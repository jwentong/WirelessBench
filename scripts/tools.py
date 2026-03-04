# -*- coding: utf-8 -*-
# @Date    : 1/20/2026
# @Author  : Jingwen
# @Desc    : Tool infrastructure and registry for WirelessBench

"""
Tool Infrastructure for WirelessBench
======================================

Provides the core tool framework used by all WirelessBench tools:

1. ToolMetrics    - Automatic performance tracking for tool calls
2. ToolSchema     - Unified tool description format (EasyTool-inspired)
3. BaseTool       - Abstract base class for all tools
4. ToolRegistry   - Tool management and discovery
5. CalculatorTool - General-purpose math expression evaluator (SymPy)

Domain-specific tools are in:
  - scripts/wireless_tools.py   (PythonExecutor, UnitConverter, WirelessFormulaLibrary)
  - scripts/enhanced_tools.py   (TelecomFormulaRetriever, TelecomPrecisionCalculator)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time
from collections import deque
from scripts.logs import logger


# ==================== Tool Metrics (Performance Tracking) ====================

class ToolMetrics:
    """
    Automatic metrics collection for tools.

    Tracks:
    - Total calls, success/failure counts
    - Execution time statistics (min, max, avg)
    - Recent call history (rolling window)
    """

    def __init__(self, name: str, history_size: int = 100):
        self.name = name
        self.history_size = history_size

        # Counters
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

        # Time tracking
        self.total_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0

        # Recent history
        self.recent_calls: deque = deque(maxlen=history_size)
        self.recent_errors: deque = deque(maxlen=50)

    def record(self, duration: float, success: bool, error: Optional[str] = None):
        """Record a tool call."""
        self.total_calls += 1
        self.total_execution_time += duration
        self.min_execution_time = min(self.min_execution_time, duration)
        self.max_execution_time = max(self.max_execution_time, duration)

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.recent_errors.append({"timestamp": time.time(), "error": error})

        self.recent_calls.append({
            "timestamp": time.time(),
            "duration": duration,
            "success": success,
        })

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics."""
        if self.total_calls == 0:
            return {
                "tool_name": self.name,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0,
            }

        recent_successes = sum(1 for c in self.recent_calls if c["success"])
        recent_success_rate = recent_successes / len(self.recent_calls) if self.recent_calls else 0.0

        return {
            "tool_name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls,
            "avg_execution_time": self.total_execution_time / self.total_calls,
            "min_execution_time": self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
            "max_execution_time": self.max_execution_time,
            "recent_success_rate": recent_success_rate,
            "recent_error_count": len(self.recent_errors),
        }

    def reset(self):
        """Reset all metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        self.recent_calls.clear()
        self.recent_errors.clear()


# ==================== Tool Schema ====================

class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


class ToolSchema(BaseModel):
    """Unified tool description format (EasyTool-inspired)."""
    name: str = Field(description="Tool name (unique identifier)")
    description: str = Field(description="Concise description of what the tool does")
    category: str = Field(default="general", description="Tool category")
    parameters: List[ToolParameter] = Field(default_factory=list)
    usage_example: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default="1.0")

    def to_llm_description(self) -> str:
        """Generate concise description for LLM consumption."""
        param_descs = []
        for param in self.parameters:
            required_mark = "*" if param.required else ""
            default_info = f" (default: {param.default})" if param.default is not None else ""
            param_descs.append(
                f"  - {param.name}{required_mark} ({param.type}): {param.description}{default_info}"
            )

        params_text = "\n".join(param_descs) if param_descs else "  No parameters"

        description = (
            f"Tool: {self.name}\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{params_text}"
        )
        if self.usage_example:
            description += f"\nExample: {self.usage_example}"
        return description.strip()

    def to_openai_function_format(self) -> Dict:
        """Convert to OpenAI Function Calling format."""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {"type": param.type, "description": param.description}
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }


# ==================== Base Tool ====================

class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Features:
    - Unified schema definition
    - Auto-generate LLM-friendly descriptions
    - Automatic metrics collection via ``__call__``
    """

    def __init__(self, schema: ToolSchema = None, enable_metrics: bool = True):
        if schema is not None:
            self.schema = schema
            self.name = schema.name
        else:
            try:
                self.name = self.schema.name
            except (AttributeError, NotImplementedError):
                self.name = self.__class__.__name__

        self.metrics = ToolMetrics(name=self.name) if enable_metrics else None

    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with automatic metrics collection."""
        start_time = time.time()
        try:
            result = await self.execute(**kwargs)
            execution_time = time.time() - start_time

            if isinstance(result, str):
                result = {"success": True, "result": result, "error": None}

            if self.metrics:
                success = result.get("success", False)
                error = result.get("error") if not success else None
                self.metrics.record(duration=execution_time, success=success, error=error)

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            if self.metrics:
                self.metrics.record(duration=execution_time, success=False, error=str(e))
            raise

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool invocation (implement in subclass)."""
        pass

    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return self.schema

    def get_llm_description(self) -> str:
        """Get LLM-friendly description."""
        return self.schema.to_llm_description()

    def get_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        if not self.metrics:
            return {"tool_name": self.name, "metrics_enabled": False}
        return self.metrics.get_stats()

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate parameters against schema."""
        required_params = [p.name for p in self.schema.parameters if p.required]
        missing = set(required_params) - set(kwargs.keys())
        if missing:
            return False, f"Missing required parameters: {missing}"
        return True, None


# ==================== Calculator Tool ====================

class CalculatorTool(BaseTool):
    """Mathematical expression evaluator using SymPy."""

    def __init__(self):
        schema = ToolSchema(
            name="calculator",
            description=(
                "Evaluate PURE mathematical expressions using SymPy. "
                "Supports arithmetic, algebra, trigonometry, calculus, etc. "
                "CANNOT handle Python code syntax (lists, loops, conditionals). "
                "For those, use python_executor instead."
            ),
            category="compute",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Pure mathematical expression. E.g., '2+2', 'sqrt(16)', 'sin(pi/2)'",
                    required=True,
                ),
            ],
            usage_example='calculator(expression="2 + 2 * 3")',
            source="sympy",
        )
        super().__init__(schema)

    async def execute(self, expression: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate mathematical expression."""
        if not expression:
            expression = kwargs.get("expr") or kwargs.get("formula") or kwargs.get("input")
            if not expression:
                return {"success": False, "result": None, "error": "Missing 'expression' parameter."}

        valid, error = self.validate_parameters(expression=expression)
        if not valid:
            return {"success": False, "result": None, "error": error}

        try:
            import sympy

            expression = expression.strip()
            python_only_patterns = ["len(", "[", "for ", " in ", "if ", "else", "def ", "lambda"]
            if any(p in expression for p in python_only_patterns):
                return {
                    "success": False,
                    "result": None,
                    "error": f"Expression contains Python syntax. Use python_executor instead: {expression[:100]}",
                }

            result = sympy.sympify(expression)
            evaluated = sympy.N(result, 10)
            return {"success": True, "result": str(evaluated), "error": None}

        except Exception as e:
            logger.error(f"Calculator tool error: {e}")
            return {"success": False, "result": None, "error": f"Invalid expression: {str(e)[:200]}"}


# ==================== Tool Registry ====================

class ToolRegistry:
    """
    Manage all available tools.

    Features:
    - Register / retrieve tools by name or category
    - Generate concise descriptions (token-efficient)
    - Export to OpenAI Function Calling format
    - Aggregate usage reports
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[str, List[str]] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        category = tool.schema.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)
        logger.info(f"Registered tool: {tool.name} ({category})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool instance by name."""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category."""
        return [self.tools[n] for n in self.categories.get(category, [])]

    def get_all_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas."""
        return [t.get_schema() for t in self.tools.values()]

    def get_concise_description(self, max_tools: int = None) -> str:
        """Generate concise tool descriptions for LLM prompts."""
        tools_list = list(self.tools.values())[:max_tools] if max_tools else list(self.tools.values())
        descriptions = [t.get_llm_description() for t in tools_list]
        header = f"Available Tools ({len(tools_list)} total):"
        return header + "\n\n" + "\n\n---\n\n".join(descriptions)

    def export_to_openai_functions(self) -> List[Dict]:
        """Export all tools to OpenAI Function Calling format."""
        return [t.schema.to_openai_function_format() for t in self.tools.values()]

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with automatic metrics."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}",
            }
        try:
            return await tool(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "result": None, "error": str(e)}

    def get_usage_report(self) -> Dict[str, Any]:
        """Generate comprehensive usage report for all tools."""
        report: Dict[str, Any] = {"total_tools": len(self.tools), "tools": {}, "summary": {}}
        all_stats = []
        for name, tool in self.tools.items():
            stats = tool.get_statistics()
            report["tools"][name] = stats
            if stats.get("total_calls", 0) > 0:
                all_stats.append((name, stats))

        if all_stats:
            total_calls = sum(s["total_calls"] for _, s in all_stats)
            report["summary"]["total_calls"] = total_calls
            all_stats.sort(key=lambda x: x[1]["total_calls"], reverse=True)
            report["summary"]["most_used_tool"] = all_stats[0][0]
            report["summary"]["least_used_tool"] = all_stats[-1][0]

            stats_with_calls = [(n, s) for n, s in all_stats if s["total_calls"] > 0]
            if stats_with_calls:
                stats_with_calls.sort(key=lambda x: x[1]["success_rate"], reverse=True)
                report["summary"]["highest_success_rate"] = {
                    "tool": stats_with_calls[0][0],
                    "rate": stats_with_calls[0][1]["success_rate"],
                }
                report["summary"]["lowest_success_rate"] = {
                    "tool": stats_with_calls[-1][0],
                    "rate": stats_with_calls[-1][1]["success_rate"],
                }

        return report
