# -*- coding: utf-8 -*-
# @Date    : 1/21/2026
# @Author  : Jingwen
# @Desc    : WCNS (Wireless Communication Network Slicing) Benchmark
#            Evaluates agent's ability to make network slicing decisions

import re
import math
from typing import Callable, List, Optional, Tuple, Dict, Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class WCNSBenchmark(BaseBenchmark):
    """
    Benchmark for Wireless Communication Network Slicing (WCNS) dataset.
    
    Evaluates:
    1. Slice Type Classification (eMBB vs URLLC)
    2. Bandwidth Allocation Accuracy (Proportional Fairness)
    3. Throughput Calculation Accuracy (Shannon Formula)
    
    Key Formulas:
    - Bandwidth = SliceCapacity / (ExistingUsers + 1)
    - Throughput = 10 × B × log₁₀(1 + 10^(CQI/10))
    
    Scoring Strategy:
    - Slice Type: 25% weight (binary: correct/incorrect)
    - CQI: 15% weight (must be obtained via ray_tracing tool)
    - Bandwidth: 35% weight (relative error based)
    - Throughput: 25% weight (relative error based)
    """
    
    # Slice type constants
    SLICE_TYPES = {"embb", "urllc"}
    
    # Bandwidth constraints for validation
    EMBB_BANDWIDTH_RANGE = (6.0, 20.0)  # MHz
    URLLC_BANDWIDTH_RANGE = (1.0, 5.0)   # MHz
    
    # Scoring weights (updated for 4-metric evaluation with CQI)
    SLICE_TYPE_WEIGHT = 0.25
    CQI_WEIGHT = 0.15
    BANDWIDTH_WEIGHT = 0.35
    THROUGHPUT_WEIGHT = 0.25
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
    
    def extract_slice_type(self, text: str) -> Optional[str]:
        """
        Extract slice type from prediction text.
        Looks for eMBB or URLLC keywords.
        
        Args:
            text: Prediction text
            
        Returns:
            'eMBB' or 'URLLC' or None
        """
        if text is None:
            return None
        
        text_lower = str(text).lower()
        
        # Check for explicit mentions
        embb_patterns = [
            r'\bembb\b',
            r'e-?mbb',
            r'enhanced mobile broadband',
            r'high bandwidth',
            r'high-bandwidth'
        ]
        
        urllc_patterns = [
            r'\burllc\b',
            r'u-?rllc',
            r'ultra[\s-]?reliable',
            r'low[\s-]?latency',
            r'mission[\s-]?critical'
        ]
        
        embb_score = sum(1 for p in embb_patterns if re.search(p, text_lower))
        urllc_score = sum(1 for p in urllc_patterns if re.search(p, text_lower))
        
        if embb_score > urllc_score:
            return "eMBB"
        elif urllc_score > embb_score:
            return "URLLC"
        
        # Fallback: look for direct mentions
        if 'embb' in text_lower:
            return "eMBB"
        elif 'urllc' in text_lower:
            return "URLLC"
        
        return None
    
    def extract_bandwidth(self, text: str) -> Optional[float]:
        """
        Extract bandwidth value from prediction text.
        Looks for patterns like "X MHz" or "bandwidth: X"
        
        Args:
            text: Prediction text
            
        Returns:
            Bandwidth value in MHz or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        
        # Patterns for bandwidth extraction
        bandwidth_patterns = [
            # Direct MHz mention
            r'(\d+\.?\d*)\s*(?:mhz|MHz)',
            # "bandwidth: X" or "bandwidth = X"
            r'bandwidth[:\s=]+(\d+\.?\d*)',
            r'allocated_bandwidth[:\s=]+(\d+\.?\d*)',
            # JSON-like format
            r'"bandwidth"[:\s]+(\d+\.?\d*)',
            r'"allocated_bandwidth"[:\s]+(\d+\.?\d*)',
            # Natural language
            r'allocate[d]?\s+(\d+\.?\d*)\s*(?:mhz|MHz)?',
        ]
        
        for pattern in bandwidth_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[-1])  # Use last match
                    # Validate reasonable range (0.1 to 100 MHz)
                    if 0.1 <= value <= 100:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def extract_rate(self, text: str) -> Optional[float]:
        """
        Extract throughput/rate value from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            Throughput value in Mbps or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        
        # Patterns for throughput extraction (prioritize throughput over rate)
        throughput_patterns = [
            # Throughput patterns (prioritized)
            r'throughput[:\s=]+(\d+\.?\d*)',
            r'"throughput"[:\s]+(\d+\.?\d*)',
            r'THROUGHPUT[:\s=]+(\d+\.?\d*)',
            # Direct Mbps mention
            r'(\d+\.?\d*)\s*(?:mbps|Mbps|MB/s)',
            # "rate: X" or "rate = X"
            r'rate[:\s=]+(\d+\.?\d*)',
            r'expected_rate[:\s=]+(\d+\.?\d*)',
            # JSON-like format
            r'"rate"[:\s]+(\d+\.?\d*)',
            r'"expected_rate"[:\s]+(\d+\.?\d*)',
        ]
        
        for pattern in throughput_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[-1])
                    # Validate reasonable range (0.1 to 1000 Mbps)
                    if 0.1 <= value <= 1000:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def extract_cqi(self, text: str) -> Optional[int]:
        """
        Extract CQI value from prediction text.
        Looks for patterns like "CQI: X" or "CQI = X"
        
        Args:
            text: Prediction text
            
        Returns:
            CQI value (1-15) or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        
        cqi_patterns = [
            r'CQI[:\s=]+(\d+)',
            r'"cqi"[:\s]+(\d+)',
            r'cqi[:\s=]+(\d+)',
            r'Channel Quality Indicator[:\s()]+(\d+)',
        ]
        
        for pattern in cqi_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[-1])
                    if 1 <= value <= 15:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def calculate_cqi_score(self, expected: int, predicted: int) -> float:
        """
        Calculate score for CQI prediction.
        Exact match = 1.0, off by 1 = 0.8, off by 2 = 0.5, more = 0.0
        
        Args:
            expected: Expected CQI (1-15)
            predicted: Predicted CQI (1-15)
            
        Returns:
            Score between 0 and 1
        """
        if expected is None or predicted is None:
            return 0.0
        
        diff = abs(expected - predicted)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.5
        elif diff <= 4:
            return 0.2
        else:
            return 0.0
    
    def calculate_throughput_score(self, expected: float, predicted: float) -> float:
        """
        Calculate score for throughput prediction.
        Uses relative error with tolerance levels.
        
        Args:
            expected: Expected throughput in Mbps
            predicted: Predicted throughput in Mbps
            
        Returns:
            Score between 0 and 1
        """
        if expected is None or predicted is None:
            return 0.0
        
        # Handle zero expected
        if abs(expected) < 1e-10:
            return 1.0 if abs(predicted) < 1e-10 else 0.0
        
        # Calculate relative error
        relative_error = abs(expected - predicted) / abs(expected)
        
        # Scoring tiers (same as bandwidth)
        if relative_error < 0.02:  # < 2% error
            return 1.0
        elif relative_error < 0.05:  # < 5% error
            return 0.95
        elif relative_error < 0.10:  # < 10% error
            return 0.85
        elif relative_error < 0.20:  # < 20% error
            return 0.70
        elif relative_error < 0.30:  # < 30% error
            return 0.50
        elif relative_error < 0.50:  # < 50% error
            return 0.30
        else:
            return 0.0
    
    def calculate_slice_type_score(self, expected: str, predicted: str) -> float:
        """
        Calculate score for slice type prediction.
        
        Args:
            expected: Expected slice type
            predicted: Predicted slice type
            
        Returns:
            1.0 for correct, 0.0 for incorrect
        """
        if expected is None or predicted is None:
            return 0.0
        
        expected_norm = expected.lower().strip()
        predicted_norm = predicted.lower().strip()
        
        # Handle variations
        expected_norm = 'embb' if 'embb' in expected_norm else ('urllc' if 'urllc' in expected_norm else expected_norm)
        predicted_norm = 'embb' if 'embb' in predicted_norm else ('urllc' if 'urllc' in predicted_norm else predicted_norm)
        
        return 1.0 if expected_norm == predicted_norm else 0.0
    
    def calculate_bandwidth_score(self, expected: float, predicted: float, 
                                   slice_type: str = None) -> float:
        """
        Calculate score for bandwidth allocation.
        Uses relative error with tolerance levels.
        
        Args:
            expected: Expected bandwidth in MHz
            predicted: Predicted bandwidth in MHz
            slice_type: Optional slice type for constraint validation
            
        Returns:
            Score between 0 and 1
        """
        if expected is None or predicted is None:
            return 0.0
        
        # Validate constraints based on slice type
        if slice_type:
            if slice_type.lower() == 'embb':
                min_bw, max_bw = self.EMBB_BANDWIDTH_RANGE
            else:
                min_bw, max_bw = self.URLLC_BANDWIDTH_RANGE
            
            # Penalize if prediction is out of valid range
            if not (min_bw <= predicted <= max_bw):
                # Still give partial credit if close to range
                if predicted < min_bw:
                    distance = (min_bw - predicted) / min_bw
                else:
                    distance = (predicted - max_bw) / max_bw
                
                if distance < 0.2:  # Within 20% of range
                    return 0.5
                return 0.2
        
        # Handle zero expected
        if abs(expected) < 1e-10:
            return 1.0 if abs(predicted) < 1e-10 else 0.0
        
        # Calculate relative error
        relative_error = abs(expected - predicted) / abs(expected)
        
        # Scoring tiers
        if relative_error < 0.02:  # < 2% error
            return 1.0
        elif relative_error < 0.05:  # < 5% error
            return 0.95
        elif relative_error < 0.10:  # < 10% error
            return 0.85
        elif relative_error < 0.20:  # < 20% error
            return 0.70
        elif relative_error < 0.30:  # < 30% error
            return 0.50
        elif relative_error < 0.50:  # < 50% error
            return 0.30
        else:
            return 0.0
    
    def calculate_score(self, expected_output: Dict[str, Any], 
                       prediction: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate combined score for network slicing prediction.
        
        Args:
            expected_output: Expected answer dict with slice_type, bandwidth, throughput
            prediction: Model prediction text
            
        Returns:
            (combined_score, extracted_values_dict)
        """
        # Extract values from prediction
        pred_slice_type = self.extract_slice_type(prediction)
        pred_bandwidth = self.extract_bandwidth(prediction)
        pred_throughput = self.extract_rate(prediction)
        pred_cqi = self.extract_cqi(prediction)
        
        # Get expected values (support both old 'allocated_bandwidth' and new 'bandwidth' keys)
        exp_slice_type = expected_output.get("slice_type")
        exp_bandwidth = expected_output.get("bandwidth") or expected_output.get("allocated_bandwidth")
        exp_throughput = expected_output.get("throughput") or expected_output.get("expected_rate")
        exp_cqi = expected_output.get("cqi")
        
        # Calculate individual scores
        slice_score = self.calculate_slice_type_score(exp_slice_type, pred_slice_type)
        bandwidth_score = self.calculate_bandwidth_score(exp_bandwidth, pred_bandwidth, exp_slice_type)
        throughput_score = self.calculate_throughput_score(exp_throughput, pred_throughput)
        cqi_score = self.calculate_cqi_score(exp_cqi, pred_cqi) if exp_cqi is not None else 0.0
        
        # Combined score: if CQI is in the answer, use 4-metric scoring; otherwise 3-metric
        if exp_cqi is not None:
            combined_score = (self.SLICE_TYPE_WEIGHT * slice_score + 
                             self.CQI_WEIGHT * cqi_score +
                             self.BANDWIDTH_WEIGHT * bandwidth_score +
                             self.THROUGHPUT_WEIGHT * throughput_score)
        else:
            # Backward compatible: old V2 data without CQI
            combined_score = (0.30 * slice_score + 
                             0.40 * bandwidth_score +
                             0.30 * throughput_score)
        
        extracted = {
            "pred_slice_type": pred_slice_type,
            "pred_cqi": pred_cqi,
            "pred_bandwidth": pred_bandwidth,
            "pred_throughput": pred_throughput,
            "slice_score": slice_score,
            "cqi_score": cqi_score,
            "bandwidth_score": bandwidth_score,
            "throughput_score": throughput_score
        }
        
        return combined_score, extracted
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), 
           retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate output with retry logic."""
        return await graph(input_text)
    
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, Dict, float, float]:
        """
        Evaluate a single network slicing problem.
        
        Args:
            problem: Problem dict with question, input, answer
            graph: Agent callable
            
        Returns:
            (question, prediction, expected_output, score, cost)
        """
        input_text = problem["question"]
        expected_answer = problem["answer"]
        
        try:
            output, cost = await self._generate_output(graph, input_text)
            
            # Calculate score
            score, extracted = self.calculate_score(expected_answer, str(output))
            
            # Log mismatches for analysis
            if score < 1.0:
                extra_info = (f"slice={extracted['slice_score']:.2f}, "
                            f"cqi={extracted['cqi_score']:.2f}, "
                            f"bw={extracted['bandwidth_score']:.2f}, "
                            f"tp={extracted['throughput_score']:.2f}, "
                            f"pred: {extracted['pred_slice_type']}/CQI{extracted['pred_cqi']}/{extracted['pred_bandwidth']}/{extracted['pred_throughput']}")
                self.log_mismatch(
                    input_text,
                    expected_answer,
                    output,
                    extracted,
                    extract_answer_code=extra_info
                )
            
            return input_text, output, expected_answer, score, cost
        
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping sample. Error: {e}")
            return input_text, str(e), expected_answer, 0.0, 0.0
    
    def get_result_columns(self) -> List[str]:
        """Get column names for result DataFrame."""
        return ["question", "prediction", "expected_output", "score", "cost"]


class WCNSBenchmarkWithCoT(WCNSBenchmark):
    """
    Extended WCNS Benchmark that also evaluates Chain-of-Thought reasoning.
    
    Additional evaluation dimensions:
    1. Workflow Step Completeness
    2. Tool Usage Correctness
    3. Reasoning Quality
    """
    
    # Expected workflow steps
    WORKFLOW_STEPS = [
        "intent_understanding",
        "slice_allocation", 
        "bandwidth_allocation",
        "qos_evaluation",
        "adjustment_check"
    ]
    
    # Expected tools
    EXPECTED_TOOLS = {
        "knowledge_base_query",
        "network_monitor",
        "beamforming_tool",
        "slice_allocation",
        "check_and_adjust_capacity"
    }
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
    
    def evaluate_cot_completeness(self, prediction: str) -> float:
        """
        Evaluate if the prediction includes all workflow steps.
        
        Returns:
            Score between 0 and 1 based on step coverage
        """
        prediction_lower = str(prediction).lower()
        
        step_keywords = {
            "intent_understanding": ["intent", "understand", "classify", "request analysis"],
            "slice_allocation": ["slice type", "embb", "urllc", "slice allocation"],
            "bandwidth_allocation": ["bandwidth", "shannon", "rate calculation", "spectral"],
            "qos_evaluation": ["qos", "quality", "evaluation", "satisfy", "requirement"],
            "adjustment_check": ["adjust", "capacity", "available", "dynamic"]
        }
        
        found_steps = 0
        for step, keywords in step_keywords.items():
            if any(kw in prediction_lower for kw in keywords):
                found_steps += 1
        
        return found_steps / len(self.WORKFLOW_STEPS)
    
    def evaluate_tool_usage(self, prediction: str) -> float:
        """
        Evaluate if prediction mentions using appropriate tools.
        
        Returns:
            Score between 0 and 1 based on tool mention coverage
        """
        prediction_lower = str(prediction).lower()
        
        tool_patterns = {
            "knowledge_base": ["knowledge base", "knowledge_base", "kb query"],
            "network_monitor": ["network monitor", "monitor", "network state"],
            "beamforming": ["beamforming", "beam", "ray tracing"],
            "shannon": ["shannon", "capacity formula", "log", "spectral efficiency"],
            "slice_allocation": ["slice_allocation", "allocate slice"],
            "capacity_check": ["capacity", "adjust", "workload"]
        }
        
        found_tools = 0
        for tool, patterns in tool_patterns.items():
            if any(p in prediction_lower for p in patterns):
                found_tools += 1
        
        return found_tools / len(tool_patterns)
    
    def calculate_score(self, expected_output: Dict[str, Any],
                       prediction: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate combined score including CoT evaluation.
        
        Weights:
        - Base score (slice + bandwidth): 70%
        - CoT completeness: 20%
        - Tool usage: 10%
        """
        # Get base score
        base_score, extracted = super().calculate_score(expected_output, prediction)
        
        # CoT evaluation
        cot_score = self.evaluate_cot_completeness(prediction)
        tool_score = self.evaluate_tool_usage(prediction)
        
        # Combined score with CoT bonus
        combined_score = (0.70 * base_score + 
                         0.20 * cot_score + 
                         0.10 * tool_score)
        
        extracted["cot_score"] = cot_score
        extracted["tool_score"] = tool_score
        
        return combined_score, extracted
