# -*- coding: utf-8 -*-
# @Date    : 1/21/2026
# @Author  : Jingwen
# @Desc    : WCMSA (Wireless Communication Mobile Service Assurance) Benchmark
#            V3: Supports tool-necessary dataset with CQI calculation and QoS verification

import re
import math
from typing import Callable, List, Optional, Tuple, Dict, Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class WCMSABenchmark(BaseBenchmark):
    """
    Benchmark for Wireless Communication Mobile Service Assurance (WCMSA) dataset.
    
    V3 Evaluates (6 metrics for tool-necessary version):
    1. Position Prediction Accuracy (x, y coordinates)
    2. CQI Prediction Accuracy (calculated from position)
    3. Slice Type Classification (eMBB vs URLLC)
    4. Bandwidth Allocation Accuracy
    5. Throughput Calculation Accuracy
    6. QoS Verification (bonus)
    
    Scoring weights (V3):
    - Position Prediction: 15%
    - CQI Prediction: 15%
    - Slice Type: 20%
    - Bandwidth: 25%
    - Throughput: 20%
    - QoS Verification: 5% (bonus)
    """
    
    # Scoring weights (V3 - 6 metrics for tool-necessary version)
    POSITION_WEIGHT = 0.15
    CQI_WEIGHT = 0.15
    SLICE_TYPE_WEIGHT = 0.20
    BANDWIDTH_WEIGHT = 0.25
    THROUGHPUT_WEIGHT = 0.20
    QOS_WEIGHT = 0.05  # Bonus for QoS verification
    
    # Bandwidth constraints
    EMBB_BANDWIDTH_RANGE = (6.0, 20.0)  # MHz
    URLLC_BANDWIDTH_RANGE = (1.0, 5.0)   # MHz
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
    
    def extract_position(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract predicted position (x, y) from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            (x, y) or (None, None)
        """
        if text is None:
            return None, None
        
        text_str = str(text)
        
        # Patterns for position extraction - ORDER MATTERS! More specific patterns first
        patterns = [
            # PRIORITY 1: "Predicted Position:" followed by coordinates
            r'[Pp]redicted\s+[Pp]osition[:\s]*\(?\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)',
            # PRIORITY 2: "**Predicted Position:**" markdown format
            r'\*\*[Pp]redicted\s+[Pp]osition[:\*\s]*\(?\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)',
            # PRIORITY 3: "next position" pattern
            r'next\s+position[:\s]*\(?\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)',
            # JSON-like: "predicted_position": {"x": -62.01, "y": 106.82}
            r'"predicted_position"[:\s]*\{[^}]*"x"[:\s]*([-+]?\d+\.?\d*)[^}]*"y"[:\s]*([-+]?\d+\.?\d*)',
            # Backtick format: `(-62.01, 106.82)`
            r'`\s*\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)\s*`',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    # Use the LAST match for more specific patterns (likely the final answer)
                    x, y = float(matches[-1][0]), float(matches[-1][1])
                    # Validate reasonable range
                    if -500 <= x <= 500 and -500 <= y <= 500:
                        return x, y
                except (ValueError, IndexError):
                    continue
        
        return None, None
    
    def extract_cqi(self, text: str) -> Optional[int]:
        """
        Extract predicted CQI from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            CQI value (1-15) or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        
        # Patterns for CQI extraction
        patterns = [
            r'"predicted_cqi"[:\s]*(\d+)',
            r'predicted[\s_]*cqi[:\s=]*(\d+)',
            r'cqi[:\s=]*(\d+)',
            r'cqi\s+(?:is\s+)?(\d+)',
            r'channel quality[:\s]*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    cqi = int(matches[-1])
                    if 1 <= cqi <= 15:
                        return cqi
                except ValueError:
                    continue
        
        return None
    
    def extract_slice_type(self, text: str) -> Optional[str]:
        """
        Extract slice type from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            'eMBB' or 'URLLC' or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        text_lower = text_str.lower()
        
        # PRIORITY 1: Look for explicit "Slice Type:" pattern in the final answer
        slice_type_patterns = [
            r'[Ss]lice\s+[Tt]ype[:\s]*[`\*]*\s*(eMBB|URLLC|embb|urllc)',
            r'\*\*[Ss]lice\s+[Tt]ype[:\*\s]*[`\*]*\s*(eMBB|URLLC|embb|urllc)',
            r'[Ss]lice[:\s]+[`\*]*(eMBB|URLLC|embb|urllc)',
        ]
        
        for pattern in slice_type_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                # Use the LAST match (likely the final answer)
                result = matches[-1].upper()
                if 'EMBB' in result:
                    return "eMBB"
                elif 'URLLC' in result:
                    return "URLLC"
        
        # PRIORITY 2: Count explicit mentions (but give more weight to later occurrences)
        embb_patterns = [r'\bembb\b', r'e-?mbb', r'enhanced mobile broadband']
        urllc_patterns = [r'\burllc\b', r'u-?rllc', r'ultra[\s-]?reliable', r'low[\s-]?latency']
        
        embb_score = sum(1 for p in embb_patterns if re.search(p, text_lower))
        urllc_score = sum(1 for p in urllc_patterns if re.search(p, text_lower))
        
        if embb_score > urllc_score:
            return "eMBB"
        elif urllc_score > embb_score:
            return "URLLC"
        
        # Direct mentions
        if 'embb' in text_lower:
            return "eMBB"
        elif 'urllc' in text_lower:
            return "URLLC"
        
        return None
    
    def extract_bandwidth(self, text: str) -> Optional[float]:
        """
        Extract bandwidth value from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            Bandwidth value in MHz or None
        """
        if text is None:
            return None
        
        text_str = str(text)
        
        # PRIORITY 1: Look for explicit "Bandwidth:" pattern in the final answer
        priority_patterns = [
            r'[Bb]andwidth[:\s]*[`\*]*\s*([\d.]+)\s*(?:MHz|mhz)?',
            r'\*\*[Bb]andwidth[:\*\s]*[`\*]*\s*([\d.]+)\s*(?:MHz|mhz)?',
        ]
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    # Use the LAST match (likely the final answer)
                    value = float(matches[-1])
                    if 1.0 <= value <= 30:  # Valid range for both eMBB and URLLC
                        return value
                except ValueError:
                    continue
        
        # PRIORITY 2: Other patterns
        patterns = [
            r'"allocated_bandwidth"[:\s]*([\d.]+)',
            r'allocat(?:ed|e)[:\s]*([\d.]+)\s*(?:mhz)?',
            r'([\d.]+)\s*mhz\s*bandwidth',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[-1])
                    if 1.0 <= value <= 30:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def calculate_position_score(self, expected: Dict[str, float], 
                                  pred_x: Optional[float], 
                                  pred_y: Optional[float]) -> float:
        """
        Calculate score for position prediction based on Euclidean distance.
        
        Args:
            expected: Expected position dict with 'x' and 'y'
            pred_x: Predicted x coordinate
            pred_y: Predicted y coordinate
            
        Returns:
            Score between 0 and 1
        """
        if pred_x is None or pred_y is None:
            return 0.0
        
        exp_x = expected.get('x', 0)
        exp_y = expected.get('y', 0)
        
        # Euclidean distance
        distance = math.sqrt((exp_x - pred_x)**2 + (exp_y - pred_y)**2)
        
        # Continuous scoring: smooth decay with 20m threshold
        # Provides gradient signal for optimizer to distinguish small improvements
        # score ≈ 1.0 at 0m, 0.95 at 1m, 0.75 at 5m, 0.50 at 10m, 0.0 at 20m+
        if distance >= 20.0:
            return 0.0
        return max(0.0, 1.0 - (distance / 20.0) ** 1.2)
    
    def calculate_cqi_score(self, expected: int, predicted: Optional[int]) -> float:
        """
        Calculate score for CQI prediction.
        
        Args:
            expected: Expected CQI (1-15)
            predicted: Predicted CQI
            
        Returns:
            Score between 0 and 1
        """
        if predicted is None:
            return 0.0
        
        diff = abs(expected - predicted)
        
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.90
        elif diff == 2:
            return 0.75
        elif diff == 3:
            return 0.50
        elif diff <= 5:
            return 0.25
        else:
            return 0.0
    
    def calculate_slice_type_score(self, expected: str, predicted: Optional[str]) -> float:
        """
        Calculate score for slice type classification.
        
        Args:
            expected: Expected slice type
            predicted: Predicted slice type
            
        Returns:
            1.0 for correct, 0.0 for incorrect
        """
        if expected is None or predicted is None:
            return 0.0
        
        exp_norm = expected.lower().strip()
        pred_norm = predicted.lower().strip()
        
        exp_norm = 'embb' if 'embb' in exp_norm else ('urllc' if 'urllc' in exp_norm else exp_norm)
        pred_norm = 'embb' if 'embb' in pred_norm else ('urllc' if 'urllc' in pred_norm else pred_norm)
        
        return 1.0 if exp_norm == pred_norm else 0.0
    
    def calculate_bandwidth_score(self, expected: float, 
                                   predicted: Optional[float],
                                   slice_type: str = None) -> float:
        """
        Calculate score for bandwidth allocation.
        
        Args:
            expected: Expected bandwidth in MHz
            predicted: Predicted bandwidth in MHz
            slice_type: Slice type for range validation
            
        Returns:
            Score between 0 and 1
        """
        if expected is None or predicted is None:
            return 0.0
        
        # Validate range
        if slice_type:
            if slice_type.lower() == 'embb':
                min_bw, max_bw = self.EMBB_BANDWIDTH_RANGE
            else:
                min_bw, max_bw = self.URLLC_BANDWIDTH_RANGE
            
            if not (min_bw <= predicted <= max_bw):
                # Out of range penalty
                if predicted < min_bw:
                    distance = (min_bw - predicted) / min_bw
                else:
                    distance = (predicted - max_bw) / max_bw
                
                return 0.5 if distance < 0.2 else 0.2
        
        if abs(expected) < 1e-10:
            return 1.0 if abs(predicted) < 1e-10 else 0.0
        
        # Relative error
        relative_error = abs(expected - predicted) / abs(expected)
        
        if relative_error < 0.02:
            return 1.0
        elif relative_error < 0.05:
            return 0.95
        elif relative_error < 0.10:
            return 0.85
        elif relative_error < 0.20:
            return 0.70
        elif relative_error < 0.30:
            return 0.50
        elif relative_error < 0.50:
            return 0.30
        else:
            return 0.0
    
    def extract_throughput(self, text: str) -> Optional[float]:
        """Extract throughput value from prediction text."""
        if text is None:
            return None
        
        text_str = str(text)
        
        patterns = [
            r'throughput[:\s=]*([\d.]+)\s*(?:mbps)?',
            r'"throughput"[:\s]*([\d.]+)',
            r'([\d.]+)\s*mbps',
            r'rate[:\s=]*([\d.]+)\s*(?:mbps)?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[-1])
                    if 0.1 <= value <= 1000:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def extract_cqi(self, text: str) -> Optional[int]:
        """Extract predicted CQI from prediction text."""
        if text is None:
            return None
        
        text_str = str(text)
        
        patterns = [
            r'[Pp]redicted\s+CQI[:\s]*(\d+)',
            r'\*\*[Pp]redicted\s+CQI[:\*\s]*(\d+)',
            r'CQI[:\s]*(\d+)',
            r'"predicted_cqi"[:\s]*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    cqi = int(matches[-1])
                    if 1 <= cqi <= 15:
                        return cqi
                except ValueError:
                    continue
        
        return None
    
    def extract_qos_satisfied(self, text: str) -> Optional[bool]:
        """Extract QoS satisfied status from prediction text."""
        if text is None:
            return None
        
        text_lower = str(text).lower()
        
        # Look for explicit QoS statements
        patterns = [
            r'qos\s+satisfied[:\s]*(yes|no|true|false)',
            r'qos[:\s]*(satisfied|not\s+satisfied|met|not\s+met)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(1)
                if result in ['yes', 'true', 'satisfied', 'met']:
                    return True
                elif result in ['no', 'false', 'not satisfied', 'not met']:
                    return False
        
        return None
    
    def calculate_cqi_score(self, expected: int, predicted: Optional[int]) -> float:
        """Calculate score for CQI prediction."""
        if expected is None or predicted is None:
            return 0.0
        
        diff = abs(expected - predicted)
        
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.90
        elif diff == 2:
            return 0.75
        elif diff == 3:
            return 0.50
        elif diff <= 5:
            return 0.25
        else:
            return 0.0
    
    def calculate_qos_score(self, expected: bool, predicted: Optional[bool]) -> float:
        """Calculate score for QoS verification."""
        if expected is None or predicted is None:
            return 0.0
        return 1.0 if expected == predicted else 0.0
    
    def calculate_throughput_score(self, expected: float, predicted: Optional[float]) -> float:
        """Calculate score for throughput prediction."""
        if expected is None or predicted is None:
            return 0.0
        
        if abs(expected) < 1e-10:
            return 1.0 if abs(predicted) < 1e-10 else 0.0
        
        relative_error = abs(expected - predicted) / abs(expected)
        
        if relative_error < 0.02:
            return 1.0
        elif relative_error < 0.05:
            return 0.95
        elif relative_error < 0.10:
            return 0.85
        elif relative_error < 0.20:
            return 0.70
        elif relative_error < 0.30:
            return 0.50
        elif relative_error < 0.50:
            return 0.30
        else:
            return 0.0
    
    def calculate_score(self, expected_output: Dict[str, Any],
                       prediction: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate combined score for MSA prediction.
        
        V3: Supports 6 metrics for tool-necessary version
        
        Args:
            expected_output: Expected answer dict
            prediction: Model prediction text
            
        Returns:
            (combined_score, extracted_values_dict)
        """
        # Extract values from prediction
        pred_x, pred_y = self.extract_position(prediction)
        pred_cqi = self.extract_cqi(prediction)
        pred_slice = self.extract_slice_type(prediction)
        pred_bw = self.extract_bandwidth(prediction)
        pred_tp = self.extract_throughput(prediction)
        pred_qos = self.extract_qos_satisfied(prediction)
        
        # Get expected values (support both V2 and V3 format)
        exp_position = expected_output.get("predicted_position", {})
        exp_cqi = expected_output.get("predicted_cqi")
        exp_slice = expected_output.get("slice_type")
        exp_bw = expected_output.get("bandwidth") or expected_output.get("allocated_bandwidth")
        exp_tp = expected_output.get("throughput") or expected_output.get("expected_rate")
        exp_qos = expected_output.get("qos_satisfied")
        
        # Calculate individual scores
        position_score = self.calculate_position_score(exp_position, pred_x, pred_y)
        slice_score = self.calculate_slice_type_score(exp_slice, pred_slice)
        bandwidth_score = self.calculate_bandwidth_score(exp_bw, pred_bw, exp_slice)
        throughput_score = self.calculate_throughput_score(exp_tp, pred_tp)
        
        # V3 specific scores (CQI and QoS)
        cqi_score = 0.0
        qos_score = 0.0
        
        if exp_cqi is not None:
            # V3 dataset with CQI
            cqi_score = self.calculate_cqi_score(exp_cqi, pred_cqi)
            qos_score = self.calculate_qos_score(exp_qos, pred_qos)
            
            # V3 weighted score (6 metrics)
            combined_score = (
                self.POSITION_WEIGHT * position_score +
                self.CQI_WEIGHT * cqi_score +
                self.SLICE_TYPE_WEIGHT * slice_score +
                self.BANDWIDTH_WEIGHT * bandwidth_score +
                self.THROUGHPUT_WEIGHT * throughput_score +
                self.QOS_WEIGHT * qos_score
            )
        else:
            # V2 dataset (backward compatible, 4 metrics)
            combined_score = (
                0.20 * position_score +
                0.25 * slice_score +
                0.30 * bandwidth_score +
                0.25 * throughput_score
            )
        
        extracted = {
            "pred_position": {"x": pred_x, "y": pred_y},
            "pred_cqi": pred_cqi,
            "pred_slice_type": pred_slice,
            "pred_bandwidth": pred_bw,
            "pred_throughput": pred_tp,
            "pred_qos_satisfied": pred_qos,
            "position_score": position_score,
            "cqi_score": cqi_score,
            "slice_score": slice_score,
            "bandwidth_score": bandwidth_score,
            "throughput_score": throughput_score,
            "qos_score": qos_score
        }
        
        return combined_score, extracted
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate output with retry logic."""
        return await graph(input_text)
    
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, Dict, float, float]:
        """
        Evaluate a single MSA problem.
        
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
            
            # Log mismatches
            if score < 1.0:
                # V3 format with all 6 metrics
                extra_info = (f"pos={extracted['position_score']:.2f}, "
                            f"cqi={extracted['cqi_score']:.2f}, "
                            f"slice={extracted['slice_score']:.2f}, "
                            f"bw={extracted['bandwidth_score']:.2f}, "
                            f"tp={extracted['throughput_score']:.2f}, "
                            f"qos={extracted['qos_score']:.2f}")
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


class WCMSABenchmarkWithCoT(WCMSABenchmark):
    """
    Extended WCMSA Benchmark that evaluates Chain-of-Thought reasoning.
    
    Additional evaluation:
    1. Workflow Step Completeness (6 steps)
    2. Tool Usage Correctness
    3. Proactive Allocation Detection
    """
    
    # Expected workflow steps
    WORKFLOW_STEPS = [
        "position_prediction",
        "cqi_prediction",
        "intent_understanding",
        "slice_allocation",
        "resource_allocation",
        "qos_evaluation"
    ]
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
    
    def evaluate_cot_completeness(self, prediction: str) -> float:
        """
        Evaluate if prediction includes all workflow steps.
        
        Returns:
            Score between 0 and 1
        """
        prediction_lower = str(prediction).lower()
        
        step_keywords = {
            "position_prediction": ["position predict", "kalman", "trajectory", "next position"],
            "cqi_prediction": ["cqi", "channel quality", "ray tracing", "snr"],
            "intent_understanding": ["intent", "service type", "request", "qos requirement"],
            "slice_allocation": ["slice type", "embb", "urllc", "slice allocation"],
            "resource_allocation": ["bandwidth", "resource", "shannon", "allocation"],
            "qos_evaluation": ["qos", "satisfied", "requirement", "evaluation"]
        }
        
        found_steps = 0
        for step, keywords in step_keywords.items():
            if any(kw in prediction_lower for kw in keywords):
                found_steps += 1
        
        return found_steps / len(self.WORKFLOW_STEPS)
    
    def evaluate_tool_usage(self, prediction: str) -> float:
        """
        Evaluate tool usage mentions.
        
        Returns:
            Score between 0 and 1
        """
        prediction_lower = str(prediction).lower()
        
        tool_patterns = {
            "kalman_filter": ["kalman", "filter", "trajectory predict"],
            "ray_tracing": ["ray tracing", "cqi calculator", "path loss"],
            "knowledge_base": ["knowledge base", "intent", "service type"],
            "network_monitor": ["network state", "utilization", "monitor"],
            "shannon_formula": ["shannon", "spectral efficiency", "capacity formula"],
            "beamforming": ["beamforming", "resource allocation", "bandwidth"]
        }
        
        found_tools = sum(1 for patterns in tool_patterns.values() 
                        if any(p in prediction_lower for p in patterns))
        
        return found_tools / len(tool_patterns)
    
    def evaluate_proactive_detection(self, prediction: str, expected: Dict) -> float:
        """
        Evaluate if proactive allocation was correctly detected.
        
        Returns:
            1.0 if correctly identified, 0.5 if partially, 0.0 otherwise
        """
        prediction_lower = str(prediction).lower()
        
        proactive_keywords = ["proactive", "degradation", "predicted cqi", "channel change"]
        mentions_proactive = any(kw in prediction_lower for kw in proactive_keywords)
        
        # Check if expected answer suggests proactive allocation was needed
        current_cqi = expected.get("metadata", {}).get("current_cqi", 15)
        predicted_cqi = expected.get("predicted_cqi", 15)
        needs_proactive = predicted_cqi < current_cqi - 1
        
        if needs_proactive and mentions_proactive:
            return 1.0
        elif not needs_proactive and not mentions_proactive:
            return 1.0
        elif mentions_proactive:
            return 0.5  # Mentioned but not needed
        else:
            return 0.3 if needs_proactive else 0.7
    
    def calculate_score(self, expected_output: Dict[str, Any],
                       prediction: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate combined score including CoT evaluation.
        
        Weights:
        - Base score (position + CQI + slice + bandwidth): 65%
        - CoT completeness: 20%
        - Tool usage: 10%
        - Proactive detection: 5%
        """
        # Get base score
        base_score, extracted = super().calculate_score(expected_output, prediction)
        
        # CoT evaluation
        cot_score = self.evaluate_cot_completeness(prediction)
        tool_score = self.evaluate_tool_usage(prediction)
        proactive_score = self.evaluate_proactive_detection(prediction, expected_output)
        
        # Combined score
        combined_score = (
            0.65 * base_score +
            0.20 * cot_score +
            0.10 * tool_score +
            0.05 * proactive_score
        )
        
        extracted["cot_score"] = cot_score
        extracted["tool_score"] = tool_score
        extracted["proactive_score"] = proactive_score
        
        return combined_score, extracted
