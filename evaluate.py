#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WirelessBench - Evaluation Entry Point

Usage:
    python evaluate.py --benchmark WCHW --split test
    python evaluate.py --benchmark WCNS --split validate
    python evaluate.py --benchmark WCMSA --split test

This script provides a simple interface to evaluate predictions against
WirelessBench ground truth. You can use it in two ways:

1. Evaluate a predictions file (JSONL format):
    python evaluate.py --benchmark WCHW --split test --predictions results.jsonl

2. Evaluate with a custom agent callable (programmatic usage):
    See examples/run_evaluation.py
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Any

from benchmarks.wchw import WCHWBenchmark
from benchmarks.wcns import WCNSBenchmark
from benchmarks.wcmsa import WCMSABenchmark
from scripts.logs import logger


BENCHMARKS = {
    "WCHW": WCHWBenchmark,
    "WCNS": WCNSBenchmark,
    "WCMSA": WCMSABenchmark,
}

BENCHMARK_DESCRIPTIONS = {
    "WCHW": "Wireless Communication Homework (math problem solving)",
    "WCNS": "Wireless Communication Network Slicing (5G resource allocation)",
    "WCMSA": "Wireless Communication Mobile Service Assurance (proactive allocation)",
}


def load_predictions(prediction_file: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file.
    
    Expected format per line:
        {"question": "...", "prediction": "..."}
    """
    predictions = []
    with open(prediction_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def evaluate_predictions_wchw(benchmark: WCHWBenchmark, 
                               data: List[dict], 
                               predictions: List[dict]) -> float:
    """Evaluate WCHW predictions against ground truth."""
    scores = []
    for item, pred in zip(data, predictions):
        expected = item["answer"]
        prediction = pred.get("prediction", "")
        answer_type = benchmark.classify_answer_type(expected)
        
        if answer_type in ['numeric', 'scientific']:
            expected_val, _ = benchmark.normalize_answer(expected, item["question"])
            pred_val, _ = benchmark.normalize_answer(str(prediction), item["question"])
            if pred_val is None:
                pred_val = benchmark.extract_number(prediction)
            score, _ = benchmark.calculate_score(expected_val, pred_val, answer_type)
        else:
            score, _ = benchmark.calculate_score(expected, prediction, answer_type)
        
        scores.append(score)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg


def evaluate_predictions_wcns(benchmark: WCNSBenchmark,
                                data: List[dict],
                                predictions: List[dict]) -> float:
    """Evaluate WCNS predictions against ground truth."""
    scores = []
    for item, pred in zip(data, predictions):
        expected = item["answer"]
        prediction = pred.get("prediction", "")
        score, _ = benchmark.calculate_score(expected, str(prediction))
        scores.append(score)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg


def evaluate_predictions_wcmsa(benchmark: WCMSABenchmark,
                                 data: List[dict],
                                 predictions: List[dict]) -> float:
    """Evaluate WCMSA predictions against ground truth."""
    scores = []
    for item, pred in zip(data, predictions):
        expected = item["answer"]
        prediction = pred.get("prediction", "")
        score, _ = benchmark.calculate_score(expected, str(prediction))
        scores.append(score)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg


async def load_data(file_path: str) -> List[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="WirelessBench - Wireless Communication Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Evaluate WCHW test predictions:
    python evaluate.py --benchmark WCHW --split test --predictions my_results.jsonl

  List available benchmarks:
    python evaluate.py --list

  Show dataset statistics:
    python evaluate.py --benchmark WCHW --split test --stats
        """
    )
    
    parser.add_argument("--benchmark", "-b", type=str, choices=BENCHMARKS.keys(),
                        help="Benchmark to evaluate: WCHW, WCNS, or WCMSA")
    parser.add_argument("--split", "-s", type=str, choices=["test", "validate"],
                        default="test", help="Dataset split (default: test)")
    parser.add_argument("--predictions", "-p", type=str,
                        help="Path to predictions JSONL file")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results",
                        help="Output directory for results (default: evaluation_results)")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics and exit")
    
    args = parser.parse_args()
    
    # List benchmarks
    if args.list:
        print("\n=== WirelessBench - Available Benchmarks ===\n")
        for name, desc in BENCHMARK_DESCRIPTIONS.items():
            print(f"  {name}: {desc}")
        print("\nDataset splits: test, validate")
        print("Data location:  data/datasets/<benchmark>_<split>.jsonl")
        print()
        return
    
    if not args.benchmark:
        parser.print_help()
        return
    
    # Data path
    data_path = f"data/datasets/{args.benchmark.lower()}_{args.split}.jsonl"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found: {data_path}")
        print("Make sure the dataset files are in the data/datasets/ directory.")
        sys.exit(1)
    
    # Load data
    data = asyncio.run(load_data(data_path))
    
    # Show statistics
    if args.stats:
        print(f"\n=== {args.benchmark} ({args.split}) Dataset Statistics ===\n")
        print(f"  Total samples: {len(data)}")
        print(f"  Data file:     {data_path}")
        
        # Show sample
        if data:
            sample = data[0]
            print(f"\n  Sample question: {sample.get('question', 'N/A')[:100]}...")
            answer = sample.get('answer', 'N/A')
            if isinstance(answer, dict):
                print(f"  Sample answer:   {json.dumps(answer)[:100]}...")
            else:
                print(f"  Sample answer:   {str(answer)[:100]}...")
        print()
        return
    
    # Evaluate predictions
    if not args.predictions:
        print(f"Error: --predictions file required for evaluation.")
        print(f"Format: JSONL with fields 'question' and 'prediction' per line.")
        sys.exit(1)
    
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    predictions = load_predictions(args.predictions)
    
    if len(predictions) != len(data):
        print(f"Warning: {len(predictions)} predictions vs {len(data)} ground truth samples.")
        print(f"Evaluating first {min(len(predictions), len(data))} samples.")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize benchmark
    benchmark_cls = BENCHMARKS[args.benchmark]
    benchmark = benchmark_cls(
        name=args.benchmark,
        file_path=data_path,
        log_path=args.output
    )
    
    # Run evaluation
    print(f"\n=== Evaluating {args.benchmark} ({args.split}) ===\n")
    
    if args.benchmark == "WCHW":
        avg_score = evaluate_predictions_wchw(benchmark, data, predictions)
    elif args.benchmark == "WCNS":
        avg_score = evaluate_predictions_wcns(benchmark, data, predictions)
    elif args.benchmark == "WCMSA":
        avg_score = evaluate_predictions_wcmsa(benchmark, data, predictions)
    
    print(f"  Average Score: {avg_score:.4f}")
    print(f"  Samples:       {min(len(predictions), len(data))}")
    print(f"  Results saved to: {args.output}/")
    print()
    
    # Save summary
    summary = {
        "benchmark": args.benchmark,
        "split": args.split,
        "num_samples": min(len(predictions), len(data)),
        "average_score": round(avg_score, 5),
    }
    
    summary_path = os.path.join(args.output, f"{args.benchmark}_{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
