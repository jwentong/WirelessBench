# -*- coding: utf-8 -*-
"""
Text → JSONL Converter
======================

Reads a plain-text file that contains one JSON object per line (or broken
JSON), cleans it, splits it into test / validation subsets, and writes
two JSONL files.

Usage:
    python -m preprocessing.txt_to_jsonl --input raw_questions.txt
    python -m preprocessing.txt_to_jsonl --input raw.txt --ratio 0.8 \
        --output-test wchw_test.jsonl --output-validate wchw_validate.jsonl
"""

import json
import re
import copy
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# ============================================================
# Parsing helpers
# ============================================================

def fix_json_escapes(line: str) -> str:
    """Fix common escape-sequence issues inside quoted strings."""

    def _fix_quoted(match):
        content = match.group(1)
        content = re.sub(r"(?<!\\)\\(?![\\\"/bfnrtu])", r"\\\\", content)
        return f'"{content}"'

    return re.sub(r'"([^"]*)"', _fix_quoted, line)


def load_text_file(path: str) -> List[Dict]:
    """Parse a text file that contains one JSON object per line.

    Falls back to ``fix_json_escapes`` when the raw line is invalid JSON.
    """
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    data.append(json.loads(fix_json_escapes(line)))
                except (json.JSONDecodeError, Exception) as exc:
                    print(f"  Warning: skipping line {num} – {str(exc)[:80]}")
    print(f"  Loaded {len(data)} records from {path}")
    return data


# ============================================================
# ID assignment & splitting
# ============================================================

def assign_ids(data: List[Dict], prefix: str = "", start: int = 1) -> List[Dict]:
    """Assign sequential IDs with an optional prefix."""
    for i, item in enumerate(data, start):
        item["id"] = f"{prefix}{i}" if prefix else i
    return data


def split_data(
    data: List[Dict], test_ratio: float = 0.75, seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Randomly split data into (test, validate), assigning IDs to each."""
    rng = random.Random(seed)
    shuffled = copy.deepcopy(data)
    rng.shuffle(shuffled)

    cut = int(len(shuffled) * test_ratio)
    test = assign_ids(shuffled[:cut], prefix="test_")
    val  = assign_ids(shuffled[cut:], prefix="val_")
    return test, val


# ============================================================
# IO
# ============================================================

def save_jsonl(data: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"  Saved {len(data)} records → {path}")


# ============================================================
# Verification
# ============================================================

def verify_split(original: List[Dict], test: List[Dict], val: List[Dict]) -> None:
    """Print basic integrity checks after splitting."""
    total = len(test) + len(val)
    print(f"\n  Original: {len(original)}  |  Test: {len(test)}  |  Val: {len(val)}  |  Sum: {total}")
    if total != len(original):
        print(f"  WARNING: {abs(len(original) - total)} records lost!")
    else:
        print("  OK – no data loss")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a raw text file (one JSON per line) to split JSONL datasets",
    )
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output-test", default="data/datasets/wchw_test.jsonl")
    parser.add_argument("--output-validate", default="data/datasets/wchw_validate.jsonl")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Fraction for test set (default 0.75)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f"File not found: {args.input}")
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be in (0, 1)")

    random.seed(args.seed)

    data = load_text_file(args.input)
    if not data:
        print("No valid records found – exiting.")
        return

    # Show field summary
    sample = data[0]
    print(f"  Fields: {list(sample.keys())}")
    print(f"  Sample question: {str(sample.get('question', ''))[:100]}…")

    test, val = split_data(data, args.ratio, args.seed)
    save_jsonl(test, args.output_test)
    save_jsonl(val, args.output_validate)
    verify_split(data, test, val)

    print("\nDone.")


if __name__ == "__main__":
    main()
