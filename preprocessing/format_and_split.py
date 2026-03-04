# -*- coding: utf-8 -*-
"""
Dataset Formatter and Splitter
==============================

Normalises field order, translates residual Chinese text to English,
and splits a JSONL dataset into test / validation subsets.

Usage:
    python -m preprocessing.format_and_split --input data/raw/wchw_expanded.jsonl
    python -m preprocessing.format_and_split --input data.jsonl --ratio 0.8
    python -m preprocessing.format_and_split --input data.jsonl \
        --output-test data/datasets/wchw_test.jsonl \
        --output-validate data/datasets/wchw_validate.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm

# ============================================================
# Chinese → English translation map for residual Chinese text
# ============================================================

_TRANSLATIONS: Dict[str, str] = {
    # Technical terms
    "信道": "channel", "带宽": "bandwidth", "频率": "frequency",
    "调制": "modulation", "编码": "coding", "信噪比": "SNR",
    "比特率": "bit rate", "误码率": "BER", "功率": "power",
    "信号": "signal", "噪声": "noise", "接收": "receive",
    "发送": "transmit", "载波": "carrier", "符号": "symbol",
    "相位": "phase", "幅度": "amplitude",
    # Units
    "赫兹": "Hz", "千赫": "kHz", "兆赫": "MHz", "吉赫": "GHz",
    "比特": "bit", "字节": "byte", "分贝": "dB", "瓦特": "W", "毫瓦": "mW",
    # Structural
    "给定": "Given", "求": "Find", "计算": "Calculate",
    "确定": "Determine", "已知": "Known", "解": "Solution",
    "结果": "Result", "公式": "Formula", "代入": "Substitute",
    "因此": "Therefore", "所以": "Thus", "最终": "Final",
    "样本": "Sample", "题目": "question", "答案": "answer",
    "解析": "solution", "步骤": "Step",
}


def translate_chinese_to_english(text: str) -> str:
    """Replace known Chinese tokens with English equivalents."""
    for zh, en in _TRANSLATIONS.items():
        text = text.replace(zh, en)
    return text


# ============================================================
# Core helpers
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load records from a JSONL file, skipping malformed lines."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping line {num} – {str(exc)[:80]}")
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    """Write records to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"  Saved {len(data)} records → {path}")


def normalize_record(item: Dict) -> Dict:
    """Ensure consistent field order (question → answer → cot → id)
    and translate residual Chinese."""
    return {
        "question": translate_chinese_to_english(str(item.get("question", ""))),
        "answer":   translate_chinese_to_english(str(item.get("answer", ""))),
        "cot":      translate_chinese_to_english(str(item.get("cot", ""))),
        "id":       str(item.get("id", "")),
    }


def reassign_ids(data: List[Dict], prefix: str = "Q") -> List[Dict]:
    """Assign sequential IDs with the given prefix."""
    for i, item in enumerate(data, 1):
        item["id"] = f"{prefix}_{i}"
    return data


def split_dataset(
    data: List[Dict],
    test_ratio: float = 0.75,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle and split data into (test, validate) subsets."""
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * test_ratio)
    return shuffled[:cut], shuffled[cut:]


# ============================================================
# Pipeline
# ============================================================

def format_and_split(
    input_file: str,
    output_test: str,
    output_validate: str,
    test_ratio: float = 0.75,
    seed: int = 42,
) -> None:
    """Full pipeline: load → normalise → split → save."""

    print("=" * 70)
    print("Dataset Formatting & Splitting")
    print("=" * 70)

    # 1. Load
    print(f"\n1. Loading {input_file} …")
    raw = load_jsonl(input_file)
    print(f"   {len(raw)} records loaded")

    # 2. Normalise
    print("2. Normalising fields & translating Chinese …")
    clean = []
    for item in tqdm(raw, desc="   Processing"):
        rec = normalize_record(item)
        if rec["question"] and rec["answer"]:
            clean.append(rec)
    print(f"   {len(clean)} valid records retained")

    # 3. Split
    print(f"3. Splitting (test {test_ratio:.0%} / validate {1 - test_ratio:.0%}) …")
    test, val = split_dataset(clean, test_ratio, seed)
    print(f"   Test: {len(test)}  |  Validate: {len(val)}")

    # 4. Re-ID
    print("4. Reassigning sequential IDs …")
    test = reassign_ids(test, prefix="test")
    val  = reassign_ids(val,  prefix="val")

    # 5. Save
    print("5. Saving …")
    save_jsonl(test, output_test)
    save_jsonl(val,  output_validate)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Input:     {input_file} ({len(raw)})")
    print(f"  Test:      {output_test} ({len(test)})")
    print(f"  Validate:  {output_validate} ({len(val)})")
    print(f"{'=' * 70}\n")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Format and split wireless dataset into test / validate JSONL",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-test", default="data/datasets/wchw_test.jsonl")
    parser.add_argument("--output-validate", default="data/datasets/wchw_validate.jsonl")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Fraction for test set (default 0.75)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        parser.error("--ratio must be in (0, 1)")
    if not Path(args.input).exists():
        parser.error(f"File not found: {args.input}")

    format_and_split(
        input_file=args.input,
        output_test=args.output_test,
        output_validate=args.output_validate,
        test_ratio=args.ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
