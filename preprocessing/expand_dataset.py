# -*- coding: utf-8 -*-
"""
LLM-Powered Dataset Expansion for WirelessBench
=================================================

Generates new wireless communication questions by calling an LLM with
carefully engineered prompts that rotate through knowledge domains.

Usage:
    python -m preprocessing.expand_dataset --input data/datasets/wchw_validate.jsonl \\
        --output wchw_expanded.jsonl --target 500 --model gpt-4o

    python -m preprocessing.expand_dataset --list-models
    python -m preprocessing.expand_dataset --dry-run --target 500
"""

import json
import re
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "gpt-4o"
BATCH_SIZE = 5

# Knowledge domains cover the full WCHW scope
KNOWLEDGE_DOMAINS = [
    "Digital modulation (BPSK, QPSK, 16-QAM, 64-QAM, FSK, MSK, GMSK)",
    "Analog modulation (AM, FM, PM, DSB, SSB, VSB)",
    "Information theory (entropy, mutual information, channel capacity, coding theorem)",
    "Channel models (AWGN, Rayleigh fading, Rician fading, multipath fading)",
    "PCM & quantization (sampling theorem, quantization error, SQNR, A-law / mu-law)",
    "Filter design (matched filter, raised-cosine filter, equalizer)",
    "Channel coding (Hamming code, convolutional code, Turbo code, LDPC)",
    "Spread spectrum (DSSS, FHSS, processing gain)",
    "OFDM (subcarrier, cyclic prefix, PAPR, spectral efficiency)",
    "Antenna & propagation (free-space loss, Doppler shift, coherence time)",
]

# ============================================================
# Prompt Template (English)
# ============================================================

EXPANSION_PROMPT = """You are a senior expert in wireless communication principles with extensive experience in designing educational assessments. Your task is to systematically expand a high-quality exercise dataset based on the provided sample questions.


## Reference Samples
The following samples demonstrate the expected style, difficulty level, and formatting conventions (for reference only — do not copy directly):
{sample_questions}

## Batch Task Specification
- Number of questions to generate: {batch_size}
- Focus knowledge domain: {focus_domain}
- Starting ID: expand_{start_id}

## Expansion Strategies (Apply these methods in a balanced manner)

### Strategy 1: Systematic Parameter Variation
- Replace key numerical parameters while preserving the question structure
- Values must fall within engineering-realistic ranges (e.g., bandwidth 10 kHz–10 MHz, symbol rate 1 kbps–10 Mbps)
- Avoid special values that make calculations trivially simple

### Strategy 2: Bidirectional Problem Type Conversion
- Transform "given A, find B" into "given B, find A" or "given A and B, find C"
- E.g.: Given SNR → find BER ⟺ Given target BER → find minimum required SNR

### Strategy 3: Cross-Topic Knowledge Integration
- Combine 2–3 related concepts into comprehensive problems
- E.g.: Source encoding → Modulation → Channel → Demodulation (end-to-end analysis)

### Strategy 4: Concept Deepening and Boundary Exploration
- Design questions exploring the applicability boundaries of core formulas
- E.g.: Limiting conditions (SNR→∞ or SNR→0), practical constraints (quantization error)

## Quality Requirements

### Question Design Standards
1. **Parameter validity**: All values must reflect realistic engineering scenarios
2. **Computational feasibility**: Avoid calculations requiring table lookups or numerical integration
3. **Unambiguous wording**: Specify all necessary conditions
4. **Appropriate difficulty**: Solution should require 3–6 steps

### Chain-of-Thought (CoT) Standards
1. Step 1: State given conditions and objective
2. Intermediate steps: List formulas → Substitute values → Compute step by step
3. Final step: Present the answer with appropriate units
4. Show unit conversions explicitly (MHz↔Hz, dB↔linear, bps↔symbol rate)

## Diversity Constraints
- No single strategy should account for more than 40% of generated questions
- Adjacent questions should involve different core formulas
- Avoid parameter values that differ from samples by only simple integer multiples

## Output Format
Output strictly in JSON Lines format — one complete JSON object per line, no additional text:

{{"question": "Complete question description", "answer": "Concise answer (with units)", "cot": "Step 1: ...\\nStep 2: ...\\nFinal answer: ...", "id": "expand_{start_id}"}}

Begin generating {batch_size} questions:"""

# ============================================================
# Core Functions
# ============================================================

def load_config(config_path: str) -> Dict:
    """Load model configuration from YAML file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("models", config)


def load_dataset(path: str) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_dataset(data: List[Dict], path: str):
    """Save dataset as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} records to {path}")


def call_llm(client: OpenAI, model_name: str, prompt: str, max_retries: int = 3) -> str:
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a wireless communication expert skilled at designing "
                            "high-quality exercises and chain-of-thought solutions. "
                            "Output JSON Lines format, one JSON object per line."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return ""


def parse_llm_response(response: str, start_id: int) -> List[Dict]:
    """Parse LLM response to extract JSON objects."""
    results = []

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    obj.setdefault("cot", "")
                    obj.setdefault("id", f"expand_{start_id + len(results)}")
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # Fallback: regex extraction
    if not results:
        for match in re.findall(r'\{[^{}]*"question"[^{}]*\}', response, re.DOTALL):
            try:
                obj = json.loads(match)
                if "question" in obj and "answer" in obj:
                    obj.setdefault("cot", "")
                    obj.setdefault("id", f"expand_{start_id + len(results)}")
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    return results


def expand_dataset(
    original_data: List[Dict],
    target_count: int,
    client: OpenAI,
    model_name: str,
    batch_size: int = BATCH_SIZE,
) -> List[Dict]:
    """Expand dataset to *target_count* items using LLM generation."""
    current_count = len(original_data)
    needed = target_count - current_count
    if needed <= 0:
        print(f"Dataset already has {current_count} records (target: {target_count})")
        return original_data

    print(f"\n{'=' * 60}")
    print(f"Dataset Expansion — {current_count} → {target_count}  (generate {needed})")
    print(f"Model: {model_name}  |  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    expanded = original_data.copy()
    generated = 0
    batch_num = 0
    pbar = tqdm(total=needed, desc="Generating")

    while generated < needed:
        batch_num += 1
        samples = random.sample(original_data, min(3, len(original_data)))
        sample_text = "\n".join(
            f"Sample {i + 1}: {json.dumps(s, ensure_ascii=False)}" for i, s in enumerate(samples)
        )

        domain_idx = batch_num % len(KNOWLEDGE_DOMAINS)
        focus_domain = KNOWLEDGE_DOMAINS[domain_idx]
        start_id = current_count + generated + 1
        remaining = min(batch_size, needed - generated)

        prompt = EXPANSION_PROMPT.format(
            batch_size=remaining,
            sample_questions=sample_text,
            focus_domain=focus_domain,
            start_id=start_id,
        )

        response = call_llm(client, model_name, prompt)
        if response:
            new_items = parse_llm_response(response, start_id)
            if new_items:
                for i, item in enumerate(new_items):
                    item["id"] = f"expand_{start_id + i}"
                expanded.extend(new_items)
                generated += len(new_items)
                pbar.update(len(new_items))
            else:
                tqdm.write(f"  Batch {batch_num}: failed to parse response")
        else:
            tqdm.write(f"  Batch {batch_num}: API call failed")

        time.sleep(1)
        if len(expanded) >= target_count:
            break

    pbar.close()
    return expanded[:target_count]


def validate_dataset(data: List[Dict]) -> List[Dict]:
    """Remove duplicates and records missing required fields."""
    seen = set()
    valid = []
    for item in data:
        if "question" not in item or "answer" not in item:
            continue
        q_hash = item["question"][:100]
        if q_hash in seen:
            continue
        seen.add(q_hash)
        item.setdefault("cot", "")
        valid.append(item)
    removed = len(data) - len(valid)
    if removed:
        print(f"Removed {removed} invalid / duplicate records")
    return valid


def reassign_ids(data: List[Dict], prefix: str = "val_") -> List[Dict]:
    """Assign sequential IDs."""
    for i, item in enumerate(data, 1):
        item["id"] = f"{prefix}{i}"
    return data


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Expand WirelessBench WCHW dataset using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="data/datasets/wchw_validate.jsonl")
    parser.add_argument("--output", type=str, default="wchw_expanded.jsonl")
    parser.add_argument("--target", type=int, default=500, help="Target record count")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Model config YAML")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without API calls")
    args = parser.parse_args()

    if args.list_models:
        models = load_config(args.config)
        print("\nAvailable models:")
        for name in models:
            print(f"  - {name}")
        return

    random.seed(42)
    models_config = load_config(args.config)
    if args.model not in models_config:
        print(f"Error: model '{args.model}' not in config. Available: {list(models_config.keys())}")
        return

    model_cfg = models_config[args.model]
    original = load_dataset(args.input)
    print(f"Loaded {len(original)} records from {args.input}")

    if args.dry_run:
        print("[DRY RUN] Configuration validated.")
        return

    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])

    expanded = expand_dataset(original, args.target, client, args.model, args.batch_size)
    cleaned = validate_dataset(expanded)
    final = reassign_ids(cleaned, prefix="val_")
    save_dataset(final, args.output)


if __name__ == "__main__":
    main()
