# -*- coding: utf-8 -*-
"""
Wireless Dataset Visualisation
==============================

Generates publication-quality charts that characterise the WCHW dataset:
  - Dataset split (pie)
  - Question-type distribution (pie + bar)
  - Knowledge-point coverage (horizontal bar + pie)
  - Difficulty distribution (bar)
  - Full text report

Usage:
    python -m preprocessing.visualize \
        --test data/datasets/wchw_test.jsonl \
        --val  data/datasets/wchw_validate.jsonl \
        --outdir figures/

    python -m preprocessing.visualize --help
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Matplotlib defaults (academic style)
# ============================================================
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ============================================================
# Analyser
# ============================================================

class WirelessDatasetAnalyzer:
    """Comprehensive visualisation analyser for the WCHW dataset."""

    # --- Knowledge-point keyword taxonomy ---
    KNOWLEDGE_POINTS: Dict[str, List[str]] = {
        "Channel Capacity & Info Theory": [
            "shannon", "capacity", "entropy", "information",
            "mutual information", "spectral efficiency", "bsc",
        ],
        "Modulation & Demodulation": [
            "bpsk", "qpsk", "qam", "psk", "fsk", "ask", "ook",
            "modulation", "demodulation", "coherent", "noncoherent",
        ],
        "Channel Modelling & Propagation": [
            "free-space", "two-ray", "path loss", "shadowing",
            "fading", "rayleigh", "rician", "log-normal",
            "multipath", "delay spread", "doppler", "channel gain",
        ],
        "Error Control Coding": [
            "hamming", "linear code", "block code", "convolutional",
            "parity", "syndrome", "error correction", "coding gain",
        ],
        "Signal Processing & Detection": [
            "snr", "ber", "eb/n0", "noise", "awgn", "detection",
            "receiver", "matched filter", "probability of error",
        ],
        "Analog Modulation": [
            "am", "fm", "pm", "dsb", "ssb", "vsb", "carrier",
            "envelope", "phase deviation", "frequency deviation",
        ],
        "Digital Signal Processing": [
            "pcm", "quantization", "sampling", "nyquist",
            "raised-cosine", "pulse shaping", "roll-off", "isi", "baseband",
        ],
        "Multiple Access & Cellular": [
            "tdma", "fdma", "cdma", "ofdm", "cell", "handoff",
            "cellular", "reuse", "cluster", "frequency reuse",
        ],
        "Power & Link Budget": [
            "transmit power", "received power", "dbm", "db",
            "gain", "loss", "attenuation", "thermal noise", "noise figure",
        ],
    }

    QUESTION_TYPES: Dict[str, List[str]] = {
        "Calculation":    ["compute", "calculate", "find", "determine", "estimate"],
        "Conversion":     ["convert", "express", "transform"],
        "Design":         ["design", "require", "minimum", "maximum", "optimal"],
        "Analysis":       ["analyze", "compare", "explain", "derive", "show"],
        "Specification":  ["specify", "what is", "list", "identify"],
    }

    def __init__(self, test_file: str, val_file: str, outdir: str = "."):
        self.test_file = test_file
        self.val_file = val_file
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.test_data: List[Dict] = []
        self.val_data: List[Dict] = []
        self.all_data: List[Dict] = []

    # ------ data loading ------

    def load_data(self) -> None:
        with open(self.test_file, "r", encoding="utf-8") as f:
            self.test_data = [json.loads(l) for l in f if l.strip()]
        with open(self.val_file, "r", encoding="utf-8") as f:
            self.val_data = [json.loads(l) for l in f if l.strip()]
        self.all_data = self.test_data + self.val_data
        print(f"Loaded  test={len(self.test_data)}  val={len(self.val_data)}  total={len(self.all_data)}")

    # ------ classifiers ------

    def classify_question_type(self, question: str) -> str:
        q = question.lower()
        for qtype, kws in self.QUESTION_TYPES.items():
            if any(k in q for k in kws):
                return qtype
        return "Other"

    def classify_knowledge_point(self, question: str) -> List[str]:
        q = question.lower()
        hits = [kp for kp, kws in self.KNOWLEDGE_POINTS.items() if any(k in q for k in kws)]
        return hits or ["General"]

    def estimate_difficulty(self, item: Dict) -> str:
        question = item.get("question", "")
        cot = item.get("cot", "")
        combined = (question + " " + cot).lower()

        steps = len(re.findall(r"step \d+:", cot, re.I))
        complex_ops = sum(
            1 for p in (r"exp\(", r"log", r"sqrt", r"sin\(", r"cos\(", r"integral", r"matrix")
            if re.search(p, combined)
        )
        word_count = len(question.split())

        score = 0
        score += min(steps // 2, 3)
        score += min(complex_ops, 2)
        score += 1 if word_count > 25 else 0

        if score <= 2:
            return "Easy"
        if score <= 4:
            return "Medium"
        return "Hard"

    # ------ aggregation ------

    def analyze_question_types(self) -> Counter:
        return Counter(self.classify_question_type(d["question"]) for d in self.all_data)

    def analyze_knowledge_points(self) -> Counter:
        c: Counter = Counter()
        for d in self.all_data:
            for kp in self.classify_knowledge_point(d["question"]):
                c[kp] += 1
        return c

    def analyze_difficulty(self) -> Counter:
        return Counter(self.estimate_difficulty(d) for d in self.all_data)

    # ------ plotting helpers ------

    def _save(self, fig, name: str) -> None:
        path = self.outdir / name
        fig.savefig(str(path), dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"  Saved {path}")

    def plot_pie(self, counter: Counter, title: str, filename: str) -> None:
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        labels = [i[0] for i in sorted_items]
        sizes  = [i[1] for i in sorted_items]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))

        fig, ax = plt.subplots(figsize=(8, 5))
        wedges, _, autotexts = ax.pie(
            sizes, labels=None, autopct="%1.1f%%", startangle=140,
            colors=colors, pctdistance=0.78, explode=[0.008] * len(sizes), radius=0.98,
        )
        for t in autotexts:
            t.set(color="white", fontweight="bold", fontsize=8.5)
        ax.legend(wedges, labels, title="Categories", loc="center left",
                  bbox_to_anchor=(0.96, 0.5), fontsize=8.5, title_fontsize=9.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=3, y=0.99)
        plt.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.68)
        self._save(fig, filename)

    def plot_bar(self, counter: Counter, title: str, xlabel: str,
                 ylabel: str, filename: str, horizontal: bool = False) -> None:
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        labels = [i[0] for i in sorted_items]
        values = [i[1] for i in sorted_items]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))

        fig, ax = plt.subplots(figsize=(12, 8))
        if horizontal:
            bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            for i, (bar, v) in enumerate(zip(bars, values)):
                ax.text(v + max(values) * 0.01, i, str(v), va="center", fontsize=10, fontweight="bold")
        else:
            bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(axis="x" if horizontal else "y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        self._save(fig, filename)

    def plot_dataset_split(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sizes = [len(self.test_data), len(self.val_data)]
        colors = plt.cm.viridis([0.3, 0.7])
        wedges, _, autotexts = ax.pie(
            sizes, labels=None,
            autopct=lambda p: f"{p:.1f}%\n({int(p / 100 * sum(sizes))})",
            startangle=90, colors=colors, pctdistance=0.65,
            explode=[0.05, 0.05], radius=0.98,
        )
        for t in autotexts:
            t.set(color="white", fontweight="bold", fontsize=10)
        ax.legend(wedges, ["Test Set", "Validation Set"], title="Dataset Split",
                  loc="center left", bbox_to_anchor=(0.85, 0.5), fontsize=10, title_fontsize=11)
        ax.set_title("Dataset Split Distribution", fontsize=12, fontweight="bold", pad=3, y=0.99)
        plt.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.62)
        self._save(fig, "dataset_split.png")

    # ------ report ------

    def generate_report(self) -> str:
        lines = [
            "=" * 80,
            "WIRELESS DATASET ANALYSIS REPORT",
            "=" * 80,
            f"\nTotal: {len(self.all_data)}  (test={len(self.test_data)}, val={len(self.val_data)})",
        ]

        lines += ["\n" + "-" * 80, "Question Type Distribution:", "-" * 80]
        for qt, cnt in self.analyze_question_types().most_common():
            lines.append(f"  {qt:20s}: {cnt:4d}  ({cnt / len(self.all_data) * 100:5.1f}%)")

        lines += ["\n" + "-" * 80, "Knowledge Point Coverage:", "-" * 80]
        for kp, cnt in self.analyze_knowledge_points().most_common():
            lines.append(f"  {kp:42s}: {cnt:4d}  ({cnt / len(self.all_data) * 100:5.1f}%)")

        lines += ["\n" + "-" * 80, "Difficulty Distribution:", "-" * 80]
        for lv in ("Easy", "Medium", "Hard"):
            cnt = self.analyze_difficulty().get(lv, 0)
            lines.append(f"  {lv:20s}: {cnt:4d}  ({cnt / len(self.all_data) * 100:5.1f}%)")

        lines.append("\n" + "=" * 80)
        report = "\n".join(lines)

        report_path = self.outdir / "analysis_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"  Saved {report_path}")
        return report

    # ------ main pipeline ------

    def run(self) -> None:
        self.load_data()
        print("\nGenerating visualisations …")

        self.plot_dataset_split()

        qt = self.analyze_question_types()
        self.plot_pie(qt, "Question Type Distribution", "question_types_pie.png")
        self.plot_bar(qt, "Question Type Distribution", "Question Type", "Count", "question_types_bar.png")

        kp = self.analyze_knowledge_points()
        self.plot_bar(kp, "Knowledge Point Coverage", "Count", "Knowledge Point",
                      "knowledge_points_bar.png", horizontal=True)
        self.plot_pie(kp, "Knowledge Point Distribution", "knowledge_points_pie.png")

        report = self.generate_report()
        print(report)
        print("\nAnalysis complete.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualise WCHW dataset statistics")
    parser.add_argument("--test", default="data/datasets/wchw_test.jsonl", help="Test JSONL")
    parser.add_argument("--val",  default="data/datasets/wchw_validate.jsonl", help="Validation JSONL")
    parser.add_argument("--outdir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    for p in (args.test, args.val):
        if not Path(p).exists():
            parser.error(f"File not found: {p}")

    analyzer = WirelessDatasetAnalyzer(args.test, args.val, args.outdir)
    analyzer.run()


if __name__ == "__main__":
    main()
