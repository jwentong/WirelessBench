<p align="center">
  <h1 align="center">WirelessBench</h1>
  <p align="center">
    A Comprehensive Benchmark for Evaluating LLM-based Agents on Wireless Communication Tasks
  </p>
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#dataset-format">Dataset Format</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**WirelessBench** is a benchmark suite for evaluating the capabilities of Large Language Models (LLMs) and LLM-based agents on real-world wireless communication tasks. It covers three core domains of 5G and beyond wireless systems:

| Benchmark | Full Name | Test | Validate | Task Type |
|-----------|-----------|------|----------|-----------|
| **WCHW** | Wireless Comm Homework | 1,044 | 348 | Math problem solving |
| **WCNS** | Wireless Comm Network Slicing | 750 | 250 | 5G resource allocation |
| **WCMSA** | Wireless Comm Mobile Service Assurance | 750 | 250 | Proactive resource management |

**Total: 3,392 evaluation samples** across three benchmarks.

## Benchmarks

### WCHW — Wireless Communication Homework

Tests fundamental wireless communication knowledge through quantitative problem solving.

**Topics covered:**
- Shannon channel capacity
- Bit Error Rate (BER) computation
- Modulation & demodulation (AM, FM, PM, QAM, PSK)
- Fading channel analysis (Rayleigh, Rician)
- Error-correcting codes (Hamming, convolutional)
- Signal-to-noise ratio (SNR) calculations
- Nyquist bandwidth & sampling theory

**Answer types:** Numeric (with units), mathematical formulas, scientific notation, text descriptions.

**Scoring:** Relative error based (1%/5%/10% tolerance tiers) for numeric answers; symbolic matching for formulas; keyword matching for text answers.

### WCNS — Wireless Communication Network Slicing

Tests the ability to perform 5G network slicing decisions including service classification, bandwidth allocation, and throughput calculation.

**Task pipeline:**
1. **Service Classification** → eMBB or URLLC
2. **CQI Determination** → Channel Quality Indicator via ray tracing
3. **Bandwidth Allocation** → Proportional fairness: `BW = SliceCapacity / (ExistingUsers + 1)`
4. **Throughput Calculation** → Shannon formula: `Throughput = 10 × B × log₁₀(1 + 10^(CQI/10))`

**Scoring weights:** Slice Type (25%) + CQI (15%) + Bandwidth (35%) + Throughput (25%)

### WCMSA — Wireless Communication Mobile Service Assurance

Tests proactive 5G resource allocation with mobility prediction. This is the most complex benchmark, requiring multi-step reasoning with tool usage.

**Task pipeline:**
1. **Position Prediction** → Kalman filter trajectory forecasting
2. **CQI Prediction** → Ray tracing from predicted position
3. **Service Classification** → eMBB / URLLC
4. **Bandwidth Allocation** → Based on predicted channel conditions
5. **Throughput Calculation** → Shannon formula
6. **QoS Verification** → Check if requirements are met

**Scoring weights:** Position (15%) + CQI (15%) + Slice Type (20%) + Bandwidth (25%) + Throughput (20%) + QoS (5%)

## Quick Start

### Installation

```bash
git clone https://github.com/jwentong/WirelessBench.git
cd WirelessBench
pip install -r requirements.txt
```

### Project Structure

```
WirelessBench/
├── evaluate.py              # CLI evaluation entry point
├── requirements.txt         # Python dependencies
├── benchmarks/              # Evaluation metrics & scoring
│   ├── benchmark.py         # Base benchmark class
│   ├── wchw.py             # WCHW evaluation
│   ├── wcns.py             # WCNS evaluation
│   ├── wcmsa.py            # WCMSA evaluation
│   └── utils.py            # Utilities
├── data/
│   ├── datasets/            # Benchmark datasets (JSONL)
│   │   ├── wchw_test.jsonl
│   │   ├── wchw_validate.jsonl
│   │   ├── wcns_test.jsonl
│   │   ├── wcns_validate.jsonl
│   │   ├── wcmsa_test.jsonl
│   │   └── wcmsa_validate.jsonl
│   ├── maps/                # OpenStreetMap data (HKUST campus)
│   ├── ray_tracing/         # Pre-computed ray tracing results
│   └── knowledge_base/      # Intent classification knowledge
├── scripts/                 # Supporting tools & utilities
│   ├── evaluator.py         # Evaluator orchestrator
│   ├── tools.py             # Tool infrastructure
│   ├── wireless_tools.py    # Wireless domain tools
│   ├── enhanced_tools.py    # RAG & calculator tools
│   ├── logs.py              # Logging
│   └── telecom_tools/       # Telecom-specific tools
└── config/
    └── config.example.yaml  # LLM configuration template
```

## Evaluation

### CLI Usage

```bash
# List available benchmarks
python evaluate.py --list

# Show dataset statistics
python evaluate.py --benchmark WCHW --split test --stats

# Evaluate predictions
python evaluate.py --benchmark WCHW --split test --predictions my_results.jsonl
python evaluate.py --benchmark WCNS --split test --predictions my_results.jsonl
python evaluate.py --benchmark WCMSA --split test --predictions my_results.jsonl
```

### Predictions File Format

Your predictions file should be a JSONL file with one JSON object per line:

```json
{"question": "Calculate the Shannon capacity...", "prediction": "The capacity is 16 kbit/s"}
{"question": "A user at position (100, 50)...", "prediction": "Slice Type: eMBB, Bandwidth: 10 MHz, Throughput: 45.2 Mbps"}
```

### Programmatic Usage

```python
import asyncio
from benchmarks.wchw import WCHWBenchmark

# Initialize benchmark
benchmark = WCHWBenchmark(
    name="WCHW",
    file_path="data/datasets/wchw_test.jsonl",
    log_path="results/"
)

# Define your agent
async def my_agent(question: str):
    # Your LLM/agent logic here
    answer = "..."
    cost = 0.0  # API cost tracking
    return answer, cost

# Run evaluation
avg_score, avg_cost, total_cost = asyncio.run(
    benchmark.run_baseline(my_agent)
)
print(f"Score: {avg_score:.4f}")
```

## Dataset Format

### WCHW

```json
{
    "question": "A binary channel has a bit rate of 36 kbit/s. The SNR is 20 dB. Calculate the channel capacity using Shannon's formula.",
    "answer": "36 kbit/s"
}
```

### WCNS

```json
{
    "question": "A user requests video streaming service at location (114.265, 22.337) in the HKUST North area. The eMBB slice has 100 MHz capacity with 9 existing users. Determine the network slicing decision.",
    "answer": {
        "slice_type": "eMBB",
        "cqi": 12,
        "bandwidth": 10.0,
        "throughput": 45.23
    }
}
```

### WCMSA

```json
{
    "question": "A mobile user has historical positions [...]. The user requests real-time video service. Predict the next position, determine CQI, and allocate resources proactively.",
    "answer": {
        "predicted_position": {"x": -62.01, "y": 106.82},
        "predicted_cqi": 11,
        "slice_type": "eMBB",
        "bandwidth": 8.33,
        "throughput": 38.5,
        "qos_satisfied": true
    }
}
```

## Scoring Details

### WCHW Scoring

| Answer Type | Method | Score Tiers |
|-------------|--------|-------------|
| Numeric | Relative error | 1.0 (<1%), 0.9 (<5%), 0.7 (<10%), 0.0 (>10%) |
| Formula | Symbolic matching | 1.0 (exact), 0.8 (partial), 0.5 (similar), 0.0 |
| Text | Keyword matching | 1.0 (>80% match), 0.8 (>60%), 0.5 (>40%), 0.0 |

### WCNS Scoring

| Metric | Weight | Method |
|--------|--------|--------|
| Slice Type | 25% | Binary (correct/incorrect) |
| CQI | 15% | Stepped (exact=1.0, ±1=0.8, ±2=0.5) |
| Bandwidth | 35% | Relative error tiers |
| Throughput | 25% | Relative error tiers |

### WCMSA Scoring

| Metric | Weight | Method |
|--------|--------|--------|
| Position | 15% | Euclidean distance (0-20m range) |
| CQI | 15% | Stepped scoring |
| Slice Type | 20% | Binary |
| Bandwidth | 25% | Relative error tiers |
| Throughput | 20% | Relative error tiers |
| QoS | 5% | Binary |

## Supporting Tools

WirelessBench provides domain-specific tools that agents can use during evaluation:

- **Ray Tracing Tool** — Look up CQI values based on geographic coordinates using pre-computed ray tracing data
- **Telecom Formula Retriever** — RAG-based retrieval of relevant wireless communication formulas
- **Telecom Calculator** — Precision calculator for wireless communication computations
- **Python Executor** — Safe Python code execution for complex calculations

See `scripts/tools.py` and `scripts/wireless_tools.py` for tool interfaces.

## Configuration

Copy the example configuration and fill in your LLM API credentials:

```bash
cp config/config.example.yaml config/config.yaml
```

```yaml
models:
  "gpt-4-turbo":
    api_type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "your-api-key"
    temperature: 0
```

## License

This project is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.

See [LICENSE](LICENSE) for details.

## Citation

If you use WirelessBench in your research, please cite:

```bibtex
@misc{wirelessbench2026,
    title={WirelessBench: A Comprehensive Benchmark for LLM-based Wireless Communication Agents},
    author={Jingwen Tong},
    year={2026},
    url={https://github.com/jwentong/WirelessBench}
}
```

## Acknowledgments

The evaluation framework is inspired by [AFlow](https://arxiv.org/abs/2410.10762) (ICLR 2025). The ray tracing data is based on the HKUST campus environment.
