# Benchmarking Scripts

Measure model performance and find bottlenecks.

## Quick Setup

```bash
# One-step setup (creates venv, installs deps, downloads models)
bash setup.sh

# Activate environment
source venv/bin/activate

# Quick test
python scripts/demo_inference.py
```

## Scripts

| Script | Function | Flags |
|--------|----------|-------|
| `benchmark_model.py` | Benchmark single model | `--model`, `--device`, `--batch-sizes` |
| `profile_model.py` | Profile single model | `--model`, `--device`, `--all` |
| `plot_results.py` | Visualize results | `--output-dir` |
| `generate_report.py` | Summary report | `--benchmark-dir` |
| `demo_inference.py` | Quick test | none |

## Batch Operations

| Script | Function | Usage |
|--------|----------|-------|
| `benchmark_all.sh` | Benchmark all models | `bash scripts/benchmark_all.sh [cpu\|mps\|cuda]` |
| `profile_all.sh` | Profile all models | `bash scripts/profile_all.sh [cpu\|mps\|cuda]` |
| `report.sh` | Create plots + report | `bash scripts/report.sh` |
| `run_full_suite.sh` | Complete workflow | `bash scripts/run_full_suite.sh [cpu\|mps\|cuda]` |

## Script Details

### benchmark_model.py
Measures inference performance.

**Flags:**
- `--model`: `resnet50`, `gpt2`, `distilbert`
- `--device`: `cpu`, `mps`, `cuda`
- `--batch-sizes`: List of batch sizes (default: 1 4 8)
- `--warmup`: Warmup iterations (default: 20)
- `--iterations`: Measurement iterations (default: 100)

**Output:** CSV/JSON in `week1/benchmarks/`

### profile_model.py
Identifies performance bottlenecks.

**Flags:**
- `--model`: `resnet50`, `gpt2`, `distilbert`
- `--device`: `cpu`, `mps`, `cuda`
- `--batch-size`: Single batch size (default: 8)
- `--all`: Run all profiling tools

**Output:** Chrome trace in `week1/experiments/{model}/trace.json`

### plot_results.py
Generates comparison plots.

**Flags:**
- `--output-dir`: CSV directory (default: week1/benchmarks)

**Output:** `performance_comparison.png`

### generate_report.py
Creates summary table.

**Flags:**
- `--benchmark-dir`: Results directory (default: week1/benchmarks)

### demo_inference.py
Tests models with sample data.

## Usage

```bash
# Quick start - run everything
bash scripts/run_full_suite.sh cpu

# Individual operations
python scripts/benchmark_model.py --model resnet50 --device cpu --batch-sizes 1 4 8
python scripts/profile_model.py --model gpt2 --device mps --all

# Batch operations
bash scripts/benchmark_all.sh cpu
bash scripts/profile_all.sh mps
bash scripts/report.sh
```

## Metrics

## Metrics

- **P50/P90/P99:** Latency percentiles
- **Throughput:** Items per second
- **Memory:** Allocated vs reserved

## Device Notes

- **CPU:** Always available, slower
- **MPS:** Apple Silicon only (M1/M2/M3)
- **CUDA:** Requires NVIDIA GPU

## Outputs

```
week1/
├── benchmarks/         # CSV/JSON results
├── experiments/        # Chrome traces
└── plots/             # Visualizations
```

## Troubleshooting

**Device not available:** Check hardware/drivers  
**Out of memory:** Reduce batch sizes  
**No results:** Run benchmarks first
