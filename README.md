# Inference Optimization

ML inference benchmarking and profiling toolkit for PyTorch models.

## Setup

```bash
# Local setup
bash setup.sh
source venv/bin/activate

# Or use Docker (for CUDA)
docker-compose up -d
docker-compose exec optimizer bash

# Quick validation
python test_models.py
python scripts/demo_inference.py
```

## Quick Start

```bash
# Benchmark a model
python scripts/benchmark_model.py --model resnet50 --device cpu --batch-sizes 1 4 8

# Profile for bottlenecks
python scripts/profile_model.py --model gpt2 --device mps --all

# Run complete suite (benchmark + profile + report)
bash scripts/run_full_suite.sh cpu

# Individual batch operations
bash scripts/benchmark_all.sh cpu    # All models
bash scripts/profile_all.sh mps      # All models
bash scripts/report.sh               # Generate report

# Visualize
python scripts/plot_results.py
```

## Project Structure

```
src/
├── benchmarking/      # Latency, throughput, memory measurement
├── models/           # ResNet50, GPT-2, DistilBERT wrappers
├── profiling/        # PyTorch profiler, roofline, layer timing
└── utils/            # Logging, data loading, timing utilities

scripts/
├── benchmark_model.py    # Single model benchmarking
├── profile_model.py      # Single model profiling
├── plot_results.py       # Visualization
├── generate_report.py    # Summary report
├── demo_inference.py     # Quick test
├── benchmark_all.sh      # Batch benchmark
├── profile_all.sh        # Batch profile
├── report.sh            # Generate report + plots
└── run_full_suite.sh    # Complete workflow

week1/
├── benchmarks/       # CSV/JSON results
├── experiments/      # Profiling traces
└── plots/           # Visualizations
```

## Supported Models

- ResNet50 (image classification)
- GPT-2 (text generation)
- DistilBERT (NLP embeddings)

## Supported Devices

- CPU (always available)
- MPS (Apple Silicon M1/M2/M3)
- CUDA (NVIDIA GPUs)

## Documentation

- [README.md](README.md) - This file (overview and quick start)
- [scripts/README.md](scripts/README.md) - Detailed script documentation
