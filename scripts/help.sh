#!/bin/bash
# Quick reference for all available commands

cat << 'EOF'
Inference Optimization Toolkit - Command Reference
==================================================

SETUP
-----
bash setup.sh                     # One-time setup (venv + deps + models)
source venv/bin/activate          # Activate environment

QUICK TEST
----------
python scripts/demo_inference.py  # Test all models work

SINGLE MODEL
------------
python scripts/benchmark_model.py --model MODEL --device DEVICE --batch-sizes 1 4 8
python scripts/profile_model.py --model MODEL --device DEVICE --all

BATCH OPERATIONS
----------------
bash scripts/benchmark_all.sh DEVICE    # Benchmark all models
bash scripts/profile_all.sh DEVICE      # Profile all models
bash scripts/report.sh                  # Generate plots + report
bash scripts/run_full_suite.sh DEVICE   # Complete workflow

MODELS
------
resnet50, gpt2, distilbert

DEVICES
-------
cpu, mps, cuda

OUTPUTS
-------
week1/benchmarks/     # CSV/JSON benchmark results
week1/experiments/    # Profiling traces (chrome://tracing)
week1/plots/          # Visualizations

EXAMPLES
--------
bash scripts/run_full_suite.sh cpu
bash scripts/benchmark_all.sh mps
python scripts/benchmark_model.py --model resnet50 --device cpu --batch-sizes 1 8 16

See scripts/README.md for detailed documentation.
EOF
