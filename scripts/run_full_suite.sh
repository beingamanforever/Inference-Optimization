#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:."

DEVICE=${1:-"cpu"}

echo "Running full benchmarking suite on device: $DEVICE"
echo ""
echo "This will:"
echo "  1. Benchmark all models"
echo "  2. Profile all models"
echo "  3. Generate visualizations and report"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Starting benchmarking suite..."
echo "=" * 60

# Step 1: Benchmark all models
echo ""
echo "[1/3] Running benchmarks..."
bash scripts/benchmark_all.sh "$DEVICE"

# Step 2: Profile all models
echo ""
echo "[2/3] Running profiling..."
bash scripts/profile_all.sh "$DEVICE"

# Step 3: Generate reports
echo ""
echo "[3/3] Generating reports..."
bash scripts/report.sh

echo ""
echo "=" * 60
echo "Full suite complete!"
echo ""
echo "Results:"
echo "  Benchmarks:  week1/benchmarks/"
echo "  Profiles:    week1/experiments/"
echo "  Plots:       week1/plots/"
