#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:."

BATCH_SIZES=(1 4 8 16)
MODELS=("resnet50" "gpt2" "distilbert")
DEVICE=${1:-"cpu"}

echo "Running benchmarks on device: $DEVICE"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "Benchmarking $MODEL..."
    python3 scripts/benchmark_model.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-sizes ${BATCH_SIZES[@]} \
        --warmup 20 \
        --iterations 100 \
        --output-dir week1/benchmarks
    echo ""
done

echo "Benchmarks complete. Results in week1/benchmarks/"