#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:."

MODELS=("resnet50" "gpt2" "distilbert")
DEVICE=${1:-"cpu"}

echo "Running profiling on device: $DEVICE"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "Profiling $MODEL..."
    python3 scripts/profile_model.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size 8 \
        --all \
        --output-dir week1/experiments
    echo ""
done

echo "Profiling complete. Traces in week1/experiments/"