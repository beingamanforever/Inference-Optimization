#!/bin/bash
set -e

echo "Setting up Inference Optimization environment..."
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment (cross-platform)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/4] Creating project directories..."
mkdir -p week1/benchmarks
mkdir -p week1/experiments/{resnet50,gpt2,distilbert}
mkdir -p week1/plots

echo "[4/4] Downloading model weights..."
python3 -c "
import torch
import torchvision.models as models
from transformers import GPT2LMHeadModel, DistilBertModel, GPT2Tokenizer, DistilBertTokenizer

print('  - ResNet50...')
models.resnet50(weights='ResNet50_Weights.DEFAULT')

print('  - GPT-2...')
GPT2Tokenizer.from_pretrained('gpt2')
GPT2LMHeadModel.from_pretrained('gpt2')

print('  - DistilBERT...')
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
DistilBertModel.from_pretrained('distilbert-base-uncased')

print('  Done.')
"

echo ""
echo "Setup complete. Activate environment with: source venv/bin/activate"
echo "Run quick test: python scripts/demo_inference.py"