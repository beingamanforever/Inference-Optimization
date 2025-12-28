#!/usr/bin/env python3
"""
Quick validation test for all models.
Ensures models load and run inference correctly.
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ResNet50ModelWrapper, GPT2ModelWrapper, DistilBERTModelWrapper


def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_model(name: str, model_class, device: str):
    """Test a single model."""
    print(f"\nTesting {name}...")
    
    # Initialize
    model = model_class(device=device)
    print(f"  - Model loaded: {model.get_model_info()['num_parameters']:,} params")
    
    # Test forward pass with different batch sizes
    for bs in [1, 4]:
        inputs = model.prepare_input(batch_size=bs)
        outputs = model.forward(inputs)
        print(f"  - Batch {bs}: {list(inputs.shape) if hasattr(inputs, 'shape') else 'dict'} -> {list(outputs.shape)}")
    
    print(f"  ✓ {name} working correctly")


def main():
    """Run validation tests."""
    print("="*60)
    print("  Model Validation Test")
    print("="*60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"Device: {device}")
    
    try:
        # Test all models
        print_section("Testing Models")
        test_model("ResNet50", ResNet50ModelWrapper, device)
        test_model("GPT-2", GPT2ModelWrapper, device)
        test_model("DistilBERT", DistilBERTModelWrapper, device)
        
        # Summary
        print("\n" + "="*60)
        print("  ✓ ALL MODELS VALIDATED")
        print("="*60)
        
        print("\nRun experiments with:")
        print(f"  bash scripts/run_full_suite.sh {device}")
        print()
        
    except Exception as e:
        print("\n" + "="*60)
        print("  ✗ VALIDATION FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
