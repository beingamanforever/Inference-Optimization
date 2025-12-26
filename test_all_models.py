#!/usr/bin/env python3
"""
Complete Test Suite for All Models
Week 1: Profiling & Bottleneck Analysis

This script tests all model implementations and utilities
to ensure everything is working correctly before benchmarking.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ResNet50Model, GPT2Model, DistilBERTModel
from src.utils import CUDATimer, CPUTimer, measure_latency, print_timing_stats
from src.utils import DummyDataLoader, ImageDataGenerator, TextDataGenerator
from src.utils import setup_logger


def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_base_functionality():
    """Test BaseModel functionality."""
    print_section("Testing BaseModel Functionality")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test with ResNet as example
    model = ResNet50Model(device=device)
    
    print("\n✓ Model initialization works")
    print(f"✓ Model info: {model.get_model_info()['num_parameters']:,} params")
    print(f"✓ Model device: {model.device}")
    
    # Test repr
    print(f"\n✓ Model repr:\n{model}")


def test_resnet50():
    """Test ResNet-50 implementation."""
    print_section("Testing ResNet-50 Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize
    print("1. Initializing ResNet-50...")
    model = ResNet50Model(device=device, pretrained=True)
    print("   ✓ Model loaded")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_sizes = [1, 8, 16]
    
    for bs in batch_sizes:
        inputs = model.prepare_input(batch_size=bs)
        outputs = model.forward(inputs)
        
        assert inputs.shape == (bs, 3, 224, 224), f"Wrong input shape: {inputs.shape}"
        assert outputs.shape == (bs, 1000), f"Wrong output shape: {outputs.shape}"
        print(f"   ✓ Batch size {bs}: {inputs.shape} -> {outputs.shape}")
    
    # Test predictions
    print("\n3. Testing predictions...")
    inputs = model.prepare_input(batch_size=2)
    predictions = model.predict(inputs, top_k=3)
    print(f"   ✓ Got predictions for {len(predictions)} images")
    print(f"   ✓ Top prediction: {predictions[0][0]['class_name']}")
    
    # Test warmup
    print("\n4. Testing warmup...")
    model.warmup(num_iterations=3, batch_size=4)
    print("   ✓ Warmup complete")
    
    print("\n✓ ResNet-50 all tests passed!")


def test_gpt2():
    """Test GPT-2 implementation."""
    print_section("Testing GPT-2 Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize
    print("1. Initializing GPT-2...")
    model = GPT2Model(device=device, seq_length=128)
    print("   ✓ Model loaded")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_sizes = [1, 4, 8]
    
    for bs in batch_sizes:
        inputs = model.prepare_input(batch_size=bs)
        outputs = model.forward(inputs)
        
        assert inputs['input_ids'].shape == (bs, 128), f"Wrong input shape"
        assert outputs.shape[0] == bs, f"Wrong output batch size"
        assert outputs.shape[1] == 128, f"Wrong output sequence length"
        print(f"   ✓ Batch size {bs}: {inputs['input_ids'].shape} -> {outputs.shape}")
    
    # Test text generation
    print("\n3. Testing text generation...")
    try:
        prompt = "The future of AI is"
        generated = model.generate(prompt, max_length=30)
        print(f"   ✓ Generated: {generated[:60]}...")
    except Exception as e:
        print(f"   ⚠ Generation failed (may need internet): {e}")
    
    # Test warmup
    print("\n4. Testing warmup...")
    model.warmup(num_iterations=3, batch_size=4)
    print("   ✓ Warmup complete")
    
    print("\n✓ GPT-2 all tests passed!")


def test_distilbert():
    """Test DistilBERT implementation."""
    print_section("Testing DistilBERT Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize
    print("1. Initializing DistilBERT...")
    model = DistilBERTModel(device=device, seq_length=128)
    print("   ✓ Model loaded")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_sizes = [1, 4, 8]
    
    for bs in batch_sizes:
        inputs = model.prepare_input(batch_size=bs)
        outputs = model.forward(inputs)
        
        assert inputs['input_ids'].shape == (bs, 128), f"Wrong input shape"
        assert outputs.shape[0] == bs, f"Wrong output batch size"
        print(f"   ✓ Batch size {bs}: {inputs['input_ids'].shape} -> {outputs.shape}")
    
    # Test embeddings
    print("\n3. Testing sentence embeddings...")
    texts = [
        "This is a test sentence.",
        "Another test sentence here."
    ]
    embeddings = model.get_embeddings(texts)
    print(f"   ✓ Embeddings shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(texts), "Wrong number of embeddings"
    
    # Test warmup
    print("\n4. Testing warmup...")
    model.warmup(num_iterations=3, batch_size=4)
    print("   ✓ Warmup complete")
    
    print("\n✓ DistilBERT all tests passed!")


def test_timing_utils():
    """Test timing utilities."""
    print_section("Testing Timing Utilities")
    
    # Test CUDA timer
    if torch.cuda.is_available():
        print("1. Testing CUDA Timer...")
        timer = CUDATimer()
        
        x = torch.randn(1000, 1000, device="cuda")
        timer.start()
        y = torch.matmul(x, x)
        elapsed = timer.stop()
        
        print(f"   ✓ CUDA Timer: {elapsed:.2f} ms")
        
        # Test context manager
        with timer.measure():
            z = torch.matmul(x, x)
        print(f"   ✓ Context manager: {timer.elapsed_time:.2f} ms")
    
    # Test CPU timer
    print("\n2. Testing CPU Timer...")
    timer = CPUTimer()
    
    timer.start()
    result = sum([i**2 for i in range(10000)])
    elapsed = timer.stop()
    
    print(f"   ✓ CPU Timer: {elapsed:.2f} ms")
    
    # Test measure_latency
    print("\n3. Testing measure_latency...")
    
    def dummy_op():
        x = torch.randn(100, 100)
        return x @ x
    
    latencies = measure_latency(
        dummy_op,
        warmup_iterations=5,
        measurement_iterations=20,
        device="cpu"
    )
    
    print(f"   ✓ Measured {len(latencies)} iterations")
    print(f"   ✓ Mean latency: {sum(latencies)/len(latencies):.2f} ms")
    
    print("\n✓ Timing utilities all tests passed!")


def test_data_utils():
    """Test data loading utilities."""
    print_section("Testing Data Loading Utilities")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test DummyDataLoader
    print("1. Testing DummyDataLoader...")
    loader = DummyDataLoader(
        data_shape=(3, 224, 224),
        num_samples=32,
        batch_size=8,
        device=device
    )
    
    print(f"   ✓ Created loader with {len(loader)} batches")
    
    for i, batch in enumerate(loader):
        print(f"   ✓ Batch {i}: shape={batch.shape}")
        if i >= 1:  # Only check first 2 batches
            break
    
    # Test ImageDataGenerator
    print("\n2. Testing ImageDataGenerator...")
    images = ImageDataGenerator.generate_batch(
        batch_size=4,
        device=device
    )
    print(f"   ✓ Generated images: {images.shape}")
    
    # Test TextDataGenerator
    print("\n3. Testing TextDataGenerator...")
    input_ids, attention_mask = TextDataGenerator.generate_batch(
        batch_size=4,
        seq_length=128,
        vocab_size=30522,
        device=device
    )
    print(f"   ✓ Generated tokens: {input_ids.shape}")
    print(f"   ✓ Generated mask: {attention_mask.shape}")
    
    print("\n✓ Data utilities all tests passed!")


def test_logging_utils():
    """Test logging utilities."""
    print_section("Testing Logging Utilities")
    
    print("1. Setting up logger...")
    logger = setup_logger(
        name="test_logger",
        log_dir="test_logs"
    )
    
    logger.info("Test log message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    
    print("   ✓ Logger created (check test_logs/)")
    
    print("\n✓ Logging utilities all tests passed!")


def quick_benchmark():
    """Run a quick benchmark of all models."""
    print_section("Quick Performance Benchmark")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    
    models = [
        ("ResNet-50", ResNet50Model(device=device)),
        ("GPT-2", GPT2Model(device=device, seq_length=128)),
        ("DistilBERT", DistilBERTModel(device=device, seq_length=128))
    ]
    
    print(f"Device: {device}, Batch size: {batch_size}\n")
    
    for name, model in models:
        # Prepare input
        inputs = model.prepare_input(batch_size=batch_size)
        
        # Measure latency
        def forward_fn():
            return model.forward(inputs)
        
        latencies = measure_latency(
            forward_fn,
            warmup_iterations=5,
            measurement_iterations=20,
            device=device
        )
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = (batch_size * 1000) / avg_latency
        
        print(f"{name:15s}: {avg_latency:6.2f} ms/batch, {throughput:7.1f} samples/sec")
    
    print("\n✓ Quick benchmark complete!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  COMPLETE TEST SUITE - Week 1 Implementation")
    print("  Testing all models and utilities")
    print("="*70)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Run all tests
        test_base_functionality()
        test_resnet50()
        test_gpt2()
        test_distilbert()
        test_timing_utils()
        test_data_utils()
        test_logging_utils()
        quick_benchmark()
        
        # Final summary
        print("\n" + "="*70)
        print("  ✓ ALL TESTS PASSED!")
        print("  Your implementation is ready for Week 1 experiments!")
        print("="*70)
        
        print("\nNext steps:")
        print("  1. Run: python scripts/run_benchmarks.sh")
        print("  2. Run: python scripts/run_profiling.sh")
        print("  3. Analyze results in week1/experiments/")
        print("\nGood luck! 🚀\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("  ✗ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
