"""Benchmark models across different devices and batch sizes.

Usage:
    python scripts/benchmark_model.py --model resnet50 --device cpu --batch-sizes 1 4 8 16
    python scripts/benchmark_model.py --model gpt2 --device mps --batch-sizes 1 2 4
    python scripts/benchmark_model.py --model distilbert --device cuda --batch-sizes 1 8 16 32
"""
import argparse
import os
import torch
import numpy as np
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ResNet50ModelWrapper, GPT2ModelWrapper, DistilBERTModelWrapper
from src.benchmarking import Benchmark, BenchmarkReporter


def set_deterministic(seed: int = 42):
    """Set seeds for reproducible benchmarking."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["resnet50", "gpt2", "distilbert"],
        help="Model to benchmark"
    )
    parser.add_argument(
        "--batch-sizes", 
        type=int, 
        nargs="+", 
        default=[1, 4, 8],
        help="List of batch sizes to test"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "mps", "cuda"],
        help="Device to run benchmarks on"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="week1/benchmarks",
        help="Directory to save results"
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=20,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=100,
        help="Number of measurement iterations"
    )
    args = parser.parse_args()

    # Set deterministic behavior
    set_deterministic(seed=42)

    # Validate device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print(f"Warning: MPS requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Model mapping
    model_map = {
        "resnet50": ResNet50ModelWrapper,
        "gpt2": GPT2ModelWrapper,
        "distilbert": DistilBERTModelWrapper
    }

    print(f"\nBenchmarking {args.model} on {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 60)
    
    # Load model
    model_wrapper = model_map[args.model](device=args.device)
    
    # Initialize benchmark engine
    benchmark = Benchmark(
        model_wrapper, 
        warmup_iter=args.warmup, 
        measure_iter=args.iterations
    )

    # Run batch size sweep
    results = benchmark.run_batch_sweep(args.batch_sizes)

    # Report results
    reporter = BenchmarkReporter(output_dir=args.output_dir)
    reporter.add_results(results)
    
    print("\n" + "=" * 60)
    reporter.print_terminal_summary()
    print("=" * 60 + "\n")
    
    # Save artifacts
    filename = f"{args.model}_{args.device}"
    reporter.save_reports(filename_prefix=filename)
    
    # Print best configuration
    best_throughput = reporter.get_best_throughput()
    best_latency = reporter.get_best_latency()
    
    print("\nOptimization Summary:")
    print(f"  Best Throughput: {best_throughput.throughput_items_per_sec:.2f} items/s "
          f"(batch_size={best_throughput.batch_size})")
    print(f"  Best Latency:    {best_latency.p50_latency_ms:.2f} ms "
          f"(batch_size={best_latency.batch_size})")
    print()


if __name__ == "__main__":
    main()