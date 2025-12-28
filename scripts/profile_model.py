"""Profile model execution for bottleneck analysis.

Usage:
    python scripts/profile_model.py --model resnet50 --device cpu --batch-size 8
    python scripts/profile_model.py --model gpt2 --device mps --all
"""
import argparse
import os
import torch
import numpy as np
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ResNet50ModelWrapper, GPT2ModelWrapper, DistilBERTModelWrapper
from src.profiling import (
    InferenceProfiler, 
    PythonOverheadAnalyzer, 
    LayerTimer,
    MemoryTracker
)


def set_deterministic(seed: int = 42):
    """Set seeds for reproducible profiling."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Profile model inference bottlenecks")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["resnet50", "gpt2", "distilbert"],
        help="Model to profile"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for profiling"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Batch size for profiling"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="week1/experiments",
        help="Output directory for traces"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all profiling tools (PyTorch profiler, layer timer, memory, overhead)"
    )
    args = parser.parse_args()

    # Set deterministic behavior
    set_deterministic(seed=42)

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Model mapping
    model_map = {
        "resnet50": ResNet50ModelWrapper,
        "gpt2": GPT2ModelWrapper,
        "distilbert": DistilBERTModelWrapper
    }

    print(f"\nInitializing {args.model} on {args.device}...")
    print("=" * 60)
    model_wrapper = model_map[args.model](device=args.device)
    
    output_path = os.path.join(args.output_dir, f"{args.model}_{args.device}")
    os.makedirs(output_path, exist_ok=True)

    # 1. PyTorch Profiler (always run)
    print(f"\n[1/4] Running PyTorch Profiler (batch_size={args.batch_size})...")
    profiler = InferenceProfiler(model_wrapper, device=args.device)
    metrics = profiler.run_profile(
        batch_size=args.batch_size, 
        output_dir=output_path,
        warmup_iters=10,
        profile_iters=20
    )
    
    print(f"\nProfiling Results:")
    print(f"  Total CPU Time:    {metrics['total_cpu_time_ms']:.2f} ms")
    print(f"  Operator Time:     {metrics['operator_time_ms']:.2f} ms")
    print(f"  Python Overhead:   {metrics['python_overhead_pct']:.2f}%")
    print(f"  Trace saved to:    {output_path}/trace.json")

    # Optional: Run additional profiling tools
    if args.all:
        # 2. Layer Timer
        print(f"\n[2/4] Running Layer Timer...")
        timer = LayerTimer(model_wrapper)
        timer.profile_layers(batch_size=args.batch_size, top_n=10)

        # 3. Memory Tracker
        print(f"\n[3/4] Memory Usage Analysis...")
        tracker = MemoryTracker(device=args.device)
        tracker.reset_peak_stats()
        _ = model_wrapper.forward(model_wrapper.prepare_input(args.batch_size))
        tracker.print_stats()

        # 4. Python Overhead Analyzer
        print(f"\n[4/4] Python Overhead Analysis...")
        analyzer = PythonOverheadAnalyzer(model_wrapper)
        analyzer.profile_inference(iterations=30, batch_size=args.batch_size, top_n=15)

    print("\n" + "=" * 60)
    print(f"Profiling complete for {args.model} on {args.device}")
    print(f"Results saved to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()