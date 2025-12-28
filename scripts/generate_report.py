"""Generate summary report from benchmark results.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --benchmark-dir week2/benchmarks
"""
import os
import json
import glob
import argparse
from typing import Dict, List, Any


def load_benchmark_results(benchmark_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON benchmark files from directory.
    
    Args:
        benchmark_dir: Directory containing benchmark JSON files
        
    Returns:
        List of benchmark result dictionaries
    """
    json_pattern = os.path.join(benchmark_dir, "*_results.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        return []
    
    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # data is a list of benchmark results
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    
    return all_results


def generate_summary_report(benchmark_dir: str = "week1/benchmarks"):
    """Generate and print summary report from benchmarks.
    
    Args:
        benchmark_dir: Directory containing benchmark results
    """
    results = load_benchmark_results(benchmark_dir)
    
    if not results:
        print(f"Error: No benchmark results found in {benchmark_dir}")
        print(f"Run benchmarks first: python scripts/benchmark_model.py --model resnet50 --device cpu")
        return

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 90)

    # Group by model and device
    grouped = {}
    for result in results:
        key = (result['model'], result['device'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Print formatted table
    print(f"\n{'Model':<15} | {'Device':<8} | {'Batch':<6} | {'P50 (ms)':<10} | "
          f"{'P99 (ms)':<10} | {'Throughput':<12} | {'Mem (MB)':<10}")
    print("-" * 90)

    for (model, device), group in sorted(grouped.items()):
        for result in sorted(group, key=lambda x: x['batch_size']):
            print(
                f"{model:<15} | {device:<8} | {result['batch_size']:<6} | "
                f"{result['latency_p50_ms']:<10.2f} | {result['latency_p99_ms']:<10.2f} | "
                f"{result['throughput_items_per_sec']:<12.2f} | "
                f"{result['memory_allocated_mb']:<10.1f}"
            )

    print("=" * 90)

    # Find best configurations
    print("\nOPTIMIZATION INSIGHTS:")
    print("-" * 90)
    
    # Best throughput
    best_throughput = max(results, key=lambda x: x['throughput_items_per_sec'])
    print(f"\nHighest Throughput:")
    print(f"  Model: {best_throughput['model']} on {best_throughput['device']}")
    print(f"  Batch Size: {best_throughput['batch_size']}")
    print(f"  Throughput: {best_throughput['throughput_items_per_sec']:.2f} items/sec")

    # Best latency
    best_latency = min(results, key=lambda x: x['latency_p50_ms'])
    print(f"\nLowest Latency (P50):")
    print(f"  Model: {best_latency['model']} on {best_latency['device']}")
    print(f"  Batch Size: {best_latency['batch_size']}")
    print(f"  Latency: {best_latency['latency_p50_ms']:.2f} ms")

    # Device comparison
    print(f"\nDevice Comparison (Average across all configs):")
    devices = {}
    for result in results:
        dev = result['device']
        if dev not in devices:
            devices[dev] = {'throughput': [], 'latency': []}
        devices[dev]['throughput'].append(result['throughput_items_per_sec'])
        devices[dev]['latency'].append(result['latency_p50_ms'])
    
    for device, metrics in sorted(devices.items()):
        avg_throughput = sum(metrics['throughput']) / len(metrics['throughput'])
        avg_latency = sum(metrics['latency']) / len(metrics['latency'])
        print(f"  {device.upper():>4}: Avg Throughput = {avg_throughput:>8.2f} items/sec, "
              f"Avg Latency = {avg_latency:>6.2f} ms")

    print("\n" + "=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark summary report")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="week1/benchmarks",
        help="Directory containing benchmark results"
    )
    args = parser.parse_args()

    generate_summary_report(benchmark_dir=args.benchmark_dir)


if __name__ == "__main__":
    main()