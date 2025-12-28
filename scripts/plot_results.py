"""Generate visualization plots from benchmark results.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --output-dir week2/benchmarks
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse


def generate_plots(benchmark_dir: str = "week1/benchmarks", output_file: str = None):
    """Generate comparison plots from benchmark CSV files.
    
    Args:
        benchmark_dir: Directory containing CSV benchmark results
        output_file: Custom output filename (default: benchmark_dir/performance_comparison.png)
    """
    # Find all CSV files
    csv_pattern = os.path.join(benchmark_dir, "*_results.csv")
    all_files = glob.glob(csv_pattern)
    
    if not all_files:
        print(f"Error: No CSV files found in {benchmark_dir}")
        print(f"Run benchmarks first: python scripts/benchmark_model.py --model resnet50 --device cpu")
        return

    print(f"Found {len(all_files)} benchmark files:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")

    # Load and combine all results
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # Verify required columns exist
    required_cols = ["batch_size", "throughput_items_per_sec", "latency_p50_ms", "model", "device"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns in CSV: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Set visual style
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Throughput comparison
    sns.lineplot(
        data=df,
        x="batch_size",
        y="throughput_items_per_sec",
        hue="model",
        style="device",
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax1
    )
    ax1.set_title("Throughput Scaling by Batch Size", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Throughput (items/sec)", fontsize=12)
    ax1.legend(title="Model & Device", loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency comparison (P50)
    sns.lineplot(
        data=df,
        x="batch_size",
        y="latency_p50_ms",
        hue="model",
        style="device",
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax2
    )
    ax2.set_title("P50 Latency by Batch Size", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Latency P50 (ms)", fontsize=12)
    ax2.legend(title="Model & Device", loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = os.path.join(benchmark_dir, "performance_comparison.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plots saved to: {output_file}")
    print(f"View the comparison charts to analyze performance.\n")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualization plots")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="week1/benchmarks",
        help="Directory containing benchmark CSV files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Custom output filename for plot"
    )
    args = parser.parse_args()

    generate_plots(benchmark_dir=args.output_dir, output_file=args.output_file)


if __name__ == "__main__":
    main()