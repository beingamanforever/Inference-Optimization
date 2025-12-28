import json
import pandas as pd
from typing import List, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .metrics import InferenceMetrics


class BenchmarkReporter:
    """
    Presentation layer for benchmark results.

    Responsibilities:
    - Human-readable CLI summary for quick analysis
    - CSV export for data analysis and plotting
    - JSON export for programmatic access and storage

    Design principle: This class is a pure presentation layer.
    It MUST NOT recompute metrics, interpret performance, or modify data.
    """

    def __init__(self, output_dir: str = "experiments"):
        """
        Initialize the benchmark reporter.
        
        Args:
            output_dir: Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[InferenceMetrics] = []
        self.console = Console()

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------

    def add_result(self, metrics: InferenceMetrics) -> None:
        """
        Add a single benchmark result.
        
        Args:
            metrics: Completed InferenceMetrics object with performance data
            
        Raises:
            ValueError: If metrics validation fails
        """
        is_valid, error_msg = metrics.validate()
        if not is_valid:
            raise ValueError(f"Invalid metrics: {error_msg}")
        
        self.results.append(metrics)

    def add_results(self, metrics_list: List[InferenceMetrics]) -> None:
        """
        Add multiple benchmark results at once.
        
        Args:
            metrics_list: List of InferenceMetrics objects
            
        Raises:
            ValueError: If any metrics validation fails
        """
        for metrics in metrics_list:
            self.add_result(metrics)

    def clear_results(self) -> None:
        """Clear all stored benchmark results."""
        self.results.clear()

    def get_results(self) -> List[InferenceMetrics]:
        """
        Get a copy of all stored results.
        
        Returns:
            List of InferenceMetrics objects
        """
        return self.results.copy()

    # ------------------------------------------------------------------
    # CLI Summary (Human-Facing)
    # ------------------------------------------------------------------

    def print_terminal_summary(self) -> None:
        """
        Print a formatted table of benchmark results to the terminal.
        
        The table includes key decision-making metrics:
        - Latency percentiles (P50, P99)
        - Throughput
        - Resource utilization
        - Memory usage
        """
        if not self.results:
            self.console.print("[yellow]Warning:[/yellow] No benchmark results to display.")
            return
            
        table = Table(title="Inference Benchmark Summary", show_header=True, header_style="bold")

        # Table structure
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Backend", style="blue")
        table.add_column("Device", style="magenta")
        table.add_column("Batch", justify="right")
        table.add_column("P50 (ms)", justify="right")
        table.add_column("P99 (ms)", justify="right", style="red")
        table.add_column("Throughput", justify="right", style="green")
        table.add_column("Util (%)", justify="right")
        table.add_column("Mem (MB)", justify="right")

        # Populate table rows
        for r in self.results:
            # Device-specific utilization display
            if r.device == "cuda":
                util_str = f"GPU: {r.gpu_utilization_percent:.1f}"
            elif r.device == "mps":
                util_str = f"CPU: {r.cpu_utilization_percent:.1f}"
            else:
                util_str = f"CPU: {r.cpu_utilization_percent:.1f}"

            table.add_row(
                r.model_name,
                r.backend,
                r.device,
                str(r.batch_size),
                f"{r.p50_latency_ms:.2f}",
                f"{r.p99_latency_ms:.2f}",
                f"{r.throughput_items_per_sec:.2f}",
                util_str,
                f"{r.memory_allocated_mb:.0f}",
            )

        self.console.print(table)

    def print_detailed_summary(self, metric_index: Optional[int] = None) -> None:
        """
        Print detailed summary for one or all benchmark results.
        
        Args:
            metric_index: Index of specific result to print. If None, prints all.
        """
        if not self.results:
            self.console.print("[yellow]Warning:[/yellow] No benchmark results to display.")
            return

        if metric_index is not None:
            if 0 <= metric_index < len(self.results):
                print(self.results[metric_index].summary())
            else:
                self.console.print(f"[red]Error:[/red] Index {metric_index} out of range (0-{len(self.results)-1})")
        else:
            for metrics in self.results:
                print(metrics.summary())

    # ------------------------------------------------------------------
    # File Export (Machine-Facing)
    # ------------------------------------------------------------------

    def save_reports(self, filename_prefix: str = "benchmark") -> None:
        """
        Export benchmark results to CSV and JSON files.
        
        CSV format:
          - Optimized for data analysis tools (pandas, R, Excel)
          - Easy to plot and visualize
          - Batch size comparisons
          
        JSON format:
          - Complete data preservation
          - Programmatic access
          - Version control friendly
        
        Args:
            filename_prefix: Prefix for output filenames
            
        Raises:
            IOError: If file writing fails
            ValueError: If no results to save
        """
        if not self.results:
            self.console.print("[yellow]Warning:[/yellow] No benchmark results to save.")
            return

        # Convert all metrics to dictionaries
        data = [r.to_dict() for r in self.results]

        try:
            # CSV export
            csv_path = self.output_dir / f"{filename_prefix}_results.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            self.console.print(f"[green]Success:[/green] CSV saved to {csv_path}")

            # JSON export
            json_path = self.output_dir / f"{filename_prefix}_results.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            self.console.print(f"[green]Success:[/green] JSON saved to {json_path}")
            
            self.console.print(f"[bold green]All reports saved to {self.output_dir}[/bold green]")
            
        except IOError as e:
            self.console.print(f"[red]Error:[/red] Failed to save reports: {e}")
            raise
        except Exception as e:
            self.console.print(f"[red]Error:[/red] Unexpected error during save: {e}")
            raise

    def save_summary_report(self, filename: str = "summary.txt") -> None:
        """
        Save a text summary of all benchmark results.
        
        Args:
            filename: Output filename for the summary
            
        Raises:
            IOError: If file writing fails
        """
        if not self.results:
            self.console.print("[yellow]Warning:[/yellow] No benchmark results to save.")
            return

        summary_path = self.output_dir / filename
        
        try:
            with open(summary_path, "w") as f:
                f.write("BENCHMARK RESULTS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                for i, metrics in enumerate(self.results, 1):
                    f.write(f"Result {i}:\n")
                    f.write(metrics.summary())
                    f.write("\n")
                
            self.console.print(f"[green]Success:[/green] Summary saved to {summary_path}")
            
        except IOError as e:
            self.console.print(f"[red]Error:[/red] Failed to save summary: {e}")
            raise

    # ------------------------------------------------------------------
    # Analysis Helpers
    # ------------------------------------------------------------------

    def get_best_throughput(self) -> Optional[InferenceMetrics]:
        """
        Find the configuration with highest throughput.
        
        Returns:
            InferenceMetrics object with best throughput, or None if no results
        """
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.throughput_items_per_sec)

    def get_best_latency(self) -> Optional[InferenceMetrics]:
        """
        Find the configuration with lowest P50 latency.
        
        Returns:
            InferenceMetrics object with best latency, or None if no results
        """
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.p50_latency_ms)

    def filter_by_device(self, device: str) -> List[InferenceMetrics]:
        """
        Filter results by device type.
        
        Args:
            device: Device type ('cuda', 'mps', or 'cpu')
            
        Returns:
            List of InferenceMetrics for the specified device
        """
        return [r for r in self.results if r.device == device]

    def filter_by_batch_size(self, batch_size: int) -> List[InferenceMetrics]:
        """
        Filter results by batch size.
        
        Args:
            batch_size: Batch size to filter by
            
        Returns:
            List of InferenceMetrics with the specified batch size
        """
        return [r for r in self.results if r.batch_size == batch_size]
