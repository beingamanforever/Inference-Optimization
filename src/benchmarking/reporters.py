import json
import pandas as pd
from typing import List
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .metrics import InferenceMetrics


class BenchmarkReporter:
    """
    Presentation layer for benchmark results.

    Responsibilities:
    - Human-readable CLI summary (quick sanity checks)
    - CSV export (analysis, plots)
    - JSON export (ground-truth artifact)

    This class MUST NOT:
    - recompute metrics
    - average across runs
    - interpret performance
    """

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[InferenceMetrics] = []
        self.console = Console()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_result(self, metrics: InferenceMetrics):
        """
        Add a completed benchmark result.
        Metrics are assumed to be final and correct.
        """
        self.results.append(metrics)

    # ------------------------------------------------------------------
    # CLI summary (human-facing)
    # ------------------------------------------------------------------

    def print_terminal_summary(self):
        """
        Print a concise CLI table.
        Focuses on *decision-driving* metrics only.
        """
        table = Table(title="Inference Benchmark Summary")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Backend", style="blue")
        table.add_column("Device", style="magenta")
        table.add_column("Batch", justify="right")

        table.add_column("P50 (ms)", justify="right")
        table.add_column("P99 (ms)", justify="right", style="red")

        table.add_column("Throughput", justify="right", style="green")
        table.add_column("Util (%)", justify="right")
        table.add_column("Mem (MB)", justify="right")

        for r in self.results:
            # Unified utilization display
            util_str = (
                f"G:{r.gpu_utilization_percent:.1f}"
                if r.device == "cuda"
                else f"C:{r.cpu_utilization_percent:.1f}"
            )

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

    # ------------------------------------------------------------------
    # Persistent reports (machine-facing)
    # ------------------------------------------------------------------

    def save_reports(self, filename_prefix: str = "benchmark"):
        """
        Save benchmark results to CSV and JSON.

        CSV:
          - plotting
          - Excel / Sheets
          - batch-size sweeps

        JSON:
          - ground truth
          - regression comparison
          - programmatic access
        """
        if not self.results:
            print("⚠️ No benchmark results to save.")
            return

        data = [r.to_dict() for r in self.results]

        # CSV
        csv_path = self.output_dir / f"{filename_prefix}_results.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # JSON
        json_path = self.output_dir / f"{filename_prefix}_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Reports saved to {self.output_dir}")
