"""
Benchmarking framework for inference optimization.

Exposes:
- Benchmark: executes benchmarks and collects metrics
- InferenceMetrics: structured performance metrics
- BenchmarkReporter: presentation and persistence of results
"""

from .benchmark import Benchmark
from .metrics import InferenceMetrics
from .reporters import BenchmarkReporter

__all__ = [
    "Benchmark",
    "InferenceMetrics",
    "BenchmarkReporter",
]
