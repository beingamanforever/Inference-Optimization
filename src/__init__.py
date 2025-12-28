# src/__init__.py
"""Inference Optimization Package

A comprehensive toolkit for benchmarking and profiling deep learning models.

Components:
    - models: Model wrappers (ResNet50, GPT-2, DistilBERT)
    - benchmarking: Performance measurement and reporting
    - profiling: Detailed bottleneck analysis tools
"""

__version__ = "0.1.0"

# Model Wrappers
from .models import (
    ResNet50ModelWrapper,
    GPT2ModelWrapper,
    DistilBERTModelWrapper
)

# Benchmarking Suite
from .benchmarking import (
    Benchmark,
    InferenceMetrics,
    BenchmarkReporter
)

# Profiling Tools
from .profiling import (
    InferenceProfiler,
    MemoryTracker,
    RooflineCalculator,
    PythonOverheadAnalyzer,
    LayerTimer
)

__all__ = [
    # Models
    "ResNet50ModelWrapper",
    "GPT2ModelWrapper",
    "DistilBERTModelWrapper",
    # Benchmarking
    "Benchmark",
    "InferenceMetrics",
    "BenchmarkReporter",
    # Profiling
    "InferenceProfiler",
    "MemoryTracker",
    "RooflineCalculator",
    "PythonOverheadAnalyzer",
    "LayerTimer",
]