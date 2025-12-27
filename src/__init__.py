"""
Inference Optimizer Package
Week 1: Profiling & Bottleneck Analysis

This package provides tools for benchmarking and profiling 
PyTorch models for inference optimization.
"""

__version__ = "0.1.0"
__author__ = "Inference Optimization Team"

# Package-level imports for convenience
from src.models import ResNet50Model, GPT2Model, DistilBERTModel
from src.benchmarking import Benchmark, InferenceMetrics
# from src.profiling import InferenceProfiler

__all__ = [
    "ResNet50Model",
    "GPT2Model", 
    "DistilBERTModel",
    "Benchmark",
    "InferenceMetrics",
    # "InferenceProfiler",
]