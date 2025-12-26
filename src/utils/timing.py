"""Timing utilities for inference benchmarking.

This module provides high-precision timing utilities for measuring
model inference latency on both CPU and CUDA devices.
"""

import time
import torch
from typing import Callable, List, Optional
from statistics import mean, stdev


class CUDATimer:
    """High-precision timer for CUDA operations.
    
    Uses CUDA events for accurate GPU timing, which is more reliable
    than CPU-side timing for GPU operations.
    
    Example:
        timer = CUDATimer()
        timer.start()
        output = model(input)
        elapsed = timer.stop()
        print(f"Inference took {elapsed:.2f} ms")
    """
    
    def __init__(self):
        """Initialize CUDA timer with start and end events."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Use CPUTimer instead.")
        
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0
    
    def start(self):
        """Record start time."""
        torch.cuda.synchronize()
        self.start_event.record()
    
    def stop(self) -> float:
        """Record end time and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        return self.elapsed_time
    
    def measure(self):
        """Context manager for timing code blocks.
        
        Example:
            with timer.measure():
                output = model(input)
            print(f"Took {timer.elapsed_time:.2f} ms")
        """
        return self
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Exit context manager."""
        self.stop()


class CPUTimer:
    """High-precision timer for CPU operations.
    
    Uses time.perf_counter() for accurate CPU timing.
    
    Example:
        timer = CPUTimer()
        timer.start()
        output = model(input)
        elapsed = timer.stop()
        print(f"Inference took {elapsed:.2f} ms")
    """
    
    def __init__(self):
        """Initialize CPU timer."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_time = 0.0
    
    def start(self):
        """Record start time."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Record end time and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        self.end_time = time.perf_counter()
        self.elapsed_time = (self.end_time - self.start_time) * 1000  # Convert to ms
        return self.elapsed_time
    
    def measure(self):
        """Context manager for timing code blocks.
        
        Example:
            with timer.measure():
                output = model(input)
            print(f"Took {timer.elapsed_time:.2f} ms")
        """
        return self
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Exit context manager."""
        self.stop()


def measure_latency(
    fn: Callable,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    device: str = "cuda"
) -> List[float]:
    """Measure latency of a function over multiple iterations.
    
    This function performs warmup iterations to stabilize GPU/CPU state,
    then measures latency over multiple iterations for statistical analysis.
    
    Args:
        fn: Function to measure (should be a callable with no arguments)
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
        device: Device type ('cuda' or 'cpu')
        
    Returns:
        List of latency measurements in milliseconds
        
    Example:
        def forward_pass():
            return model(inputs)
        
        latencies = measure_latency(
            forward_pass,
            warmup_iterations=10,
            measurement_iterations=100,
            device="cuda"
        )
        print(f"Mean: {mean(latencies):.2f} ms")
        print(f"Std: {stdev(latencies):.2f} ms")
    """
    # Select appropriate timer
    if device == "cuda" and torch.cuda.is_available():
        timer = CUDATimer()
    else:
        timer = CPUTimer()
    
    # Warmup phase
    for _ in range(warmup_iterations):
        fn()
    
    # Synchronize before measurements
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement phase
    latencies = []
    for _ in range(measurement_iterations):
        timer.start()
        fn()
        elapsed = timer.stop()
        latencies.append(elapsed)
    
    return latencies


def print_timing_stats(latencies: List[float], name: str = "Operation"):
    """Print statistical summary of timing measurements.
    
    Args:
        latencies: List of latency measurements in milliseconds
        name: Name of the operation being measured
    """
    if not latencies:
        print(f"No timing data for {name}")
        return
    
    mean_latency = mean(latencies)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0.0
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    p50 = sorted_latencies[int(n * 0.50)]
    p95 = sorted_latencies[int(n * 0.95)]
    p99 = sorted_latencies[int(n * 0.99)]
    
    print(f"\n{'='*60}")
    print(f"Timing Statistics: {name}")
    print(f"{'='*60}")
    print(f"  Iterations:    {len(latencies)}")
    print(f"  Mean:          {mean_latency:.4f} ms")
    print(f"  Std Dev:       {std_latency:.4f} ms")
    print(f"  Min:           {min_latency:.4f} ms")
    print(f"  Max:           {max_latency:.4f} ms")
    print(f"  Median (P50):  {p50:.4f} ms")
    print(f"  P95:           {p95:.4f} ms")
    print(f"  P99:           {p99:.4f} ms")
    print(f"{'='*60}\n")


class Stopwatch:
    """Simple stopwatch for timing code sections.
    
    Example:
        sw = Stopwatch()
        sw.start("data_loading")
        # ... load data ...
        sw.stop("data_loading")
        
        sw.start("inference")
        # ... run inference ...
        sw.stop("inference")
        
        sw.print_summary()
    """
    
    def __init__(self):
        """Initialize stopwatch."""
        self.timings = {}
        self.active_timers = {}
    
    def start(self, name: str):
        """Start timing a named section.
        
        Args:
            name: Name of the section to time
        """
        self.active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing a named section.
        
        Args:
            name: Name of the section to stop timing
            
        Returns:
            Elapsed time in milliseconds
        """
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = (time.perf_counter() - self.active_timers[name]) * 1000
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        
        del self.active_timers[name]
        return elapsed
    
    def get_stats(self, name: str) -> dict:
        """Get statistics for a named section.
        
        Args:
            name: Name of the section
            
        Returns:
            Dictionary with timing statistics
        """
        if name not in self.timings:
            return {}
        
        times = self.timings[name]
        return {
            "count": len(times),
            "mean": mean(times),
            "std": stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "total": sum(times)
        }
    
    def print_summary(self):
        """Print summary of all timed sections."""
        print(f"\n{'='*60}")
        print("Stopwatch Summary")
        print(f"{'='*60}")
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean']:.4f} ms")
            print(f"  Std:    {stats['std']:.4f} ms")
            print(f"  Min:    {stats['min']:.4f} ms")
            print(f"  Max:    {stats['max']:.4f} ms")
            print(f"  Total:  {stats['total']:.4f} ms")
        
        print(f"{'='*60}\n")


# Test code
if __name__ == "__main__":
    print("Testing timing utilities...")
    
    # Test CUDA timer
    if torch.cuda.is_available():
        print("\n=== Testing CUDA Timer ===")
        timer = CUDATimer()
        
        # Simple matrix multiplication
        x = torch.randn(1000, 1000, device="cuda")
        
        timer.start()
        y = torch.matmul(x, x)
        elapsed = timer.stop()
        print(f"Matrix multiplication: {elapsed:.4f} ms")
        
        # Test context manager
        with timer.measure():
            z = torch.matmul(x, x)
        print(f"Context manager: {timer.elapsed_time:.4f} ms")
    
    # Test CPU timer
    print("\n=== Testing CPU Timer ===")
    cpu_timer = CPUTimer()
    
    cpu_timer.start()
    result = sum([i**2 for i in range(100000)])
    elapsed = cpu_timer.stop()
    print(f"CPU computation: {elapsed:.4f} ms")
    
    # Test measure_latency
    print("\n=== Testing measure_latency ===")
    
    def dummy_operation():
        x = torch.randn(500, 500)
        return x @ x
    
    latencies = measure_latency(
        dummy_operation,
        warmup_iterations=5,
        measurement_iterations=50,
        device="cpu"
    )
    
    print_timing_stats(latencies, "Dummy Operation")
    
    # Test Stopwatch
    print("=== Testing Stopwatch ===")
    sw = Stopwatch()
    
    sw.start("task1")
    time.sleep(0.01)
    sw.stop("task1")
    
    sw.start("task2")
    time.sleep(0.02)
    sw.stop("task2")
    
    sw.start("task1")
    time.sleep(0.015)
    sw.stop("task1")
    
    sw.print_summary()
    
    print("✓ All timing tests passed!")
