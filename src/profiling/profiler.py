"""PyTorch profiler wrapper for inference analysis.

Provides detailed profiling including:
- CPU/CUDA execution traces
- Memory profiling
- Operator-level timing
- Chrome trace export for visualization
"""
import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, Any, Optional, List


class InferenceProfiler:
    """Wrapper for PyTorch profiler with model inference analysis.
    
    Attributes:
        model_wrapper: Model instance implementing BaseModel interface
        device: Device model runs on ('cpu', 'cuda', or 'mps')
        activities: List of profiling activities (CPU, CUDA)
    """
    
    def __init__(self, model_wrapper, device: str = "cpu"):
        """Initialize profiler with model and device configuration.
        
        Args:
            model_wrapper: Model wrapper instance
            device: Target device for profiling
        """
        self.model_wrapper = model_wrapper
        self.device = device
        
        # Configure profiling activities based on device
        self.activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
        if "cuda" in self.device and torch.cuda.is_available():
            self.activities.append(ProfilerActivity.CUDA)

    def run_profile(self, batch_size: int, output_dir: str, warmup_iters: int = 5, profile_iters: int = 10) -> Dict[str, Any]:
        """Execute profiling and export Chrome trace file.
        
        Args:
            batch_size: Number of samples per batch
            output_dir: Directory to save trace file
            warmup_iters: Number of warmup iterations before profiling
            profile_iters: Number of profiled iterations
            
        Returns:
            Dictionary containing profiling analysis results
            
        Raises:
            RuntimeError: If profiling fails
        """
        os.makedirs(output_dir, exist_ok=True)
        inputs = self.model_wrapper.prepare_input(batch_size)
        trace_file = os.path.join(output_dir, "trace.json")
        
        # Step 1: Warmup to stabilize timing
        print(f"Profiling warmup: {warmup_iters} iterations...")
        for _ in range(warmup_iters):
            self.model_wrapper.forward(inputs)
            if "cuda" in self.device:
                torch.cuda.synchronize()
            elif "mps" in self.device:
                torch.mps.synchronize()

        # Step 2: Profiling execution
        print(f"Profiling execution: {profile_iters} iterations...")
        with profile(
            activities=self.activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True  # Track FLOPs if available
        ) as prof:
            with record_function("model_inference_block"):
                for _ in range(profile_iters):
                    self.model_wrapper.forward(inputs)
                    
                    # Ensure operations complete before timing
                    if "cuda" in self.device:
                        torch.cuda.synchronize()
                    elif "mps" in self.device:
                        torch.mps.synchronize()

        # Step 3: Export trace file for visualization
        print(f"Exporting Chrome trace to {trace_file}...")
        try:
            prof.export_chrome_trace(trace_file)
            print(f"Trace exported successfully. View at chrome://tracing")
        except Exception as e:
            print(f"Warning: Failed to export trace: {e}")
        
        # Step 4: Analyze bottlenecks
        return self._analyze_bottlenecks(prof)

    def _analyze_bottlenecks(self, prof) -> Dict[str, Any]:
        """Analyze profiling results and identify bottlenecks.
        
        Calculates:
        - Total execution time (including Python overhead)
        - Pure operator time (actual compute)
        - Python/framework overhead percentage
        - Top operators by execution time
        
        Args:
            prof: PyTorch profiler object
            
        Returns:
            Dictionary with timing breakdown and top operators
        """
        key_averages = prof.key_averages()
        
        # Self CPU time = actual operator execution (no Python overhead)
        # Microseconds to milliseconds conversion
        operator_time_ms = sum(item.self_cpu_time_total for item in key_averages) / 1000.0
        
        # Total CPU time includes Python/PyTorch framework overhead
        total_recorded_time_ms = sum(item.cpu_time_total for item in key_averages) / 1000.0
        
        # Calculate overhead percentage
        # Overhead = (Total - Operator) / Total * 100
        overhead_pct = 0.0
        if total_recorded_time_ms > 0:
            overhead_pct = max(0.0, ((total_recorded_time_ms - operator_time_ms) / total_recorded_time_ms) * 100.0)

        # Collect top operators
        print("\nTop operators by CPU time:")
        print(key_averages.table(sort_by="cpu_time_total", row_limit=10))

        return {
            "total_cpu_time_ms": round(total_recorded_time_ms, 2),
            "operator_time_ms": round(operator_time_ms, 2),
            "python_overhead_pct": round(overhead_pct, 2),
            "top_operators": key_averages.table(sort_by="cpu_time_total", row_limit=5)
        }