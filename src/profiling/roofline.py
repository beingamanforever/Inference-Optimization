"""Roofline model analysis for compute vs memory bottleneck identification.

The Roofline Model helps identify whether performance is limited by:
- Compute (arithmetic intensity is high, hitting peak FLOPs)
- Memory bandwidth (arithmetic intensity is low, hitting memory ceiling)

Key Metrics:
- Operational Intensity (OI) = FLOPs / Bytes Moved (FLOP/byte)
- Achieved Performance = FLOPs / Time (GFLOP/s)
"""
from typing import Dict, Any


class RooflineCalculator:
    """Calculator for Roofline Model metrics.
    
    The Roofline Model provides insights into performance bottlenecks:
    - Low OI → Memory-bound (need better data reuse)
    - High OI → Compute-bound (need faster arithmetic)
    """
    
    @staticmethod
    def calculate_metrics(flops: float, bytes_moved: float, latency_ms: float) -> Dict[str, Any]:
        """Calculate Roofline Model metrics.
        
        Formulas:
            Operational Intensity (OI) = Total FLOPs / Total Bytes Moved
                - Units: FLOP/byte
                - Higher OI = more compute per byte (compute-bound)
                - Lower OI = less compute per byte (memory-bound)
            
            Performance = FLOPs / Execution Time
                - Units: GFLOP/s
                - Measures achieved computational throughput
        
        Args:
            flops: Total floating-point operations
            bytes_moved: Total bytes transferred (memory bandwidth)
            latency_ms: Execution time in milliseconds
            
        Returns:
            Dictionary containing:
                - operational_intensity: FLOPs per byte
                - gflops_per_sec: Achieved performance in GFLOP/s
                - performance_bound: "compute" or "memory" or "unknown"
        
        Example:
            >>> calc = RooflineCalculator()
            >>> metrics = calc.calculate_metrics(
            ...     flops=4.1e9,        # 4.1 GFLOPs
            ...     bytes_moved=25e6,    # 25 MB
            ...     latency_ms=50.0      # 50 ms
            ... )
            >>> print(metrics['operational_intensity'])  # ~164 FLOP/byte
        """
        # Calculate Operational Intensity (FLOP/byte)
        oi = flops / bytes_moved if bytes_moved > 0 else 0.0
        
        # Calculate achieved performance (GFLOP/s)
        # Convert: FLOPs / (milliseconds / 1000) = FLOPs per second
        # Then divide by 1e9 to get GFLOP/s
        latency_sec = latency_ms / 1000.0
        performance_gflops = (flops / 1e9) / latency_sec if latency_ms > 0 else 0.0
        
        # Estimate bottleneck type (rough heuristic)
        # OI < 10: Memory-bound (limited by bandwidth)
        # OI > 100: Compute-bound (limited by ALU throughput)
        # 10 <= OI <= 100: Balanced or unknown
        performance_bound = "unknown"
        if oi > 0:
            if oi < 10.0:
                performance_bound = "memory"
            elif oi > 100.0:
                performance_bound = "compute"
            else:
                performance_bound = "balanced"
        
        return {
            "operational_intensity": round(oi, 2),
            "gflops_per_sec": round(performance_gflops, 2),
            "performance_bound": performance_bound
        }