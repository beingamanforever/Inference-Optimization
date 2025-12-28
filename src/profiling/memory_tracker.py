"""Memory usage tracking for CPU and GPU.

Tracks:
- CPU RAM usage (RSS - Resident Set Size)
- GPU memory allocation and reservation (CUDA/MPS)
- Peak memory usage for optimization analysis
"""
import torch
import psutil
import os
from typing import Dict, Any


class MemoryTracker:
    """Tracker for memory usage across CPU and GPU devices.
    
    Attributes:
        device: Device to track ('cpu', 'cuda', or 'mps')
        process: psutil Process handle for current Python process
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize memory tracker.
        
        Args:
            device: Device to monitor memory for
        """
        self.device = device
        self.process = psutil.Process(os.getpid())

    def get_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary containing:
                - cpu_ram_used_gb: CPU RAM used by process (GB)
                - gpu_reserved_mb: GPU memory reserved (CUDA/MPS) (MB)
                - gpu_allocated_mb: GPU memory actually allocated (MB)
                - gpu_peak_mb: Peak GPU memory usage (MB)
        """
        stats = {
            "cpu_ram_used_gb": round(self.process.memory_info().rss / (1024**3), 3)
        }
        
        # CUDA memory tracking
        if "cuda" in self.device and torch.cuda.is_available():
            stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
            stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
            stats["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
        
        # MPS memory tracking (Apple Silicon)
        elif "mps" in self.device and torch.backends.mps.is_available():
            stats["gpu_allocated_mb"] = round(torch.mps.current_allocated_memory() / (1024**2), 2)
            stats["gpu_reserved_mb"] = round(torch.mps.driver_allocated_memory() / (1024**2), 2)
            # MPS doesn't have peak tracking, use current as approximation
            stats["gpu_peak_mb"] = stats["gpu_allocated_mb"]
        
        return stats
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics.
        
        Useful for measuring memory usage of specific operations.
        """
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Note: MPS doesn't have reset_peak functionality
    
    def print_stats(self) -> None:
        """Print formatted memory statistics."""
        stats = self.get_stats()
        print("\nMemory Usage:")
        print("=" * 50)
        print(f"CPU RAM Used:      {stats['cpu_ram_used_gb']:.3f} GB")
        
        if "gpu_allocated_mb" in stats:
            print(f"GPU Allocated:     {stats['gpu_allocated_mb']:.2f} MB")
            print(f"GPU Reserved:      {stats['gpu_reserved_mb']:.2f} MB")
            print(f"GPU Peak:          {stats['gpu_peak_mb']:.2f} MB")
        
        print("=" * 50)