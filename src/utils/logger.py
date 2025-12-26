"""Logging utilities for inference optimization."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "inference_optimizer",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = "logs"
) -> logging.Logger:
    """Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional log file name (if None, uses timestamp)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        
        file_path = log_path / log_file
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


class BenchmarkLogger:
    """Specialized logger for benchmarking results."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize with existing logger.
        
        Args:
            logger: Logger instance from setup_logger()
        """
        self.logger = logger
    
    def log_system_info(self, info: dict):
        """Log system information.
        
        Args:
            info: Dictionary with system info (CPU, GPU, memory, etc.)
        """
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 60)
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_model_info(self, model_name: str, info: dict):
        """Log model information.
        
        Args:
            model_name: Name of the model
            info: Dictionary with model info
        """
        self.logger.info("-" * 60)
        self.logger.info(f"MODEL: {model_name}")
        self.logger.info("-" * 60)
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_benchmark_start(self, model_name: str, batch_size: int, device: str):
        """Log start of benchmark.
        
        Args:
            model_name: Name of model being benchmarked
            batch_size: Batch size
            device: Device (cpu/cuda)
        """
        self.logger.info("")
        self.logger.info(f"Starting benchmark: {model_name}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Device: {device}")
    
    def log_benchmark_results(self, results: dict):
        """Log benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        self.logger.info("Benchmark Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_comparison(self, baseline: float, optimized: float, metric: str = "Latency"):
        """Log before/after comparison.
        
        Args:
            baseline: Baseline value
            optimized: Optimized value
            metric: Name of metric being compared
        """
        speedup = baseline / optimized
        improvement_pct = ((baseline - optimized) / baseline) * 100
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"{metric} Comparison")
        self.logger.info("=" * 60)
        self.logger.info(f"  Baseline:    {baseline:.4f}")
        self.logger.info(f"  Optimized:   {optimized:.4f}")
        self.logger.info(f"  Speedup:     {speedup:.2f}x")
        self.logger.info(f"  Improvement: {improvement_pct:.2f}%")
        self.logger.info("=" * 60)


def get_system_info() -> dict:
    """Get system information for logging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import torch
    import psutil
    
    info = {
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
    }
    
    # GPU info
    if torch.cuda.is_available():
        info["CUDA Available"] = True
        info["CUDA Version"] = torch.version.cuda
        info["GPU Count"] = torch.cuda.device_count()
        info["GPU Name"] = torch.cuda.get_device_name(0)
        info["GPU Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    else:
        info["CUDA Available"] = False
    
    return info


# Test code
if __name__ == "__main__":
    print("Testing logging utilities...")
    
    # Test basic logger
    print("\n=== Testing Basic Logger ===")
    logger = setup_logger(
        name="test_logger",
        level=logging.DEBUG,
        log_dir="test_logs"
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test BenchmarkLogger
    print("\n=== Testing BenchmarkLogger ===")
    bench_logger = BenchmarkLogger(logger)
    
    # Log system info
    sys_info = get_system_info()
    bench_logger.log_system_info(sys_info)
    
    # Log model info
    model_info = {
        "Parameters": "25.6M",
        "Architecture": "ResNet-50",
        "Input Size": "224x224x3"
    }
    bench_logger.log_model_info("ResNet-50", model_info)
    
    # Log benchmark
    bench_logger.log_benchmark_start("ResNet-50", batch_size=8, device="cuda")
    
    results = {
        "latency_mean_ms": 12.34,
        "latency_std_ms": 0.56,
        "throughput_samples_per_sec": 648.3,
        "memory_allocated_mb": 245.8
    }
    bench_logger.log_benchmark_results(results)
    
    # Log comparison
    bench_logger.log_comparison(
        baseline=12.34,
        optimized=6.17,
        metric="Latency (ms)"
    )
    
    print("\n✓ All tests passed!")
    print(f"Check log files in: test_logs/")