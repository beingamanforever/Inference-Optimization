import time
import torch
<<<<<<< HEAD
try:
    import pynvml
except ImportError:
    pynvml = None
import psutil
import os
=======
import pynvml
import psutil
import os
import gc
>>>>>>> 01490da (Restructure: Add complete benchmarking suite with profiling tools)
from typing import List
from .metrics import InferenceMetrics


class Benchmark:
    def __init__(self, model, warmup_iter: int = 20, measure_iter: int = 100):
        self.model = model
        self.warmup_iter = warmup_iter
        self.measure_iter = measure_iter
        self.device = model.device

        # Track this process only
        self.process = psutil.Process(os.getpid())

<<<<<<< HEAD
        # NVML setup
        self.nvml_available = False
        self.gpu_handle = None
        if self.device == "cuda" and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_available = True
            except Exception:
                print("Warning: NVML not available")

    @torch.inference_mode()
    def run_batch_sweep(self, batch_sizes: List[int]) -> List[InferenceMetrics]:
        results = []
        print(f"🚀 Benchmarking {self.model.__class__.__name__} on {self.device}")

        for bs in batch_sizes:
            results.append(self._measure_batch(bs))

        return results

    def _measure_batch(self, batch_size: int) -> InferenceMetrics:
        inputs = self.model.prepare_input(batch_size)

        # --------------------
        # Warmup
        # --------------------
        for _ in range(self.warmup_iter):
            self.model.forward(inputs)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # --------------------
        # Measurement setup
        # --------------------
        latencies_ms = []
        compute_latencies_ms = []
        queue_wait_latencies_ms = []

        self.process.cpu_percent(interval=None)

        gpu_utils = []

        if self.device == "cuda":
            start_events = [torch.cuda.Event(True) for _ in range(self.measure_iter)]
            end_events = [torch.cuda.Event(True) for _ in range(self.measure_iter)]
            torch.cuda.reset_peak_memory_stats()

        # --------------------
        # Timed execution
        # --------------------
        total_start = time.perf_counter()

        if self.device == "cuda":
=======
        # NVML setup for GPU monitoring
        self.nvml_available = False
        self.gpu_handle = None
        if self.device == "cuda":
            try:
                pynvml.nvmlInit() # connect to NVIDIA driver
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_available = True
            except Exception as e:
                print(f"Warning: NVML not available - {e}")
                print("GPU utilization metrics will not be collected")

    def __del__(self):
        """Cleanup NVML resources on object destruction."""
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown() # disconnect from NVIDIA driver
            except Exception:
                pass

    @torch.inference_mode()
    # orchestrate benchmarking over multiple batch sizes
    def run_batch_sweep(self, batch_sizes: List[int]) -> List[InferenceMetrics]:
        results = []
        print(f"Benchmarking {self.model.__class__.__name__} on {self.device}")
        for bs in batch_sizes:
            gc.collect() # garbage collector 
            if self.device == "cuda": # cleanr pytorch's internal cache
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            results.append(self._measure_batch(bs))
        return results

    
    def _measure_batch(self, batch_size: int) -> InferenceMetrics:
        """Measure inference performance for a specific batch size.
        Args:
            batch_size: Number of samples to process in each batch
        Returns:
            InferenceMetrics object containing all performance metrics
        """
        inputs = self.model.prepare_input(batch_size) # creates a batch of data

        # warmup phase: ensures when we start measuring, things are stable
        # Stabilize caches, memory allocation, and kernel compilation
        for _ in range(self.warmup_iter):
            self.model.forward(inputs)

        # ensure all warmup operations complete before measurement
        if self.device == "cuda":
            torch.cuda.synchronize() # Wait for GPU to finish
        elif self.device == "mps":
            torch.mps.synchronize() # Wait for MPS to finish

        # measurement setup 
        latencies_ms = [] # End-to-end time per iteration
        compute_latencies_ms = []  # Pure GPU/CPU compute time
        queue_wait_latencies_ms = [] # Time waiting in queue
        gpu_utils = [] # GPU utilization % samples

        # initialize CPU utilization tracking (right now returns 0.0)
        self.process.cpu_percent(interval=None)

        # prepare GPU-specific timing infrastructure
        if self.device == "cuda":
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.measure_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.measure_iter)]
            # Reset peak memory to measure only this batch's allocation
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS doesn't have Events, so we'll use CPU timing
            torch.mps.empty_cache()
            
        # Timed execution
        total_start = time.perf_counter()

        if self.device == "cuda":
            # GPU measurement using CUDA Events for precise timing
>>>>>>> 01490da (Restructure: Add complete benchmarking suite with profiling tools)
            for i in range(self.measure_iter):
                start_events[i].record()
                self.model.forward(inputs)
                end_events[i].record()
<<<<<<< HEAD
                torch.cuda.synchronize()

                if self.nvml_available:
                    gpu_utils.append(
                        pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    )

            for s, e in zip(start_events, end_events):
                compute_latencies_ms.append(s.elapsed_time(e))
                latencies_ms.append(s.elapsed_time(e))
                queue_wait_latencies_ms.append(0.0)

        else:
            for _ in range(self.measure_iter):
                start = time.perf_counter()
                self.model.forward(inputs)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies_ms.append(latency_ms)
                compute_latencies_ms.append(latency_ms)
                queue_wait_latencies_ms.append(0.0)

        total_end = time.perf_counter()

        # --------------------
        # Resource metrics
        # --------------------
=======
                torch.cuda.synchronize()  # ensure completion before next iteration

                # Sample GPU utilization if available
                if self.nvml_available:
                    try:
                        util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_utils.append(util_rates.gpu)
                    except Exception:
                        # skip this sample if utilization query fails
                        pass

            # Extract timing measurements from CUDA events
            for s, e in zip(start_events, end_events):
                elapsed_ms = s.elapsed_time(e)
                compute_latencies_ms.append(elapsed_ms)
                latencies_ms.append(elapsed_ms)
                queue_wait_latencies_ms.append(0.0) # Queue wait is zero for synchronous execution

        else: # cpu / mps measurements
            # (mps doesn't support CUDA events, so it uses the same timing as CPU)
            for _ in range(self.measure_iter):
                start = time.perf_counter()
                self.model.forward(inputs)
                if self.device == "mps":
                    torch.mps.synchronize()  # minor ensure completion
                end = time.perf_counter()
                latency_ms = (end - start) * 1000.0
                latencies_ms.append(latency_ms)
                compute_latencies_ms.append(latency_ms)
                queue_wait_latencies_ms.append(0.0)  # Queue wait is zero for synchronous execution

        total_end = time.perf_counter()

        # resource metrics collection, memory and utilization
>>>>>>> 01490da (Restructure: Add complete benchmarking suite with profiling tools)
        if self.device == "cuda":
            mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
<<<<<<< HEAD
        else:
            mem_allocated = self.process.memory_info().rss / (1024 ** 2)
            mem_reserved = 0.0
            gpu_util = 0.0

        cpu_util = self.process.cpu_percent(interval=None)

        # --------------------
        # Final metrics object
        # --------------------
=======
            cpu_util = self.process.cpu_percent(interval=None)
        elif self.device == "mps":
            mem_allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
            mem_reserved = torch.mps.driver_allocated_memory() / (1024 ** 2)
            gpu_util = 0.0  # mps doesn't expose utilization metrics
            cpu_util = self.process.cpu_percent(interval=None)
        else:
            mem_info = self.process.memory_info()
            mem_allocated = mem_info.rss / (1024 ** 2)  # resident set size
            mem_reserved = mem_info.vms / (1024 ** 2)   # virtual memory size
            gpu_util = 0.0
            cpu_util = self.process.cpu_percent(interval=None)

        # final metrics object
>>>>>>> 01490da (Restructure: Add complete benchmarking suite with profiling tools)
        return InferenceMetrics(
            model_name=self.model.__class__.__name__,
            backend=self.model.backend,
            device=self.device,
            batch_size=batch_size,
            latencies_ms=latencies_ms,
            compute_latencies_ms=compute_latencies_ms,
            queue_wait_latencies_ms=queue_wait_latencies_ms,
            total_items_processed=batch_size * self.measure_iter,
            total_time_seconds=(total_end - total_start),
            gpu_utilization_percent=gpu_util,
            cpu_utilization_percent=cpu_util,
            memory_allocated_mb=mem_allocated,
            memory_reserved_mb=mem_reserved,
        )
