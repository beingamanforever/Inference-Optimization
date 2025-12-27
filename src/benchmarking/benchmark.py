import time
import torch
try:
    import pynvml
except ImportError:
    pynvml = None
import psutil
import os
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
            for i in range(self.measure_iter):
                start_events[i].record()
                self.model.forward(inputs)
                end_events[i].record()
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
        if self.device == "cuda":
            mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        else:
            mem_allocated = self.process.memory_info().rss / (1024 ** 2)
            mem_reserved = 0.0
            gpu_util = 0.0

        cpu_util = self.process.cpu_percent(interval=None)

        # --------------------
        # Final metrics object
        # --------------------
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
