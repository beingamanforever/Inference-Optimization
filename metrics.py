from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class InferenceMetrics:
    """
    Metrics container for ML inference optimization.

    This class is designed to answer the core questions of inference systems:
    1. Is optimization helping real users?
    2. Where is time going (compute vs system)?
    3. What limits scale (throughput, batching, KV-cache)?
    4. Why does tail latency exist?
    """

    # --------------------
    # Identification
    # --------------------
    model_name: str
    backend: str              # PyTorch / ONNX Runtime / TensorRT
    device: str               # CPU / GPU
    batch_size: int

    # --------------------
    # End-to-End Latency
    # Raw samples in milliseconds (after warmup)
    # --------------------
    latencies_ms: List[float] = field(default_factory=list)

    # --------------------
    # Latency decomposition (milliseconds)
    # --------------------
    compute_latencies_ms: List[float] = field(default_factory=list)
    queue_wait_latencies_ms: List[float] = field(default_factory=list)

    # --------------------
    # Throughput (ground truth)
    # --------------------
    total_items_processed: int = 0
    total_time_seconds: float = 0.0

    # --------------------
    # Resource utilization
    # --------------------
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # --------------------
    # LLM-specific metrics
    # --------------------
    prefill_latency_ms: Optional[float] = None          # Initial prompt processing
    decode_latency_ms_per_token: Optional[float] = None # Per-token decode latency
    tokens_processed: Optional[int] = None
    kv_cache_memory_mb: Optional[float] = None
    kv_cache_decode_speedup: Optional[float] = None     # e.g. 5x, 10x

    # --------------------
    # Throughput stability
    # --------------------
    throughput_samples: List[float] = field(default_factory=list)

    # ======================================================================
    # Latency statistics — END TO END (USER VISIBLE)
    # ======================================================================

    @property
    def n_samples(self) -> int:
        """
        Number of latency samples.
        Required to judge statistical reliability of percentiles.
        """
        return len(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        """Typical user experience."""
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p90_latency_ms(self) -> float:
        """Upper-bound for most users."""
        return float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """Common SLO target."""
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        """Production pain / tail latency."""
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def min_latency_ms(self) -> float:
        """Best-case execution path."""
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        """Worst-case latency spike (GC, sync, queueing)."""
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    # ======================================================================
    # Latency decomposition
    # ======================================================================

    @property
    def mean_compute_latency_ms(self) -> float:
        """
        Time spent doing actual model computation.
        Dominates → kernel fusion, quantization.
        """
        return float(np.mean(self.compute_latencies_ms)) if self.compute_latencies_ms else 0.0

    @property
    def mean_queue_wait_latency_ms(self) -> float:
        """
        Time spent waiting before execution.
        Dominates → async execution, batching, scheduler tuning.
        """
        return float(np.mean(self.queue_wait_latencies_ms)) if self.queue_wait_latencies_ms else 0.0

    # ======================================================================
    # Throughput
    # ======================================================================

    @property
    def throughput_items_per_sec(self) -> float:
        """
        True throughput measured over wall-clock time.
        Primary optimization target for batching, async, KV-cache.
        """
        if self.total_time_seconds > 0:
            return self.total_items_processed / self.total_time_seconds
        return 0.0

    @property
    def throughput_stability(self) -> float:
        """
        Variance of throughput over time.
        High variance → throttling, fragmentation, scheduler instability.
        """
        return float(np.std(self.throughput_samples)) if self.throughput_samples else 0.0

    # ======================================================================
    # LLM-specific derived metrics
    # ======================================================================

    @property
    def tokens_per_second(self) -> float:
        """
        Core LLM throughput metric.
        """
        if self.tokens_processed is not None and self.total_time_seconds > 0:
            return self.tokens_processed / self.total_time_seconds
        return 0.0

    # ======================================================================
    # Export
    # ======================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Export metrics for JSON / CSV / dashboards.
        """
        return {
            # Identification
            "model": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "batch_size": self.batch_size,

            # Latency (end-to-end)
            "n_samples": self.n_samples,
            "latency_p50_ms": round(self.p50_latency_ms, 4),
            "latency_p90_ms": round(self.p90_latency_ms, 4),
            "latency_p95_ms": round(self.p95_latency_ms, 4),
            "latency_p99_ms": round(self.p99_latency_ms, 4),
            "latency_min_ms": round(self.min_latency_ms, 4),
            "latency_max_ms": round(self.max_latency_ms, 4),

            # Latency decomposition
            "compute_latency_ms": round(self.mean_compute_latency_ms, 4),
            "queue_wait_latency_ms": round(self.mean_queue_wait_latency_ms, 4),

            # Throughput
            "throughput_items_per_sec": round(self.throughput_items_per_sec, 2),
            "throughput_stability": round(self.throughput_stability, 4),

            # Resource usage
            "gpu_util_percent": round(self.gpu_utilization_percent, 2),
            "cpu_util_percent": round(self.cpu_utilization_percent, 2),
            "memory_allocated_mb": round(self.memory_allocated_mb, 2),
            "memory_reserved_mb": round(self.memory_reserved_mb, 2),

            # LLM-specific
            "prefill_latency_ms": self.prefill_latency_ms,
            "decode_latency_ms_per_token": self.decode_latency_ms_per_token,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "kv_cache_memory_mb": self.kv_cache_memory_mb,
            "kv_cache_decode_speedup": self.kv_cache_decode_speedup,
        }
