"""Adaptive parallelism routing demo (Chapter 19)."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
    _NVML_INITIALIZED = False
except Exception:  # pragma: no cover - NVML optional on dev machines
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False
    _NVML_INITIALIZED = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ParallelismStrategy(Enum):
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"
    HYBRID = "hybrid"
    DATA = "data_parallel"


@dataclass
class WorkloadMetrics:
    seq_len: int
    batch_size: int
    gpu_mem_util: float
    concurrent_reqs: int
    avg_latency_ms: float
    throughput_tokens: float
    prefill_tokens: int = 0
    decode_tokens: int = 0


@dataclass
class StrategyConfig:
    strategy: ParallelismStrategy
    tp: int
    pp: int
    dp: int
    est_latency_ms: float
    est_throughput: float
    mem_efficiency: float


class DynamicParallelismRouter:
    def __init__(self, gpus: int = 8) -> None:
        self.gpus = gpus
        self.current = ParallelismStrategy.TENSOR
        self.last_switch = 0.0
        self.cooldown = 5.0
        self.profiles: Dict[ParallelismStrategy, StrategyConfig] = {
            ParallelismStrategy.TENSOR: StrategyConfig(ParallelismStrategy.TENSOR, gpus, 1, 1, 50.0, 900.0, 0.7),
            ParallelismStrategy.PIPELINE: StrategyConfig(ParallelismStrategy.PIPELINE, 1, gpus, 1, 85.0, 700.0, 0.9),
            ParallelismStrategy.HYBRID: StrategyConfig(ParallelismStrategy.HYBRID, gpus // 2, 2, 1, 65.0, 800.0, 0.85),
            ParallelismStrategy.DATA: StrategyConfig(ParallelismStrategy.DATA, 1, 1, gpus, 120.0, 1400.0, 0.6),
        }

    def choose(self, metrics: WorkloadMetrics) -> ParallelismStrategy:
        long_prefill = metrics.prefill_tokens > 0 and metrics.prefill_tokens >= metrics.decode_tokens * 2
        steady_decode = metrics.decode_tokens > 0 and metrics.decode_tokens <= 256
        if metrics.concurrent_reqs > 32 and steady_decode:
            return ParallelismStrategy.DATA
        if metrics.seq_len > 1024 or metrics.gpu_mem_util > 0.85 or long_prefill:
            if metrics.seq_len > 4096 or metrics.gpu_mem_util > 0.92:
                return ParallelismStrategy.PIPELINE
            return ParallelismStrategy.HYBRID
        return ParallelismStrategy.TENSOR

    def score(self, metrics: WorkloadMetrics, strategy: ParallelismStrategy) -> float:
        cfg = self.profiles[strategy]
        latency = max(0.0, 1.0 - metrics.avg_latency_ms / 200.0)
        throughput = min(1.0, metrics.throughput_tokens / 2000.0)
        memory = 1.0 - metrics.gpu_mem_util
        if strategy == ParallelismStrategy.TENSOR:
            return 0.6 * latency + 0.3 * throughput + 0.1 * memory
        if strategy == ParallelismStrategy.PIPELINE:
            return 0.2 * latency + 0.3 * throughput + 0.5 * memory
        if strategy == ParallelismStrategy.HYBRID:
            return 0.4 * latency + 0.4 * throughput + 0.2 * memory
        return 0.1 * latency + 0.7 * throughput + 0.2 * memory

    def maybe_switch(self, metrics: WorkloadMetrics) -> Optional[StrategyConfig]:
        now = time.perf_counter()
        if now - self.last_switch < self.cooldown:
            return None
        proposed = self.choose(metrics)
        if proposed == self.current:
            return None
        current_score = self.score(metrics, self.current)
        proposed_score = self.score(metrics, proposed)
        if proposed_score > current_score * 1.1:
            logger.info("Switching %s -> %s (score %.2f -> %.2f)", self.current.value, proposed.value, current_score, proposed_score)
            self.current = proposed
            self.last_switch = now
            return self.profiles[proposed]
        return None


def collect_metrics() -> WorkloadMetrics:
    seq_len = random.choice([128, 256, 512, 1024, 2048, 4096])
    load = random.uniform(0.3, 0.9)
    prefill_tokens = seq_len
    decode_tokens = random.randint(32, 256)
    return WorkloadMetrics(
        seq_len=seq_len,
        batch_size=random.randint(4, 32),
        gpu_mem_util=0.4 + load * 0.5,
        concurrent_reqs=int(load * 50),
        avg_latency_ms=60 + load * 90,
        throughput_tokens=1100 * (2.0 - load),
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )


def simulate(router: DynamicParallelismRouter, seconds: float = 10.0) -> None:
    start = time.perf_counter()
    switches = 0
    processed = 0
    latency_total = 0.0
    while time.perf_counter() - start < seconds:
        metrics = collect_metrics()
        cfg = router.maybe_switch(metrics)
        if cfg:
            switches += 1
        cfg = router.profiles[router.current]
        simulated_latency = cfg.est_latency_ms * (metrics.seq_len / 1024.0) * (1.0 + metrics.gpu_mem_util * 0.3)
        processing_time = simulated_latency / 1000.0
        time.sleep(processing_time)
        processed += 1
        latency_total += simulated_latency
        if processed % 50 == 0:
            avg = latency_total / processed
            logger.info("Processed %d requests, avg latency %.1f ms, strategy %s, switches %d",
                        processed, avg, router.current.value, switches)

    print(f"Processed {processed} requests in {seconds:.1f}s, switches={switches}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    router = DynamicParallelismRouter(gpus=8)
    print("Chapter 19: Adaptive parallelism demo")
    simulate(router)


_DEFAULT_ROUTER = DynamicParallelismRouter(gpus=8)


def telemetry_gpu_mem_util(default: float = 0.0, device_index: int = 0) -> float:
    """Optional NVML-backed memory utilisation fraction."""
    global _NVML_INITIALIZED
    if not _NVML_AVAILABLE:
        return default
    try:
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()  # type: ignore[attr-defined]
            _NVML_INITIALIZED = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)  # type: ignore[attr-defined]
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
        return max(default, util.memory / 100.0)
    except Exception:
        return default


def choose_worker_pool(
    seq_len: int,
    gpu_mem_util: float,
    concurrent_reqs: int,
    *,
    batch_size: int = 1,
    prefill_tokens: int = 0,
    decode_tokens: int = 0,
    telemetry_fn: Optional[Callable[[], float]] = None,
) -> StrategyConfig:
    """
    Return StrategyConfig matching the book's decision function.

    Args:
        seq_len: Prompt sequence length.
        gpu_mem_util: Current GPU memory utilisation (0.0 - 1.0).
        concurrent_reqs: Outstanding decoding requests.
        batch_size: Batch size for the new request.
        prefill_tokens: Estimated tokens processed during prefill.
        decode_tokens: Estimated tokens for decode phase.
        telemetry_fn: Optional callable returning live GPU memory utilisation.
    """
    live_util = telemetry_fn() if telemetry_fn is not None else None
    util = float(live_util) if live_util is not None else gpu_mem_util
    metrics = WorkloadMetrics(
        seq_len=seq_len,
        batch_size=batch_size,
        gpu_mem_util=util,
        concurrent_reqs=concurrent_reqs,
        avg_latency_ms=80.0,
        throughput_tokens=1200.0,
        prefill_tokens=prefill_tokens if prefill_tokens else seq_len,
        decode_tokens=decode_tokens if decode_tokens else max(128, seq_len // 4),
    )
    strategy = _DEFAULT_ROUTER.choose(metrics)
    return _DEFAULT_ROUTER.profiles[strategy]


if __name__ == "__main__":
    main()
