"""Pure protocol helpers for the bounded B200 ceiling characterization."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BandwidthCandidate:
    """One requested-byte bandwidth strategy in the bounded screen."""

    operation: str
    strategy: str
    payload_mib: int
    threads: int | None = None
    blocks_per_sm: int | None = None
    chunk_kib: int | None = None

    @property
    def candidate_id(self) -> str:
        fields = [self.operation, self.strategy, f"{self.payload_mib}mib"]
        if self.threads is not None:
            fields.append(f"t{self.threads}")
        if self.blocks_per_sm is not None:
            fields.append(f"bsm{self.blocks_per_sm}")
        if self.chunk_kib is not None:
            fields.append(f"chunk{self.chunk_kib}kib")
        return "_".join(fields)

    def to_dict(self) -> dict[str, int | str | None]:
        return {"candidate_id": self.candidate_id, **asdict(self)}


@dataclass(frozen=True)
class GemmCandidate:
    """One square BF16 GEMM size and physical operand layout."""

    size: int
    layout: str

    @property
    def candidate_id(self) -> str:
        return f"bf16_gemm_m{self.size}_k{self.size}_n{self.size}_{self.layout.lower()}"

    def to_dict(self) -> dict[str, int | str]:
        return {"candidate_id": self.candidate_id, **asdict(self)}


def parse_positive_csv(value: str, *, name: str) -> list[int]:
    """Parse a nonempty, unique comma-separated list of positive integers."""
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated integers") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} values must be unique")
    return parsed


def parse_layout_csv(value: str) -> list[str]:
    """Parse unique GEMM layouts in a stable order."""
    layouts = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not layouts:
        raise ValueError("layouts must not be empty")
    allowed = {"nn", "nt", "tn", "tt"}
    invalid = [layout for layout in layouts if layout not in allowed]
    if invalid:
        raise ValueError(f"unsupported GEMM layouts: {invalid}")
    if len(set(layouts)) != len(layouts):
        raise ValueError("layouts must be unique")
    return layouts


def bandwidth_candidates(
    payload_mib: Sequence[int],
    *,
    threads: Sequence[int] = (128, 256, 512),
    blocks_per_sm: Sequence[int] = (4, 8, 16),
    chunk_kib: Sequence[int] = (256, 1024, 4096),
) -> list[BandwidthCandidate]:
    """Build the fixed copy, read, and write search without hidden expansion."""
    candidates: list[BandwidthCandidate] = []
    for payload in payload_mib:
        candidates.append(BandwidthCandidate("copy", "torch_copy", payload))
        candidates.append(
            BandwidthCandidate(
                "copy",
                "native_scalar",
                payload,
                threads=256,
                blocks_per_sm=8,
            )
        )
        for thread_count in threads:
            for block_count in blocks_per_sm:
                candidates.append(
                    BandwidthCandidate(
                        "copy",
                        "native_uint4",
                        payload,
                        threads=thread_count,
                        blocks_per_sm=block_count,
                    )
                )
        for thread_count in threads:
            for chunk in chunk_kib:
                candidates.append(
                    BandwidthCandidate(
                        "read",
                        "native_chunk_sum",
                        payload,
                        threads=thread_count,
                        chunk_kib=chunk,
                    )
                )
        for thread_count in threads:
            for block_count in blocks_per_sm:
                candidates.append(
                    BandwidthCandidate(
                        "write",
                        "native_high_entropy_uint4",
                        payload,
                        threads=thread_count,
                        blocks_per_sm=block_count,
                    )
                )
    return candidates


def gemm_candidates(sizes: Sequence[int], layouts: Sequence[str]) -> list[GemmCandidate]:
    """Build the explicit BF16 GEMM search matrix."""
    return [GemmCandidate(size=size, layout=layout) for size in sizes for layout in layouts]


def confirmation_invocations_per_sample(
    screen_p50_ms: float,
    *,
    repeats: int,
    minimum_window_gpu_ms: float,
    margin: float = 1.05,
) -> int:
    """Choose an aggregate count expected to sustain one confirmation window."""
    if not math.isfinite(screen_p50_ms) or screen_p50_ms <= 0:
        raise ValueError("screen_p50_ms must be finite and positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not math.isfinite(minimum_window_gpu_ms) or minimum_window_gpu_ms <= 0:
        raise ValueError("minimum_window_gpu_ms must be finite and positive")
    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError("margin must be finite and at least one")
    target_sample_ms = minimum_window_gpu_ms * margin / repeats
    return max(1, math.ceil(target_sample_ms / screen_p50_ms))


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int | str]:
    """Return a deterministic percentile bootstrap interval for the median."""
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("bootstrap input must not be empty")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("bootstrap input must be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(samples, k=len(samples))) for _ in range(resamples)]
    medians.sort()
    low_index = math.floor(0.025 * (resamples - 1))
    high_index = math.ceil(0.975 * (resamples - 1))
    return {
        "method": "deterministic_percentile_bootstrap_of_median",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "low": medians[low_index],
        "high": medians[high_index],
    }
