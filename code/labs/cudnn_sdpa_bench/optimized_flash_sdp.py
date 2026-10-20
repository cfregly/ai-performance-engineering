"""Optimized variant for the cuDNN SDPA lab. It pins the Flash backend by default."""

from __future__ import annotations

from labs.cudnn_sdpa_bench.baseline_flash_sdp import FlashSDPLabBenchmark, _parse_backend_overrides

_DEFAULT_BACKEND = "flash"


class OptimizedFlashSDPLabBenchmark(FlashSDPLabBenchmark):
    """Flash-pinned SDPA benchmark that reads only --optimized-backend overrides."""

    def __init__(self) -> None:
        super().__init__(backend=_DEFAULT_BACKEND)

    def apply_target_overrides(self, argv: list[str]) -> None:
        backend = _parse_backend_overrides(argv)["optimized"]
        if backend is not None:
            self.backend = backend


def get_benchmark() -> FlashSDPLabBenchmark:
    # Same benchmark as the baseline, pinned to the Flash backend.
    return OptimizedFlashSDPLabBenchmark()
