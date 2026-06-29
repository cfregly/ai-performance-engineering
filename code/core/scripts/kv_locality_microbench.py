"""Microbenchmark for H2D transfers across pinned/pageable/NUMA-host slabs."""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from functools import partial
from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

resolve_device = partial(require_cuda_device, "CUDA device required for KV locality microbench")


class _NumaHelper:
    """Best-effort libnuma wrapper used to move pinned slabs between nodes."""

    def __init__(self) -> None:
        self._lib = self._load()
        self.nodes = self._detect_nodes()
        self.preferred = self.nodes[0] if self.nodes else None

    def _load(self) -> Optional[ctypes.CDLL]:
        for lib in ("libnuma.so.1", "libnuma.so"):
            try:
                handle = ctypes.CDLL(lib)
            except OSError:
                continue
            try:
                if handle.numa_available() < 0:  # type: ignore[attr-defined]
                    continue
            except Exception:
                continue
            return handle
        return None

    def _detect_nodes(self) -> list[int]:
        root = "/sys/devices/system/node"
        if not os.path.isdir(root):
            return []
        nodes: list[int] = []
        for entry in os.listdir(root):
            if entry.startswith("node") and entry[4:].isdigit():
                nodes.append(int(entry[4:]))
        return sorted(nodes)

    def move_to_node(self, tensor: torch.Tensor, node: int) -> bool:
        if self._lib is None:
            return False
        try:
            ptr = ctypes.c_void_p(tensor.data_ptr())
            length = ctypes.c_size_t(tensor.numel() * tensor.element_size())
            res = self._lib.numa_tonode_memory(ptr, length, ctypes.c_int(node))  # type: ignore[attr-defined]
            return res == 0
        except Exception:
            return False

    def remote_node(self) -> Optional[int]:
        if len(self.nodes) < 2:
            return None
        if self.preferred is None:
            return self.nodes[0]
        for n in self.nodes:
            if n != self.preferred:
                return n
        return None


class KvLocalityMicrobench(VerificationPayloadMixin, BaseBenchmark):
    """Compare H2D copy time for HBM vs pinned-local vs pinned-remote vs pageable."""

    allowed_benchmark_fn_antipatterns = ("sync",)

    def __init__(self) -> None:
        super().__init__()
        args = _CLI_ARGS
        self.rows = args.rows
        self.cols = args.cols
        self.iters = args.iters
        self.device = None
        self.dst = None
        self.pageable = None
        self.pinned_local = None
        self.pinned_remote = None
        self.hbm = None
        self.copy_stream = None
        self.helper = _NumaHelper()
        self.results: Dict[str, float] = {}
        self.output: Optional[torch.Tensor] = None
        self._output_values: Optional[list[float]] = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._signature_tensor: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.device = resolve_device()
        shape = (self.rows, self.cols)
        self.dst = torch.empty(shape, device=self.device, dtype=torch.float16)
        self.hbm = torch.empty(shape, device=self.device, dtype=torch.float16)
        self.pageable = torch.randn(shape, device="cpu", dtype=torch.float16)
        self.pinned_local = torch.randn(shape, device="cpu", dtype=torch.float16, pin_memory=True)
        if self.helper.preferred is not None:
            self.helper.move_to_node(self.pinned_local, self.helper.preferred)

        remote_node = self.helper.remote_node()
        if remote_node is not None:
            self.pinned_remote = torch.randn(shape, device="cpu", dtype=torch.float16, pin_memory=True)
            self.helper.move_to_node(self.pinned_remote, remote_node)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._output_tensor = torch.empty(4, dtype=torch.float32)
        self._signature_tensor = torch.empty(3, dtype=torch.float32)
        self._signature_tensor[0] = self.rows
        self._signature_tensor[1] = self.cols
        self._signature_tensor[2] = self.iters

    def _bench_copy(self, src: torch.Tensor) -> float:
        if self.dst is None or self.copy_stream is None:
            raise RuntimeError("Destination tensor not initialized; call setup() before benchmarking copies")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.iters):
            with torch.cuda.stream(self.copy_stream):
                self.dst.copy_(src, non_blocking=True)
        self.copy_stream.synchronize()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / float(self.iters)

    def benchmark_fn(self) -> None:
        if any(x is None for x in [self.dst, self.pageable, self.pinned_local, self.hbm]):
            raise RuntimeError("Buffers not initialized")

        self.results["hbm_to_hbm_ms"] = self._bench_copy(self.hbm)  # type: ignore[arg-type]
        self.results["pageable_to_hbm_ms"] = self._bench_copy(self.pageable)  # type: ignore[arg-type]
        self.results["pinned_local_to_hbm_ms"] = self._bench_copy(self.pinned_local)  # type: ignore[arg-type]
        if self.pinned_remote is not None:
            self.results["pinned_remote_to_hbm_ms"] = self._bench_copy(self.pinned_remote)
        ordered = [
            self.results["hbm_to_hbm_ms"],
            self.results["pageable_to_hbm_ms"],
            self.results["pinned_local_to_hbm_ms"],
            self.results.get("pinned_remote_to_hbm_ms", -1.0),
        ]
        self._output_values = ordered
        self.output = None

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def capture_verification_payload(self) -> None:
        if self._output_values is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output_tensor is None or self._signature_tensor is None:
            raise RuntimeError("setup() must initialize verification tensors")
        for idx, value in enumerate(self._output_values):
            self._output_tensor[idx] = value
        self.output = self._output_tensor
        self._set_verification_payload(
            inputs={"shape": self._signature_tensor},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-6, 1e-6),
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self.results:
            return None
        metrics = {k: float(v * 1000.0) for k, v in self.results.items()}  # seconds -> ms
        metrics["shape_rows"] = float(self.rows)
        metrics["shape_cols"] = float(self.cols)
        if self.helper.preferred is not None:
            metrics["numa_local_node"] = float(self.helper.preferred)
        remote = self.helper.remote_node()
        if remote is not None:
            metrics["numa_remote_node"] = float(remote)
        return metrics

    def teardown(self) -> None:
        self.dst = None
        self.pageable = None
        self.pinned_local = None
        self.pinned_remote = None
        self.hbm = None
        self.copy_stream = None
        self._output_tensor = None
        self._signature_tensor = None
        self.results = {}
        self.output = None
        self._output_values = None
        torch.cuda.empty_cache()


def get_benchmark() -> BaseBenchmark:
    return KvLocalityMicrobench()


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rows", type=int, default=256, help="Rows of the test tensor.")
    parser.add_argument("--cols", type=int, default=4096, help="Cols of the test tensor.")
    parser.add_argument("--iters", type=int, default=200, help="Iterations per copy variant.")
    return parser.parse_known_args()[0]


_CLI_ARGS = _parse_cli_args()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.setup()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
