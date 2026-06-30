"""Shared helpers for the NVSHMEM IBGDA microbenchmark."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from core.benchmark.cuda_binary_benchmark import BinaryRunResult, CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BenchmarkConfig


def _default_symmetric_size() -> str:
    """Best-effort symmetric heap sizing for the microbench."""
    return os.getenv("NVSHMEM_SYMMETRIC_SIZE", "128M")


class NvshmemIbgdaMicrobench(CudaBinaryBenchmark):
    """Wrap the nvshmem_ibgda_microbench CUDA binary for the harness."""
    multi_gpu_required = True
    preferred_ncu_replay_mode = "range"
    # Keep NCU replay bounded for multi-process microbenchmarks.
    ncu_env_overrides = {
        "AISP_NCU_PROFILE_ITERS": "200",
        "AISP_NCU_PROFILE_CTAS": "4",
    }

    def __init__(
        self,
        *,
        enable_ibgda: bool,
        mode: str = "p",
        bytes_per_message: int = 1024,
        ctas: int = 32,
        threads: int = 256,
        iters: int = 500,
        world_size: int = 2,
        ibgda_batch: int = 1,
        proxy_penalty_ms: int = 0,
        symmetric_size: str = _default_symmetric_size(),
    ) -> None:
        self.enable_ibgda = enable_ibgda
        self.mode = mode
        self.bytes_per_message = bytes_per_message
        self.ctas = ctas
        self.threads = threads
        self.iters = iters
        self.world_size = max(1, world_size)
        self.ibgda_batch = max(1, ibgda_batch)
        self.proxy_penalty_ms = max(0, proxy_penalty_ms)
        self.symmetric_size = symmetric_size
        self.nvshmemrun: Optional[str] = None
        self._parsed_metrics: Dict[str, float] = {}
        self._last_output: Optional[torch.Tensor] = None
        self._last_output_tensor: Optional[torch.Tensor] = None
        self._last_output_values: list[float] = [0.0]
        self._last_output_ready = False
        self._payload_input_tensors: Optional[Dict[str, torch.Tensor]] = None
        self._payload_device = torch.device("cpu")

        args = self._build_run_args()

        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name="nvshmem_ibgda_microbench",
            friendly_name=f"nvshmem_ibgda_{mode}",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            run_args=args,
            time_regex=None,  # Use harness timing instead of stdout parsing.
            workload_params={
                "mode_id": ord(mode[0]),
                "bytes": bytes_per_message,
                "ctas": ctas,
                "threads": threads,
                "iters": iters,
                "world_size": self.world_size,
                "ibgda": int(enable_ibgda),
                "batch_size": 1,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(requests_per_iteration=1.0)

    # --------------------------------------------------------------------- Setup/teardown
    def setup(self) -> None:
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError("SKIPPED: nvshmem_ibgda_microbench requires >=2 GPUs")

        lib_candidates = [
            Path(os.getenv("NVSHMEM_HOME", "")) / "libnvshmem_host.so",
            Path("/usr/lib/x86_64-linux-gnu/libnvshmem_host.so"),
            Path("/usr/lib/aarch64-linux-gnu/libnvshmem_host.so"),
        ]
        if not any(p.exists() for p in lib_candidates):
            raise RuntimeError(
                "SKIPPED: NVSHMEM not installed (libnvshmem_host.so not found; set NVSHMEM_HOME)"
            )

        # Prefer nvshmemrun from PATH, fall back to NVSHMEM_HOME/bin.
        launcher = shutil.which("nvshmemrun")
        if launcher is None:
            nvshmem_home = os.getenv("NVSHMEM_HOME")
            candidate = (
                Path(nvshmem_home) / "bin" / "nvshmemrun"
                if nvshmem_home
                else None
            )
            if candidate and candidate.exists():
                launcher = str(candidate)

        if launcher is None and self.world_size > 1:
            raise RuntimeError("SKIPPED: nvshmemrun not found in PATH or NVSHMEM_HOME/bin")

        self.nvshmemrun = launcher

        config = getattr(self, "_config", None)
        if config and getattr(config, "enable_ncu", False):
            ncu_timeout = getattr(config, "ncu_timeout_seconds", None)
            if ncu_timeout:
                self.timeout_seconds = max(self.timeout_seconds, int(ncu_timeout))

        self._maybe_apply_ncu_profile_overrides()

        try:
            super().setup()
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: failed to build nvshmem_ibgda_microbench ({exc})") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._payload_device = device
        self._last_output_tensor = torch.empty(1, device=device, dtype=torch.float32)
        self._payload_input_tensors = {
            "mode": torch.empty(1, device=device, dtype=torch.int64),
            "bytes": torch.empty(1, device=device, dtype=torch.int64),
            "ctas": torch.empty(1, device=device, dtype=torch.int64),
            "threads": torch.empty(1, device=device, dtype=torch.int64),
            "world_size": torch.empty(1, device=device, dtype=torch.int64),
        }
        self._payload_input_tensors["mode"][0] = ord(self.mode[0])
        self._payload_input_tensors["bytes"][0] = self.bytes_per_message
        self._payload_input_tensors["ctas"][0] = self.ctas
        self._payload_input_tensors["threads"][0] = self.threads
        self._payload_input_tensors["world_size"][0] = self.world_size

    # --------------------------------------------------------------------- Runtime helpers
    def _runtime_env(self) -> Dict[str, str]:
        env = {
            "NVSHMEM_SYMMETRIC_SIZE": self.symmetric_size,
        }
        if self.enable_ibgda:
            env.update(
                {
                    "NVSHMEM_IB_ENABLE_IBGDA": "1",
                    "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
                    "NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE": "gpumem",
                    "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH": str(self.ibgda_batch),
                }
            )
        else:
            env.update(
                {
                    "NVSHMEM_IB_ENABLE_IBGDA": "0",
                    "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH": "1",
                }
            )
        return env

    def _parse_metrics(self, stdout: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        bw = re.search(r"bw=([0-9.]+)\s*GB/s", stdout)
        if bw:
            metrics["bandwidth_gbps"] = float(bw.group(1))
        mops = re.search(r"rate=([0-9.]+)\s*MOPS", stdout)
        if mops:
            metrics["mops"] = float(mops.group(1))
        return metrics

    def _build_run_args(self) -> list[str]:
        return [
            f"--mode={self.mode}",
            f"--bytes={self.bytes_per_message}",
            f"--ctas={self.ctas}",
            f"--threads={self.threads}",
            f"--iters={self.iters}",
            f"--proxy-penalty-ms={self.proxy_penalty_ms}",
        ]

    def _maybe_apply_ncu_profile_overrides(self) -> None:
        profile_iters = os.getenv("AISP_NCU_PROFILE_ITERS")
        profile_ctas = os.getenv("AISP_NCU_PROFILE_CTAS")
        updated = False
        if profile_iters:
            self.iters = min(self.iters, int(profile_iters))
            updated = True
        if profile_ctas:
            self.ctas = min(self.ctas, int(profile_ctas))
            updated = True
        if updated:
            self.run_args = self._build_run_args()
            self._workload_params["iters"] = self.iters
            self._workload_params["ctas"] = self.ctas

    def _run_once(self) -> BinaryRunResult:
        if self.exec_path is None:
            raise RuntimeError("Executable path not set (build step missing)")
        env = os.environ.copy()
        env.update(self._runtime_env())

        if self.nvshmemrun is not None:
            cmd = [
                self.nvshmemrun,
                "-np",
                str(self.world_size),
                str(self.exec_path),
                *self.run_args,
            ]
        else:
            # Single-GPU/single-PE fallback when nvshmemrun is unavailable.
            cmd = [str(self.exec_path), *self.run_args]

        start = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=self.chapter_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.binary_name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        self._parsed_metrics = self._parse_metrics(completed.stdout)
        if self.world_size <= 1 and not self._parsed_metrics:
            # Single-PE dry run: no metrics emitted.
            self._parsed_metrics = {"single_pe": 1.0}
        self._parsed_metrics["elapsed_ms"] = elapsed_ms
        return BinaryRunResult(time_ms=elapsed_ms, raw_stdout=completed.stdout, raw_stderr=completed.stderr)

    def _run_verify(self) -> Optional[float]:
        if self._verify_exec_path is None:
            raise RuntimeError("Verify binary not built (call _build_binary_verify first)")
        env = os.environ.copy()
        env.update(self._runtime_env())
        if self.nvshmemrun is not None:
            cmd = [
                self.nvshmemrun,
                "-np",
                str(self.world_size),
                str(self._verify_exec_path),
                *self.run_args,
            ]
        else:
            cmd = [str(self._verify_exec_path), *self.run_args]
        completed = subprocess.run(
            cmd,
            cwd=self.chapter_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self._verify_exec_path.name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        match = self._verify_checksum_pattern.search(completed.stdout)
        if match:
            self._verify_checksum = float(match.group(1))
            return self._verify_checksum
        return None

    # --------------------------------------------------------------------- Benchmark API
    def benchmark_fn(self) -> None:
        self._last_result = self._run_once()
        device = self._payload_device
        self._last_output_values[0] = self._parsed_metrics.get("bandwidth_gbps", 0.0)
        self._last_output_ready = True
        self._last_output = None
        self._payload_device = device

    def capture_verification_payload(self) -> None:
        if not self._last_output_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        device = self._payload_device
        if (
            self._last_output_tensor is None
            or self._payload_input_tensors is None
            or self._last_output_tensor.device != device
        ):
            raise RuntimeError("setup() must initialize verification tensors")
        self._last_output_tensor[0] = self._last_output_values[0]
        self._last_output = self._last_output_tensor
        self._set_verification_payload(
            inputs=self._payload_input_tensors,
            output=self._last_output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_config(self) -> BenchmarkConfig:
        # Single execution; wall-clock timer is provided by harness.
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=self.timeout_seconds,
            setup_timeout_seconds=60,
            enable_memory_tracking=False,
            multi_gpu_required=True,
        )

    def get_input_signature(self) -> dict:
        params = dict(self._workload_params)
        params.pop("ibgda", None)
        dtype = params.pop("dtype", "float32")
        batch_size = int(params.pop("batch_size", 1))
        normalized_dims = {k: int(v) for k, v in params.items()}
        return simple_signature(batch_size=batch_size, dtype=dtype, **normalized_dims)

    def validate_result(self) -> Optional[str]:
        if self._last_result is None:
            return "Binary did not execute"
        if not self._parsed_metrics:
            if self.world_size <= 1:
                return None
            return "Failed to parse metrics from nvshmem_ibgda_microbench output"
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._parsed_metrics
