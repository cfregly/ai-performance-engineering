"""Utilities for wrapping standalone CUDA binaries in the benchmark harness.

The chapter Makefiles build architecture-specific executables such as
``baseline_hbm3e_copy_sm100``. This helper compiles the requested binary (if
needed) and measures it by launching the executable from Python so the result
can participate in the standardized harness / metrics pipeline.

Verify Mode Support:
    CUDA binaries can participate in verification by emitting checksums when
    built with -DVERIFY=1. The harness parses "VERIFY_CHECKSUM: <float>" from
    stdout and compares baseline vs optimized checksums.
    
    Include cuda_verify.cuh in your CUDA code:
        #include "cuda_verify.cuh"
        VERIFY_CHECKSUM(buffer, size, &checksum);
        VERIFY_PRINT_CHECKSUM(checksum);
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from core.benchmark.detect_sm import SM_MAP
from core.benchmark.timing_parser import parse_kernel_time_ms
from core.benchmark.tma_checks import require_tma_instructions
from core.benchmark.verification import simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.cuda_capabilities import pipeline_runtime_allowed


# Default regex for parsing VERIFY_CHECKSUM from stdout
VERIFY_CHECKSUM_REGEX = r"VERIFY_CHECKSUM:\s*([0-9.eE+-]+)"

ARCH_SUFFIX = {
    "sm_100": "_sm100",
    "sm_103": "_sm103",
    "sm_120": "_sm120",
    "sm_121": "_sm121",
}
SUPPORTED_CAPABILITY_ARCH = {
    capability: architecture
    for capability, architecture in SM_MAP.items()
    if architecture in ARCH_SUFFIX
}


def _run_subprocess_capture(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while capturing stdout/stderr without pipe breakage.

    Use temporary files instead of PIPE to avoid BrokenPipe when the parent
    stdout is detached (e.g., queue runner sessions).
    """
    env = os.environ.copy()
    owner_run_id = str(os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", "")).strip()
    owner_pid = str(os.environ.get("AISP_BENCHMARK_OWNER_PID", "")).strip()
    if owner_run_id:
        env.setdefault("AISP_BENCHMARK_OWNER_RUN_ID", owner_run_id)
    if owner_pid:
        env.setdefault("AISP_BENCHMARK_OWNER_PID", owner_pid)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=False,
            stdout=stdout_file,
            stderr=stderr_file,
            timeout=timeout,
            env=env,
        )
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            args=completed.args,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _extract_binary_skip_message(completed: subprocess.CompletedProcess[str]) -> Optional[str]:
    """Return a canonical skip message for the binary exit-code contract."""
    if completed.returncode != 3:
        return None
    for output in (completed.stdout, completed.stderr):
        for line in output.splitlines():
            marker_index = line.find("SKIPPED:")
            if marker_index >= 0:
                return line[marker_index:].strip()
    return None


def detect_supported_arch() -> str:
    """Infer the CUDA architecture for building binaries from the active GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to benchmark CUDA binaries")
    
    major, minor = torch.cuda.get_device_capability()
    architecture = SUPPORTED_CAPABILITY_ARCH.get((major, minor))
    if architecture is not None:
        return architecture
    
    raise RuntimeError(
        f"Unsupported compute capability {major}.{minor}. "
        f"Supported architectures: {sorted(ARCH_SUFFIX)}"
    )


@dataclass
class BinaryRunResult:
    """Holds metrics parsed from a CUDA binary execution."""
    time_ms: Optional[float]
    raw_stdout: str
    raw_stderr: str


class CudaBinaryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper that builds and runs a CUDA executable.
    
    Supports two build modes:
    - Perf mode (default): No -DVERIFY flag, optimized for timing
    - Verify mode: Built with -DVERIFY=1, emits checksums for verification
    
    Attributes:
        verify_checksum_regex: Regex pattern for parsing VERIFY_CHECKSUM from stdout
        _verify_checksum: Last parsed verify checksum (None if not in verify mode)
        _verify_exec_path: Path to verify-mode binary (separate from perf binary)
    """

    # The signature for CUDA binaries encodes workload parameters (e.g., bytes, M/N/K).
    # There are no meaningful Python tensor inputs to validate against those parameters.
    parameter_signature_only: bool = True
    _is_deterministic: bool = True
    allowed_benchmark_fn_antipatterns = ("io",)
    
    def __init__(
        self,
        chapter_dir: Path,
        binary_name: str,
        friendly_name: str,
        *,
        iterations: int = 3,
        warmup: int = 5,
        timeout_seconds: int = 15,
        run_args: Sequence[str] = (),
        requires_pipeline_api: bool = False,
        require_tma_instructions: bool = False,
        workload_params: Optional[dict] = None,
        time_regex: Optional[str] = None,
        verify_checksum_regex: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.chapter_dir = chapter_dir
        self.binary_name = binary_name
        self.friendly_name = friendly_name
        self.iterations = iterations
        self.time_regex = time_regex  # Custom regex for parsing timing output
        self.warmup = warmup
        self.timeout_seconds = timeout_seconds
        self.run_args = list(run_args)
        self.requires_pipeline_api = requires_pipeline_api
        self.require_tma_instructions = require_tma_instructions
        self.use_reported_time = True
        self._workload_params = workload_params or {}
        
        # Verify mode support
        self.verify_checksum_regex = verify_checksum_regex or VERIFY_CHECKSUM_REGEX
        self._verify_checksum_pattern = re.compile(self.verify_checksum_regex)
        self._verify_checksum: Optional[float] = None
        self._verify_exec_path: Optional[Path] = None
        
        self.arch: Optional[str] = None
        self.exec_path: Optional[Path] = None
        self._last_result: Optional[BinaryRunResult] = None
    
    # ------------------------------------------------------------------ Helper API
    def _build_binary(self, verify_mode: bool = False) -> Path:
        """Compile the requested CUDA binary.
        
        Args:
            verify_mode: If True, build with -DVERIFY=1 flag for verification
            
        Returns:
            Path to the built binary
        """
        self.arch = detect_supported_arch()
        suffix = ARCH_SUFFIX[self.arch]
        
        # Verify builds get a _verify suffix to keep them separate
        if verify_mode:
            target = f"{self.binary_name}_verify{suffix}"
        else:
            target = f"{self.binary_name}{suffix}"
        
        build_cmd = ["make", f"ARCH={self.arch}", target]
        if verify_mode:
            build_cmd.append("VERIFY=1")
        
        try:
            completed = _run_subprocess_capture(
                build_cmd,
                cwd=self.chapter_dir,
                timeout=60,  # 60 second timeout - CUDA compilation can take time for complex kernels
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Build timeout: {target} compilation exceeded 60 seconds")
        
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to build {target} (arch={self.arch}).\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        
        path = self.chapter_dir / target
        if not path.exists():
            raise FileNotFoundError(f"Built binary not found at {path}")
        
        if verify_mode:
            self._verify_exec_path = path
        else:
            self.exec_path = path
            if self.require_tma_instructions:
                require_tma_instructions(self.exec_path)
        
        return path
    
    def _build_binary_verify(self) -> Path:
        """Build the verify-mode binary with -DVERIFY=1.
        
        Returns:
            Path to the verify binary
        """
        return self._build_binary(verify_mode=True)
    
    def _run_verify(self) -> Optional[float]:
        """Run the verify binary and parse checksum from stdout.
        
        Returns:
            Parsed checksum, or None if not found
        """
        if self._verify_exec_path is None:
            raise RuntimeError("Verify binary not built (call _build_binary_verify first)")
        
        try:
            completed = _run_subprocess_capture(
                [str(self._verify_exec_path), *self.run_args],
                cwd=self.chapter_dir,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Verify timeout: {self._verify_exec_path.name} exceeded {self.timeout_seconds} seconds")
        
        if completed.returncode != 0:
            skip_message = _extract_binary_skip_message(completed)
            if skip_message is not None:
                raise RuntimeError(skip_message)
            raise RuntimeError(
                f"Verify binary {self._verify_exec_path.name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        
        # Parse checksum from stdout
        match = self._verify_checksum_pattern.search(completed.stdout)
        if match:
            self._verify_checksum = float(match.group(1))
            return self._verify_checksum
        
        return None
    
    def check_perf_binary_clean(self) -> tuple[bool, Optional[str]]:
        """Check that perf binary doesn't contain VERIFY symbols.
        
        Uses nm to inspect the binary for VERIFY-related symbols.
        Perf binaries should NOT have any VERIFY code paths.
        
        Returns:
            Tuple of (is_clean, error_message)
        """
        if self.exec_path is None:
            return False, "Perf binary not built"
        
        try:
            # Use nm to list symbols
            completed = subprocess.run(
                ["nm", str(self.exec_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if completed.returncode != 0:
                # nm might fail on some binaries, skip check
                return True, None
            
            # Check for VERIFY-related symbols
            verify_patterns = [
                r"VERIFY",
                r"verify_checksum",
                r"_verify_sum",
            ]
            
            for pattern in verify_patterns:
                if re.search(pattern, completed.stdout, re.IGNORECASE):
                    return False, f"Found VERIFY symbol matching '{pattern}' in perf binary"
            
            return True, None
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If nm not available or times out, skip check
            return True, None
    
    def _run_once(self) -> BinaryRunResult:
        """Execute the compiled binary and parse its runtime."""
        if self.exec_path is None:
            raise RuntimeError("Executable path not set (build step missing)")
        
        try:
            completed = _run_subprocess_capture(
                [str(self.exec_path), *self.run_args],
                cwd=self.chapter_dir,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Execution timeout: {self.exec_path.name} exceeded {self.timeout_seconds} seconds")
        
        if completed.returncode != 0:
            skip_message = _extract_binary_skip_message(completed)
            if skip_message is not None:
                raise RuntimeError(skip_message)
            raise RuntimeError(
                f"{self.exec_path.name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        
        time_ms = parse_kernel_time_ms(completed.stdout, self.time_regex)
        if time_ms is None:
            raise RuntimeError(
                f"Could not parse execution time from output of {self.exec_path.name}.\n"
                f"stdout:\n{completed.stdout}"
            )
        
        return BinaryRunResult(time_ms=time_ms, raw_stdout=completed.stdout, raw_stderr=completed.stderr)
    
    # ------------------------------------------------------------------ Benchmark API
    def setup(self) -> None:
        """Build the executable once before benchmarking."""
        if self.requires_pipeline_api:
            supported, reason = pipeline_runtime_allowed()
            if not supported:
                raise RuntimeError(f"SKIPPED: CUDA Pipeline API unavailable ({reason})")
        self._build_binary(verify_mode=False)
    
    def benchmark_fn(self) -> None:
        """Launch the executable and record its runtime."""
        self._last_result = self._run_once()
    
    def teardown(self) -> None:
        """No runtime resources to release."""
        pass

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        """Return a placeholder tensor for aliasing checks.

        CUDA binaries do not expose runtime input tensors to the Python harness.
        Verification uses checksum outputs emitted by the verify-mode binary.
        """
        return {"placeholder": torch.zeros(1, dtype=torch.float32)}
    
    def get_config(self) -> BenchmarkConfig:
        """Configure harness to use CPU timers (subprocess measures wall time)."""
        return BenchmarkConfig(
            iterations=self.iterations,
            warmup=self.warmup,
            timeout_seconds=self.timeout_seconds,
            device=torch.device("cuda"),
            enable_memory_tracking=False,
            enable_profiling=False,
            # External binaries report their own timing via TIME_MS.
            # The Python harness wraps a process launch, so:
            # - Adaptive iteration scaling would explode (reported TIME_MS is tiny vs process overhead)
            # - CUDA-event vs wall-clock cross-validation is not meaningful
            adaptive_iterations=False,
            cross_validate_timing=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Ensure we captured at least one timing measurement."""
        if self._last_result is None:
            return "Binary did not execute"
        if self._last_result.time_ms is None or self._last_result.time_ms <= 0:
            return f"Invalid runtime parsed: {self._last_result.time_ms} ms"
        return None
    
    def get_input_signature(self) -> dict:
        """MANDATORY: Return input signature for verification.
        
        CUDA binaries have workload parameters baked in at compile time.
        Subclasses MUST either:
        1. Pass workload_params to __init__, OR
        2. Override this method to return explicit parameters
        
        NO AUTO-INFERENCE. NO FALLBACKS. EVERYTHING EXPLICIT.
        
        Common parameters to include:
        - N, M, K: Matrix/vector sizes
        - batch_size, seq_len: Sequence dimensions
        - hidden_dim, num_heads: Model dimensions
        
        Returns:
            Dict with workload parameters (MUST be non-empty)
            
        Raises:
            NotImplementedError: If workload_params not provided and method not overridden
        """
        if self._workload_params:
            params = dict(self._workload_params)
            dtype = params.pop("dtype", "float32")
            batch_size = int(params.pop("batch_size", 1))
            normalized_dims = {k: int(v) for k, v in params.items()}
            return simple_signature(batch_size=batch_size, dtype=dtype, **normalized_dims)
        raise NotImplementedError(
            f"{self.__class__.__name__} must provide workload_params to __init__ or override "
            "get_input_signature(). NO AUTO-INFERENCE. NO FALLBACKS. "
            "Return a dict with workload parameters (e.g., {'M': 4096, 'N': 4096, 'K': 4096})."
        )
    
    def get_output_tolerance(self) -> tuple[float, float]:
        """Verification tolerance for checksum-based CUDA binaries.

        CUDA binaries typically emit a single checksum float when built with
        `-DVERIFY=1`. For reduced-precision formats (FP8/FP4), small numeric
        differences across implementations can accumulate into a noticeably
        different checksum. Use dtype-aware tolerances to avoid false negatives
        while still catching large correctness regressions.
        """
        dtype = str(self._workload_params.get("dtype", "float32")).lower()
        if dtype in {"fp4", "nvfp4", "cuda_r_4f_e2m1"}:
            return (1e-1, 1e-2)
        if dtype in {
            "fp8",
            "nvfp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "cuda_r_8f_e4m3",
            "cuda_r_8f_e5m2",
        }:
            return (1e-2, 1e-2)
        if dtype in {"tf32", "tfloat32"}:
            return (1e-2, 1e-2)
        if dtype in {"fp16", "float16", "half", "bf16", "bfloat16"}:
            return (1e-3, 1e-3)
        return (1e-5, 1e-5)
    
    def get_verify_output(self) -> "torch.Tensor":
        """Return checksum tensor from last verify run.
        
        If verify binary hasn't been run, builds and runs it to get checksum.
        Returns a tensor containing the checksum for comparison with optimized.
        
        Returns:
            Tensor containing the checksum value
        """
        if self._last_result is None:
            raise RuntimeError(
                "get_verify_output() requires benchmark_fn() to run before verification; "
                "CUDA binaries emit checksums only when executed."
            )
        if self._verify_checksum is None:
            try:
                self.run_verify()
            except Exception as e:
                if str(e).startswith("SKIPPED:"):
                    raise RuntimeError(str(e)) from e
                raise RuntimeError(
                    f"CUDA binary verification failed: {e}. "
                    "Ensure the binary supports -DVERIFY=1 build mode."
                ) from e
        
        if self._verify_checksum is None:
            raise RuntimeError(
                "CUDA binary did not emit VERIFY_CHECKSUM. "
                "Include cuda_verify.cuh and call VERIFY_PRINT_CHECKSUM()."
            )
        
        return torch.tensor([self._verify_checksum], dtype=torch.float32)
    
    def run_verify(self) -> Optional[float]:
        """Build verify binary and run to get checksum.
        
        Convenience method that builds the verify binary (if not already built)
        and runs it to capture the checksum.
        
        Returns:
            Parsed checksum, or None if not found
        """
        if self._verify_exec_path is None:
            self._build_binary_verify()
        return self._run_verify()
    
    # Convenience accessors -----------------------------------------------------
    @property
    def last_time_ms(self) -> Optional[float]:
        return None if self._last_result is None else self._last_result.time_ms
    
    @property
    def last_stdout(self) -> Optional[str]:
        return None if self._last_result is None else self._last_result.raw_stdout
