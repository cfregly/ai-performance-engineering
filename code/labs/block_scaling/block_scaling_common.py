# Adapted from NVIDIA CUTLASS `dense_blockscaled_gemm_persistent.py`.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the Blackwell block scaling lab.

This lab recreates the practical flow from Colfax Research's Blackwell block
scaling tutorial while fitting into this repo's lab conventions:

- `baseline_block_scaling.py` measures a conservative software path that
  materializes block scales, applies them in BF16, and then calls matmul.
- `optimized_block_scaling.py` compiles the CUTLASS/CuTe blockscaled GEMM once
  during setup and measures only the hardware-supported execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import importlib.util
import io
import os
from pathlib import Path
from contextlib import redirect_stdout
from types import ModuleType
from typing import Any, Optional

import torch

BLOCK_SCALING_SOURCE_URL = (
    "https://research.colfax-intl.com/"
    "cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# sm_100 (B200) example shipped by the pinned cutlass submodule (v4.1.0).
SM100_EXAMPLE_PATH = (
    REPO_ROOT
    / "third_party"
    / "cutlass"
    / "examples"
    / "python"
    / "CuTeDSL"
    / "blackwell"
    / "dense_blockscaled_gemm_persistent.py"
)
# sm_103 (GB300 / Blackwell Ultra) example vendored from cutlass main (BSD-3).
# The pinned v4.1.0 submodule predates sm_103 and its DSL-4.1 example uses APIs
# removed in cutlass-dsl 4.5.x, so the GB300 path needs the sm103-specific
# example. Imports resolve from the pip nvidia-cutlass-dsl[cu13]>=4.5.2 package
# (no sibling-file deps); see vendor/README.md for provenance.
SM103_EXAMPLE_PATH = (
    Path(__file__).resolve().parent
    / "vendor"
    / "sm103_dense_blockscaled_gemm_persistent.py"
)
# Back-compat default (the sm_100 path); load_cutlass_example_module() selects
# the sm_103 example per-arch at load time via _resolve_cutlass_example_path().
CUTLASS_EXAMPLE_PATH = SM100_EXAMPLE_PATH

DEFAULT_MNKL = (8192, 8192, 1024, 1)
DEFAULT_MMA_TILER_MN = (256, 128)
DEFAULT_CLUSTER_SHAPE_MN = (2, 1)


@dataclass(frozen=True)
class BlockScalingConfig:
    """Configuration shared by the software and hardware paths."""

    mnkl: tuple[int, int, int, int] = DEFAULT_MNKL
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN
    sf_vec_size: int = 16
    tolerance: float = 1e-1
    software_dtype: torch.dtype = torch.bfloat16

    @property
    def m(self) -> int:
        return self.mnkl[0]

    @property
    def n(self) -> int:
        return self.mnkl[1]

    @property
    def k(self) -> int:
        return self.mnkl[2]

    @property
    def l(self) -> int:
        return self.mnkl[3]


@dataclass
class BlockScalingProblem:
    """Prepared tensors and optional compiled hardware kernel."""

    config: BlockScalingConfig
    module: ModuleType
    a_ref: torch.Tensor
    b_ref: torch.Tensor
    c_ref: torch.Tensor
    sfa_ref: torch.Tensor
    sfb_ref: torch.Tensor
    baseline_a: torch.Tensor
    baseline_b: torch.Tensor
    baseline_sfa: torch.Tensor
    baseline_sfb: torch.Tensor
    a_tensor: Any
    b_tensor: Any
    sfa_tensor: Any
    sfb_tensor: Any
    c_tensor: Any
    current_stream: Any
    compiled_gemm: Optional[Any] = None
    c_ref_device: Optional[torch.Tensor] = None
    prescaled_a: Optional[torch.Tensor] = None
    prescaled_b: Optional[torch.Tensor] = None

    def run_software(self) -> torch.Tensor:
        """Scale A/B in BF16 and execute a batched matmul."""
        with torch.inference_mode():
            a_scaled = self.baseline_a * self.baseline_sfa
            b_scaled = self.baseline_b * self.baseline_sfb
            return torch.bmm(a_scaled, b_scaled.transpose(1, 2)).permute(1, 2, 0).contiguous()

    def run_prescaled_bf16_gemm(self) -> torch.Tensor:
        """Execute only the BF16 GEMM on inputs that were scaled ahead of time."""
        with torch.inference_mode():
            if self.prescaled_a is None or self.prescaled_b is None:
                self.prescaled_a = (self.baseline_a * self.baseline_sfa).contiguous()
                self.prescaled_b = (self.baseline_b * self.baseline_sfb).contiguous()
            return (
                torch.bmm(self.prescaled_a, self.prescaled_b.transpose(1, 2))
                .permute(1, 2, 0)
                .contiguous()
            )

    def run_hardware(self) -> None:
        """Execute the compiled blockscaled kernel once."""
        if self.compiled_gemm is None:
            raise RuntimeError("Hardware blockscaled GEMM is not compiled")
        with torch.inference_mode():
            self.compiled_gemm(
                self.a_tensor,
                self.b_tensor,
                self.sfa_tensor,
                self.sfb_tensor,
                self.c_tensor,
                self.current_stream,
            )

    def extract_hardware_output(self) -> torch.Tensor:
        """Convert the CUTLASS output tensor into a standard torch tensor."""
        if self.c_ref_device is None:
            self.c_ref_device = self.c_ref.cuda()
        self.module.cute.testing.convert(
            self.c_tensor,
            self.module.from_dlpack(self.c_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
        )
        return self.c_ref_device

    def verify_close(self) -> dict[str, float]:
        """Compare the hardware output against the software reference path."""
        self.run_hardware()
        torch.cuda.synchronize()
        hardware = self.extract_hardware_output()
        software = self.run_software().float()
        torch.cuda.synchronize()
        diff = (hardware - software).abs()
        torch.testing.assert_close(
            hardware,
            software,
            atol=self.config.tolerance,
            rtol=1e-2,
        )
        error_stats = torch.empty(2, device=diff.device, dtype=diff.dtype)
        error_stats[0].copy_(diff.max())
        error_stats[1].copy_(diff.mean())
        error_stats_host = error_stats.detach().cpu()
        max_abs_error = float(error_stats_host[0])
        mean_abs_error = float(error_stats_host[1])
        return {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
        }


def _parse_tuple_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    parts = tuple(int(piece.strip()) for piece in value.split(","))
    if len(parts) != len(default):
        raise ValueError(f"{name} must contain exactly {len(default)} integers, got {value!r}")
    return parts


def _parse_dtype_env(name: str, default: torch.dtype) -> torch.dtype:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"{name} must be one of bf16/fp16, got {value!r}")


def parse_int_tuple(value: str, *, expected_len: int, name: str) -> tuple[int, ...]:
    """Parse a fixed-length comma-separated integer tuple."""
    parts = tuple(int(piece.strip()) for piece in value.split(","))
    if len(parts) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} integers, got {value!r}")
    return parts


def parse_software_dtype(value: str) -> torch.dtype:
    """Parse a user-facing dtype string for the software reference path."""
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"software dtype must be one of bf16/fp16, got {value!r}")


def load_lab_config_from_env() -> BlockScalingConfig:
    """Read block scaling knobs from the environment."""
    return BlockScalingConfig(
        mnkl=_parse_tuple_env("AISP_BLOCK_SCALING_MNKL", DEFAULT_MNKL),
        mma_tiler_mn=_parse_tuple_env("AISP_BLOCK_SCALING_MMA_TILER_MN", DEFAULT_MMA_TILER_MN),
        cluster_shape_mn=_parse_tuple_env(
            "AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN",
            DEFAULT_CLUSTER_SHAPE_MN,
        ),
        sf_vec_size=int(os.getenv("AISP_BLOCK_SCALING_SF_VEC_SIZE", "16")),
        tolerance=float(os.getenv("AISP_BLOCK_SCALING_TOLERANCE", "1e-1")),
        software_dtype=_parse_dtype_env(
            "AISP_BLOCK_SCALING_SOFTWARE_DTYPE",
            torch.bfloat16,
        ),
    )


def override_config(
    config: BlockScalingConfig,
    *,
    mnkl: Optional[tuple[int, int, int, int]] = None,
    mma_tiler_mn: Optional[tuple[int, int]] = None,
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: Optional[int] = None,
    tolerance: Optional[float] = None,
    software_dtype: Optional[torch.dtype] = None,
) -> BlockScalingConfig:
    """Return a config with optional CLI-style overrides applied."""
    return replace(
        config,
        mnkl=config.mnkl if mnkl is None else mnkl,
        mma_tiler_mn=config.mma_tiler_mn if mma_tiler_mn is None else mma_tiler_mn,
        cluster_shape_mn=(
            config.cluster_shape_mn if cluster_shape_mn is None else cluster_shape_mn
        ),
        sf_vec_size=config.sf_vec_size if sf_vec_size is None else int(sf_vec_size),
        tolerance=config.tolerance if tolerance is None else float(tolerance),
        software_dtype=config.software_dtype if software_dtype is None else software_dtype,
    )


def resolve_cuda_device(*, require_blackwell: bool) -> torch.device:
    """Require a CUDA device and optionally require Blackwell / SM100+."""
    if not torch.cuda.is_available():
        raise RuntimeError("The block scaling lab requires CUDA.")
    device = torch.device("cuda")
    if require_blackwell and torch.cuda.get_device_capability(device) < (10, 0):
        raise RuntimeError("The hardware blockscaled path requires Blackwell / SM100+.")
    return device


@lru_cache(maxsize=1)
def _is_blackwell_ultra() -> bool:
    """True on GB300 / Blackwell Ultra (sm_103, compute capability 10.3)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 3


def _resolve_cutlass_example_path() -> Path:
    """Pick the arch-matched blockscaled example: the vendored sm_103 example on
    Blackwell Ultra, else the sm_100 submodule example."""
    if _is_blackwell_ultra() and SM103_EXAMPLE_PATH.exists():
        return SM103_EXAMPLE_PATH
    return CUTLASS_EXAMPLE_PATH


def load_cutlass_example_module() -> ModuleType:
    """Load the NVIDIA CUTLASS blockscaled example as a Python module."""
    example_path = _resolve_cutlass_example_path()
    if not example_path.exists():
        raise FileNotFoundError(f"Missing CUTLASS example: {example_path}")
    spec = importlib.util.spec_from_file_location(
        "aisp_dense_blockscaled_gemm_persistent",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The sm_103 example imports cutlass.torch locally inside run(); the lab's
    # build_problem accesses it as module.cutlass_torch (the sm_100 example
    # exposed it at module level), so expose it here when absent.
    if not hasattr(module, "cutlass_torch"):
        import cutlass.torch as _cutlass_torch

        module.cutlass_torch = _cutlass_torch
    return module


def _select_blockscaled_kernel(module: ModuleType) -> tuple[Any, bool]:
    """Return (kernel_class, needs_use_tma_store).

    The sm_103 (GB300) kernel's ``can_implement`` and ``__init__`` take an extra
    trailing ``use_tma_store: bool`` that the sm_100 (B200) kernel does not.
    """
    sm103_cls = getattr(module, "Sm103BlockScaledPersistentDenseGemmKernel", None)
    if sm103_cls is not None:
        return sm103_cls, True
    return module.Sm100BlockScaledPersistentDenseGemmKernel, False


def _mark_compact_tensor(
    tensor: Any,
    *,
    leading_dim: int,
    stride_order: tuple[int, int, int],
    divisibility: int,
) -> None:
    tensor.mark_compact_shape_dynamic(
        mode=leading_dim,
        stride_order=stride_order,
        divisibility=divisibility,
    )


def _create_scale_factor_tensor(
    module: ModuleType,
    l: int,
    mn: int,
    k: int,
    sf_vec_size: int,
    dtype: Any,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    """Create the expanded FP32 reference scale tensor and the compact CUTLASS tensor."""

    def ceil_div(lhs: int, rhs: int) -> int:
        return (lhs + rhs - 1) // rhs

    ref_shape = (l, mn, ceil_div(k, sf_vec_size))
    mma_shape = (
        l,
        ceil_div(mn, 32 * 4),
        ceil_div(ceil_div(k, sf_vec_size), 4),
        32,
        4,
        4,
    )

    ref_f32_cpu = module.cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=(1, 2, 0),
        init_type=module.cutlass_torch.TensorInitType.RANDOM,
        init_config=module.cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )
    compact_f32_cpu = module.cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=(3, 4, 1, 5, 2, 0),
        init_type=module.cutlass_torch.TensorInitType.RANDOM,
        init_config=module.cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )

    module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        module.from_dlpack(ref_f32_cpu),
        module.from_dlpack(compact_f32_cpu),
    )
    compact_f32 = compact_f32_cpu.cuda()

    expanded_ref = (
        ref_f32_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, ceil_div(k, sf_vec_size), sf_vec_size)
        .reshape(l, mn, ceil_div(k, sf_vec_size) * sf_vec_size)
        .permute(1, 2, 0)
    )[:, :k, :]

    compact_tensor, compact_torch = module.cutlass_torch.cute_tensor_like(
        compact_f32_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    compact_tensor = module.cutlass_torch.convert_cute_tensor(
        compact_f32,
        compact_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return expanded_ref, compact_tensor, compact_torch


def theoretical_flops(config: BlockScalingConfig) -> float:
    """Return the fused GEMM work per iteration."""
    return float(2 * config.m * config.n * config.k * config.l)


def tflops_from_latency_ms(config: BlockScalingConfig, latency_ms: float) -> float:
    """Convert average latency into TFLOP/s."""
    seconds = latency_ms / 1_000.0
    if seconds <= 0:
        return 0.0
    return theoretical_flops(config) / seconds / 1e12


def direct_colfax_reference_latency_ms(
    config: BlockScalingConfig,
    *,
    warmup: int,
    iterations: int,
    skip_ref_check: bool = True,
    use_cold_l2: bool = False,
    verbose: bool = False,
) -> float:
    """Benchmark the original CUTLASS example entrypoint on the same workload."""
    module = load_cutlass_example_module()
    run_kwargs = dict(
        mnkl=config.mnkl,
        ab_dtype=module.cutlass.Float4E2M1FN,
        sf_dtype=module.cutlass.Float8E8M0FNU,
        sf_vec_size=config.sf_vec_size,
        c_dtype=module.cutlass.BFloat16,
        a_major="k",
        b_major="k",
        c_major="n",
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        tolerance=config.tolerance,
        warmup_iterations=warmup,
        iterations=iterations,
        skip_ref_check=skip_ref_check,
        use_cold_l2=use_cold_l2,
    )
    if verbose:
        latency_us = module.run(**run_kwargs)
    else:
        with redirect_stdout(io.StringIO()):
            latency_us = module.run(**run_kwargs)
    return float(latency_us) / 1_000.0


def verification_inputs(config: BlockScalingConfig) -> dict[str, torch.Tensor]:
    """Return compact metadata tensors that encode the workload configuration."""
    dtype_code = 0 if config.software_dtype == torch.bfloat16 else 1
    return {
        "mnkl": torch.tensor(config.mnkl, dtype=torch.int64, device="cpu"),
        "mma_tiler_mn": torch.tensor(config.mma_tiler_mn, dtype=torch.int64, device="cpu"),
        "cluster_shape_mn": torch.tensor(config.cluster_shape_mn, dtype=torch.int64, device="cpu"),
        "sf_meta": torch.tensor([config.sf_vec_size, dtype_code], dtype=torch.int64, device="cpu"),
    }


def verification_output_slice(
    output: torch.Tensor,
    buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a representative output tile for harness verification."""
    output_slice = output[
        : min(128, output.shape[0]),
        : min(128, output.shape[1]),
        : min(1, output.shape[2]),
    ]
    if buffer is None:
        return output_slice.detach().to(torch.float32).contiguous()
    verify_output = buffer[
        : output_slice.shape[0],
        : output_slice.shape[1],
        : output_slice.shape[2],
    ]
    verify_output.copy_(output_slice)
    return verify_output


def measure_cuda_callable(fn: Any, *, warmup: int, iterations: int) -> float:
    """Time a CUDA callable with CUDA events and return average milliseconds."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iterations):
        fn()
    end.record(current_stream)
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def build_problem(
    config: BlockScalingConfig,
    *,
    compile_hardware: bool,
) -> BlockScalingProblem:
    """Create tensors for the software path and optionally compile the hardware path."""
    resolve_cuda_device(require_blackwell=compile_hardware)
    module = load_cutlass_example_module()

    kernel_cls, needs_tma_store = _select_blockscaled_kernel(module)
    if compile_hardware:
        can_impl_args = [
            module.cutlass.Float4E2M1FN,
            module.cutlass.Float8E8M0FNU,
            config.sf_vec_size,
            module.cutlass.BFloat16,
            config.mma_tiler_mn,
            config.cluster_shape_mn,
            config.m,
            config.n,
            config.k,
            config.l,
            "k",
            "k",
            "n",
        ]
        if needs_tma_store:
            can_impl_args.append(True)
        if not kernel_cls.can_implement(*can_impl_args):
            raise TypeError(
                "Unsupported block scaling configuration: "
                f"mnkl={config.mnkl}, mma_tiler_mn={config.mma_tiler_mn}, "
                f"cluster_shape_mn={config.cluster_shape_mn}"
            )

    a_ref = module.cutlass_torch.matrix(config.l, config.m, config.k, False, module.cutlass.Float32)
    b_ref = module.cutlass_torch.matrix(config.l, config.n, config.k, False, module.cutlass.Float32)
    c_ref = module.cutlass_torch.matrix(config.l, config.m, config.n, False, module.cutlass.Float32)

    a_tensor, _ = module.cutlass_torch.cute_tensor_like(
        a_ref,
        module.cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, _ = module.cutlass_torch.cute_tensor_like(
        b_ref,
        module.cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor, _ = module.cutlass_torch.cute_tensor_like(
        c_ref,
        module.cutlass.BFloat16,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    _mark_compact_tensor(a_tensor, leading_dim=1, stride_order=(2, 0, 1), divisibility=2)
    _mark_compact_tensor(b_tensor, leading_dim=1, stride_order=(2, 0, 1), divisibility=2)
    _mark_compact_tensor(c_tensor, leading_dim=1, stride_order=(2, 0, 1), divisibility=1)

    sfa_ref, sfa_tensor, _ = _create_scale_factor_tensor(
        module,
        config.l,
        config.m,
        config.k,
        config.sf_vec_size,
        module.cutlass.Float8E8M0FNU,
    )
    sfb_ref, sfb_tensor, _ = _create_scale_factor_tensor(
        module,
        config.l,
        config.n,
        config.k,
        config.sf_vec_size,
        module.cutlass.Float8E8M0FNU,
    )

    baseline_a = a_ref.to(device="cuda", dtype=config.software_dtype).permute(2, 0, 1).contiguous()
    baseline_b = b_ref.to(device="cuda", dtype=config.software_dtype).permute(2, 0, 1).contiguous()
    baseline_sfa = sfa_ref.to(device="cuda", dtype=config.software_dtype).permute(2, 0, 1).contiguous()
    baseline_sfb = sfb_ref.to(device="cuda", dtype=config.software_dtype).permute(2, 0, 1).contiguous()

    current_stream = module.cutlass_torch.default_stream()
    compiled_gemm = None
    if compile_hardware:
        ctor_args = [config.sf_vec_size, config.mma_tiler_mn, config.cluster_shape_mn]
        if needs_tma_store:
            ctor_args.append(True)
        gemm = kernel_cls(*ctor_args)
        max_active_clusters = module.cutlass.utils.HardwareInfo().get_max_active_clusters(
            config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
        )
        compiled_gemm = module.cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            max_active_clusters,
            current_stream,
        )

    return BlockScalingProblem(
        config=config,
        module=module,
        a_ref=a_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        sfa_ref=sfa_ref,
        sfb_ref=sfb_ref,
        baseline_a=baseline_a,
        baseline_b=baseline_b,
        baseline_sfa=baseline_sfa,
        baseline_sfb=baseline_sfb,
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        c_tensor=c_tensor,
        current_stream=current_stream,
        compiled_gemm=compiled_gemm,
    )
