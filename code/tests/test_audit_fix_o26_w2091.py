"""Regressions for W2-091: the software-pipelining pair must stay controlled.

The launcher runs one block-scoped cuda::pipeline kernel for the optimized arm.
The optimized arm's description must name that kernel's mechanism, not a TMA
path that is never dispatched, and the extension must not carry unlaunched
kernels. The CUDA source checks pair with a B200 run of the GPU test below.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from labs.software_pipelining.baseline_tile_pipeline import (
    get_benchmark as get_baseline_benchmark,
)
from labs.software_pipelining.optimized_tile_pipeline import (
    get_benchmark as get_optimized_benchmark,
)
from labs.software_pipelining.software_pipelining_common import (
    reference_tile_pipeline,
)

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "software_pipelining"
KERNEL_SOURCE = (LAB_DIR / "software_pipelining_kernels.cu").read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    assert source.count(start) == 1, start
    body = source.split(start, 1)[1]
    assert end in body, end
    return body.split(end, 1)[0]


def _launcher() -> str:
    return _section(KERNEL_SOURCE, "torch::Tensor launch_common(", "}  // namespace")


def _kernel_body(name: str) -> str:
    start = f"__global__ void {name}("
    body = KERNEL_SOURCE.split(start, 1)[1]
    next_kernel = body.find("__global__ void ")
    return body if next_kernel < 0 else body[:next_kernel]


def _full_tile_compute_loop(kernel_body: str) -> str:
    return _section(kernel_body, "const float4 a = lhs_vec[v];", "out_vec[v] = r;")


def test_optimized_description_names_the_dispatched_cuda_pipeline_kernel() -> None:
    notes = get_optimized_benchmark().describe()
    launcher = _launcher()
    pipeline_branch = _section(launcher, "if (use_pipeline) {", "} else {")
    dispatched = re.findall(r"(\w+)<<<", pipeline_branch)
    assert dispatched == ["optimized_tile_pipeline_kernel"]
    optimized_kernel = _kernel_body("optimized_tile_pipeline_kernel")
    assert "cuda::make_pipeline(cta, pipe_state)" in optimized_kernel
    assert "cuda::memcpy_async(" in optimized_kernel

    assert "cuda::pipeline" in notes
    assert "cuda::memcpy_async" in notes
    assert "only the serialized versus two-stage schedule differs" in notes
    for stale_claim in ("TMA", "cp.async.bulk", "warp-specialized", "mbarrier", "Falls back", "sm_90"):
        assert stale_claim not in notes, stale_claim


def test_every_kernel_in_the_extension_is_launched() -> None:
    defined = set(re.findall(r"__global__\s+void\s+(\w+)\s*\(", KERNEL_SOURCE))
    launched = set(re.findall(r"(\w+)<<<", _launcher()))
    assert defined == {"baseline_tile_pipeline_kernel", "optimized_tile_pipeline_kernel"}
    assert defined == launched
    assert "__CUDA_ARCH__" not in KERNEL_SOURCE


def test_pair_shares_launch_configuration_and_compute_loop() -> None:
    launches = re.findall(r"(\w+)<<<([^>]*)>>>", _launcher())
    assert sorted(name for name, _ in launches) == [
        "baseline_tile_pipeline_kernel",
        "optimized_tile_pipeline_kernel",
    ]
    assert {config for _, config in launches} == {"grid, block, 0, stream"}
    assert _full_tile_compute_loop(
        _kernel_body("baseline_tile_pipeline_kernel")
    ) == _full_tile_compute_loop(_kernel_body("optimized_tile_pipeline_kernel"))

    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()
    assert baseline.workload == optimized.workload
    assert "TMA" not in baseline.describe()


def test_launcher_rejects_inputs_the_int_and_float4_paths_cannot_handle() -> None:
    checks = _section(KERNEL_SOURCE, "torch::Tensor check_inputs(", "torch::Tensor launch_common(")
    assert "lhs.numel() <= std::numeric_limits<int>::max() - kTileElems" in checks
    assert "repeat_fmas <= std::numeric_limits<int>::max()" in checks
    assert "reinterpret_cast<std::uintptr_t>(lhs.data_ptr()) % 16 == 0" in checks
    assert "reinterpret_cast<std::uintptr_t>(rhs.data_ptr()) % 16 == 0" in checks


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: the software-pipelining kernels need a CUDA device",
)
def test_pair_matches_reference_on_full_tail_and_short_inputs() -> None:
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip(f"SKIPPED: cuda::pipeline needs SM80+ (found SM {major}.{minor})")
    from labs.software_pipelining.software_pipelining_extension import (
        load_software_pipelining_extension,
    )

    module = load_software_pipelining_extension()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(4242)
    # Full tiles only, full tiles plus a ragged tail, and a single partial tile.
    for length in (1 << 20, (1 << 20) + 37, 1000):
        lhs = torch.randn(length, generator=generator, device="cuda")
        rhs = torch.randn(length, generator=generator, device="cuda")
        reference = reference_tile_pipeline(lhs, rhs, 2)
        baseline = module.baseline_tile_pipeline(lhs, rhs, 2)
        optimized = module.optimized_tile_pipeline(lhs, rhs, 2)
        torch.cuda.synchronize()
        torch.testing.assert_close(baseline, reference, rtol=0.0, atol=5e-4)
        torch.testing.assert_close(optimized, reference, rtol=0.0, atol=5e-4)
        torch.testing.assert_close(optimized, baseline, rtol=1e-6, atol=1e-6)

    padded = torch.randn(4097, generator=generator, device="cuda")
    misaligned = padded[1:]
    assert misaligned.is_contiguous()
    for op in (module.baseline_tile_pipeline, module.optimized_tile_pipeline):
        with pytest.raises(RuntimeError, match="16-byte aligned lhs data"):
            op(misaligned, padded[:4096], 2)
