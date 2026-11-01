"""Assigned-host GPU checks, separate from CPU fixtures.

Set DNR_REQUIRE_GPU_TESTS=1 to fail rather than skip if a selected test's
dependencies are unavailable. Retain JUnit per-test outcomes and runtime fields.
Opcode and library-kernel assertions are stack-specific regression checks.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path

import pytest

from divide_not_reciprocal import constants as C, exhaustive, hardsigmoid, timing, triton_div
from divide_not_reciprocal.ulp import ordered, patterns, tally

torch = np = DEV = RNG = None


def unavailable(reason):
    if os.environ.get("DNR_REQUIRE_GPU_TESTS") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def dependency(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        unavailable(f"needs {name}: {exc}")


@pytest.fixture(autouse=True)
def gpu_runtime(request):
    global torch, np, DEV, RNG
    torch, np = dependency("torch"), dependency("numpy")
    if not torch.cuda.is_available():
        unavailable("needs an assigned CUDA GPU")
    DEV, RNG = torch.device("cuda"), np.random.default_rng(7)
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "cuda_build": torch.version.cuda, "numpy": np.__version__,
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
               "device": str(torch.cuda.get_device_properties(DEV))}
    if request.node.name in {"test_triton_kernels_compute_what_their_names_say",
                             "test_timed_kernels_are_the_named_kernels_and_agree"}:
        runtime["triton"] = dependency("triton").__version__
    if request.node.name.startswith("test_cuda_"):
        for name, resolve in (("nvcc", exhaustive.nvcc), ("cuobjdump", exhaustive.cuobjdump),
                              ("host_compiler", exhaustive.host_compiler)):
            try:
                exe = resolve()
            except RuntimeError as exc:
                unavailable(str(exc))
            result = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
            runtime[name] = {"path": exe, "stdout": result.stdout, "stderr": result.stderr}
    request.node.user_properties.append(("gpu_runtime", json.dumps(runtime, sort_keys=True)))
    with np.errstate(invalid="ignore"):
        yield


def assert_timed_expression(mode, out, source, divide):
    """Check each timed expression against its own contract, not a composed bound."""
    if mode == 0:
        assert int((ordered(out) - ordered(divide)).abs().max()) <= C.DIV_FULL_MAX_ULP
    elif mode == 1:
        assert torch.equal(out.view(torch.int32), source.view(torch.int32))
    elif mode == 2:
        assert torch.equal(out.view(torch.int32), divide.view(torch.int32))
    else:
        raise ValueError(f"unknown timed Triton mode {mode}")


def random_patterns(n):
    """n random bit patterns as float32 on the GPU and as a numpy float32 array (the CPU reference)."""
    bits = RNG.integers(0, 1 << 32, size=n, dtype=np.uint32)
    cpu = bits.view(np.float32)
    gpu = torch.from_numpy(bits.astype(np.int64)).to(DEV)
    gpu = torch.where(gpu >= (1 << 31), gpu - (1 << 32), gpu).to(torch.int32).view(torch.float32)
    return gpu, cpu


def cpu_classes(p: np.ndarray, q: np.ndarray):
    """Numpy's own tally (float32 on the CPU is IEEE): same, 1 ulp away, 1 ulp toward, 2+ ulp, NaN."""
    pi, qi = p.view(np.int32).astype(np.int64), q.view(np.int32).astype(np.int64)
    o = lambda i: np.where(i < 0, -(i & 0x7FFFFFFF), i)
    nan = np.isnan(p) | np.isnan(q)
    d = np.where(nan, 0, o(pi) - o(qi))
    away = (pi & 0x7FFFFFFF) > (qi & 0x7FFFFFFF)
    return (int(((d == 0) & ~nan).sum()), int(((np.abs(d) == 1) & away).sum()),
            int(((np.abs(d) == 1) & ~away).sum()), int((np.abs(d) >= 2).sum()), int(nan.sum()))


def test_patterns_cover_the_whole_space_in_order():
    x = patterns(0, 8, DEV)
    assert x.view(torch.int32).tolist() == list(range(8))
    hi = patterns((1 << 32) - 4, 1 << 32, DEV)
    assert hi.view(torch.int32).tolist() == [-4, -3, -2, -1]          # the top patterns wrap to negative int32
    assert torch.isnan(hi).all()                                      # 0xFFFFFFFC.. are negative NaNs


def test_tally_matches_numpy_on_random_patterns():
    gpu, cpu = random_patterns(1 << 20)
    q = torch.div(gpu, torch.full((1,), 6.0, device=DEV))
    p = gpu * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=DEV)
    t = tally(p, q, gpu)
    same, away, toward, far, nan = cpu_classes(cpu * np.float32(1.0 / 6.0), cpu / np.float32(6.0))
    assert (t.same, t.away1, t.toward1, t.far, t.nan_both) == (same, away, toward, far, nan)
    assert t.max_ulp == 1 and t.toward1 == 0 and t.far == 0        # signed-zero directions are classified by magnitude


def test_tensor_divisor_reference_is_ieee_like_the_cpu():
    """torch.div(x, six) with a tensor divisor matches numpy's float32 division bit for bit; the Python-scalar
    path does not (it is the reciprocal multiply)."""
    gpu, cpu = random_patterns(1 << 22)
    ref = cpu / np.float32(6.0)
    ieee = triton_div.ieee_div6(gpu).cpu().numpy()
    scalar = torch.div(gpu, 6.0).cpu().numpy()
    nan = np.isnan(ref)
    assert np.array_equal(ieee.view(np.uint32)[~nan], ref.view(np.uint32)[~nan])
    assert np.array_equal(scalar.view(np.uint32)[~nan], (cpu * np.float32(1.0 / 6.0)).view(np.uint32)[~nan])
    assert not np.array_equal(scalar.view(np.uint32)[~nan], ref.view(np.uint32)[~nan])


def test_cuda_count_matches_numpy_on_one_binade(tmp_path):
    """The CUDA walk over the binade [1, 2) gives numpy's counts for the reciprocal multiply, and the control
    variant (__fdiv_rn) gives zero, with the PTX saying which instruction each is."""
    work = tmp_path
    b = exhaustive.build(work, "default", [])
    lo, hi = C.f32_bits(1.0), C.f32_bits(2.0)
    cpu = np.arange(lo, hi, dtype=np.uint32).view(np.float32)
    same, away, toward, far, nan = cpu_classes(cpu * np.float32(1.0 / 6.0), cpu / np.float32(6.0))
    t = exhaustive.run(b, 0, lo, hi)
    e = 127
    assert t.hist[e][exhaustive.SAME] == same and t.hist[e][exhaustive.AWAY1] == away
    assert t.hist[e][exhaustive.TOWARD1] == toward == 0 and t.hist[e][exhaustive.FAR] == far == 0
    assert away == C.PREDICTED_FLIPS_PER_BINADE
    assert sum(sum(r) for r in t.hist) == hi - lo
    assert "mul.f32" in t.ptx_ops and "FMUL" in t.sass_ops
    ctrl = exhaustive.run(b, 3, lo, hi)
    assert ctrl.hist[e][exhaustive.SAME] == hi - lo and "div.rn.f32" in ctrl.ptx_ops
    fast = exhaustive.build(work, "fast_math", ["--use_fast_math"])
    fast_div = exhaustive.run(fast, 1, lo, lo + 4096)
    assert "div.rn" not in fast_div.ptx_ops
    assert fast_div.max_ulp[e] <= C.DIV_APPROX_MAX_ULP


def test_triton_kernels_compute_what_their_names_say():
    gpu, cpu = random_patterns(1 << 22)
    nan = np.isnan(cpu)
    ref = (cpu / np.float32(6.0)).view(np.uint32)[~nan]
    recip = (cpu * np.float32(1.0 / 6.0)).view(np.uint32)[~nan]
    out_rn, k_rn = triton_div.div6(gpu, 2)
    out_lit, k_lit = triton_div.div6(gpu, 3)
    out_slash, k_slash = triton_div.div6(gpu, 0)
    out_fdiv, k_fdiv = triton_div.div6(gpu, 1)
    assert np.array_equal(out_rn.cpu().numpy().view(np.uint32)[~nan], ref)          # tl.div_rn is the IEEE quotient
    assert np.array_equal(out_lit.cpu().numpy().view(np.uint32)[~nan], recip)       # the literal is fl(1/6)
    assert "div.rn.f32" in triton_div.ptx_ops(k_rn)
    assert "mul.f32" in triton_div.ptx_ops(k_lit)
    assert "div.full.f32" in triton_div.ptx_ops(k_slash)
    import triton
    if triton.__version__.split("+")[0] == "3.7.1":
        assert "div.full.f32" in triton_div.ptx_ops(k_fdiv) and "div.rn" not in triton_div.ptx_ops(k_fdiv)
    # Other Triton versions may honor ieee_rounding. The precision guarantee is tested via div_rn above.
    d = (ordered(out_slash) - ordered(torch.div(gpu, torch.full((1,), 6.0, device=DEV)))).abs()
    assert int(d[~torch.isnan(gpu)].max()) <= C.DIV_FULL_MAX_ULP                  # the PTX ISA bound holds


def test_hardsigmoid_source_read():
    """F.hardsigmoid on the GPU is clamp(x + 3, 0, 6) * fl(1/6) bit for bit (the kernel the post quotes)."""
    gpu, _ = random_patterns(1 << 22)
    import torch.nn.functional as F
    hs = F.hardsigmoid(gpu)
    docs_typed, docs_ieee, source, divide = hardsigmoid.references(gpu)
    finite = torch.isfinite(gpu)
    assert torch.equal(hs[finite].view(torch.int32), source[finite].view(torch.int32))
    assert not torch.equal(hs[finite].view(torch.int32), divide[finite].view(torch.int32))


def test_timed_kernels_are_the_named_kernels_and_agree():
    """Each timing case launches one kernel; each Triton mode meets its own expression contract."""
    x = torch.randn(1 << 20, device=DEV) * 4
    out = torch.empty_like(x)
    configs = {m: (4096, 8) for m in range(timing.N_TRITON)}
    cases = timing.make_cases(x, out, configs)
    for c in cases:
        c()
    torch.cuda.synchronize()
    _, _, source, divide = hardsigmoid.references(x)
    for m in range(timing.N_TRITON):
        cases[m]()
        torch.cuda.synchronize()
        assert_timed_expression(m, out, source, divide)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for c in cases:
            c()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert len(kernels) == len(cases), kernels
    assert all("hardsigmoid_kernel" in k for k in kernels[:timing.N_TRITON]), kernels
    assert "hardsigmoid" in kernels[timing.N_TRITON].lower(), kernels
    assert "Mul" in kernels[timing.N_TRITON + 1] and "Div" in kernels[timing.N_TRITON + 2], kernels   # the scalar path is a multiply


def test_cuda_signed_zero_and_ftz_direction_regression(tmp_path):
    """Hand-selected intervals catch the direction defect that the old CPU mirror shared."""
    work = tmp_path
    default = exhaustive.build(work, "default", [])
    # x = -3 * minsub: exact divide rounds to -zero (tie to even), reciprocal product to -minsub.
    t = exhaustive.run(default, 0, 0x80000003, 0x80000004)
    assert t.hist[256][exhaustive.AWAY1] == 1 and t.hist[256][exhaustive.TOWARD1] == 0
    fast = exhaustive.build(work, "fast_math", ["--use_fast_math"])
    # x = +/- 4..8 * minsub: IEEE quotient is +/- minsub, FTZ product is signed zero.
    # All five moves per sign are toward zero; floating comparisons under FTZ used to split the labels.
    for lo in (0x00000004, 0x80000004):
        t = exhaustive.run(fast, 0, lo, lo + 5)
        row = t.hist[lo >> 23]
        assert row[exhaustive.TOWARD1] == 5 and row[exhaustive.AWAY1] == row[exhaustive.FAR] == 0
