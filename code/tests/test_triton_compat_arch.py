"""Unit tests for the Triton SM-arch compat shim (core/benchmark/triton_compat.py).

The 'a' suffix must be preserved for every arch (it is required for
arch-conditional ISA: tcgen05/TMA on sm_100a/sm_103a, wgmma on sm_90a) and only
GB10's sm_121[a] may be clamped, to sm_120. De-suffixing sm_103a -> sm_103 made
LLVM unable to select tcgen05 intrinsics on GB300 (uncatchable "LLVM ERROR:
Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st", proven 2026-06-11; see
code/upstream/triton-tcgen05-wait-st/STATUS.md probe D).
"""

from __future__ import annotations

import inspect
import os
import sys
from types import ModuleType

import pytest

from core.benchmark.triton_compat import _canonicalize_triton_arch


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # GB10 clamp: the only rewrite this shim is allowed to make.
        ("sm_121a", "sm_120"),
        ("sm_121", "sm_120"),
        # Everything else passes through verbatim, suffix included.
        ("sm_120a", "sm_120a"),
        ("sm_120", "sm_120"),
        ("sm_103a", "sm_103a"),
        ("sm_103", "sm_103"),
        ("sm_101a", "sm_101a"),
        ("sm_100a", "sm_100a"),
        ("sm_100", "sm_100"),
        ("sm_90a", "sm_90a"),
        ("sm_90", "sm_90"),
        ("sm_80", "sm_80"),
        # Non-SM tokens are returned untouched.
        ("gfx942", "gfx942"),
    ],
)
def test_canonicalize_preserves_suffix_and_clamps_only_121(raw: str, expected: str) -> None:
    assert _canonicalize_triton_arch(raw) == expected


def test_canonicalize_normalizes_spelling() -> None:
    assert _canonicalize_triton_arch(" SM103a ") == "sm_103a"
    assert _canonicalize_triton_arch("sm121a") == "sm_120"


def test_codegen_arch_clamp_is_gb10_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TRITON_CODEGEN_ARCH clamp must hit ONLY GB10 (CC 12.1): a major>=12
    catch-all would force sm_120 on real sm_120 GPUs (blocking sm_120a
    arch-conditional codegen) and mis-target future major-13 arches."""
    from core.benchmark import triton_compat

    monkeypatch.setattr(triton_compat.torch.cuda, "is_available", lambda: True)

    for capability, expected in [
        ((12, 1), "sm_120"),  # GB10: the one arch the clamp exists for
        ((12, 0), None),  # real sm_120 GPU: leave Triton's own target alone
        ((13, 0), None),  # future arch: never clamp
        ((10, 3), None),  # GB300: never clamp
    ]:
        monkeypatch.delenv("TRITON_CODEGEN_ARCH", raising=False)
        monkeypatch.setattr(
            triton_compat.torch.cuda, "get_device_capability", lambda c=capability: c
        )
        triton_compat._clamp_triton_codegen_arch()
        assert os.environ.get("TRITON_CODEGEN_ARCH") == expected, capability

    # A user-set env var is normalized (sm_121a -> sm_120) but an 'a' suffix on a
    # non-121 arch survives verbatim.
    monkeypatch.setattr(triton_compat.torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setenv("TRITON_CODEGEN_ARCH", "sm_121a")
    triton_compat._clamp_triton_codegen_arch()
    assert os.environ["TRITON_CODEGEN_ARCH"] == "sm_120"
    monkeypatch.setenv("TRITON_CODEGEN_ARCH", "sm_103a")
    triton_compat._clamp_triton_codegen_arch()
    assert os.environ["TRITON_CODEGEN_ARCH"] == "sm_103a"


def test_patched_sm_arch_from_capability() -> None:
    """The live patch must apply the same rule to Triton's capability lookup."""
    triton = pytest.importorskip("triton")  # noqa: F841
    triton_compiler = pytest.importorskip("triton.backends.nvidia.compiler")

    from core.benchmark.triton_compat import ENABLE_TRITON_PATCH, ensure_triton_compat

    if not ENABLE_TRITON_PATCH:
        pytest.skip("ENABLE_TRITON_PATCH=0")
    ensure_triton_compat()
    if not getattr(triton_compiler, "_sm_arch_patch_applied", False):
        pytest.skip("Triton too old for the SM-arch patch")

    patched = triton_compiler.sm_arch_from_capability
    orig = patched.__wrapped__

    # GB10: the one capability the shim rewrites.
    assert patched(121) == "sm_120"

    # All other capabilities: identical to what Triton itself picked, and in
    # particular any 'a' suffix Triton emitted survives (tcgen05/TMA need it).
    for capability in (120, 103, 100, 90):
        raw = orig(capability)
        assert patched(capability) == _canonicalize_triton_arch(raw)
        if raw.endswith("a"):
            assert patched(capability).endswith("a"), (
                f"'a' suffix stripped from {raw} for capability {capability}; "
                "this re-introduces the GB300 tcgen05.wait.st LLVM-select abort"
            )

    # Triton 3.7 on the GB300 stack maps CC 10.3 -> sm_103a; if that holds,
    # pin the exact value so a regression is unambiguous.
    if orig(103) == "sm_103a":
        assert patched(103) == "sm_103a"


def _fake_triton_modules(make_ptx, make_cubin, *, backend_metaclass=type):
    triton = ModuleType("triton")
    triton.__version__ = "3.8.0"
    backends = ModuleType("triton.backends")
    nvidia = ModuleType("triton.backends.nvidia")
    compiler = ModuleType("triton.backends.nvidia.compiler")

    def sm_arch_from_capability(capability: int) -> str:
        return f"sm_{capability}a" if capability in {90, 100, 103} else f"sm_{capability}"

    class CUDABackend(metaclass=backend_metaclass):
        pass

    CUDABackend.make_ptx = make_ptx
    CUDABackend.make_cubin = make_cubin
    compiler.sm_arch_from_capability = sm_arch_from_capability
    compiler.CUDABackend = CUDABackend
    triton.backends = backends
    backends.nvidia = nvidia
    nvidia.compiler = compiler
    return triton, backends, nvidia, compiler


def test_triton_38_backend_signature_and_call_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.benchmark import triton_compat

    calls: list[tuple[str, object, object, object, int]] = []

    def make_ptx(self, src, metadata, opt, capability):
        calls.append(("ptx", src, metadata, opt, capability))
        return "ptx-result"

    def make_cubin(self, src, metadata, opt, capability):
        calls.append(("cubin", src, metadata, opt, capability))
        return b"cubin-result"

    triton, backends, nvidia, compiler = _fake_triton_modules(make_ptx, make_cubin)
    for name, module in {
        "triton": triton,
        "triton.backends": backends,
        "triton.backends.nvidia": nvidia,
        "triton.backends.nvidia.compiler": compiler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(triton_compat, "_TRITON_ARCH_PATCHED", False)

    triton_compat.ensure_triton_compat()

    backend = compiler.CUDABackend()
    metadata: dict[str, object] = {}
    option = object()
    assert backend.make_ptx("src", metadata, option, 121) == "ptx-result"
    assert backend.make_cubin("src", metadata, option, 103) == b"cubin-result"
    assert calls == [
        ("ptx", "src", metadata, option, 120),
        ("cubin", "src", metadata, option, 103),
    ]
    expected = ("self", "src", "metadata", "opt", "capability")
    assert tuple(inspect.signature(compiler.CUDABackend.make_ptx).parameters) == expected
    assert tuple(inspect.signature(compiler.CUDABackend.make_cubin).parameters) == expected
    assert compiler.sm_arch_from_capability(121) == "sm_120"
    assert compiler.sm_arch_from_capability(103) == "sm_103a"


def test_triton_backend_signature_drift_fails_before_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.benchmark import triton_compat

    def make_ptx(self, src, metadata, opt):
        return "unsupported"

    def make_cubin(self, src, metadata, opt, capability):
        return b"cubin"

    triton, backends, nvidia, compiler = _fake_triton_modules(make_ptx, make_cubin)
    for name, module in {
        "triton": triton,
        "triton.backends": backends,
        "triton.backends.nvidia": nvidia,
        "triton.backends.nvidia.compiler": compiler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(triton_compat, "_TRITON_ARCH_PATCHED", False)

    with pytest.raises(RuntimeError, match="CUDABackend.make_ptx"):
        triton_compat.ensure_triton_compat()

    assert not getattr(compiler, "_sm_arch_patch_applied", False)


def test_triton_backend_patch_failure_rolls_back_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.benchmark import triton_compat

    failure_pending = True

    class FailMakeCubinOnce(type):
        def __setattr__(cls, name, value):
            nonlocal failure_pending
            if name == "make_cubin" and failure_pending and hasattr(value, "__wrapped__"):
                failure_pending = False
                raise RuntimeError("injected make_cubin assignment failure")
            super().__setattr__(name, value)

    calls: list[tuple[str, int]] = []

    def make_ptx(self, src, metadata, opt, capability):
        calls.append(("ptx", capability))
        return "ptx-result"

    def make_cubin(self, src, metadata, opt, capability):
        calls.append(("cubin", capability))
        return b"cubin-result"

    triton, backends, nvidia, compiler = _fake_triton_modules(
        make_ptx,
        make_cubin,
        backend_metaclass=FailMakeCubinOnce,
    )
    for name, module in {
        "triton": triton,
        "triton.backends": backends,
        "triton.backends.nvidia": nvidia,
        "triton.backends.nvidia.compiler": compiler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(triton_compat, "_TRITON_ARCH_PATCHED", False)

    original_sm_arch = compiler.sm_arch_from_capability
    original_make_ptx = compiler.CUDABackend.make_ptx
    original_make_cubin = compiler.CUDABackend.make_cubin

    with pytest.raises(RuntimeError, match="Changes were rolled back"):
        triton_compat.ensure_triton_compat()

    assert compiler.sm_arch_from_capability is original_sm_arch
    assert compiler.CUDABackend.make_ptx is original_make_ptx
    assert compiler.CUDABackend.make_cubin is original_make_cubin
    assert "_sm_arch_patch_applied" not in vars(compiler)
    assert "_capability_clamp_patch" not in vars(compiler.CUDABackend)
    assert triton_compat._TRITON_ARCH_PATCHED is False

    triton_compat.ensure_triton_compat()

    backend = compiler.CUDABackend()
    assert backend.make_ptx("src", {}, object(), 121) == "ptx-result"
    assert backend.make_cubin("src", {}, object(), 103) == b"cubin-result"
    assert calls == [("ptx", 120), ("cubin", 103)]
    assert compiler._sm_arch_patch_applied is True
    assert compiler.CUDABackend._capability_clamp_patch is True
    assert triton_compat._TRITON_ARCH_PATCHED is True
