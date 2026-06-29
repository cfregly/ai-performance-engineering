from __future__ import annotations
"""GPU occupancy heuristics with PyTorch (CUDA 13.0 / PyTorch 2.10).

The focus is on sizing work to keep SMs busy, batching small ops, fusing chains,
and choosing tensor layouts that map to efficient kernels.
"""
try:
    from ch08.arch_config import (
        get_compile_mode,
        maybe_compile,
        should_use_compile,
    )
except ImportError:  # Fallback if the helper is absent.
    def maybe_compile(fn, *, default_mode: str = "reduce-overhead"):
        return fn

    def should_use_compile() -> bool:
        return False

    def get_compile_mode(default: str = "reduce-overhead") -> str:
        return default

import time
import torch


def _need_cuda() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available; occupancy demos require a GPU.")
        return False
    return True


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<32}: {elapsed_ms:7.3f} ms")
    return elapsed_ms


def matmul_size_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    small = (128, 128)
    large = (4096, 4096)
    a_small = torch.randn(*small, device=device)
    b_small = torch.randn(*small, device=device)
    a_large = torch.randn(*large, device=device)
    b_large = torch.randn(*large, device=device)

    print("\n=== Matrix size and occupancy ===")
    _benchmark("mm 128x128", lambda: torch.mm(a_small, b_small), iters=200)
    _benchmark("mm 4096x4096", lambda: torch.mm(a_large, b_large), iters=5)


def batching_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    batched = torch.randn(32, 256, 256, device=device)
    tensors = batched.unbind(0)
    individual_out = torch.empty_like(batched)
    individual_outputs = individual_out.unbind(0)
    batched_out = torch.empty_like(batched)

    print("\n=== Batching small pointwise ops ===")

    def individual():
        for t, out_t in zip(tensors, individual_outputs, strict=True):
            torch.relu(t, out=out_t)
            torch.nn.functional.silu(out_t, inplace=True)
        return individual_out

    def batched_op():
        torch.relu(batched, out=batched_out)
        return torch.nn.functional.silu(batched_out, inplace=True)

    indiv_ms = _benchmark("Individual 32x", individual, iters=5)
    batch_ms = _benchmark("Batched (N=32)", batched_op, iters=5)
    print(f"Speedup (individual / batched): {indiv_ms / batch_ms:5.2f}x")


def compile_fusion_demo() -> None:
    if not _need_cuda():
        return
    if not hasattr(torch, "compile"):
        print("torch.compile not available; skipping fusion demo.")
        return

    device = torch.device("cuda")
    x = torch.randn(2048, 2048, device=device)

    def eager(inp):
        out = torch.nn.functional.relu(inp)
        out = out * 2.0
        out = torch.nn.functional.gelu(out)
        return out + 1.0

    compile_mode = get_compile_mode("reduce-overhead")
    if not should_use_compile():
        print(
            "torch.compile disabled (set USE_COMPILE=1 to enable); showing eager timings only."
        )
        _benchmark("Eager (warm-up)", lambda: eager(x), iters=1)
        _benchmark("Eager (no compile)", lambda: eager(x), iters=10)
        print(f"Re-run with USE_COMPILE=1 COMPILE_MODE={compile_mode} for fusion demo.")
        return

    compiled = maybe_compile(eager, default_mode=compile_mode)
    if compiled is eager:
        print(
            f"torch.compile request failed or was disabled; run with USE_COMPILE=1 "
            f"COMPILE_MODE={compile_mode} for the fused comparison."
        )
        eager_ms = _benchmark("Eager (fallback)", lambda: eager(x), iters=10)
        print(f"Repeat with compilation enabled to observe fusion benefits.")
        return

    compiled(x)  # warm-up

    print("\n=== torch.compile for occupancy (kernel fusion) ===")
    eager_ms = _benchmark("Eager", lambda: eager(x), iters=10)
    compiled_ms = _benchmark(
        f"Compiled ({compile_mode})", lambda: compiled(x), iters=10
    )
    print(f"Speedup (eager / compiled): {eager_ms / compiled_ms:5.2f}x")


def layout_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    conv = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device)

    nchw = torch.randn(32, 128, 64, 64, device=device)
    nhwc = nchw.to(memory_format=torch.channels_last)

    print("\n=== Memory layout impact (channels-last) ===")
    print(f"NCHW contiguous: {nchw.is_contiguous()}")
    print(f"NHWC contiguous (channels_last): {nhwc.is_contiguous(memory_format=torch.channels_last)}")

    _benchmark("Conv2d NCHW", lambda: conv(nchw), iters=20)
    _benchmark("Conv2d NHWC", lambda: conv(nhwc), iters=20)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; nothing to demonstrate")
        return

    props = torch.cuda.get_device_properties(0)
    print("=== PyTorch occupancy heuristics ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SM count: {props.multi_processor_count}, total memory: {props.total_memory / 1e9:.1f} GB")

    matmul_size_demo()
    batching_demo()
    compile_fusion_demo()
    layout_demo()

    print("\nSuggested profiling commands:")
    print("- ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python occupancy_pytorch.py")
    print("- torch.profiler.profile(..., activities=[ProfilerActivity.CUDA])")


if __name__ == "__main__":
    main()
