"""Optional Triton matmul runner to compare against CUTLASS profiler output."""
# ruff: noqa: N803

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import arch_config  # noqa: F401  # triggers triton_compat patch

from .shapes import GemmShape, transformer_gemm_shapes

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pylint: disable=broad-except
    TRITON_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "cutlass_profiler"


def _dtype_from_str(dtype: str) -> torch.dtype:
    mapping = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype}")
    return mapping[dtype]


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Simple block matmul kernel using Triton language (no triton.ops dependency)."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_block_ptr = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_block_ptr = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_remaining = K
    a_ptrs = a_block_ptr
    b_ptrs = b_block_ptr
    while k_remaining > 0:
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    c = acc
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def benchmark_triton_matmul(
    shape: GemmShape, warmup: int, iters: int, dtype: torch.dtype
) -> dict[str, float] | None:
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        return None

    device = torch.device("cuda")
    a = torch.randn((shape.m, shape.k), device=device, dtype=dtype)
    b = torch.randn((shape.k, shape.n), device=device, dtype=dtype)
    c = torch.empty((shape.m, shape.n), device=device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(shape.m, meta["BLOCK_M"]),
            triton.cdiv(shape.n, meta["BLOCK_N"]),
        )

    # Warmup/JIT compile
    for _ in range(warmup):
        _matmul_kernel[grid](
            a,
            b,
            c,
            shape.m,
            shape.n,
            shape.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    for _ in range(iters):
        start.record(current_stream)
        _matmul_kernel[grid](
            a,
            b,
            c,
            shape.m,
            shape.n,
            shape.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )
        end.record(current_stream)
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    if not times_ms:
        return None

    best_ms = min(times_ms)
    flops = 2.0 * shape.m * shape.n * shape.k
    tflops = flops / (best_ms * 1e-3) / 1e12
    return {"runtime_ms": best_ms, "tflops": tflops}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Triton matmul on transformer-ish shapes.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Where to store JSON results.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for Triton matmul.")
    parser.add_argument("--iters", type=int, default=10, help="Measured iterations.")
    parser.add_argument("--shapes", type=str, nargs="*", default=None, help="Subset of shape names to run.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping Triton matmul.", file=sys.stderr)
        return 1
    if not TRITON_AVAILABLE:
        print("Triton is not installed (or triton.language unavailable); install triton to run this comparison.", file=sys.stderr)
        return 1

    shapes = transformer_gemm_shapes()
    if args.shapes:
        name_set = set(args.shapes)
        shapes = [s for s in shapes if s.name in name_set]
    if not shapes:
        print("No shapes selected.", file=sys.stderr)
        return 1

    dtype = _dtype_from_str(shapes[0].dtype)
    results = []
    failed = []

    for shape in shapes:
        print(f"→ Triton matmul {shape.name} (m={shape.m}, n={shape.n}, k={shape.k})")
        try:
            metrics = benchmark_triton_matmul(shape, args.warmup, args.iters, dtype)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"   FAILED: {exc}", file=sys.stderr)
            failed.append(shape.name)
            continue

        if metrics is None:
            print("   No metrics produced.", file=sys.stderr)
            failed.append(shape.name)
            continue

        print(f"   runtime={metrics['runtime_ms']:.3f} ms, throughput={metrics['tflops']:.2f} TFLOP/s")
        results.append(
            {
                **shape.as_dict(),
                **metrics,
                "provider": "triton",
                "kernel": "triton.ops.matmul",
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "triton_matmul_results.json"
    with output_path.open("w") as f:
        json.dump({"provider": "triton", "results": results, "failed": failed}, f, indent=2)

    print(f"\nTriton results written to {output_path}")
    if failed:
        print(f"{len(failed)} shape(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
