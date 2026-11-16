"""Lightweight tests used by `aisp test ...`."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing in docs builds
    torch = None  # type: ignore

from core.diagnostics import microbench


def _print(payload: Dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def gpu_bandwidth(args: Any) -> int:
    """Quick PCIe bandwidth probe."""
    res = microbench.pcie_bandwidth_test(size_mb=16, iters=3)
    _print(res, getattr(args, "json", False))
    return 0


def network_test(args: Any) -> int:
    """Loopback throughput sanity check."""
    res = microbench.network_loopback_test(size_mb=8, port=5789)
    _print(res, getattr(args, "json", False))
    return 0


def warmup_audit(args: Any, script: str | None = None, iterations: int = 10) -> int:
    """Show first-iter vs steady-state timing for a tiny matmul."""
    if torch is None or not torch.cuda.is_available():
        print("CUDA + torch required for warmup audit.")
        return 1
    steps = getattr(args, "iterations", iterations) or iterations
    size = 512
    x = torch.randn(size, size, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = x @ x  # noqa: F841
    torch.cuda.synchronize()
    first = time.perf_counter() - t0

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(max(steps - 1, 1)):
        y = x @ x  # noqa: F841
    torch.cuda.synchronize()
    steady = (time.perf_counter() - t1) / max(steps - 1, 1)

    payload = {
        "first_iter_ms": first * 1000,
        "steady_iter_ms": steady * 1000,
        "jit_overhead_ms": max(first - steady, 0) * 1000,
        "iterations": steps,
    }
    _print(payload, getattr(args, "json", False))
    return 0


def speedtest(args: Any) -> int:
    """Micro speed tests across GEMM/memory depending on selection."""
    choice = getattr(args, "type", "all")
    json_out = getattr(args, "json", False)
    gemm_size = getattr(args, "gemm_size", 512)
    precision = getattr(args, "precision", "fp16")
    mem_size_mb = getattr(args, "mem_size_mb", 16)
    mem_stride = getattr(args, "mem_stride", 128)
    results: List[Dict[str, Any]] = []

    if choice in ("all", "gemm"):
        res = microbench.tensor_core_bench(size=gemm_size, precision=precision)
        res["name"] = "tensor_core_gemm"
        results.append(res)
    if choice in ("all", "memory"):
        res = microbench.mem_hierarchy_test(size_mb=mem_size_mb, stride=mem_stride)
        res["name"] = "mem_hierarchy"
        results.append(res)
    if choice in ("all", "attention"):
        # Approximate attention by softmax on a small matrix
        attn = {"name": "attention_softmax"}
        if torch is None or not torch.cuda.is_available():
            attn["error"] = "CUDA not available"
        else:
            q = torch.randn(256, 256, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.softmax(q, dim=-1)
            torch.cuda.synchronize()
            attn["latency_ms"] = (time.perf_counter() - start) * 1000
        results.append(attn)

    payload = {"tests": results}
    if json_out:
        _print(payload, json_out)
    else:
        print("Speed tests:")
        for r in results:
            name = r.get("name", "unknown")
            if "tflops" in r and r.get("tflops") is not None:
                size = r.get("size", "n/a")
                precision = r.get("precision", "n/a")
                print(f"  - {name}: {r['tflops']:.4f} TFLOP/s @ size={size}, precision={precision}")
            elif "bandwidth_gbps" in r and r.get("bandwidth_gbps") is not None:
                size_mb = r.get("size_mb", "n/a")
                stride = r.get("stride", "n/a")
                print(f"  - {name}: {r['bandwidth_gbps']:.4f} GB/s @ size={size_mb} MB, stride={stride}")
            elif "latency_ms" in r and r.get("latency_ms") is not None:
                print(f"  - {name}: {r['latency_ms']:.3f} ms")
            else:
                print(f"  - {name}: {r.get('error', 'no data')}")
        print("Use --json for machine-readable output.")
    return 0


def diagnostics(args: Any) -> int:
    """Bundle of quick probes for automation."""
    summary = {
        "pcie": microbench.pcie_bandwidth_test(size_mb=getattr(args, "mem_size_mb", 8), iters=2),
        "memory": microbench.mem_hierarchy_test(
            size_mb=getattr(args, "mem_size_mb", 8), stride=getattr(args, "mem_stride", 256)
        ),
        "tensor_core": microbench.tensor_core_bench(
            size=getattr(args, "gemm_size", 256), precision=getattr(args, "precision", "fp16")
        ),
    }
    _print(summary, getattr(args, "json", False))
    return 0


def mem_roofline(args: Any) -> int:
    """Stride sweep across memory hierarchy and show ASCII roofline-style table."""
    size_mb = getattr(args, "size_mb", 32)
    strides = getattr(args, "strides", None) or [32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    for stride in strides:
        res = microbench.mem_hierarchy_test(size_mb=size_mb, stride=stride)
        rows.append((stride, res.get("bandwidth_gbps") or 0.0))
    if getattr(args, "json", False):
        _print({"size_mb": size_mb, "rows": rows}, True)
        return 0

    header = f"Stride sweep ({size_mb} MB)"
    print(header)
    print("-" * len(header))
    max_bw = max([bw for _, bw in rows] + [1e-9])
    for stride, bw in rows:
        bar_len = int((bw / max_bw) * 40) if max_bw > 0 else 0
        bar = "█" * max(1, bar_len) if bw > 0 else ""
        print(f"{stride:>6} bytes : {bw:>8.3f} GB/s | {bar}")
    return 0
