"""Lightweight hardware microbenchmarks for quick diagnostics.

IMPORTANT: These routines are diagnostic tools, not comparable benchmarks.
They intentionally bypass the benchmark harness and therefore do NOT include
the 95 validity protections (stream auditing, cache isolation, clock locking,
etc.). Use them to sanity-check a system, not to claim baseline-vs-optimized
speedups.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import typer


_DIAGNOSTIC_META: Dict[str, Any] = {
    "diagnostic_only": True,
    "harness_protections": False,
    "note": (
        "Diagnostic-only microbenchmark (bypasses harness protections). "
        "Do not use for baseline/optimized comparisons."
    ),
}


def _with_meta(result: Dict[str, Any]) -> Dict[str, Any]:
    if "_meta" in result:
        raise RuntimeError("microbench result dict must not already contain '_meta'")
    return {"_meta": dict(_DIAGNOSTIC_META), **result}


def _now() -> float:
    return time.perf_counter()


def _summarize_samples(samples_s: list[float]) -> Dict[str, Any]:
    if not samples_s:
        return {
            "samples_completed": 0,
            "p50_seconds": None,
            "p90_seconds": None,
            "mean_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    ordered = sorted(samples_s)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = int(round(p * (n - 1)))
        return ordered[idx]

    return {
        "samples_completed": n,
        "p50_seconds": _pct(0.50),
        "p90_seconds": _pct(0.90),
        "mean_seconds": float(statistics.mean(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def _gpu_state_payload(before_state: Any, after_state: Any) -> Dict[str, Any]:
    from dataclasses import asdict

    from core.harness.validity_checks import check_gpu_state_consistency

    consistent, warnings_list = check_gpu_state_consistency(before_state, after_state)
    return {
        "gpu_state_before": asdict(before_state),
        "gpu_state_after": asdict(after_state),
        "gpu_state_consistent": bool(consistent),
        "gpu_warnings": warnings_list,
    }


def disk_io_test(
    file_size_mb: int = 256,
    block_size_kb: int = 1024,
    tmp_dir: str | None = None,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """Simple sequential disk write/read benchmark.

    Args:
        file_size_mb: Size of file to write/read.
        block_size_kb: Block size used for write/read.
        tmp_dir: Optional directory for the test file.
        timeout_seconds: Optional max duration; returns partial progress when exceeded.
    """
    tmp_path = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    tmp_path.mkdir(parents=True, exist_ok=True)
    file_path = tmp_path / "microbench_io.bin"

    total_bytes = file_size_mb * 1024 * 1024
    block_bytes = block_size_kb * 1024
    data = os.urandom(block_bytes)
    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False

    # Write
    start = _now()
    with open(file_path, "wb") as f:
        written = 0
        while written < total_bytes:
            f.write(data)
            written += len(data)
            if deadline and _now() > deadline:
                timeout_hit = True
                break
    write_time = _now() - start

    read_time = None
    bytes_read = 0

    # Read (only if we did not time out during writes)
    if not timeout_hit:
        start = _now()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(block_bytes)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if deadline and _now() > deadline:
                    timeout_hit = True
                    break
        read_time = _now() - start

    try:
        file_path.unlink()
    except Exception:
        pass

    return _with_meta({
        "file_size_mb": file_size_mb,
        "block_size_kb": block_size_kb,
        "write_seconds": write_time,
        "write_gbps": (written / write_time) / 1e9 if write_time > 0 else None,
        "read_seconds": read_time,
        "read_gbps": (bytes_read / read_time) / 1e9 if read_time and read_time > 0 else None,
        "bytes_written": written,
        "bytes_read": bytes_read,
        "path": str(tmp_path),
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
    })


def pcie_bandwidth_test(
    size_mb: int = 256,
    iters: int = 10,
    warmup: int = 3,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """Measure H2D and D2H bandwidth using torch CUDA if available."""
    try:
        import torch
    except ImportError as e:
        return _with_meta({"error": f"torch not available: {e}"})

    if not torch.cuda.is_available():
        return _with_meta({"error": "CUDA not available"})

    device = torch.device("cuda")
    device_index = device.index if device.index is not None else 0
    size_bytes = size_mb * 1024 * 1024
    tensor_cpu = torch.empty(size_bytes // 4, dtype=torch.float32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device=device)

    from core.harness.validity_checks import capture_gpu_state

    gpu_state_before = capture_gpu_state(device_index)
    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False

    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        tensor_gpu.copy_(tensor_cpu, non_blocking=True)
    torch.cuda.synchronize()
    # H2D
    start = _now()
    h2d_iters = 0
    for _ in range(iters):
        tensor_gpu.copy_(tensor_cpu, non_blocking=True)
        h2d_iters += 1
        if deadline and _now() > deadline:
            timeout_hit = True
            break
    torch.cuda.synchronize()
    elapsed_h2d = _now() - start
    h2d_time = elapsed_h2d / max(h2d_iters, 1)

    # D2H
    d2h_time = None
    d2h_iters = 0
    if not timeout_hit:
        for _ in range(max(0, warmup)):
            tensor_cpu.copy_(tensor_gpu, non_blocking=True)
        torch.cuda.synchronize()
        start = _now()
        for _ in range(iters):
            tensor_cpu.copy_(tensor_gpu, non_blocking=True)
            d2h_iters += 1
            if deadline and _now() > deadline:
                timeout_hit = True
                break
        torch.cuda.synchronize()
        elapsed_d2h = _now() - start
        d2h_time = elapsed_d2h / max(d2h_iters, 1)

    gpu_state_after = capture_gpu_state(device_index)
    return _with_meta({
        "size_mb": size_mb,
        "iters": iters,
        "warmup": warmup,
        "h2d_completed": h2d_iters,
        "d2h_completed": d2h_iters,
        "h2d_seconds_per_iter": h2d_time,
        "d2h_seconds_per_iter": d2h_time,
        "h2d_gbps": (size_bytes / h2d_time) / 1e9 if h2d_time and h2d_time > 0 else None,
        "d2h_gbps": (size_bytes / d2h_time) / 1e9 if d2h_time and d2h_time > 0 else None,
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
        **_gpu_state_payload(gpu_state_before, gpu_state_after),
    })


def mem_hierarchy_test(
    size_mb: int = 256,
    stride: int = 128,
    iters: int = 10,
    warmup: int = 3,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """Crude stride-based bandwidth test on GPU memory."""
    try:
        import torch
    except ImportError as e:
        return _with_meta({"error": f"torch not available: {e}"})
    if not torch.cuda.is_available():
        return _with_meta({"error": "CUDA not available"})

    device = torch.device("cuda")
    device_index = device.index if device.index is not None else 0
    n = (size_mb * 1024 * 1024) // 4
    x = torch.arange(n, device=device, dtype=torch.float32)
    from core.harness.validity_checks import capture_gpu_state

    gpu_state_before = capture_gpu_state(device_index)

    def _run_once() -> torch.Tensor:
        # stride access; clone ensures real reads happen
        return x[::stride].clone()

    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        y = _run_once()
    torch.cuda.synchronize()

    samples_s: list[float] = []
    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False
    y = None
    for _ in range(max(0, iters)):
        start = _now()
        y = _run_once()
        torch.cuda.synchronize()
        samples_s.append(_now() - start)
        if deadline and _now() > deadline:
            timeout_hit = True
            break

    summary = _summarize_samples(samples_s)
    elapsed = summary["p50_seconds"] if summary["p50_seconds"] is not None else None
    y_final = y if y is not None else _run_once()
    bytes_moved = y_final.numel() * 4
    gpu_state_after = capture_gpu_state(device_index)
    return _with_meta({
        "size_mb": size_mb,
        "stride": stride,
        "iters": iters,
        "warmup": warmup,
        "bandwidth_gbps": (bytes_moved / elapsed) / 1e9 if elapsed and elapsed > 0 else None,
        "elements": y_final.numel(),
        "timing": summary,
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
        **_gpu_state_payload(gpu_state_before, gpu_state_after),
    })


def _prepare_fp8_matmul(size: int, device):
    """Prepare real E4M3 operands and the PyTorch scaled-mm operation.

    Sampling float8 directly and ordinary @ are not supported FP8 GEMM APIs.
    Unit scales preserve the quantized operand values; no different precision
    is substituted if the actual scaled-mm backend rejects this device.
    """
    import torch
    if not hasattr(torch, "float8_e4m3fn") or not callable(getattr(torch, "_scaled_mm", None)):
        raise RuntimeError("FP8 requires torch.float8_e4m3fn and torch._scaled_mm")
    if size <= 0 or size % 16:
        raise ValueError("FP8 scaled-mm size must be a positive multiple of 16")
    a = torch.randn((size, size), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((size, size), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    scale_a = torch.ones((), device=device, dtype=torch.float32)
    scale_b = torch.ones((), device=device, dtype=torch.float32)

    def run():
        return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=False)

    return a, b, run


def tensor_core_bench(
    size: int = 4096,
    precision: str = "fp16",
    iters: int = 10,
    warmup: int = 3,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """Matmul throughput benchmark to stress tensor cores."""
    try:
        import torch
    except ImportError as e:
        return _with_meta({"error": f"torch not available: {e}"})
    if not torch.cuda.is_available():
        return _with_meta({"error": "CUDA not available"})

    precision_lower = precision.lower()
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
        "fp32": torch.float32,
    }
    if precision_lower == "fp8":
        if not hasattr(torch, "float8_e4m3fn") or not callable(getattr(torch, "_scaled_mm", None)):
            return _with_meta({"error": "FP8 requires float8_e4m3fn and torch._scaled_mm; no precision fallback"})
        dtype_map["fp8"] = torch.float8_e4m3fn
    elif precision_lower == "int8":
        return _with_meta({"error": "INT8 matmul not supported in this minimal microbench; use fp16/bf16/tf32"})

    dtype = dtype_map.get(precision_lower)
    if dtype is None:
        return _with_meta({"error": f"unsupported precision: {precision}"})

    device = torch.device("cuda")
    device_index = device.index if device.index is not None else 0
    from core.harness.validity_checks import capture_gpu_state

    gpu_state_before = capture_gpu_state(device_index)
    if precision_lower == "fp8":
        try:
            a, b, run_matmul = _prepare_fp8_matmul(size, device)
            # Establish actual backend support before reporting or timing FP8.
            run_matmul()
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            return _with_meta({"error": f"FP8 scaled-mm unavailable/failed: {exc}", "precision": precision})
    else:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)

        def run_matmul():
            return a @ b
    torch.cuda.synchronize()

    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False

    for _ in range(max(0, warmup)):
        c = run_matmul()
    torch.cuda.synchronize()

    samples_s: list[float] = []
    c = None
    for _ in range(max(0, iters)):
        start = _now()
        c = run_matmul()
        torch.cuda.synchronize()
        samples_s.append(_now() - start)
        if deadline and _now() > deadline:
            timeout_hit = True
            break

    summary = _summarize_samples(samples_s)
    elapsed = summary["p50_seconds"] if summary["p50_seconds"] is not None else None
    flops = 2 * (size ** 3)
    tflops = (flops / elapsed) / 1e12 if elapsed and elapsed > 0 else None
    gpu_state_after = capture_gpu_state(device_index)

    c_final = c if c is not None else run_matmul()
    return _with_meta({
        "size": size,
        "precision": precision,
        "tflops": tflops,
        "elapsed_seconds": elapsed,
        "iters": iters,
        "warmup": warmup,
        "timing": summary,
        "output_shape": list(c_final.shape),
        "placeholder_used": False,
        "operand_dtype": str(a.dtype),
        "output_dtype": str(c_final.dtype),
        "matmul_api": "torch._scaled_mm" if precision_lower == "fp8" else "torch.matmul",
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
        **_gpu_state_payload(gpu_state_before, gpu_state_after),
    })


def sfu_bench(
    size: int = 64 * 1024 * 1024,
    iters: int = 10,
    warmup: int = 3,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """SFU-heavy benchmark using sin/cos operations."""
    try:
        import torch
    except ImportError as e:
        return _with_meta({"error": f"torch not available: {e}"})
    if not torch.cuda.is_available():
        return _with_meta({"error": "CUDA not available"})

    device = torch.device("cuda")
    device_index = device.index if device.index is not None else 0
    from core.harness.validity_checks import capture_gpu_state

    gpu_state_before = capture_gpu_state(device_index)
    x = torch.linspace(0, 10, steps=size, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False

    for _ in range(max(0, warmup)):
        y = torch.sin(x) + torch.cos(x)
    torch.cuda.synchronize()

    samples_s: list[float] = []
    y = None
    for _ in range(max(0, iters)):
        start = _now()
        y = torch.sin(x) + torch.cos(x)
        torch.cuda.synchronize()
        samples_s.append(_now() - start)
        if deadline and _now() > deadline:
            timeout_hit = True
            break

    summary = _summarize_samples(samples_s)
    elapsed = summary["p50_seconds"] if summary["p50_seconds"] is not None else None
    ops = size * 4  # approx operations per element
    gops = (ops / elapsed) / 1e9 if elapsed and elapsed > 0 else None
    gpu_state_after = capture_gpu_state(device_index)

    y_final = y if y is not None else (torch.sin(x) + torch.cos(x))
    return _with_meta({
        "elements": size,
        "elapsed_seconds": elapsed,
        "gops": gops,
        "iters": iters,
        "warmup": warmup,
        "timing": summary,
        "result_sample": float(y_final[0].item()) if y_final.numel() > 0 else None,
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
        **_gpu_state_payload(gpu_state_before, gpu_state_after),
    })


def network_loopback_test(size_mb: int = 64, port: int = 50007, timeout_seconds: float | None = None) -> Dict[str, Any]:
    """Simple loopback TCP throughput test (localhost)."""
    total_bytes = size_mb * 1024 * 1024
    payload = b"x" * 65536
    deadline = _now() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timeout_hit = False

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
    except PermissionError as exc:
        return _with_meta({
            "size_mb": size_mb,
            "elapsed_seconds": None,
            "bytes_sent": 0,
            "throughput_gbps": None,
            "error": f"Loopback sockets not permitted: {exc}",
            "notes": "Loopback TCP only; use iperf for real NIC tests",
            "timeout_seconds": timeout_seconds,
            "timeout_hit": False,
        })

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        conn, _ = srv.accept()
        received = 0
        while received < total_bytes:
            data = conn.recv(len(payload))
            if not data:
                break
            received += len(data)
        conn.close()
        srv.close()

    import threading
    t = threading.Thread(target=server, daemon=True)
    t.start()
    time.sleep(0.1)

    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start = _now()
    cli.connect(("127.0.0.1", port))
    sent = 0
    while sent < total_bytes:
        cli.sendall(payload)
        sent += len(payload)
        if deadline and _now() > deadline:
            timeout_hit = True
            break
    cli.shutdown(socket.SHUT_WR)
    cli.close()
    t.join()
    elapsed = _now() - start
    return _with_meta({
        "size_mb": size_mb,
        "elapsed_seconds": elapsed,
        "bytes_sent": sent,
        "throughput_gbps": (sent / elapsed) / 1e9 if elapsed > 0 else None,
        "notes": "Loopback TCP only; use iperf for real NIC tests",
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
    })


def _print(res: Dict[str, Any], json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps(res, indent=2, default=str))
    else:
        note = (res.get("_meta") or {}).get("note")
        if note:
            typer.secho(f"WARNING: {note}", fg=typer.colors.YELLOW, err=True)
        typer.echo(res)

# CLI/aisp wrappers (SimpleNamespace args)
def disk(args: Any) -> int:
    res = disk_io_test(
        file_size_mb=getattr(args, "file_size_mb", getattr(args, "size_mb", 256)),
        block_size_kb=getattr(args, "block_size_kb", getattr(args, "block_kb", 1024)),
        tmp_dir=getattr(args, "tmp_dir", None),
    )
    _print(res, getattr(args, "json", False))
    return 0


def pcie(args: Any) -> int:
    res = pcie_bandwidth_test(
        size_mb=getattr(args, "size_mb", 256),
        iters=getattr(args, "iters", 10),
    )
    _print(res, getattr(args, "json", False))
    return 0


def mem_hierarchy(args: Any) -> int:
    res = mem_hierarchy_test(
        size_mb=getattr(args, "size_mb", 256),
        stride=getattr(args, "stride", 128),
    )
    _print(res, getattr(args, "json", False))
    return 0


def tensor_core(args: Any) -> int:
    res = tensor_core_bench(
        size=getattr(args, "size", 4096),
        precision=getattr(args, "precision", "fp16"),
    )
    _print(res, getattr(args, "json", False))
    return 0


def sfu(args: Any) -> int:
    res = sfu_bench(size=getattr(args, "elements", 64 * 1024 * 1024))
    _print(res, getattr(args, "json", False))
    return 0


def loopback(args: Any) -> int:
    res = network_loopback_test(
        size_mb=getattr(args, "size_mb", 256),
        port=getattr(args, "port", 5789),
    )
    _print(res, getattr(args, "json", False))
    return 0


app = typer.Typer(help="Lightweight microbenchmarks (disk, PCIe, memory, tensor core, SFU, loopback).")


@app.command("disk")
def cli_disk(
    file_size_mb: int = typer.Option(256, "--file-size-mb", "-s", help="Size of the temp file to write/read (MB)"),
    block_size_kb: int = typer.Option(1024, "--block-size-kb", "-b", help="Block size for IO (KB)"),
    tmp_dir: Optional[str] = typer.Option(None, "--tmp-dir", "-t", help="Optional directory for the temp file"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(disk_io_test(file_size_mb=file_size_mb, block_size_kb=block_size_kb, tmp_dir=tmp_dir), json_out)


@app.command("pcie")
def cli_pcie(
    size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
    iters: int = typer.Option(10, "--iters", "-i", help="Number of repetitions"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(pcie_bandwidth_test(size_mb=size_mb, iters=iters), json_out)


@app.command("mem")
def cli_mem(
    size_mb: int = typer.Option(256, "--size-mb", "-s", help="Tensor size (MB)"),
    stride: int = typer.Option(128, "--stride", help="Stride for access pattern"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(mem_hierarchy_test(size_mb=size_mb, stride=stride), json_out)


@app.command("tensor")
def cli_tensor(
    size: int = typer.Option(4096, "--size", "-s", help="Matrix size (N x N)"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="Precision: fp16/bf16/tf32/fp32/fp8"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(tensor_core_bench(size=size, precision=precision), json_out)


@app.command("sfu")
def cli_sfu(
    elements: int = typer.Option(64 * 1024 * 1024, "--elements", "-n", help="Number of elements"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(sfu_bench(size=elements), json_out)


@app.command("loopback")
def cli_loopback(
    size_mb: int = typer.Option(64, "--size-mb", "-s", help="Transfer size in MB"),
    port: int = typer.Option(50007, "--port", "-p", help="Port to use"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(network_loopback_test(size_mb=size_mb, port=port), json_out)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
