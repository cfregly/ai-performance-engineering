#!/usr/bin/env python3
"""Probe GPU capabilities dynamically and cache the results as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by portable hosts
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]  # core/scripts/utilities -> core/scripts -> core -> repo_root
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_PATH = ARTIFACTS_DIR / "hardware_capabilities.json"

CUDA_ATTR_CLUSTER_LAUNCH = 164


def _run_command(
    cmd: List[str], *, timeout: int = 60, env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """Run a subprocess command with a sanitized environment."""
    run_env = os.environ.copy()
    # Strip LD_PRELOAD – on aarch64 host images we occasionally inherit an x86
    # NCCL preload value from the container, which causes noisy warnings when
    # probing toolchain support. Removing it keeps the probe output clean while
    # still honouring LD_LIBRARY_PATH for CUDA/NVCC discovery.
    run_env.pop("LD_PRELOAD", None)
    if env:
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
        env=run_env,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _locate_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def _probe_tma_compiler_support(sm_tag: str) -> Tuple[bool, Optional[str]]:
    ptxas = _locate_tool("ptxas")
    if not ptxas:
        return False, "ptxas not found"
    sm_name = sm_tag
    ptx = textwrap.dedent(
        f"""
        .version 9.0
        .target sm_{sm_name}
        .address_size 64

        .visible .entry tma_probe() {{
            .reg .b64 %rd<2>;
            .reg .pred %p<2>;
            .reg .b32 %r<2>;
            .reg .b64 %desc;
            mov.b64 %rd0, 0;
            mov.b32 %r0, 0;
            mov.b32 %r1, 0;
            mov.b64 %desc, %rd0;
            @%p0 tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%desc], %rd0;
            ret;
        }}
        """
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ptx_path = tmp_path / "tma_probe.ptx"
        cubin_path = tmp_path / "tma_probe.cubin"
        ptx_path.write_text(ptx)
        code, _, err = _run_command(
            [ptxas, "--gpu-name", f"sm_{sm_name}", str(ptx_path), "-o", str(cubin_path)]
        )
        if code == 0:
            return True, None
        # Blackwell ptxas requires the architecture-specific 'a' suffix for TMA /
        # tensormap instructions. Retry with the 'a' variant for ANY Blackwell SM
        # (sm_100 -> sm_100a, sm_103 -> sm_103a). Without the sm_103 retry, GB300
        # is wrongly reported TMA-unsupported and every TMA lab skips.
        if sm_tag.startswith("10") and not sm_name.endswith("a"):
            sm_name = f"{sm_tag}a"
            ptx = ptx.replace(f"sm_{sm_tag}", f"sm_{sm_name}")
            ptx_path.write_text(ptx)
            code, _, err = _run_command(
                [ptxas, "--gpu-name", f"sm_{sm_name}", str(ptx_path), "-o", str(cubin_path)]
            )
            if code == 0:
                return True, None
        return False, err or "ptxas failed"


def _compile_and_run(source: str, sm_tag: str, exe_name: str) -> Tuple[int, str, str]:
    nvcc = _locate_tool("nvcc")
    if not nvcc:
        return 1, "", "nvcc not found"
    # Blackwell (SM 10.x) requires the 'a' suffix for cluster/DSMEM features
    arch_tag = f"sm_{sm_tag}a" if sm_tag.startswith("10") else f"sm_{sm_tag}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cu_path = tmp_path / f"{exe_name}.cu"
        exe_path = tmp_path / exe_name
        cu_path.write_text(source)
        cmd = [
            nvcc,
            "-std=c++17",
            "-rdc=true",
            "-O2",
            f"-arch={arch_tag}",
            str(cu_path),
            "-lcuda",
            "-lcudadevrt",
            "-o",
            str(exe_path),
        ]
        code, _, err = _run_command(cmd, timeout=120)
        if code != 0:
            return code, "", err or "nvcc compilation failed"
        return _run_command([str(exe_path)], timeout=60)


def _probe_dsmem_support(sm_tag: str) -> Tuple[bool, Optional[str]]:
    # CUDA 13.0 changed cudaLaunchKernelEx to a template API.
    # Use static shared memory (not extern __shared__) for reliable DSMEM access.
    source = textwrap.dedent(
        """
        #include <cuda_runtime.h>
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        __global__ void dsmem_probe_kernel(int *out) {
            __shared__ int smem[32];  // Static shared memory for reliable DSMEM
            auto cluster = cg::this_cluster();
            unsigned int block_rank = cluster.block_rank();
            if (block_rank == 0 && threadIdx.x == 0) {
                smem[0] = 123;
            }
            cluster.sync();
            if (block_rank == 1 && threadIdx.x == 0) {
                int *remote = cluster.map_shared_rank(smem, 0);
                out[0] = remote[0];  // Should read 123 from block 0's smem
            }
            cluster.sync();
        }

        int main() {
            int *out = nullptr;
            if (cudaMalloc(&out, sizeof(int)) != cudaSuccess) {
                return 3;
            }
            cudaMemset(out, 0, sizeof(int));
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(2);
            config.blockDim = dim3(32);
            config.dynamicSmemBytes = 0;  // Using static shared memory
            config.stream = nullptr;
            // Use attributes API for cluster dimension (works in CUDA 12.x and 13.0)
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim.x = 2;
            attrs[0].val.clusterDim.y = 1;
            attrs[0].val.clusterDim.z = 1;
            config.attrs = attrs;
            config.numAttrs = 1;
            // CUDA 13.0 template API: pass kernel args directly
            cudaError_t launch_err = cudaLaunchKernelEx(&config, dsmem_probe_kernel, out);
            cudaError_t sync_err = cudaDeviceSynchronize();
            int result = 0;
            cudaMemcpy(&result, out, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(out);
            // Verify the kernel actually worked (result should be 123)
            if (launch_err == cudaSuccess && sync_err == cudaSuccess && result == 123) {
                return 0;
            }
            if (launch_err == cudaErrorNotSupported) {
                return 2;
            }
            return 1;
        }
        """
    )
    code, out, err = _compile_and_run(source, sm_tag, "dsmem_probe")
    if code == 0:
        return True, None
    if code == 2:
        return False, "cudaErrorNotSupported"
    if err:
        return False, err
    return False, out or "DSMEM probe failed"


def _query_cluster_launch(device_index: int) -> Optional[int]:
    try:
        import ctypes

        libcudart = ctypes.CDLL("libcudart.so")
        func = libcudart.cudaDeviceGetAttribute
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        value = ctypes.c_int()
        result = func(ctypes.byref(value), ctypes.c_int(CUDA_ATTR_CLUSTER_LAUNCH), ctypes.c_int(device_index))
        if result != 0:
            return None
        return int(value.value)
    except OSError:
        return None


def _is_grace_host() -> bool:
    """True when the host CPU is Grace (ARM), i.e. a coherent Grace-Blackwell node."""
    return platform.machine().lower() in ("aarch64", "arm64")


def _has_grace_coherence(props) -> bool:
    """Grace-Blackwell coherent memory: a Grace (ARM) host with a Blackwell-class
    GPU. GB200/GB300 report CC 10.x; GB10 reports CC 12.x. A non-Grace (x86)
    B200/B300 has no NVLink-C2C coherent fabric."""
    if props.major >= 12:
        return True
    return props.major >= 10 and _is_grace_host()


def _build_features(props) -> List[str]:
    features: List[str] = []
    if props.major >= 10:
        features.append("HBM3e")
        features.append("Stream-ordered memory APIs")
    if _has_grace_coherence(props):
        features.append("NVLink-C2C")
        features.append("Grace-Blackwell coherence fabric")
    return features


def _derive_tma_limits(props) -> Tuple[int, int, int]:
    if props.major >= 10 and props.major < 12:
        return 1024, 128, 128
    if props.major >= 12:
        return 256, 64, 32
    if props.major >= 9:
        return 1024, 128, 128
    return 256, 64, 32


def _probe_device(device_index: int) -> Dict[str, Any]:
    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(device_index)
    sm_tag = f"{props.major}{props.minor}"
    tma_supported = props.major >= 9
    tma_compiler_supported = False
    tma_note = None
    if tma_supported:
        tma_compiler_supported, tma_note = _probe_tma_compiler_support(sm_tag)
        if not tma_compiler_supported:
            # Treat missing compiler/toolchain support as effectively unsupported.
            tma_supported = False
    else:
        tma_note = "Compute capability < 9.0"
    max_1d, max_2d_w, max_2d_h = _derive_tma_limits(props)
    cluster_launch_attr = _query_cluster_launch(device_index)
    cluster_supported = True if cluster_launch_attr is None else bool(cluster_launch_attr > 0)
    cluster_note = None
    if cluster_launch_attr is None:
        cluster_note = "cudaDevAttrClusterLaunch unavailable; assuming support"
    elif not cluster_supported:
        cluster_note = "Driver reports cluster launch disabled"
    dsmem_supported = False
    dsmem_note = None
    if cluster_supported:
        dsmem_supported, dsmem_note = _probe_dsmem_support(sm_tag)
    else:
        dsmem_note = cluster_note or "Cluster launch disabled"
    tensor_core_desc = "5th Gen" if props.major >= 10 else ("4th Gen" if props.major == 9 else "Unknown")
    entry: Dict[str, Any] = {
        "device_index": device_index,
        "name": props.name,
        "key": f"sm_{props.major}{props.minor}",
        "architecture": (
            "grace_blackwell" if props.major >= 12
            else "blackwell_ultra" if (props.major == 10 and props.minor >= 3)
            else "blackwell" if props.major >= 10
            else "hopper" if props.major >= 9
            else "other"
        ),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "num_sms": props.multi_processor_count,
        "warp_size": props.warp_size,
        "max_threads_per_block": getattr(props, "max_threads_per_block", 1024),
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "max_shared_mem_per_block": props.shared_memory_per_block,
        "max_shared_mem_per_sm": props.shared_memory_per_multiprocessor,
        "l2_cache_bytes": getattr(props, "L2_cache_size", 0),
        "features": _build_features(props),
        "tensor_cores": tensor_core_desc,
        "memory_bandwidth_tbps": None,
        "max_unified_memory_tb": 30 if props.major >= 12 else None,
        "nvlink_c2c": _has_grace_coherence(props),
        "grace_coherence": _has_grace_coherence(props),
        "notes": [],
        "tma": {
            "supported": tma_supported,
            "compiler_support": tma_compiler_supported,
            "max_1d": max_1d,
            "max_2d_width": max_2d_w,
            "max_2d_height": max_2d_h,
        },
        "cluster": {
            "supports_clusters": cluster_supported,
            "has_dsmem": dsmem_supported,
            "max_cluster_size": 8 if props.major >= 10 else 4 if props.major >= 9 else 1,
            "notes": cluster_note,
        },
    }
    if tma_note:
        entry["notes"].append(f"TMA probe: {tma_note}")
    if dsmem_note:
        entry["notes"].append(f"DSMEM probe: {dsmem_note}")
    driver_version = None
    if hasattr(torch.cuda, "_get_device_properties"):
        try:
            driver_version = torch._C._cuda_getDriverVersion()
        except Exception:
            driver_version = None
    if driver_version:
        major = driver_version // 1000
        minor = (driver_version % 1000) // 10
        patch = driver_version % 10
        if patch:
            entry["driver_version"] = f"{major}.{minor}.{patch}"
        else:
            entry["driver_version"] = f"{major}.{minor}"
    entry["cuda_runtime_version"] = torch.version.cuda
    return entry


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CUDA hardware capabilities.")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help=(
            "Exit successfully without writing a cache when PyTorch or CUDA is unavailable. "
            "Use this for portable preflight/setup paths; strict B200 runs should omit it."
        ),
    )
    return parser.parse_args(argv)


def _skip_or_raise(reason: str, *, allow_unavailable: bool) -> None:
    if allow_unavailable:
        print(f"SKIPPED: {reason}; hardware capability cache not refreshed.")
        return
    raise RuntimeError(reason)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if torch is None:
        _skip_or_raise("PyTorch is not installed", allow_unavailable=args.allow_unavailable)
        return
    if not torch.cuda.is_available():
        _skip_or_raise("CUDA not available", allow_unavailable=args.allow_unavailable)
        return
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    devices: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        devices.append(_probe_device(idx))
    data = {
        "version": 1,
        "timestamp": time.time(),
        "devices": devices,
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
