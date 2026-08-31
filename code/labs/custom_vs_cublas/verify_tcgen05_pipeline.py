"""Full-output GPU correctness checks for the W1-002/W1-026 stage pipelines.

Run serially on an explicitly assigned SM100/SM103 GPU. This is correctness
validation, not a timing benchmark. A CPU invocation fails without claiming a
pass. --list-cases only prints the planned workload and does not initialize CUDA.

For allocation-boundary checking, run from code/ under compute-sanitizer with
PYTORCH_NO_CUDA_MEMORY_CACHING=1 and --error-exitcode=1. Keep sanitizer output
alongside the JSON receipt; this script does not claim sanitizer qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = {
    "cluster": ("tcgen05_cluster.cu", "matmul_tcgen05_cluster"),
    "no_wait": ("tcgen05_no_wait.cu", "matmul_tcgen05_no_wait"),
    "no_wait_swizzle": ("tcgen05_no_wait_swizzle.cu", "matmul_tcgen05_no_wait_swizzle"),
    "warp_parallel": ("tcgen05_warp_parallel.cu", "matmul_tcgen05_warp_parallel"),
    "tma_multicast": ("experimental/tcgen05_tma_multicast.cu", "matmul_tcgen05_tma_multicast"),
    "no_wait_5stage": ("experimental/tcgen05_no_wait_5stage.cu", "matmul_tcgen05_no_wait_5stage"),
    "no_mma_barrier": ("experimental/tcgen05_no_mma_barrier.cu", "matmul_tcgen05_no_mma_barrier"),
}

# Include initial fills, first reuse, multiple parity wraps, and long K loops.
# M=128/384 have odd logical M-tile counts and need a padded cluster CTA.
K_TILE_COUNTS = (1, 2, 3, 4, 5, 8, 9, 17, 33, 129)
SHAPES = tuple((m, 256, 64 * tiles) for m in (128, 256, 384) for tiles in K_TILE_COUNTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1009])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.list_cases:
        print(json.dumps({"status": "planned_only", "variants": args.variants,
                          "shapes_m_n_k": SHAPES, "seeds": args.seeds,
                          "repeats": args.repeats}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for an execution receipt")
    if args.output.exists():
        parser.error("--output must be a new path; preserve prior attempts")

    import torch
    from core.benchmark.verification import get_tolerance_for_dtype
    from labs.custom_vs_cublas.tcgen05_loader import _load_kernel

    lab_dir = Path(__file__).resolve().parent
    source_paths = [lab_dir / VARIANTS[name][0] for name in args.variants]
    receipt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete", "scope": "GPU full-output correctness only",
        "sanitizer_qualified": False,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "source_sha256": {str(path.relative_to(lab_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "requested": {"variants": args.variants, "shapes_m_n_k": SHAPES,
                      "seeds": args.seeds, "repeats": args.repeats},
        "modules": [], "checks": [],
    }
    current_case = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Real CUDA device required; no CPU substitute is valid")
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(f"This regression requires SM100/SM103, got {capability}")
        receipt["device"] = {"index": device, "name": torch.cuda.get_device_name(device),
                             "capability": capability}
        receipt["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=lab_dir, text=True
        ).strip()
        tolerance = get_tolerance_for_dtype(torch.float16)
        receipt["tolerance"] = {"rtol": tolerance.rtol, "atol": tolerance.atol,
                                "source": "get_tolerance_for_dtype(torch.float16)"}
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for variant in args.variants:
                source, symbol = VARIANTS[variant]
                current_case = {"variant": variant, "stage": "compile"}
                # Keep architecture and runtime identity in the module namespace
                # as well as the loader's source hash; don't reuse another cell's build.
                runtime_tag = re.sub(r"[^a-zA-Z0-9_]", "_", f"{capability}_{torch.__version__}_{torch.version.cuda}")
                module = _load_kernel(lab_dir / source, f"audit_pipeline_{variant}_{runtime_tag}")
                module_path = Path(module.__file__)
                receipt["modules"].append({"variant": variant, "path": str(module_path),
                                           "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()})
                kernel = getattr(module, symbol)
                for m, n, k in SHAPES:
                    for seed in args.seeds:
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator) / math.sqrt(k)
                        b = torch.randn((n, k), device="cuda", dtype=torch.float16, generator=generator)
                        original_a, original_b = a.clone(), b.clone()
                        expected = (a.float() @ b.float().T).half()
                        # This guard is supplementary. compute-sanitizer with
                        # uncached allocations is the allocation-boundary gate.
                        guard = torch.full((32768,), 0x5A5A, dtype=torch.int32, device="cuda")
                        for repeat in range(args.repeats):
                            current_case = {"variant": variant, "m": m, "n": n, "k": k,
                                            "seed": seed, "repeat": repeat}
                            actual = kernel(a, b)
                            torch.cuda.synchronize(device)
                            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                                raise AssertionError(f"Output contract mismatch: {actual.shape}, {actual.dtype}")
                            if not bool(torch.isfinite(actual).all()):
                                raise AssertionError("Nonfinite output")
                            torch.testing.assert_close(actual, expected, rtol=tolerance.rtol, atol=tolerance.atol)
                            torch.testing.assert_close(a, original_a, rtol=0, atol=0)
                            torch.testing.assert_close(b, original_b, rtol=0, atol=0)
                            if not bool((guard == 0x5A5A).all()):
                                raise AssertionError("Supplementary allocation guard changed")
                            receipt["checks"].append({**current_case, "status": "pass",
                                                      "compared_elements": m * n,
                                                      "max_abs_error": float((actual.float() - expected.float()).abs().max())})
                        print(f"checked {variant} M={m} N={n} K={k} seed={seed}", flush=True)
            for path in source_paths:
                relative_path = str(path.relative_to(lab_dir))
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["source_sha256"][relative_path]:
                    raise RuntimeError(f"Kernel source changed during verification: {relative_path}")
            receipt["status"] = "passed_full_output_checks"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"case": current_case, "type": type(exc).__name__, "message": str(exc)}
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output_file:
            output_file.write(json.dumps(receipt, indent=2) + "\n")
    print(f"{receipt['status']}: {args.output}")
    return 0 if receipt["status"] == "passed_full_output_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
