"""Complete prefill/decode payload controls and separately gated real CUDA runs.

CPU tests exercise tensor helpers, payload construction and validation only.
They do not run or simulate CUDA-only setup, TMA, copy extensions or graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from labs.persistent_decode.persistent_decode_common import (
    DecodeInputs,
    DecodeOptions,
    build_decode_input_signature,
    build_prefill_decode_verification_buffers,
    get_decode_options,
    set_decode_options,
)

VARIANTS = {
    "baseline_tma": ("baseline_tma_prefill_decode", "BaselineTmaPrefillDecodeBenchmark"),
    "optimized_tma": ("optimized_tma_prefill_decode", "OptimizedTmaPrefillDecodeBenchmark"),
    "baseline_native": ("baseline_native_tma_prefill_decode", "BaselineNativeTmaPrefillDecodeBenchmark"),
    "optimized_native": ("optimized_native_tma_prefill_decode", "OptimizedNativeTmaPrefillDecodeBenchmark"),
}
CUDA_CASES = [
    ("baseline_tma", "none", 32),
    ("baseline_native", "none", 32),
    ("optimized_native", "none", 32),
    ("optimized_tma", "full", 32),
    ("optimized_tma", "piecewise", 32),
    ("optimized_tma", "full_and_piecewise", 1),
    ("optimized_tma", "full_and_piecewise", 32),
]


def _benchmark_class(variant):
    module, name = VARIANTS[variant]
    return getattr(importlib.import_module("labs.persistent_decode." + module), name)


def _payload_control(variant, dtype):
    # The real constructors require CUDA and may allocate streams. Bypass only
    # that unrelated setup to exercise actual CPU-capable payload/tensor methods.
    bench = object.__new__(_benchmark_class(variant))
    bench.batch, bench.seq_len, bench.head_dim = 3, 11, 16
    indices = torch.arange(3 * 11 * 16).reshape(3, 11, 16)
    q = ((indices % 7 + 1) / 8).to(dtype)
    k = ((indices % 5 + 1) / 4).to(dtype)
    v = ((indices % 11 + 1) / 8).to(dtype)
    bench.inputs = DecodeInputs(q, k, v, torch.empty_like(q), torch.arange(3),
                                torch.full((3,), 11), torch.zeros(1))
    bench.prefill_src = torch.arange(34, dtype=torch.float32).reshape(2, 17) / 8
    bench.prefill_dst = bench.prefill_src.clone()  # Payload data, not GPU-copy evidence.
    bench._prefill_work = tuple(zip(bench.prefill_src.unbind(0), bench.prefill_dst.unbind(0), strict=True))
    bench._product_buffer = torch.empty(3, 16, dtype=dtype)
    bench._dot_buffer = torch.empty(3, 1, dtype=dtype)
    bench._decode_step_views = tuple(zip(q.unbind(1), k.unbind(1), v.unbind(1),
                                         bench.inputs.out.unbind(1), strict=True))
    if hasattr(bench, "_decode_host_loop"):
        bench._decode_host_loop()
    else:
        bench._decode_body(q, k, v, bench.inputs.out)
    bench.output = bench._output_view = bench.inputs.out
    (bench._verify_output_buffer, bench._verify_decode_view,
     bench._verify_prefill_view) = build_prefill_decode_verification_buffers(bench.inputs, bench.prefill_dst)
    return bench


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_capture_includes_complete_decode_and_prefill_with_reused_storage(variant, dtype):
    bench = _payload_control(variant, dtype)
    data = bench.inputs
    reference = ((data.q.double() * data.k.double()).sum(-1, keepdim=True) * data.v.double()).to(dtype)
    pointer = bench._verify_output_buffer.data_ptr()
    for _ in range(2):
        bench.capture_verification_payload()
        payload = bench.get_verify_output()
        split = data.out.numel()
        assert payload.shape == (split + bench.prefill_dst.numel(),)
        torch.testing.assert_close(payload[:split].view_as(data.out), reference.float(), rtol=0, atol=0)
        torch.testing.assert_close(payload[split:].view_as(bench.prefill_src), bench.prefill_src, rtol=0, atol=0)
        assert bench._verify_output_buffer.data_ptr() == pointer
        assert payload.data_ptr() != pointer
        assert bench.validate_result() is None
        assert set(bench.get_verify_inputs()) == {"q", "k", "v", "prefill_src"}
        signature = bench.get_input_signature()
        assert tuple(signature.shapes["output"]) == tuple(payload.shape)
        assert tuple(signature.shapes["prefill_src"]) == (2, 17)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fault", ["decode_tail", "prefill_tail", "decode_nan", "prefill_nan", "prefill_alias"])
def test_capture_and_validation_reject_corruption_outside_the_old_slice(variant, fault):
    bench = _payload_control(variant, torch.float32)
    old_slice = bench.inputs.out[:1, :8].clone()
    if fault == "decode_tail":
        bench.inputs.out[-1, -1, -1] += 1024
    elif fault == "prefill_tail":
        bench.prefill_dst[-1, -1] += 1024
    elif fault == "decode_nan":
        bench.inputs.out[-1, -1, -1] = float("nan")
    elif fault == "prefill_nan":
        bench.prefill_dst[-1, -1] = float("nan")
    else:
        bench.prefill_dst = bench.prefill_src
    # The former payload omitted both the last decode element and all prefill.
    torch.testing.assert_close(old_slice, bench.inputs.out[:1, :8], rtol=0, atol=0)
    assert bench.validate_result() is not None
    with pytest.raises(AssertionError):
        bench.capture_verification_payload()


def test_baseline_prefill_is_the_same_copy_only_workload_as_its_peer():
    bench = _payload_control("baseline_tma", torch.float32)
    bench.prefill_dst.fill_(float("nan"))
    bench._prefill_sequential()  # Actual production CPU tensor operations.
    torch.testing.assert_close(bench.prefill_dst, bench.prefill_src, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 11, 16), (8, 32, 64), (12, 64, 64)])
def test_static_decode_signature_describes_every_output_element(shape):
    batch, seq_len, head_dim = shape
    signature = build_decode_input_signature(batch=batch, seq_len=seq_len,
                                             head_dim=head_dim, quantization="fp32")
    assert tuple(signature["shapes"]["output"]) == shape


def _cuda_unavailable_reason():
    if not torch.cuda.is_available():
        return "Actual CUDA device required; CPU payload checks are not GPU qualification"
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        return "These existing extensions target sm_100/103/120/121"
    if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) < 13:
        return "Descriptor-copy extension requires the actual CUDA 13+ build"
    if shutil.which("nvcc") is None:
        return "nvcc is required to build the actual copy extensions"
    return None


def _cuda_worker(variant, mode, capture_limit, output, report):
    """Run real setup/extensions/graphs; called only in a bounded child."""
    reason = _cuda_unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    previous = get_decode_options()
    bench = None
    records = report["checks"]
    try:
        set_decode_options(DecodeOptions(tier="small", quantization="fp32"))
        cls = _benchmark_class(variant)
        if variant == "optimized_tma":
            from labs.persistent_decode.optimized_tma_prefill_decode import GraphMode
            bench = cls(graph_mode=GraphMode(mode), max_capture_seq=capture_limit)
        else:
            bench = cls()
        bench.prefill_chunks, bench.prefill_chunk_elems = 2, 256
        bench.setup()
        caller = torch.cuda.Stream()
        shape = bench.inputs.q.shape
        positions = torch.arange(bench.inputs.q.numel(), device=bench.device).reshape(shape)
        prefill_positions = torch.arange(bench.prefill_src.numel(), device=bench.device).reshape_as(bench.prefill_src)
        caller.wait_stream(torch.cuda.current_stream())
        for iteration in range(3):
            with torch.cuda.stream(caller):
                # Real device work deliberately delays the caller's writes.
                # No host/device synchronization is inserted before benchmark_fn.
                torch.cuda._sleep(2_000_000)
                bench.inputs.q.copy_((positions % 7 + 1) / 8 + iteration / 8)
                bench.inputs.k.copy_((positions % 5 + 1) / 4 + iteration / 4)
                bench.inputs.v.copy_((positions % 11 + 1) / 8 + iteration / 8)
                bench.prefill_src.copy_(prefill_positions / 8 + iteration)
                bench.inputs.out.fill_(float("nan"))
                bench.prefill_dst.fill_(float("nan"))
                bench.benchmark_fn()
                # Preserve complete actual tensors before a validator can fail.
                # These reads occur after the benchmark, never before its launch.
                actual_decode = bench.inputs.out.detach().cpu()
                actual_prefill = bench.prefill_dst.detach().cpu()
                q, k, v = (tensor.detach().cpu().double() for tensor in
                           (bench.inputs.q, bench.inputs.k, bench.inputs.v))
                expected_decode = ((q * k).sum(-1, keepdim=True) * v).float()
                expected_prefill = bench.prefill_src.detach().cpu()
                torch.save({"actual_decode": actual_decode, "actual_prefill": actual_prefill,
                            "decode_reference": expected_decode, "prefill_reference": expected_prefill},
                           output / f"iteration-{iteration}.pt")
                bench.capture_verification_payload()
                actual = bench.get_verify_output().cpu()
                torch.save(actual, output / f"payload-{iteration}.pt")
                split = expected_decode.numel()
                torch.testing.assert_close(actual[:split].view_as(expected_decode), expected_decode, rtol=0, atol=0)
                torch.testing.assert_close(actual[split:].view_as(expected_prefill), expected_prefill, rtol=0, atol=0)
                assert bench.output.shape == shape
                assert bench.validate_result() is None
                if variant == "optimized_tma":
                    expected_path = "full_graph" if mode == "full" or (mode == "full_and_piecewise" and shape[1] <= capture_limit) else "piecewise_graph"
                    assert bench._pending_graph_path == expected_path
                bench.inputs.out[-1, -1, -1] += 1024
                with pytest.raises(AssertionError):
                    bench.capture_verification_payload()
                bench.inputs.out.copy_(expected_decode.to(bench.device))
                bench.prefill_dst[-1, -1] += 1024
                with pytest.raises(AssertionError):
                    bench.capture_verification_payload()
                records.append({"iteration": iteration, "complete_outputs": True,
                                "changed_inputs": True, "decode_and_prefill_corruption_rejected": True})
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        caller.synchronize()
    finally:
        if bench is not None:
            bench.teardown()
        set_decode_options(previous)
    return records


@pytest.mark.parametrize("variant,mode,capture_limit", CUDA_CASES)
def test_real_cuda_prefill_decode_full_outputs_and_changed_inputs(variant, mode, capture_limit, tmp_path):
    reason = _cuda_unavailable_reason()
    if reason:
        pytest.skip(reason)
    from tests.cuda.validation_process import run_command
    output = tmp_path / "actual-cuda"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
               "--mode", mode, "--capture-limit", str(capture_limit), "--output-dir", str(output)]
    result = run_command(command, timeout=300)
    (tmp_path / "command-output.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "PASS" and len(report["checks"]) == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--capture-limit", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), CODE / "labs/persistent_decode/persistent_decode_common.py",
                    CODE / "labs/persistent_decode/tma_extension.py",
                    CODE / "core/common/headers/tma_helpers.cuh",
                    CODE / "core/common/headers/tma_2d_layout.hpp",
                    CODE / "core/common/headers/arch_detection.cuh",
                    CODE / "tests/cuda/validation_process.py"]
    source_paths.extend(CODE / "labs/persistent_decode" / (module + ".py") for module, _ in VARIANTS.values())
    report = {"status": "HOLD", "checks": [], "torch": torch.__version__, "cuda": torch.version.cuda,
              "source_sha256": {str(path.relative_to(CODE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              "scope": "Real CUDA whole-output/fresh-input checks only; no sanitizer or performance qualification"}
    reason = _cuda_unavailable_reason()
    if reason or args.worker is None:
        report["reason"] = reason or "Use pytest for the seven bounded CUDA worker cases"
        result = 3
    else:
        report["status"] = "RUNNING"
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            _cuda_worker(args.worker, args.mode, args.capture_limit, args.output_dir, report)
            report["status"] = "PASS"
            result = 0
        except BaseException:
            report.update(status="FAIL", reason=traceback.format_exc())
            result = 1
    report["artifacts_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(args.output_dir.iterdir())
                                  if path.is_file() and path.name != "report.json"}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reason": report.get("reason"), "checks": len(report["checks"])}, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
