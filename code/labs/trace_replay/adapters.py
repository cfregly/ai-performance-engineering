"""Fixed adapters for the repository's scan, softmax and NVFP4 operators."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch

from ch09.online_softmax_common import OnlineSoftmaxBenchmark, materialized_softmax
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.nvfp4_quantization.benchmarks import NVFP4Benchmark, NVFP4Workload, validate_inputs
from labs.nvfp4_quantization.reference import Quantized, evaluate_reference
from labs.prefix_scan.benchmarks import PrefixScanBenchmark
from labs.prefix_scan.reference import inclusive_scan
from labs.trace_replay.schema import load_bundle, validate_parameters
from labs.trace_replay.storage import load_event_inputs, read_tensor, sha256_file

CODE_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def source_paths(operations):
    paths = set(Path(__file__).parent.glob("*.py"))
    paths.add(CODE_ROOT / "core/utils/cooperative_subprocess.py")
    for operation in operations:
        if operation == "prefix_scan":
            paths.update(CODE_ROOT / "labs/prefix_scan" / name for name in ("benchmarks.py", "reference.py", "kernels.py"))
        elif operation == "online_softmax":
            paths.update(CODE_ROOT / "ch09" / name for name in ("online_softmax_common.py", "online_softmax_triton.py"))
        elif operation.startswith("nvfp4."):
            paths.update(CODE_ROOT / "labs/nvfp4_quantization" / name for name in ("benchmarks.py", "reference.py", "kernels.py"))
            paths.add(CODE_ROOT / "core/utils/nvfp4_layout.py")
        else:
            raise ValueError(f"unsupported operator: {operation}")
    return sorted(paths)


def source_identity(operations):
    return [{"path": path.relative_to(CODE_ROOT).as_posix(), "sha256": sha256_file(path)} for path in source_paths(operations)]


def verify_sources(bundle):
    actual = source_identity(event["operation"] for event in bundle.events)
    if actual != bundle.manifest["source_files"]:
        raise ValueError("implementation source hashes differ from the captured bundle")


def validate_operator_inputs(operation, parameters, inputs):
    validate_parameters(operation, parameters)
    if not isinstance(inputs, dict) or not inputs or any(not isinstance(value, torch.Tensor) for value in inputs.values()):
        raise ValueError("operator inputs must be a nonempty tensor dictionary")
    if any(not value.is_contiguous() for value in inputs.values()):
        raise ValueError("operator inputs must be contiguous")
    if operation == "prefix_scan":
        if set(inputs) != {"x"} or inputs["x"].dtype != torch.int32 or tuple(inputs["x"].shape) != (parameters["numel"],):
            raise ValueError("scan requires x with the declared int32 vector shape")
    elif operation == "online_softmax":
        if set(inputs) != {"x"} or inputs["x"].dtype != torch.float32 or tuple(inputs["x"].shape) != (parameters["rows"], parameters["cols"]):
            raise ValueError("softmax requires x with the declared FP32 matrix shape")
        if not bool(torch.isfinite(inputs["x"]).all()):
            raise ValueError("softmax capture requires finite input values")
    else:
        kind = operation.removeprefix("nvfp4.")
        expected = {"x", "global_scale", "valid_rows"}
        if kind == "add_rmsnorm":
            expected.update({"residual", "weight"})
        if set(inputs) != expected:
            raise ValueError("NVFP4 retained input names do not match the operation")
        validate_inputs(NVFP4Workload(kind, **parameters), inputs)
        if any(not bool(torch.isfinite(value).all()) for value in inputs.values() if value.is_floating_point()):
            raise ValueError("NVFP4 capture requires finite input values")


def output_dict(operation, result):
    """Expose full original buffers, including NVFP4 padding and residual."""
    if operation.startswith("nvfp4."):
        if not isinstance(result, Quantized):
            raise TypeError("NVFP4 operations must return Quantized")
        values = {"packed": result.packed, "scales": result.scales}
        if result.residual is not None:
            values["residual"] = result.residual
        return values
    if not isinstance(result, torch.Tensor):
        raise TypeError("scan and softmax operations must return a tensor")
    return {"output": result}


def reference_output(operation, parameters, inputs):
    validate_operator_inputs(operation, parameters, inputs)
    if operation == "prefix_scan":
        result = inclusive_scan(inputs["x"])
    elif operation == "online_softmax":
        result = materialized_softmax(inputs["x"])
    else:
        result = evaluate_reference(operation.removeprefix("nvfp4."), **inputs)
    return output_dict(operation, result)


def output_tolerances(operation):
    if operation == "online_softmax":
        return {"output": (2e-5, 2e-7)}
    if operation == "prefix_scan":
        return {"output": (0.0, 0.0)}
    result = {"packed": (0.0, 0.0), "scales": (0.0, 0.0)}
    if operation == "nvfp4.add_rmsnorm":
        result["residual"] = (0.0, 0.0)
    return result


def native_benchmark(operation, parameters, inputs, optimized):
    if operation == "prefix_scan":
        return PrefixScanBenchmark(optimized, **parameters, input_tensors=inputs)
    if operation == "online_softmax":
        return OnlineSoftmaxBenchmark(optimized, **parameters, input_tensors=inputs)
    return NVFP4Benchmark(
        operation.removeprefix("nvfp4."), optimized,
        NVFP4Workload(operation.removeprefix("nvfp4."), **parameters), input_tensors=inputs,
    )


class ReplayBenchmark(BaseBenchmark):
    """Replay one retained operation through its existing native benchmark.

    Retained values are independent of the harness seed. Live jitter still
    mutates the exact tensors read by benchmark_fn(). Reference-only execution
    is an explicit CPU correctness diagnostic and is never a GPU timing route.
    """

    _is_deterministic = True

    def __init__(self, optimized=False, reference_only=False):
        self.allow_cpu = bool(reference_only)
        super().__init__()
        self.optimized = bool(optimized)
        self.reference_only = bool(reference_only)
        self.bundle_path = ""
        self.event_id = ""
        self.expected_bundle_sha256 = None
        self.selected_bundle_sha256 = None
        self.inputs = None
        self.output = None
        self.operation = None
        self.parameters = None
        self.native = None
        self._captured_inputs = None
        self._captured_outputs = None
        self.captured_output_checked = False
        self.executed = False
        self.input_jitter_bounds = {"x": (-(2**31), 2**31)}
        self.phase_durations_s = {}

    def _declare_event(self, bundle, event):
        self.operation = event["operation"]
        self.parameters = event["parameters"]
        self.signature_equivalence_group = f"retained-operator:{bundle.sha256}:{self.event_id}"
        self.signature_equivalence_ignore_fields = ()
        self.story_metadata = {
            "replay_bundle_sha256": bundle.sha256,
            "replay_bundle_pin_sha256": self.expected_bundle_sha256 or self.selected_bundle_sha256,
            "replay_bundle_pin_source": (
                "argument" if self.expected_bundle_sha256 is not None else "selection"
            ),
            "replay_event_id": self.event_id,
        }
        units = self.parameters["numel"] if self.operation == "prefix_scan" else (
            self.parameters["rows"] * self.parameters["cols"] * self.parameters.get("batch", 1)
        )
        unit_name = (
            "scanned_values" if self.operation == "prefix_scan"
            else "normalized_values" if self.operation == "online_softmax"
            else "quantized_values"
        )
        self.register_workload_metadata(custom_units_per_iteration=units, custom_unit_name=unit_name)

    def apply_target_overrides(self, argv):
        parser = argparse.ArgumentParser(prog="trace_replay", add_help=False, allow_abbrev=False)
        parser.add_argument("--bundle", required=True)
        parser.add_argument("--event", required=True)
        parser.add_argument("--bundle-sha256")
        args = parser.parse_args(argv)
        if self.inputs is not None:
            parser.error("trace selection must happen before setup")
        if args.bundle_sha256 is not None and not _SHA256.fullmatch(args.bundle_sha256):
            raise ValueError("--bundle-sha256 must contain 64 lowercase hex digits")
        bundle_path = str(Path(args.bundle).absolute())
        if (bundle_path, args.event) != (self.bundle_path, self.event_id):
            self.selected_bundle_sha256 = None
        self.bundle_path = bundle_path
        self.event_id = args.event
        self.expected_bundle_sha256 = args.bundle_sha256
        self._declare_event(*self._event())

    def _event(self):
        if not self.bundle_path or not self.event_id:
            raise ValueError("trace replay requires --bundle and --event")
        bundle = load_bundle(Path(self.bundle_path))
        if self.expected_bundle_sha256 is not None and bundle.sha256 != self.expected_bundle_sha256:
            raise ValueError(
                "replay bundle SHA-256 differs from --bundle-sha256: "
                f"expected {self.expected_bundle_sha256}, found {bundle.sha256}"
            )
        if self.selected_bundle_sha256 is None:
            self.selected_bundle_sha256 = bundle.sha256
        elif bundle.sha256 != self.selected_bundle_sha256:
            raise ValueError(
                "replay bundle SHA-256 changed after selection: "
                f"expected {self.selected_bundle_sha256}, found {bundle.sha256}"
            )
        return bundle, bundle.event(self.event_id)

    def setup(self):
        self.executed = False
        self.output = None
        self.inputs = None
        self.native = None
        self._captured_inputs = None
        self._captured_outputs = None
        self.captured_output_checked = False
        if self.reference_only and self.optimized:
            raise ValueError("CPU reference diagnostics have no optimized performance arm")
        if not self.reference_only and not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: native trace replay requires an NVIDIA CUDA GPU")
        started = time.perf_counter()
        bundle, event = self._event()
        verify_sources(bundle)
        self._declare_event(bundle, event)
        target = "cpu" if self.reference_only else self.device
        retained = load_event_inputs(bundle, self.event_id, device=target)
        validate_operator_inputs(self.operation, self.parameters, retained)
        if self.operation == "online_softmax" and self.parameters["cols"] == 1:
            raise RuntimeError("SKIPPED: one-column softmax has constant output, so the required live input jitter cannot establish input dependence")
        if self.operation in {"nvfp4.quantize", "nvfp4.silu_mul"} and not bool(retained["valid_rows"].any()):
            raise RuntimeError(f"SKIPPED: fully masked {self.operation} output is constant, so the required live input jitter cannot establish input dependence")
        self._captured_inputs = {name: value.detach().cpu().clone() for name, value in retained.items()}
        self._captured_outputs = {name: read_tensor(bundle.root, descriptor) for name, descriptor in event["outputs"].items()}
        if not self.reference_only:
            torch.cuda.synchronize(self.device)
        self.phase_durations_s = {"bundle_load_and_transfer": time.perf_counter() - started}
        started = time.perf_counter()
        if self.reference_only:
            self.inputs = retained
        else:
            self.native = native_benchmark(self.operation, self.parameters, retained, self.optimized)
            self.native.device = self.device
            self.native.setup()
            self.inputs = self.native.inputs if self.operation.startswith("nvfp4.") else {"x": self.native.x}
            metadata = self.native.get_workload_metadata()
            if metadata is not None:
                self._workload_metadata = metadata
            torch.cuda.synchronize(self.device)
        self.phase_durations_s["native_setup"] = time.perf_counter() - started

    def benchmark_fn(self):
        self.executed = False
        self.output = None
        if self.inputs is None:
            raise RuntimeError("setup() must run before replay")
        if self.reference_only:
            self.output = reference_output(self.operation, self.parameters, self.inputs)
        else:
            self.native.benchmark_fn()
            self.output = output_dict(self.operation, self.native.output)
        self.executed = True

    def capture_verification_payload(self):
        if not self.executed or not self.output:
            raise RuntimeError("benchmark_fn() must produce actual outputs before verification")
        started = time.perf_counter()
        expected = reference_output(self.operation, self.parameters, self.inputs)
        tolerances = output_tolerances(self.operation)
        if set(self.output) != set(expected) or set(self.output) != set(tolerances):
            raise AssertionError("replay output names do not match the complete operation result")
        for name, reference in expected.items():
            actual = self.output[name]
            if actual.dtype != reference.dtype:
                raise AssertionError(f"replay output dtype differs for {name}")
            rtol, atol = tolerances[name]
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        if "scales" in self.output and not bool(torch.isfinite(self.output["scales"].view(torch.float8_e4m3fn).float()).all()):
            raise AssertionError("NVFP4 scale overflow or nonfinite output")
        self.captured_output_checked = False
        unchanged_inputs = all(
            torch.equal(value.detach().cpu().reshape(-1).view(torch.uint8), self._captured_inputs[name].reshape(-1).view(torch.uint8))
            for name, value in self.inputs.items()
        )
        if unchanged_inputs:
            if set(self.output) != set(self._captured_outputs):
                raise AssertionError("captured output names do not match the complete replay result")
            for name, captured in self._captured_outputs.items():
                rtol, atol = tolerances[name]
                torch.testing.assert_close(self.output[name].detach().cpu(), captured, rtol=rtol, atol=atol)
            self.captured_output_checked = True
        self.phase_durations_s["full_correctness"] = time.perf_counter() - started

    def get_verify_inputs(self):
        if not self.executed or self.inputs is None:
            raise RuntimeError("replay has not produced verification inputs")
        if self.operation == "prefix_scan":
            return {"x": self.inputs["x"].view(1, -1)}
        return {"x": self.inputs["x"], **{name: value for name, value in self.inputs.items() if name != "x"}}

    def get_verify_output(self):
        if not self.executed or not self.output:
            raise RuntimeError("replay has not produced verification outputs")
        return {name: tensor.detach().clone() for name, tensor in self.output.items()}

    def get_input_signature(self):
        inputs = self.get_verify_inputs()
        return InputSignature(
            shapes={name: tuple(value.shape) for name, value in inputs.items()},
            dtypes={name: str(value.dtype) for name, value in inputs.items()},
            batch_size=1 if self.operation == "prefix_scan" else self.parameters.get("batch", self.parameters["rows"]),
            parameter_count=self.inputs["weight"].numel() if "weight" in self.inputs else 0,
            precision_flags=PrecisionFlags(bf16=self.operation.startswith("nvfp4."), tf32=False),
        )

    def get_output_tolerance(self):
        operation = self.operation or self._event()[1]["operation"]
        return (2e-5, 2e-7) if operation == "online_softmax" else (0.0, 0.0)

    def get_output_tolerances(self):
        return output_tolerances(self.operation or self._event()[1]["operation"])

    def get_config(self):
        return BenchmarkConfig(iterations=30, warmup=10)

    def get_custom_metrics(self):
        return {
            **{f"excluded_{name}_seconds": value for name, value in self.phase_durations_s.items()},
            "captured_output_checked": float(self.captured_output_checked),
        }

    def teardown(self):
        if self.native is not None:
            self.native.teardown()
        self.native = self.inputs = self.output = None
        self._captured_inputs = self._captured_outputs = None
        self.executed = False
