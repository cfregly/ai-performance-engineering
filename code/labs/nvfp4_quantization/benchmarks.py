"""Three BF16-to-NVFP4 operation pairs with explicit workload configurations."""

import argparse
import importlib
from dataclasses import dataclass

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.nvfp4_quantization.reference import Quantized, evaluate_reference


@dataclass(frozen=True)
class NVFP4Workload:
    kind: str
    batch: int
    rows: int
    cols: int

    def __post_init__(self):
        if self.kind not in {"add_rmsnorm", "quantize", "silu_mul"}:
            raise ValueError("unknown NVFP4 task")
        if (
            any(type(v) is not int or v <= 0 for v in (self.batch, self.rows, self.cols))
            or self.cols % 16
            or self.cols > 16384
        ):
            raise ValueError(
                "positive dimensions and K divisible by 16, at most 16384, are required"
            )


WORKLOADS = {
    "rmsnorm_2048": NVFP4Workload("add_rmsnorm", 1, 128, 2048),
    "rmsnorm_4096": NVFP4Workload("add_rmsnorm", 1, 128, 4096),
    "rmsnorm_8192": NVFP4Workload("add_rmsnorm", 1, 128, 8192),
    "quantize_14336": NVFP4Workload("quantize", 1, 128, 14336),
    "silu_mul_7168": NVFP4Workload("silu_mul", 8, 256, 7168),
    "silu_mul_14336": NVFP4Workload("silu_mul", 8, 256, 14336),
}

DEFAULT_WORKLOADS = {
    "add_rmsnorm": "rmsnorm_2048",
    "quantize": "quantize_14336",
    "silu_mul": "silu_mul_7168",
}


def build_inputs(workload, device):
    w = workload
    width = w.cols * (2 if w.kind == "silu_mul" else 1)
    inputs = {
        "x": torch.randn(w.batch, w.rows, width, device=device, dtype=torch.bfloat16),
        "global_scale": torch.ones(w.batch, device=device, dtype=torch.float32),
        "valid_rows": (
            torch.randint(1, w.rows + 1, (w.batch,), dtype=torch.int32, device=device)
            if w.kind == "silu_mul"
            else torch.full((w.batch,), w.rows, dtype=torch.int32, device=device)
        ),
    }
    if w.kind == "add_rmsnorm":
        inputs["residual"] = torch.randn_like(inputs["x"])
        inputs["weight"] = torch.randn(w.cols, device=device, dtype=torch.bfloat16)
    return inputs


def require_sm100(device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: NVFP4 pairs require a CUDA Blackwell SM100 GPU")
    if device is not None and torch.device(device).type != "cuda":
        raise ValueError("fused NVFP4 inputs must be on CUDA")
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("SKIPPED: NVFP4 pairs are scoped to Blackwell SM100")


def validate_inputs(workload, inputs):
    w = workload
    x = inputs["x"]
    expected_shape = (w.batch, w.rows, w.cols * (2 if w.kind == "silu_mul" else 1))
    if tuple(x.shape) != expected_shape or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("input must match the contiguous BF16 workload shape")
    for name, shape, dtype in [
        ("global_scale", (w.batch,), torch.float32),
        ("valid_rows", (w.batch,), torch.int32),
    ]:
        tensor = inputs[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"invalid {name} shape, dtype, device or layout")
    if not torch.isfinite(inputs["global_scale"]).all() or not (inputs["global_scale"] > 0).all():
        raise ValueError("global scales must be finite and positive")
    if not ((inputs["valid_rows"] >= 0) & (inputs["valid_rows"] <= w.rows)).all():
        raise ValueError("valid row counts must be within the workload")
    if w.kind == "add_rmsnorm":
        if not (inputs["valid_rows"] == w.rows).all():
            raise ValueError("RMSNorm tasks require all residual rows")
        for name, shape in [("residual", expected_shape), ("weight", (w.cols,))]:
            tensor = inputs[name]
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != x.dtype
                or tensor.device != x.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"invalid {name} tensor")


class FusedPlan:
    def __init__(self, workload, inputs):
        require_sm100(inputs["x"].device)
        validate_inputs(workload, inputs)
        self.workload, self.inputs = workload, inputs
        w = workload
        try:
            self.kernels = importlib.import_module("labs.nvfp4_quantization.kernels")
        except ModuleNotFoundError as exc:
            if exc.name not in {"triton", "triton.language"}:
                raise
            raise RuntimeError("SKIPPED: fused NVFP4 requires Triton") from exc
        device = inputs["x"].device
        if device.type != "cuda":
            raise ValueError("fused plan inputs must be on CUDA")
        self.padded_rows = (w.rows + 127) // 128 * 128
        self.padded_groups = (w.cols // 16 + 3) // 4 * 4
        packed = torch.empty(w.batch, w.rows, w.cols // 2, device=device, dtype=torch.uint8)
        scale_shape = (
            (w.batch, self.padded_rows, self.padded_groups)
            if w.kind == "silu_mul"
            else (w.batch, w.rows, w.cols // 16)
        )
        # Only padding is invariant. Every logical scale and packed value is
        # written by each kernel invocation, including masked expert rows.
        scales = torch.zeros(scale_shape, device=device, dtype=torch.uint8)
        residual = torch.empty_like(inputs["x"]) if w.kind == "add_rmsnorm" else None
        self.output = Quantized(
            packed, scales, (w.batch, w.rows, w.cols), w.kind == "silu_mul", residual
        )

    def run(self):
        w, tensors = self.workload, self.inputs
        self.kernels.fused_quantize[(w.batch * w.rows,)](
            tensors["x"],
            tensors.get("residual", tensors["x"]),
            tensors.get("weight", tensors["x"]),
            tensors["global_scale"],
            tensors["valid_rows"],
            self.output.packed,
            self.output.scales,
            self.output.residual if self.output.residual is not None else tensors["x"],
            w.rows,
            w.cols,
            w.kind,
            self.padded_rows,
            self.padded_groups,
            1e-6,
            1 << (w.cols - 1).bit_length(),
            num_warps=8,
            enable_fp_fusion=False,
        )
        return self.output


class NVFP4Benchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, kind, optimized, workload=None):
        super().__init__()
        if kind not in DEFAULT_WORKLOADS:
            raise ValueError("unknown NVFP4 operation")
        self.kind = kind
        self.workload = workload or WORKLOADS[DEFAULT_WORKLOADS[kind]]
        if self.workload.kind != kind:
            raise ValueError("benchmark and workload operation differ")
        self.optimized = optimized
        self.inputs = self.plan = self.output = None
        self._register_workload_metadata()

    def _register_workload_metadata(self):
        self.register_workload_metadata(
            custom_units_per_iteration=self.workload.batch
            * self.workload.rows
            * self.workload.cols,
            custom_unit_name="quantized_values",
        )

    def apply_target_overrides(self, argv):
        parser = argparse.ArgumentParser(
            prog=f"nvfp4_quantization:{self.kind}", add_help=False, allow_abbrev=False
        )
        parser.add_argument(
            "--workload",
            choices=[name for name, w in WORKLOADS.items() if w.kind == self.kind],
            default=DEFAULT_WORKLOADS[self.kind],
        )
        args = parser.parse_args(argv)
        if self.inputs is not None:
            parser.error("workload selection must happen before setup")
        self.workload = WORKLOADS[args.workload]
        self._register_workload_metadata()

    def setup(self):
        self.output = None
        # BaseBenchmark resolves device lazily and raises on a CPU-only host.
        require_sm100(self.device if torch.cuda.is_available() else None)
        self.inputs = build_inputs(self.workload, self.device)
        validate_inputs(self.workload, self.inputs)
        self.plan = FusedPlan(self.workload, self.inputs) if self.optimized else None

    def benchmark_fn(self):
        if self.inputs is None:
            raise RuntimeError("setup() must run first")
        self.output = None
        self.output = (
            self.plan.run()
            if self.optimized
            else evaluate_reference(self.workload.kind, **self.inputs)
        )

    def capture_verification_payload(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        expected = evaluate_reference(self.workload.kind, **self.inputs)
        for name in ("packed", "scales", "residual"):
            value, reference = getattr(self.output, name), getattr(expected, name)
            if reference is not None:
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
        if not torch.isfinite(self.output.scales.view(torch.float8_e4m3fn).float()).all():
            raise AssertionError("NVFP4 scale overflow or nonfinite input")
        # Every byte is represented exactly; this is not a checksum or sample.
        buffers = [self.output.packed.flatten().float(), self.output.scales.flatten().float()]
        if self.output.residual is not None:
            buffers.append(self.output.residual.flatten().float())
        self._set_verification_payload(
            inputs=self.inputs,
            output=torch.cat(buffers),
            batch_size=self.workload.batch * self.workload.rows,
            parameter_count=self.inputs["weight"].numel() if "weight" in self.inputs else 0,
            precision_flags={"bf16": True},
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self):
        return BenchmarkConfig(iterations=30, warmup=10)

    def teardown(self):
        self.inputs = self.plan = self.output = None
