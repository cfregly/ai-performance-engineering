"""Exercise retained tensor capture, native replay, and real verification checks."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verify_runner import VerifyConfig
from core.harness.benchmark_harness import BaseBenchmark
from tests.protection_test_utils import make_runner, preserve_rng_state

CODE = Path(__file__).resolve().parents[1]


class _ChildStoryMetadataTransportBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Distinguish a real isolated worker receipt from its parent declaration."""

    allow_cpu = True
    _is_deterministic = True

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None
        self.story_metadata = {
            "receipt_origin": "parent_initialization",
            "process_id": os.getpid(),
            "replay_bundle_sha256": "a" * 64,
        }
        self.register_workload_metadata(samples_per_iteration=8.0)

    def setup(self) -> None:
        self.output = None
        self.story_metadata = {
            "receipt_origin": "worker_setup",
            "process_id": os.getpid(),
            "replay_bundle_sha256": "b" * 64,
        }

    def benchmark_fn(self) -> None:
        self.output = self.input.square()

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=8,
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self):
        if self.output is None or not torch.equal(self.output, self.input.square()):
            return "isolated story metadata transport produced an invalid output"
        return None


@pytest.fixture(autouse=True)
def _preserve_test_rng_state():
    with preserve_rng_state():
        yield


def _plan(path: Path, operations: tuple[str, ...] = ("prefix_scan", "online_softmax")) -> Path:
    parameters = {
        "prefix_scan": {"numel": 17},
        "online_softmax": {"rows": 3, "cols": 17},
        "nvfp4.quantize": {"batch": 1, "rows": 3, "cols": 48},
        "nvfp4.add_rmsnorm": {"batch": 1, "rows": 3, "cols": 48},
        "nvfp4.silu_mul": {"batch": 2, "rows": 3, "cols": 48},
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "native-operator-capture-plan.v1",
                "events": [
                    {
                        "event_id": f"event-{index}",
                        "operation": operation,
                        "parameters": parameters[operation],
                        "seed": 501 + index,
                        "source": {
                            "kind": "generated_by_recipe",
                            "description": "Small retained input for native replay correctness",
                        },
                    }
                    for index, operation in enumerate(operations)
                ],
            }
        )
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([0, -32768, 16256, -16512, 32640, -128, 32705, 1], dtype=torch.int16)
        .view(torch.bfloat16)
        .reshape(2, 4),
        torch.tensor([-(2**31), -1, 0, 1, 2**31 - 1, 17], dtype=torch.int32).reshape(2, 3),
        torch.tensor([0, 1, 127, 128, 254, 255], dtype=torch.uint8).reshape(2, 3),
    ],
    ids=["bf16-bit-patterns", "int32-limits", "uint8-packed-values"],
)
def test_raw_tensor_roundtrip_preserves_dtype_shape_and_every_byte(tmp_path: Path, tensor) -> None:
    from labs.trace_replay.storage import read_tensor, write_tensor

    path = tmp_path / "tensor.bin"
    descriptor = write_tensor(path, tensor)
    restored = read_tensor(tmp_path, descriptor, device="cpu")

    assert restored.dtype == tensor.dtype
    assert restored.shape == tensor.shape
    assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
    expected_bytes = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert path.read_bytes() == expected_bytes
    assert descriptor["nbytes"] == len(expected_bytes)
    assert descriptor["sha256"] == hashlib.sha256(expected_bytes).hexdigest()


def test_raw_tensor_read_rejects_changed_content(tmp_path: Path) -> None:
    from labs.trace_replay.storage import read_tensor, write_tensor

    path = tmp_path / "tensor.bin"
    descriptor = write_tensor(path, torch.arange(16, dtype=torch.int32))
    contents = bytearray(path.read_bytes())
    contents[0] ^= 1
    path.write_bytes(contents)

    with pytest.raises((ValueError, RuntimeError), match="(?i)hash|sha256|digest"):
        read_tensor(tmp_path, descriptor)


@pytest.mark.parametrize("path", ["../outside.bin", "/tmp/outside.bin"])
def test_raw_tensor_read_rejects_paths_outside_bundle(tmp_path: Path, path: str) -> None:
    from labs.trace_replay.storage import read_tensor, write_tensor

    descriptor = write_tensor(tmp_path / "tensor.bin", torch.arange(8, dtype=torch.uint8))
    descriptor["path"] = path

    with pytest.raises((ValueError, RuntimeError), match="(?i)path|relative|bundle|outside"):
        read_tensor(tmp_path, descriptor)


def test_raw_tensor_read_rejects_symlink_escape(tmp_path: Path) -> None:
    from labs.trace_replay.storage import read_tensor, write_tensor

    root = tmp_path / "bundle"
    root.mkdir()
    descriptor = write_tensor(tmp_path / "outside.bin", torch.arange(8, dtype=torch.uint8))
    (root / "linked.bin").symlink_to(tmp_path / "outside.bin")
    descriptor["path"] = "linked.bin"

    with pytest.raises((ValueError, RuntimeError), match="(?i)path|symlink|bundle|outside"):
        read_tensor(root, descriptor)


@pytest.mark.parametrize("change", [{"dtype": "float128"}, {"shape": [1, 999]}, {"nbytes": 1}])
def test_raw_tensor_read_rejects_invalid_typed_descriptor(tmp_path: Path, change: dict) -> None:
    from labs.trace_replay.storage import read_tensor, write_tensor

    descriptor = write_tensor(tmp_path / "tensor.bin", torch.arange(8, dtype=torch.int32))

    with pytest.raises((ValueError, RuntimeError)):
        read_tensor(tmp_path, descriptor | change)


def test_failure_receipts_preserve_earlier_failures(tmp_path: Path) -> None:
    from labs.trace_replay.runner import write_failure_receipt

    first = write_failure_receipt(tmp_path, "event-0", "verification", "first failure")
    first_bytes = first.read_bytes()
    second = write_failure_receipt(tmp_path, "event-0", "verification", "second failure")

    assert first != second
    assert first.read_bytes() == first_bytes
    assert "first failure" in first_bytes.decode()
    assert "second failure" in second.read_text()
    receipt = json.loads(first_bytes)
    assert receipt["event_id"] == "event-0"
    assert receipt["phase"] == "verification"
    assert receipt["error"] == "first failure"
    assert receipt["created_at"]


@pytest.mark.parametrize("change", ["version", "unknown-operation", "duplicate-event", "extra-field"])
def test_capture_rejects_invalid_plan_schema(tmp_path: Path, change: str) -> None:
    from labs.trace_replay.runner import capture_plan

    path = _plan(tmp_path / "plan.json")
    plan = json.loads(path.read_text())
    if change == "version":
        plan["schema_version"] = "native-operator-capture-plan.v999"
    elif change == "unknown-operation":
        plan["events"][0]["operation"] = "arbitrary.python.call"
    elif change == "duplicate-event":
        plan["events"][1]["event_id"] = plan["events"][0]["event_id"]
    else:
        plan["events"][0]["execute"] = "untrusted command"
    path.write_text(json.dumps(plan))

    with pytest.raises((ValueError, RuntimeError)):
        capture_plan(path, tmp_path / "capture", device="cpu")


@pytest.mark.parametrize("change", ["manifest-hash", "version", "tensor-path", "extra-field"])
def test_bundle_loading_rejects_manifest_and_schema_changes(tmp_path: Path, change: str) -> None:
    from labs.trace_replay.runner import capture_plan
    from labs.trace_replay.schema import load_bundle

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )
    manifest_path = bundle.root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if change == "version":
        manifest["schema_version"] = "native-operator-trace.v999"
    elif change == "tensor-path":
        manifest["events"][0]["inputs"]["x"]["path"] = "../outside.bin"
    else:
        manifest["unrecognized_field"] = True
    raw = (json.dumps(manifest) + "\n").encode()
    manifest_path.write_bytes(raw)
    if change != "manifest-hash":
        (bundle.root / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")

    with pytest.raises((ValueError, RuntimeError)):
        load_bundle(bundle.root)


def test_capture_never_overwrites_an_existing_bundle(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan

    plan = _plan(tmp_path / "plan.json", ("prefix_scan",))
    bundle = capture_plan(plan, tmp_path / "capture", device="cpu")
    before = {path: path.read_bytes() for path in bundle.root.rglob("*") if path.is_file()}

    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        capture_plan(plan, bundle.root, device="cpu")

    assert all(path.read_bytes() == data for path, data in before.items())


def _observed_call(event_id, operation, parameters, inputs, **status):
    return {
        "event_id": event_id,
        "operation": operation,
        "parameters": parameters,
        "inputs": inputs,
        "source": {
            "kind": "observed_call",
            "description": "Inputs supplied by a test caller",
            "caller": "native-replay-correctness",
        },
        **status,
    }


def _strided_observed_input(operation):
    if operation == "prefix_scan":
        storage = torch.arange(16, dtype=torch.int32)
        return storage, storage[::2], {"numel": 8}
    storage = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    return storage, storage[:, ::2], {"rows": 4, "cols": 6}


@pytest.mark.parametrize("operation", ["prefix_scan", "online_softmax"])
def test_public_capture_rejects_completed_strided_inputs_before_cloning(tmp_path: Path, operation):
    from labs.trace_replay.runner import capture_event

    storage, x, parameters = _strided_observed_input(operation)
    original_storage = storage.clone()
    original_stride = x.stride()
    output_dir = tmp_path / "capture"
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="completed observed calls require contiguous input tensors"):
        capture_event(
            operation, parameters, {"x": x}, event_id="strided",
            source={"kind": "observed_call", "description": "Caller supplied a stride-two input view"},
            output_dir=output_dir, device="cpu",
        )

    assert torch.equal(storage, original_storage)
    assert x.stride() == original_stride
    assert not x.is_contiguous()
    assert not output_dir.exists()


@pytest.mark.parametrize("operation", ["prefix_scan", "online_softmax"])
@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_public_capture_retains_strided_values_from_incomplete_calls(tmp_path: Path, operation, status):
    from labs.trace_replay.runner import capture_events
    from labs.trace_replay.storage import read_tensor

    storage, x, parameters = _strided_observed_input(operation)
    original_storage = storage.clone()
    original_values = x.clone()
    original_stride = x.stride()
    error = "Source operation did not complete"

    bundle = capture_events(
        [_observed_call("incomplete", operation, parameters, {"x": x}, status=status, error=error)],
        tmp_path / "capture", device="cpu",
    )

    event = bundle.event("incomplete")
    assert event["status"] == status
    assert event["error"] == error
    assert event["outputs"] == {}
    assert torch.equal(read_tensor(bundle.root, event["inputs"]["x"]), original_values)
    assert torch.equal(storage, original_storage)
    assert x.stride() == original_stride
    receipt = json.loads(next((bundle.root / "failures").glob("*.json")).read_text())
    assert receipt["status"] == status
    assert receipt["phase"] == "source"


def test_observed_roster_preserves_order_values_and_source_dispositions(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_events
    from labs.trace_replay.storage import load_event_inputs, read_tensor

    scan = torch.tensor([2, -1, 4], dtype=torch.int32)
    failed_input = torch.tensor([[1.0, 2.0, 3.0]])
    softmax = torch.tensor([[-3.0, 0.0, 2.0], [1.0, -1.0, 5.0]])
    calls = [
        _observed_call("source-z", "prefix_scan", {"numel": 3}, {"x": scan}),
        _observed_call(
            "source-a",
            "online_softmax",
            {"rows": 1, "cols": 3},
            {"x": failed_input},
            status="failed",
            error="source operation failed before producing output",
        ),
        _observed_call(
            "source-cancelled",
            "prefix_scan",
            {"numel": 3},
            {},
            status="cancelled",
            error="caller cancelled this source operation",
        ),
        _observed_call(
            "source-resumed", "online_softmax", {"rows": 2, "cols": 3}, {"x": softmax}
        ),
    ]
    original_scan = scan.clone()
    bundle = capture_events(calls, tmp_path / "capture", device="cpu")
    scan.zero_()

    assert [event["event_id"] for event in bundle.events] == [call["event_id"] for call in calls]
    assert [event["status"] for event in bundle.events] == [
        "completed", "failed", "cancelled", "completed"
    ]
    assert bundle.manifest["plan_sha256"] is None
    assert all(event["seed"] is None for event in bundle.events)
    assert all(event["source"] == call["source"] for event, call in zip(bundle.events, calls, strict=True))
    assert torch.equal(load_event_inputs(bundle, "source-z")["x"], original_scan)
    assert torch.equal(
        read_tensor(bundle.root, bundle.event("source-a")["inputs"]["x"]), failed_input
    )
    assert bundle.event("source-a")["outputs"] == {}
    assert bundle.event("source-cancelled")["inputs"] == {}
    assert bundle.event("source-cancelled")["outputs"] == {}
    receipts = {
        row["event_id"]: row
        for path in (bundle.root / "failures").glob("*.json")
        if (row := json.loads(path.read_text()))["phase"] == "source"
    }
    assert receipts.keys() == {"source-a", "source-cancelled"}
    for call in calls[1:3]:
        receipt = receipts[call["event_id"]]
        assert receipt["status"] == call["status"]
        assert receipt["error"] == call["error"]
        assert bundle.event(call["event_id"])["error"] == call["error"]


def test_observed_generator_snapshots_reused_inputs_and_metadata_at_each_yield(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_events
    from labs.trace_replay.storage import load_event_inputs

    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    parameters = {"numel": 3}
    source = {"kind": "observed_call", "description": "first call", "caller": "reused-buffer-caller"}

    def calls():
        for index in range(2):
            source["description"] = f"observed call {index}"
            yield {
                "event_id": f"event-{index}", "operation": "prefix_scan",
                "parameters": parameters, "inputs": {"x": x}, "source": source,
            }
            x.add_(10)
        parameters["numel"] = 999
        source["description"] = "caller advanced beyond these observations"

    bundle = capture_events(calls(), tmp_path / "capture", device="cpu")

    assert [event["status"] for event in bundle.events] == ["completed", "completed"]
    for index, event in enumerate(bundle.events):
        assert event["parameters"] == {"numel": 3}
        assert event["source"] == {
            "kind": "observed_call", "description": f"observed call {index}",
            "caller": "reused-buffer-caller",
        }
        assert load_event_inputs(bundle, event["event_id"])["x"].tolist() == [
            1 + 10 * index, 2 + 10 * index, 3 + 10 * index,
        ]


def test_finite_input_nvfp4_arithmetic_overflow_retains_failed_outputs(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_events
    from labs.trace_replay.storage import read_tensor

    # Overflow the residual sum so NaN arises before FP8 conversion. This avoids
    # depending on how a CPU build casts an out-of-range finite scale to FP8.
    x = torch.full((1, 2, 16), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16)
    inputs = {
        "x": x,
        "residual": x.clone(),
        "weight": torch.ones(16, dtype=torch.bfloat16),
        "global_scale": torch.ones(1, dtype=torch.float32),
        "valid_rows": torch.tensor([2], dtype=torch.int32),
    }
    assert all(bool(torch.isfinite(value).all()) for value in inputs.values())
    bundle = capture_events(
        [_observed_call("overflow", "nvfp4.add_rmsnorm", {"batch": 1, "rows": 2, "cols": 16}, inputs)],
        tmp_path / "capture",
        device="cpu",
    )
    event = bundle.event("overflow")

    assert event["status"] == "failed"
    assert "overflow or nonfinite" in event["error"]
    assert event["outputs"].keys() == {"packed", "scales", "residual"}
    scales = read_tensor(bundle.root, event["outputs"]["scales"])
    assert scales.dtype == torch.uint8
    assert not bool(torch.isfinite(scales.view(torch.float8_e4m3fn).float()).all())
    assert read_tensor(bundle.root, event["outputs"]["packed"]).numel() == 16
    residual = read_tensor(bundle.root, event["outputs"]["residual"])
    assert residual.dtype == torch.bfloat16
    assert residual.shape == (1, 2, 16)
    assert bool(torch.isinf(residual).all())
    receipt_paths = list((bundle.root / "failures").glob("*.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text())
    assert receipt["status"] == "failed"
    assert receipt["phase"] == "capture"
    assert "overflow or nonfinite" in receipt["error"]


def test_valid_hashes_do_not_authorize_changed_captured_oracle_values(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan
    from labs.trace_replay.schema import load_bundle
    from labs.trace_replay.storage import read_tensor

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )
    manifest_path = bundle.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["events"][0]["outputs"]["output"]
    changed = read_tensor(bundle.root, descriptor)
    changed[0] += 1
    raw = changed.view(torch.uint8).numpy().tobytes()
    (bundle.root / descriptor["path"]).write_bytes(raw)
    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_bytes = (json.dumps(manifest) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    (bundle.root / "manifest.sha256").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n")
    changed_bundle = load_bundle(bundle.root)
    benchmark = _configured_replay(changed_bundle, "event-0")
    benchmark.setup()
    try:
        benchmark.benchmark_fn()
        assert not torch.equal(benchmark.output["output"], changed)
        with pytest.raises(AssertionError):
            benchmark.capture_verification_payload()
        assert benchmark.captured_output_checked is False
    finally:
        benchmark.teardown()


@pytest.mark.parametrize("operation", ["online_softmax", "nvfp4.quantize", "nvfp4.silu_mul"])
def test_invariant_outputs_have_explicit_replay_diagnostic(tmp_path: Path, operation: str) -> None:
    from core.benchmark.quarantine import detect_skip_flags
    from labs.trace_replay.runner import capture_events

    if operation == "online_softmax":
        parameters = {"rows": 2, "cols": 1}
        inputs = {"x": torch.tensor([[3.0], [-4.0]])}
    else:
        parameters = {"batch": 1, "rows": 2, "cols": 16}
        width = 32 if operation == "nvfp4.silu_mul" else 16
        inputs = {
            "x": torch.arange(2 * width, dtype=torch.bfloat16).reshape(1, 2, width) / 16,
            "global_scale": torch.ones(1),
            "valid_rows": torch.zeros(1, dtype=torch.int32),
        }
    bundle = capture_events(
        [_observed_call("invariant", operation, parameters, inputs)],
        tmp_path / "capture",
        device="cpu",
    )
    assert bundle.event("invariant")["status"] == "completed"
    benchmark = _configured_replay(bundle, "invariant")

    assert detect_skip_flags(benchmark) is None
    with pytest.raises(RuntimeError, match="SKIPPED:.*constant.*jitter"):
        benchmark.setup()
    assert detect_skip_flags(benchmark) is None


def test_profiler_receipts_require_explicit_native_capture_status(tmp_path: Path) -> None:
    from labs.trace_replay.runner import profiler_receipts

    reports, issues = profiler_receipts([{"optimizations": [{}]}], tmp_path)

    assert reports == []
    assert len(issues) == 4
    assert all("did not explicitly succeed" in issue for issue in issues)


def test_profiler_receipts_reject_forged_reports_outside_event_capture(tmp_path: Path) -> None:
    from labs.trace_replay.runner import profiler_receipts

    outside = tmp_path / "outside"
    outside.mkdir()
    event_root = tmp_path / "event"
    outcome = {"optimizations": [{}]}
    for arm, record in (("baseline", outcome), ("optimized", outcome["optimizations"][0])):
        record[f"{arm}_profiler_statuses"] = {"nsys": "succeeded", "ncu": "succeeded"}
        for profiler in ("nsys", "ncu"):
            path = outside / f"{arm}.{profiler}-rep"
            path.write_text("Untrusted metadata does not establish a native profiler capture.\n")
            record[f"{arm}_{profiler}_rep"] = str(path)

    reports, issues = profiler_receipts([outcome], event_root)

    assert reports == []
    assert len(issues) == 4


@pytest.mark.skipif(os.name != "posix", reason="owned CLI process groups require POSIX signals")
def test_owned_cli_interrupt_waits_for_real_child_cleanup(tmp_path: Path, monkeypatch) -> None:
    from labs.trace_replay.runner import wait_for_owned_cli

    marker = tmp_path / "cleanup.txt"
    child_code = """
import signal
import sys
from pathlib import Path

def cleanup(_signal, _frame):
    Path(sys.argv[1]).write_text("cleanup-completed\\n")
    raise SystemExit(23)

signal.signal(signal.SIGINT, cleanup)
print("ready", flush=True)
signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    original_wait = process.wait
    wait_calls = 0
    receipt = {}

    def interrupt_wait(*args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls <= 2:
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    try:
        assert process.stdout.readline() == "ready\n"
        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(process, "wait", interrupt_wait)
            with pytest.raises(KeyboardInterrupt):
                wait_for_owned_cli(process, receipt)
        assert marker.read_text() == "cleanup-completed\n"
        assert process.returncode == receipt["returncode"] == 23
        assert receipt["execution"] == {
            "process_id": process.pid,
            "process_group_id": process.pid,
            "start_new_session": True,
        }
        assert receipt["cancellation"]["signal_forwarded"] is True
        assert receipt["cancellation"]["status"] == "cleanup_completed"
        assert receipt["cancellation"]["repeated_interrupts"] == 1
        assert receipt["cancellation"]["completed_at"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            original_wait()
        process.communicate()


@pytest.mark.parametrize("kind", ["quantize", "add_rmsnorm", "silu_mul"])
def test_nvfp4_output_dictionary_retains_complete_typed_buffers(kind: str) -> None:
    from labs.nvfp4_quantization.reference import evaluate_reference
    from labs.trace_replay.adapters import output_dict

    generator = torch.Generator().manual_seed(701)
    width = 96 if kind == "silu_mul" else 48
    inputs = {
        "x": torch.randn(2, 3, width, generator=generator).bfloat16(),
        "global_scale": torch.tensor([0.5, 1.5]),
        "valid_rows": torch.tensor([0, 3] if kind == "silu_mul" else [3, 3], dtype=torch.int32),
    }
    if kind == "add_rmsnorm":
        inputs["residual"] = torch.randn(2, 3, width, generator=generator).bfloat16()
        inputs["weight"] = torch.randn(width, generator=generator).bfloat16()
    result = evaluate_reference(kind, **inputs)
    outputs = output_dict(f"nvfp4.{kind}", result)

    expected = {"packed": result.packed, "scales": result.scales}
    if kind == "add_rmsnorm":
        expected["residual"] = result.residual
    assert outputs.keys() == expected.keys()
    for name, value in expected.items():
        assert outputs[name].dtype == value.dtype
        assert outputs[name].shape == value.shape
        assert torch.equal(outputs[name].view(torch.uint8), value.contiguous().view(torch.uint8))
    assert outputs["packed"].dtype == outputs["scales"].dtype == torch.uint8
    if kind == "add_rmsnorm":
        assert outputs["residual"].dtype == torch.bfloat16


def test_cpu_capture_loads_independent_inputs_and_replays_native_references(tmp_path: Path) -> None:
    from labs.nvfp4_quantization.reference import evaluate_reference
    from labs.trace_replay.adapters import ReplayBenchmark, output_dict
    from labs.trace_replay.runner import capture_plan
    from labs.trace_replay.storage import load_event_inputs, read_tensor

    operations = (
        "prefix_scan",
        "online_softmax",
        "nvfp4.quantize",
        "nvfp4.add_rmsnorm",
        "nvfp4.silu_mul",
    )
    plan = _plan(tmp_path / "plan.json", operations)
    bundle = capture_plan(plan, tmp_path / "capture", device="cpu")
    assert bundle.manifest["schema_version"] == "native-operator-trace.v1"
    assert bundle.manifest["capture_device"] == "cpu"
    assert bundle.manifest["capture_backend"] == "cpu_reference"
    assert bundle.manifest["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert bundle.manifest["source_files"]
    assert all(len(source["sha256"]) == 64 for source in bundle.manifest["source_files"])
    assert isinstance(bundle.manifest["runtime"], dict)
    assert str(torch.__version__) in json.dumps(bundle.manifest["runtime"])
    assert len(bundle.sha256) == 64
    assert len(bundle.events) == len(operations)

    for event, operation in zip(bundle.events, operations, strict=True):
        assert event["operation"] == operation
        assert event["status"] == "completed"
        assert event["source"]["kind"] == "generated_by_recipe"
        inputs = load_event_inputs(bundle, event["event_id"], device="cpu")
        original = {name: value.clone() for name, value in inputs.items()}
        inputs["x"].zero_()
        loaded_again = load_event_inputs(bundle, event["event_id"], device="cpu")
        for name, value in original.items():
            assert torch.equal(loaded_again[name].view(torch.uint8), value.view(torch.uint8))

        benchmark = ReplayBenchmark(optimized=False, reference_only=True)
        benchmark.device = torch.device("cpu")
        benchmark.apply_target_overrides(["--bundle", str(bundle.root), "--event", event["event_id"]])
        benchmark.setup()
        try:
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            if operation == "prefix_scan":
                expected = output_dict(operation, original["x"].long().cumsum(0).int())
            elif operation == "online_softmax":
                expected = output_dict(operation, torch.softmax(original["x"].double(), 1).float())
            else:
                expected = output_dict(
                    operation, evaluate_reference(operation.removeprefix("nvfp4."), **original)
                )
            assert benchmark.output.keys() == expected.keys()
            assert event["outputs"].keys() == expected.keys()
            for name, value in expected.items():
                assert benchmark.output[name].dtype == value.dtype
                assert benchmark.output[name].shape == value.shape
                torch.testing.assert_close(
                    benchmark.output[name],
                    value,
                    rtol=2e-5 if operation == "online_softmax" else 0,
                    atol=2e-7 if operation == "online_softmax" else 0,
                )
                torch.testing.assert_close(
                    read_tensor(bundle.root, event["outputs"][name]),
                    value,
                    rtol=2e-5 if operation == "online_softmax" else 0,
                    atol=2e-7 if operation == "online_softmax" else 0,
                )
        finally:
            benchmark.teardown()


def _configured_replay(
    bundle, event_id, *, benchmark_type=None, reference_only=True, bundle_sha256=None
):
    from labs.trace_replay.adapters import ReplayBenchmark

    benchmark_type = benchmark_type or ReplayBenchmark
    benchmark = benchmark_type(optimized=False, reference_only=reference_only)
    if reference_only:
        benchmark.device = torch.device("cpu")
    argv = ["--bundle", str(bundle.root), "--event", event_id]
    if bundle_sha256 is not None:
        argv.extend(["--bundle-sha256", bundle_sha256])
    benchmark.apply_target_overrides(argv)
    return benchmark


@pytest.mark.parametrize("explicit_pin", [False, True], ids=["selection-pin", "argument-pin"])
def test_replay_adapter_rejects_valid_bundle_swap_before_setup(
    tmp_path: Path, explicit_pin: bool
) -> None:
    from labs.trace_replay.runner import capture_plan

    active = tmp_path / "capture"
    first = capture_plan(
        _plan(tmp_path / "plan-a.json", ("prefix_scan",)), active, device="cpu"
    )
    benchmark = _configured_replay(
        first,
        "event-0",
        bundle_sha256=first.sha256 if explicit_pin else None,
    )
    assert benchmark.selected_bundle_sha256 == first.sha256
    first.root.rename(tmp_path / "capture-a")
    second_plan = _plan(tmp_path / "plan-b.json", ("prefix_scan",))
    second_payload = json.loads(second_plan.read_text())
    second_payload["events"][0]["seed"] = 9001
    second_plan.write_text(json.dumps(second_payload) + "\n")
    second = capture_plan(second_plan, active, device="cpu")
    assert second.sha256 != first.sha256

    with pytest.raises(ValueError, match="replay bundle SHA-256"):
        benchmark.setup()

    assert benchmark.inputs is None
    assert benchmark.executed is False


def test_replay_adapter_rejects_wrong_argument_digest_at_selection(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )

    with pytest.raises(ValueError, match="differs from --bundle-sha256"):
        _configured_replay(bundle, "event-0", bundle_sha256="0" * 64)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_replay_adapter_rejects_invalid_argument_digest(tmp_path: Path, digest: str) -> None:
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )

    with pytest.raises(ValueError, match="64 lowercase hex digits"):
        _configured_replay(bundle, "event-0", bundle_sha256=digest)


def test_replay_adapter_rejects_corrupted_selected_tensor_during_setup(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )
    benchmark = _configured_replay(bundle, "event-0", bundle_sha256=bundle.sha256)
    descriptor = bundle.event("event-0")["inputs"]["x"]
    tensor_path = bundle.root / descriptor["path"]
    contents = bytearray(tensor_path.read_bytes())
    contents[0] ^= 1
    tensor_path.write_bytes(contents)

    with pytest.raises(ValueError, match="retained tensor hash or byte count mismatch"):
        benchmark.setup()

    assert benchmark.inputs is None
    assert benchmark.executed is False


def test_actual_cpu_replay_adapter_passes_default_verification(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan", "online_softmax")),
        tmp_path / "capture",
        device="cpu",
    )
    before = {path: path.read_bytes() for path in bundle.root.rglob("*") if path.is_file()}
    for event in bundle.events:
        baseline = _configured_replay(bundle, event["event_id"])
        candidate = _configured_replay(bundle, event["event_id"])
        assert baseline.expected_bundle_sha256 is None
        assert baseline.selected_bundle_sha256 == bundle.sha256
        assert baseline.story_metadata == {
            "replay_bundle_sha256": bundle.sha256,
            "replay_bundle_pin_sha256": bundle.sha256,
            "replay_bundle_pin_source": "selection",
            "replay_event_id": event["event_id"],
        }
        with preserve_rng_state():
            result = make_runner(tmp_path / event["event_id"]).verify_pair(
                baseline, candidate, VerifyConfig()
            )
        assert result.passed, (event["event_id"], result.reason, result.details)
        assert not (result.details or {}).get("warnings")
    assert all(path.read_bytes() == data for path, data in before.items())


def test_actual_replay_adapter_cached_output_fails_live_jitter(tmp_path: Path) -> None:
    from labs.trace_replay.adapters import ReplayBenchmark
    from labs.trace_replay.runner import capture_plan

    class CachedReplay(ReplayBenchmark):
        def setup(self):
            super().setup()
            self.cached_output = None

        def benchmark_fn(self):
            if self.cached_output is None:
                super().benchmark_fn()
                self.cached_output = {name: value.clone() for name, value in self.output.items()}
            else:
                self.output = {name: value.clone() for name, value in self.cached_output.items()}
                self.executed = True

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("online_softmax",)),
        tmp_path / "capture",
        device="cpu",
    )
    baseline = _configured_replay(bundle, "event-0")
    cached = _configured_replay(bundle, "event-0", benchmark_type=CachedReplay)

    with preserve_rng_state():
        result = make_runner(tmp_path).verify_pair(baseline, cached, VerifyConfig())

    assert cached._is_deterministic is True
    assert not result.passed
    assert result.reason == "jitter_fail", result.details


def test_real_isolated_worker_story_metadata_is_authoritative() -> None:
    from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness

    benchmark = _ChildStoryMetadataTransportBenchmark()
    parent_process_id = os.getpid()
    assert benchmark.story_metadata == {
        "receipt_origin": "parent_initialization",
        "process_id": parent_process_id,
        "replay_bundle_sha256": "a" * 64,
    }
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        use_subprocess=False,
        adaptive_iterations=False,
        min_run_time_ms=0.0,
        min_total_duration_ms=0.0,
        enable_profiling=False,
        enable_memory_tracking=False,
        enforce_environment_validation=False,
    )
    harness = BenchmarkHarness(config=config)
    harness._ensure_runtime_initialized()

    result = harness._benchmark_with_subprocess(benchmark, config)

    assert not result.errors, result.errors
    assert result.runtime_provenance is not None
    assert result.story_metadata == {
        "receipt_origin": "worker_setup",
        "process_id": result.runtime_provenance.process_id,
        "replay_bundle_sha256": "b" * 64,
    }
    assert result.story_metadata["process_id"] != parent_process_id
    assert benchmark.story_metadata["replay_bundle_sha256"] == "a" * 64


def _identity_outcome(bundle_sha256, event_id):
    metadata = {
        "replay_bundle_sha256": bundle_sha256,
        "replay_bundle_pin_sha256": bundle_sha256,
        "replay_bundle_pin_source": "argument",
        "replay_event_id": event_id,
    }
    return {
        "baseline_story_metadata": dict(metadata),
        "optimizations": [
            {
                "story_metadata": dict(metadata),
                "input_verification": {
                    "equivalence": {
                        "group": f"retained-operator:{bundle_sha256}:{event_id}",
                        "ignore_fields": [],
                    }
                },
            }
        ],
    }


def test_replay_identity_receipts_match_coordinator_pin() -> None:
    from labs.trace_replay.runner import replay_identity_receipts

    digest = "a" * 64
    receipts, issues = replay_identity_receipts(
        [_identity_outcome(digest, "event-0")], "event-0", digest
    )

    assert issues == []
    assert receipts == [
        {
            "arm": "baseline",
            "bundle_sha256": digest,
            "bundle_pin_sha256": digest,
            "bundle_pin_source": "argument",
            "event_id": "event-0",
        },
        {
            "arm": "optimized",
            "bundle_sha256": digest,
            "bundle_pin_sha256": digest,
            "bundle_pin_source": "argument",
            "event_id": "event-0",
        },
    ]


@pytest.mark.parametrize(
    "change",
    ["equivalence", "baseline-digest", "optimized-pin", "event"],
)
def test_replay_identity_receipts_reject_mismatches(change: str) -> None:
    from labs.trace_replay.runner import replay_identity_receipts

    digest = "a" * 64
    outcome = _identity_outcome(digest, "event-0")
    if change == "equivalence":
        outcome["optimizations"][0]["input_verification"]["equivalence"]["group"] = (
            f"retained-operator:{'b' * 64}:event-0"
        )
    elif change == "baseline-digest":
        outcome["baseline_story_metadata"]["replay_bundle_sha256"] = "b" * 64
    elif change == "optimized-pin":
        outcome["optimizations"][0]["story_metadata"]["replay_bundle_pin_sha256"] = "b" * 64
    else:
        outcome["optimizations"][0]["story_metadata"]["replay_event_id"] = "event-1"

    _, issues = replay_identity_receipts([outcome], "event-0", digest)

    assert issues


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU coordinator path requires no CUDA")
def test_real_cpu_replay_cli_writes_bundle_pin_before_cuda_skip(tmp_path: Path) -> None:
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "labs.trace_replay.trace_replay_tool",
            "replay",
            "--bundle",
            str(bundle.root),
            "--output-dir",
            str(tmp_path / "replay"),
        ],
        cwd=CODE,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)

    assert result["status"] == "failed"
    assert result["bundle_sha256"] == bundle.sha256
    assert result["events"][0]["status"] == "skipped"
    command = json.loads((tmp_path / "replay" / "event-0" / "command.json").read_text())
    assert command["bundle_sha256"] == bundle.sha256
    target_override = command["argv"][command["argv"].index("--target-extra-arg") + 1]
    label, override = target_override.split("=", 1)
    assert label == "labs/trace_replay:trace_replay"
    worker_argv = shlex.split(override)
    pin_index = worker_argv.index("--bundle-sha256")
    assert worker_argv[pin_index + 1] == bundle.sha256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
@pytest.mark.parametrize(
    "operation",
    ["prefix_scan", "online_softmax", "nvfp4.quantize", "nvfp4.add_rmsnorm", "nvfp4.silu_mul"],
)
def test_cuda_capture_replays_actual_native_pair_with_default_verification(
    tmp_path: Path, operation: str
) -> None:
    pytest.importorskip("triton")
    if operation.startswith("nvfp4.") and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("NVFP4 replay requires Blackwell SM100")
    from labs.trace_replay.adapters import ReplayBenchmark
    from labs.trace_replay.runner import capture_plan

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", (operation,)), tmp_path / "capture", device="cuda"
    )
    assert bundle.manifest["capture_backend"] == "cuda_native"
    baseline = _configured_replay(bundle, "event-0", reference_only=False)
    candidate = ReplayBenchmark(optimized=True)
    candidate.apply_target_overrides(["--bundle", str(bundle.root), "--event", "event-0"])

    with preserve_rng_state():
        result = make_runner(tmp_path).verify_pair(baseline, candidate, VerifyConfig())

    assert result.passed, (operation, result.reason, result.details)
    assert not (result.details or {}).get("warnings")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
def test_cuda_coordinator_reconciles_worker_bundle_receipts(tmp_path: Path) -> None:
    pytest.importorskip("triton")
    from labs.trace_replay.runner import capture_plan, replay_bundle

    bundle = capture_plan(
        _plan(tmp_path / "plan.json", ("prefix_scan",)), tmp_path / "capture", device="cpu"
    )
    result = replay_bundle(bundle.root, tmp_path / "replay")

    assert result["status"] == "completed", result
    row = result["events"][0]
    assert row["status"] == "completed"
    assert row.get("bundle_identity_issues") is None
    assert [receipt["bundle_sha256"] for receipt in row["bundle_identity_receipts"]] == [
        bundle.sha256,
        bundle.sha256,
    ]
    assert [receipt["bundle_pin_sha256"] for receipt in row["bundle_identity_receipts"]] == [
        bundle.sha256,
        bundle.sha256,
    ]


def test_real_cli_capture_and_validate_retains_reference_bundle(tmp_path: Path) -> None:
    from labs.trace_replay.schema import load_bundle

    plan = _plan(tmp_path / "plan.json")
    output = tmp_path / "captured"
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    capture = subprocess.run(
        [
            sys.executable,
            "-m",
            "labs.trace_replay.trace_replay_tool",
            "capture",
            "--plan",
            str(plan),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=CODE,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert capture.returncode == 0, capture.stdout + capture.stderr
    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "labs.trace_replay.trace_replay_tool",
            "validate",
            "--bundle",
            str(output),
        ],
        cwd=CODE,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
    assert len(load_bundle(output).events) == 2


class _FixedCaptureWork(VerificationPayloadMixin, BaseBenchmark):
    """Use fixed captured data while retaining mutable inputs for live jitter."""

    allow_cpu = True
    _is_deterministic = True

    def __init__(self, *, cached: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.captured = torch.linspace(-2, 2, 32).reshape(4, 8)
        self.cached = cached
        self.input = self.output = None
        self.setup_inputs: list[torch.Tensor] = []
        self.executed_inputs: list[torch.Tensor] = []
        self.register_workload_metadata(samples_per_iteration=4, requests_per_iteration=1)

    def setup(self) -> None:
        self.input = self.captured.clone()
        self.output = None
        self.setup_inputs.append(self.input.clone())

    def benchmark_fn(self) -> None:
        if self.input is None:
            raise RuntimeError("setup() must run first")
        self.executed_inputs.append(self.input.clone())
        if self.output is None or not self.cached:
            self.output = self.input.sin().mul(2)

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run first")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=4,
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.input = self.output = None
        self._verification_payload = None
        super().teardown()


def test_fixed_capture_default_verification_runs_live_input_perturbation(tmp_path: Path) -> None:
    baseline = _FixedCaptureWork()
    candidate = _FixedCaptureWork()
    captured_bytes = candidate.captured.view(torch.uint8).clone()

    with preserve_rng_state():
        result = make_runner(tmp_path).verify_pair(baseline, candidate, VerifyConfig())

    assert result.passed, (result.reason, result.details)
    assert not (result.details or {}).get("warnings")
    assert len(candidate.setup_inputs) == 3
    assert all(torch.equal(value, candidate.captured) for value in candidate.setup_inputs)
    assert len(candidate.executed_inputs) == 4
    assert not torch.equal(candidate.executed_inputs[-1], candidate.captured)
    assert torch.equal(candidate.captured.view(torch.uint8), captured_bytes)


def test_fixed_capture_deterministic_flag_does_not_admit_cached_output(tmp_path: Path) -> None:
    baseline = _FixedCaptureWork()
    cached = _FixedCaptureWork(cached=True)

    with preserve_rng_state():
        result = make_runner(tmp_path).verify_pair(baseline, cached, VerifyConfig())

    assert not result.passed
    assert result.reason == "jitter_fail"
    assert "output unchanged" in result.details["error"]
    assert len(cached.setup_inputs) == 3
    assert len(cached.executed_inputs) == 4
    assert not torch.equal(cached.executed_inputs[-1], cached.captured)
