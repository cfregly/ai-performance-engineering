"""Real bounded child-result round trips preserve every reference byte."""

import json
import time

import pytest
import torch

from core.benchmark.verification import PrecisionFlags
from labs.train_distributed.training_utils import child_result as transport


def _context(tmp_path, monkeypatch, *, limit=256 * 1024, atol=0.0):
    contract = transport.TorchrunChildResultContract(
        profile="tests:lossless-child-storage",
        input_names=("features",),
        output_names=("prediction",),
        per_rank_batch_size=2,
        parameter_count=8,
        precision_flags=PrecisionFlags(tf32=False),
        output_tolerance=(0.0, atol),
        independent_reference="separately-computed-addition",
        max_rank_payload_bytes=limit,
    )
    wall, monotonic = time.time_ns(), time.monotonic_ns()
    for key, value in {
        transport.RESULT_DIR_ENV: str(tmp_path),
        transport.RUN_ID_ENV: "lossless-storage-roundtrip",
        transport.CONTRACT_ENV: json.dumps(contract.to_dict()),
        transport.WORLD_SIZE_ENV: "1",
        transport.ITERATIONS_ENV: "3",
        transport.LAUNCH_WALL_NS_ENV: str(wall),
        transport.LAUNCH_MONOTONIC_NS_ENV: str(monotonic),
        "RANK": "0",
        "WORLD_SIZE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    return {
        "contract": contract,
        "run_id": "lossless-storage-roundtrip",
        "world_size": 1,
        "requested_iterations": 3,
        "launch_wall_ns": wall,
        "launch_monotonic_ns": monotonic,
    }


def _write(tmp_path, *, reference_delta=0.0, signed_zero=False):
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    changed = inputs + 1
    # Four output tensors alone exceed the test cap; two fit comfortably.
    values = torch.arange(16_384, dtype=torch.float32).reshape(2, -1)
    output = values + inputs.sum()
    reference = values + inputs.sum() + reference_delta
    if signed_zero:
        output[0, 0] = 0.0
        reference[0, 0] = -0.0
    changed_output = values + changed.sum()
    changed_reference = values + changed.sum() + reference_delta
    expected = {
        "outputs": output,
        "reference_outputs": reference,
        "sensitivity_outputs": changed_output,
        "sensitivity_reference_outputs": changed_reference,
    }
    path = transport.write_training_child_result(
        inputs={"features": inputs},
        outputs={"prediction": output},
        reference_outputs={"prediction": reference},
        sensitivity_inputs={"features": changed},
        sensitivity_outputs={"prediction": changed_output},
        sensitivity_reference_outputs={"prediction": changed_reference},
        completed_iterations=3,
    )
    assert path == tmp_path / "rank-0.pt"
    return path, expected


def _validate(tmp_path, context):
    return transport.validate_training_child_result_bundle(
        tmp_path,
        **context,
        finish_wall_ns=time.time_ns(),
        finish_monotonic_ns=time.monotonic_ns(),
    )


def _bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def test_full_reference_roundtrip_fits_unchanged_bound(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    path, expected = _write(tmp_path)
    assert path.stat().st_size < context["contract"].max_rank_payload_bytes
    loaded = torch.load(path, weights_only=True)
    for name, tensor in expected.items():
        assert torch.equal(_bytes(loaded[name]["prediction"]), _bytes(tensor))
    for left, right in (("outputs", "reference_outputs"),
                        ("sensitivity_outputs", "sensitivity_reference_outputs")):
        assert loaded[left]["prediction"].data_ptr() == loaded[right]["prediction"].data_ptr()
    bundle = _validate(tmp_path, context)
    assert torch.equal(bundle["verify_output"]["rank-0:prediction"], expected["outputs"])


@pytest.mark.parametrize("signed_zero", [False, True])
def test_distinct_reference_bytes_survive_roundtrip(tmp_path, monkeypatch, signed_zero):
    context = _context(tmp_path, monkeypatch, limit=512 * 1024, atol=1.0)
    path, expected = _write(
        tmp_path, reference_delta=0.0 if signed_zero else 0.25, signed_zero=signed_zero
    )
    loaded = torch.load(path, weights_only=True)
    actual = loaded["outputs"]["prediction"]
    reference = loaded["reference_outputs"]["prediction"]
    assert actual.data_ptr() != reference.data_ptr()
    for name, tensor in expected.items():
        assert torch.equal(_bytes(loaded[name]["prediction"]), _bytes(tensor))
    _validate(tmp_path, context)


def test_invalid_full_reference_is_rejected_before_packing(tmp_path, monkeypatch):
    _context(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="independent reference"):
        _write(tmp_path, reference_delta=1.0)
    assert not list(tmp_path.iterdir())


def test_storage_sharing_does_not_bypass_payload_limit(tmp_path, monkeypatch):
    _context(tmp_path, monkeypatch, limit=64 * 1024)
    with pytest.raises(RuntimeError, match=r"size limit: \d+ bytes > 65536 bytes"):
        _write(tmp_path)
    assert not list(tmp_path.iterdir())


def test_parent_rejects_changed_reference_after_serialization(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    path, _ = _write(tmp_path)
    loaded = torch.load(path, weights_only=True)
    loaded["reference_outputs"]["prediction"] = loaded["reference_outputs"]["prediction"].clone()
    loaded["reference_outputs"]["prediction"][-1, -1] += 1
    torch.save(loaded, path)
    with pytest.raises(RuntimeError, match="independent reference"):
        _validate(tmp_path, context)
