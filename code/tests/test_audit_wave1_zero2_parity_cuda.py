"""Opt-in, real two-GPU gate for the common ZeRO training workload.

No CPU substitute: the numerical gate requires two allocated CUDA devices and
NCCL. From code/: AISP_RUN_ZERO2_PARITY_CUDA=1 python -m pytest -q
tests/test_audit_wave1_zero2_parity_cuda.py. To retain a standalone receipt use
python tests/test_audit_wave1_zero2_parity_cuda.py --output-dir <new-dir> --execute.
The default suite runs only host negative controls and skips the CUDA gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

WORLD = 2
UPDATES = 4  # One warmup-equivalent update, then three further updates.
MICROS = 2
HIDDEN = 17
BATCH = 5
SEED = 711
SOURCE_PATHS = (
    "labs/train_distributed/zero2_common.py",
    "labs/train_distributed/baseline_zero2.py",
    "labs/train_distributed/optimized_zero2.py",
    "labs/train_distributed/baseline_zero2_multigpu.py",
    "labs/train_distributed/optimized_zero2_multigpu.py",
    "labs/train_distributed/training_utils/torchrun_harness.py",
    "tests/test_audit_wave1_zero2_parity_cuda.py",
)
LIMITATIONS = [
    "Small eager numerical workload; not the default hidden-size 10000 / 12 GiB payload scale.",
    "No torch.compile, throughput, memory-saving, CUDA sanitizer or pinned-stack qualification.",
    "Exercises production common training_step/components, not the four CLI main functions.",
    "Common warmup/timing regions are source-reviewed; this gate does not measure them.",
    "TorchrunScriptBenchmark's toy signature payload does not verify actual training.",
]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _hold_reason(execute: bool) -> str | None:
    if not execute:
        return "Explicit opt-in required after allocating exactly two CUDA devices."
    if not torch.cuda.is_available():
        return "CUDA unavailable; no GPU work launched."
    if not dist.is_available() or not dist.is_nccl_available():
        return "NCCL unavailable; no GPU work launched."
    if torch.cuda.device_count() != WORLD:
        return "Expose exactly two allocated CUDA devices with CUDA_VISIBLE_DEVICES."
    for rank in range(WORLD):
        if torch.cuda.get_device_capability(rank)[0] < 8:
            return f"Visible CUDA device {rank} lacks native BF16 support (requires CC >= 8)."
    return None


def _snapshot(model, loss) -> dict:
    return {
        "loss": loss.detach().cpu().clone(),
        "parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
        "gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
    }


def _assert_snapshot(actual, expected) -> None:
    # Every element is checked. A checksum or a subset of parameters cannot pass.
    assert actual.keys() == expected.keys()
    assert torch.isfinite(actual["loss"]).all()
    assert torch.isfinite(expected["loss"]).all()
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=1e-5, atol=1e-5)
    for kind, rtol, atol in (("parameters", 1e-5, 1e-6), ("gradients", 3e-4, 3e-6)):
        assert actual[kind].keys() == expected[kind].keys()
        for name, got in actual[kind].items():
            wanted = expected[kind][name]
            assert torch.isfinite(got).all(), (kind, name)
            assert torch.isfinite(wanted).all(), (kind, name)
            torch.testing.assert_close(got, wanted, rtol=rtol, atol=atol,
                                       msg=lambda msg, kind=kind, name=name: f"{kind}/{name}: {msg}")


def _reference_update(model, optimizer, generators, device, rank, payload):
    """Dense independent AdamW: average both ranks' losses before clipping.

    This function uses no DDP, production training_step or communication hook.
    Each rank's microbatch has the production shape, including BF16 autocast.
    """
    optimizer.zero_grad(set_to_none=True)
    local_last = None
    for _micro in range(MICROS):
        for source_rank, generator in enumerate(generators):
            rx = torch.empty(BATCH, HIDDEN, device=device, dtype=torch.float32)
            ry = torch.empty_like(rx)
            rx.normal_(generator=generator)
            ry.normal_(generator=generator)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(rx), ry) / MICROS
            if payload:
                loss = loss + model.extra_grad_payload.sum() * 0.0
            (loss / WORLD).backward()
            if source_rank == rank:
                local_last = (loss.detach(), rx.detach(), ry.detach())

    # Independent implementation of the same global L2 clipping rule.
    gradients = [p.grad for p in model.parameters()]
    norm = torch.linalg.vector_norm(torch.cat([g.detach().float().flatten() for g in gradients]))
    assert float(norm) > 1.0, "Constructed workload must actually exercise clipping"
    coefficient = (1.0 / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(gradient.dtype))
    optimizer.step()
    return local_last, float(norm)


def _optimizer_state(model, optimizer, *, optimized):
    local = optimizer.optim if optimized else optimizer
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for parameter, state in local.state.items():
        result[names[id(parameter)]] = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
    return result


def _check_optimizer_configuration(optimizer, *, optimized):
    local = optimizer.optim if optimized else optimizer
    assert isinstance(local, torch.optim.AdamW)
    for group in local.param_groups:
        assert group["lr"] == .01
        assert group["betas"] == (.9, .95)
        assert group["weight_decay"] == .05
        assert group["fused"] is True


def _assert_optimizer_state(actual, expected, update):
    assert actual
    for name, state in actual.items():
        assert state.keys() == expected[name].keys()
        assert int(state["step"]) == update + 1
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.isfinite(state[key]).all()
            torch.testing.assert_close(state[key], expected[name][key], rtol=5e-4, atol=3e-7,
                                       msg=lambda msg, name=name, key=key: f"optimizer/{name}/{key}: {msg}")


def _check_replicas_and_ownership(model, optimizer, *, optimized, device):
    for parameter in model.parameters():
        for tensor in (parameter.detach(), parameter.grad):
            replicas = [torch.empty_like(tensor) for _ in range(WORLD)]
            dist.all_gather(replicas, tensor.contiguous())
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    local = optimizer.optim if optimized else optimizer
    ownership = torch.tensor([int(p in local.state) for p in model.parameters()],
                             device=device, dtype=torch.int32)
    local_count = int(ownership.sum())
    if optimized:
        assert 0 < local_count < ownership.numel()
    else:
        assert local_count == ownership.numel()
    dist.all_reduce(ownership, op=dist.ReduceOp.SUM)
    assert torch.all(ownership == (1 if optimized else WORLD))
    return local_count


def _worker(rank: int, rendezvous: str, output_dir: str) -> None:
    from labs.train_distributed.zero2_common import (
        build_model, build_training_components, training_step,
    )

    output = Path(output_dir)
    report = {"rank": rank, "status": "RUNNING", "checks": [], "world_size": WORLD}
    try:
        torch.set_num_threads(1)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=WORLD,
                                device_id=device, timeout=timedelta(seconds=30))
        properties = torch.cuda.get_device_properties(device)
        report["device"] = {"ordinal": rank, "name": properties.name,
                            "capability": list(torch.cuda.get_device_capability(device))}
        _write_json(output / f"rank-{rank}.json", report)
        for payload in (False, True):
            torch.manual_seed(51)
            initial = build_model(HIDDEN, device)
            assert all(p.dtype == torch.float32 for p in initial.parameters())
            assert sum(isinstance(layer, torch.nn.Linear) for layer in initial) == 7
            assert sum(isinstance(layer, torch.nn.GELU) for layer in initial) == 6
            with torch.no_grad():
                initial[-1].bias.fill_(100.0)
            if payload:
                # A separate odd BF16 bucket exercises the real RS/AG padding path.
                initial.register_parameter("extra_grad_payload", torch.nn.Parameter(
                    torch.zeros(7, device=device, dtype=torch.bfloat16)))
            baseline_updates = []
            for optimized in (False, True):
                model, reference = copy.deepcopy(initial), copy.deepcopy(initial)
                ddp, optimizer = build_training_components(
                    model, .01, optimized=optimized, device_ids=[rank])
                dense = torch.optim.AdamW(reference.parameters(), lr=.01, betas=(.9, .95),
                                          weight_decay=.05, fused=True)
                _check_optimizer_configuration(optimizer, optimized=optimized)
                actual_rng = torch.Generator(device=device).manual_seed(SEED + rank)
                reference_rngs = [torch.Generator(device=device).manual_seed(SEED + other)
                                  for other in range(WORLD)]
                x = torch.empty(BATCH, HIDDEN, device=device, dtype=torch.float32)
                y = torch.empty_like(x)
                pointers = (x.data_ptr(), y.data_ptr())
                case = f"{'rsag-zero' if optimized else 'dense-ddp'}-{'odd-bf16-payload' if payload else 'model'}"
                for update in range(UPDATES):
                    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
                    actual_loss = training_step(ddp, optimizer, x, y, actual_rng, MICROS,
                                                extra_param=model.extra_grad_payload if payload else None)
                    last, norm = _reference_update(reference, dense, reference_rngs, device, rank, payload)
                    expected_loss, expected_x, expected_y = last
                    actual = _snapshot(model, actual_loss)
                    expected = _snapshot(reference, expected_loss)
                    state = _optimizer_state(model, optimizer, optimized=optimized)
                    expected_state = _optimizer_state(reference, dense, optimized=False)
                    artifact = output / f"rank-{rank}-{case}-update-{update}.pt"
                    # Preserve complete arrays before numerical assertions, including failed attempts.
                    torch.save({"actual": actual, "reference": expected, "optimizer": state,
                                "reference_optimizer": expected_state,
                                "actual_last_inputs": (x.detach().cpu(), y.detach().cpu()),
                                "expected_last_inputs": (expected_x.cpu(), expected_y.cpu()),
                                "reference_preclip_norm": norm}, artifact)
                    assert pointers == (x.data_ptr(), y.data_ptr())
                    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
                    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
                    _assert_snapshot(actual, expected)
                    _assert_optimizer_state(state, expected_state, update)
                    assert any(not torch.equal(p, before[name]) for name, p in actual["parameters"].items())
                    if optimized:
                        _assert_snapshot(actual, baseline_updates[update])
                    else:
                        baseline_updates.append(actual)
                    count = _check_replicas_and_ownership(model, optimizer, optimized=optimized, device=device)
                    report["checks"].append({"case": case, "update": update,
                                             "local_optimizer_state_entries": count,
                                             "reference_preclip_norm": norm,
                                             "full_tensor_artifact": artifact.name, "status": "PASS"})
                    _write_json(output / f"rank-{rank}.json", report)
                dist.barrier()
                del ddp, optimizer, model, reference, dense
        report["status"] = "PASS"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        _write_json(output / f"rank-{rank}.json", report)
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_gate(output: Path, *, execute: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier attempt.
    report = {"status": "HOLD", "checks": [], "limitations": LIMITATIONS,
              "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
              "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
              "source_sha256": {name: hashlib.sha256((CODE / name).read_bytes()).hexdigest()
                                for name in SOURCE_PATHS}}
    reason = _hold_reason(execute)
    if reason:
        report["reason"] = reason
        _write_json(output / "report.json", report)
        return report
    report["status"] = "RUNNING"
    _write_json(output / "report.json", report)
    context = None
    try:
        context = torch.multiprocessing.spawn(
            _worker, args=((output / "rendezvous").as_uri(), str(output)),
            nprocs=WORLD, join=False)
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Two-GPU common training validation exceeded 180 seconds")
        ranks = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(WORLD)]
        assert all(rank["status"] == "PASS" and len(rank["checks"]) == 4 * UPDATES for rank in ranks)
        assert all(check["status"] == "PASS" for rank in ranks for check in rank["checks"])
        report.update(status="PASS", checks=ranks)
    except BaseException:
        report.update(status="FAIL", reason=traceback.format_exc())
    finally:
        if context is not None:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        report["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(output.iterdir())
                                     if p.is_file() and p.name != "report.json"}
        _write_json(output / "report.json", report)
    return report


def test_cuda_gate_requires_opt_in_before_querying_devices(monkeypatch):
    def unexpected_query():
        pytest.fail("A non-opted-in suite must not query CUDA or launch workers")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_query)
    assert "opt-in" in _hold_reason(False)


def test_unavailable_cuda_produces_hold_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.multiprocessing, "spawn", lambda *a, **kw: pytest.fail("No GPU launch"))
    report = _run_gate(tmp_path / "attempt", execute=True)
    assert report["status"] == "HOLD" and report["checks"] == []
    assert report["source_sha256"] and "CUDA unavailable" in report["reason"]
    assert json.loads((tmp_path / "attempt/report.json").read_text()) == report


@pytest.mark.parametrize("corruption", ["late_parameter", "late_gradient", "nan", "missing_parameter"])
def test_full_tensor_validator_rejects_corruption(corruption):
    expected = {"loss": torch.tensor(1.),
                "parameters": {"first": torch.zeros(2, 3), "last": torch.zeros(2, 3)},
                "gradients": {"first": torch.zeros(2, 3), "last": torch.zeros(2, 3)}}
    actual = copy.deepcopy(expected)
    if corruption == "late_parameter":
        actual["parameters"]["last"][-1, -1] = .1
    elif corruption == "late_gradient":
        actual["gradients"]["last"][-1, -1] = .1
    elif corruption == "nan":
        actual["loss"].fill_(float("nan"))
    else:
        del actual["parameters"]["last"]
    with pytest.raises(AssertionError):
        _assert_snapshot(actual, expected)


def test_both_common_training_variants_on_two_real_cuda_ranks(tmp_path):
    execute = os.environ.get("AISP_RUN_ZERO2_PARITY_CUDA") == "1"
    report = _run_gate(tmp_path / "cuda-attempt", execute=execute)
    if report["status"] == "HOLD":
        pytest.skip("HOLD: " + report["reason"])
    assert report["status"] == "PASS", report.get("reason")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="Confirm two visible GPUs are allocated for this numerical gate")
    args = parser.parse_args()
    report = _run_gate(args.output_dir.resolve(), execute=args.execute)
    print(json.dumps({"status": report["status"], "reason": report.get("reason"),
                      "report": str(args.output_dir / "report.json")}, indent=2))
    return {"PASS": 0, "HOLD": 3, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
