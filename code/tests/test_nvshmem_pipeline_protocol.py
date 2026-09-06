from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.nvshmem_pipeline_parallel_multigpu import (
    NVSHMEMPipelineEngine,
    PipelineStageModule,
    PipelineTransferBuffer,
    _create_pipeline_control_group,
    _gather_pipeline_verification,
    _require_global_nvshmem,
    _require_transport_consensus,
)
from ch04.nvshmem_pipeline_result import (
    NVSHMEM_PIPELINE_RESULT_CALLBACK,
    NVSHMEM_PIPELINE_RESULT_LAUNCH_WALL_NS_ENV,
    NVSHMEM_PIPELINE_RESULT_SCHEMA,
    NVSHMEMPipelineChildResultMixin,
    NVSHMEMPipelineWorkloadResult,
    write_nvshmem_pipeline_child_result,
)


def _gloo_pipeline_worker(rank: int, rendezvous_path: str, output_path: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(42 + rank)
        device = torch.device("cpu")
        hidden_dim = 4
        microbatch_size = 2
        seq_len = 3
        num_microbatches = 4
        stage = PipelineStageModule(hidden_dim).to(dtype=torch.float16)
        engine = NVSHMEMPipelineEngine(
            stage=stage,
            stage_id=rank,
            num_stages=2,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
            activation_shape=(microbatch_size, seq_len, hidden_dim),
            device=device,
            world_size=2,
            transport="nccl",
        )
        input_batches = (
            [
                torch.randn(
                    microbatch_size,
                    seq_len,
                    hidden_dim,
                    dtype=torch.float16,
                )
                for _ in range(num_microbatches)
            ]
            if rank == 0
            else None
        )
        engine.run_1f1b_schedule(input_batches)
        engine.finish_transfers()
        inputs, actual, reference, parameter_count = _gather_pipeline_verification(
            rank=rank,
            world_size=2,
            stage=stage,
            engine=engine,
            input_batches=input_batches,
            hidden_dim=hidden_dim,
            microbatch_size=microbatch_size,
            seq_len=seq_len,
            num_microbatches=num_microbatches,
            device=device,
        )
        torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
        engine.close()

        # Exercise the exact ready/consumed P2P control primitives on real
        # process groups. This is sideband evidence only, not NVSHMEM payload
        # evidence; the latter remains a CUDA/NVSHMEM runtime requirement.
        ready_group = _create_pipeline_control_group(2, 5_000)
        consumed_group = _create_pipeline_control_group(2, 5_000)
        token = torch.tensor([rank + 1], dtype=torch.int32)
        received = torch.zeros_like(token)
        if rank == 0:
            dist.isend(token, dst=1, group=ready_group).wait()
            dist.recv(received, src=1, group=consumed_group)
            sideband_ok = int(received.item()) == 2
        else:
            dist.recv(received, src=0, group=ready_group)
            sideband_ok = int(received.item()) == 1
            dist.isend(token, dst=0, group=consumed_group).wait()

        try:
            _require_transport_consensus("nccl" if rank == 0 else "nvshmem", device)
        except RuntimeError as exc:
            mismatch = str(exc)
        else:
            raise AssertionError("Divergent rank transports unexpectedly passed consensus")

        os.environ["AISP_DISABLE_SYMMEM_PIPELINE"] = "1"
        try:
            _require_global_nvshmem(device)
        except RuntimeError as exc:
            unavailable = str(exc)
        else:
            raise AssertionError("Disabled NVSHMEM transport unexpectedly passed consensus")

        dist.barrier()
        if rank == 0:
            torch.save(
                {
                    "inputs": inputs,
                    "actual": actual,
                    "reference": reference,
                    "parameter_count": parameter_count,
                    "mismatch": mismatch,
                    "unavailable": unavailable,
                    "sideband_ok": sideband_ok,
                },
                output_path,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _join_spawn(context: Any, *, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if context.join(timeout=min(0.25, deadline - time.monotonic())):
            return
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=2.0)
    pytest.fail("Two-rank Gloo pipeline control exceeded its bounded timeout")


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_real_two_rank_gloo_schedule_matches_full_serial_pipeline(tmp_path: Path) -> None:
    rendezvous_path = tmp_path / "gloo-store"
    output_path = tmp_path / "result.pt"
    context = mp.spawn(
        _gloo_pipeline_worker,
        args=(str(rendezvous_path), str(output_path)),
        nprocs=2,
        join=False,
    )
    _join_spawn(context)

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["inputs"].shape == (4, 2, 3, 4)
    assert payload["actual"].shape == payload["reference"].shape
    torch.testing.assert_close(payload["actual"], payload["reference"], rtol=1e-3, atol=1e-3)
    assert payload["parameter_count"] > 0
    assert "selected different" in payload["mismatch"]
    assert "unavailable on at least one rank" in payload["unavailable"]
    assert payload["sideband_ok"] is True


class _LoggedTensor:
    def __init__(self, tensor: torch.Tensor, events: list[tuple[Any, ...]], label: str):
        self.tensor = tensor
        self.events = events
        self.label = label

    def copy_(self, other: torch.Tensor, non_blocking: bool = False) -> _LoggedTensor:
        self.events.append((self.label, "copy", non_blocking))
        self.tensor.copy_(other)
        return self

    def clone(self) -> torch.Tensor:
        self.events.append((self.label, "clone"))
        return self.tensor.clone()


class _SequencingHandle:
    events: list[tuple[Any, ...]] = []
    next_slot = 0

    def __init__(self, tensor: torch.Tensor):
        slot = type(self).next_slot
        type(self).next_slot += 1
        self.slot = slot
        self.world_size = 2
        self.buffer = _LoggedTensor(tensor, self.events, f"local-{slot}")
        self.remote = _LoggedTensor(torch.zeros_like(tensor), self.events, f"remote-{slot}")

    def get_buffer(self, rank: int) -> _LoggedTensor:
        self.events.append((self.slot, "get_buffer", rank))
        return self.remote

    def put_signal(self, rank: int, *, channel: int, timeout_ms: int) -> None:
        raise AssertionError("PyTorch 2.9 put_signal is a silent no-op")

    def wait_signal(self, rank: int, *, channel: int, timeout_ms: int) -> None:
        raise AssertionError("PyTorch 2.9 wait_signal is a silent no-op")


class _SequencingWork:
    def __init__(self, events: list[tuple[Any, ...]], label: str):
        self.events = events
        self.label = label

    def wait(self) -> None:
        self.events.append((self.label, "wait"))


def test_nvshmem_sequencing_model_waits_before_reuse_and_acks_after_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect API ordering only; this is not positive NVSHMEM runtime evidence."""
    import ch04.nvshmem_pipeline_parallel_multigpu as pipeline

    _SequencingHandle.events = []
    _SequencingHandle.next_slot = 0
    monkeypatch.setattr(pipeline, "SymmetricMemoryHandle", _SequencingHandle)
    groups = iter(("ready", "consumed"))
    monkeypatch.setattr(
        pipeline,
        "_create_pipeline_control_group",
        lambda _world_size, _timeout_ms: next(groups),
    )

    def fake_isend(
        _tensor: torch.Tensor, *, dst: int, group: str
    ) -> _SequencingWork:
        label = f"{group}-to-{dst}"
        _SequencingHandle.events.append((label, "isend"))
        return _SequencingWork(_SequencingHandle.events, label)

    def fake_recv(_tensor: torch.Tensor, *, src: int, group: str) -> None:
        _SequencingHandle.events.append((f"{group}-from-{src}", "recv"))

    monkeypatch.setattr(
        pipeline,
        "_complete_pipeline_stream",
        lambda _device: _SequencingHandle.events.append(("stream", "complete")),
    )
    monkeypatch.setattr(pipeline.dist, "isend", fake_isend)
    monkeypatch.setattr(pipeline.dist, "recv", fake_recv)
    channel = PipelineTransferBuffer(
        shape=(2,),
        dtype=torch.float32,
        device=torch.device("cpu"),
        world_size=2,
        transport="nvshmem",
        signal_timeout_ms=1234,
    )

    channel.send(torch.tensor([1.0, 2.0]), target_rank=1)
    channel.send(torch.tensor([3.0, 4.0]), target_rank=1)
    channel.send(torch.tensor([5.0, 6.0]), target_rank=1)
    channel.receive(source_rank=0)

    events = _SequencingHandle.events
    third_copy = [
        index
        for index, event in enumerate(events)
        if event == ("remote-0", "copy", True)
    ][1]
    ready_completion = events.index(("ready-to-1", "wait"))
    reuse_ack = events.index(("consumed-from-1", "recv"))
    first_copy = events.index(("remote-0", "copy", True))
    first_copy_fence = events.index(("stream", "complete"), first_copy)
    first_ready = events.index(("ready-to-1", "isend"))
    local_clone = events.index(("local-0", "clone"))
    consumed_ack = events.index(("consumed-to-0", "isend"))
    clone_fence = events.index(("stream", "complete"), local_clone)
    assert first_copy < first_copy_fence < first_ready
    assert ready_completion < reuse_ack < third_copy
    assert local_clone < clone_fence < consumed_ack


class _ResultConsumer(NVSHMEMPipelineChildResultMixin):
    pass


def _configuration(transport: str) -> dict[str, int | str]:
    return {
        "schedule": "1f1b",
        "batch_size": 4,
        "num_microbatches": 2,
        "microbatch_size": 2,
        "seq_len": 3,
        "hidden_dim": 2,
        "transport": transport,
    }


def test_pipeline_child_result_requires_fresh_full_rank_actual_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _ResultConsumer()
    configuration = _configuration("nccl")
    result_env = consumer.prepare_nvshmem_pipeline_child_result(
        variant="baseline",
        world_size=2,
        iterations=1,
        configuration=configuration,
    )
    result_dir = Path(result_env["AISP_NVSHMEM_PIPELINE_RESULT_DIR"])
    launch_wall_ns = time.time_ns()
    for name, value in result_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        NVSHMEM_PIPELINE_RESULT_LAUNCH_WALL_NS_ENV,
        str(launch_wall_ns),
    )
    inputs = torch.arange(24, dtype=torch.float16).reshape(2, 2, 3, 2)
    output = inputs + 1
    try:
        for rank in range(2):
            result = NVSHMEMPipelineWorkloadResult(
                rank=rank,
                world_size=2,
                iterations=1,
                time_per_iter_ms=2.5,
                transport="nccl",
                configuration=configuration,
                verify_inputs={"pipeline_inputs": inputs},
                verify_output=output,
                reference_output=output.clone(),
                batch_size=4,
                parameter_count=10,
                output_tolerance=(0.0, 0.0),
            )
            assert write_nvshmem_pipeline_child_result(result, variant="baseline")
        finish_wall_ns = time.time_ns()
        consumer.consume_nvshmem_pipeline_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=finish_wall_ns,
            returncode=0,
        )
        signature = consumer.get_input_signature()
        assert consumer.validate_result() is None
        torch.testing.assert_close(consumer.get_verify_output(), output, rtol=0, atol=0)
        assert signature.collective_algorithm == "1f1b-point-to-point"
        assert signature.async_completion_policy == "wait_for_async_before_timed_close"
        assert signature.pipeline_stage_boundaries == [(0, 0), (1, 1)]
        assert not result_dir.exists()
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def test_pipeline_child_result_rejects_incomplete_quorum() -> None:
    consumer = _ResultConsumer()
    result_env = consumer.prepare_nvshmem_pipeline_child_result(
        variant="optimized",
        world_size=2,
        iterations=1,
        configuration=_configuration("nvshmem"),
    )
    result_dir = Path(result_env["AISP_NVSHMEM_PIPELINE_RESULT_DIR"])
    try:
        with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
            consumer.consume_nvshmem_pipeline_child_results(
                launch_wall_ns=time.time_ns(),
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
        assert result_dir.exists()
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("module_name", "variant", "transport"),
    (
        ("ch04.baseline_nvshmem_pipeline_parallel_multigpu", "baseline", "nccl"),
        ("ch04.optimized_nvshmem_pipeline_parallel_multigpu", "optimized", "nvshmem"),
    ),
)
def test_pipeline_wrappers_launch_explicit_transport_and_fresh_result_callback(
    module_name: str,
    variant: str,
    transport: str,
) -> None:
    module = __import__(module_name, fromlist=["get_benchmark"])
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    config.nproc_per_node = 2
    config.nnodes = 1
    spec = benchmark.get_torchrun_spec(config)
    result_dir = Path(spec.env["AISP_NVSHMEM_PIPELINE_RESULT_DIR"])
    try:
        assert spec.module_name == "core.harness.benchmark_worker"
        assert spec.script_path is None
        assert spec.script_args[:5] == [
            "--module",
            "ch04.nvshmem_worker",
            "--callable",
            "main",
            "--",
        ]
        assert spec.script_args[5:9] == ["--workload", "pipeline", "--variant", variant]
        assert spec.script_args[-2:] == ["--transport", transport]
        assert spec.result_callback == NVSHMEM_PIPELINE_RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == 1
        assert spec.env["AISP_DISABLE_SYMMEM_PIPELINE"] == (
            "1" if transport == "nccl" else "0"
        )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def test_pipeline_result_schema_is_dedicated() -> None:
    assert NVSHMEM_PIPELINE_RESULT_SCHEMA == "aisp.nvshmem.pipeline-result.v1"
