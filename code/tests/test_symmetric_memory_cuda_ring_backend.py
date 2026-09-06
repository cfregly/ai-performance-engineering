from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ch04 import symmetric_memory_example
from core.harness.benchmark_harness import BenchmarkConfig
from core.optimization import symmetric_memory_patch


class _FakeTensor:
    is_cuda = True
    device = torch.device("cuda", 0)
    shape = (2, 4)
    dtype = torch.float32

    def numel(self) -> int:
        return 8

    def copy_(self, _other: object) -> _FakeTensor:
        return self


class _FakeRawHandle:
    def __init__(self, buffer: _FakeTensor) -> None:
        self._buffer = buffer

    def get_buffer(
        self,
        _rank: int,
        _shape: tuple[int, ...],
        _dtype: torch.dtype,
    ) -> _FakeTensor:
        return self._buffer

    def barrier(self, *, channel: int = 0, timeout_ms: int = 0) -> None:
        del channel, timeout_ms


class _FakeSymmetricMemoryModule:
    def __init__(
        self,
        *,
        observed_backend: str = "CUDA",
        update_backend_on_set: bool = True,
        reject_redundant_set: bool = False,
    ) -> None:
        self.observed_backend = observed_backend
        self.update_backend_on_set = update_backend_on_set
        self.reject_redundant_set = reject_redundant_set
        self.calls: list[tuple[str, Any]] = []
        self.tensor = _FakeTensor()
        self.handle = _FakeRawHandle(self.tensor)

    def set_backend(self, backend: str) -> None:
        self.calls.append(("set_backend", backend))
        if self.reject_redundant_set and backend == self.observed_backend:
            raise RuntimeError("redundant setter is forbidden after allocation")
        if self.update_backend_on_set:
            self.observed_backend = backend

    def get_backend(self, device: torch.device) -> str:
        self.calls.append(("get_backend", device))
        return self.observed_backend

    def empty(self, shape: tuple[int, ...], **_kwargs: object) -> _FakeTensor:
        self.calls.append(("empty", shape))
        return self.tensor

    def rendezvous(self, _tensor: _FakeTensor, group: object) -> _FakeRawHandle:
        self.calls.append(("rendezvous", group))
        return self.handle

    def is_nvshmem_available(self) -> bool:
        return True


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module: _FakeSymmetricMemoryModule,
) -> object:
    group = object()
    monkeypatch.setattr(symmetric_memory_patch, "_symm_mem", module)
    monkeypatch.setattr(symmetric_memory_patch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        symmetric_memory_patch,
        "dist",
        SimpleNamespace(
            get_rank=lambda *, group: 0,
            get_world_size=lambda *, group: 2,
        ),
    )
    return group


def test_cuda_backend_is_selected_and_observed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeSymmetricMemoryModule(observed_backend="NVSHMEM")
    group = _install_fake_runtime(monkeypatch, module)

    handle = symmetric_memory_patch.create_symmetric_memory_handle(
        _FakeTensor(),
        group=group,
        backend="CUDA",
    )

    assert handle.backend == "CUDA"
    assert [name for name, _value in module.calls] == [
        "get_backend",
        "set_backend",
        "get_backend",
        "empty",
        "rendezvous",
    ]
    assert module.calls[1] == ("set_backend", "CUDA")


def test_cuda_backend_identity_mismatch_fails_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeSymmetricMemoryModule(
        observed_backend="NVSHMEM",
        update_backend_on_set=False,
    )
    group = _install_fake_runtime(monkeypatch, module)

    with pytest.raises(RuntimeError, match="backend identity mismatch"):
        symmetric_memory_patch.create_symmetric_memory_handle(
            _FakeTensor(),
            group=group,
            backend="CUDA",
        )

    assert [name for name, _value in module.calls] == [
        "get_backend",
        "set_backend",
        "get_backend",
    ]


def test_default_backend_remains_nvshmem(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _FakeSymmetricMemoryModule(observed_backend="NVSHMEM")
    group = _install_fake_runtime(monkeypatch, module)

    handle = symmetric_memory_patch.create_symmetric_memory_handle(
        _FakeTensor(),
        group=group,
    )

    assert handle.backend == "NVSHMEM"
    assert all(name != "set_backend" for name, _value in module.calls)


def test_repeated_cuda_handles_do_not_repeat_global_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeSymmetricMemoryModule(
        observed_backend="NVSHMEM",
        reject_redundant_set=True,
    )
    group = _install_fake_runtime(monkeypatch, module)

    first = symmetric_memory_patch.create_symmetric_memory_handle(
        _FakeTensor(),
        group=group,
        backend="CUDA",
    )
    second = symmetric_memory_patch.create_symmetric_memory_handle(
        _FakeTensor(),
        group=group,
        backend="CUDA",
    )

    assert first.backend == second.backend == "CUDA"
    assert [call for call in module.calls if call[0] == "set_backend"] == [
        ("set_backend", "CUDA")
    ]


@pytest.mark.parametrize(
    ("module_name", "expected_backend"),
    (
        ("ch04.baseline_symmetric_memory_multigpu", "NCCL"),
        ("ch04.optimized_symmetric_memory_multigpu", "CUDA"),
    ),
)
def test_ring_receipt_binds_requested_and_observed_transport(
    module_name: str,
    expected_backend: str,
) -> None:
    import importlib

    benchmark = importlib.import_module(module_name).get_benchmark()
    spec = benchmark.get_torchrun_spec(BenchmarkConfig(nproc_per_node=2, nnodes=1))
    result_dir = Path(spec.env["AISP_NVSHMEM_CHILD_RESULT_DIR"])
    try:
        context = benchmark._nvshmem_child_result_context
        assert context is not None
        assert context["configuration"]["process_group_backend"] == "NCCL"
        assert context["configuration"]["requested_transport_backend"] == expected_backend
        assert context["configuration"]["observed_transport_backend"] == expected_backend
        assert spec.script_args[4] == "--"
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def test_baseline_setup_has_no_symmetric_memory_capability_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ch04 import baseline_symmetric_memory_multigpu as baseline

    monkeypatch.setattr(baseline.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(baseline.torch.cuda, "manual_seed_all", lambda _seed: None)
    benchmark = baseline.get_benchmark()

    benchmark.setup()

    assert benchmark._benchmark_ready is True


def test_ring_launch_requires_one_node_world_and_nccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("GROUP_RANK", "0")
    monkeypatch.setattr(
        symmetric_memory_example.dist,
        "get_backend",
        lambda _group: "nccl",
    )

    assert symmetric_memory_example._require_single_node_nccl_world(2) == "NCCL"


@pytest.mark.parametrize(
    ("local_world_size", "group_rank", "backend", "message"),
    (
        ("1", "0", "nccl", "one WORLD group on one node"),
        ("2", "1", "nccl", "one WORLD group on one node"),
        ("2", "0", "gloo", "requires an NCCL process group"),
    ),
)
def test_ring_launch_rejects_wrong_scope_or_control_backend(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: str,
    group_rank: str,
    backend: str,
    message: str,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
    monkeypatch.setenv("GROUP_RANK", group_rank)
    monkeypatch.setattr(
        symmetric_memory_example.dist,
        "get_backend",
        lambda _group: backend,
    )

    with pytest.raises(RuntimeError, match=message):
        symmetric_memory_example._require_single_node_nccl_world(2)


def test_cuda_ring_records_full_iteration_topology_and_replays_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Any, ...]] = []

    class ControlTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def flatten(self) -> ControlTensor:
            return self

        def detach(self) -> ControlTensor:
            return self

        def clone(self) -> ControlTensor:
            calls.append(("clone", self.name))
            return ControlTensor(f"{self.name}-snapshot")

        def copy_(
            self,
            source: ControlTensor,
            *,
            non_blocking: bool = False,
        ) -> ControlTensor:
            calls.append(("copy", self.name, source.name, non_blocking))
            return self

    class IndexedBuffer:
        def __init__(self, prefix: str) -> None:
            self.slots = [ControlTensor(f"{prefix}-{index}") for index in range(2)]

        def __getitem__(self, index: int) -> ControlTensor:
            return self.slots[index]

    class ControlHandle:
        backend = "CUDA"

        def __init__(self) -> None:
            self.buffer = IndexedBuffer("local")
            self.peer = IndexedBuffer("peer")

        def get_buffer(self, rank: int) -> IndexedBuffer:
            calls.append(("get_buffer", rank))
            return self.peer

        def barrier(self, *, channel: int, timeout_ms: int) -> None:
            calls.append(("barrier", channel, timeout_ms))

    class ControlGraph:
        def __init__(self) -> None:
            calls.append(("graph_create",))

        def replay(self) -> None:
            calls.append(("graph_replay",))

    class GraphCapture:
        def __enter__(self) -> None:
            calls.append(("graph_capture_enter",))

        def __exit__(self, *exc: object) -> None:
            calls.append(("graph_capture_exit", *exc))

    class ControlEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.index = sum(call[0] == "event_create" for call in calls)
            calls.append(("event_create", self.index))

        def record(self) -> None:
            calls.append(("event_record", self.index))

        def elapsed_time(self, other: ControlEvent) -> float:
            calls.append(("elapsed_time", self.index, other.index))
            return 80.0

    class NvtxRange:
        def __enter__(self) -> None:
            calls.append(("nvtx_enter",))

        def __exit__(self, *exc: object) -> None:
            calls.append(("nvtx_exit", *exc))

    handle = ControlHandle()
    input_tensor = ControlTensor("input")
    monkeypatch.setattr(symmetric_memory_example.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(symmetric_memory_example.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        symmetric_memory_example.dist,
        "barrier",
        lambda: pytest.fail("CUDA symmetric ring must not use a host barrier"),
    )
    monkeypatch.setattr(
        symmetric_memory_example.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        symmetric_memory_example.torch.cuda,
        "synchronize",
        lambda _device: calls.append(("synchronize",)),
    )
    monkeypatch.setattr(symmetric_memory_example.torch.cuda, "CUDAGraph", ControlGraph)
    monkeypatch.setattr(
        symmetric_memory_example.torch.cuda,
        "graph",
        lambda _graph: GraphCapture(),
    )
    monkeypatch.setattr(symmetric_memory_example.torch.cuda, "Event", ControlEvent)
    monkeypatch.setattr(
        symmetric_memory_example.torch,
        "stack",
        lambda _values: ControlTensor("stacked-seed"),
    )
    monkeypatch.setattr(
        symmetric_memory_example.torch,
        "empty_like",
        lambda _value: pytest.fail("CUDA ring must use its symmetric receive slot"),
    )
    monkeypatch.setattr(
        symmetric_memory_example,
        "create_symmetric_memory_handle",
        lambda *_args, **_kwargs: handle,
    )
    monkeypatch.setattr(
        symmetric_memory_example,
        "nvtx_range",
        lambda *_args, **_kwargs: NvtxRange(),
    )

    measured = symmetric_memory_example.benchmark_symmetric_ring(
        input_tensor,
        iterations=4,
        return_output=True,
        backend="CUDA",
    )

    # These fakes record Python dispatch only. Replay is intentionally opaque:
    # this CPU control does not simulate execution of captured CUDA operations.
    assert isinstance(measured, tuple)
    elapsed, output = measured
    assert elapsed == 20.0
    assert output.name == "local-1-snapshot"
    assert calls.count(("graph_replay",)) == 1
    assert [call[1] for call in calls if call[0] == "barrier"] == [0, 1] * 9
    assert len([call for call in calls if call[:2] == ("copy", "peer-0")]) == 5
    assert len([call for call in calls if call[:2] == ("copy", "peer-1")]) == 4
    capture_exit = next(
        index for index, call in enumerate(calls) if call[0] == "graph_capture_exit"
    )
    capture_enter = calls.index(("graph_capture_enter",))
    assert calls.index(("synchronize",)) < capture_enter
    captured_dispatches = [
        call[0:2]
        for call in calls[capture_enter + 1 : capture_exit]
    ]
    assert captured_dispatches == [
        ("copy", "peer-0"),
        ("barrier", 0),
        ("barrier", 1),
        ("copy", "peer-1"),
        ("barrier", 0),
        ("barrier", 1),
    ] * 2
    assert not any(call[:2] == ("copy", "input") for call in calls)
    start = calls.index(("event_record", 0))
    replay = calls.index(("graph_replay",))
    end = calls.index(("event_record", 1))
    output_clone = calls.index(("clone", "local-1"))
    assert capture_exit < start < replay < end < output_clone
    assert "symmetric_ring_cuda_graph_capture_wall_ms" in capsys.readouterr().out
