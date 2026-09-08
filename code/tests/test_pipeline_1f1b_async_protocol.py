from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import ch04.pipeline_parallel_common as pipeline


@dataclass
class _FakeP2POp:
    kind: str
    tensor: torch.Tensor
    peer: int


class _FakeWork:
    def __init__(
        self,
        transport: _FakeTransport,
        operation: _FakeP2POp,
        group_index: int,
        operation_index: int,
    ) -> None:
        self.transport = transport
        self.kind = operation.kind
        self.peer = operation.peer
        self.group_index = group_index
        self.operation_index = operation_index
        self.tensor_ref = weakref.ref(operation.tensor)
        self.wait_calls = 0

    def wait(self) -> bool:
        self.wait_calls += 1
        self.transport.events.append(
            f"wait:{self.group_index}:{self.operation_index}:{self.kind}"
        )
        tensor = self.tensor_ref()
        if tensor is None:
            raise RuntimeError("P2P tensor owner was released before Work completion")
        if self.kind == "recv":
            tensor.fill_(float(self.transport.next_recv_value))
            self.transport.next_recv_value += 1
        if (self.group_index, self.operation_index) in self.transport.wait_failures:
            raise RuntimeError("injected P2P wait failure")
        return True


class _FakeTransport:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.groups: list[list[_FakeWork]] = []
        self.wait_failures: set[tuple[int, int]] = set()
        self.next_recv_value = 100
        self.isend_op = object()
        self.irecv_op = object()

    def p2p_op(
        self,
        operation: Any,
        tensor: torch.Tensor,
        peer: int,
    ) -> _FakeP2POp:
        if operation is self.isend_op:
            kind = "send"
        elif operation is self.irecv_op:
            kind = "recv"
        else:
            raise AssertionError("unexpected P2P operation")
        return _FakeP2POp(kind=kind, tensor=tensor, peer=peer)

    def batch(self, operations: list[_FakeP2POp]) -> list[_FakeWork]:
        group_index = len(self.groups)
        self.events.append(
            f"batch:{group_index}:" + ",".join(operation.kind for operation in operations)
        )
        works = [
            _FakeWork(self, operation, group_index, operation_index)
            for operation_index, operation in enumerate(operations)
        ]
        self.groups.append(works)
        return works


def _install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: _FakeTransport,
) -> None:
    monkeypatch.setattr(pipeline.dist, "isend", transport.isend_op)
    monkeypatch.setattr(pipeline.dist, "irecv", transport.irecv_op)
    monkeypatch.setattr(pipeline.dist, "P2POp", transport.p2p_op)
    monkeypatch.setattr(pipeline.dist, "batch_isend_irecv", transport.batch)

    def reject_blocking_p2p(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("optimized 1F1B must not use unbatched blocking P2P")

    monkeypatch.setattr(pipeline.dist, "send", reject_blocking_p2p)
    monkeypatch.setattr(pipeline.dist, "recv", reject_blocking_p2p)


def _run_rank_zero(
    transport: _FakeTransport,
    *,
    fail_forward_index: int | None = None,
) -> tuple[list[int], list[int], pipeline.PipelineIterationCapture]:
    forward_indices: list[int] = []
    backward_values: list[int] = []
    capture = pipeline.PipelineIterationCapture.create(3)

    def get_microbatch(index: int) -> torch.Tensor:
        transport.events.append(f"get:{index}")
        return torch.tensor([float(index)])

    def forward_step(value: torch.Tensor) -> torch.Tensor:
        index = int(value.item())
        transport.events.append(f"forward:{index}")
        forward_indices.append(index)
        if index == fail_forward_index:
            raise RuntimeError("injected forward failure")
        return value + 10.0

    def backward_step(_activation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        incoming = int(value.item())
        transport.events.append(f"backward:{incoming}")
        backward_values.append(incoming)
        return value + 100.0

    pipeline.run_1f1b_iteration(
        rank=0,
        world_size=2,
        num_micro_batches=3,
        get_rank0_microbatch=get_microbatch,
        recv_forward_buffers=[],
        recv_backward_buffers=[torch.empty(1) for _ in range(3)],
        forward_step=forward_step,
        backward_step=backward_step,
        activation_slots=[None],
        capture=capture,
    )
    return forward_indices, backward_values, capture


def test_rank_zero_lookahead_precedes_wait_and_drains_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport()
    _install_fake_transport(monkeypatch, transport)

    forward_indices, backward_values, capture = _run_rank_zero(transport)

    first_exchange = transport.events.index("batch:1:send,recv")
    lookahead = transport.events.index("forward:2")
    first_wait = transport.events.index("wait:1:0:send")
    assert first_exchange < lookahead < first_wait
    assert forward_indices == [0, 1, 2]
    assert backward_values == [100, 101, 102]
    assert [len(group) for group in transport.groups] == [1, 2, 2, 1]
    assert all(work.wait_calls == 1 for group in transport.groups for work in group)

    forward_inputs, backward_inputs, backward_outputs = capture.concatenate()
    torch.testing.assert_close(
        forward_inputs,
        torch.tensor([0.0, 1.0, 2.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        backward_inputs,
        torch.tensor([100.0, 101.0, 102.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        backward_outputs,
        torch.tensor([200.0, 201.0, 202.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_lookahead_failure_still_drains_every_posted_exchange_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport()
    _install_fake_transport(monkeypatch, transport)

    with pytest.raises(RuntimeError, match="injected forward failure"):
        _run_rank_zero(transport, fail_forward_index=2)

    assert len(transport.groups) == 2
    assert all(work.wait_calls == 1 for work in transport.groups[-1])


def test_wait_failure_does_not_abandon_the_other_exchange_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport()
    transport.wait_failures.add((1, 0))
    _install_fake_transport(monkeypatch, transport)

    with pytest.raises(RuntimeError, match="injected P2P wait failure"):
        _run_rank_zero(transport)

    assert len(transport.groups) == 2
    assert [work.wait_calls for work in transport.groups[-1]] == [1, 1]
