"""Compatibility contracts for Chapter 4 distributed collectives."""

from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (CODE_ROOT / "ch04" / name).read_text(encoding="utf-8")


def test_gradient_compression_uses_public_out_of_place_all_reduce() -> None:
    source = _source("gradient_compression_multigpu_worker.py")
    collective = source.split("def _collective", maxsplit=1)[1].split("def _complete", maxsplit=1)[
        0
    ]

    assert "torch.distributed._functional_collectives" not in source
    assert "result = value.clone()" in collective
    assert "work = dist.all_reduce(" in collective
    assert "async_op=True" in collective


def test_optimized_torchcomms_uses_public_async_all_reduce() -> None:
    source = _source("optimized_torchcomms.py")
    step = source.split("def _step", maxsplit=1)[1].split("for _ in range(max(warmup", maxsplit=1)[
        0
    ]

    assert "torch.distributed._functional_collectives" not in source
    assert "work = dist.all_reduce(" in step
    assert "op=dist.ReduceOp.AVG" in step
    assert "async_op=True" in step
    assert step.index("aux_out = aux_block.forward_prepared(inputs)") < step.index("work.wait()")
