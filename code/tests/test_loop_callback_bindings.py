from __future__ import annotations

import ast
import threading
from collections import Counter
from pathlib import Path

from ch17.early_rejection import Priority, Request, _schedule_completion

REPO_ROOT = Path(__file__).resolve().parent.parent


def _function_node(source: str, function_name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_ch04_collective_callbacks_bind_each_loop_operand() -> None:
    source = (REPO_ROOT / "ch04" / "multi_node_blackwell.py").read_text(encoding="utf-8")
    callbacks = [
        call.args[0]
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_bench"
        and call.args
    ]

    assert all(isinstance(callback, ast.Lambda) for callback in callbacks)
    lambda_callbacks = [callback for callback in callbacks if isinstance(callback, ast.Lambda)]
    bound_names = Counter(
        tuple(argument.arg for argument in callback.args.args) for callback in lambda_callbacks
    )
    assert bound_names == Counter(
        {
            ("tensor",): 3,
            ("out", "tensor"): 1,
            ("output", "reducescatter_input"): 1,
        }
    )

    loop_operands = {"tensor", "out", "output", "reducescatter_input"}
    for callback in lambda_callbacks:
        defaults_by_name = dict(
            zip(
                (argument.arg for argument in callback.args.args[-len(callback.args.defaults) :]),
                callback.args.defaults,
                strict=True,
            )
        )
        referenced_operands = {
            node.id
            for node in ast.walk(callback.body)
            if isinstance(node, ast.Name) and node.id in loop_operands
        }
        assert referenced_operands == defaults_by_name.keys()
        assert all(
            isinstance(default, ast.Name) and default.id == name
            for name, default in defaults_by_name.items()
        )


def test_ch04_symmetric_memory_default_is_created_inside_each_call() -> None:
    source = (REPO_ROOT / "ch04" / "symmetric_memory_example.py").read_text(encoding="utf-8")
    function = _function_node(source, "benchmark_multigpu_symmetric_memory")
    defaults_by_name = dict(
        zip(
            (argument.arg for argument in function.args.args[-len(function.args.defaults) :]),
            function.args.defaults,
            strict=True,
        )
    )

    tensor_sizes_default = defaults_by_name["tensor_sizes"]
    assert isinstance(tensor_sizes_default, ast.Constant)
    assert tensor_sizes_default.value is None
    assert (
        "if tensor_sizes is None:\n"
        "        tensor_sizes = [(1024,), (1024 * 256,), (1024 * 1024,)]" in source
    )


class _CompletionRecorder:
    def __init__(self) -> None:
        self.completed: list[tuple[Request, float, float]] = []
        self.lock = threading.Lock()

    def complete_request(
        self,
        request: Request,
        actual_ttft: float,
        actual_tpot: float,
    ) -> None:
        with self.lock:
            self.completed.append((request, actual_ttft, actual_tpot))


def test_ch17_deferred_completion_keeps_each_submitted_request() -> None:
    recorder = _CompletionRecorder()
    timers: list[threading.Timer] = []

    for index in range(3):
        request = Request(
            id=f"request-{index}",
            prompt_length=16,
            expected_output_length=8,
            priority=Priority.STANDARD,
            arrival_time=0.0,
        )
        timers.append(
            _schedule_completion(
                recorder,  # type: ignore[arg-type]
                request,
                actual_ttft=100.0 + index,
                actual_tpot=10.0 + index,
                delay_seconds=0.01,
            )
        )

    for timer in timers:
        timer.join(timeout=1.0)

    assert all(timer.daemon and not timer.is_alive() for timer in timers)
    assert sorted(
        (request.id, actual_ttft, actual_tpot)
        for request, actual_ttft, actual_tpot in recorder.completed
    ) == [
        ("request-0", 100.0, 10.0),
        ("request-1", 101.0, 11.0),
        ("request-2", 102.0, 12.0),
    ]

    source = (REPO_ROOT / "ch17" / "early_rejection.py").read_text(encoding="utf-8")
    scenario_loop = source.split("for scenario in scenarios:", maxsplit=1)[1].split(
        "# Print stats for this scenario",
        maxsplit=1,
    )[0]
    assert "_schedule_completion(qos, request, actual_ttft, actual_tpot)" in scenario_loop
    assert "def complete_later" not in scenario_loop
