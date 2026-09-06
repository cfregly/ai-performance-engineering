from __future__ import annotations

import ast
from pathlib import Path


def _symmetric_ring_function() -> tuple[ast.FunctionDef, str]:
    source_path = (
        Path(__file__).resolve().parents[1] / "ch04" / "symmetric_memory_example.py"
    )
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark_symmetric_ring"
    )
    return function, ast.get_source_segment(source, function) or ""


def test_symmetric_ring_cuda_branch_uses_two_device_fences() -> None:
    function, source = _symmetric_ring_function()
    barrier_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "handle"
        and node.func.attr == "barrier"
    ]

    assert len(barrier_calls) == 4
    assert [
        (ast.literal_eval(call.keywords[0].value), ast.literal_eval(call.keywords[1].value))
        for call in barrier_calls
    ] == [(0, 5000), (1, 5000), (0, 5000), (1, 5000)]
    assert source.count("next_buf[buf_idx].copy_(flat, non_blocking=True)") == 4
    assert source.count("recv_tensor.copy_(local[buf_idx], non_blocking=True)") == 4
    assert source.count("dist.barrier()") == 4
    assert "torch.cuda.current_stream().synchronize()" not in source
    assert source.count("torch.cuda.synchronize(device)") == 1


def test_symmetric_ring_timed_iteration_preserves_publish_consume_order() -> None:
    function, _source = _symmetric_ring_function()
    timed_with = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "nvtx_range"
            for item in node.items
        )
    )
    backend_branch = next(node for node in timed_with.body if isinstance(node, ast.If))
    measured_loop = next(node for node in backend_branch.body if isinstance(node, ast.For))
    operations = [
        ast.unparse(statement.value.func)
        for statement in measured_loop.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]

    assert operations == [
        "next_buf[buf_idx].copy_",
        "handle.barrier",
        "recv_tensor.copy_",
        "handle.barrier",
    ]

    channels = [
        ast.literal_eval(statement.value.keywords[0].value)
        for statement in measured_loop.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "handle.barrier"
    ]
    assert channels == [0, 1]

    legacy_loop = next(
        node for node in backend_branch.orelse if isinstance(node, ast.For)
    )
    legacy_operations = [
        ast.unparse(statement.value.func)
        for statement in legacy_loop.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    assert legacy_operations == [
        "next_buf[buf_idx].copy_",
        "dist.barrier",
        "recv_tensor.copy_",
        "dist.barrier",
    ]


def test_symmetric_ring_default_remains_nvshmem_and_worker_requests_cuda() -> None:
    function, _source = _symmetric_ring_function()
    defaults = function.args.kw_defaults
    backend_index = [argument.arg for argument in function.args.kwonlyargs].index("backend")

    assert ast.literal_eval(defaults[backend_index]) == "NVSHMEM"

    module_source = (
        Path(__file__).resolve().parents[1] / "ch04" / "symmetric_memory_example.py"
    ).read_text(encoding="utf-8")
    assert "backend=SYMMETRIC_RING_TRANSPORT_BACKEND" in module_source
