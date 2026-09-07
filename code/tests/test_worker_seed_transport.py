"""Controls for parent-to-worker benchmark seed transport."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.utils.worker_seed import apply_worker_seed
from tests.protection_test_utils import preserve_rng_state

CODE_ROOT = Path(__file__).resolve().parents[1]
WORKERS = {
    "ch04/baseline_pipeline_parallel.py": "_run_worker",
    "ch04/baseline_pipeline_parallel_multigpu.py": "_run_worker",
    "ch04/baseline_tensor_parallel.py": "_run_worker",
    "ch04/baseline_tensor_parallel_allgather_multigpu.py": "_run_worker",
    "ch04/baseline_tensor_parallel_multigpu.py": "_run_worker",
    "ch04/baseline_torchcomms.py": "_run_worker",
    "ch04/baseline_torchcomms_multigpu.py": "_run_worker",
    "ch04/optimized_pipeline_parallel_1f1b.py": "_run_worker",
    "ch04/optimized_pipeline_parallel_multigpu_1f1b.py": "_run_worker",
    "ch04/optimized_tensor_parallel_allgather_multigpu.py": "_run_worker",
    "ch04/optimized_tensor_parallel_async.py": "_run_worker",
    "ch04/optimized_tensor_parallel_multigpu.py": "_run_worker",
    "ch04/optimized_torchcomms.py": "_run_worker",
    "ch04/optimized_torchcomms_multigpu.py": "_run_worker",
    "ch15/baseline_disaggregated_inference_multigpu.py": "_run_torchrun_worker",
}
WORKER_ADAPTERS = {
    "ch15/optimized_disaggregated_inference_multigpu.py": "_run_torchrun_worker",
}


def _call_name(call: ast.Call) -> str:
    node = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_flag(call: ast.Call, flag: str) -> bool:
    return any(isinstance(argument, ast.Constant) and argument.value == flag for argument in call.args)


def _distributed_seed_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    seed: int,
    result_path: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        apply_worker_seed(seed)
        sample = torch.randn(16)
        gathered = [torch.empty_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered, sample)
        if rank == 0:
            Path(result_path).write_text(
                json.dumps(
                    {
                        "initial_seed": int(torch.initial_seed()),
                        "samples": [tensor.tolist() for tensor in gathered],
                    }
                ),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


def _expected_sample(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(16, generator=generator)


def _run_two_rank_sample(tmp_path: Path, seed: int) -> dict[str, object]:
    rendezvous_path = tmp_path / f"seed-{seed}.rdzv"
    result_path = tmp_path / f"seed-{seed}.json"
    mp.spawn(
        _distributed_seed_worker,
        args=(2, rendezvous_path.as_uri(), seed, str(result_path)),
        nprocs=2,
        join=True,
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_apply_worker_seed_preserves_legacy_42_and_honors_fresh_seed() -> None:
    with preserve_rng_state():
        torch.manual_seed(42)
        legacy = torch.randn(16)

        apply_worker_seed(42)
        transported_42 = torch.randn(16)
        observed_42 = int(torch.initial_seed())

        apply_worker_seed(1042)
        transported_1042 = torch.randn(16)
        observed_1042 = int(torch.initial_seed())

    assert torch.equal(transported_42, legacy)
    assert torch.equal(transported_42, _expected_sample(42))
    assert observed_42 == 42
    assert observed_1042 == 1042
    assert torch.equal(transported_1042, _expected_sample(1042))
    assert not torch.equal(transported_42, transported_1042)


def test_two_real_gloo_workers_use_the_transported_seed(tmp_path: Path) -> None:
    run_42 = _run_two_rank_sample(tmp_path, 42)
    run_1042 = _run_two_rank_sample(tmp_path, 1042)
    samples_42 = [torch.tensor(sample) for sample in run_42["samples"]]
    samples_1042 = [torch.tensor(sample) for sample in run_1042["samples"]]

    assert run_42["initial_seed"] == 42
    assert run_1042["initial_seed"] == 1042
    assert all(torch.equal(sample, _expected_sample(42)) for sample in samples_42)
    assert all(torch.equal(sample, _expected_sample(1042)) for sample in samples_1042)
    assert torch.equal(samples_42[0], samples_42[1])
    assert torch.equal(samples_1042[0], samples_1042[1])
    assert not torch.equal(samples_42[0], samples_1042[0])


def test_all_worker_launchers_transport_config_seed_to_worker() -> None:
    literal_seed_calls = {
        "torch.manual_seed",
        "torch.cuda.manual_seed",
        "torch.cuda.manual_seed_all",
    }
    for relative_path, worker_name in WORKERS.items():
        source = (CODE_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        worker = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == worker_name
        )
        worker_arguments = {
            argument.arg
            for argument in (*worker.args.posonlyargs, *worker.args.args, *worker.args.kwonlyargs)
        }
        assert "seed" in worker_arguments
        helper_calls = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call) and _call_name(node) == "apply_worker_seed"
        ]
        assert len(helper_calls) == 1
        assert ast.unparse(helper_calls[0].args[0]) == "seed"
        assert not any(
            isinstance(node, ast.Call)
            and _call_name(node) in literal_seed_calls
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 42
            for node in ast.walk(worker)
        )

        seed_parser_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _call_name(node).endswith("add_argument")
            and _is_flag(node, "--seed")
        ]
        assert len(seed_parser_calls) == 1
        parser_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in seed_parser_calls[0].keywords}
        assert parser_keywords["type"] == "int"
        assert parser_keywords["default"] == "42"

        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_worker_call = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and _call_name(node).endswith(worker_name)
        )
        main_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in main_worker_call.keywords}
        assert main_keywords["seed"] == "args.seed"

        config_maps = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "config_arg_map"
            and isinstance(node.value, ast.Dict)
        ]
        assert len(config_maps) == 1
        config_map = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(config_maps[0].keys, config_maps[0].values, strict=True)
        }
        assert config_map["seed"] == "--seed"


def test_imported_worker_adapters_forward_cli_seed() -> None:
    for relative_path, worker_name in WORKER_ADAPTERS.items():
        source = (CODE_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        seed_parser_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _call_name(node).endswith("add_argument")
            and _is_flag(node, "--seed")
        ]
        assert len(seed_parser_calls) == 1
        parser_keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in seed_parser_calls[0].keywords
        }
        assert parser_keywords["type"] == "int"
        assert parser_keywords["default"] == "42"

        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_worker_call = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and _call_name(node).endswith(worker_name)
        )
        main_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in main_worker_call.keywords}
        assert main_keywords["seed"] == "args.seed"


def test_disaggregated_adapters_parse_actual_cli_seed() -> None:
    modules = (
        "ch15.baseline_disaggregated_inference_multigpu",
        "ch15.optimized_disaggregated_inference_multigpu",
    )
    for module_name in modules:
        script = (
            "import json, sys; "
            f"from {module_name} import _parse_args; "
            "sys.argv = ['worker', '--seed', '1042']; "
            "print(json.dumps(vars(_parse_args()), sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=CODE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout)["seed"] == 1042
