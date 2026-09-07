"""Regression coverage for parent/worker RNG ordering in torchrun launches."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import torch

from core.harness.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    LaunchVia,
    TorchrunLaunchSpec,
)
from tests.protection_test_utils import preserve_rng_state


class _RngReferenceTarget:
    name = "torchrun_parent_worker_rng_reference"

    def __init__(self, script_path: Path, output_path: Path) -> None:
        self.script_path = script_path
        self.output_path = output_path
        self.parent_reference: torch.Tensor | None = None

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        # Chapter torchrun adapters construct their verification reference here.
        # This draw must therefore happen after the harness applies config.seed.
        self.parent_reference = torch.randn(32)
        return TorchrunLaunchSpec(
            script_path=self.script_path,
            script_args=[str(self.output_path)],
            config_arg_map={},
            multi_gpu_required=False,
            name=self.name,
        )


def _free_rendezvous_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def _run_seeded_reference(
    script_path: Path,
    output_path: Path,
    *,
    seed: int,
    unrelated_draws: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(seed + 9_000)
    torch.randn(unrelated_draws)

    config = BenchmarkConfig(
        device=torch.device("cpu"),
        seed=seed,
        iterations=1,
        warmup=5,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        measurement_timeout_seconds=60,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        enforce_environment_validation=False,
        rdzv_endpoint=_free_rendezvous_endpoint(),
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    target = _RngReferenceTarget(script_path, output_path)

    result = harness._benchmark_with_torchrun(target, config)

    assert not result.errors, result.errors
    assert target.parent_reference is not None
    child_payload = json.loads(output_path.read_text(encoding="utf-8"))
    return (
        target.parent_reference,
        torch.tensor(child_payload["sample"], dtype=torch.float32),
        int(child_payload["initial_seed"]),
    )


def test_torchrun_spec_reference_uses_active_seed_before_worker_launch(tmp_path: Path) -> None:
    script_path = tmp_path / "rng_tensor_worker.py"
    script_path.write_text(
        """import json
from pathlib import Path
import sys
import torch

sample = torch.randn(32)
Path(sys.argv[1]).write_text(
    json.dumps({"initial_seed": int(torch.initial_seed()), "sample": sample.tolist()}),
    encoding="utf-8",
)
print("rank0 RNG tensor emitted", flush=True)
""",
        encoding="utf-8",
    )

    with preserve_rng_state():
        parent_42, child_42, child_seed_42 = _run_seeded_reference(
            script_path,
            tmp_path / "seed-42.json",
            seed=42,
            unrelated_draws=7,
        )
        parent_1042, child_1042, child_seed_1042 = _run_seeded_reference(
            script_path,
            tmp_path / "seed-1042.json",
            seed=1042,
            unrelated_draws=113,
        )

    expected_42 = torch.randn(32, generator=torch.Generator().manual_seed(42))
    expected_1042 = torch.randn(32, generator=torch.Generator().manual_seed(1042))
    assert child_seed_42 == 42
    assert child_seed_1042 == 1042
    torch.testing.assert_close(parent_42, child_42, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parent_1042, child_1042, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parent_42, expected_42, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parent_1042, expected_1042, rtol=0.0, atol=0.0)
    assert not torch.equal(parent_42, parent_1042)
