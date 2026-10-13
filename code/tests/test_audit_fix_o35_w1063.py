"""W1-063: torchrun token-rate parsing must not read a label digit as the rate."""

import socket
from pathlib import Path

import pytest
import torch

from core.benchmark.run_manifest import capture_runtime_provenance
from core.harness.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    TorchrunLaunchSpec,
)


# Lines in the exact format the repo's torchrun workers print, for example
# ch04/ddp_worker.py, ch13/expert_parallel_common.py,
# ch17/prefill_decode_disagg_multigpu_common.py and
# labs/decode_optimization/decode_multigpu_demo.py.
@pytest.mark.parametrize(
    ("line", "expected"),
    [(f"rank0 tokens/s: {123.45:.2f} tokens/s", 123.45),
     (f"rank0 tokens/s: {0.5:.2f} tokens/s", 0.5),
     (f"rank0 decode tokens/s: {456.78:.2f} tokens/s", 456.78),
     (f"rank0 tokens/s: {2048.0:.2f} tokens/s (local {1024.0:.2f})", 2048.0)],
)
def test_rank0_worker_line_reports_declared_rate(line, expected):
    assert BenchmarkHarness._extract_tokens_per_s([line]) == expected


def test_rank0_label_digit_does_not_mask_rate_across_lines():
    lines = ["rank0 time_per_iter_ms: 1.500000000", "rank0 tokens/s: 0.25 tokens/s"]
    assert BenchmarkHarness._extract_tokens_per_s(lines) == 0.25


@pytest.mark.parametrize(
    ("line", "expected"),
    [("Throughput: 2048.0 tokens/sec", 2048.0),
     ("1,500 tokens/second", 1500.0),
     ("1.5e+05 tokens/s", 150000.0),
     ("3E2 tok/s", 300.0)],
)
def test_number_first_rate_units_and_forms(line, expected):
    assert BenchmarkHarness._extract_tokens_per_s([line]) == expected


@pytest.mark.parametrize(
    "line",
    ["step 3/32 tokens/s=1234",
     "gpu1 toks/s",
     "v2.5 tok/s",
     "12,34 tok/s",
     "loss=12.7 tokens/step=4096",
     "1.23M tokens/s"],
)
def test_digits_inside_labels_or_numbers_are_not_a_rate(line):
    assert BenchmarkHarness._extract_tokens_per_s([line]) is None


def test_non_finite_rate_fails_instead_of_recording_inf():
    with pytest.raises(RuntimeError, match="non-finite token rate"):
        BenchmarkHarness._extract_tokens_per_s(["rank0 tokens/s: 1e999 tokens/s"])


class _RankZeroRateProcess:
    name = "audit_w1063_rank0_rate"

    def __init__(self, path: Path):
        self.path = path

    def get_torchrun_spec(self, config):
        return TorchrunLaunchSpec(
            script_path=self.path, config_arg_map={"iterations": "--steps"}, name=self.name,
        )


def test_real_torchrun_records_rank0_declared_rate(tmp_path):
    runtime = capture_runtime_provenance()
    if not runtime.library_versions_complete:
        pytest.skip(
            "SKIPPED: this host's installed-library provenance is incomplete, so the "
            "torchrun parity check rejects the run before throughput is parsed"
        )
    script = tmp_path / "rank0_rate.py"
    script.write_text(
        "import argparse\nimport torch\n"
        "p=argparse.ArgumentParser(); p.add_argument('--steps', type=int); a=p.parse_args()\n"
        "x=torch.arange(8); y=x.clone()\n"
        "for _ in range(a.steps): y.add_(x)\n"
        "assert y.tolist() == (torch.arange(8) * (a.steps + 1)).tolist()\n"
        "print('rank1 tokens/s: 999.00 tokens/s')\n"
        "print(f'rank0 tokens/s: {123.45:.2f} tokens/s', flush=True)\n"
    )
    config = BenchmarkConfig(
        device=torch.device("cpu"), iterations=3, warmup=1,
        use_subprocess=False, enable_profiling=False, enable_memory_tracking=False,
        enforce_environment_validation=False, min_run_time_ms=10,
        measurement_timeout_seconds=40, nproc_per_node=1,
        rdzv_backend="static", rdzv_endpoint="127.0.0.1:29500",
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    harness._ensure_runtime_initialized()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    result = harness._benchmark_with_torchrun(_RankZeroRateProcess(script), config)
    assert not result.errors, result.errors
    assert result.throughput.tokens_per_s == 123.45
    assert result.custom_metrics["tokens_per_s"] == 123.45
