from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core.benchmark.bench_commands import (
    _collect_gpu_requirements,
    _collect_multi_gpu_examples,
)

CODE_ROOT = Path(__file__).resolve().parents[1]

MISROUTED_THIN_WRAPPER_TARGETS = {
    "ch04:no_overlap",
    "ch17:prefill_decode_disagg_batched_multigpu",
    "ch17:prefill_decode_disagg_overlap_multigpu",
    "ch17:prefill_decode_disagg_tpot_long_multigpu",
    "ch17:prefill_decode_disagg_ttft_multigpu",
    "labs/cache_aware_disagg_inference:cache_aware_disagg_multigpu",
    "labs/fullstack_cluster:moe_hybrid_ep_multigpu",
    "labs/train_distributed:ddp_compression_multigpu_int8",
    "labs/train_distributed:ddp_compression_multigpu_powersgd",
    "labs/train_distributed:ddp_flash_multigpu",
    "labs/train_distributed:ddp_multigpu",
    "labs/train_distributed:fsdp2_multigpu",
    "labs/train_distributed:fsdp_multigpu",
    "labs/train_distributed:pipeline_1f1b_multigpu",
    "labs/train_distributed:pipeline_1f1b_to_gpipe_multigpu",
    "labs/train_distributed:pipeline_dualpipe_multigpu",
    "labs/train_distributed:pipeline_dualpipev_multigpu",
    "labs/train_distributed:pipeline_gpipe_multigpu",
    "labs/train_distributed:pipeline_gpipe_to_dualpipe_multigpu",
    "labs/train_distributed:pipeline_gpipe_to_dualpipev_multigpu",
    "labs/train_distributed:symmem_training_multigpu",
    "labs/train_distributed:zero1_multigpu",
    "labs/train_distributed:zero2_multigpu",
    "labs/train_distributed:zero3_multigpu",
}


def test_real_inventory_routes_thin_wrappers_without_importing_workloads() -> None:
    script = """
import json
import sys
from pathlib import Path

from core.benchmark.e2e_sweep import discover_benchmark_e2e_inventory

inventory = discover_benchmark_e2e_inventory(Path.cwd())
workload_prefixes = ("ch02.", "ch04.", "ch13.", "ch15.", "ch17.", "labs.")
print("INVENTORY=" + json.dumps({
    "targets": inventory["targets"],
    "loaded_workloads": sorted(
        name for name in sys.modules if name.startswith(workload_prefixes)
    ),
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("INVENTORY=")
    )
    payload = json.loads(payload_line.removeprefix("INVENTORY="))
    by_target = {entry["target"]: entry for entry in payload["targets"]}

    assert set(by_target) >= MISROUTED_THIN_WRAPPER_TARGETS
    assert {
        target: (
            by_target[target]["multi_gpu"],
            by_target[target]["minimum_gpu_count"],
            by_target[target]["requires_torchrun"],
        )
        for target in MISROUTED_THIN_WRAPPER_TARGETS
    } == {target: (True, 2, True) for target in MISROUTED_THIN_WRAPPER_TARGETS}
    assert by_target["ch04:torchcomms"]["multi_gpu"] is False
    assert by_target["ch04:torchcomms"]["minimum_gpu_count"] == 1
    assert by_target["ch04:torchcomms"]["requires_torchrun"] is True
    assert by_target["ch04:bandwidth_benchmark_suite"]["multi_gpu"] is True
    assert by_target["ch04:bandwidth_benchmark_suite"]["requires_torchrun"] is False
    assert payload["loaded_workloads"] == []


def test_static_inventory_preserves_two_and_four_gpu_prerequisites(tmp_path: Path) -> None:
    (tmp_path / "core" / "benchmark").mkdir(parents=True)
    chapter_dir = tmp_path / "ch99"
    chapter_dir.mkdir()
    two_gpu_source = """
class TwoGpuBenchmark:
    multi_gpu_required = True

def get_benchmark():
    return TwoGpuBenchmark()
"""
    four_gpu_source = """
class FourGpuBenchmark:
    def get_config(self):
        return BenchmarkConfig(
            multi_gpu_required=True,
            required_world_size=4,
        )

def get_benchmark():
    return FourGpuBenchmark()
"""
    for variant in ("baseline", "optimized"):
        (chapter_dir / f"{variant}_two_gpu.py").write_text(two_gpu_source, encoding="utf-8")
        (chapter_dir / f"{variant}_four_gpu.py").write_text(four_gpu_source, encoding="utf-8")

    assert _collect_gpu_requirements(chapter_dir) == {
        "four_gpu": 4,
        "two_gpu": 2,
    }


def test_cli_inventory_uses_the_same_declared_routing() -> None:
    multi_gpu = _collect_multi_gpu_examples(CODE_ROOT / "ch04")

    assert multi_gpu["no_overlap"] is True
    assert multi_gpu["bandwidth_benchmark_suite"] is True
    assert multi_gpu["torchcomms"] is False
