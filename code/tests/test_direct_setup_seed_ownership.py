"""Controls for caller-owned seeds in direct benchmarks and templates."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import torch
import torch.nn as nn

from ch06.roofline_analysis_ilp import RooflineAnalysisILPBenchmark
from ch18.scheduling_vllm_sglang import SchedulingBenchmark
from templates.benchmark_compliant import CompliantBenchmark
from tests.protection_test_utils import preserve_rng_state

CODE_ROOT = Path(__file__).resolve().parents[1]
SETUP_CLASSES = {
    "ch06/roofline_analysis_ilp.py": ("RooflineAnalysisILPBenchmark",),
    "ch15/kv_cache_management_math.py": ("KVCacheManagementMathBenchmark",),
    "ch14/flash_attention_sdpa_bench.py": ("FlashAttentionSdpaBenchBenchmark",),
    "ch14/flex_attention_sparse_demo.py": ("FlexAttentionSparseDemoBenchmark",),
    "ch14/triton_persistent_batched_bench.py": ("TritonPersistentBatchedBenchBenchmark",),
    "ch14/sliding_window_demo.py": ("SlidingWindowDemoBenchmark",),
    "ch14/triton_persistent_demo.py": ("TritonPersistentDemoBenchmark",),
    "ch13/fp8_perchannel_bench.py": ("OptimizedFP8PerChannelBenchmark",),
    "ch13/fp8_perchannel_demo.py": ("FP8PerChannelDemoBenchmark",),
    "ch13/fp8_static_demo.py": ("FP8StaticDemoBenchmark",),
    "ch04/optimizer_central_nvlink.py": ("OptimizedOptimizerCentralNvlinkBenchmark",),
    "ch04/nvls_collectives.py": ("NVLSCollectivesBenchmark",),
    "ch04/optimizer_replicated.py": ("BaselineOptimizerReplicatedBenchmark",),
    "ch04/ddp_nvlink_overlap.py": ("OptimizedDdpNvlinkOverlapBenchmark",),
    "ch04/ddp_nvlink_naive.py": ("BaselineDdpNvlinkNaiveBenchmark",),
    "templates/benchmark_compliant.py": ("CompliantBenchmark",),
    "templates/benchmark_template.py": ("MyBenchmark",),
    "ch18/run_vllm_decoder.py": ("VLLMMoEInferenceBenchmark",),
    "ch18/scheduling_vllm_sglang.py": ("SchedulingBenchmark",),
    "core/benchmark/examples.py": ("InferenceBenchmarkBase",),
    "labs/moe_optimization_journey/level4_triton.py": ("Level4Triton",),
    "labs/moe_optimization_journey/level6_full_stack.py": ("Level6FullStack",),
}
GLOBAL_SEED_CALLS = {
    "random.seed",
    "torch.manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
}


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def test_direct_benchmark_setups_do_not_override_caller_seed() -> None:
    checked = 0
    for relative_path, class_names in SETUP_CLASSES.items():
        tree = ast.parse((CODE_ROOT / relative_path).read_text(encoding="utf-8"))
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in class_names
        }
        assert classes.keys() == set(class_names)
        for class_name in class_names:
            setup = next(
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == "setup"
            )
            assert not any(
                isinstance(node, ast.Call)
                and _call_name(node) in GLOBAL_SEED_CALLS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 42
                for node in ast.walk(setup)
            )
            checked += 1
    assert checked == 22


def test_cpu_scheduler_preserves_legacy_42_and_uses_fresh_python_seed() -> None:
    with preserve_rng_state():
        random.seed(42)
        legacy = SchedulingBenchmark()
        legacy.setup()
        legacy_lengths = legacy.request_lengths

        random.seed(1042)
        fresh = SchedulingBenchmark()
        fresh.setup()
        fresh_lengths = fresh.request_lengths

    expected_fresh_rng = random.Random(1042)
    assert legacy_lengths == [24, 7, 4, 27, 12, 11, 11, 8]
    assert fresh_lengths == [expected_fresh_rng.randint(4, 32) for _ in range(8)]
    assert fresh_lengths != legacy_lengths


def test_cpu_template_matches_legacy_42_and_changes_at_1042() -> None:
    with preserve_rng_state():
        torch.manual_seed(42)
        expected_model = nn.Linear(128, 128)
        expected_input = torch.randn(8, 128)

        torch.manual_seed(42)
        legacy = CompliantBenchmark()
        legacy.device = torch.device("cpu")
        legacy.setup()

        torch.manual_seed(1042)
        fresh = CompliantBenchmark()
        fresh.device = torch.device("cpu")
        fresh.setup()

    torch.testing.assert_close(legacy.model.weight, expected_model.weight, rtol=0, atol=0)
    torch.testing.assert_close(legacy.model.bias, expected_model.bias, rtol=0, atol=0)
    torch.testing.assert_close(legacy.input, expected_input, rtol=0, atol=0)
    assert not torch.equal(legacy.model.weight, fresh.model.weight)
    assert not torch.equal(legacy.input, fresh.input)


def test_nonstochastic_cpu_setup_does_not_replace_active_seed() -> None:
    with preserve_rng_state():
        torch.manual_seed(1042)
        benchmark = RooflineAnalysisILPBenchmark()
        benchmark.setup()
        observed_seed = torch.initial_seed()

    assert observed_seed == 1042
