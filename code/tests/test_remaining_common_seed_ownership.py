"""Real setup controls for the final shared benchmark seed repairs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from ch15.guided_decoding_common import GuidedDecodingBenchmark, GuidedDecodingConfig
from tests.protection_test_utils import preserve_rng_state

CODE_ROOT = Path(__file__).resolve().parents[1]
SETUP_FILES = (
    "ch15/moe_overlap_local_route_common.py",
    "ch15/prefill_decode_disagg_common.py",
    "ch15/moe_comm_exchange_benchmarks.py",
    "ch15/moe_inference_common.py",
    "ch15/moe_routing_benchmark_common.py",
    "ch15/medusa_eagle_speculative_benchmarks.py",
    "ch15/disaggregated_inference_single_common.py",
    "ch15/speculative_decoding_benchmarks.py",
    "ch15/guided_decoding_common.py",
    "ch04/gradient_fusion_common.py",
    "ch04/gradient_compression_common.py",
    "ch04/single_gpu_transfer_common.py",
    "labs/flashattention4/flashattention4_benchmarks.py",
    "labs/moe_optimization_journey/moe_benchmark.py",
    "labs/decode_optimization/decode_common.py",
    "labs/persistent_decode/paged_kv_offload_common.py",
    "labs/persistent_decode/nvlink_offload_common.py",
    "labs/moe_parallelism/benchmarking.py",
    "labs/blackwell_matmul/blackwell_benchmarks.py",
    "labs/parameterized_cuda_graphs/parameterized_cuda_graphs_common.py",
    "labs/cache_aware_disagg_inference/cache_aware_disagg_common.py",
    "labs/occupancy_tuning/triton_matmul_schedules.py",
)


def test_remaining_common_setups_do_not_replace_the_caller_seed() -> None:
    for relative_path in SETUP_FILES:
        tree = ast.parse((CODE_ROOT / relative_path).read_text(encoding="utf-8"))
        setups = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "setup"]
        assert setups, relative_path
        for setup in setups:
            for call in (n for n in ast.walk(setup) if isinstance(n, ast.Call)):
                name = ast.unparse(call.func)
                assert name not in {"torch.manual_seed", "torch.cuda.manual_seed_all"}, relative_path
                if name == "gen.manual_seed":
                    assert ast.unparse(call.args[0]) == "setup_seed"


@pytest.mark.parametrize("reuse_gpu_mask", [False, True])
def test_real_guided_decoding_setup_and_output_follow_caller_seed(reuse_gpu_mask: bool) -> None:
    snapshots = []
    with preserve_rng_state():
        for seed in (42, 42, 1042):
            torch.manual_seed(seed)
            benchmark = GuidedDecodingBenchmark(
                reuse_gpu_mask=reuse_gpu_mask,
                label="seed_control",
                cfg=GuidedDecodingConfig(batch_size=2, steps=2, vocab_size=128, allowed_count=32, output_slice=16),
            )
            benchmark.device = torch.device("cpu")
            benchmark.setup()
            benchmark.benchmark_fn()
            assert torch.initial_seed() == seed
            snapshots.append((benchmark.logits.clone(), benchmark.allowed_token_ids.clone(), benchmark.output.clone()))
            benchmark.teardown()
    assert all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[1], strict=True))
    assert all(not torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[2], strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA generator lifecycle")
@pytest.mark.parametrize("kind", ["moe", "decode"])
def test_real_cuda_graph_cleanup_preserves_fresh_setup_seed(kind: str) -> None:
    snapshots = []
    with preserve_rng_state():
        for seed in (42, 42, 1042):
            if kind == "moe":
                from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark

                benchmark = MoEJourneyBenchmark()
                benchmark.VOCAB_SIZE = 128
                benchmark.HIDDEN_SIZE = 64
                benchmark.INTERMEDIATE_SIZE = 128
                benchmark.NUM_HEADS = 4
                benchmark.NUM_EXPERTS = 4
                benchmark.BATCH_SIZE = 1
                benchmark.SEQ_LEN = 8
            else:
                from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig

                benchmark = DecodeBenchmark(DecodeConfig(
                    batch_size=1, prompt_tokens=8, decode_tokens=2, hidden_size=64, vocab_size=128,
                ))
            torch.manual_seed(seed)
            benchmark.setup()
            assert torch.initial_seed() == seed
            assert torch.cuda.initial_seed() == seed
            if kind == "moe":
                snapshot = (next(benchmark.model.parameters()).detach().clone(), benchmark.input_ids.clone())
            else:
                snapshot = (benchmark.embedding.weight.detach().clone(), benchmark.host_prompt.clone())
            benchmark.benchmark_fn()
            assert benchmark.output is not None
            assert torch.isfinite(benchmark.output).all()
            snapshots.append((*snapshot, benchmark.output.detach().clone()))
            benchmark.teardown()
    assert all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[1], strict=True))
    assert all(not torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[2], strict=True))
