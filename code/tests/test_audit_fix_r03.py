"""Regression coverage for audit residual W2-056 (ch15 KV-transfer verification)."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch15.baseline_disaggregated_inference_multigpu import (
    DECODE_OUTPUT_TOLERANCE,
    TORCHRUN_LAUNCH_WALL_NS_ENV,
    TORCHRUN_RESULT_CALLBACK,
    TORCHRUN_RESULT_DIR_ENV,
    TORCHRUN_RESULT_LABEL_ENV,
    TORCHRUN_RESULT_TOKEN_ENV,
    BaselineDisaggregatedInferenceMultiGPUBenchmark,
    DisaggConfig,
    _allocate_decode_outputs,
    _apply_profile_overrides,
    _build_moe_config,
    _build_pair_groups,
    _build_prefill_reference_output,
    _kv_position_weights,
    _make_torchrun_result_contract,
    _run_decode,
    _run_prefill,
    _share_prefill_weights,
    _transferred_kv_context,
    _write_torchrun_rank_result,
)
from ch15.optimized_disaggregated_inference_multigpu import (
    OptimizedDisaggregatedInferenceMultiGPUBenchmark,
)
from core.optimization.moe_inference import SimpleMoEGPT, allocate_kv_cache

CPU = torch.device("cpu")


def test_transferred_kv_context_weights_positions_by_recency() -> None:
    cache = torch.arange(24, dtype=torch.float32).view(1, 3, 8)

    kv_context, context_fp32 = _transferred_kv_context(cache, context_window=2)

    expected = ((cache[:, 0] + 2.0 * cache[:, 1]) / 3.0).unsqueeze(1)
    torch.testing.assert_close(context_fp32, expected)
    torch.testing.assert_close(kv_context, expected)

    weights = _kv_position_weights(2, CPU)
    _, reused_fp32 = _transferred_kv_context(cache, 2, position_weights=weights)
    torch.testing.assert_close(reused_fp32, context_fp32, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cache, torch.arange(24, dtype=torch.float32).view(1, 3, 8))
    with pytest.raises(ValueError, match="position_weights"):
        _transferred_kv_context(cache, 2, position_weights=_kv_position_weights(3, CPU))

    changed_tail = cache.clone()
    changed_tail[:, 2] += 100.0
    _, tail_context = _transferred_kv_context(changed_tail, context_window=2)
    torch.testing.assert_close(tail_context, expected)

    bf16_context, bf16_fp32 = _transferred_kv_context(cache.to(torch.bfloat16), 2)
    assert bf16_context.dtype == torch.bfloat16
    assert bf16_fp32.dtype == torch.float32
    assert bf16_fp32.shape == (1, 1, 8)


def test_permuted_kv_transfer_changes_decode_verification_output() -> None:
    cfg = DisaggConfig(
        vocab_size=64,
        hidden_size=16,
        ffn_size=16,
        num_layers=1,
        num_moe_layers=1,
        num_experts=2,
        top_k=1,
        batch_size=1,
        requests_per_rank=1,
        context_window=6,
        decode_tokens=3,
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(1234)
    model = SimpleMoEGPT(_build_moe_config(cfg), device=CPU).eval()
    # Small integers keep every partial sum exact, so a mean-only signal would
    # be bit-identical for any reordering of the window.
    kv_prompt = torch.randint(
        -8,
        8,
        (cfg.batch_size, cfg.context_window, cfg.hidden_size),
        generator=generator,
    ).to(torch.float32)
    permuted_prompt = kv_prompt.flip(1).contiguous()
    assert not torch.equal(kv_prompt, permuted_prompt)
    torch.testing.assert_close(
        kv_prompt.mean(dim=1),
        permuted_prompt.mean(dim=1),
        rtol=0.0,
        atol=0.0,
    )
    seed_tokens = torch.tensor([[3]], dtype=torch.long)

    reference = _run_decode(cfg, model, [kv_prompt], [seed_tokens], CPU)[0].clone()
    repeat = _run_decode(cfg, model, [kv_prompt.clone()], [seed_tokens], CPU)[0].clone()
    permuted = _run_decode(cfg, model, [permuted_prompt], [seed_tokens], CPU)[0].clone()

    assert reference.shape == (cfg.batch_size, cfg.hidden_size + 1)
    torch.testing.assert_close(repeat, reference, rtol=0.0, atol=0.0)
    hidden = cfg.hidden_size
    assert not torch.equal(permuted[:, :hidden], reference[:, :hidden])


class _TwoRankLaunch:
    """One CPU stand-in for a two-rank launch that runs the real ch15 code paths.

    Rank 0 recomputes its reference with the worker's reference helper. Rank 1
    decodes the KV that rank 0 produced, through the persistent cache and output
    slots the timed decode rank uses. Only process launch is replaced.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("AISP_NCU_PROFILE_REQUESTS", "2")
        monkeypatch.setenv("AISP_NCU_PROFILE_CONTEXT", "6")
        monkeypatch.setenv("AISP_NCU_PROFILE_DECODE", "2")
        monkeypatch.setenv("AISP_NCU_PROFILE_BATCH", "1")
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.cfg = _apply_profile_overrides(DisaggConfig())
        self.benchmark = BaselineDisaggregatedInferenceMultiGPUBenchmark.__new__(
            BaselineDisaggregatedInferenceMultiGPUBenchmark
        )
        self.benchmark.world_size = 2
        self.benchmark.label = "baseline_disaggregated_inference_multigpu"
        self.benchmark.worker_module = "ch15.baseline_disaggregated_inference_multigpu"
        self.benchmark._subprocess_verify_output = torch.ones(1)
        self.config = self.benchmark.get_config()
        self.spec = self.benchmark.get_torchrun_spec(self.config)
        self.result_dir = Path(self.spec.env[TORCHRUN_RESULT_DIR_ENV])
        self.contract = _make_torchrun_result_contract(
            self.cfg,
            label=self.benchmark.label,
            world_size=2,
            iterations=int(self.config.iterations),
            warmup=int(self.config.warmup),
        )
        generator = torch.Generator().manual_seed(42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = SimpleMoEGPT(_build_moe_config(self.cfg), device=CPU).eval()
        self.prompts = torch.randint(
            0,
            self.cfg.vocab_size,
            (self.cfg.requests_per_rank, self.cfg.batch_size, self.cfg.context_window),
            generator=generator,
        )

    def child_env(self, launch_wall_ns: int) -> None:
        for name in (TORCHRUN_RESULT_DIR_ENV, TORCHRUN_RESULT_TOKEN_ENV, TORCHRUN_RESULT_LABEL_ENV):
            self.monkeypatch.setenv(name, self.spec.env[name])
        self.monkeypatch.setenv(TORCHRUN_LAUNCH_WALL_NS_ENV, str(launch_wall_ns))

    def outputs(
        self,
        corrupt: Callable[[torch.Tensor], torch.Tensor] = lambda kv: kv,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = _build_prefill_reference_output(self.cfg, self.model, self.prompts, CPU)
        sent_kv, sent_seeds = _run_prefill(self.cfg, self.model, self.prompts)
        received_kv = [corrupt(kv.clone()) for kv in sent_kv]
        timed = _run_decode(
            self.cfg,
            self.model,
            received_kv,
            [seed.clone() for seed in sent_seeds],
            CPU,
            kv_cache=allocate_kv_cache(
                self.cfg.batch_size,
                self.cfg.tokens_per_request,
                self.cfg.hidden_size,
                self.cfg.dtype,
                CPU,
            ),
            outputs=_allocate_decode_outputs(self.cfg, CPU),
        )
        return reference, torch.cat(timed, dim=0)

    def publish(self, reference: torch.Tensor, timed: torch.Tensor) -> None:
        assert _write_torchrun_rank_result(
            self.contract, rank=0, output=reference, prompts=self.prompts
        )
        assert _write_torchrun_rank_result(self.contract, rank=1, output=timed, prompts=None)

    def consume(self, launch_wall_ns: int, *, returncode: int = 0) -> None:
        self.benchmark.consume_torchrun_rank_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=time.time_ns(),
            returncode=returncode,
        )


@pytest.fixture
def two_rank_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _TwoRankLaunch:
    return _TwoRankLaunch(monkeypatch, tmp_path)


def test_torchrun_spec_binds_a_fresh_rank_result_transport(two_rank_launch: _TwoRankLaunch) -> None:
    launch = two_rank_launch
    spec = launch.spec

    assert spec.result_callback == TORCHRUN_RESULT_CALLBACK
    assert callable(getattr(launch.benchmark, TORCHRUN_RESULT_CALLBACK))
    assert launch.result_dir.is_dir() and not any(launch.result_dir.iterdir())
    assert launch.result_dir.parent == launch.tmp_path
    assert spec.env[TORCHRUN_RESULT_LABEL_ENV] == launch.benchmark.label
    assert len(spec.env[TORCHRUN_RESULT_TOKEN_ENV]) == 32
    assert spec.timing_iterations_per_sample == int(launch.config.iterations)
    assert "_subprocess_verify_output" not in vars(launch.benchmark)
    with pytest.raises(RuntimeError, match="fresh results from every timed torchrun rank"):
        launch.benchmark._prepare_verification_payload()

    second = launch.benchmark.get_torchrun_spec(launch.config)
    assert second.env[TORCHRUN_RESULT_TOKEN_ENV] != spec.env[TORCHRUN_RESULT_TOKEN_ENV]
    assert second.env[TORCHRUN_RESULT_DIR_ENV] != spec.env[TORCHRUN_RESULT_DIR_ENV]


def test_writer_publishes_nothing_without_a_callback_bound_launch(
    two_rank_launch: _TwoRankLaunch,
) -> None:
    launch = two_rank_launch
    reference, _ = launch.outputs()
    launch.monkeypatch.delenv(TORCHRUN_LAUNCH_WALL_NS_ENV, raising=False)

    assert not _write_torchrun_rank_result(
        launch.contract, rank=0, output=reference, prompts=launch.prompts
    )
    assert not any(launch.result_dir.iterdir())


def test_timed_decode_outputs_become_the_verification_payload(
    two_rank_launch: _TwoRankLaunch,
) -> None:
    launch = two_rank_launch
    launch_wall_ns = time.time_ns()
    launch.child_env(launch_wall_ns)
    reference, timed = launch.outputs()
    torch.testing.assert_close(timed, reference, rtol=0.0, atol=0.0)
    launch.publish(reference, timed)

    launch.consume(launch_wall_ns)

    benchmark = launch.benchmark
    torch.testing.assert_close(benchmark.get_verify_output(), timed, rtol=0.0, atol=0.0)
    assert benchmark.get_output_tolerance() == DECODE_OUTPUT_TOLERANCE
    assert torch.equal(benchmark.get_verify_inputs()["prompt"], launch.prompts)
    signature = benchmark.get_input_signature()
    rows = launch.cfg.requests_per_rank * launch.cfg.batch_size
    assert signature.world_size == 2
    assert signature.batch_size == rows
    assert signature.shapes["output"] == (rows, launch.cfg.hidden_size + 1)
    assert signature.collective_type == "send_recv"
    assert signature.parameter_count == 2 * sum(p.numel() for p in launch.model.parameters())
    assert signature.precision_flags.bf16
    benchmark._prepare_verification_payload()
    assert not launch.result_dir.exists()


def test_corrupted_kv_handoff_fails_the_launch(two_rank_launch: _TwoRankLaunch) -> None:
    launch = two_rank_launch
    launch_wall_ns = time.time_ns()
    launch.child_env(launch_wall_ns)
    reference, timed = launch.outputs(corrupt=lambda kv: kv.flip(1).contiguous())
    launch.publish(reference, timed)

    with pytest.raises(RuntimeError, match="differs from the reference that prefill rank 0"):
        launch.consume(launch_wall_ns)

    assert launch.result_dir.is_dir()
    with pytest.raises(RuntimeError, match="fresh results from every timed torchrun rank"):
        launch.benchmark._prepare_verification_payload()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_rank", "quorum is incomplete"),
        ("previous_launch", "Stale disaggregated rank result"),
        ("foreign_token", "token mismatch"),
        ("failed_child", "clean torchrun exit"),
    ],
)
def test_missing_or_stale_rank_results_fail_the_launch(
    two_rank_launch: _TwoRankLaunch,
    defect: str,
    message: str,
) -> None:
    launch = two_rank_launch
    launch_wall_ns = time.time_ns()
    launch.child_env(launch_wall_ns)
    reference, timed = launch.outputs()
    if defect == "missing_rank":
        assert _write_torchrun_rank_result(
            launch.contract, rank=0, output=reference, prompts=launch.prompts
        )
    elif defect == "foreign_token":
        launch.monkeypatch.setenv(TORCHRUN_RESULT_TOKEN_ENV, "0" * 32)
        launch.publish(reference, timed)
    else:
        launch.publish(reference, timed)
    if defect == "previous_launch":
        launch_wall_ns = time.time_ns()

    with pytest.raises(RuntimeError, match=message):
        launch.consume(launch_wall_ns, returncode=1 if defect == "failed_child" else 0)
    with pytest.raises(RuntimeError, match="fresh results from every timed torchrun rank"):
        launch.benchmark._prepare_verification_payload()


def test_local_setup_drops_results_from_an_earlier_torchrun_launch(
    two_rank_launch: _TwoRankLaunch,
) -> None:
    launch = two_rank_launch
    launch_wall_ns = time.time_ns()
    launch.child_env(launch_wall_ns)
    reference, timed = launch.outputs()
    launch.publish(reference, timed)
    launch.consume(launch_wall_ns)
    benchmark = launch.benchmark
    assert "_subprocess_verify_output" in vars(benchmark)

    # Stop setup() at its first hardware gate so the test stays on the control path.
    launch.monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="SKIPPED: CUDA required"):
        benchmark.setup()

    assert benchmark._torchrun_result_context is None
    assert benchmark._torchrun_result_bundle is None
    for attribute in (
        "_subprocess_verify_inputs",
        "_subprocess_verify_output",
        "_subprocess_output_tolerance",
        "_subprocess_input_signature",
    ):
        assert attribute not in vars(benchmark)


def _pair_weight_worker(rank: int, world_size: int, init_method: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        num_pairs = world_size // 2
        pair_id = rank % num_pairs
        pair_groups = _build_pair_groups(num_pairs)
        cfg = DisaggConfig(
            vocab_size=32,
            hidden_size=8,
            ffn_size=8,
            num_experts=2,
            top_k=1,
            batch_size=1,
            requests_per_rank=1,
            context_window=4,
            decode_tokens=1,
            dtype=torch.float32,
        )
        # A different seed per rank makes every rank start with its own weights.
        torch.manual_seed(100 + rank)
        model = SimpleMoEGPT(_build_moe_config(cfg), device=CPU).eval()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        _share_prefill_weights(model, prefill_rank=pair_id, group=pair_groups[pair_id])
        after = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        torch.save({"before": before, "after": after}, Path(result_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU distributed path requires the Gloo backend",
)
def test_each_decode_rank_receives_its_prefill_rank_weights(tmp_path: Path) -> None:
    world_size = 4
    num_pairs = world_size // 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _pair_weight_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), str(result_dir)),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
    states = {
        rank: torch.load(result_dir / f"rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    }

    def same(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
        return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)

    for pair_id in range(num_pairs):
        decode_rank = pair_id + num_pairs
        prefill_weights = states[pair_id]["before"]
        assert not same(states[decode_rank]["before"], prefill_weights)
        assert same(states[pair_id]["after"], prefill_weights)
        assert same(states[decode_rank]["after"], prefill_weights)
    assert not same(states[0]["after"], states[1]["after"])


def test_both_arms_share_the_rank_result_transport() -> None:
    assert (
        OptimizedDisaggregatedInferenceMultiGPUBenchmark.consume_torchrun_rank_results
        is BaselineDisaggregatedInferenceMultiGPUBenchmark.consume_torchrun_rank_results
    )
    assert (
        BaselineDisaggregatedInferenceMultiGPUBenchmark.story_metadata["verification_mode"]
        == "timed_torchrun_rank_outputs"
    )
