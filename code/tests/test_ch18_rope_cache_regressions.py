"""Full cache controls for the production sequential RoPE benchmark loop."""

import torch

from ch18.baseline_rope_q_cache import BaselineRopeQCacheBenchmark
from ch18.optimized_rope_q_cache import OptimizedRopeQCacheBenchmark
from ch18.rope_q_cache_common import RopeQCacheConfig, build_rope_tables


def test_rope_direct_cache_writes_match_every_step_and_preserve_unused_tail():
    class CpuBaseline(BaselineRopeQCacheBenchmark):
        allow_cpu = True

    class CpuOptimized(OptimizedRopeQCacheBenchmark):
        allow_cpu = True

    cfg = RopeQCacheConfig(batch_size=2, heads=3, head_dim=8, steps=5, max_seq_len=7, dtype=torch.float32)
    torch.manual_seed(18)
    baseline, optimized = CpuBaseline(cfg), CpuOptimized(cfg)
    inputs = torch.randn(cfg.steps, cfg.batch_size, cfg.hidden_size)
    weight = torch.randn(cfg.hidden_size, cfg.hidden_size)
    cos, sin = build_rope_tables(cfg.max_seq_len, cfg.head_dim, torch.device('cpu'), cfg.dtype)
    for benchmark in (baseline, optimized):
        benchmark.device = torch.device('cpu')
        benchmark.inputs = inputs.clone()
        benchmark.q_weight = weight.clone()
        benchmark.cos, benchmark.sin = cos.clone(), sin.clone()
        benchmark.cache = torch.full((cfg.batch_size, cfg.heads, cfg.max_seq_len, cfg.head_dim), float('nan'))
        benchmark.q_buffer = torch.empty(cfg.batch_size, cfg.hidden_size)
        benchmark.q_heads = benchmark.q_buffer.view(cfg.batch_size, cfg.heads, cfg.head_dim)
        benchmark._input_step_views = list(benchmark.inputs.unbind(0))
        benchmark._cache_step_views = [benchmark.cache[:, :, step, :] for step in range(cfg.steps)]
        benchmark._cos_step_views = [cos[step].view(1, 1, cfg.head_dim) for step in range(cfg.steps)]
        benchmark._sin_step_views = [sin[step].view(1, 1, cfg.head_dim) for step in range(cfg.steps)]
        benchmark._step_groups = list(zip(benchmark._input_step_views, benchmark._cache_step_views, benchmark._cos_step_views, benchmark._sin_step_views, strict=True))
        benchmark._input_step_count = benchmark._cache_step_count = cfg.steps
        benchmark._cos_step_count = benchmark._sin_step_count = benchmark._step_group_count = cfg.steps
        benchmark._output_view = benchmark.cache[:, :, :cfg.steps, :]
    baseline.rope_out = torch.empty(cfg.batch_size, cfg.heads, cfg.head_dim)
    for repetition in range(3):
        changed_inputs = inputs + repetition
        expected = []
        for step, x in enumerate(changed_inputs):
            q = (x @ weight).view(cfg.batch_size, cfg.heads, cfg.head_dim)
            half = cfg.head_dim // 2
            rotated = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            expected.append(q * cos[step] + rotated * sin[step])
        expected = torch.stack(expected, dim=2)
        for benchmark in (baseline, optimized):
            benchmark.inputs.copy_(changed_inputs)
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            torch.testing.assert_close(benchmark.get_verify_output(), expected)
            torch.testing.assert_close(benchmark.get_verify_inputs()['inputs'], changed_inputs)
            assert torch.isnan(benchmark.cache[:, :, cfg.steps:, :]).all()
