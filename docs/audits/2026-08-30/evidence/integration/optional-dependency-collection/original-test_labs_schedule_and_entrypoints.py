from core.harness.benchmark_harness import BaseBenchmark
from labs.occupancy_tuning import optimized_proton_matmul_bm64_bn256_bk32 as wide_n_module
from labs.occupancy_tuning import optimized_proton_matmul_bm64_bn64_bk32_nw2 as low_warp_module
from labs.real_world_models.deepseek_r1_moe_optimization import get_benchmark as get_deepseek_benchmark
from labs.real_world_models.gpt4_architecture_optimization import get_benchmark as get_gpt4_benchmark


def test_wide_n_schedule_matches_filename() -> None:
    schedule = wide_n_module.WIDE_N_LATENCY_SCHEDULE
    assert schedule.name == "bm64_bn256_bk32"
    assert (schedule.block_m, schedule.block_n, schedule.block_k) == (64, 256, 32)


def test_low_warp_schedule_matches_filename() -> None:
    schedule = low_warp_module.LATENCY_FRIENDLY_SCHEDULE
    assert schedule.name == "bm64_bn64_bk32_nw2"
    assert (schedule.block_m, schedule.block_n, schedule.block_k, schedule.num_warps) == (64, 64, 32, 2)


def test_real_world_model_entrypoints_return_benchmarks() -> None:
    deepseek = get_deepseek_benchmark()
    gpt4 = get_gpt4_benchmark()
    assert isinstance(deepseek, BaseBenchmark)
    assert isinstance(gpt4, BaseBenchmark)
