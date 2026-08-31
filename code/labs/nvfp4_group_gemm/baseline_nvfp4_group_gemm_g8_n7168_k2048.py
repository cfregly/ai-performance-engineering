"""Independent NVFP4 grouped GEMM baseline: unpack, scale, and FP64 matmul."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.reference_math import reference_group_gemm, prepare_reference
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES, NVFP4GroupGemmBenchmark, attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[1]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=reference_group_gemm,
        prepare=prepare_reference,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=f"nvfp4_group_gemm_{case.name}_baseline_reference_fp64",
    )
    return attach_benchmark_metadata(bench, __file__)
