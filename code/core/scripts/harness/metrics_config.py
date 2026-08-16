"""Per-chapter profiler metric configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List

if TYPE_CHECKING:
    from core.scripts.harness.example_registry import Example


BASE_NCU_METRICS: List[str] = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__sass_average_branch_targets_threads_uniform.pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_data_bytes_mem_shared_op_ld.sum",
    "sm__sass_data_bytes_mem_shared_op_st.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "gpu__time_duration.avg",
]

BASE_NSYS_TRACE_MODULES: List[str] = [
    "cuda",
    "nvtx",
    "osrt",
    "cublas",
    "cudnn",
]

BASE_NSYS_EXTRA_ARGS: List[str] = []


@dataclass
class ProfilerOverrides:
    """Overrides for profiler invocations."""

    ncu_metrics: List[str] = field(default_factory=list)
    nsys_trace: List[str] = field(default_factory=list)
    nsys_extra_args: List[str] = field(default_factory=list)
    pytorch_modes: List[str] = field(default_factory=list)

    def merge(self, other: "ProfilerOverrides") -> None:
        self._extend_unique(self.ncu_metrics, other.ncu_metrics)
        self._extend_unique(self.nsys_trace, other.nsys_trace)
        self._extend_unique(self.nsys_extra_args, other.nsys_extra_args)
        self._extend_unique(self.pytorch_modes, other.pytorch_modes)

    @staticmethod
    def _extend_unique(target: List[str], values: Iterable[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)


def _overrides_from_lists(
    *,
    ncu_metrics: Iterable[str] | None = None,
    nsys_trace: Iterable[str] | None = None,
    nsys_extra_args: Iterable[str] | None = None,
    pytorch_modes: Iterable[str] | None = None,
) -> ProfilerOverrides:
    override = ProfilerOverrides()
    if ncu_metrics:
        override.ncu_metrics.extend(ncu_metrics)
    if nsys_trace:
        override.nsys_trace.extend(nsys_trace)
    if nsys_extra_args:
        override.nsys_extra_args.extend(nsys_extra_args)
    if pytorch_modes:
        override.pytorch_modes.extend(pytorch_modes)
    return override


CONFIG_BY_TAG: Dict[str, ProfilerOverrides] = {
    "ch04": _overrides_from_lists(
        ncu_metrics=[
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch05": _overrides_from_lists(
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
    ),
    "ch07": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_average_branch_targets_threads_uniform.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
        ]
    ),
    "ch08": _overrides_from_lists(
        ncu_metrics=[
            "gpu__time_duration.avg",
            "sm__sass_average_branch_targets_threads_uniform.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch09": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ]
    ),
    "ch10": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch11": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch12": _overrides_from_lists(
        nsys_extra_args=["--gpu-metrics-device=all"],
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
    ),
    "ch13": _overrides_from_lists(
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
        pytorch_modes=["blackwell"],
    ),
    "ch15": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ]
    ),
    "ch16": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        nsys_extra_args=["--gpu-metrics-device=all"],
    ),
    "ch17": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "gpu__time_duration.avg",
        ]
    ),
    "ch18": _overrides_from_lists(
        ncu_metrics=[
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
        nsys_extra_args=["--gpu-metrics-device=all"],
    ),
    "ch19": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch20": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
}

CONFIG_BY_EXAMPLE: Dict[str, ProfilerOverrides] = {
    "ch13_memory_profiling": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_average_branch_targets_threads_uniform.pct",
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
        ],
        pytorch_modes=["memory"],
    ),
    "ch13_fsdp": _overrides_from_lists(
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
        pytorch_modes=["distributed"],
    ),
    "ch13_train_deepseek_coder": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        pytorch_modes=["blackwell"],
    ),
    "ch14_torch_compiler_examples": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__sass_average_branch_targets_threads_uniform.pct",
        ],
        pytorch_modes=["blackwell"],
    ),
    "ch14_triton_examples": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
        ],
    ),
    "ch15_disaggregated_inference_multigpu": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    ),
    "ch16_inference_profiling": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        nsys_extra_args=["--gpu-metrics-device=all"],
        pytorch_modes=["profiling"],
    ),
    "ch17_dynamic_routing": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "gpu__time_duration.avg",
        ],
    ),
    "ch17_early_rejection": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__time_duration.avg",
        ],
    ),
    "ch18_flexdecoding": _overrides_from_lists(
        ncu_metrics=[
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
        ],
    ),
    "ch19_adaptive_parallelism": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
    ),
    "ch19_token_precision_switching": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    ),
    "ch19_dynamic_precision": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    ),
    "ch19_dynamic_quantized_cache": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_data_bytes_mem_shared_op_ld.sum",
            "sm__sass_data_bytes_mem_shared_op_st.sum",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    ),
    "ch20_ai_kernel_generator": _overrides_from_lists(
        ncu_metrics=[
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
    ),
}


def resolve_overrides(example: "Example") -> ProfilerOverrides:
    result = ProfilerOverrides()

    for tag in example.tags:
        tag_key = tag.lower()
        override = CONFIG_BY_TAG.get(tag_key)
        if override is not None:
            result.merge(override)

    example_override = CONFIG_BY_EXAMPLE.get(example.name)
    if example_override is not None:
        result.merge(example_override)

    return result


__all__ = [
    "BASE_NCU_METRICS",
    "BASE_NSYS_TRACE_MODULES",
    "BASE_NSYS_EXTRA_ARGS",
    "ProfilerOverrides",
    "resolve_overrides",
]
