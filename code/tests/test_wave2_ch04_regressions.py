from __future__ import annotations

import inspect
from contextlib import nullcontext

import pytest
import torch

import ch04.bandwidth_benchmark_suite_multigpu as bandwidth_suite
import ch04.ddp_nvlink_overlap as ddp_overlap
import ch04.gradient_compression_common as gradient_compression
import ch04.multi_node_blackwell as multi_node
import ch04.nccl_blackwell_config as nccl_config
import ch04.nvls_collectives as nvls_collectives
import ch04.optimized_pipeline_parallel_multigpu_1f1b as pipeline_1f1b


def test_allgather_bus_traffic_excludes_the_local_shard() -> None:
    data_bytes = 4096

    assert bandwidth_suite._collective_bus_bytes("allgather", data_bytes, 2) == 4096
    assert bandwidth_suite._collective_bus_bytes("allgather", data_bytes, 4) == 12288
    assert bandwidth_suite._collective_bus_bytes("allreduce", data_bytes, 2) == 4096
    assert bandwidth_suite._collective_bus_bytes("reducescatter", data_bytes, 2) == 2048

    with pytest.raises(ValueError, match="world_size must be >=2"):
        bandwidth_suite._collective_bus_bytes("allgather", data_bytes, 1)


def test_ddp_reductions_are_aggregated_once_for_every_replica() -> None:
    reductions = [torch.tensor([2.0, 4.0]), torch.tensor([6.0, 8.0])]
    destination = torch.empty(2)

    result = ddp_overlap._aggregate_reduced_gradients(
        destination,
        reductions,
        range(1, len(reductions)),
        scale=0.25,
    )

    torch.testing.assert_close(result, torch.tensor([2.0, 3.0]))
    assert result.data_ptr() == destination.data_ptr()

    setup_source = inspect.getsource(ddp_overlap.OptimizedDdpNvlinkOverlapBenchmark.setup)
    benchmark_source = inspect.getsource(
        ddp_overlap.OptimizedDdpNvlinkOverlapBenchmark.benchmark_fn
    )
    assert "][: len(self._reduction_results)]" not in setup_source
    assert "for _model_idx, model, update_buffer in self._model_update_groups" in benchmark_source
    assert "root_buf = reduction_results[model_idx]" not in benchmark_source
    assert "grad_snapshot = self._micro_grad_buffers[micro][model_idx]" in benchmark_source


def test_comm_only_fp32_baseline_models_one_transfer_copy() -> None:
    class _CopyCounter:
        def __init__(self) -> None:
            self.sources: list[object] = []

        def copy_(self, source: object) -> None:
            self.sources.append(source)

    input_payload = object()
    output = _CopyCounter()
    benchmark = gradient_compression.GradientCompressionBenchmark.__new__(
        gradient_compression.GradientCompressionBenchmark
    )
    benchmark.inputs = [input_payload]
    benchmark.output = None
    benchmark.compression = "none"
    benchmark.multi_gpu = False
    benchmark.simulate_single_gpu_transfer = True
    benchmark._fp32_outputs = [output]
    benchmark._nvtx_range = lambda _name: nullcontext()

    benchmark.benchmark_fn()

    assert output.sources == [input_payload]
    assert benchmark.output is output


def test_multinode_step_throughput_waits_for_cuda_event_completion() -> None:
    train_source = inspect.getsource(multi_node.train_multi_node)

    start_record = train_source.index("step_start_event.record()")
    model_work = train_source.index("logits = model(input_ids)")
    end_record = train_source.index("step_end_event.record()")
    end_sync = train_source.index("step_end_event.synchronize()")
    elapsed = train_source.index("step_start_event.elapsed_time(step_end_event)")
    throughput = train_source.index("throughput = tokens_per_step / step_time")

    assert start_record < model_work < end_record < end_sync < elapsed < throughput
    assert "step_time = time.perf_counter() - step_start" not in train_source


def test_nccl_configuration_does_not_emit_fabricated_blackwell_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabricated = {
        "NCCL_NVLINK_TCE_ENABLE",
        "NCCL_NVLINK_C2C_ENABLE",
        "NCCL_GRACE_BLACKWELL",
    }
    isolated_environment: dict[str, str] = {}
    monkeypatch.setattr(nccl_config.os, "environ", isolated_environment)

    configured = nccl_config.configure_nccl_for_blackwell(verbose=False)

    assert fabricated.isdisjoint(configured)
    assert all(name not in isolated_environment for name in fabricated)
    signature = inspect.signature(nccl_config.configure_nccl_for_blackwell)
    assert "enable_tce" not in signature.parameters
    assert "enable_nvlink_c2c" not in signature.parameters


def test_nvls_environment_is_set_or_validated_before_communicator_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)

    nvls_collectives._configure_nvls_environment(communicator_initialized=False)

    assert nvls_collectives.os.environ["NCCL_NVLS_ENABLE"] == "1"
    assert "NVLS" in nvls_collectives.os.environ["NCCL_ALGO"].split(",")

    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    with pytest.raises(RuntimeError, match="before NCCL communicator initialization"):
        nvls_collectives._configure_nvls_environment(communicator_initialized=True)

    setup_source = inspect.getsource(nvls_collectives.NVLSCollectivesBenchmark.setup)
    assert setup_source.index("_configure_nvls_environment") < setup_source.index(
        "dist.init_process_group"
    )


@pytest.mark.parametrize(
    ("world_size", "micro_batches", "expected"),
    [
        (2, 2, (1, 1)),
        (4, 4, (3, 1)),
        (8, 16, (7, 9)),
    ],
)
def test_default_sized_1f1b_schedule_has_steady_state_work(
    world_size: int,
    micro_batches: int,
    expected: tuple[int, int],
) -> None:
    assert pipeline_1f1b._resolve_1f1b_schedule(world_size, micro_batches) == expected


def test_1f1b_rejects_a_schedule_with_no_steady_state() -> None:
    with pytest.raises(ValueError, match="at least one steady-state microbatch"):
        pipeline_1f1b._resolve_1f1b_schedule(world_size=4, num_micro_batches=3)
