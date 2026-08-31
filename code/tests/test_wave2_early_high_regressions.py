from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional

import ch02.nvlink_c2c_bandwidth_benchmark as nvlink_bandwidth
import ch02.optimized_grace_coherent_memory as grace_coherent
import ch04.ddp_nvlink_overlap as ddp_nvlink_overlap
import ch04.multi_node_blackwell as multi_node_blackwell
import ch04.optimizer_central_nvlink as optimizer_central_nvlink
import ch05.optimized_ai as optimized_ai

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_sm12_is_not_inferred_to_be_a_gb200_or_gb300_superchip() -> None:
    source = (CODE_ROOT / "ch02" / "memory_transfer_nvlink_demo.cu").read_text(
        encoding="utf-8"
    )

    assert "bool is_datacenter_blackwell = (prop.major == 10);" in source
    assert "bool is_sm12_blackwell = (prop.major == 12);" in source
    assert "SM 12.x identifies GB10/workstation-class GPUs, not a GB200/GB300 Superchip" in source
    assert "bool is_grace_blackwell = (prop.major >= 12);" not in source
    assert "has_grace_cpu || is_grace_blackwell" not in source


def test_nvlink_bandwidth_converts_mib_payloads_to_decimal_gb_per_second() -> None:
    bandwidth = nvlink_bandwidth._decimal_gb_per_second(
        size_mb=1024,
        iterations=2,
        elapsed_seconds=2.0,
    )
    bidirectional = nvlink_bandwidth._decimal_gb_per_second(
        size_mb=512,
        iterations=2,
        elapsed_seconds=2.0,
        directions=2,
    )

    assert bandwidth == pytest.approx(1.073741824)
    assert bidirectional == pytest.approx(1.073741824)


def test_resident_grace_path_reports_zero_explicit_transfer_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ResidentImpl:
        strategy = "zero_copy"
        is_grace_blackwell = True

        def __init__(self, *, size_mb: int, iterations: int) -> None:
            self.size_mb = size_mb
            self.iterations = iterations

        def run_step(self) -> float:
            return 0.25

    monkeypatch.setattr(grace_coherent, "OptimizedGraceCoherentMemory", _ResidentImpl)
    benchmark = grace_coherent.OptimizedGraceCoherentMemoryBenchmark(
        size_mb=8,
        iterations=1,
    )

    benchmark.benchmark_fn()
    metrics = benchmark.get_custom_metrics()

    assert benchmark.get_workload_metadata().bytes_per_iteration == 0.0
    assert benchmark.bandwidth_gb_s is None
    assert metrics == {
        "coherent_memory.explicit_transfer_bytes": 0.0,
        "coherent_memory.resident_elements": float(8 * 1024 * 1024 // 4),
        "coherent_memory.resident_compute": 1.0,
    }
    assert not any(key.startswith("transfer.") for key in metrics)


def test_ddp_nvlink_validates_only_the_root_routes_without_manual_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[tuple[int, int]] = []
    monkeypatch.setattr(ddp_nvlink_overlap.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(ddp_nvlink_overlap, "skip_if_insufficient_gpus", lambda count: None)
    monkeypatch.setattr(
        ddp_nvlink_overlap,
        "require_peer_access",
        lambda src, dst, script_name=None: routes.append((src, dst)),
    )

    ddp_nvlink_overlap._require_root_peer_access()

    assert routes == [(1, 0), (0, 1), (2, 0), (0, 2)]
    assert "enable_peer_access" not in inspect.getsource(ddp_nvlink_overlap)


def test_central_optimizer_validates_root_routes_without_manual_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[tuple[int, int]] = []
    monkeypatch.setattr(optimizer_central_nvlink.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        optimizer_central_nvlink,
        "skip_if_insufficient_gpus",
        lambda count: None,
    )
    monkeypatch.setattr(
        optimizer_central_nvlink,
        "require_peer_access",
        lambda src, dst, script_name=None: routes.append((src, dst)),
    )
    benchmark = optimizer_central_nvlink.OptimizedOptimizerCentralNvlinkBenchmark.__new__(
        optimizer_central_nvlink.OptimizedOptimizerCentralNvlinkBenchmark
    )

    benchmark._require_root_peer_access()

    assert routes == [(1, 0), (0, 1), (2, 0), (0, 2)]
    assert "enable_peer_access" not in inspect.getsource(optimizer_central_nvlink)


def test_multinode_attention_uses_complete_heads_for_sdpa() -> None:
    torch.manual_seed(7)
    block = multi_node_blackwell.MultiNodeTransformerBlock(
        d_model=12,
        num_heads=3,
        d_ff=24,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 5, 12)

    with torch.inference_mode():
        actual = block(inputs)

        normalized = block.attn_norm(inputs)
        q = block.q_proj(normalized).view(2, 5, 3, 4).transpose(1, 2)
        k = block.k_proj(normalized).view(2, 5, 3, 4).transpose(1, 2)
        v = block.v_proj(normalized).view(2, 5, 3, 4).transpose(1, 2)
        attention = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attention = attention.transpose(1, 2).contiguous().flatten(2)
        after_attention = block.o_proj(attention) + inputs
        expected = block.ff2(functional.gelu(block.ff1(block.ff_norm(after_attention))))
        expected = expected + after_attention

    assert block.head_dim == 4
    torch.testing.assert_close(actual, expected)


def test_tensor_parallel_size_must_preserve_whole_attention_heads() -> None:
    class _TensorParallelMesh:
        @staticmethod
        def size() -> int:
            return 4

    model = SimpleNamespace(layers=[SimpleNamespace(num_heads=6)])
    device_mesh = {"tp": _TensorParallelMesh()}

    with pytest.raises(ValueError, match=r"num_heads \(6\) must be divisible by tp_size \(4\)"):
        multi_node_blackwell.apply_tensor_parallelism(model, device_mesh)


def test_pinned_host_slot_waits_for_dma_before_cpu_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _CopyDoneEvent:
        def synchronize(self) -> None:
            calls.append("dma_complete")

    benchmark = optimized_ai.OptimizedAIBenchmark.__new__(optimized_ai.OptimizedAIBenchmark)
    benchmark.mapped_inputs = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    benchmark.host_views = [np.zeros((2, 2), dtype=np.float32)]
    benchmark.host_copy_done_events = [_CopyDoneEvent()]
    original_copyto = np.copyto

    def _recording_copyto(destination: np.ndarray, source: np.ndarray) -> None:
        calls.append("host_overwrite")
        original_copyto(destination, source)

    monkeypatch.setattr(optimized_ai.np, "copyto", _recording_copyto)

    benchmark._stage_to_host(0, 1)

    assert calls == ["dma_complete", "host_overwrite"]
    np.testing.assert_array_equal(benchmark.host_views[0], benchmark.mapped_inputs[1])
    enqueue_source = inspect.getsource(optimized_ai.OptimizedAIBenchmark._enqueue_copy)
    assert enqueue_source.index("copy_(") < enqueue_source.index(
        "host_copy_done_events[slot].record"
    )
