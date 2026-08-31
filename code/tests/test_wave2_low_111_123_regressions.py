from __future__ import annotations

import ast
import contextlib
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]


def _load_function_from_source(path: Path, function_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_w2_111_tensor_parallel_demo_shards_every_layer(capsys: pytest.CaptureFixture[str]) -> None:
    from ch15.disaggregated_inference_multigpu import ParallelismManager

    manager = ParallelismManager(SimpleNamespace(num_gpus=8))
    manager.setup_tensor_parallelism()
    output = capsys.readouterr().out

    assert output.count("across all 6 transformer layers") == 8
    assert "Tensor shard 8/8" in output
    assert "Layers 0-0" not in output


def test_w2_112_large_flexattention_parameter_count_uses_actual_ffn_width() -> None:
    path = CODE_ROOT / "ch18" / "flex_attention_large_model.py"
    estimate_parameter_count = _load_function_from_source(path, "estimate_parameter_count")

    assert estimate_parameter_count(1, 16, 64) == 4 * 16**2 + 2 * 16 * 64
    assert estimate_parameter_count(3, 16, 64) == 3 * 12 * 16**2

    source = path.read_text(encoding="utf-8")
    assert "estimate_memory(n_layers, d_model, d_ff, batch, seq_len)" in source
    assert "5 * d_model * d_model" not in source


def test_w2_113_block_sparse_sparsity_counts_every_local_block() -> None:
    from ch18.flexattention_block_sparse import BlockSparseFlexAttention

    attention = object.__new__(BlockSparseFlexAttention)
    attention.seq_length = 8192
    attention.block_size = 256
    expected = (
        1
        - (32 * 256**2 + 8192 * 31) / 8192**2
    ) * 100
    assert attention._calculate_sparsity() == pytest.approx(expected)
    assert attention._calculate_sparsity() == pytest.approx(96.49658203125)

    attention.seq_length = 10
    attention.block_size = 4
    assert attention._calculate_sparsity() == pytest.approx(44.0)


def test_w2_114_fp6_header_uses_packed_storage_and_b200_dense_rates() -> None:
    source = (CODE_ROOT / "ch19" / "native_fp6_quantization.py").read_text(
        encoding="utf-8"
    )
    header = source.split('"""', 2)[1]

    assert "62.5% vs FP16" in header
    assert "~4.5 PFLOPS dense for either FP6 or FP8" in header
    assert "~9 PFLOPS with structured sparsity" in header
    assert "up to 2.67x as many packed weights as FP16" in header
    assert "~1400 TFLOPS" not in header
    assert "~1200 FP8" not in header


def test_w2_115_moe_metadata_counts_rows_as_tokens() -> None:
    for filename in ("baseline_moe.py", "optimized_moe.py"):
        source = (CODE_ROOT / "ch20" / filename).read_text(encoding="utf-8")
        init = source.split("def __init__", 2)[2].split("def setup", 1)[0]
        assert "tokens_per_iteration=float(self.batch)" in init
        assert "self.batch * self.hidden_dim" not in init

    for filename in ("expectations_b200.json", "expectations_4x_gb200.json"):
        examples = json.loads((CODE_ROOT / "ch20" / filename).read_text(encoding="utf-8"))[
            "examples"
        ]
        assert "moe" not in examples


def test_w2_116_rejection_analyzer_uses_cuda_key_and_hardware_file(tmp_path: Path) -> None:
    from core.analysis.analyze_expectation_rejections import (
        render_expectation_rejection_ledger,
    )

    chapter = tmp_path / "ch09"
    chapter.mkdir()
    expectation_path = chapter / "expectations_4x_gb200.json"
    expectation_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "hardware_key": "4x_gb200",
                "examples": {
                    "gemm_cuda": {
                        "example": "gemm",
                        "type": "cuda",
                        "provenance": {
                            "hardware_key": "4x_gb200",
                            "git_commit": "stored-cuda-sha",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    event_path = run_dir / "logs" / "benchmark_events.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        json.dumps(
            {
                "event_type": "expectation_update",
                "chapter": "ch09",
                "example": "gemm_cuda",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.0,
                "delta_pct": -50.0,
                "new_provenance": {"hardware_key": "4x_gb200"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = render_expectation_rejection_ledger(repo_root=tmp_path, run_dir=run_dir)
    row = json.loads(outputs["json"].read_text(encoding="utf-8"))[0]
    assert row["expectations_file"] == "ch09/expectations_4x_gb200.json"
    assert row["stored_git_commit"] == "stored-cuda-sha"


def test_w2_118_performance_targets_use_consistent_units_and_bounds() -> None:
    from core.benchmark.performance_targets import _DEFAULT_TARGETS

    allreduce = _DEFAULT_TARGETS["ch04"]["metrics"]["allreduce_bandwidth_gbs"]
    assert allreduce == {
        "min": 70,
        "target": 100,
        "unit": "GB/s",
        "realistic_max": 100,
    }

    fp16 = _DEFAULT_TARGETS["overall"]["fp16_compute_tflops"]
    assert fp16["target"] <= fp16["realistic_max"]
    assert fp16["realistic_max"] == 2250


def test_w2_119_tma_dispatch_handles_every_selected_tile_shape() -> None:
    source = (CODE_ROOT / "core" / "common" / "headers" / "cuda13_demos.cuh").read_text(
        encoding="utf-8"
    )
    dispatch = source.split("auto launch_demo", 1)[1].split(
        "cudaError_t launch_err", 1
    )[0]
    for width in (32, 64, 128):
        for height in (32, 64):
            expected = (
                f"std::integral_constant<int, {width}>{{}}, "
                f"std::integral_constant<int, {height}>{{}}"
            )
            assert expected in dispatch


def test_w2_120_grace_coherence_requires_a_grace_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.scripts.utilities import probe_hardware_capabilities as probe

    cc12 = SimpleNamespace(major=12)
    monkeypatch.setattr(probe, "_is_grace_host", lambda: False)
    assert probe._has_grace_coherence(cc12) is False
    assert "NVLink-C2C" not in probe._build_features(cc12)

    monkeypatch.setattr(probe, "_is_grace_host", lambda: True)
    assert probe._has_grace_coherence(cc12) is True
    assert "NVLink-C2C" in probe._build_features(cc12)


def test_w2_121_dashboard_treats_nvidia_memory_units_as_mib() -> None:
    gpu_card = (CODE_ROOT / "dashboard" / "web" / "src" / "components" / "GpuCard.tsx").read_text(
        encoding="utf-8"
    )
    assistant = (
        CODE_ROOT / "dashboard" / "web" / "src" / "components" / "tabs" / "AIAssistantTab.tsx"
    ).read_text(encoding="utf-8")
    gpu_types = (CODE_ROOT / "dashboard" / "web" / "src" / "types" / "index.ts").read_text(
        encoding="utf-8"
    )

    for source in (gpu_card, assistant):
        assert "mebibytesToBytes" in source
        assert "* 1e6" not in source
    assert "MiB (nvidia-smi units)" in gpu_types


def _import_autotune(monkeypatch: pytest.MonkeyPatch):
    loader = types.ModuleType("tcgen05_loader")
    for name in (
        "matmul_tcgen05",
        "matmul_tcgen05_pipelined",
        "matmul_tcgen05_3stage",
        "matmul_tcgen05_swizzled",
        "matmul_tcgen05_cluster",
        "matmul_tcgen05_warp_spec",
    ):
        setattr(loader, name, lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "tcgen05_loader", loader)
    sys.modules.pop("labs.custom_vs_cublas.autotune", None)
    return importlib.import_module("labs.custom_vs_cublas.autotune")


def test_w2_122_autotune_reports_all_kernel_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _import_autotune(monkeypatch)
    monkeypatch.setattr(autotune, "_load_cache", lambda: {})
    monkeypatch.setattr(autotune, "_get_device_key", lambda _device: "GPU-2_10.0")
    monkeypatch.setattr(autotune, "_resolve_cuda_device", lambda _device: torch.device("cuda:2"))
    monkeypatch.setattr(autotune.torch, "randn", lambda *_args, **_kwargs: object())
    entered_devices: list[torch.device] = []

    def device_context(device: torch.device):
        entered_devices.append(device)
        return contextlib.nullcontext()

    monkeypatch.setattr(autotune.torch.cuda, "device", device_context)
    monkeypatch.setattr(autotune.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        autotune,
        "KERNELS",
        {"basic": object(), "pipelined": object()},
    )

    def fail_kernel(*_args, **_kwargs):
        raise RuntimeError("launch rejected")

    monkeypatch.setattr(autotune, "_benchmark_kernel", fail_kernel)
    with pytest.raises(RuntimeError, match=r"all 2 kernels failed") as excinfo:
        autotune.autotune(16, 32, 64, verbose=False, device=torch.device("cuda:2"))
    assert "basic=RuntimeError: launch rejected" in str(excinfo.value)
    assert "pipelined=RuntimeError: launch rejected" in str(excinfo.value)
    assert entered_devices == [torch.device("cuda:2")]


def test_w2_123_autotune_fingerprints_and_tunes_the_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _import_autotune(monkeypatch)
    seen_indices: list[int] = []

    def properties(index: int):
        seen_indices.append(index)
        return SimpleNamespace(name=f"GPU-{index}", major=10, minor=index)

    monkeypatch.setattr(autotune.torch.cuda, "get_device_properties", properties)
    assert autotune._get_device_key(torch.device("cuda:3")) == "GPU-3_10.3"
    assert seen_indices == [3]

    selected: dict[str, object] = {}

    def select_kernel(M, N, K, verbose=True, device=None):  # noqa: N803
        selected.update(M=M, N=N, K=K, verbose=verbose, device=device)
        return "winner"

    sentinel = object()
    monkeypatch.setattr(autotune, "autotune", select_kernel)
    monkeypatch.setattr(autotune, "KERNELS", {"winner": lambda _a, _b: sentinel})
    entered_devices: list[torch.device] = []

    def device_context(device: torch.device):
        entered_devices.append(device)
        return contextlib.nullcontext()

    monkeypatch.setattr(autotune.torch.cuda, "device", device_context)
    a = SimpleNamespace(shape=(2, 4), device=torch.device("cuda:3"))
    b = SimpleNamespace(shape=(5, 4), device=torch.device("cuda:3"))
    assert autotune.matmul_autotuned(a, b) is sentinel
    assert selected == {
        "M": 2,
        "N": 5,
        "K": 4,
        "verbose": False,
        "device": torch.device("cuda:3"),
    }
    assert entered_devices == [torch.device("cuda:3")]
