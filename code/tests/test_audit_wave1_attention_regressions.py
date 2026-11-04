"""Attention/decode host contracts plus explicit, real CUDA acceptance gates."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core.benchmark.numerical_accuracy import (
    ScaleInvariantAccuracyLimits,
    assert_low_precision_attention_accuracy,
    low_precision_attention_limits,
)

CODE = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA runtime required; host checks are not GPU qualification")

_CUTE_ENVIRONMENT_VERSIONS = {
    "torch": "2.14.1+cu130",
    "triton": "3.8.0",
    "flash-attn-4": "4.0.0b36",
    "nvidia-cutlass-dsl": "4.8.0",
    "quack-kernels": "0.6.5",
    "apache-tvm-ffi": "0.1.11",
    "torch-c-dlpack-ext": "0.1.5+torch2141.nobackend1",
}


def _write_distribution(root: Path, name: str, version: str) -> None:
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def _run_cute_resolver(
    tmp_path: Path,
    *,
    public_api: bool,
    version_overrides: dict[str, str] | None = None,
    include_flash_attn_2: bool = False,
) -> subprocess.CompletedProcess[str]:
    versions = _CUTE_ENVIRONMENT_VERSIONS | (version_overrides or {})
    for distribution, version in versions.items():
        _write_distribution(tmp_path, distribution, version)
    if include_flash_attn_2:
        _write_distribution(tmp_path, "flash-attn", "2.8.3.post2")

    cute_package = tmp_path / "flash_attn" / "cute"
    cute_package.mkdir(parents=True)
    (tmp_path / "flash_attn" / "__init__.py").write_text("", encoding="utf-8")
    if public_api:
        cute_init = "def flash_attn_func(*args, **kwargs):\n    return 'public-fa4'\n"
    else:
        cute_init = ""
        (cute_package / "interface.py").write_text(
            "def _flash_attn_fwd(*args, **kwargs):\n    return 'private-fa2'\n",
            encoding="utf-8",
        )
    (cute_package / "__init__.py").write_text(cute_init, encoding="utf-8")

    python_path = [str(tmp_path), str(CODE)]
    if inherited := os.environ.get("PYTHONPATH"):
        python_path.append(inherited)
    environment = os.environ | {"PYTHONPATH": os.pathsep.join(python_path)}
    # Import the real tool first, then isolate metadata lookup to the fixture.
    # Installed FA2 must not leak into a control for the shared FA4 environment.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "from labs.flexattention.flex_attention_cute import _resolve_cute_forward; "
                "sys.path = [entry for entry in sys.path if not "
                "{'site-packages', 'dist-packages'}.intersection(Path(entry).parts)]; "
                "forward = _resolve_cute_forward(); "
                "assert forward.__module__ == 'flash_attn.cute'; "
                "assert forward() == 'public-fa4'"
            ),
        ],
        cwd=CODE,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )


def test_cudnn_intent_and_real_backend_context_are_exclusive():
    from labs.cudnn_sdpa_bench.baseline_flash_sdp import _sdpa_context, _select_backend
    assert _select_backend("cudnn") == "cudnn"  # Even when this host has no GPU.
    flags = torch.backends.cuda
    with _sdpa_context("cudnn"):
        assert flags.cudnn_sdp_enabled()
        assert not flags.flash_sdp_enabled()
        assert not flags.math_sdp_enabled()
        assert not flags.mem_efficient_sdp_enabled()
    with _sdpa_context("flash"):
        assert flags.flash_sdp_enabled() and not flags.cudnn_sdp_enabled()
    with pytest.raises(ValueError): _select_backend("invented")


def test_math_backend_executes_real_cpu_attention():
    from labs.cudnn_sdpa_bench.baseline_flash_sdp import _sdpa_context
    q = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 10
    with _sdpa_context("math"):
        result = F.scaled_dot_product_attention(q, -q, q.flip(-2))
    expected = torch.softmax(q @ (-q).transpose(-1, -2) / 2, dim=-1) @ q.flip(-2)
    torch.testing.assert_close(result, expected)


def test_cudnn_sdpa_baseline_isolated_worker_preserves_device_metadata():
    from core.benchmark.models import BenchmarkResult
    from core.harness.isolated_runner import run_benchmark
    from labs.cudnn_sdpa_bench.baseline_flash_sdp import FlashSDPLabBenchmark

    assert FlashSDPLabBenchmark().backend == "math"

    response = run_benchmark(
        {
            "benchmark_module_path": str(
                CODE / "labs" / "cudnn_sdpa_bench" / "baseline_flash_sdp.py"
            ),
            "benchmark_class_name": "FlashSDPLabBenchmark",
            "config_dict": {
                "iterations": 1,
                "warmup": 5,
                "measurement_timeout_seconds": 2,
                "setup_timeout_seconds": 2,
                "enforce_environment_validation": False,
                "monitor_gpu_state": False,
                "track_memory_allocations": False,
                "clear_l2_cache": False,
                "lock_gpu_clocks": False,
            },
            "mode": "custom",
            "device": "cpu",
            "initial_state": {"backend": "cudnn"},
        }
    )

    result = BenchmarkResult.model_validate_json(response["result_json"])
    assert result.device == "cpu"
    assert not any("missing 1 required positional argument: 'backend'" in error for error in result.errors)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fault", ["correct", "zero", "half_scale", "last_element", "nan", "alias", "short"])
def test_attention_accuracy_gate_checks_complete_scale_invariant_output(dtype, fault):
    rounded = (torch.arange(1, 1 + 2 * 3 * 4 * 8, dtype=torch.float32) / 128).reshape(2, 3, 4, 8).to(dtype)
    expected = rounded.double()
    actual = rounded.clone()
    if fault == "zero":
        actual.zero_()
    elif fault == "half_scale":
        actual.mul_(0.5)
    elif fault == "last_element":
        actual[-1, -1, -1, -1] += 0.25
    elif fault == "nan":
        actual[-1, -1, -1, -1] = float("nan")
    elif fault == "alias":
        expected = actual
    elif fault == "short":
        actual = actual[..., :-1]

    limits = low_precision_attention_limits(dtype)
    assert limits.relative_l2 == torch.finfo(dtype).eps
    assert limits.normalized_max_abs == torch.finfo(dtype).eps
    if fault == "correct":
        assert assert_low_precision_attention_accuracy(actual, expected) == {
            "relative_l2": 0.0,
            "normalized_max_abs": 0.0,
        }
    else:
        with pytest.raises(AssertionError):
            assert_low_precision_attention_accuracy(actual, expected)


@pytest.mark.parametrize("bad", [False, True, -1e-9, float("nan"), float("inf"), "0", None])
def test_scale_invariant_limits_reject_malformed_values(bad):
    with pytest.raises(ValueError):
        ScaleInvariantAccuracyLimits(bad, 0)
    with pytest.raises(ValueError):
        ScaleInvariantAccuracyLimits(0, bad)

    limits = ScaleInvariantAccuracyLimits(0.25, 0.25)
    for metric in ("relative_l2", "normalized_max_abs"):
        errors = {"relative_l2": 0.0, "normalized_max_abs": 0.0}
        errors[metric] = bad
        with pytest.raises(ValueError):
            limits.check(errors, label="adverse-control")


@pytest.mark.parametrize("mode", ["dense", "causal", "alibi", "softcap", "windowed", "alibi_windowed"])
@pytest.mark.parametrize("seq", [0, 1, 7, 64])
def test_flop_count_matches_actual_reference_mask(mode, seq):
    from labs.flashattention4.flashattention4_common import build_dense_attention_mask, count_nonmasked_attention_elements
    mask = build_dense_attention_mask(mode, seq_len=seq, window_size=3, device=torch.device("cpu"))
    actual = seq * seq if mask is None else int(mask.count_nonzero())
    assert count_nonmasked_attention_elements(mode, q_seq_len=seq, kv_seq_len=seq, window_size=3) == actual


@pytest.mark.parametrize("q_len,k_len", [(3, 7), (7, 3), (1, 1), (0, 7)])
def test_rectangular_causal_flops(q_len, k_len):
    from labs.flashattention4.flashattention4_common import count_nonmasked_attention_elements
    expected = int((torch.arange(q_len)[:, None] >= torch.arange(k_len)[None, :]).sum())
    for mode in ("causal", "alibi", "softcap"):
        assert count_nonmasked_attention_elements(mode, q_seq_len=q_len, kv_seq_len=k_len) == expected


def test_cute_layout_preserves_values_and_shape():
    from labs.flexattention.flex_attention_cute import to_cute_layout
    q = torch.arange(2 * 3 * 7 * 5).reshape(2, 3, 7, 5)
    converted = to_cute_layout(q, q + 100, q - 100)
    for original, result in zip((q, q + 100, q - 100), converted):
        assert result.shape == (2, 7, 3, 5) and result.is_contiguous()
        torch.testing.assert_close(result.transpose(1, 2), original, rtol=0, atol=0)
    with pytest.raises(ValueError): to_cute_layout(q, q[:, :, :1], q)
    result = subprocess.run([sys.executable, "-m", "labs.flexattention.flex_attention_cute", "--help"],
        cwd=CODE, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("with_logsumexp", [False, True])
def test_cute_public_pair_preserves_output_tensor(with_logsumexp):
    from labs.flexattention.flex_attention_cute import extract_cute_output

    output = torch.randn(2, 4, 3, 8)
    logsumexp = torch.randn(2, 3, 4) if with_logsumexp else None
    assert extract_cute_output((output, logsumexp)) is output


@pytest.mark.parametrize("result", [None, (), (None, None), (1, None), (None,), (None, None, None)])
def test_cute_public_pair_rejects_invalid_return(result):
    from labs.flexattention.flex_attention_cute import extract_cute_output

    with pytest.raises(RuntimeError):
        extract_cute_output(result)


def test_cute_public_pair_rejects_invalid_logsumexp():
    from labs.flexattention.flex_attention_cute import extract_cute_output

    with pytest.raises(RuntimeError, match="logsumexp must be a tensor or None"):
        extract_cute_output((torch.ones(1), "invalid"))


def test_cute_environment_uses_shared_fa4_stack_and_preserves_separate_fa2_contract():
    import json

    from labs.flexattention.flex_attention_cute import CUTE_ENVIRONMENT_VERSIONS

    requirements = (CODE / "labs/flexattention/requirements_cute.txt").read_text(encoding="utf-8")
    active_lines = [
        line for line in requirements.splitlines() if line and not line.startswith("#")
    ]
    assert active_lines == ["-r ../../requirements_torch2141_linux_x86_64.txt"]

    lock = (CODE / "requirements_torch2141_linux_x86_64.txt").read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in lock
    assert "--find-links vendor/python-wheels" in lock
    for distribution, version in CUTE_ENVIRONMENT_VERSIONS.items():
        assert f"{distribution}=={version} " in lock
    assert CUTE_ENVIRONMENT_VERSIONS == _CUTE_ENVIRONMENT_VERSIONS

    setup = (CODE / "setup.sh").read_text(encoding="utf-8")
    default_requirements = (CODE / "requirements_latest.txt").read_text(encoding="utf-8")
    default_requirement_lines = {
        line
        for line in default_requirements.splitlines()
        if line and not line.startswith(("#", "--"))
    }
    assert "torch==2.14.1+cu130" in default_requirement_lines
    assert "flash-attn-4[cu13]==4.0.0b36" in default_requirement_lines
    assert not any(line.startswith("flash-attn==") for line in default_requirement_lines)

    # The shared runtime installs FA4 and only rebuilds Transformer Engine.
    # FA2 owns the same flash_attn namespace, so its source build is an explicit
    # contract for a separate, otherwise fresh environment.
    assert '--package transformer-engine-torch' in setup
    assert '--package flash-attn' not in setup
    assert 'metadata.distribution("flash-attn")' in setup
    assert 'metadata.version("flash-attn-4") != "4.0.0b36"' in setup

    native_contract = json.loads(
        (CODE / "native_torch_extensions.json").read_text(encoding="utf-8")
    )
    assert native_contract["runtime"]["torch"] == "2.14.1+cu130"
    fa2 = next(
        package
        for package in native_contract["packages"]
        if package["id"] == "flash-attn"
    )
    assert (
        fa2["source"]["declared_upstream_commit"]
        == "49774b31c8d1172a5b28f72e9aed0f322bf4ccba"
    )
    assert (
        fa2["sdist"]["sha256"]
        == "d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"
    )
    assert fa2["expected_metadata"]["version"] == "2.8.3.post2"
    assert fa2["forbidden_distributions"] == ["flash-attn", "flash-attn-4"]
    assert fa2["build_environment"]["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"
    assert fa2["build_environment"]["FLASH_ATTN_CUDA_ARCHS"] == "100"


def test_cute_resolver_uses_public_fa4_api_from_exact_isolated_environment(tmp_path):
    result = _run_cute_resolver(tmp_path, public_api=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "defect,expected_error",
    [
        ("private_api_only", "must export public flash_attn.cute.flash_attn_func"),
        ("wrong_fa4_version", "flash-attn-4==4.0.0b19, expected 4.0.0b36"),
        ("fa2_overlap", "must not contain flash-attn==2.8.3.post2"),
    ],
)
def test_cute_resolver_rejects_private_or_nonisolated_environments(
    tmp_path, defect, expected_error
):
    result = _run_cute_resolver(
        tmp_path,
        public_api=defect != "private_api_only",
        version_overrides={"flash-attn-4": "4.0.0b19"}
        if defect == "wrong_fa4_version"
        else None,
        include_flash_attn_2=defect == "fa2_overlap",
    )
    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize("prefetch", [False, True])
def test_host_stage_waits_for_ownership_fence_before_actual_cpu_write(prefetch):
    from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark
    bench = PagedKVOffloadBenchmark(PagedKVConfig(batch_size=1, num_heads=1, head_dim=2,
        max_seq_len=4, page_tokens=2, prefer_fp8=False))
    bench.host_cache = torch.arange(16, dtype=torch.float32).reshape(2, 1, 1, 4, 2)
    bench.staging = torch.full((2, 1, 1, 2, 2), -999.)
    bench.prefetch_staging = torch.full_like(bench.staging, -999.)
    target = bench.prefetch_staging if prefetch else bench.staging
    waiting, released = threading.Event(), threading.Event()
    # This is a host ownership-fence test, not a simulated GPU correctness result.
    class HostFence:
        def synchronize(self):
            waiting.set()
            if not released.wait(3): raise RuntimeError("test fence timeout")
    bench._staging_copy_events[id(target)] = HostFence()
    bench._staging_copy_pending.add(id(target))
    errors = []
    def stage():
        try: bench._stage_page(2, into_prefetch=prefetch)
        except BaseException as exc: errors.append(exc)
    worker = threading.Thread(target=stage)
    worker.start()
    try:
        assert waiting.wait(2)
        assert torch.all(target == -999)
    finally:
        released.set(); worker.join(3)
    assert not worker.is_alive() and not errors
    torch.testing.assert_close(target, bench.host_cache[..., 2:4, :])
    assert id(target) not in bench._staging_copy_pending


@pytest.mark.parametrize("fault", ["correct", "dropped_tail", "nan", "corrupt", "alias"])
def test_decode_actual_full_output_verifier(fault):
    from labs.persistent_decode.persistent_decode_common import DecodeInputs, validate_decode_output
    q = torch.full((12, 9, 8), 2.)
    k, v = torch.full_like(q, 3), torch.arange(q.numel(), dtype=torch.float32).reshape_as(q) + 1
    out = (q * k).sum(-1, keepdim=True) * v
    if fault == "dropped_tail": out[8:].zero_()
    if fault == "nan": out[-1, -1, -1] = float("nan")
    if fault == "corrupt": out[-1, -1, -1] *= 2
    if fault == "alias": out = q
    data = DecodeInputs(q, k, v, out, torch.arange(12), torch.full((12,), 9), torch.zeros(1))
    if fault == "correct": validate_decode_output(data)
    else:
        with pytest.raises(AssertionError): validate_decode_output(data)


def test_decode_program_overrides_reject_invalid_values_and_rng_is_harness_owned():
    from labs.persistent_decode.persistent_decode_common import DecodeOptions, get_decode_options, set_decode_options, get_decode_profile, build_inputs
    old = get_decode_options()
    try:
        for programs in (0, -1):
            set_decode_options(DecodeOptions(num_programs=programs))
            with pytest.raises(ValueError): get_decode_profile()
        set_decode_options(DecodeOptions(tier="large", num_programs=1))
        assert get_decode_profile().num_programs == 1
        with torch.random.fork_rng():
            torch.manual_seed(17); a = build_inputs(device=torch.device("cpu")).q
            torch.manual_seed(18); b = build_inputs(device=torch.device("cpu")).q
        assert not torch.equal(a, b)
    finally:
        set_decode_options(old)


@CUDA
@pytest.mark.parametrize("programs", [1, 8, 12])
def test_real_triton_grid_stride_covers_every_sequence(programs):
    from labs.persistent_decode.optimized_persistent_decode_triton import persistent_decode_kernel
    from labs.persistent_decode.persistent_decode_common import build_inputs, validate_decode_output
    data = build_inputs(12, 32, 64, torch.device("cuda"))
    for _ in range(3):
        data.out.fill_(float("nan"))
        persistent_decode_kernel[(programs,)](data.q, data.k, data.v, data.out, data.work_seq_ids,
            data.work_steps, 12, head_dim=64, max_steps=32, BLOCK_K=64, num_warps=2, num_stages=1)
        torch.cuda.synchronize()
        validate_decode_output(data)


@CUDA
def test_real_cuda_reduction_repeated_multwarp_reuse():
    from labs.persistent_decode.optimized_persistent_decode_cuda import _load_extension
    ext = _load_extension()
    for dim in (33, 64, 128):
        q, k, v = [torch.randn(12, 64, dim, device="cuda") for _ in range(3)]
        expected = (q.double() * k.double()).sum(-1, keepdim=True) * v.double()
        out = torch.empty_like(q)
        for _ in range(25):
            ext.persistent_decode(q, k, v, out, 2)
            torch.testing.assert_close(out, expected.float())


@CUDA
@pytest.mark.parametrize("seq,dim", [(1, 16), (63, 32), (65, 40), (129, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_real_triton_attention_tail_and_causal(seq, dim, causal):
    from labs.flashattention_gluon.flashattention_gluon_common import gluon_flash_attention
    q = torch.full((2, 3, seq, dim), .25, device="cuda", dtype=torch.float16)
    k = -torch.rand_like(q) - .5  # Real scores negative; zero-padded phantom keys would dominate.
    v = torch.randn_like(q)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        # Keep the reference independent of import-time TF32 configuration.
        expected = F.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), is_causal=causal
        ).to(q.dtype)
    actual = gluon_flash_attention(q, k, v, causal=causal)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(NotImplementedError): gluon_flash_attention(q, k, v, dropout_p=.1)


@CUDA
def test_real_cute_layout_and_attention():
    from labs.flexattention.flex_attention_cute import (
        _resolve_cute_forward,
        extract_cute_output,
        to_cute_layout,
    )
    forward = _resolve_cute_forward()
    q, k, v = [torch.randn(2, 3, 64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    result = forward(*to_cute_layout(q, k, v))
    actual = extract_cute_output(result)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        # Independent reference: ambient FP32/TF32 policy must not define the
        # expected result of a different fused attention implementation.
        expected = F.scaled_dot_product_attention(
            q.double(), k.double(), v.double()
        ).transpose(1, 2)
    assert_low_precision_attention_accuracy(actual, expected)


@CUDA
@pytest.mark.parametrize("copy_stream", [False, True])
def test_real_pinned_stage_dma_reuse(copy_stream):
    from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark
    cfg = PagedKVConfig(batch_size=1, num_heads=1, head_dim=64, max_seq_len=256,
        page_tokens=128, decode_tokens=1, repeat_pages=4, use_pinned_stage=True,
        use_async_stream=copy_stream, use_memmap=True, prefer_fp8=False)
    bench = PagedKVOffloadBenchmark(cfg)
    bench.setup()
    try:
        for repeat in range(8):
            for start in (0, 128):
                staged, count = bench._stage_page(start)
                expected = staged.clone()
                bench._copy_to_device(staged, count)
                # Immediately reuse CPU storage; _stage_page must wait for its read.
                bench._stage_page(128 - start)
                torch.cuda.synchronize()
                torch.testing.assert_close(bench.hot_k_bufs[0], expected[0].cuda())
                torch.testing.assert_close(bench.hot_v_bufs[0], expected[1].cuda())
    finally:
        bench.teardown()


@CUDA
def test_real_async_input_allocation_lifetime_and_actual_stream():
    from core.common.async_input_pipeline import AsyncInputPipelineBenchmark, PipelineConfig
    bench = AsyncInputPipelineBenchmark(PipelineConfig(batch_size=2, feature_shape=(3, 16, 16),
        dataset_size=8, pin_memory=True, non_blocking=True, use_copy_stream=True))
    bench.setup()
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    snapshots = []
    try:
        with torch.cuda.stream(consumer), torch.inference_mode():
            for _ in range(12):
                bench.benchmark_fn()
                snapshots.append((bench._last_batch.clone(), bench.output.clone()))
        consumer.synchronize()
        with torch.inference_mode():
            for batch, actual in snapshots:
                torch.testing.assert_close(actual, bench.model(batch))
        assert bench.compute_stream == consumer
    finally:
        torch.cuda.synchronize(); bench.teardown()


@CUDA
def test_real_flashmla_book_kernel_stable_full_dot(tmp_path):
    from torch.utils.cpp_extension import load_inline
    source = '#define FLASHMLA_NO_MAIN\n#include "' + str(CODE / "ch18/flashmla_kernel.cu") + '"\n'
    source += r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
torch::Tensor run_flashmla(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor lengths) {
    auto out = torch::empty_like(q);
    int d = q.size(2), threads = 1; while (threads < d) threads *= 2;
    flashmla_decode<<<q.size(0)*q.size(1),threads,0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()), reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        lengths.data_ptr<int>(), q.size(1), d, k.stride(0));
    AT_CUDA_CHECK(cudaGetLastError()); return out;
}
'''
    ext = load_inline(name="audit_flashmla_reference", cpp_sources="torch::Tensor run_flashmla(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
        cuda_sources=source, functions=["run_flashmla"], build_directory=str(tmp_path), verbose=False)
    for dim in (33, 64, 128, 1024):
        q = torch.randn(3, 3, dim, device="cuda", dtype=torch.float16) * 20
        k, v = [torch.randn(3, 7, 3, dim, device="cuda", dtype=torch.float16) for _ in range(2)]
        lengths = torch.tensor([7, 3, 0], device="cuda", dtype=torch.int32)
        actual = ext.run_flashmla(q, k, v, lengths)
        for batch, count in enumerate((7, 3, 0)):
            if count == 0:
                torch.testing.assert_close(actual[batch], torch.zeros_like(actual[batch]), rtol=0, atol=0)
                continue
            score = torch.einsum("hd,shd->hs", q[batch].double(), k[batch, :count].double()) / dim ** .5
            expected = torch.einsum("hs,shd->hd", score.softmax(-1), v[batch, :count].double()).half()
            torch.testing.assert_close(actual[batch], expected)


def test_host_prefetch_failed_join_preserves_worker_owned_storage():
    from labs.persistent_decode.paged_kv_offload_common import PagedKVOffloadBenchmark
    bench = PagedKVOffloadBenchmark()
    released = threading.Event()
    worker = threading.Thread(target=released.wait)
    bench._prefetch_thread = worker
    bench.prefetch_staging = torch.ones(2)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="cannot release"):
            bench.teardown()
        assert bench._prefetch_thread is worker and bench.prefetch_staging is not None
    finally:
        released.set(); worker.join(3)
        bench._stop_host_prefetch_thread()


@CUDA
def test_real_cudnn_pin_profiles_actual_model_dispatch(tmp_path):
    from labs.cudnn_sdpa_bench.baseline_flash_sdp import (
        FlashSDPLabBenchmark,
        SDPAAttentionModule,
    )
    model = SDPAAttentionModule(hidden_dim=512, num_heads=8, backend="cudnn").cuda().half().eval()
    inputs = torch.randn(8, 256, 512, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        model(inputs)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            actual = model(inputs)
        torch.cuda.synchronize()
        prof.export_chrome_trace(str(tmp_path / "cudnn-actual-model.json"))
        keys = {event.key for event in prof.key_averages()}
        assert "aten::_scaled_dot_product_cudnn_attention" in keys, sorted(keys)
        assert "aten::_scaled_dot_product_flash_attention" not in keys
        # Compare the dispatched attention against an independent FP64 oracle
        # using the same projected FP16 inputs. Harness import enables TF32,
        # so selecting MATH alone does not make its FP32 matmuls a strict oracle.
        q, k, v = model.qkv(inputs).view(8, 256, 3, 8, 64).unbind(dim=2)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(
                q.transpose(1, 2).double(), k.transpose(1, 2).double(),
                v.transpose(1, 2).double(),
            ).transpose(1, 2).reshape_as(actual)
        assert_low_precision_attention_accuracy(actual, expected)

        # Exercise the benchmark's production capture path with the exact
        # dispatched output, then prove a corrupted full output cannot pass.
        benchmark = FlashSDPLabBenchmark(backend="cudnn")
        benchmark.model = model
        benchmark.inputs = inputs
        benchmark.output = actual
        benchmark._payload_parameter_count = sum(p.numel() for p in model.parameters())
        benchmark.capture_verification_payload()
        benchmark.output = actual.clone()
        benchmark.output[..., -1].add_(1)
        with pytest.raises(AssertionError, match="attention accuracy failed"):
            benchmark.capture_verification_payload()


@CUDA
@pytest.mark.parametrize("mode", ["full", "piecewise", "full_and_piecewise"])
def test_real_decode_graphs_cover_all_sequences_with_one_program(mode):
    from labs.persistent_decode.optimized_persistent_decode_graphs import OptimizedPersistentDecodeGraphsBenchmark, GraphMode
    from labs.persistent_decode.persistent_decode_common import DecodeOptions, get_decode_options, set_decode_options, validate_decode_output
    old = get_decode_options()
    bench = None
    try:
        set_decode_options(DecodeOptions(tier="large", num_programs=1))
        bench = OptimizedPersistentDecodeGraphsBenchmark(graph_mode=GraphMode(mode))
        bench.setup()
        for _ in range(3):
            bench.inputs.out.fill_(float("nan"))
            bench.benchmark_fn()
            torch.cuda.synchronize()
            validate_decode_output(bench.inputs)
            assert bench.output.shape == (12, 64, 64)
    finally:
        if bench is not None: bench.teardown()
        set_decode_options(old)


@CUDA
@pytest.mark.parametrize("host_worker", [False, True])
def test_real_paged_prefetch_repeated_page_content(host_worker):
    from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark
    cfg = PagedKVConfig(batch_size=1, num_heads=2, head_dim=64, max_seq_len=256,
        page_tokens=128, decode_tokens=2, repeat_pages=3, use_pinned_stage=True,
        use_async_stream=True, use_memmap=True, prefer_fp8=False, prefetch_next_page=True,
        use_host_prefetch_thread=host_worker)
    bench = PagedKVOffloadBenchmark(cfg)
    bench.setup()
    try:
        for _ in range(8):
            last_start = (bench.page_cursor + (cfg.repeat_pages - 1) * cfg.page_tokens) % cfg.max_seq_len
            expected_kv = bench.host_cache[..., last_start:last_start + cfg.page_tokens, :].clone().cuda()
            bench.benchmark_fn()
            torch.cuda.synchronize()
            torch.testing.assert_close(bench._payload_k, expected_kv[0])
            torch.testing.assert_close(bench._payload_v, expected_kv[1])
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                # Preserve reference precision even when harness defaults have
                # globally enabled TF32 for performance measurements.
                expected = F.scaled_dot_product_attention(
                    bench.q.double(), expected_kv[0].double(), expected_kv[1].double(),
                )
            assert_low_precision_attention_accuracy(
                bench.output, expected[:, :, :1, :bench._verify_head_dim]
            )
            bench.capture_verification_payload()
        bench.output[..., -1].add_(1)
        with pytest.raises(AssertionError, match="attention accuracy failed"):
            bench.capture_verification_payload()
    finally:
        bench.teardown()


@CUDA
def test_real_thread_async_copy_eager_and_graph_current_stream():
    from labs.persistent_decode.tma_extension import load_async_copy
    ext = load_async_copy()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for length in (0, 1, 127, 128, 129, 4097):
        with torch.cuda.stream(stream):
            src = torch.arange(length, device="cuda", dtype=torch.float32)
            out = torch.empty_like(src)
            ext.async_copy(src, out)
            torch.testing.assert_close(out, src, rtol=0, atol=0)
        if length:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream): ext.async_copy(src, out)
            with torch.cuda.stream(stream):
                for step in range(3):
                    src.add_(step + 1); out.fill_(float("nan")); graph.replay()
                    torch.testing.assert_close(out, src, rtol=0, atol=0)
    overlapping = torch.arange(130, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="overlap"):
        ext.async_copy(overlapping[:-1], overlapping[1:])
    stream.synchronize()


@CUDA
@pytest.mark.parametrize("seed", [42, 1042])
def test_real_eager_vs_graph_persistent_decode_repeated_states(seed: int):
    from tests.protection_test_utils import preserve_rng_state

    with preserve_rng_state():
        from labs.decode_optimization.decode_common import DecodeConfig
        from labs.decode_optimization.baseline_decode_warp_specialized import PersistentPrefillBaselineBenchmark
        from labs.decode_optimization.optimized_decode_warp_specialized import CUDAGraphPersistentDecodeBenchmark
        cfg = DecodeConfig(batch_size=2, prompt_tokens=16, decode_tokens=4, hidden_size=64,
            vocab_size=128, use_pinned_host=True, use_copy_stream=True, use_compute_stream=True)
        baseline, graph = PersistentPrefillBaselineBenchmark(cfg), CUDAGraphPersistentDecodeBenchmark(cfg)
        torch.manual_seed(seed)
        baseline.setup()
        torch.manual_seed(seed)
        graph.setup()
        consumer = torch.cuda.Stream()
        consumer.wait_stream(torch.cuda.current_stream())
        try:
            torch.testing.assert_close(baseline._prefilled_state, graph._prefilled_state, rtol=0, atol=0)
            for _ in range(4):
                with torch.cuda.stream(consumer):
                    baseline.benchmark_fn(); graph.benchmark_fn()
                    torch.testing.assert_close(graph.state_buffer, baseline.state_buffer)
                    torch.testing.assert_close(graph.current_tokens, baseline.current_tokens, rtol=0, atol=0)
            consumer.synchronize()
        finally:
            torch.cuda.synchronize(); baseline.teardown(); graph.teardown()
