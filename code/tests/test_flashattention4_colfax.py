"""Correctness and discovery coverage for the pinned Colfax FA4 lab pairs."""

from __future__ import annotations

import importlib
import math
import os
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BaseBenchmark
from core.harness.validity_checks import check_setup_precomputation
from labs.flashattention4.colfax_benchmarks import (
    ColfaxBenchmark,
    ColfaxConfig,
    build_inputs,
    default_config,
    load_upstream,
    python_source_tree_fingerprint,
    reference_attention,
    source_manifest,
    verify_python_source_tree,
    verify_source_files,
    verify_vcs_direct_url,
)
from tests.protection_test_utils import preserve_rng_state

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "flashattention4"
EXPECTED_PINS = {
    "decode": {
        "pr": 2817,
        "commit": "a93c9a8fb95516a08e0974743bd5723122613377",
        "python_source_tree": {
            "format": "sha256-path-content-sha256-v1",
            "file_count": 52,
            "sha256": "5fc8266486cef1883118d18278eefe4d320556c32365b59f842649827354117a",
        },
    },
    "backward": {
        "pr": 2804,
        "commit": "c33d03d9f3edc850ecb5e21466d5dd31331ca7b6",
        "python_source_tree": {
            "format": "sha256-path-content-sha256-v1",
            "file_count": 52,
            "sha256": "019cd50a4f70211405703f926cd64f691c71e03773a722bb49ea878a2552a5ba",
        },
    },
}


def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Deliberately scalar CPU oracle, independent of the module implementation."""
    batch, query_tokens, query_heads, head_dim = q.shape
    key_tokens, kv_heads = k.shape[1:3]
    group_size = query_heads // kv_heads
    output = torch.empty_like(q, dtype=torch.float64)
    scale = head_dim**-0.5

    for batch_index in range(batch):
        for query_index in range(query_tokens):
            for query_head in range(query_heads):
                kv_head = query_head // group_size
                scores = [
                    sum(
                        float(q[batch_index, query_index, query_head, dim])
                        * float(k[batch_index, key_index, kv_head, dim])
                        for dim in range(head_dim)
                    )
                    * scale
                    for key_index in range(key_tokens)
                ]
                score_max = max(scores)
                unnormalized = [math.exp(score - score_max) for score in scores]
                normalizer = sum(unnormalized)
                for dim in range(head_dim):
                    output[batch_index, query_index, query_head, dim] = (
                        sum(
                            weight * float(v[batch_index, key_index, kv_head, dim])
                            for key_index, weight in enumerate(unnormalized)
                        )
                        / normalizer
                    )
    return output


def _torch_attention_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    causal: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Independent CPU FP64 PyTorch oracle, including autograd for backward."""
    requires_grad = "dout" in inputs
    q = inputs["q"].detach().cpu().double().requires_grad_(requires_grad)
    k = inputs["k"].detach().cpu().double().requires_grad_(requires_grad)
    v = inputs["v"].detach().cpu().double().requires_grad_(requires_grad)
    group_size = q.shape[2] // k.shape[2]
    expanded_k = k.repeat_interleave(group_size, dim=2)
    expanded_v = v.repeat_interleave(group_size, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, expanded_k) * q.shape[-1] ** -0.5
    if causal:
        query_positions = torch.arange(q.shape[1], device=q.device)[:, None]
        key_positions = torch.arange(k.shape[1], device=q.device)[None, :]
        bottom_right = query_positions + k.shape[1] - q.shape[1]
        scores = scores.masked_fill(key_positions > bottom_right, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_v)
    if not requires_grad:
        return output, None
    gradients = torch.autograd.grad(
        output,
        (q, k, v),
        grad_outputs=inputs["dout"].detach().cpu().double(),
    )
    return output, gradients


def test_default_configs_are_frozen_and_match_the_two_colfax_workloads() -> None:
    decode = default_config("decode")
    backward = default_config("backward")

    assert decode == ColfaxConfig("decode", 32, 1, 131072, 16, 1, 64)
    assert backward == ColfaxConfig("backward", 4, 16384, 16384, 32, 32, 64)
    decode.validate()
    backward.validate()
    with pytest.raises(FrozenInstanceError):
        decode.head_dim = 128  # type: ignore[misc]
    with pytest.raises(ValueError, match="kind must be decode or backward"):
        default_config("prefill")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(default_config("decode"), kind="prefill"), "kind must be"),
        (replace(default_config("decode"), batch_size=0), "batch_size must be"),
        (replace(default_config("decode"), seqlen_q=True), "seqlen_q must be"),
        (replace(default_config("decode"), seqlen_k=-1), "seqlen_k must be"),
        (replace(default_config("decode"), query_heads=0), "query_heads must be"),
        (replace(default_config("decode"), kv_heads=0), "kv_heads must be"),
        (replace(default_config("decode"), head_dim=0), "head_dim must be"),
        (replace(default_config("decode"), query_heads=15, kv_heads=2), "divisible"),
        (replace(default_config("decode"), seqlen_q=257, seqlen_k=256), "must not exceed"),
        (replace(default_config("decode"), head_dim=32), "head_dim 64 or 128"),
        (
            replace(default_config("decode"), seqlen_q=17, seqlen_k=256),
            "packed decode queries",
        ),
        (replace(default_config("decode"), seqlen_k=128), "at least two"),
        (replace(default_config("backward"), head_dim=128), "head_dim 64"),
        (replace(default_config("backward"), seqlen_q=1024), "equal Q and KV"),
    ],
)
def test_config_validation_rejects_invalid_or_unsupported_shapes(
    config: ColfaxConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_benchmark_constructor_rejects_a_mismatched_config_kind() -> None:
    with pytest.raises(ValueError, match="config kind must match benchmark kind"):
        ColfaxBenchmark("decode", optimized=False, config=default_config("backward"))


def test_build_inputs_uses_the_callers_cpu_rng_stream() -> None:
    config = ColfaxConfig("backward", 1, 2, 2, 2, 2, 64)
    shapes = {
        "q": (1, 2, 2, 64),
        "k": (1, 2, 2, 64),
        "v": (1, 2, 2, 64),
        "dout": (1, 2, 2, 64),
    }

    with preserve_rng_state():
        snapshots = []
        for seed in (20260907, 20260907, 20260908):
            torch.manual_seed(seed)
            tensors = build_inputs(config, torch.device("cpu"))
            assert torch.initial_seed() == seed
            assert {name: tuple(tensor.shape) for name, tensor in tensors.items()} == shapes
            assert all(tensor.dtype == torch.bfloat16 for tensor in tensors.values())
            snapshots.append({name: tensor.clone() for name, tensor in tensors.items()})

    assert all(torch.equal(snapshots[0][name], snapshots[1][name]) for name in shapes)
    assert any(not torch.equal(snapshots[0][name], snapshots[2][name]) for name in shapes)

    with preserve_rng_state():
        torch.manual_seed(4242)
        _ = torch.randn(11)
        build_inputs(config, torch.device("cpu"))
        actual_tail = torch.randn(8)

        torch.manual_seed(4242)
        _ = torch.randn(11)
        for shape in shapes.values():
            _ = torch.randn(shape, dtype=torch.bfloat16)
        expected_tail = torch.randn(8)

    torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)


def test_build_inputs_only_adds_dout_for_backward() -> None:
    decode = ColfaxConfig("decode", 1, 1, 256, 2, 1, 64)
    backward = ColfaxConfig("backward", 1, 2, 2, 2, 2, 64)

    with preserve_rng_state():
        torch.manual_seed(7)
        decode_inputs = build_inputs(decode, torch.device("cpu"))
        torch.manual_seed(7)
        backward_inputs = build_inputs(backward, torch.device("cpu"))

    assert set(decode_inputs) == {"q", "k", "v"}
    assert set(backward_inputs) == {"q", "k", "v", "dout"}


def test_reference_attention_expands_gqa_heads_against_a_scalar_oracle() -> None:
    q = torch.tensor(
        [
            [
                [[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7], [0.1, -0.8]],
                [[-0.3, 0.9], [0.4, -0.2], [0.8, 0.1], [-0.5, 0.6]],
            ]
        ],
        dtype=torch.float64,
    )
    k = torch.tensor(
        [[[[0.3, 0.1], [-0.2, 0.4]], [[0.7, -0.5], [0.6, 0.2]], [[-0.1, 0.8], [0.5, -0.7]]]],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [[[[1.0, -1.0], [0.5, 0.2]], [[0.0, 2.0], [-0.5, 1.5]], [[3.0, 0.5], [2.0, -2.0]]]],
        dtype=torch.float64,
    )

    actual = reference_attention(q, k, v)
    expected = _manual_attention(q, k, v)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_reference_attention_uses_bottom_right_causal_alignment() -> None:
    q = torch.zeros((1, 2, 1, 1), dtype=torch.float32)
    k = torch.zeros((1, 4, 1, 1), dtype=torch.float32)
    v = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)

    actual = reference_attention(q, k, v, causal=True)

    # Q row 0 aligns with K row 2 and can see K[0:3]; Q row 1 can see all K.
    expected = torch.tensor([2.0, 2.5]).reshape(1, 2, 1, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_source_manifests_and_requirement_files_pin_distinct_exact_revisions() -> None:
    manifests = {kind: source_manifest(kind) for kind in EXPECTED_PINS}

    for kind, expected in EXPECTED_PINS.items():
        manifest = manifests[kind]
        assert manifest["pr"] == expected["pr"]
        assert manifest["commit"] == expected["commit"]
        assert {
            key: manifest["python_source_tree"][key]
            for key in ("format", "file_count", "sha256")
        } == expected["python_source_tree"]
        assert re.fullmatch(r"[0-9a-f]{64}", manifest["python_source_tree"]["sha256"])
        assert manifest["python_source_tree"]["include"] == "**/*.py"
        assert manifest["python_source_tree"]["excluded_generated"] == [
            "**/__pycache__/**",
            "**/*.pyc",
            "../flash_attn_4-*.dist-info/**",
        ]
        assert manifest["installed_vcs"] == {
            "distribution": "flash-attn-4",
            "subdirectory": "flash_attn/cute",
            "metadata": "direct_url.json",
        }
        requirement = (LAB_DIR / f"requirements_colfax_{kind}.txt").read_text(encoding="utf-8")
        assert (
            f"flash-attention.git@{expected['commit']}#subdirectory=flash_attn/cute" in requirement
        )

    assert manifests["decode"]["commit"] != manifests["backward"]["commit"]


@pytest.mark.parametrize("kind", ["decode", "backward"])
def test_source_verification_rejects_a_real_directory_with_mismatched_files(
    tmp_path: Path,
    kind: str,
) -> None:
    (tmp_path / "interface.py").write_text(f"not Colfax {kind}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=rf"SKIPPED: Colfax {kind}.*source tree mismatch"):
        verify_source_files(tmp_path, kind)


def test_python_source_tree_fingerprint_rejects_a_tampered_helper(tmp_path: Path) -> None:
    (tmp_path / "interface.py").write_text("from .helper import run\n", encoding="utf-8")
    helper = tmp_path / "helper.py"
    helper.write_text("def run(): return 1\n", encoding="utf-8")
    expected = python_source_tree_fingerprint(tmp_path)

    helper.write_text("def run(): return 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Python source tree mismatch"):
        verify_python_source_tree(tmp_path, expected, kind="decode", commit="fixture")


@pytest.mark.parametrize("kind", ["decode", "backward"])
def test_vcs_direct_url_requires_the_exact_requested_commit(kind: str) -> None:
    manifest = source_manifest(kind)
    direct_url = {
        "subdirectory": "flash_attn/cute",
        "url": f"{manifest['repository']}.git",
        "vcs_info": {
            "commit_id": manifest["commit"],
            "requested_revision": manifest["commit"],
            "vcs": "git",
        },
    }
    assert verify_vcs_direct_url(direct_url, kind) == manifest

    direct_url["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(RuntimeError, match="requires an exact VCS install"):
        verify_vcs_direct_url(direct_url, kind)


def test_real_discovery_finds_both_colfax_pairs() -> None:
    pairs = {
        example: (baseline.name, [path.name for path in optimized])
        for baseline, optimized, example in discover_benchmarks(LAB_DIR, warn_missing=False)
        if example.startswith("flashattention4_")
    }

    assert pairs["flashattention4_decode"] == (
        "baseline_flashattention4_decode.py",
        ["optimized_flashattention4_decode.py"],
    )
    assert pairs["flashattention4_backward"] == (
        "baseline_flashattention4_backward.py",
        ["optimized_flashattention4_backward.py"],
    )


@pytest.mark.parametrize(
    ("module_name", "kind", "optimized"),
    [
        ("baseline_flashattention4_decode", "decode", False),
        ("optimized_flashattention4_decode", "decode", True),
        ("baseline_flashattention4_backward", "backward", False),
        ("optimized_flashattention4_backward", "backward", True),
    ],
)
def test_wrapper_factories_construct_without_resolving_a_device(
    module_name: str,
    kind: str,
    optimized: bool,
) -> None:
    module = importlib.import_module(f"labs.flashattention4.{module_name}")

    benchmark = module.get_benchmark()

    assert isinstance(benchmark, BaseBenchmark)
    assert isinstance(benchmark, ColfaxBenchmark)
    assert benchmark.kind == kind
    assert benchmark.optimized is optimized
    assert benchmark.spec == default_config(kind)
    assert benchmark._device is None


def test_opt_in_real_colfax_sm100_lifecycle_and_full_outputs() -> None:
    kind = os.environ.get("AISP_TEST_COLFAX_KIND")
    if kind is None:
        pytest.skip("set AISP_TEST_COLFAX_KIND=decode or backward for the real Colfax FA4 test")
    if kind not in EXPECTED_PINS:
        pytest.fail("AISP_TEST_COLFAX_KIND must be exactly decode or backward")
    if not torch.cuda.is_available():
        pytest.fail(f"AISP_TEST_COLFAX_KIND={kind} requested, but CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        pytest.fail(f"AISP_TEST_COLFAX_KIND={kind} requires exact SM100; detected {capability}")

    interface = load_upstream(kind)
    verified = verify_source_files(Path(interface.__file__).resolve().parent, kind)
    assert verified["commit"] == EXPECTED_PINS[kind]["commit"]
    assert interface._get_device_arch() == 100

    config = (
        ColfaxConfig("decode", 1, 2, 256, 16, 2, 64)
        if kind == "decode"
        else ColfaxConfig("backward", 1, 256, 256, 4, 4, 64)
    )
    arm_results = []
    for optimized in (False, True):
        benchmark = ColfaxBenchmark(kind, optimized=optimized, config=config)
        try:
            torch.manual_seed(20260907)
            decode_switch = (
                interface.utils._fa_disable_s_ping_pong_enabled if kind == "decode" else None
            )
            setup_valid, setup_error = check_setup_precomputation(
                lambda benchmark=benchmark: {"output": benchmark.output}, benchmark.setup
            )
            assert setup_valid, setup_error
            assert benchmark.output is None
            if kind == "decode":
                assert interface.utils._fa_disable_s_ping_pong_enabled is decode_switch
            with pytest.raises(RuntimeError, match=r"benchmark_fn\(\) must replay"):
                benchmark.capture_verification_payload()

            benchmark.benchmark_fn()
            captured_outputs = (
                (benchmark.output,)
                if isinstance(benchmark.output, torch.Tensor)
                else benchmark.output
            )
            assert captured_outputs is not None
            for _ in range(2):
                for tensor in captured_outputs:
                    tensor.fill_(float("nan"))
                benchmark.benchmark_fn()
                assert all(torch.isfinite(tensor).all() for tensor in captured_outputs)

            before_perturbation = tuple(tensor.detach().clone() for tensor in captured_outputs)
            input_name = "q" if kind == "decode" else "dout"
            assert benchmark.inputs is not None
            input_tensor = benchmark.inputs[input_name]
            original_input = input_tensor.detach().clone()
            with torch.no_grad():
                input_tensor.mul_(0.5).add_(0.25)
            benchmark.benchmark_fn()
            after_perturbation = tuple(tensor.detach().clone() for tensor in captured_outputs)
            assert any(
                not torch.equal(before, after)
                for before, after in zip(before_perturbation, after_perturbation, strict=True)
            )
            perturbed_inputs = {
                name: tensor.detach().clone() for name, tensor in benchmark.inputs.items()
            }
            reference_output, reference_gradients = _torch_attention_oracle(
                perturbed_inputs,
                causal=config.causal,
            )
            expected_perturbed = (
                (reference_output,) if kind == "decode" else reference_gradients
            )
            assert expected_perturbed is not None
            for actual, expected in zip(after_perturbation, expected_perturbed, strict=True):
                torch.testing.assert_close(
                    actual.detach().cpu().double(), expected, rtol=3e-2, atol=3e-2
                )

            with torch.no_grad():
                input_tensor.copy_(original_input)
            benchmark.benchmark_fn()
            restored_outputs = tuple(tensor.detach().clone() for tensor in captured_outputs)
            for before, restored in zip(before_perturbation, restored_outputs, strict=True):
                torch.testing.assert_close(before, restored, rtol=0, atol=0)

            benchmark.capture_verification_payload()
            assert benchmark.inputs is not None
            inputs = {name: tensor.detach().clone() for name, tensor in benchmark.inputs.items()}
            if kind == "decode":
                assert isinstance(benchmark.output, torch.Tensor)
                outputs = (benchmark.output.detach().clone(),)
            else:
                assert isinstance(benchmark.output, tuple) and len(benchmark.output) == 3
                outputs = tuple(tensor.detach().clone() for tensor in benchmark.output)
            payload = benchmark.get_verify_output()
            expected_payload = (
                outputs[0]
                if kind == "decode"
                else torch.cat([tensor.reshape(-1) for tensor in outputs])
            )
            torch.testing.assert_close(payload, expected_payload, rtol=0, atol=0)
            assert benchmark.get_output_tolerance() == (
                (2e-2, 2e-2) if kind == "decode" else (0.0, 0.0)
            )
            assert benchmark.validate_result() is None
            arm_results.append((inputs, outputs))
        finally:
            benchmark.teardown()

    baseline_inputs, baseline_outputs = arm_results[0]
    candidate_inputs, candidate_outputs = arm_results[1]
    assert baseline_inputs.keys() == candidate_inputs.keys()
    for name in baseline_inputs:
        torch.testing.assert_close(baseline_inputs[name], candidate_inputs[name], rtol=0, atol=0)

    reference_output, reference_gradients = _torch_attention_oracle(
        baseline_inputs,
        causal=config.causal,
    )
    if kind == "decode":
        for outputs in (baseline_outputs, candidate_outputs):
            torch.testing.assert_close(
                outputs[0].detach().cpu().double(), reference_output, rtol=3e-2, atol=3e-2
            )
    else:
        assert reference_gradients is not None
        for outputs in (baseline_outputs, candidate_outputs):
            assert len(outputs) == len(reference_gradients) == 3
            for actual, expected in zip(outputs, reference_gradients, strict=True):
                torch.testing.assert_close(
                    actual.detach().cpu().double(), expected, rtol=3e-2, atol=3e-2
                )

    pair_tolerance = (2e-2, 2e-2) if kind == "decode" else (0.0, 0.0)
    for baseline, candidate in zip(baseline_outputs, candidate_outputs, strict=True):
        torch.testing.assert_close(
            baseline,
            candidate,
            rtol=pair_tolerance[0],
            atol=pair_tolerance[1],
        )
