from __future__ import annotations

import inspect

import torch

from labs.block_scaling.block_scaling_common import (
    BlockScalingConfig,
    BlockScalingProblem,
    DEFAULT_CLUSTER_SHAPE_MN,
    DEFAULT_MMA_TILER_MN,
    DEFAULT_MNKL,
    load_lab_config_from_env,
    override_config,
    parse_int_tuple,
    parse_software_dtype,
    verification_inputs,
    verification_output_slice,
)


def test_load_block_scaling_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("AISP_BLOCK_SCALING_MNKL", raising=False)
    monkeypatch.delenv("AISP_BLOCK_SCALING_MMA_TILER_MN", raising=False)
    monkeypatch.delenv("AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN", raising=False)
    monkeypatch.delenv("AISP_BLOCK_SCALING_SF_VEC_SIZE", raising=False)
    monkeypatch.delenv("AISP_BLOCK_SCALING_TOLERANCE", raising=False)
    monkeypatch.delenv("AISP_BLOCK_SCALING_SOFTWARE_DTYPE", raising=False)

    config = load_lab_config_from_env()

    assert config.mnkl == DEFAULT_MNKL
    assert config.mma_tiler_mn == DEFAULT_MMA_TILER_MN
    assert config.cluster_shape_mn == DEFAULT_CLUSTER_SHAPE_MN
    assert config.sf_vec_size == 16
    assert config.tolerance == 0.1
    assert config.software_dtype == torch.bfloat16


def test_load_block_scaling_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AISP_BLOCK_SCALING_MNKL", "4096,2048,1024,2")
    monkeypatch.setenv("AISP_BLOCK_SCALING_MMA_TILER_MN", "128,256")
    monkeypatch.setenv("AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN", "1,2")
    monkeypatch.setenv("AISP_BLOCK_SCALING_SF_VEC_SIZE", "32")
    monkeypatch.setenv("AISP_BLOCK_SCALING_TOLERANCE", "0.25")
    monkeypatch.setenv("AISP_BLOCK_SCALING_SOFTWARE_DTYPE", "fp16")

    config = load_lab_config_from_env()

    assert config.mnkl == (4096, 2048, 1024, 2)
    assert config.mma_tiler_mn == (128, 256)
    assert config.cluster_shape_mn == (1, 2)
    assert config.sf_vec_size == 32
    assert config.tolerance == 0.25
    assert config.software_dtype == torch.float16

    inputs = verification_inputs(config)
    assert tuple(inputs["mnkl"].tolist()) == config.mnkl
    assert tuple(inputs["mma_tiler_mn"].tolist()) == config.mma_tiler_mn
    assert tuple(inputs["cluster_shape_mn"].tolist()) == config.cluster_shape_mn
    assert tuple(inputs["sf_meta"].tolist()) == (32, 1)


def test_override_config_and_parse_helpers() -> None:
    config = BlockScalingConfig()
    updated = override_config(
        config,
        mnkl=parse_int_tuple("4096,8192,1024,1", expected_len=4, name="mnkl"),
        mma_tiler_mn=parse_int_tuple("128,128", expected_len=2, name="mma"),
        cluster_shape_mn=parse_int_tuple("1,2", expected_len=2, name="cluster"),
        sf_vec_size=32,
        tolerance=0.25,
        software_dtype=parse_software_dtype("fp16"),
    )

    assert updated.mnkl == (4096, 8192, 1024, 1)
    assert updated.mma_tiler_mn == (128, 128)
    assert updated.cluster_shape_mn == (1, 2)
    assert updated.sf_vec_size == 32
    assert updated.tolerance == 0.25
    assert updated.software_dtype == torch.float16
    assert config.mnkl == DEFAULT_MNKL


def test_verification_output_slice_caps_to_small_tile() -> None:
    output = torch.randn(256, 192, 4)
    sliced = verification_output_slice(output)

    assert sliced.shape == (128, 128, 1)
    torch.testing.assert_close(sliced, output[:128, :128, :1].float())


def test_block_scaling_verify_close_batches_error_materialization() -> None:
    source = inspect.getsource(BlockScalingProblem.verify_close)

    assert "max_abs_error, mean_abs_error = torch.stack((diff.max(), diff.mean())).tolist()" in source
    assert "diff.max().item()" not in source
    assert "diff.mean().item()" not in source
