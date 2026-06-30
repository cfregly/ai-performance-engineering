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
    measure_cuda_callable,
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

    buffer = torch.empty(128, 128, 1, dtype=torch.float32)
    buffered = verification_output_slice(output, buffer)
    assert buffered.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(buffered, output[:128, :128, :1].float())


def test_block_scaling_verify_close_batches_error_materialization() -> None:
    source = inspect.getsource(BlockScalingProblem.verify_close)

    assert "error_stats = torch.empty(2, device=diff.device, dtype=diff.dtype)" in source
    assert "error_stats[0].copy_(diff.max())" in source
    assert "error_stats[1].copy_(diff.mean())" in source
    assert "error_stats_host = error_stats.detach().cpu()" in source
    assert "max_abs_error = float(error_stats_host[0])" in source
    assert "mean_abs_error = float(error_stats_host[1])" in source
    assert "error_stats.detach().cpu().tolist()" not in source
    assert "torch.stack((diff.max(), diff.mean())).tolist()" not in source
    assert "diff.max().item()" not in source
    assert "diff.mean().item()" not in source


def test_block_scaling_extract_hardware_output_reuses_device_buffer() -> None:
    source = inspect.getsource(BlockScalingProblem.extract_hardware_output)

    assert "self.c_ref_device = self.c_ref.cuda()" in source
    assert "return self.c_ref_device" in source
    assert ".float().clone()" not in source


def test_block_scaling_cuda_timing_records_on_current_stream() -> None:
    source = inspect.getsource(measure_cuda_callable)

    assert source.count("torch.cuda.Event(enable_timing=True)") == 2
    assert source.count("current_stream = torch.cuda.current_stream()") == 1
    assert "start.record(current_stream)" in source
    assert "end.record(current_stream)" in source
    assert "start.record()" not in source
    assert "end.record()" not in source
