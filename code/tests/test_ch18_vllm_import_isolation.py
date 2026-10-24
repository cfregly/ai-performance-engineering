from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "ch18.baseline_vllm_v1_integration",
    "ch18.optimized_vllm_v1_integration",
)


_IMPORT_PROBE = textwrap.dedent(
    r"""
    import __future__
    import argparse
    import collections.abc
    import ctypes
    import gc
    import importlib
    import importlib.metadata
    import importlib.util
    import json
    import os
    from pathlib import Path
    import random
    import sys
    import time
    import types
    import typing

    source_path = Path(sys.argv[1])
    module_name = sys.argv[2]
    runtime_calls = []
    vllm_attribute_reads = []
    dlopen_calls = []

    def install_package(name, path=None):
        module = types.ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        sys.modules[name] = module
        return module

    def install_module(name, **attributes):
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    def forbidden_runtime_call(name):
        def fail(*args, **kwargs):
            runtime_calls.append(name)
            raise AssertionError(f"{name} ran during module import")

        return fail

    class BaseBenchmark:
        pass

    class BenchmarkConfig:
        pass

    class WorkloadMetadata:
        pass

    class PrecisionFlags:
        pass

    class VerificationPayloadMixin:
        pass

    class ServingStackPins:
        pass

    class Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    install_package("ch18", source_path.parent)
    install_package("core")
    install_package("core.benchmark")
    install_package("core.harness")
    install_package("core.utils")
    install_module("torch")
    install_module(
        "ch18.vllm_process_cleanup",
        shutdown_vllm_runtime=lambda *args, **kwargs: None,
    )
    install_module(
        "core.benchmark.verification",
        PrecisionFlags=PrecisionFlags,
        simple_signature=lambda *args, **kwargs: None,
    )
    install_module(
        "core.benchmark.verification_mixin",
        VerificationPayloadMixin=VerificationPayloadMixin,
    )
    install_module(
        "core.harness.benchmark_harness",
        BaseBenchmark=BaseBenchmark,
        BenchmarkConfig=BenchmarkConfig,
        WorkloadMetadata=WorkloadMetadata,
    )
    install_module(
        "core.harness.serving_stack",
        ServingStackPins=ServingStackPins,
        configure_serving_stack_cache_env=forbidden_runtime_call(
            "configure_serving_stack_cache_env"
        ),
        configure_serving_stack_runtime_env=forbidden_runtime_call(
            "configure_serving_stack_runtime_env"
        ),
        get_serving_stack_pins=forbidden_runtime_call("get_serving_stack_pins"),
        is_vllm_native_extension_abi_error=lambda exc: False,
        load_vllm_native_extensions=forbidden_runtime_call(
            "load_vllm_native_extensions"
        ),
        preload_serving_stack_shared_libs=forbidden_runtime_call(
            "preload_serving_stack_shared_libs"
        ),
    )
    install_module("core.utils.logger", get_logger=lambda name: Logger())
    vllm_sentinel = install_package("vllm")

    def record_vllm_attribute(name):
        vllm_attribute_reads.append(name)
        raise AttributeError(name)

    vllm_sentinel.__getattr__ = record_vllm_attribute
    install_package("vllm.inputs")

    def record_dlopen(*args, **kwargs):
        dlopen_calls.append(str(args[0]) if args else "")
        raise AssertionError("ctypes.CDLL ran during module import")

    ctypes.CDLL = record_dlopen

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {source_path}")
    subject = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = subject

    before_env = dict(os.environ)
    before_path = list(sys.path)
    before_modules = {name: id(module) for name, module in sys.modules.items()}
    spec.loader.exec_module(subject)
    after_modules = {name: id(module) for name, module in sys.modules.items()}

    print(
        json.dumps(
            {
                "env_unchanged": before_env == dict(os.environ),
                "path_unchanged": before_path == sys.path,
                "modules_unchanged": before_modules == after_modules,
                "numba_unloaded": "numba" not in sys.modules,
                "vllm_unchanged": sys.modules["vllm"] is vllm_sentinel,
                "runtime_calls": runtime_calls,
                "vllm_attribute_reads": vllm_attribute_reads,
                "dlopen_calls": dlopen_calls,
            },
            sort_keys=True,
        )
    )
    """
)


@pytest.mark.parametrize("module_name", MODULES)
def test_vllm_benchmark_import_is_runtime_side_effect_free(module_name: str) -> None:
    source_path = REPO_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _IMPORT_PROBE, str(source_path), module_name],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "dlopen_calls": [],
        "env_unchanged": True,
        "modules_unchanged": True,
        "numba_unloaded": True,
        "path_unchanged": True,
        "runtime_calls": [],
        "vllm_attribute_reads": [],
        "vllm_unchanged": True,
    }


@pytest.mark.parametrize("module_name", MODULES)
def test_vllm_benchmark_runtime_probe_uses_current_native_and_prompt_apis(
    module_name: str,
) -> None:
    source_path = REPO_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    source = source_path.read_text(encoding="utf-8")

    assert "load_vllm_native_extensions()" in source
    assert "vllm_module.TokensPrompt" in source
    assert "vllm.inputs.data" not in source
    assert 'importlib.import_module("vllm._C")' not in source
    assert "install_local_module_override" not in source
    assert "build_repo_python_env" not in source
    assert "hack/numba" not in source


def test_ch18_vllm_benchmarks_use_the_installed_numba_package() -> None:
    numba = importlib.import_module("numba")
    module_path = Path(numba.__file__).resolve()

    assert module_path.is_file()
    assert REPO_ROOT / "hack" / "numba" not in module_path.parents
    assert callable(numba.njit)


@pytest.mark.parametrize("module_name", MODULES)
def test_vllm_benchmark_runtime_readiness_calls_shared_native_loader(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    calls: list[str] = []
    monkeypatch.setattr(module, "_assert_serving_stack_versions", lambda _pins: None)
    monkeypatch.setattr(
        module,
        "load_vllm_native_extensions",
        lambda: calls.append("loaded"),
    )

    module._assert_vllm_runtime_ready(None, object())

    assert calls == ["loaded"]


@pytest.mark.parametrize("module_name", MODULES)
def test_vllm_benchmark_accepts_offline_model_and_context_overrides(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    benchmark = module.get_benchmark()

    benchmark.apply_target_overrides(
        [
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--max-model-len",
            "2048",
        ]
    )

    assert benchmark.runner.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert benchmark.runner.max_model_len == 2048
    assert benchmark.runner.max_tokens == 128
    assert benchmark.runner.batch_size == 8
    workload = benchmark.get_workload_metadata()
    assert workload is not None
    assert workload.requests_per_iteration == 8.0
    assert workload.tokens_per_iteration == 1024.0
    signature = benchmark.get_input_signature()
    assert signature["shapes"]["workload"] == [128]

    captured: dict[str, object] = {}

    def fake_llm(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    benchmark.runner._llm_cls = fake_llm
    benchmark.runner._new_llm()
    assert captured["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert captured["max_model_len"] == 2048


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (["--model", ""], "--model must be a non-empty"),
        (["--max-model-len", "128"], "must be greater than max_tokens"),
        (["--max-model-len", "not-an-int"], "require --model NAME"),
        (["--batch-size", "1"], "unsupported vLLM target override"),
    ),
)
def test_vllm_benchmark_rejects_unsupported_or_invalid_target_overrides(
    module_name: str,
    argv: list[str],
    message: str,
) -> None:
    module = importlib.import_module(module_name)
    benchmark = module.get_benchmark()

    with pytest.raises(ValueError, match=message):
        benchmark.apply_target_overrides(argv)


def test_decode_kernel_uses_current_vllm_unified_attention_api() -> None:
    source = (REPO_ROOT / "ch18" / "decode_kernels.py").read_text(encoding="utf-8")

    assert "vllm.v1.attention.ops.triton_unified_attention" in source
    assert "self._unified_attention(" in source
    assert "cu_seqlens_q=self.query_start_loc[: batch + 1]" in source
    assert "paged_attention_v1" not in source
    assert "vllm.inputs.data" not in source
