from __future__ import annotations

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
        preload_serving_stack_shared_libs=forbidden_runtime_call(
            "preload_serving_stack_shared_libs"
        ),
    )
    install_module("core.utils.logger", get_logger=lambda name: Logger())
    install_module(
        "core.utils.python_entrypoints",
        build_repo_python_env=forbidden_runtime_call("build_repo_python_env"),
        install_local_module_override=forbidden_runtime_call(
            "install_local_module_override"
        ),
    )

    numba_sentinel = install_module("numba")
    vllm_sentinel = install_package("vllm")

    def record_vllm_attribute(name):
        vllm_attribute_reads.append(name)
        raise AttributeError(name)

    vllm_sentinel.__getattr__ = record_vllm_attribute
    install_package("vllm.inputs")
    install_module("vllm.inputs.data")

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
                "numba_unchanged": sys.modules["numba"] is numba_sentinel,
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
        "numba_unchanged": True,
        "path_unchanged": True,
        "runtime_calls": [],
        "vllm_attribute_reads": [],
        "vllm_unchanged": True,
    }
