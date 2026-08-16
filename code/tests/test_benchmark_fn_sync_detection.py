from __future__ import annotations

import ast
import importlib.util
import textwrap
from pathlib import Path

from core.benchmark.contract import BenchmarkContract
from core.harness.validity_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)
from core.hot_path_checks import benchmark_fn_antipattern_warnings_for_class


def _parse_class(source: str) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(source))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _load_fixture_module(relative_path: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_warns_on_sync_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                self._synchronize()
                torch.cuda.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("_synchronize()" in warning for warning in warnings)
    assert any("torch.cuda.synchronize()" in warning for warning in warnings)


def test_contract_warns_on_stream_and_event_synchronize_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                stream.synchronize()
                self._poll_event.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("stream/event synchronize()" in warning for warning in warnings)


def test_contract_warns_on_local_event_variable_synchronize_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                end_event = torch.cuda.Event(enable_timing=True)
                alias = end_event
                alias.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("stream/event synchronize()" in warning for warning in warnings)


def test_runtime_sync_check_detects_hot_path_synchronization() -> None:
    class SyncBench:
        def _synchronize(self) -> None:
            pass

        def benchmark_fn(self) -> None:
            self._synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("_synchronize()" in finding for finding in findings)


def test_runtime_sync_check_detects_stream_or_event_synchronization() -> None:
    class SyncBench:
        def benchmark_fn(self) -> None:
            stream.synchronize()  # noqa: F821 - undefined receiver is the inspected fixture
            self._poll_event.synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("stream/event synchronize()" in finding for finding in findings)


def test_runtime_sync_check_detects_local_event_variable_synchronization() -> None:
    class SyncBench:
        def benchmark_fn(self) -> None:
            end_event = torch.cuda.Event(enable_timing=True)  # noqa: F821 - inspected fixture
            alias = end_event
            alias.synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("stream/event synchronize()" in finding for finding in findings)


def test_contract_warns_on_sync_inside_same_class_helper_called_by_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def _helper(self):
                torch.cuda.synchronize()

            def benchmark_fn(self):
                self._helper()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("torch.cuda.synchronize()" in warning for warning in warnings)


def test_runtime_sync_check_detects_same_class_helper_synchronization() -> None:
    class SyncBench:
        def _helper(self) -> None:
            torch.cuda.synchronize()  # noqa: F821 - undefined module is the inspected fixture

        def benchmark_fn(self) -> None:
            self._helper()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("torch.cuda.synchronize()" in finding for finding in findings)


def test_runtime_sync_check_ignores_clean_benchmark_fn() -> None:
    class CleanBench:
        def benchmark_fn(self) -> None:
            x = 1 + 1
            assert x == 2

    ok, findings = check_benchmark_fn_sync_calls(CleanBench().benchmark_fn)

    assert ok
    assert findings == []


def test_runtime_sync_check_respects_allowlist() -> None:
    class AllowedBench:
        def benchmark_fn(self) -> None:
            torch.cuda.synchronize()  # noqa: F821 - undefined module is the inspected fixture

    ok, findings = check_benchmark_fn_sync_calls(
        AllowedBench().benchmark_fn,
        allowed_codes=("sync",),
    )

    assert ok
    assert findings == []


def test_contract_warns_on_random_input_regeneration_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                torch.randn(8, 8, device=self.device)

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_contract_warns_on_host_transfer_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                x = y.cpu()
                z = y.to("cpu")
                return x, z

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any(".cpu()" in warning for warning in warnings)
    assert any(".to('cpu')" in warning for warning in warnings)


def test_runtime_antipattern_check_detects_hot_path_allocations() -> None:
    class AntiPatternBench:
        def benchmark_fn(self) -> None:
            torch.randn(8, 8)  # noqa: F821 - undefined module is the inspected fixture

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any("regenerates random inputs" in finding for finding in findings)


def test_contract_warns_on_hot_path_logging() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def benchmark_fn(self):
                print("timed output")
                self.logger.info("timed log")
        """
    )

    warnings = benchmark_fn_antipattern_warnings_for_class(class_node)

    assert any("print()" in warning for warning in warnings)
    assert any("self.logger.info()" in warning for warning in warnings)


def test_contract_warns_on_hot_path_tensor_construction_and_numpy_io() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def benchmark_fn(self):
                values = np.load(self.path)
                return torch.from_numpy(values)
        """
    )

    warnings = benchmark_fn_antipattern_warnings_for_class(class_node)

    assert any("NumPy file I/O" in warning for warning in warnings)
    assert any("constructs a tensor view or copy" in warning for warning in warnings)


def test_contract_allows_declared_hot_path_tensor_construction_and_numpy_io() -> None:
    class_node = _parse_class(
        """
        class StoragePathBench:
            def benchmark_fn(self):
                values = np.load(self.path)
                return torch.from_numpy(values)
        """
    )

    warnings = benchmark_fn_antipattern_warnings_for_class(
        class_node,
        allowed_codes=("io", "tensor_construction"),
    )

    assert not warnings


def test_contract_covers_supported_tensor_numpy_and_logging_call_forms() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def benchmark_fn(self):
                torch.tensor([1])
                torch.as_tensor([1])
                torch.from_numpy(self.values)
                np.load(self.path)
                np.save(self.path, self.values)
                numpy.load(self.path)
                numpy.save(self.path, self.values)
                logging.info("module logger")
                logger.warning("named logger")
                self.log.error("instance log")
                helper.info("ordinary API")
        """
    )

    warnings = benchmark_fn_antipattern_warnings_for_class(class_node)

    for target in (
        "torch.tensor()",
        "torch.as_tensor()",
        "torch.from_numpy()",
        "np.load()",
        "np.save()",
        "numpy.load()",
        "numpy.save()",
        "logging.info()",
        "logger.warning()",
        "self.log.error()",
    ):
        assert any(target in warning for warning in warnings)
    assert not any("helper.info()" in warning for warning in warnings)


def test_contract_allows_declared_hot_path_logging() -> None:
    class_node = _parse_class(
        """
        class LoggingBench:
            def benchmark_fn(self):
                print("timed output")
                self.logger.info("timed log")
        """
    )

    warnings = benchmark_fn_antipattern_warnings_for_class(
        class_node,
        allowed_codes=("logging",),
    )

    assert not warnings


def test_contract_warns_on_antipattern_inside_same_class_helper_called_by_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def _helper(self):
                torch.randn(8, 8, device=self.device)

            def benchmark_fn(self):
                self._helper()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_runtime_antipattern_check_detects_same_class_helper_antipattern() -> None:
    class AntiPatternBench:
        def _helper(self) -> None:
            torch.randn(8, 8)  # noqa: F821 - undefined module is the inspected fixture

        def benchmark_fn(self) -> None:
            self._helper()

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any("regenerates random inputs" in finding for finding in findings)


def test_runtime_antipattern_check_detects_imported_helper_function_antipattern() -> None:
    module = _load_fixture_module("tests/fixtures_contract/imported_helper_function_entry.py")
    bench = module.get_benchmark()

    ok, findings = check_benchmark_fn_antipatterns(bench.benchmark_fn)

    assert not ok
    assert any("regenerates random inputs" in finding for finding in findings)


def test_runtime_antipattern_check_detects_imported_helper_object_antipattern() -> None:
    module = _load_fixture_module("tests/fixtures_contract/imported_helper_object_entry.py")
    bench = module.get_benchmark()

    ok, findings = check_benchmark_fn_antipatterns(bench.benchmark_fn)

    assert not ok
    assert any(".cpu()" in finding for finding in findings)


def test_runtime_antipattern_check_detects_host_transfer() -> None:
    class AntiPatternBench:
        def benchmark_fn(self) -> None:
            value.cpu()  # noqa: F821 - undefined receiver is the inspected fixture
            value.item()  # noqa: F821 - undefined receiver is the inspected fixture
            value.to("cpu")  # noqa: F821 - undefined receiver is the inspected fixture

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any(".cpu()" in finding for finding in findings)
    assert any(".item()" in finding for finding in findings)
    assert any(".to('cpu')" in finding for finding in findings)


def test_runtime_antipattern_check_respects_allowlist() -> None:
    class AllowedBench:
        def benchmark_fn(self) -> None:
            value.cpu()  # noqa: F821 - undefined receiver is the inspected fixture

    ok, findings = check_benchmark_fn_antipatterns(
        AllowedBench().benchmark_fn,
        allowed_codes=("host_transfer",),
    )

    assert ok
    assert findings == []
