from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CODE_ROOT / "ch10" / "cufile_gds_example.py"
FILE_SIZE = 4096


def _load_cufile_example(
    monkeypatch: pytest.MonkeyPatch,
    *,
    read_result: object = FILE_SIZE,
    write_result: object = FILE_SIZE,
    driver_open_error: str | None = None,
    driver_close_error: str | None = None,
    handle_register_error: str | None = None,
    handle_deregister_error: str | None = None,
    cuda_available: bool = True,
) -> SimpleNamespace:
    events: list[tuple[object, ...]] = []
    descriptors = []
    allocated_tensors = []
    contiguous_copies = []
    opened_files: list[tuple[object, ...]] = []
    next_pointer = iter(range(0xB200, 0xB300))

    class FakeCuFileError(Exception):
        pass

    class FakeDescr:
        def __init__(self) -> None:
            self.type = None
            self.handle = SimpleNamespace(fd=None)
            descriptors.append(self)

        def __int__(self) -> int:
            return 0xD35C

    class FakeTensor:
        is_cuda = True

        def __init__(
            self,
            size: int,
            device: object = "cuda:0",
            *,
            is_contiguous: bool = True,
        ) -> None:
            self.size = size
            self.device = device
            self.is_contiguous = is_contiguous
            self.pointer = next(next_pointer)

        def data_ptr(self) -> int:
            return self.pointer

        def numel(self) -> int:
            return self.size

        def element_size(self) -> int:
            return 1

        def contiguous(self) -> FakeTensor:
            if self.is_contiguous:
                return self
            copy = FakeTensor(self.size, self.device)
            contiguous_copies.append((self, copy))
            return copy

    fake_cufile = ModuleType("cuda.bindings.cufile")
    fake_cufile.cuFileError = FakeCuFileError
    fake_cufile.Descr = FakeDescr
    fake_cufile.FileHandleType = SimpleNamespace(OPAQUE_FD=17)

    def driver_open() -> None:
        events.append(("driver_open",))
        if driver_open_error is not None:
            raise FakeCuFileError(driver_open_error)

    fake_cufile.driver_open = driver_open

    def handle_register(descr_ptr: int) -> int:
        events.append(("handle_register", descr_ptr))
        if handle_register_error is not None:
            raise FakeCuFileError(handle_register_error)
        return 0xFACE

    def read(*args: object) -> object:
        events.append(("read", *args))
        return read_result

    def write(*args: object) -> object:
        events.append(("write", *args))
        return write_result

    fake_cufile.handle_register = handle_register
    fake_cufile.read = read
    fake_cufile.write = write

    def handle_deregister(handle: int) -> None:
        events.append(("handle_deregister", handle))
        if handle_deregister_error is not None:
            raise FakeCuFileError(handle_deregister_error)

    def driver_close() -> None:
        events.append(("driver_close",))
        if driver_close_error is not None:
            raise FakeCuFileError(driver_close_error)

    fake_cufile.handle_deregister = handle_deregister
    fake_cufile.driver_close = driver_close

    cuda_package = ModuleType("cuda")
    cuda_package.__path__ = []
    bindings_package = ModuleType("cuda.bindings")
    bindings_package.__path__ = []
    bindings_package.cufile = fake_cufile
    cuda_package.bindings = bindings_package

    class FakeCudaDeviceContext:
        def __init__(self, device: object) -> None:
            self.device = device

        def __enter__(self) -> None:
            events.append(("device_enter", self.device))

        def __exit__(self, *_exc_info: object) -> None:
            events.append(("device_exit", self.device))

    fake_torch = ModuleType("torch")
    fake_torch.uint8 = object()
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda _device: "Fake B200",
        get_device_properties=lambda _device: SimpleNamespace(major=10, minor=0),
        device=FakeCudaDeviceContext,
        synchronize=lambda device: events.append(("synchronize", device)),
    )

    def empty(size: int, *, dtype: object, device: object) -> FakeTensor:
        tensor = FakeTensor(size, device)
        allocated_tensors.append((size, dtype, device, tensor))
        return tensor

    fake_torch.empty = empty

    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings.cufile", fake_cufile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_name = "_test_cufile_gds_example_cuda131"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    closed_fds: list[int] = []
    monkeypatch.setattr(module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(module.os.path, "getsize", lambda _path: FILE_SIZE)
    monkeypatch.setattr(module.os, "O_DIRECT", 0, raising=False)

    def open_file(*args: object) -> int:
        opened_files.append(args)
        return 73

    monkeypatch.setattr(module.os, "open", open_file)
    monkeypatch.setattr(module.os, "close", closed_fds.append)
    timestamps = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(timestamps))

    return SimpleNamespace(
        module=module,
        fake_cufile=fake_cufile,
        fake_torch=fake_torch,
        FakeTensor=FakeTensor,
        events=events,
        descriptors=descriptors,
        allocated_tensors=allocated_tensors,
        contiguous_copies=contiguous_copies,
        opened_files=opened_files,
        closed_fds=closed_fds,
    )


def test_cuda_bindings_131_validates_counts_and_uses_exact_device_buffers(monkeypatch) -> None:
    env = _load_cufile_example(monkeypatch)
    demo = env.module.GPUDirectStorageDemo()

    throughput_gbps, latency_ms = demo.read_gds("/tmp/fake-gds-input.bin", device=3)
    input_tensor = env.FakeTensor(FILE_SIZE, device="cuda:7", is_contiguous=False)
    write_throughput_gbps, write_latency_ms = demo.write_gds(
        "/tmp/fake-gds-output.bin",
        input_tensor,
    )
    demo.cleanup()

    assert env.module.CUFILE_AVAILABLE is True
    assert env.allocated_tensors[0][:3] == (FILE_SIZE, env.fake_torch.uint8, "cuda:3")
    assert len(env.contiguous_copies) == 1
    contiguous_tensor = env.contiguous_copies[0][1]
    write_event = next(event for event in env.events if event[0] == "write")
    assert write_event == ("write", 0xFACE, contiguous_tensor.data_ptr(), FILE_SIZE, 0, 0)
    assert [event for event in env.events if event[0] == "synchronize"] == [
        ("synchronize", 3),
        ("synchronize", 3),
        ("synchronize", "cuda:7"),
        ("synchronize", "cuda:7"),
    ]
    assert [event for event in env.events if event[0].startswith("device_")] == [
        ("device_enter", 3),
        ("device_exit", 3),
        ("device_enter", "cuda:7"),
        ("device_exit", "cuda:7"),
    ]
    assert env.opened_files[1][1] & env.module.os.O_TRUNC
    assert env.closed_fds == [73, 73]
    assert throughput_gbps == pytest.approx(FILE_SIZE / 0.25 / 1e9)
    assert latency_ms == pytest.approx(250.0)
    assert write_throughput_gbps == pytest.approx(FILE_SIZE / 0.5 / 1e9)
    assert write_latency_ms == pytest.approx(500.0)
    assert demo.driver_initialized is False


@pytest.mark.parametrize(
    ("operation", "result", "expected_error"),
    [
        ("read", FILE_SIZE - 1, "cuFile read transferred 4095 of 4096 requested bytes"),
        ("read", FILE_SIZE + 1, "cuFile read transferred 4097 of 4096 requested bytes"),
        ("read", -5, "cuFile read returned a negative byte count -5. Expected 4096"),
        ("read", None, "cuFile read returned a non-integer byte count None. Expected 4096"),
        ("write", FILE_SIZE - 1, "cuFile write transferred 4095 of 4096 requested bytes"),
        ("write", FILE_SIZE + 1, "cuFile write transferred 4097 of 4096 requested bytes"),
        ("write", -5, "cuFile write returned a negative byte count -5. Expected 4096"),
        ("write", None, "cuFile write returned a non-integer byte count None. Expected 4096"),
    ],
)
def test_cufile_rejects_invalid_transfer_counts_and_cleans_up(
    monkeypatch, operation: str, result: object, expected_error: str
) -> None:
    result_overrides = {f"{operation}_result": result}
    env = _load_cufile_example(monkeypatch, **result_overrides)
    demo = env.module.GPUDirectStorageDemo()

    with pytest.raises(env.module.CuFileError) as exc_info:
        if operation == "read":
            demo.read_gds("/tmp/fake-gds-input.bin", device=4)
        else:
            data = env.FakeTensor(FILE_SIZE, device="cuda:6", is_contiguous=False)
            demo.write_gds("/tmp/fake-gds-output.bin", data)

    demo.cleanup()

    assert str(exc_info.value) == expected_error
    assert env.closed_fds == [73]
    assert ("handle_deregister", 0xFACE) in env.events
    assert [event for event in env.events if event[0] == "synchronize"] == [
        ("synchronize", 4 if operation == "read" else "cuda:6")
    ]
    assert env.events[-1] == ("driver_close",)


def test_transfer_error_is_not_masked_by_handle_cleanup_failure(monkeypatch) -> None:
    env = _load_cufile_example(
        monkeypatch,
        read_result=FILE_SIZE - 1,
        handle_deregister_error="deregister failed",
    )
    demo = env.module.GPUDirectStorageDemo()

    with pytest.raises(env.module.CuFileError) as exc_info:
        demo.read_gds("/tmp/fake-gds-input.bin", device=4)

    assert str(exc_info.value) == "cuFile read transferred 4095 of 4096 requested bytes"
    assert ("handle_deregister", 0xFACE) in env.events
    assert env.closed_fds == [73]


def test_source_contains_no_simulated_or_fixed_fallback_metrics() -> None:
    source = SOURCE.read_text()

    assert "simulate_gds_performance" not in source
    assert "Running in simulation mode" not in source
    assert "traditional_throughput = 8.5" not in source
    assert "gds_throughput = 10.8" not in source
    assert "Typically 20-30%" not in source


def test_main_fails_before_driver_or_metrics_when_gds_config_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _load_cufile_example(monkeypatch)
    monkeypatch.setattr(env.module.os.path, "isfile", lambda _path: False)

    assert env.module.main() == 3

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert "SKIPPED: GDS is not configured because /etc/cufile.json is missing" in captured.err
    assert "Throughput:" not in combined_output
    assert "Latency:" not in combined_output
    assert "simulation" not in combined_output.lower()
    assert env.events == []


def test_main_fails_before_driver_or_metrics_when_cuda_gpu_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _load_cufile_example(monkeypatch, cuda_available=False)

    assert env.module.main() == 3

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert "SKIPPED: PyTorch cannot see a CUDA GPU" in captured.err
    assert "Throughput:" not in combined_output
    assert "Latency:" not in combined_output
    assert env.events == []


def test_main_fails_before_metrics_when_cufile_driver_cannot_initialize(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _load_cufile_example(monkeypatch, driver_open_error="driver rejected host")

    assert env.module.main() == 3

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert "SKIPPED: the cuFile driver could not initialize" in captured.err
    assert "driver rejected host" in captured.err
    assert "Throughput:" not in combined_output
    assert "Latency:" not in combined_output
    assert env.events == [("driver_open",)]


def test_storage_registration_failure_precedes_comparison_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _load_cufile_example(monkeypatch, handle_register_error="unsupported filesystem")
    demo = env.module.GPUDirectStorageDemo()
    monkeypatch.setattr(
        demo,
        "create_test_file",
        lambda _size_mb: "/tmp/unsupported-gds-input.bin",
    )

    with pytest.raises(env.module.CuFileUnavailableError) as exc_info:
        demo.benchmark_comparison(size_mb=1)

    demo.cleanup()
    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert str(exc_info.value) == (
        "SKIPPED: cuFile could not register the input path. Use a GDS-supported local or "
        "NVMe filesystem. Path: /tmp/unsupported-gds-input.bin. Error: unsupported filesystem"
    )
    assert "Throughput:" not in combined_output
    assert "Latency:" not in combined_output
    assert not any(event[0] == "read" for event in env.events)
    assert env.closed_fds == [73]


def test_main_preserves_classified_failure_when_driver_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _load_cufile_example(monkeypatch)

    class FailingDemo:
        device_description = "Fake B200 (SM 10.0)"

        def benchmark_comparison(self, *, size_mb: int) -> dict:
            assert size_mb == 256
            raise env.module.CuFileUnavailableError("SKIPPED: unsupported GDS storage")

        def print_results(self, _results: dict) -> None:
            raise AssertionError("results must not be printed")

        def cleanup(self) -> None:
            raise env.module.CuFileError("Failed to close cuFile driver: close failed")

    monkeypatch.setattr(env.module, "GPUDirectStorageDemo", FailingDemo)

    assert env.module.main() == 3

    captured = capsys.readouterr()
    assert "SKIPPED: unsupported GDS storage" in captured.err
    assert "GDS cleanup failed: Failed to close cuFile driver: close failed" in captured.err
