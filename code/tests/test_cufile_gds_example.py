from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CODE_ROOT / "ch10" / "cufile_gds_example.py"


def test_cuda_bindings_13_cufile_read_and_write_use_pythonic_api(monkeypatch) -> None:
    events: list[tuple[object, ...]] = []
    descriptors = []

    class FakeCuFileError(Exception):
        pass

    class FakeDescr:
        def __init__(self) -> None:
            self.type = None
            self.handle = SimpleNamespace(fd=None)
            descriptors.append(self)

        def __int__(self) -> int:
            return 0xD35C

    fake_cufile = ModuleType("cuda.bindings.cufile")
    fake_cufile.cuFileError = FakeCuFileError
    fake_cufile.Descr = FakeDescr
    fake_cufile.FileHandleType = SimpleNamespace(OPAQUE_FD=17)
    fake_cufile.driver_open = lambda: events.append(("driver_open",))

    def handle_register(descr_ptr: int) -> int:
        events.append(("handle_register", descr_ptr))
        return 0xFACE

    fake_cufile.handle_register = handle_register
    fake_cufile.read = lambda *args: events.append(("read", *args))
    fake_cufile.write = lambda *args: events.append(("write", *args))
    fake_cufile.handle_deregister = lambda handle: events.append(("handle_deregister", handle))
    fake_cufile.driver_close = lambda: events.append(("driver_close",))

    cuda_package = ModuleType("cuda")
    cuda_package.__path__ = []
    bindings_package = ModuleType("cuda.bindings")
    bindings_package.__path__ = []
    bindings_package.cufile = fake_cufile
    cuda_package.bindings = bindings_package

    class FakeTensor:
        is_cuda = True

        def data_ptr(self) -> int:
            return 0xB200

        def numel(self) -> int:
            return 1024

        def element_size(self) -> int:
            return 4

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _device: "Fake B200",
        synchronize=lambda: events.append(("synchronize",)),
    )
    fake_torch.empty = lambda *_args, **_kwargs: FakeTensor()

    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings.cufile", fake_cufile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_name = "_test_cufile_gds_example_cuda13"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    closed_fds: list[int] = []
    monkeypatch.setattr(module.os.path, "getsize", lambda _path: 4096)
    monkeypatch.setattr(module.os, "O_DIRECT", 0, raising=False)
    monkeypatch.setattr(module.os, "open", lambda *_args, **_kwargs: 73)
    monkeypatch.setattr(module.os, "close", closed_fds.append)
    timestamps = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(timestamps))

    demo = module.GPUDirectStorageDemo(use_gds=True)
    throughput_gbps, latency_ms = demo.read_gds("/tmp/fake-gds-input.bin")
    write_throughput_gbps, write_latency_ms = demo.write_gds(
        "/tmp/fake-gds-output.bin",
        FakeTensor(),
    )
    demo.cleanup()

    assert module.CUFILE_AVAILABLE is True
    assert len(descriptors) == 2
    assert descriptors[0].type == fake_cufile.FileHandleType.OPAQUE_FD
    assert descriptors[0].handle.fd == 73
    assert descriptors[1].type == fake_cufile.FileHandleType.OPAQUE_FD
    assert descriptors[1].handle.fd == 73
    assert events == [
        ("driver_open",),
        ("handle_register", 0xD35C),
        ("read", 0xFACE, 0xB200, 4096, 0, 0),
        ("synchronize",),
        ("handle_deregister", 0xFACE),
        ("handle_register", 0xD35C),
        ("write", 0xFACE, 0xB200, 4096, 0, 0),
        ("synchronize",),
        ("handle_deregister", 0xFACE),
        ("driver_close",),
    ]
    assert closed_fds == [73, 73]
    assert throughput_gbps == pytest.approx(4096 / 0.25 / 1e9)
    assert latency_ms == pytest.approx(250.0)
    assert write_throughput_gbps == pytest.approx(4096 / 0.5 / 1e9)
    assert write_latency_ms == pytest.approx(500.0)
    assert demo.driver_initialized is False
