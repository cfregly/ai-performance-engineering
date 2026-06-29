from types import SimpleNamespace

from ch04 import dist_allreduce


class _FakeTensor:
    def __mul__(self, _other):
        return self

    def __getitem__(self, _index):
        return self

    def item(self):
        return 1.0


def test_main_uses_local_rank_for_single_process_nccl_fallback(monkeypatch) -> None:
    captured = {"device": None, "set_device_calls": []}

    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(
        dist_allreduce.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(data_size=16, backend="nccl"),
    )
    monkeypatch.setattr(dist_allreduce.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        dist_allreduce.torch.cuda,
        "set_device",
        lambda device_id: captured["set_device_calls"].append(device_id),
    )
    monkeypatch.setattr(
        dist_allreduce.dist,
        "init_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("env:// missing")),
    )
    monkeypatch.setattr(
        dist_allreduce.torch,
        "ones",
        lambda *_args, **kwargs: captured.__setitem__("device", kwargs["device"]) or _FakeTensor(),
    )

    dist_allreduce.main()

    assert str(captured["device"]) == "cuda:2"
    assert captured["set_device_calls"] == [2]


def test_main_uses_local_rank_instead_of_global_rank_for_nccl_tensor_device(monkeypatch) -> None:
    captured = {"device": None, "set_device_calls": [], "init_kwargs": None}

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        dist_allreduce.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(data_size=16, backend="nccl"),
    )
    monkeypatch.setattr(dist_allreduce.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        dist_allreduce.torch.cuda,
        "set_device",
        lambda device_id: captured["set_device_calls"].append(device_id),
    )
    monkeypatch.setattr(
        dist_allreduce.dist,
        "init_process_group",
        lambda **kwargs: captured.__setitem__("init_kwargs", kwargs),
    )
    monkeypatch.setattr(dist_allreduce.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist_allreduce.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist_allreduce.dist, "barrier", lambda: None)
    monkeypatch.setattr(dist_allreduce.dist, "all_reduce", lambda tensor, op=None: None)
    monkeypatch.setattr(dist_allreduce.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(
        dist_allreduce.torch,
        "ones",
        lambda *_args, **kwargs: captured.__setitem__("device", kwargs["device"]) or _FakeTensor(),
    )

    dist_allreduce.main()

    assert str(captured["device"]) == "cuda:1"
    assert captured["set_device_calls"] == [1]
    assert captured["init_kwargs"]["backend"] == "nccl"
    assert captured["init_kwargs"]["device_id"] == 1


def test_ch04_distributed_demos_use_monotonic_elapsed_timing() -> None:
    dist_source = dist_allreduce.__loader__.get_source(dist_allreduce.__name__)
    assert dist_source is not None
    barrier_source = (
        dist_allreduce.__file__.replace("dist_allreduce.py", "barrier_straggler_multigpu.py")
    )
    with open(barrier_source, encoding="utf-8") as handle:
        barrier_text = handle.read()

    assert "start = time.perf_counter()" in dist_source
    assert "elapsed = time.perf_counter() - start" in dist_source
    assert "time.time()" not in dist_source
    assert "start = time.perf_counter()" in barrier_text
    assert "elapsed = time.perf_counter() - start" in barrier_text
    assert "time.time()" not in barrier_text
