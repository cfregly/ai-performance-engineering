"""Actual two-rank cache handoff checks; Gloo does not qualify NCCL behavior."""

from datetime import timedelta
import time

import pytest
import torch
import torch.distributed as dist


def _handoff_worker(rank, rendezvous, backend):
    from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
        CacheAwareDisaggMultiGPUConfig, _extend_cache_buffer, _stage_runtime_cache,
    )

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        request = 0
        for batch in (1, 2, 3):
            for length in (1, 3, 8):
                for warm in (False, True):
                    cfg = CacheAwareDisaggMultiGPUConfig(batch_size=batch, hidden_size=4,
                                                        context_window=8, dtype=torch.float32)
                    expected = torch.arange(batch * length * 4, device=device, dtype=cfg.dtype).reshape(batch, length, 4)
                    expected = expected + request * 1000
                    active, stored = {}, {}
                    if rank == 0:
                        source = _extend_cache_buffer(
                            cfg, request_id=request, cache=torch.empty(batch, 0, 4, device=device),
                            chunk_kv=expected, kv_buffers={},
                        )
                        assert source.is_contiguous() == (batch == 1 or length == 8)
                        (stored if warm else active)[request] = source
                    metrics = dict.fromkeys(("cache_hits", "cache_misses", "peer_handoffs",
                                             "worker_switches", "kv_transfer_bytes", "shared_reload_bytes"), 0.0)
                    for owner, target in ((0, 1), (1, 0)):
                        _stage_runtime_cache(
                            cfg=cfg, rank=rank, device=device, request_id=request,
                            target_rank=target, current_owner=owner, current_cache_len=length,
                            warm_cache_store=stored, active_caches=active, metrics=metrics,
                        )
                        if rank == target:
                            torch.testing.assert_close(active[request], expected, rtol=0, atol=0)
                        else:
                            assert request not in active
                        dist.barrier()
                    assert metrics["peer_handoffs"] == 1
                    assert metrics["worker_switches"] == 1
                    assert metrics["kv_transfer_bytes"] == expected.numel() * expected.element_size()
                    request += 1
    finally:
        dist.destroy_process_group()


def _run_handoffs(tmp_path, backend):
    context = torch.multiprocessing.spawn(
        _handoff_worker, args=((tmp_path / "rendezvous").as_uri(), backend),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(f"Two-rank {backend} handoff check timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Actual Gloo unavailable")
def test_peer_handoff_preserves_full_batched_cache_on_two_cpu_ranks(tmp_path):
    _run_handoffs(tmp_path, "gloo")


@pytest.mark.skipif(not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                    reason="Actual NCCL and two CUDA GPUs required")
def test_peer_handoff_preserves_full_batched_cache_on_two_cuda_ranks(tmp_path):
    _run_handoffs(tmp_path, "nccl")
