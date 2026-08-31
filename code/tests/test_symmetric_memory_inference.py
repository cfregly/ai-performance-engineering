import os
import socket
import sys
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _setup_env(world_size: int, port: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)


def _kv_worker(rank: int, world_size: int, seq_len: int, port: int) -> None:
    _setup_env(world_size, port, rank)
    from ch16.symmetric_memory_inference import demo_kv_cache

    demo_kv_cache(batch_size=2, seq_len=seq_len)
    if dist.is_initialized():
        dist.destroy_process_group()


def _multi_model_worker(rank: int, world_size: int, size_mb: int, port: int) -> None:
    _setup_env(world_size, port, rank)
    from ch16.symmetric_memory_inference import demo_multi_model

    demo_multi_model(size_mb=size_mb)
    if dist.is_initialized():
        dist.destroy_process_group()


def _speculative_worker(rank: int, world_size: int, steps: int, port: int) -> None:
    _setup_env(world_size, port, rank)
    from ch16.symmetric_memory_inference import demo_speculative

    demo_speculative(num_steps=steps)
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_kv_cache_demo_runs_on_gloo():
    from ch16.symmetric_memory_inference import symmetric_memory_available

    if not symmetric_memory_available() or torch.cuda.device_count() < 2:
        pytest.skip("Symmetric-memory demo requires runtime support and at least two GPUs")

    world_size = 2
    port = _find_free_port()
    spawn(_kv_worker, args=(world_size, 16, port), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_multi_model_demo_runs_on_gloo():
    from ch16.symmetric_memory_inference import symmetric_memory_available

    if not symmetric_memory_available() or torch.cuda.device_count() < 2:
        pytest.skip("Symmetric-memory demo requires runtime support and at least two GPUs")

    world_size = 2
    port = _find_free_port()
    spawn(_multi_model_worker, args=(world_size, 2, port), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_speculative_demo_runs_on_gloo():
    from ch16.symmetric_memory_inference import symmetric_memory_available

    if not symmetric_memory_available() or torch.cuda.device_count() < 2:
        pytest.skip("Symmetric-memory demo requires runtime support and at least two GPUs")

    world_size = 2
    port = _find_free_port()
    spawn(_speculative_worker, args=(world_size, 2, port), nprocs=world_size, join=True)
