"""Actual sampled-loss storage, autograd independence, and CUDA transfer checks."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from labs.train_distributed.training_utils.deferred_metrics import DeferredTrainingProgress


def test_retains_each_sample_after_source_changes_without_autograd_history():
    progress = DeferredTrainingProgress(num_steps=21, interval=10, device=torch.device("cpu"))
    value = torch.tensor(1.25, requires_grad=True)
    for step, expected in [(0, 1.25), (10, 2.5), (20, 3.75)]:
        with torch.no_grad():
            value.fill_(expected)
        progress.record(step=step, loss=value, tokens=128 + step)
    with torch.no_grad():
        value.fill_(100)
    rows = progress.read()
    assert [(row.step, row.loss, row.tokens) for row in rows] == [
        (0, 1.25, 128), (10, 2.5, 138), (20, 3.75, 148)
    ]
    assert value.grad is None


def test_partial_training_does_not_read_unwritten_buffer_entries():
    progress = DeferredTrainingProgress(num_steps=100, interval=10, device=torch.device("cpu"))
    assert progress.read() == []
    progress.record(step=0, loss=torch.tensor(0.5), tokens=64)
    assert [row.loss for row in progress.read()] == [0.5]
    with pytest.raises(ValueError, match="declared step interval"):
        progress.record(step=20, loss=torch.tensor(0.6), tokens=64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA device required")
def test_cuda_records_never_transfer_loss_to_host_until_read():
    class Transfers(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.host_copies = 0
            self.scalar_reads = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            options = kwargs or {}
            if func is torch.ops.aten._to_copy.default and options.get("device") == torch.device("cpu"):
                self.host_copies += 1
            if func is torch.ops.aten._local_scalar_dense.default:
                self.scalar_reads += 1
            return func(*args, **options)

    device = torch.device("cuda", 0)
    progress = DeferredTrainingProgress(num_steps=20, interval=10, device=device)
    losses = torch.tensor([1.25, 2.5], device=device)
    trace = Transfers()
    with trace:
        progress.record(step=0, loss=losses[0], tokens=128)
        progress.record(step=10, loss=losses[1], tokens=128)
        assert trace.host_copies == 0 and trace.scalar_reads == 0
        rows = progress.read()
    assert trace.host_copies == 1 and trace.scalar_reads == 0
    assert [row.loss for row in rows] == [1.25, 2.5]
