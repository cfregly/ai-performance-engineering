"""Explicit optimizer-state sharding for the ZeRO-1 teaching examples."""

from time import perf_counter

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class OptimizerStateSharder:
    """Average all gradients, update owned parameters, and broadcast updates."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        shard_size, remainder = divmod(len(self.params), self.world_size)
        start = self.rank * shard_size + min(self.rank, remainder)
        end = start + shard_size + int(self.rank < remainder)
        self.local_indices = list(range(start, end))
        self.local_params = {self.params[i] for i in self.local_indices}
        self.owners = [
            rank
            for rank in range(self.world_size)
            for _ in range(shard_size + int(rank < remainder))
        ]
        # Gradient averaging assumes every rank starts from the same weights.
        with torch.no_grad():
            for parameter in self.params:
                dist.broadcast(parameter, src=0)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p in self.local_params]
        self.communication_time = 0.0
        self.step_time = 0.0

    def _synchronize(self) -> None:
        for device in {p.device for p in self.params if p.is_cuda}:
            torch.cuda.synchronize(device)

    def step(self, closure=None):
        start = perf_counter()
        for parameter in self.params:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)
        self._synchronize()
        self.communication_time += perf_counter() - start
        result = self.optimizer.step(closure)
        with torch.no_grad():
            for parameter, owner in zip(self.params, self.owners, strict=True):
                dist.broadcast(parameter, src=owner)
        self._synchronize()
        self.step_time += perf_counter() - start
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        # The wrapped optimizer owns only this rank's shard. Every replicated
        # gradient participates in the next all-reduce and must be cleared.
        for parameter in self.params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                with torch.no_grad():
                    parameter.grad.zero_()
