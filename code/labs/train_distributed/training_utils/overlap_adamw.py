"""Grouped AdamW overlap for the explicit, bounded DDP optimizer opt-in.

Only one backward per update, with every trainable parameter used, is supported.
The one-rank path requires a single producer CUDA stream for all gradients.
One optimizer preserves parameter state across DDP bucket rebuilds. The caller
must call step() after backward and before the next forward.
"""
import threading

import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks


def parameter_groups(parameters, bucket_bytes):
    if bucket_bytes <= 0:
        raise ValueError('bucket_bytes must be positive')
    groups, current, size = [], [], 0
    for p in reversed(parameters):
        current.append(p)
        size += p.numel() * p.element_size()
        if size >= bucket_bytes:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    return groups


def _ddp_hook(state, bucket):
    return state.reduce_and_update(bucket)


class OverlappedAdamW:
    def __init__(self, model, optimizer_factory, learning_rate, *, ddp=None, bucket_bytes=50*1024*1024):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters or any(p.device.type != 'cuda' for p in self.parameters):
            raise ValueError('OverlappedAdamW requires nonempty CUDA parameters')
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in self.parameters):
            raise ValueError('All parameters must be on one local CUDA device')
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError('Duplicate parameter ownership')
        self.optimizer = optimizer_factory(self.parameters, learning_rate, prefer_fused=True)
        if len(self.optimizer.param_groups) != 1 or not self.optimizer.defaults.get('fused'):
            raise ValueError('This experiment requires one fused AdamW parameter group')
        self.stream = torch.cuda.Stream(device=self.device)
        self.producer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(self.producer_stream)
        self.lock = threading.Lock()
        self.seen = set()
        self.updated = set()
        self.expected = {id(p) for p in self.parameters}
        self.completed_steps = 0
        self.group_updates = 0
        self.ddp = ddp
        self.handles = []
        if ddp is None:
            self.groups = parameter_groups(self.parameters, bucket_bytes)
            self.group_for = {id(p): i for i,g in enumerate(self.groups) for p in g}
            self.ready = [0] * len(self.groups)
            for p in self.parameters:
                self.handles.append(p.register_post_accumulate_grad_hook(self.parameter_ready))
        else:
            if ddp.module is not model:
                raise ValueError('DDP must wrap the supplied model')
            if not ddp.gradient_as_bucket_view:
                raise ValueError('Optimizer overlap requires DDP gradient bucket views')
            ddp.register_comm_hook(self, _ddp_hook)

    def _update(self, parameters):
        ids = {id(p) for p in parameters}
        if not ids <= self.expected or ids & self.updated:
            raise RuntimeError('Unknown or repeated parameter update')
        if any(p.grad is None for p in parameters):
            raise RuntimeError('An update was scheduled before its gradients were ready')
        original = self.optimizer.param_groups[0]['params']
        self.optimizer.param_groups[0]['params'] = parameters
        try:
            for p in parameters:
                p.grad.record_stream(self.stream)
            self.optimizer.step()
        finally:
            self.optimizer.param_groups[0]['params'] = original
        self.updated.update(ids)
        self.group_updates += 1

    def parameter_ready(self, parameter):
        with self.lock:
            if torch.cuda.current_stream(self.device) != self.producer_stream:
                raise RuntimeError('This one-rank optimizer requires a single producer CUDA stream')
            key = id(parameter)
            if key in self.seen:
                raise RuntimeError('Multiple backward calls require the synchronous accumulation path')
            self.seen.add(key)
            group = self.group_for[key]
            self.ready[group] += 1
            if self.ready[group] != len(self.groups[group]):
                return
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self._update(self.groups[group])

    def reduce_and_update(self, bucket):
        # The default hook divides by world size before its all-reduce, just as
        # ordinary DDP does. Waiting within the optimizer stream orders its read.
        future = default_hooks.allreduce_hook(self.ddp.process_group, bucket)
        with self.lock, torch.cuda.stream(self.stream):
            future.wait()
            parameters = bucket.parameters()
            for p,g in zip(parameters, bucket.gradients()):
                if p.grad is None or not p.grad.is_set_to(g):
                    raise RuntimeError('DDP parameter gradient no longer matches its bucket view')
            self._update(parameters)
        completed = torch.futures.Future()
        completed.set_result(bucket.buffer())
        return completed

    def step(self):
        with self.lock:
            if self.updated != self.expected:
                raise RuntimeError(f'Incomplete optimizer update: {len(self.updated)}/{len(self.expected)} parameters')
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.completed_steps += 1
            self.seen.clear()
            self.updated.clear()
            if self.ddp is None:
                self.ready = [0] * len(self.groups)

    def zero_grad(self, *, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
