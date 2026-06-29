"""Reusable pipeline-parallel executors for the distributed training lab."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Literal, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn

ScheduleType = Literal["gpipe", "1f1b", "dualpipe", "dualpipev"]
Tensor = torch.Tensor


def _finish_pipeline_loss(loss_buffer: Tensor, loss_count: int, n_micro: int) -> float:
    if loss_count <= 0 or n_micro <= 0:
        return 0.0
    total = loss_buffer[:loss_count].sum()
    return float(total.detach().cpu()) / n_micro


@dataclass
class PipelineConfig:
    """Configuration for a toy pipeline demo."""

    schedule: ScheduleType = "gpipe"
    n_stages: int = 4
    batch_size: int = 64
    micro_batch_size: int = 16
    input_dim: int = 50
    hidden_dim: int = 500
    depth: int = 6
    learning_rate: float = 1e-4
    non_blocking: bool = True
    dtype: torch.dtype = torch.float32
    device_ids: Optional[List[int]] = None
    seed: int = 42
    dual_window: int = 2

    @property
    def n_micro(self) -> int:
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        return self.batch_size // self.micro_batch_size


@dataclass
class StageTelemetry:
    forward_ops: int = 0
    backward_ops: int = 0
    active_ticks: int = 0
    idle_ticks: int = 0
    max_forward_queue: int = 0
    max_backward_queue: int = 0

    def merge(self, other: "StageTelemetry") -> None:
        self.forward_ops += other.forward_ops
        self.backward_ops += other.backward_ops
        self.active_ticks += other.active_ticks
        self.idle_ticks += other.idle_ticks
        self.max_forward_queue = max(self.max_forward_queue, other.max_forward_queue)
        self.max_backward_queue = max(self.max_backward_queue, other.max_backward_queue)


class PipelineTelemetry:
    """Collects queue depth and utilization stats per stage."""

    def __init__(self, n_stages: int, *, schedule: ScheduleType):
        self.schedule = schedule
        self.stage_stats: List[StageTelemetry] = [StageTelemetry() for _ in range(n_stages)]
        self.total_ticks = 0
        self._active_this_tick = [False] * n_stages
        self._n_stages = n_stages

    def start_tick(self) -> None:
        self.total_ticks += 1
        self._active_this_tick = [False] * self._n_stages

    def end_tick(self) -> None:
        for stage_id, active in enumerate(self._active_this_tick):
            if not active:
                self.stage_stats[stage_id].idle_ticks += 1

    def mark_forward(self, stage_id: int) -> None:
        stats = self.stage_stats[stage_id]
        stats.forward_ops += 1
        if not self._active_this_tick[stage_id]:
            stats.active_ticks += 1
            self._active_this_tick[stage_id] = True

    def mark_backward(self, stage_id: int) -> None:
        stats = self.stage_stats[stage_id]
        stats.backward_ops += 1
        if not self._active_this_tick[stage_id]:
            stats.active_ticks += 1
            self._active_this_tick[stage_id] = True

    def record_queues(
        self,
        fwd_queues: Iterable[Deque[Tuple[int, Tensor]]],
        bwd_queues: Iterable[Deque[Tuple[int, Optional[Tensor]]]],
    ) -> None:
        for stage_id, queue in enumerate(fwd_queues):
            self.stage_stats[stage_id].max_forward_queue = max(
                self.stage_stats[stage_id].max_forward_queue, len(queue)
            )
        for stage_id, queue in enumerate(bwd_queues):
            self.stage_stats[stage_id].max_backward_queue = max(
                self.stage_stats[stage_id].max_backward_queue, len(queue)
            )

    def merge(self, other: "PipelineTelemetry") -> None:
        if len(self.stage_stats) != len(other.stage_stats):
            raise ValueError("Cannot merge telemetry from mismatched stage counts.")
        self.total_ticks += other.total_ticks
        for idx, stage in enumerate(other.stage_stats):
            self.stage_stats[idx].merge(stage)

    def utilization(self, stage_id: int) -> float:
        stats = self.stage_stats[stage_id]
        total = stats.active_ticks + stats.idle_ticks
        return stats.active_ticks / total if total else 0.0

    def summary_lines(self, prefix: str = "") -> List[str]:
        lines = [
            f"{prefix}schedule={self.schedule} total_ticks={self.total_ticks}",
        ]
        for stage_id, stats in enumerate(self.stage_stats):
            util = self.utilization(stage_id) * 100.0
            lines.append(
                f"{prefix}stage{stage_id}: util={util:5.1f}% | "
                f"fwd={stats.forward_ops:3d} bwd={stats.backward_ops:3d} | "
                f"max_fwd_q={stats.max_forward_queue:2d} max_bwd_q={stats.max_backward_queue:2d}"
            )
        return lines


def add_pipeline_args(parser) -> None:
    """Attach shared CLI arguments to a parser."""
    parser.add_argument("--steps", type=int, default=6, help="Number of synthetic training steps.")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size.")
    parser.add_argument(
        "--n-stages",
        type=int,
        default=None,
        help="Number of pipeline stages (defaults to all visible GPUs).",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Microbatch size (defaults to batch size for baseline scripts).",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device IDs for stages (repeat IDs to simulate multi-stage on one GPU).",
    )
    parser.add_argument("--hidden-dim", type=int, default=500, help="Toy model hidden dimension.")
    parser.add_argument("--depth", type=int, default=6, help="Number of linear+ReLU blocks per stage.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print loss/telemetry every N steps.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--dual-window",
        type=int,
        default=None,
        help="Max microbatches to keep inflight on stage0 for DualPipe demos.",
    )


def parse_device_ids(raw: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated CUDA device IDs from CLI."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def resolve_n_stages(n_stages: Optional[int]) -> int:
    """Resolve pipeline stage count, defaulting to all visible GPUs."""
    if n_stages is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Pipeline demos require CUDA GPUs.")
        available = torch.cuda.device_count()
        if available < 2:
            raise RuntimeError("Pipeline demos require >=2 GPUs.")
        return available
    resolved = int(n_stages)
    if resolved < 2:
        raise ValueError("n_stages must be >= 2 for pipeline demos.")
    return resolved


def _build_toy_model(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_features = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        in_features = hidden_dim
    layers.append(nn.Linear(hidden_dim, input_dim))
    return nn.Sequential(*layers)


class PipelineExperiment:
    """Creates the toy pipeline and executes the requested schedule."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("Pipeline demos require CUDA GPUs.")
        available = torch.cuda.device_count()
        if config.device_ids is None:
            if config.n_stages > available:
                raise RuntimeError(
                    f"Need {config.n_stages} CUDA devices but only {available} detected. "
                    "Set CUDA_VISIBLE_DEVICES or lower --n_stages."
                )
            self.devices = [f"cuda:{idx}" for idx in range(config.n_stages)]
        else:
            if any(dev < 0 or dev >= available for dev in config.device_ids):
                raise RuntimeError(
                    f"device_ids must reference visible CUDA devices (0-{available - 1})."
                )
            self.devices = [f"cuda:{dev}" for dev in config.device_ids]
        if len(self.devices) < config.n_stages:
            raise ValueError("device_ids must include one entry per stage.")

        self.model = _build_toy_model(config.input_dim, config.hidden_dim, config.depth)
        self.stages = self._split_into_stages()
        self.optimizers = [torch.optim.Adam(stage.parameters(), lr=config.learning_rate) for stage in self.stages]
        self.criterion = nn.MSELoss()

    def _split_into_stages(self) -> List[nn.Sequential]:
        n_layers = len(self.model)
        per_stage = (n_layers + self.config.n_stages - 1) // self.config.n_stages
        stages: List[nn.Sequential] = []
        start = 0
        for stage_id in range(self.config.n_stages):
            end = min(start + per_stage, n_layers)
            sub = nn.Sequential(*self.model[start:end]).to(self.devices[stage_id], dtype=self.config.dtype)
            sub.train()
            stages.append(sub)
            start = end
        return stages

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    def run_batch(self, inputs: Tensor, targets: Tensor) -> Tuple[float, PipelineTelemetry]:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        if self.config.schedule == "gpipe":
            loss_value, telemetry = self._run_gpipe(inputs, targets)
        elif self.config.schedule == "1f1b":
            loss_value, telemetry = self._run_onef1b(inputs, targets)
        elif self.config.schedule == "dualpipe":
            loss_value, telemetry = self._run_dualpipe(inputs, targets)
        elif self.config.schedule == "dualpipev":
            loss_value, telemetry = self._run_dualpipev(inputs, targets)
        else:
            raise ValueError(f"Unknown schedule {self.config.schedule}")

        for opt in self.optimizers:
            opt.step()

        return loss_value, telemetry

    def _run_gpipe(self, inputs: Tensor, targets: Tensor) -> Tuple[float, PipelineTelemetry]:
        cfg = self.config
        n_micro = cfg.n_micro
        last_stage = cfg.n_stages - 1

        micro_ins = inputs.chunk(n_micro)
        micro_tgts = targets.chunk(n_micro)

        fwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        bwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        saved_activations: List[Dict[int, Tuple[Tensor, Tensor]]] = [dict() for _ in range(cfg.n_stages)]
        losses: Dict[int, Tensor] = {}

        telemetry = PipelineTelemetry(cfg.n_stages, schedule=cfg.schedule)
        loss_buffer = torch.empty(n_micro, dtype=torch.float64, device=self.devices[last_stage])
        loss_count = 0

        next_micro = 0
        forward_ticks = n_micro + cfg.n_stages - 1
        for _ in range(forward_ticks):
            telemetry.start_tick()
            pending: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
            if next_micro < n_micro:
                fwd_queues[0].append((next_micro, micro_ins[next_micro]))
                next_micro += 1

            telemetry.record_queues(fwd_queues, bwd_queues)

            for stage_id in range(cfg.n_stages):
                device = self.devices[stage_id]
                if not fwd_queues[stage_id]:
                    continue
                mb_id, data = fwd_queues[stage_id].popleft()
                x = data.to(device, non_blocking=cfg.non_blocking).detach().requires_grad_(True)
                out = self.stages[stage_id](x)
                saved_activations[stage_id][mb_id] = (x, out)
                telemetry.mark_forward(stage_id)

                if stage_id < last_stage:
                    out_next = out.detach().to(self.devices[stage_id + 1], non_blocking=cfg.non_blocking)
                    pending[stage_id + 1].append((mb_id, out_next))
                else:
                    y = micro_tgts[mb_id].to(device, non_blocking=cfg.non_blocking)
                    loss = self.criterion(out, y) / n_micro
                    losses[mb_id] = loss
                    loss_buffer[loss_count].copy_(loss.detach())
                    loss_count += 1

            for stage_id in range(1, cfg.n_stages):
                while pending[stage_id]:
                    fwd_queues[stage_id].append(pending[stage_id].popleft())

            telemetry.end_tick()

        for mb_id in reversed(range(n_micro)):
            bwd_queues[last_stage].append((mb_id, None))

        backward_ticks = n_micro + cfg.n_stages - 1
        for _ in range(backward_ticks):
            telemetry.start_tick()
            pending: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
            telemetry.record_queues(fwd_queues, bwd_queues)

            for stage_id in reversed(range(cfg.n_stages)):
                device = self.devices[stage_id]
                if not bwd_queues[stage_id]:
                    continue

                mb_id, grad_output = bwd_queues[stage_id].popleft()
                telemetry.mark_backward(stage_id)

                if stage_id == last_stage and grad_output is None:
                    retain = mb_id != 0
                    loss = losses.pop(mb_id)
                    loss.backward(retain_graph=retain)
                    x, out = saved_activations[stage_id].pop(mb_id)
                else:
                    x, out = saved_activations[stage_id].pop(mb_id)
                    out.backward(gradient=grad_output)

                if stage_id > 0:
                    grad_input = x.grad.detach().to(self.devices[stage_id - 1], non_blocking=cfg.non_blocking)
                    pending[stage_id - 1].append((mb_id, grad_input))
                    x.grad = None

            for stage_id in range(cfg.n_stages - 1):
                while pending[stage_id]:
                    bwd_queues[stage_id].append(pending[stage_id].popleft())

            telemetry.end_tick()

        avg_loss = _finish_pipeline_loss(loss_buffer, loss_count, n_micro)
        return avg_loss, telemetry

    def _run_onef1b(self, inputs: Tensor, targets: Tensor) -> Tuple[float, PipelineTelemetry]:
        cfg = self.config
        n_micro = cfg.n_micro
        last_stage = cfg.n_stages - 1

        micro_ins = inputs.chunk(n_micro)
        micro_tgts = targets.chunk(n_micro)

        fwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        bwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        saved_activations: List[Dict[int, Tuple[Tensor, Tensor]]] = [dict() for _ in range(cfg.n_stages)]
        losses: Dict[int, Tensor] = {}

        telemetry = PipelineTelemetry(cfg.n_stages, schedule=cfg.schedule)
        loss_buffer = torch.empty(n_micro, dtype=torch.float64, device=self.devices[last_stage])
        loss_count = 0

        next_micro = 0
        completed = 0
        pending_fwd: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        pending_bwd: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]

        while completed < n_micro:
            telemetry.start_tick()
            for queue in pending_fwd:
                queue.clear()
            for queue in pending_bwd:
                queue.clear()
            if next_micro < n_micro:
                fwd_queues[0].append((next_micro, micro_ins[next_micro]))
                next_micro += 1

            telemetry.record_queues(fwd_queues, bwd_queues)

            for stage_id in range(cfg.n_stages):
                device = self.devices[stage_id]

                if fwd_queues[stage_id]:
                    mb_id, data = fwd_queues[stage_id].popleft()
                    x = data.to(device, non_blocking=cfg.non_blocking).detach().requires_grad_(True)
                    out = self.stages[stage_id](x)
                    saved_activations[stage_id][mb_id] = (x, out)
                    telemetry.mark_forward(stage_id)

                    if stage_id < last_stage:
                        out_next = out.detach().to(self.devices[stage_id + 1], non_blocking=cfg.non_blocking)
                        pending_fwd[stage_id + 1].append((mb_id, out_next))
                    else:
                        y = micro_tgts[mb_id].to(device, non_blocking=cfg.non_blocking)
                        loss = self.criterion(out, y) / n_micro
                        losses[mb_id] = loss
                        loss_buffer[loss_count].copy_(loss.detach())
                        loss_count += 1
                        pending_bwd[stage_id].append((mb_id, None))

                if bwd_queues[stage_id]:
                    mb_id, grad_output = bwd_queues[stage_id].popleft()
                    telemetry.mark_backward(stage_id)
                    x, out = saved_activations[stage_id].pop(mb_id)

                    if stage_id == last_stage and grad_output is None:
                        retain = mb_id != n_micro - 1
                        loss = losses.pop(mb_id)
                        loss.backward(retain_graph=retain)
                    else:
                        out.backward(gradient=grad_output)

                    if stage_id > 0:
                        grad_input = x.grad.detach().to(self.devices[stage_id - 1], non_blocking=cfg.non_blocking)
                        pending_bwd[stage_id - 1].append((mb_id, grad_input))
                        x.grad = None
                    else:
                        completed += 1

            for stage_id in range(1, cfg.n_stages):
                if pending_fwd[stage_id]:
                    fwd_queues[stage_id].extend(pending_fwd[stage_id])
                    pending_fwd[stage_id].clear()

            for stage_id in range(cfg.n_stages):
                if pending_bwd[stage_id]:
                    bwd_queues[stage_id].extend(pending_bwd[stage_id])
                    pending_bwd[stage_id].clear()

            telemetry.end_tick()

        avg_loss = _finish_pipeline_loss(loss_buffer, loss_count, n_micro)
        return avg_loss, telemetry

    def _run_dualpipe(self, inputs: Tensor, targets: Tensor) -> Tuple[float, PipelineTelemetry]:
        cfg = self.config
        n_micro = cfg.n_micro
        if n_micro < cfg.n_stages:
            raise ValueError("DualPipe requires #microbatches >= #stages to keep the pipe busy.")
        last_stage = cfg.n_stages - 1
        dual_window = max(1, cfg.dual_window)

        micro_ins = inputs.chunk(n_micro)
        micro_tgts = targets.chunk(n_micro)

        fwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        bwd_queues: List[Deque[Tuple[int, Optional[Tensor]]]] = [deque() for _ in range(cfg.n_stages)]
        saved_activations: List[Dict[int, Tuple[Tensor, Tensor]]] = [dict() for _ in range(cfg.n_stages)]
        losses: Dict[int, Tensor] = {}

        telemetry = PipelineTelemetry(cfg.n_stages, schedule=cfg.schedule)
        loss_buffer = torch.empty(n_micro, dtype=torch.float64, device=self.devices[last_stage])
        loss_count = 0
        completed = 0
        next_micro = 0

        def any_work_pending() -> bool:
            if completed < n_micro:
                return True
            return any(queue for queue in fwd_queues) or any(queue for queue in bwd_queues)

        while any_work_pending():
            telemetry.start_tick()
            # Keep stage0 prefetched with up to dual_window microbatches
            while next_micro < n_micro and len(fwd_queues[0]) < dual_window:
                fwd_queues[0].append((next_micro, micro_ins[next_micro]))
                next_micro += 1

            telemetry.record_queues(fwd_queues, bwd_queues)

            for stage_id in range(cfg.n_stages):
                device = self.devices[stage_id]

                # Forward lane
                if fwd_queues[stage_id]:
                    mb_id, data = fwd_queues[stage_id].popleft()
                    x = data.to(device, non_blocking=cfg.non_blocking).detach().requires_grad_(True)
                    out = self.stages[stage_id](x)
                    saved_activations[stage_id][mb_id] = (x, out)
                    telemetry.mark_forward(stage_id)

                    if stage_id < last_stage:
                        out_next = out.detach().to(self.devices[stage_id + 1], non_blocking=cfg.non_blocking)
                        fwd_queues[stage_id + 1].append((mb_id, out_next))
                    else:
                        y = micro_tgts[mb_id].to(device, non_blocking=cfg.non_blocking)
                        loss = self.criterion(out, y) / n_micro
                        losses[mb_id] = loss
                        loss_buffer[loss_count].copy_(loss.detach())
                        loss_count += 1
                        bwd_queues[stage_id].append((mb_id, None))

                # Backward lane
                if bwd_queues[stage_id]:
                    mb_id, grad_output = bwd_queues[stage_id].popleft()
                    telemetry.mark_backward(stage_id)
                    x, out = saved_activations[stage_id].pop(mb_id)

                    if stage_id == last_stage and grad_output is None:
                        retain = mb_id != n_micro - 1
                        loss = losses.pop(mb_id)
                        loss.backward(retain_graph=retain)
                    else:
                        out.backward(gradient=grad_output)

                    if stage_id > 0:
                        grad_input = x.grad.detach().to(
                            self.devices[stage_id - 1],
                            non_blocking=cfg.non_blocking,
                        )
                        bwd_queues[stage_id - 1].append((mb_id, grad_input))
                        x.grad = None
                    else:
                        completed += 1

            telemetry.end_tick()

        avg_loss = _finish_pipeline_loss(loss_buffer, loss_count, n_micro)
        return avg_loss, telemetry

    def _run_dualpipev(self, inputs: Tensor, targets: Tensor) -> Tuple[float, PipelineTelemetry]:
        """Approximate the DualPipeV / ZB-V wave schedule by draining queues aggressively each tick."""

        cfg = self.config
        n_micro = cfg.n_micro
        if n_micro < cfg.n_stages:
            raise ValueError("DualPipeV requires #microbatches >= #stages to avoid underfilling the wave.")

        micro_ins = inputs.chunk(n_micro)
        micro_tgts = targets.chunk(n_micro)

        fwd_queues: List[Deque[Tuple[int, Tensor]]] = [deque() for _ in range(cfg.n_stages)]
        bwd_queues: List[Deque[Tuple[int, Optional[Tensor]]]] = [deque() for _ in range(cfg.n_stages)]
        saved_activations: List[Dict[int, Tuple[Tensor, Tensor]]] = [dict() for _ in range(cfg.n_stages)]
        losses: Dict[int, Tensor] = {}

        telemetry = PipelineTelemetry(cfg.n_stages, schedule=cfg.schedule)
        loss_buffer = torch.empty(n_micro, dtype=torch.float64, device=self.devices[cfg.n_stages - 1])
        loss_count = 0
        completed = 0
        next_micro = 0

        while completed < n_micro or any(queue for queue in fwd_queues) or any(queue for queue in bwd_queues):
            telemetry.start_tick()
            while next_micro < n_micro and len(fwd_queues[0]) < cfg.dual_window:
                fwd_queues[0].append((next_micro, micro_ins[next_micro]))
                next_micro += 1

            telemetry.record_queues(fwd_queues, bwd_queues)

            # Forward sweep (0 -> n-1)
            for stage_id in range(cfg.n_stages):
                device = self.devices[stage_id]
                while fwd_queues[stage_id]:
                    mb_id, data = fwd_queues[stage_id].popleft()
                    x = data.to(device, non_blocking=cfg.non_blocking).detach().requires_grad_(True)
                    out = self.stages[stage_id](x)
                    saved_activations[stage_id][mb_id] = (x, out)
                    telemetry.mark_forward(stage_id)

                    if stage_id < cfg.n_stages - 1:
                        out_next = out.detach().to(self.devices[stage_id + 1], non_blocking=cfg.non_blocking)
                        fwd_queues[stage_id + 1].append((mb_id, out_next))
                    else:
                        y = micro_tgts[mb_id].to(device, non_blocking=cfg.non_blocking)
                        loss = self.criterion(out, y) / n_micro
                        losses[mb_id] = loss
                        loss_buffer[loss_count].copy_(loss.detach())
                        loss_count += 1
                        bwd_queues[stage_id].append((mb_id, None))

            # Backward sweep (n-1 -> 0) with aggressive draining.
            for stage_id in reversed(range(cfg.n_stages)):
                while bwd_queues[stage_id]:
                    mb_id, grad_output = bwd_queues[stage_id].popleft()
                    telemetry.mark_backward(stage_id)
                    x, out = saved_activations[stage_id].pop(mb_id)
                    if stage_id == cfg.n_stages - 1 and grad_output is None:
                        retain = mb_id != n_micro - 1
                        loss = losses.pop(mb_id)
                        loss.backward(retain_graph=retain)
                    else:
                        out.backward(gradient=grad_output)

                    if stage_id > 0:
                        grad_input = x.grad.detach().to(
                            self.devices[stage_id - 1],
                            non_blocking=cfg.non_blocking,
                        )
                        bwd_queues[stage_id - 1].append((mb_id, grad_input))
                        x.grad = None
                    else:
                        completed += 1

            telemetry.end_tick()

        avg_loss = _finish_pipeline_loss(loss_buffer, loss_count, n_micro)
        return avg_loss, telemetry


def format_telemetry(name: str, telemetry: PipelineTelemetry) -> str:
    """Pretty string for console logging."""
    lines = [f"[{name}] " + telemetry.summary_lines(prefix="")[0]]
    for stage_line in telemetry.summary_lines(prefix="  "):
        if "schedule" in stage_line:
            continue
        lines.append(stage_line)
    return "\n".join(lines)
