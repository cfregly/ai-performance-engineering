"""Matched training workload for the dense-DDP versus sharded-AdamW comparison.

Both variants use the same model, eager FP32 execution, input generation,
accumulation, clipping, warmup and reporting. The optimized
variant changes gradient communication and optimizer-state ownership only. The
optional synthetic communication payload uses BF16 in both variants.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core.benchmark.gpu_requirements import require_min_gpus
from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.utils import get


@dataclass
class _Zero2CommunicationEvidence:
    mechanism: str
    hook_invocations: int = 0
    reduce_scatter_completions: int = 0
    all_gather_completions: int = 0
    process_group: Optional[dist.ProcessGroup] = None


def _tracked_reduce_scatter_allgather_hook(
    state: _Zero2CommunicationEvidence,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Execute and record the completed RS/AG operations for one DDP bucket."""

    state.hook_invocations += 1
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = group.size()
    buffer = bucket.buffer()
    if world_size <= 1:
        future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        future.set_result(buffer)
        return future

    buffer.div_(world_size)
    flat = buffer
    numel = flat.numel()
    shard_size = (numel + world_size - 1) // world_size
    padded = None
    if numel % world_size != 0:
        padded = flat.new_zeros(shard_size * world_size)
        padded[:numel].copy_(flat)
        flat = padded

    output = flat.new_empty(shard_size)
    if dist.get_backend(group) == "gloo":
        dist.reduce_scatter_tensor(output, flat, group=group)
        state.reduce_scatter_completions += 1
        gathered = output.new_empty(shard_size * world_size)
        dist.all_gather_into_tensor(gathered, output, group=group)
        state.all_gather_completions += 1
        if padded is not None:
            gathered = gathered[:numel]
        buffer.copy_(gathered)
        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    work = dist.reduce_scatter_tensor(output, flat, group=group, async_op=True)

    def finish_all_gather(future: torch.futures.Future) -> torch.Tensor:
        shard = future.value()[0]
        state.reduce_scatter_completions += 1
        gathered = shard.new_empty(shard_size * world_size)
        dist.all_gather_into_tensor(gathered, shard, group=group)
        state.all_gather_completions += 1
        if padded is not None:
            gathered = gathered[:numel]
        buffer.copy_(gathered)
        return buffer

    return work.get_future().then(finish_all_gather)


_tracked_reduce_scatter_allgather_hook.__annotations__["bucket"] = dist.GradBucket
_tracked_reduce_scatter_allgather_hook.__annotations__["return"] = (
    torch.futures.Future[torch.Tensor]
)


def get_zero2_communication_evidence(model: object) -> dict[str, object]:
    """Return production-owned communication evidence for a constructed DDP model."""

    evidence = getattr(model, "_aisp_zero2_communication_evidence", None)
    if not isinstance(evidence, _Zero2CommunicationEvidence):
        return {
            "mechanism": "unobserved",
            "hook_invocations": 0,
            "reduce_scatter_completions": 0,
            "all_gather_completions": 0,
        }
    return {
        "mechanism": evidence.mechanism,
        "hook_invocations": int(evidence.hook_invocations),
        "reduce_scatter_completions": int(evidence.reduce_scatter_completions),
        "all_gather_completions": int(evidence.all_gather_completions),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--extra-grad-mb", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--verification-only",
        action="store_true",
        help="Run the bounded child-result correctness profile without performance timing.",
    )
    parser.add_argument(
        "--verification-backend",
        choices=("gloo", "nccl"),
        default=None,
        help="Distributed backend for --verification-only (explicit; never a fallback).",
    )
    args = parser.parse_args(argv)
    if min(args.steps, args.hidden_size, args.batch_size, args.grad_accum) <= 0:
        parser.error("steps, hidden-size, batch-size and grad-accum must be positive")
    if args.extra_grad_mb < 0 or not 0 < args.learning_rate < float("inf"):
        parser.error("extra-grad-mb must be nonnegative and learning-rate finite positive")
    if bool(args.verification_backend) != bool(args.verification_only):
        parser.error("--verification-only and --verification-backend must be provided together")
    return args


def build_model(hidden_size, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device=device, dtype=torch.float32)


def build_training_components(model, learning_rate, *, optimized, device_ids=None):
    # Imported lazily so legacy module-level helper APIs remain compatible.
    from labs.train_distributed.optimized_zero2_multigpu import (
        _build_optimizer,
        _optimizer_cfg,
    )

    ddp = DDP(model, device_ids=device_ids, static_graph=True,
              gradient_as_bucket_view=True, bucket_cap_mb=25)
    if optimized:
        evidence = _Zero2CommunicationEvidence(
            mechanism="reduce-scatter-all-gather",
            process_group=dist.group.WORLD,
        )
        ddp.register_comm_hook(state=evidence, hook=_tracked_reduce_scatter_allgather_hook)
        optimizer = _build_optimizer(ddp.parameters(), learning_rate)
    else:
        evidence = _Zero2CommunicationEvidence(mechanism="ddp-all-reduce")
        options = _optimizer_cfg(ddp.parameters(), learning_rate)
        options.pop("optimizer_class")
        optimizer = torch.optim.AdamW(ddp.parameters(), **options)
    ddp._aisp_zero2_communication_evidence = evidence
    return ddp, optimizer


def training_step(
    model,
    optimizer,
    x,
    y,
    generator,
    grad_accum,
    *,
    extra_param=None,
    fixed_microbatches: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]] = None,
    autocast_enabled: Optional[bool] = None,
    post_clip_callback: Optional[Callable[[Iterable[nn.Parameter]], None]] = None,
):
    """Execute one production update and return its final scaled microbatch loss.

    Timed training supplies a generator. The post-timing correctness profile uses
    the same update body with fixed inputs, autocast disabled and a post-clip
    capture callback. Those controls are prepared outside the timed region.
    """

    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if fixed_microbatches is None:
        if generator is None:
            raise ValueError("training_step requires a generator or fixed microbatches")
    else:
        if generator is not None:
            raise ValueError("fixed microbatches and a generator are mutually exclusive")
        if len(fixed_microbatches) != grad_accum:
            raise ValueError("fixed microbatch count must equal grad_accum")
    use_autocast = x.is_cuda if autocast_enabled is None else bool(autocast_enabled)
    optimizer.zero_grad(set_to_none=True)
    for microbatch_index in range(grad_accum):
        if fixed_microbatches is None:
            x.normal_(generator=generator)
            y.normal_(generator=generator)
        else:
            next_x, next_y = fixed_microbatches[microbatch_index]
            if next_x.shape != x.shape or next_y.shape != y.shape:
                raise ValueError("fixed microbatch shape does not match training buffers")
            x.copy_(next_x)
            y.copy_(next_y)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=use_autocast):
            final_microbatch_loss = nn.functional.mse_loss(model(x), y) / grad_accum
        if extra_param is not None:
            final_microbatch_loss = final_microbatch_loss + extra_param.sum() * 0.0
        final_microbatch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if post_clip_callback is not None:
        post_clip_callback(model.parameters())
    optimizer.step()
    return final_microbatch_loss.detach()


def run_training(args, *, optimized, multi_gpu):
    from labs.train_distributed.zero2_child_protocol import (
        POST_TIMING_PROFILE_KIND,
        VERIFICATION_ONLY_PROFILE_KIND,
        required_profile_kind,
        result_protocol_requested,
        run_zero2_result_profile,
    )

    variant = "multigpu" if multi_gpu else "single"
    if args.compile:
        raise RuntimeError(
            "The verified ZeRO adapter requires eager FP32 execution; --compile is unsupported"
        )
    if args.verification_only:
        backend = str(args.verification_backend)
        if not result_protocol_requested():
            raise RuntimeError(
                "--verification-only requires the parent ZeRO child-result contract; "
                "standalone performance results are intentionally unavailable"
            )
        requested_kind = required_profile_kind()
        if requested_kind != VERIFICATION_ONLY_PROFILE_KIND:
            raise RuntimeError(
                "--verification-only cannot satisfy a post-timing ZeRO result contract"
            )
        local_rank = get("lrank")
        if backend == "nccl":
            require_min_gpus(2 if multi_gpu else 1, script_name="zero2.py verification-only")
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group("gloo")
        try:
            run_zero2_result_profile(
                optimized=optimized,
                variant=variant,
                device=device,
            )
            if dist.get_rank() == 0:
                print(
                    "ZeRO verification-only profile completed; no performance timing was collected.",
                    flush=True,
                )
        finally:
            dist.destroy_process_group()
        return

    if result_protocol_requested():
        requested_kind = required_profile_kind()
        if requested_kind != POST_TIMING_PROFILE_KIND:
            raise RuntimeError(
                "The normal ZeRO performance child requires a post-timing result contract"
            )
    require_min_gpus(2 if multi_gpu else 1, script_name="zero2.py")
    local_rank = get("lrank")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    label = "optimized-zero2" if optimized else "baseline-zero2"
    try:
        rank = get("rank")
        # The model initializes on CPU before moving to CUDA. Seed only that
        # generator; the per-rank input generator is explicit below.
        torch.default_generator.manual_seed(args.seed)
        model = build_model(args.hidden_size, device)
        extra_param = None
        if args.extra_grad_mb:
            numel = args.extra_grad_mb * 1024 * 1024 // 2
            extra_param = nn.Parameter(torch.zeros(numel, device=device, dtype=torch.bfloat16))
            model.register_parameter("extra_grad_payload", extra_param)
        ddp, optimizer = build_training_components(
            model, args.learning_rate, optimized=optimized, device_ids=[local_rank])
        generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
        x = torch.empty(args.batch_size, args.hidden_size, device=device)
        y = torch.empty_like(x)
        loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

        training_step(
            ddp,
            optimizer,
            x,
            y,
            generator,
            args.grad_accum,
            extra_param=extra_param,
            autocast_enabled=False,
        )
        if rank == 0:
            print_memory_stats(f"{label} warmup", ddp, optimizer, rank, device)
            print("Workload: seven GELU-separated linear layers; eager FP32 model execution; "
                  "AdamW; clip=1.0; one warmup update. RS/AG restores full gradients; no optimizer overlap.")
            if extra_param is not None:
                print("Optional synthetic communication payload uses BF16 and zero gradients in both variants.")
        dist.barrier()
        torch.cuda.synchronize(device)

        # Same measured region in both variants; logging is outside it.
        start = perf_counter()
        for _ in range(args.steps):
            loss = training_step(
                ddp,
                optimizer,
                x,
                y,
                generator,
                args.grad_accum,
                extra_param=extra_param,
                autocast_enabled=False,
            )
        torch.cuda.synchronize(device)
        total_time = perf_counter() - start
        if rank == 0:
            loss_value_buffer[0].copy_(loss)
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            samples = args.steps * args.grad_accum * args.batch_size
            print(f"[{label}] finished {args.steps} steps | loss(last microbatch)={loss_value:.6f} | "
                  f"{samples / total_time:,.0f} samples/s per rank | training_seconds={total_time:.6f}")
            print("Samples count batch rows, not hidden-vector elements. Harness process-wall time also includes setup/warmup.")
        if result_protocol_requested():
            run_zero2_result_profile(
                optimized=optimized,
                variant=variant,
                device=device,
            )
    finally:
        dist.destroy_process_group()
