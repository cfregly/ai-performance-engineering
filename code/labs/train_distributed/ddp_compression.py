"""DDP training loop with optional gradient compression hooks."""

import argparse
import os
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.common.device_utils import resolve_local_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--compression",
        choices=["none", "int8", "powersgd"],
        default="none",
        help="Gradient compression mode.",
    )
    parser.add_argument(
        "--powersgd-rank",
        type=int,
        default=1,
        help="PowerSGD low-rank approximation rank.",
    )
    parser.add_argument(
        "--powersgd-start-iter",
        type=int,
        default=2,
        help="Iteration to start PowerSGD compression (>=2 required).",
    )
    parser.add_argument(
        "--powersgd-min-compression-rate",
        type=int,
        default=1,
        help="Minimum compression rate required to apply PowerSGD.",
    )
    parser.add_argument(
        "--powersgd-batch-same-shape",
        action="store_true",
        help="Batch tensors with same shape during PowerSGD compression.",
    )
    parser.add_argument(
        "--powersgd-disable-error-feedback",
        action="store_true",
        help="Disable error feedback in PowerSGD to reduce overhead.",
    )
    parser.add_argument(
        "--powersgd-disable-warm-start",
        action="store_true",
        help="Disable PowerSGD warm start to reduce overhead.",
    )
    parser.add_argument(
        "--extra-grad-mb",
        type=int,
        default=64,
        help="Extra gradient payload size (MB) to make communication dominant.",
    )
    parser.add_argument(
        "--bucket-cap-mb",
        type=int,
        default=25,
        help="DDP bucket size (MB) for gradient bucketing.",
    )
    parser.add_argument(
        "--disable-bucket-view",
        action="store_true",
        help="Disable gradient-as-bucket-view optimization.",
    )
    parser.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Allow single-GPU runs for the compression benchmark.",
    )
    parser.add_argument(
        "--simulate-single-gpu-comm",
        action="store_true",
        help="Simulate gradient communication on a single GPU.",
    )
    parser.add_argument(
        "--naive-allreduce",
        action="store_true",
        help="Use per-parameter all-reduce instead of DDP bucketed communication.",
    )
    parser.add_argument(
        "--naive-allreduce-sync",
        action="store_true",
        help="Synchronize after each naive all-reduce to surface sync overhead.",
    )
    parser.add_argument(
        "--int8-local-scale",
        action="store_true",
        help="Use local max for INT8 scaling (skip global max reduce).",
    )
    return parser.parse_args()


class _Int8AllReduceState:
    def __init__(self, process_group: dist.ProcessGroup | None, *, use_local_scale: bool):
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.limit = max(1, 127 // max(1, self.world_size))
        self.use_local_scale = use_local_scale


def _int8_allreduce_hook(
    state: _Int8AllReduceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    tensor = bucket.buffer()
    if state.world_size < 2:
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    local_max = tensor.abs().max()
    if not state.use_local_scale:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=state.process_group)
    # Scale to keep the int8 sum in-range across all ranks.
    scale = (local_max / float(state.limit)).to(dtype=tensor.dtype)
    scale.masked_fill_(scale == 0, 1.0)
    quant = torch.clamp((tensor / scale).round(), -state.limit, state.limit).to(torch.int8)
    dist.all_reduce(quant, op=dist.ReduceOp.SUM, group=state.process_group)
    # Keep dequant in tensor dtype to reduce temporary memory overhead.
    dequant_scale = (scale / state.world_size).to(dtype=tensor.dtype)
    dequant = quant.to(dtype=tensor.dtype).mul(dequant_scale)

    fut = torch.futures.Future()
    fut.set_result(dequant)
    return fut


def main():
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD
    from torch.nn.parallel import DistributedDataParallel as DDP

    from labs.train_distributed.training_utils.utils import (
        build_dataloader,
        build_text_model,
        build_tokenizer,
        get_dataset,
    )

    expected_torch_seed = torch.initial_seed()
    expected_cuda_seed = torch.cuda.initial_seed() if torch.cuda.is_available() else None

    args = parse_args()
    local_rank = resolve_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", device_id=local_rank)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2 and not args.allow_single_gpu:
        raise RuntimeError("SKIPPED: requires >=2 GPUs for gradient compression")

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    class _SyntheticCausalLM(torch.nn.Module):
        def __init__(self, vocab_size: int = 32000, hidden_size: int = 1024):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_kwargs):
            x = self.embed(input_ids)
            logits = self.proj(x)
            if labels is None:
                labels = input_ids
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return SimpleNamespace(loss=loss, logits=logits)

    def _build_synthetic_dataloader(
        *,
        batch_size: int,
        steps: int,
        seq_len: int = 128,
        vocab_size: int = 32000,
        distributed: bool = False,
    ) -> torch.utils.data.DataLoader:
        sample_count = max(batch_size * max(steps, 1) * 2, batch_size * 64)
        input_ids = torch.randint(0, vocab_size, (sample_count, seq_len), dtype=torch.int64)
        attention_mask = torch.ones((sample_count, seq_len), dtype=torch.int64)

        class _SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return sample_count

            def __getitem__(self, idx: int):
                return {
                    "input_ids": input_ids[idx],
                    "attention_mask": attention_mask[idx],
                }

        dataset = _SyntheticDataset()
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            if distributed
            else None
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    distributed = dist.is_initialized() and dist.get_world_size() > 1

    try:
        tokenizer = build_tokenizer()
        dataset = get_dataset()["train"]
        dataloader = build_dataloader(
            dataset,
            tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            distributed=distributed,
            num_workers=2,
            prefetch_factor=2,
        )
        model = build_text_model()
    except Exception as exc:
        if is_main:
            print(
                "[ddp-compression] WARNING: using synthetic local model/dataset fallback "
                f"because Hugging Face assets were unavailable ({exc}).",
                flush=True,
            )
        dataloader = _build_synthetic_dataloader(
            batch_size=args.batch_size,
            steps=args.steps,
            distributed=distributed,
        )
        model = _SyntheticCausalLM()

    model.to(device)
    model.train()

    extra_param = None
    if args.extra_grad_mb > 0:
        elem_bytes = torch.finfo(torch.bfloat16).bits // 8
        numel = (args.extra_grad_mb * 1024 * 1024) // elem_bytes
        rows = min(4096, max(1, numel))
        cols = max(1, numel // rows)
        numel = rows * cols
        extra_param = torch.nn.Parameter(
            torch.zeros((rows, cols), device=device, dtype=torch.bfloat16)
        )
        model.register_parameter("extra_grad_payload", extra_param)

    comm_buffer = None
    if args.simulate_single_gpu_comm and world_size < 2 and args.extra_grad_mb > 0:
        elem_bytes = torch.finfo(torch.float32).bits // 8
        numel = (args.extra_grad_mb * 1024 * 1024) // elem_bytes
        comm_buffer = torch.randn(numel, device=device, dtype=torch.float32)

    def _empty_cpu_staging(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        pin_memory = device.type == "cuda" and torch.cuda.is_available()
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=pin_memory)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=dtype)

    comm_staging = {}
    if comm_buffer is not None:
        if args.compression == "none":
            comm_staging["cpu_float"] = _empty_cpu_staging(comm_buffer.shape, comm_buffer.dtype)
        elif args.compression == "int8":
            comm_staging["work"] = torch.empty_like(comm_buffer)
            comm_staging["quant"] = torch.empty(comm_buffer.shape, device=device, dtype=torch.int8)
            comm_staging["cpu_quant"] = _empty_cpu_staging(comm_buffer.shape, torch.int8)
        elif args.compression == "powersgd":
            stride = max(1, 8 // max(1, args.powersgd_rank))
            sample_numel = comm_buffer.view(-1)[::stride].numel()
            comm_staging["powersgd_stride"] = stride
            comm_staging["sample"] = torch.empty(sample_numel, device=device, dtype=comm_buffer.dtype)
            comm_staging["cpu_sample"] = _empty_cpu_staging(
                torch.Size((sample_numel,)), comm_buffer.dtype
            )

    def _simulate_single_gpu_comm(buffer: torch.Tensor) -> None:
        if args.compression == "none":
            cpu_buf = comm_staging["cpu_float"]
            cpu_buf.copy_(buffer, non_blocking=False)
            buffer.copy_(cpu_buf, non_blocking=False)
        elif args.compression == "int8":
            work = comm_staging["work"]
            quant = comm_staging["quant"]
            cpu_quant = comm_staging["cpu_quant"]
            scale = buffer.abs().max().clamp_min(1e-12).div(127.0)
            torch.div(buffer, scale, out=work)
            work.round_().clamp_(-127, 127)
            quant.copy_(work)
            cpu_quant.copy_(quant, non_blocking=False)
            quant.copy_(cpu_quant, non_blocking=False)
            buffer.copy_(quant)
            buffer.mul_(scale)
        elif args.compression == "powersgd":
            flat = buffer.view(-1)
            stride = comm_staging["powersgd_stride"]
            sample = comm_staging["sample"]
            cpu_sample = comm_staging["cpu_sample"]
            sample_view = flat[::stride]
            sample.copy_(sample_view)
            cpu_sample.copy_(sample, non_blocking=False)
            sample.copy_(cpu_sample, non_blocking=False)
            sample_view.copy_(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if args.naive_allreduce and dist.is_initialized() and world_size < 2:
        raise RuntimeError("--naive-allreduce requires >=2 GPUs")

    if args.naive_allreduce:
        ddp_model = model
    else:
        ddp_model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=not args.disable_bucket_view,
            find_unused_parameters=False,
            bucket_cap_mb=args.bucket_cap_mb,
        )

        if args.compression == "int8":
            state = _Int8AllReduceState(
                dist.group.WORLD if dist.is_initialized() else None,
                use_local_scale=args.int8_local_scale,
            )
            ddp_model.register_comm_hook(state, _int8_allreduce_hook)
        elif args.compression == "powersgd":
            state = powerSGD.PowerSGDState(
                process_group=dist.group.WORLD,
                matrix_approximation_rank=args.powersgd_rank,
                start_powerSGD_iter=args.powersgd_start_iter,
                min_compression_rate=args.powersgd_min_compression_rate,
                use_error_feedback=not args.powersgd_disable_error_feedback,
                warm_start=not args.powersgd_disable_warm_start,
                batch_tensors_with_same_shape=args.powersgd_batch_same_shape,
            )
            ddp_model.register_comm_hook(state, powerSGD.powerSGD_hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)

    num_steps = min(args.steps, len(dataloader))
    start = perf_counter()
    total_tokens = 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        outputs = ddp_model(**batch)
        loss = outputs.loss
        if extra_param is not None:
            loss = loss + extra_param.sum() * 0.0
        loss.backward()
        if args.naive_allreduce and dist.is_initialized() and world_size > 1:
            for param in ddp_model.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
                if args.naive_allreduce_sync and device.type == "cuda":
                    torch.cuda.synchronize(device)
        if comm_buffer is not None:
            _simulate_single_gpu_comm(comm_buffer)
        optimizer.step()

        total_tokens += batch["input_ids"].numel()

        if is_main and step % 10 == 0:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[ddp-compression:{args.compression}] step {step}/{num_steps} | loss={loss_value:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if is_main:
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[ddp-compression:{args.compression}] finished {num_steps} steps in {elapsed:.1f}s "
            f"({toks_per_sec:,.0f} toks/s per rank)"
        )

    if dist.is_initialized():
        dist.destroy_process_group()

    if args.compression == "powersgd":
        torch.manual_seed(expected_torch_seed)
        if expected_cuda_seed is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(expected_cuda_seed)


if __name__ == "__main__":
    main()
