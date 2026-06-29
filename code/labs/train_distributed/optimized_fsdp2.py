"""Optimized FSDP2 example with FlashAttention and optional FP8 (single GPU)."""

from __future__ import annotations

import argparse
import os

from core.common.device_utils import resolve_local_rank
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.utils import (
    ThroughputTracker,
    create_collate_fn,
    get_model_flops_per_token,
    gpu_memory_usage,
    load_tinystories_packed,
    load_tinystories,
    setup_tokenizer,
)

try:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
except Exception as exc:  # pragma: no cover - torchao may be absent in some environments
    Float8LinearConfig = None  # type: ignore[assignment]
    convert_to_float8_training = None  # type: ignore[assignment]
    _TORCHAO_IMPORT_ERROR = exc
else:
    _TORCHAO_IMPORT_ERROR = None


MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
LOCAL_DATA_PATH = Path(__file__).parent / "data" / "tinystories_sample.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized FSDP2 FP8 training")
    parser.add_argument("--steps", type=int, default=100, help="Optimizer steps to run")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, int]:
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def _build_dataloader(seq_len: int, micro_batch: int, rank: int, world_size: int):
    packed_path = os.getenv("AISP_TINYSTORIES_PACKED_PATH")
    if packed_path:
        dataset = load_tinystories_packed(packed_path, seq_len, is_main_process=rank == 0)
    else:
        tokenizer = setup_tokenizer(MODEL_ID)
        dataset = load_tinystories(tokenizer, seq_len, is_main_process=rank == 0)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=create_collate_fn(),
    )
    return dataloader, sampler


def _apply_fsdp2(model: torch.nn.Module, mesh) -> torch.nn.Module:
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError as exc:
        raise RuntimeError("_apply_fsdp2() requires the `transformers` package") from exc
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
    return model


def _fused_adamw(params, lr, *, allow_fused: bool) -> torch.optim.Optimizer:
    if not allow_fused:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=False)
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def _assert_torchao_available():
    if convert_to_float8_training is None or Float8LinearConfig is None:
        raise RuntimeError(
            "torchao.float8 is not available. Install torchao with CUDA support to run the FP8 optimized demo."
        ) from _TORCHAO_IMPORT_ERROR


def main():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("optimized_fsdp2 requires the `transformers` package") from exc

    args = parse_args()
    fp8_enabled = os.getenv("AISP_FSDP_DISABLE_FP8") != "1"
    if fp8_enabled:
        _assert_torchao_available()

    rank, world_size, local_rank = _init_distributed()

    dataloader, sampler = _build_dataloader(args.sequence_length, args.micro_batch_size, rank, world_size)
    if rank == 0:
        print("[optimized_fsdp2] dataloader ready", flush=True)

    config_path = os.getenv("AISP_TINYSTORIES_CONFIG_PATH")
    if config_path:
        config = AutoConfig.from_pretrained(config_path)
        config.use_cache = False
        config.attn_implementation = "flash_attention_2"
    else:
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
    override_layers = os.getenv("AISP_TINYSTORIES_LAYERS")
    if override_layers:
        layers = int(override_layers)
        if layers < 1 or layers > config.num_hidden_layers:
            raise ValueError(
                f"AISP_TINYSTORIES_LAYERS must be between 1 and {config.num_hidden_layers}, got {layers}."
            )
        config.num_hidden_layers = layers
    config.gradient_checkpointing = False
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if rank == 0:
        print("[optimized_fsdp2] model instantiated", flush=True)

    model = model.to(torch.cuda.current_device(), dtype=torch.bfloat16)
    if fp8_enabled:
        fp8_recipe = Float8LinearConfig(enable_fsdp_float8_all_gather=True)
        model = convert_to_float8_training(model, config=fp8_recipe)
    elif rank == 0:
        print("[optimized_fsdp2] FP8 disabled via AISP_FSDP_DISABLE_FP8=1", flush=True)

    mesh = init_device_mesh("cuda", (world_size,))
    model = _apply_fsdp2(model, mesh)
    if rank == 0:
        print("[optimized_fsdp2] fsdp2 applied", flush=True)

    optimizer = _fused_adamw(model.parameters(), args.learning_rate, allow_fused=not fp8_enabled)
    if rank == 0:
        print("[optimized_fsdp2] optimizer ready", flush=True)

    flop_per_token = get_model_flops_per_token(model.config, args.sequence_length)
    tracker = ThroughputTracker(warmup_steps=10)

    total_updates = args.steps
    optimizer_step = 0
    micro_step = 0
    epoch = 0
    is_main = rank == 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=f"cuda:{local_rank}")

    while optimizer_step < total_updates:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
            with sdpa_ctx, torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            micro_step += 1

            should_step = micro_step % args.grad_accum == 0
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            metrics = tracker.step(batch["input_ids"].numel(), flop_per_token)
            if (
                metrics
                and should_step
                and is_main
                and (optimizer_step % 5 == 0 or optimizer_step == total_updates)
            ):
                metrics.update(gpu_memory_usage(local_rank))
                loss_value_buffer[0].copy_(loss.detach())
                loss_value = float(loss_value_buffer.detach().cpu()[0]) * args.grad_accum
                msg = (
                    f"[optimized_fsdp2] step {optimizer_step}/{total_updates} "
                    f"loss={loss_value:.4f}" + ThroughputTracker.format(metrics, include_memory=True)
                )
                print(msg, flush=True)

            if optimizer_step >= total_updates:
                break

        epoch += 1

    dist.barrier()
    if is_main:
        print("[optimized_fsdp2] training completed", flush=True)

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    # Scale up by switching to a larger config (ex: tinyllama_config.json)
    # and matching it with a packed dataset at the desired sequence length.
    packed_data_path = Path(__file__).parent / "data" / "tinystories_packed_seq1024.jsonl"
    config_path = Path(__file__).parent / "data" / "tinyllama_config.json"
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "train_fsdp2.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "single",
            "--sequence-length",
            "1024",
            "--micro-batch-size",
            "2",
            "--grad-accum",
            "2",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=False,
        target_label="labs/train_distributed:fsdp2",
        default_nproc_per_node=1,
        default_iterations=10,
        env={
            "AISP_TINYSTORIES_LOCAL_PATH": str(LOCAL_DATA_PATH),
            "AISP_TINYSTORIES_PACKED_PATH": str(packed_data_path),
            "AISP_TINYSTORIES_CONFIG_PATH": str(config_path),
            "AISP_TINYSTORIES_LAYERS": "8",
            "AISP_FSDP_DISABLE_FP8": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        name="optimized_fsdp2",
    )
