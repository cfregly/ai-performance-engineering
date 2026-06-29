"""Optimized FSDP example with FlashAttention and optional FP8."""

from __future__ import annotations

import argparse
import os

from core.common.device_utils import resolve_local_rank
from functools import partial
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore

from core.benchmark.gpu_requirements import require_min_gpus
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


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized FSDP FP8 training")
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



def _build_dataloader(
    seq_len: int,
    micro_batch: int,
    rank: int,
    world_size: int,
    *,
    steps: int,
    grad_accum: int,
):
    fast_mode = os.getenv("AISP_FSDP_FAST") == "1"
    if fast_mode:
        vocab_size = int(os.getenv("AISP_TINYSTORIES_VOCAB", "32000"))
        num_samples = max(steps * grad_accum * micro_batch * world_size, 256)
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
        labels = input_ids.clone()

        class SyntheticTokenDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor):
                self._input_ids = input_ids
                self._labels = labels

            def __len__(self) -> int:
                return self._input_ids.size(0)

            def __getitem__(self, idx: int):
                return {"input_ids": self._input_ids[idx], "labels": self._labels[idx]}

        dataset = SyntheticTokenDataset(input_ids, labels)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader, sampler

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
        drop_last=False,
        collate_fn=create_collate_fn(),
    )
    return dataloader, sampler


def _wrap_fsdp(model: torch.nn.Module) -> FSDP:
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError as exc:
        raise RuntimeError("_wrap_fsdp() requires the `transformers` package") from exc
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=False,
    )


def _fused_adamw(params, lr, *, allow_fused: bool) -> torch.optim.Optimizer:
    if os.getenv("AISP_FSDP_FAST") == "1":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=False)
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
    require_min_gpus(2, script_name="optimized_fsdp_multigpu.py")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("optimized_fsdp_multigpu requires the `transformers` package") from exc

    args = parse_args()
    fp8_enabled = os.getenv("AISP_FSDP_DISABLE_FP8") != "1"
    if fp8_enabled:
        _assert_torchao_available()

    rank, world_size, local_rank = _init_distributed()

    dataloader, sampler = _build_dataloader(
        args.sequence_length,
        args.micro_batch_size,
        rank,
        world_size,
        steps=args.steps,
        grad_accum=args.grad_accum,
    )
    if rank == 0:
        print("[optimized_fsdp_multigpu] dataloader ready", flush=True)

    fast_mode = os.getenv("AISP_FSDP_FAST") == "1"
    if fast_mode:
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=int(os.getenv("AISP_TINYSTORIES_VOCAB", "32000")),
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=max(args.sequence_length, 2048),
            rms_norm_eps=1e-5,
            use_cache=False,
        )
    else:
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
    if getattr(config, "max_position_embeddings", 0) < args.sequence_length:
        config.max_position_embeddings = args.sequence_length
    config.gradient_checkpointing = False
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if rank == 0:
        print("[optimized_fsdp_multigpu] model instantiated", flush=True)

    model = model.to(torch.cuda.current_device(), dtype=torch.bfloat16)
    if fp8_enabled:
        fp8_recipe = Float8LinearConfig(enable_fsdp_float8_all_gather=True)
        model = convert_to_float8_training(model, config=fp8_recipe)
    elif rank == 0:
        print("[optimized_fsdp_multigpu] FP8 disabled via AISP_FSDP_DISABLE_FP8=1", flush=True)

    fsdp_model = _wrap_fsdp(model)
    if rank == 0:
        print("[optimized_fsdp_multigpu] fsdp wrapped", flush=True)
    optimizer = _fused_adamw(fsdp_model.parameters(), args.learning_rate, allow_fused=not fp8_enabled)
    if rank == 0:
        print("[optimized_fsdp_multigpu] optimizer ready", flush=True)

    flop_per_token = get_model_flops_per_token(fsdp_model.module.config, args.sequence_length)
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
                outputs = fsdp_model(**batch)
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
                    f"[optimized_fsdp_multigpu] step {optimizer_step}/{total_updates} "
                    f"loss={loss_value:.4f}" + ThroughputTracker.format(metrics, include_memory=True)
                )
                print(msg, flush=True)

            if optimizer_step >= total_updates:
                break

        epoch += 1

    dist.barrier()
    if is_main:
        print("[optimized_fsdp_multigpu] training completed", flush=True)

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    local_data_path = Path(__file__).parent / "data" / "tinystories_sample.jsonl"
    # Scale up by switching to a larger config (ex: tinyllama_config.json)
    # and matching it with a packed dataset at the desired sequence length.
    packed_data_path = Path(__file__).parent / "data" / "tinystories_packed_seq1024.jsonl"
    config_path = Path(__file__).parent / "data" / "tinyllama_config.json"
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "train_fsdp.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "multigpu",
            "--sequence-length",
            "1024",
            "--micro-batch-size",
            "2",
            "--grad-accum",
            "2",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:fsdp_multigpu",
        default_nproc_per_node=None,
        default_iterations=200,
        measurement_timeout_seconds=900,
        env={
            "AISP_TINYSTORIES_LOCAL_PATH": str(local_data_path),
            "AISP_TINYSTORIES_PACKED_PATH": str(packed_data_path),
            "AISP_TINYSTORIES_CONFIG_PATH": str(config_path),
            "AISP_TINYSTORIES_LAYERS": "12",
            "AISP_FSDP_DISABLE_FP8": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        name="optimized_fsdp_multigpu",
    )
