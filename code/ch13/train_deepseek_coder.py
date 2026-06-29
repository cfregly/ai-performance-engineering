from __future__ import annotations

"""Profiling helper for DeepSeek Coder training (Chapter 13).

Demonstrates DeepSeek architecture training with:
- DeepSeek Coder 6.7B model (real DeepSeek architecture)
- Warmup loop outside the profiler
- AMP/fused optimizer for B200 performance
- CUDA Graph capture compatible data
- Proper profiling workflow for large models

Note: Uses DeepSeek Coder 6.7B (manageable size for single GPU)
For full DeepSeek-V3, see multi-GPU examples in extras/ch13/fsdp_example.py
"""

import os

import heapq
import json
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from core.common.device_utils import get_preferred_device
from ch13.te_runtime_common import ensure_te_runtime_initialized


def _select_model() -> tuple[str, int, int]:
    """Choose an appropriate model/batch based on environment."""
    override = os.environ.get("DEEPSEEK_CODER_MODEL")
    quick_mode = (
        os.environ.get("QUICK_PROFILE") == "1"
        or os.environ.get("RUN_ALL_CHAPTERS") == "1"
        or os.environ.get("BENCHMARK_QUICK") == "1"
        or os.environ.get("SKIP_HEAVY_MODELS") == "1"
    )
    if override:
        return override, BATCH, PROFILE_STEPS
    if quick_mode:
        # Tiny GPT-2 keeps automation runs fast while exercising the full path.
        return "sshleifer/tiny-gpt2", 1, 2
    return MODEL_NAME, BATCH, PROFILE_STEPS

# Using real DeepSeek Coder model (6.7B parameters)
# This is a real DeepSeek architecture, not GPT-2!
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
BATCH = 2
WARMUP = 2
PROFILE_STEPS = 3


def _external_torch_profiler_active() -> bool:
    """Return True when an outer torch.profiler session is already active."""
    is_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if not callable(is_enabled):
        return False
    try:
        return bool(is_enabled())
    except Exception:
        return False


def _event_time_us(evt: object, *attrs: str) -> float:
    """Return the first profiler time attribute exposed by this PyTorch build."""
    for attr in attrs:
        value = getattr(evt, attr, None)
        if value is None:
            continue
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _build_top_op_summary(prof: Any, row_limit: int = 10) -> dict[str, object]:
    """Build a stable top-op summary across profiler API variations."""
    events = [
        evt
        for evt in prof.key_averages()
        if not str(getattr(evt, "key", "")).startswith("ProfilerStep")
    ]
    if not events:
        raise RuntimeError("torch.profiler returned no aggregated operator events to summarize.")

    has_device_time = any(
        _event_time_us(evt, "self_device_time_total", "self_cuda_time_total") > 0.0
        for evt in events
    )
    if not has_device_time:
        raise RuntimeError(
            "torch.profiler did not expose per-op self device totals; "
            "cannot build a CUDA top-op summary on this build."
        )

    metric_key = "self_device_time_total_us"
    metric_label = "self CUDA/device time"

    def _sort_value(evt: object) -> float:
        return _event_time_us(evt, "self_device_time_total", "self_cuda_time_total")

    top_ops = heapq.nlargest(row_limit, events, key=_sort_value)
    rows = []
    for evt in top_ops:
        rows.append(
            {
                "name": getattr(evt, "key", ""),
                "count": int(getattr(evt, "count", 0) or 0),
                "self_device_time_total_us": _event_time_us(
                    evt, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_time_total_us": _event_time_us(
                    evt, "device_time_total", "cuda_time_total"
                ),
                "self_cpu_time_total_us": _event_time_us(evt, "self_cpu_time_total"),
                "cpu_time_total_us": _event_time_us(evt, "cpu_time_total"),
            }
        )

    return {
        "device_times_available": True,
        "metric_key": metric_key,
        "metric_label": metric_label,
        "top_ops": rows,
        "total_events": len(events),
    }


def _format_top_ops_report(summary: dict[str, object]) -> str:
    metric_key = str(summary["metric_key"])
    metric_label = str(summary["metric_label"])
    top_ops = summary.get("top_ops", [])

    lines = [f"Top operations by {metric_label}"]
    lines.append(f"{'Operation':40s} {'Time':>10s} {'Calls':>6s}")
    for row in top_ops:
        row_dict = dict(row)
        time_ms = float(row_dict.get(metric_key, 0.0) or 0.0) / 1000.0
        lines.append(
            f"{str(row_dict.get('name', ''))[:40]:40s} {time_ms:10.2f} {int(row_dict.get('count', 0) or 0):6d}"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_te_runtime_initialized()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "transformers and its optional runtime dependencies must import cleanly "
            "to run ch13.train_deepseek_coder"
        ) from exc

    device, cuda_err = get_preferred_device()
    if cuda_err:
        raise RuntimeError(
            "CUDA must be available to run ch13.train_deepseek_coder "
            f"and collect a CUDA top-op summary: {cuda_err}"
        )
    if device.type != "cuda":
        raise RuntimeError(
            "ch13.train_deepseek_coder requires a CUDA device to collect a CUDA "
            "top-op summary."
        )
    model_name, batch_size, profile_steps = _select_model()
    if model_name != MODEL_NAME:
        print(
            f"Using lightweight model '{model_name}' "
            f"(batch_size={batch_size}, profile_steps={profile_steps}) for quick profiling."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"Model load for '{model_name}' ran out of memory; choose a smaller "
                "model explicitly via DEEPSEEK_CODER_MODEL or enable QUICK_PROFILE=1."
            ) from exc
        raise

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")

    texts = ["DeepSeek Coder is optimized for code generation." for _ in range(batch_size)]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].clone()

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    def run_train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(**batch, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.train()
    warmup_steps = min(WARMUP, profile_steps + 1)
    for _ in range(warmup_steps):
        run_train_step()

    if _external_torch_profiler_active():
        for _ in range(profile_steps):
            run_train_step()
        print("Outer torch.profiler session detected; using harness-owned profiler capture.")
        return

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Prime the first CUDA profiler session. On this torch/Kineto stack it
    # initializes tracing but does not populate per-op device totals.
    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
    ) as primer:
        run_train_step()
        primer.step()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
    ) as prof:
        run_train_step()
        prof.step()

    prof.export_chrome_trace("deepseek_coder_trace.json")
    summary = _build_top_op_summary(prof, row_limit=10)
    print(_format_top_ops_report(summary))

    hta_dir = "hta_traces"
    os.makedirs(hta_dir, exist_ok=True)
    with open(os.path.join(hta_dir, "rank_0.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
