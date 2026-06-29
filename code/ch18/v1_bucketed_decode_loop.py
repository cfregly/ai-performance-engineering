"""V1 engine polling loop with bucketed decode workspaces and masks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch

from ch18.baseline_vllm_decode_graphs import (  # noqa: E402
    DecodeMetrics,
    export_prom_metrics,
    format_metrics,
)
from ch18.decode_kernels import DEVICE, build_decode_kernel  # noqa: E402
from ch18.optimized_vllm_decode_graphs import BUCKETS, pick_bucket  # noqa: E402
from ch18.v1_engine_loop import run_engine_loop  # noqa: E402
from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack  # noqa: E402


@dataclass
class BucketWorkspace:
    batch: int
    hidden: int
    dtype: torch.dtype
    device: str = DEVICE
    tokens_kv: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    tmp: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    initialized: bool = False

    def ensure(self) -> None:
        if self.initialized:
            return
        self.tokens_kv = torch.empty(
            (2, self.batch, self.hidden),
            device=self.device,
            dtype=self.dtype,
        )
        self.tokens = self.tokens_kv[0]
        self.kv = self.tokens_kv[1]
        self.tokens_kv.normal_(mean=0.0, std=1.0)
        self.mask = torch.ones(self.batch, dtype=torch.bool, device=self.device)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self.logits = torch.empty(
                    (self.batch, self.hidden),
                    device=self.device,
                    dtype=self.dtype,
                )
                self.tmp = torch.empty_like(self.logits)
            self.stream.synchronize()
        else:
            self.logits = torch.empty(
                (self.batch, self.hidden),
                device=self.device,
                dtype=self.dtype,
            )
            self.tmp = torch.empty_like(self.logits)
        self.initialized = True

    @property
    def bytes(self) -> int:
        if (
            self.logits is None
            or self.tmp is None
            or self.tokens is None
            or self.kv is None
            or self.mask is None
        ):
            return 0
        return (
            (self.logits.numel() + self.tmp.numel()) * self.logits.element_size()
            + self.tokens.numel() * self.tokens.element_size()
            + self.kv.numel() * self.kv.element_size()
            + self.mask.numel() * self.mask.element_size()
        )


class BucketWorkspaceRegistry:
    """Per-bucket reusable workspaces for the decode step."""

    def __init__(self, buckets: Sequence[int], hidden: int, dtype: torch.dtype) -> None:
        self.workspaces = {
            b: BucketWorkspace(batch=b, hidden=hidden, dtype=dtype) for b in buckets
        }
        self._bytes_accounted: set[int] = set()

    def get(self, bucket: int) -> BucketWorkspace:
        return self.workspaces[bucket]

    def allocator_bytes_once(self, bucket: int) -> int:
        """Count allocator bytes only the first time a bucket is touched."""
        if bucket in self._bytes_accounted:
            return 0
        self._bytes_accounted.add(bucket)
        return self.workspaces[bucket].bytes


def iter_mock_engine() -> Iterator[List[MockRequestOutput]]:
    engine_core, core_client = build_demo_stack()
    for ro in run_engine_loop(engine_core, core_client):
        yield [ro]


def iter_vllm_engine(engine: object) -> Iterator[List[object]]:
    """Yield RequestOutputs per vLLM step (when vLLM is available)."""
    while engine.has_unfinished_requests():
        outputs = engine.step()
        yield outputs


def run_bucketed_loop(
    step_iter: Iterable[List[object]],
    hidden: int,
) -> DecodeMetrics:
    decode_kernel = build_decode_kernel(hidden=hidden, max_batch=max(BUCKETS))
    metrics = DecodeMetrics()
    seen_shapes: set[Tuple[int, int]] = set()
    dtype = torch.float16 if getattr(decode_kernel, "backend", "") == "vllm" else torch.float32
    registry = BucketWorkspaceRegistry(BUCKETS, hidden=hidden, dtype=dtype)
    current_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None

    for outputs in step_iter:
        if not outputs:
            continue

        batch_size = len(outputs)
        bucket = pick_bucket(batch_size)
        ws = registry.get(bucket)
        was_initialized = ws.initialized
        ws.ensure()

        # Drive the real decode kernel on dummy tensors to exercise the graph,
        # masking inactive rows to keep bucketed shapes stable.
        if ws.tokens is None or ws.kv is None or ws.mask is None:
            raise RuntimeError("workspace not initialized")
        tokens_padded = ws.tokens
        kv_padded = ws.kv
        mask = ws.mask
        if batch_size == bucket:
            mask.fill_(True)
        else:
            mask[:batch_size].fill_(True)
            mask[batch_size:bucket].fill_(False)

        if ws.stream is not None and current_stream is not None:
            current_stream.wait_stream(ws.stream)

        logits = decode_kernel(tokens_padded, kv_padded, mask)
        shape_key = (logits.shape[0], logits.shape[1])
        if shape_key not in seen_shapes:
            metrics.graph_recaptures += 1
            seen_shapes.add(shape_key)

        metrics.allocator_bytes += registry.allocator_bytes_once(bucket) if not was_initialized else 0
        metrics.tokens += batch_size
        metrics.steps += 1

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucketed V1 engine loop with decode workspaces.")
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="If set, run a tiny vLLM loop instead of the mock demo stack (requires installed model).",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model id for vLLM.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max new tokens per request (vLLM mode).")
    parser.add_argument("--prom-port", type=int, default=None, help="Optional Prometheus port to export metrics.")
    parser.add_argument(
        "--prom-duration",
        type=int,
        default=0,
        help="If --prom-port is set, keep the server alive this many seconds (0 = fire-and-exit).",
    )
    return parser.parse_args()


def build_vllm_steps(args: argparse.Namespace) -> Iterable[List[object]]:
    try:
        from vllm import EngineArgs, LLMEngine, SamplingParams  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vLLM not available: {exc}")

    engine_args = EngineArgs(model=args.model, disable_log_stats=True)
    engine = LLMEngine.from_engine_args(engine_args)

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.8, top_p=0.95)
    prompts = [
        "Explain CUDA graphs decode pitfalls briefly.",
        "List three causes of allocator churn in decode loops.",
    ]
    for i, prompt in enumerate(prompts):
        engine.add_request(request_id=f"req-{i}", prompt=prompt, sampling_params=sp)

    return iter_vllm_engine(engine)


def main() -> None:
    args = parse_args()

    if args.use_vllm:
        try:
            step_iter = build_vllm_steps(args)
        except Exception as exc:
            print(f"[warn] falling back to mock engine: {exc}")
            step_iter = iter_mock_engine()
    else:
        step_iter = iter_mock_engine()

    metrics = run_bucketed_loop(step_iter, hidden=128)
    backend = "vllm" if args.use_vllm else "torch"
    print(format_metrics("v1_bucketed_loop", metrics, backend=backend))

    if args.prom_port is not None:
        export_prom_metrics(
            "v1_bucketed_loop", metrics, backend=backend, port=args.prom_port, duration_s=args.prom_duration
        )


if __name__ == "__main__":
    main()
