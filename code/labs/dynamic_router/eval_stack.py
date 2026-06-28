"""Cheap eval stack that pairs user-visible quality with MoE/router telemetry.

This is intentionally lightweight so you can drop it into the dynamic router lab
and replace the mock generators with real engine hooks when you have vLLM or
TensorRT-LLM running. This is a lab tool (not a comparable benchmark target)
that emits a consistent artifact layout:

/artifacts/dynamic_router/cheap_eval/<run_id>/
  quality.jsonl       # per-item exact-match results
  latency.jsonl       # TTFT + decode latency samples
  tps_goodput.json    # throughput + goodput summary
  moe_router.jsonl    # router entropy / margin / drops per window
  moe_traffic.jsonl   # per-expert histogram snapshots
  sys_meta.json       # config for the run
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from labs.common.model_fetcher import ensure_gpt_oss_20b

try:  # Optional import here; runtime will error if vLLM is actually needed and missing.
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover - optional
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return _percentile_from_ordered(sorted(values), pct)


def _percentile_from_ordered(ordered: Sequence[float], pct: float) -> float:
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[int(lo)] * (hi - k) + ordered[int(hi)] * (k - lo)


def _imbalance_cv(expert_hist: Sequence[int]) -> float:
    if not expert_hist:
        return 0.0
    m = mean(expert_hist)
    if m == 0:
        return 0.0
    return pstdev(expert_hist) / m


def _dirichlet(rng: random.Random, alpha: float, k: int) -> List[float]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    if total == 0:
        return [1.0 / k] * k
    return [d / total for d in draws]


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": _percentile_from_ordered(ordered, 50),
        "p95": _percentile_from_ordered(ordered, 95),
    }


def _summarize_quality_rows(rows: Sequence[Dict]) -> Dict:
    """Aggregate per-task accuracy from already-evaluated rows."""
    per_task_acc: Dict[str, List[int]] = {}
    for row in rows:
        task = row.get("task", "unknown")
        correct = 1 if row.get("correct") else 0
        per_task_acc.setdefault(task, []).append(correct)
    per_task_summary = {
        t: (sum(v) / len(v) if v else 0.0) for t, v in per_task_acc.items()
    }
    avg_acc = (
        sum(per_task_summary.values()) / len(per_task_summary) if per_task_summary else 0.0
    )
    return {"per_task": per_task_summary, "avg_accuracy": avg_acc}


def _summarize_moe(
    moe_router_rows: Sequence[Dict],
    moe_traffic_rows: Sequence[Dict],
    *,
    experts: int,
) -> Dict[str, float]:
    """Summarize MoE health from real or simulated rows."""
    if moe_traffic_rows:
        last_hist = moe_traffic_rows[-1].get("expert_hist", [0] * experts)
        imbalance = moe_traffic_rows[-1].get("imbalance_cv", _imbalance_cv(last_hist))
    else:
        last_hist = [0] * experts
        imbalance = 0.0

    # Drops may be per-token or per-window; support both.
    dropped_tokens = 0.0
    total_tokens = 0.0
    entropy_samples: List[float] = []
    margin_samples: List[float] = []
    for row in moe_router_rows:
        dropped_tokens += float(row.get("drops", 0))
        total_tokens += float(row.get("total_tokens", 0.0))
        if "entropy" in row:
            entropy_samples.append(float(row["entropy"]))
        if "margin" in row:
            margin_samples.append(float(row.get("margin", 0.0)))
    if total_tokens <= 0 and moe_router_rows:
        total_tokens = float(len(moe_router_rows))

    drop_rate = (dropped_tokens / total_tokens) if total_tokens > 0 else 0.0
    return {
        "token_drop_rate": drop_rate,
        "imbalance_cv": imbalance,
        "entropy": mean(entropy_samples) if entropy_samples else 0.0,
        "gate_margin": mean(margin_samples) if margin_samples else 0.0,
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

QUALITY_TASKS: Dict[str, List[Tuple[str, str]]] = {
    "mmlu-mini": [
        ("What is the capital of France?", "paris"),
        ("Who wrote the novel 1984?", "george orwell"),
        ("Plants take in which gas during photosynthesis?", "carbon dioxide"),
        ("What is the largest planet in our solar system?", "jupiter"),
    ],
    "gsm8k-lite": [
        ("If 3 apples cost $6, how much do 5 apples cost?", "10"),
        ("A box has 12 red and 8 blue marbles. How many marbles total?", "20"),
        ("You need 120 pages read over 4 days. How many per day?", "30"),
        ("Sam has 15 stickers and buys 5 more packs of 4. How many now?", "35"),
    ],
    "truthfulqa-lite": [
        ("Can humans breathe underwater without equipment?", "no"),
        ("Does the sun orbit the earth?", "no"),
        ("Is water made of hydrogen and oxygen?", "yes"),
        ("Do vaccines contain microchips?", "no"),
    ],
    "domain-slice": [
        ("Return the HTTP status code for success.", "200"),
        ("Name one GPU interconnect used for KV transfer.", "nvlink"),
        ("What does TTFT stand for?", "time to first token"),
        ("What is a healthy MoE token-drop rate target under load?", "0"),
    ],
}


@dataclass
class EvalConfig:
    run_root: Path = Path("artifacts/dynamic_router/cheap_eval")
    seed: int = 0
    request_count: int = 48
    experts: int = 8
    top_k: int = 2
    latency_slo_ms: int = 4200
    ttft_slo_ms: int = 1100
    model_path: str = "gpt-oss-20b/original"
    use_vllm: bool = True
    tensor_parallel_size: Optional[int] = None
    max_gen_tokens: int = 48
    temperature: float = 0.0
    top_p: float = 0.9
    metrics_dir: Optional[Path] = None  # If set, load real metrics instead of mocks.
    allow_missing_metrics: bool = False
    baseline_scorecard: Optional[Path] = None

    # Flags consumed via aisp bench target-extra-arg; no env parsing.
    @staticmethod
    def from_flags(argv: Sequence[str], *, seed: int = 0) -> "EvalConfig":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--metrics-dir", type=Path, default=None)
        parser.add_argument("--baseline-scorecard", type=Path, default=None)
        parser.add_argument("--allow-missing-metrics", action="store_true")
        parser.add_argument("--model-path", type=str, default="gpt-oss-20b/original")
        parser.add_argument("--use-vllm", dest="use_vllm", action="store_true", default=True)
        parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
        parser.add_argument("--request-count", type=int, default=48)
        parser.add_argument("--experts", type=int, default=8)
        parser.add_argument("--top-k", type=int, default=2)
        parser.add_argument("--ttft-slo-ms", type=int, default=None)
        parser.add_argument("--latency-slo-ms", type=int, default=None)
        parser.add_argument("--max-gen-tokens", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--top-p", type=float, default=None)
        parser.add_argument("--seed", type=int, default=seed)

        args, _ = parser.parse_known_args(list(argv))

        return EvalConfig(
            run_root=Path("artifacts/dynamic_router/cheap_eval"),
            seed=args.seed,
            request_count=args.request_count or 48,
            experts=args.experts or 8,
            top_k=args.top_k or 2,
            latency_slo_ms=args.latency_slo_ms or 4200,
            ttft_slo_ms=args.ttft_slo_ms or 1100,
            model_path=args.model_path,
            use_vllm=args.use_vllm,
            tensor_parallel_size=None,
            max_gen_tokens=args.max_gen_tokens or 48,
            temperature=args.temperature if args.temperature is not None else 0.0,
            top_p=args.top_p if args.top_p is not None else 0.9,
            metrics_dir=args.metrics_dir,
            allow_missing_metrics=args.allow_missing_metrics,
            baseline_scorecard=args.baseline_scorecard,
        )


@dataclass
class LoadedMetrics:
    quality_rows: List[Dict]
    latency_rows: List[Dict]
    moe_router_rows: List[Dict]
    moe_traffic_rows: List[Dict]
    throughput_summary: Optional[Dict]


class CheapEvalStack:
    """Generates the six cheap checks and writes them to disk."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._llm: Optional["LLM"] = None
        self._llm_available: bool = False
        if cfg.metrics_dir is None:  # When replaying real metrics we don't need to spin up an LLM.
            self._init_llm_if_possible()

    def _load_real_metrics(self) -> Optional[LoadedMetrics]:
        """Load telemetry emitted by a real engine run."""
        metrics_dir = self.cfg.metrics_dir
        if metrics_dir is None:
            return None

        metrics_dir = Path(metrics_dir)
        quality_rows = _read_jsonl(metrics_dir / "quality.jsonl")
        latency_rows = _read_jsonl(metrics_dir / "latency.jsonl")
        moe_router_rows = _read_jsonl(metrics_dir / "moe_router.jsonl")
        moe_traffic_rows = _read_jsonl(metrics_dir / "moe_traffic.jsonl")
        throughput_summary = _read_json(metrics_dir / "tps_goodput.json")

        def _assert_present(name: str, rows: Sequence[Dict]) -> None:
            if not rows and not self.cfg.allow_missing_metrics:
                raise FileNotFoundError(
                    f"Expected {name} under {metrics_dir} for real telemetry. "
                    "Set EVAL_STACK_ALLOW_MISSING=1 to fall back to synthetic rows."
                )

        _assert_present("quality.jsonl", quality_rows)
        _assert_present("latency.jsonl", latency_rows)
        _assert_present("moe_router.jsonl", moe_router_rows)
        _assert_present("moe_traffic.jsonl", moe_traffic_rows)

        return LoadedMetrics(
            quality_rows=quality_rows,
            latency_rows=latency_rows,
            moe_router_rows=moe_router_rows,
            moe_traffic_rows=moe_traffic_rows,
            throughput_summary=throughput_summary or None,
        )

    def run(self) -> Dict[str, float]:
        run_id = f"full_{int(time.time())}"
        run_dir = self.cfg.run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        optimized = True

        loaded = self._load_real_metrics()
        if loaded:
            quality_rows = loaded.quality_rows
            latency_rows = loaded.latency_rows
            moe_router_rows = loaded.moe_router_rows
            moe_traffic_rows = loaded.moe_traffic_rows

            if not quality_rows and self.cfg.allow_missing_metrics:
                quality_rows, quality_summary = self._run_quality(optimized)
            else:
                quality_summary = _summarize_quality_rows(quality_rows)

            if not latency_rows and self.cfg.allow_missing_metrics:
                latency_rows = self._simulate_latency(optimized)

            if not moe_router_rows and not moe_traffic_rows and self.cfg.allow_missing_metrics:
                moe_router_rows, moe_traffic_rows, moe_summary = self._simulate_moe(optimized)
            else:
                moe_summary = _summarize_moe(
                    moe_router_rows,
                    moe_traffic_rows,
                    experts=self.cfg.experts,
                )

            throughput_summary = loaded.throughput_summary or self._compute_throughput(
                latency_rows, moe_summary["token_drop_rate"]
            )
        else:
            quality_rows, quality_summary = self._run_quality(optimized)
            latency_rows = self._simulate_latency(optimized)
            moe_router_rows, moe_traffic_rows, moe_summary = self._simulate_moe(optimized)
            throughput_summary = self._compute_throughput(latency_rows, moe_summary["token_drop_rate"])

        scorecard = self._build_scorecard(
            quality_summary=quality_summary,
            latency_rows=latency_rows,
            moe_summary=moe_summary,
            throughput_summary=throughput_summary,
        )
        scorecard = self._attach_baseline(scorecard)

        _write_jsonl(run_dir / "quality.jsonl", quality_rows)
        _write_jsonl(run_dir / "latency.jsonl", latency_rows)
        _write_jsonl(run_dir / "moe_router.jsonl", moe_router_rows)
        _write_jsonl(run_dir / "moe_traffic.jsonl", moe_traffic_rows)

        (run_dir / "tps_goodput.json").write_text(json.dumps(throughput_summary, indent=2))
        (run_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
        sys_meta = {
            "mode": "full",
            "seed": self.cfg.seed,
            "experts": self.cfg.experts,
            "top_k": self.cfg.top_k,
            "request_count": self.cfg.request_count,
            "latency_slo_ms": self.cfg.latency_slo_ms,
            "ttft_slo_ms": self.cfg.ttft_slo_ms,
            "run_dir": str(run_dir),
            "model_path": self.cfg.model_path,
            "used_vllm": bool(self._llm_available and self.cfg.use_vllm),
        }
        (run_dir / "sys_meta.json").write_text(json.dumps(sys_meta, indent=2))

        summary = {
            "run_dir": str(run_dir),
            "accuracy_overall": quality_summary["avg_accuracy"],
            "accuracy_mmlu": quality_summary["per_task"].get("mmlu-mini", 0.0),
            "accuracy_math": quality_summary["per_task"].get("gsm8k-lite", 0.0),
            "accuracy_truthful": quality_summary["per_task"].get("truthfulqa-lite", 0.0),
            "ttft_p50_ms": _percentile([r["ttft_ms"] for r in latency_rows], 50),
            "ttft_p95_ms": _percentile([r["ttft_ms"] for r in latency_rows], 95),
            "decode_p50_ms": _percentile([r["decode_ms"] for r in latency_rows], 50),
            "decode_p95_ms": _percentile([r["decode_ms"] for r in latency_rows], 95),
            "token_drop_rate": moe_summary["token_drop_rate"],
            "expert_imbalance_cv": moe_summary["imbalance_cv"],
            "router_entropy": moe_summary["entropy"],
            "gate_margin": moe_summary["gate_margin"],
            "goodput": throughput_summary["goodput"],
            "throughput_tps": throughput_summary["throughput_tps"],
            "scorecard_path": str(run_dir / "scorecard.json"),
            "baseline_scorecard": str(self.cfg.baseline_scorecard) if self.cfg.baseline_scorecard else "",
        }
        return summary

    # ------------------------------------------------------------------ Quality
    def _run_quality(self, optimized: bool) -> Tuple[List[Dict], Dict]:
        rows: List[Dict] = []
        per_task_acc: Dict[str, List[int]] = {k: [] for k in QUALITY_TASKS}
        base_acc = 0.62 if not optimized else 0.74
        consistency_bonus = 0.0 if not optimized else 0.03

        if self._llm_available and self.cfg.use_vllm:
            rows, per_task_acc = self._run_quality_with_llm()
        else:
            # Synthetic fallback when vLLM is unavailable in the current environment.
            for task, qas in QUALITY_TASKS.items():
                for _, expected in qas:
                    correct = self._rng.random() < (base_acc + consistency_bonus)
                    per_task_acc[task].append(1 if correct else 0)
                    rows.append(
                        {
                            "task": task,
                            "prompt": "synthetic",
                            "expected": expected,
                            "prediction": expected if correct else f"{expected}_wrong",
                            "correct": correct,
                        }
                    )

        per_task_summary = {t: (sum(v) / len(v) if v else 0.0) for t, v in per_task_acc.items()}
        avg_acc = (
            sum(per_task_summary.values()) / len(per_task_summary) if per_task_summary else 0.0
        )
        return rows, {"per_task": per_task_summary, "avg_accuracy": avg_acc}

    def _run_quality_with_llm(self) -> Tuple[List[Dict], Dict[str, List[int]]]:
        if self._llm is None:
            return [], {}

        prompts: List[Tuple[str, str, str]] = []  # (task, prompt, expected)
        for task, qas in QUALITY_TASKS.items():
            for prompt, answer in qas:
                prompts.append((task, prompt, answer))

        sampling = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_gen_tokens,
        )
        outputs = self._llm.generate(
            [p[1] for p in prompts],
            sampling_params=sampling,
        )

        rows: List[Dict] = []
        per_task_acc: Dict[str, List[int]] = {k: [] for k in QUALITY_TASKS}
        for meta, out in zip(prompts, outputs):
            task, prompt, expected = meta
            completion = out.outputs[0].text if out.outputs else ""
            normalized_pred = self._normalize_text(completion)
            normalized_gt = self._normalize_text(expected)
            correct = normalized_pred.startswith(normalized_gt) or normalized_gt in normalized_pred
            rows.append(
                {
                    "task": task,
                    "prompt": prompt,
                    "expected": expected,
                    "prediction": completion.strip(),
                    "correct": correct,
                    "source": "vllm",
                }
            )
            per_task_acc[task].append(int(correct))
        return rows, per_task_acc

    # ----------------------------------------------------------------- Latency
    def _simulate_latency(self, optimized: bool) -> List[Dict]:
        rows: List[Dict] = []
        rng = self._rng
        cold_bias = 1.15
        ttft_base = 420.0 if not optimized else 360.0
        decode_base = 850.0 if not optimized else 720.0
        output_choices = [128, 512, 2048]

        for i in range(self.cfg.request_count):
            warm = (i % 4) != 0  # every fourth request is a cold cache
            ttft = rng.gauss(ttft_base, ttft_base * 0.18)
            output_tokens = rng.choice(output_choices)  # type: ignore[attr-defined]
            decode_scale = math.sqrt(output_tokens / 256.0)
            decode = rng.gauss(decode_base * decode_scale, decode_base * 0.22 * decode_scale)
            if not warm:
                ttft *= cold_bias
            prompt_tokens = rng.randint(64, 1024)
            rows.append(
                {
                    "request_id": f"req-{i}",
                    "ttft_ms": max(ttft, 50.0),
                    "decode_ms": max(decode, 120.0),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "case": "warm" if warm else "cold",
                }
            )
        return rows

    # --------------------------------------------------------------------- MoE
    def _simulate_moe(self, optimized: bool) -> Tuple[List[Dict], List[Dict], Dict]:
        router_rows: List[Dict] = []
        traffic_rows: List[Dict] = []

        rng = self._rng
        experts = self.cfg.experts
        top_k = self.cfg.top_k
        expert_ids = range(experts)
        expert_hist = [0 for _ in expert_ids]
        entropy_samples: List[float] = []
        margin_samples: List[float] = []

        drop_chance = 0.0065 if not optimized else 0.0008
        imbalance_tilt = 0.35 if not optimized else 0.15
        dirichlet_alpha = 0.85 if optimized else 0.55

        total_tokens = 0
        dropped_tokens = 0
        window = 0
        tokens_per_window = 64
        total_steps = max(self.cfg.request_count * 20, tokens_per_window)

        for step in range(total_steps):
            probs = _dirichlet(
                rng,
                alpha=dirichlet_alpha,
                k=experts,
            )
            ranked_experts = sorted(expert_ids, key=probs.__getitem__, reverse=True)
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
            best_prob = probs[ranked_experts[0]]
            second_prob = probs[ranked_experts[1]] if experts > 1 else 0.0
            margin = best_prob - second_prob

            token_experts = ranked_experts[:top_k]
            for e in token_experts:
                expert_hist[e] += 1

            imbalance = _imbalance_cv(expert_hist)
            adjusted_drop = drop_chance * (1.0 + imbalance_tilt * imbalance)
            roll = rng.random()
            if optimized:
                roll *= 0.5  # optimized routing should hit capacity limits less often
            drop_event = roll < adjusted_drop

            entropy_samples.append(entropy)
            margin_samples.append(margin)
            total_tokens += 1
            dropped_tokens += 1 if drop_event else 0

            if (step + 1) % tokens_per_window == 0:
                router_rows.append(
                    {
                        "step": step,
                        "entropy": entropy,
                        "margin": margin,
                        "drops": int(drop_event),
                    }
                )
                traffic_rows.append(
                    {
                        "window": window,
                        "expert_hist": expert_hist.copy(),
                        "imbalance_cv": _imbalance_cv(expert_hist),
                    }
                )
                window += 1

        drop_rate = dropped_tokens / float(total_tokens or 1)
        summary = {
            "token_drop_rate": drop_rate,
            "imbalance_cv": _imbalance_cv(expert_hist),
            "entropy": mean(entropy_samples) if entropy_samples else 0.0,
            "gate_margin": mean(margin_samples) if margin_samples else 0.0,
        }
        return router_rows, traffic_rows, summary

    # -------------------------------------------------------------- Throughput
    def _compute_throughput(self, latency_rows: List[Dict], drop_rate: float) -> Dict[str, float]:
        total_tokens = sum(r["output_tokens"] for r in latency_rows)
        total_time_s = sum((r["ttft_ms"] + r["decode_ms"]) for r in latency_rows) / 1000.0
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        good_rows = [
            r
            for r in latency_rows
            if r["ttft_ms"] <= self.cfg.ttft_slo_ms and r["decode_ms"] <= self.cfg.latency_slo_ms
        ]
        good_tokens = sum(r["output_tokens"] for r in good_rows)
        goodput = good_tokens / total_time_s if total_time_s > 0 else 0.0
        # Penalize goodput when drop rate is high to reflect MoE instability.
        goodput *= max(0.0, 1.0 - drop_rate * 5.0)

        return {
            "throughput_tps": throughput,
            "goodput": goodput,
            "slo_hit_rate": (len(good_rows) / len(latency_rows)) if latency_rows else 0.0,
            "drop_rate": drop_rate,
        }

    # -------------------------------------------------------------- Scorecard
    def _build_scorecard(
        self,
        *,
        quality_summary: Dict,
        latency_rows: List[Dict],
        moe_summary: Dict,
        throughput_summary: Dict,
    ) -> Dict[str, Dict]:
        ttft_vals = [r["ttft_ms"] for r in latency_rows]
        decode_vals = [r["decode_ms"] for r in latency_rows]
        warm_rows = [r for r in latency_rows if r["case"] == "warm"]
        cold_rows = [r for r in latency_rows if r["case"] == "cold"]

        def _decode_bucket(target: int) -> Dict[str, float]:
            bucket = [r["decode_ms"] for r in latency_rows if r["output_tokens"] == target]
            return _percentiles(bucket) if bucket else {"p50": 0.0, "p95": 0.0}

        decode_by_len = {
            "128": _decode_bucket(128),
            "512": _decode_bucket(512),
            "2048": _decode_bucket(2048),
        }

        return {
            "quality": {
                "avg_accuracy": quality_summary["avg_accuracy"],
                "per_task": quality_summary["per_task"],
                "notes": "Use 200-500 item slice; watch delta vs baseline big run (-0.5 to -1.0 pts tolerance).",
            },
            "latency": {
                "ttft": _percentiles(ttft_vals),
                "ttft_warm": _percentiles([r["ttft_ms"] for r in warm_rows]),
                "ttft_cold": _percentiles([r["ttft_ms"] for r in cold_rows]),
                "decode": _percentiles(decode_vals),
                "decode_by_tokens": decode_by_len,
                "slo_ms": {"ttft": self.cfg.ttft_slo_ms, "latency": self.cfg.latency_slo_ms},
            },
            "moe": {
                "token_drop_rate": moe_summary["token_drop_rate"],
                "expert_imbalance_cv": moe_summary["imbalance_cv"],
                "router_entropy": moe_summary["entropy"],
                "gate_margin": moe_summary["gate_margin"],
                "targets": {
                    "token_drop_rate_max": 0.005,
                    "expert_imbalance_cv_max": 0.25,
                    "entropy_drift_pct": 0.20,
                },
            },
            "throughput": throughput_summary,
            "references": {
                "ttft_headroom_pct": 0.15,
                "decode_headroom_pct": 0.15,
                "notes": "Keep TTFT/decode p95 within +10-15% of baseline; goodput counts only SLO-hitting, non-dropped requests.",
            },
        }

    def _attach_baseline(self, scorecard: Dict) -> Dict:
        """Optionally add deltas against a baseline scorecard.json."""
        if not self.cfg.baseline_scorecard:
            return scorecard
        path = Path(self.cfg.baseline_scorecard)
        if not path.exists():
            return scorecard
        baseline = _read_json(path)

        def _pick(dct: Dict, keys: List[str], default: float = 0.0) -> float:
            cur = dct
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return float(cur) if isinstance(cur, (int, float)) else default

        scorecard["baseline_ref"] = str(path)
        scorecard["delta_vs_baseline"] = {
            "avg_accuracy": scorecard["quality"]["avg_accuracy"] - _pick(baseline, ["quality", "avg_accuracy"]),
            "ttft_p95": scorecard["latency"]["ttft"]["p95"] - _pick(baseline, ["latency", "ttft", "p95"]),
            "decode_p95": scorecard["latency"]["decode"]["p95"] - _pick(
                baseline, ["latency", "decode", "p95"]
            ),
            "drop_rate": scorecard["moe"]["token_drop_rate"] - _pick(baseline, ["moe", "token_drop_rate"]),
            "goodput": scorecard["throughput"].get("goodput", 0.0) - _pick(
                baseline, ["throughput", "goodput"]
            ),
        }
        return scorecard

    # ---------------------------------------------------------------- Private helpers

    def _init_llm_if_possible(self) -> None:
        """Attempt to bring up vLLM for real model scoring; fall back silently on failure."""
        if not self.cfg.use_vllm:
            return
        if LLM is None or SamplingParams is None:
            return
        if not torch.cuda.is_available():
            return

        model_path = ensure_gpt_oss_20b(Path(self.cfg.model_path))
        if torch.cuda.device_count() <= 0:
            return
        if not (model_path / "config.json").exists():
            return

        tp = self.cfg.tensor_parallel_size or torch.cuda.device_count()
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                self._llm = LLM(
                    model=str(model_path),
                    tensor_parallel_size=tp,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.8,
                    enforce_eager=True,
                )
                self._llm_available = True
            except Exception as exc:
                self._llm_available = False
                captured_err = buf.getvalue().strip()
                lines = [ln for ln in captured_err.splitlines() if ln]
                lines.append(f"llm_init_error: {exc}")
                try:
                    print(json.dumps({"event": "vllm_llm_init_error", "lines": lines}), file=sys.stderr)
                except Exception:
                    print("\n".join(lines), file=sys.stderr)
                return
        captured = buf.getvalue().strip()
        if captured:
            try:
                lines = [ln for ln in captured.splitlines() if ln]
                print(json.dumps({"event": "vllm_llm_init_stdout", "lines": lines}), file=sys.stderr)
            except Exception:
                print(captured, file=sys.stderr)

    @staticmethod
    def _normalize_text(text: str) -> str:
        norm = text.strip().lower()
        cleaned = []
        for ch in norm:
            if ch.isalnum() or ch.isspace():
                cleaned.append(ch)
        return " ".join("".join(cleaned).split())


def run_eval_stack(cfg: EvalConfig | None = None) -> Dict[str, float]:
    """Convenience entrypoint used by tools."""
    cfg = cfg or EvalConfig()
    runner = CheapEvalStack(cfg)
    return runner.run()
