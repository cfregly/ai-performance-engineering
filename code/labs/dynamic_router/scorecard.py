"""CLI to render a ship/no-ship table from cheap-eval artifacts.

Usage:
    python labs/dynamic_router/scorecard.py artifacts/dynamic_router/cheap_eval/baseline_* artifacts/dynamic_router/cheap_eval/optimized_*

Optional:
    --baseline <run_dir>          # reference for deltas
    --plot out.png                # save a matplotlib figure (Agg backend)
    --ttft-p95-max 1300           # override SLOs if sys_meta missing
    --latency-p95-max 4500
    --drop-max 0.005
    --imbalance-max 0.25
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _pct(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[int(lo)] * (hi - k) + ordered[int(hi)] * (k - lo)


def _load_run(run_dir: Path) -> Dict:
    quality_rows = _read_jsonl(run_dir / "quality.jsonl")
    latency_rows = _read_jsonl(run_dir / "latency.jsonl")
    moe_router_rows = _read_jsonl(run_dir / "moe_router.jsonl")
    tps_goodput = _read_json(run_dir / "tps_goodput.json")
    sys_meta = _read_json(run_dir / "sys_meta.json")

    quality_acc = (
        sum(1 for r in quality_rows if r.get("correct")) / float(len(quality_rows) or 1)
    )
    ttft_p95 = _pct([r.get("ttft_ms", 0.0) for r in latency_rows], 95)
    decode_p95 = _pct([r.get("decode_ms", 0.0) for r in latency_rows], 95)
    drop_rate = (
        sum(r.get("drops", 0) for r in moe_router_rows) / float(len(moe_router_rows) or 1)
    )
    traffic_rows = _read_jsonl(run_dir / "moe_traffic.jsonl")
    imbalance_cv = max((r.get("imbalance_cv", 0.0) for r in traffic_rows), default=0.0)

    return {
        "dir": run_dir,
        "acc": quality_acc,
        "ttft_p95": ttft_p95,
        "decode_p95": decode_p95,
        "drop_rate": drop_rate,
        "imbalance_cv": imbalance_cv,
        "goodput": tps_goodput.get("goodput", 0.0),
        "throughput": tps_goodput.get("throughput_tps", 0.0),
        "meta": sys_meta,
    }


def _fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"


def _print_table(runs: List[Dict], args: argparse.Namespace, baseline: Dict | None) -> None:
    headers = [
        "run",
        "acc",
        "ttft_p95",
        "decode_p95",
        "drop %",
        "imbalance",
        "goodput",
    ]
    print(" | ".join(headers))
    print("-" * 80)
    for r in runs:
        acc = f"{r['acc']:.3f}"
        ttft = f"{r['ttft_p95']:.0f} ms"
        decode = f"{r['decode_p95']:.0f} ms"
        drop = _fmt_pct(r["drop_rate"])
        imb = f"{r['imbalance_cv']:.3f}"
        goodput = f"{r['goodput']:.1f}"
        line = [r['dir'].name, acc, ttft, decode, drop, imb, goodput]
        print(" | ".join(line))
    print("\nPASS/FAIL vs thresholds (or sys_meta SLOs when present):")
    for r in runs:
        meta = r["meta"]
        ttft_cap = meta.get("ttft_slo_ms", args.ttft_p95_max)
        decode_cap = meta.get("latency_slo_ms", args.latency_p95_max)
        drop_cap = args.drop_max
        imb_cap = args.imbalance_max
        ok = (
            r["acc"] >= args.min_acc
            and r["ttft_p95"] <= ttft_cap
            and r["decode_p95"] <= decode_cap
            and r["drop_rate"] <= drop_cap
            and r["imbalance_cv"] <= imb_cap
        )
        status = "SHIP" if ok else "BLOCK"
        delta_str = ""
        if baseline and baseline is not r:
            delta_str = f" | Δacc={r['acc']-baseline['acc']:+.3f}"
        print(f"- {r['dir'].name}: {status} (ttft<= {ttft_cap}ms, decode<= {decode_cap}ms){delta_str}")


def _maybe_plot(runs: List[Dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional
        print(f"[scorecard] Skipping plot (matplotlib unavailable: {exc})")
        return

    labels = [r["dir"].name for r in runs]
    x = list(range(len(labels)))
    ttft_vals = [r["ttft_p95"] for r in runs]
    decode_vals = [r["decode_p95"] for r in runs]
    acc_vals = [r["acc"] for r in runs]
    drop_vals = [r["drop_rate"] * 100 for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    axes[0].bar(x, acc_vals, color="#2b8cbe")
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(x, ttft_vals, color="#a6bddb")
    axes[1].set_ylabel("TTFT p95 (ms)")

    axes[2].bar(x, decode_vals, color="#1c9099")
    axes[2].set_ylabel("Decode p95 (ms)")

    axes[3].bar(x, drop_vals, color="#e34a33")
    axes[3].set_ylabel("Drop rate (%)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[scorecard] Wrote plot to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="Run directories containing cheap-eval artifacts.")
    ap.add_argument("--baseline", type=Path, default=None, help="Optional baseline run dir for deltas.")
    ap.add_argument("--plot", type=Path, default=None, help="Optional path to save a matplotlib PNG.")
    ap.add_argument("--ttft-p95-max", type=float, default=1300.0)
    ap.add_argument("--latency-p95-max", type=float, default=4500.0)
    ap.add_argument("--drop-max", type=float, default=0.005)
    ap.add_argument("--imbalance-max", type=float, default=0.25)
    ap.add_argument("--min-acc", type=float, default=0.6)
    args = ap.parse_args()

    runs = [_load_run(r) for r in args.runs]
    baseline = _load_run(args.baseline) if args.baseline else None

    if baseline:
        # Ensure baseline shows up first
        runs = [baseline] + [r for r in runs if r["dir"] != baseline["dir"]]

    _print_table(runs, args, baseline)

    if args.plot:
        _maybe_plot(runs, args.plot)


if __name__ == "__main__":
    main()
