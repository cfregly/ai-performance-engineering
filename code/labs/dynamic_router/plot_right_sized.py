"""Plot TTFT and goodput-per-dollar for dynamic router scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_runs(paths: List[Path]) -> List[Tuple[str, Dict[str, float]]]:
    runs: List[Tuple[str, Dict[str, float]]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        label = path.stem
        if "scenario" in data:
            label = f"{data.get('scenario')}-{data.get('mode', '')}-{path.stem}"
        runs.append((label, data))
    return runs


def _plot_metric(
    runs: List[Tuple[str, Dict[str, float]]],
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for name, data in runs:
        labels.append(name)
        values.append(data.get(metric, 0.0))
    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, values, color="#4f6bed")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.1f}", ha="center", va="bottom")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dynamic-router right-sizing runs")
    parser.add_argument("runs", nargs="+", type=Path, help="JSON summaries from driver.py --log-json")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/dynamic_router/plots"),
        help="Directory to write plots",
    )
    args = parser.parse_args()

    samples = _load_runs(sorted(args.runs))
    if not samples:
        raise SystemExit("No runs provided")

    args.outdir.mkdir(parents=True, exist_ok=True)
    _plot_metric(samples, "ttft_ms_p95", "TTFT p95 (ms)", args.outdir / "ttft_p95.png")
    _plot_metric(samples, "goodput_tokens_per_dollar", "Goodput tokens per $", args.outdir / "goodput_per_dollar.png")


if __name__ == "__main__":
    main()
