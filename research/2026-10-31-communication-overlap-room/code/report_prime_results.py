#!/usr/bin/env python3
"""Explain the arithmetic in Prime Intellect's source-reported step timings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must fit a finite numeric value") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def load_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("the data must be a JSON object")
    if data.get("evidence_class") != "source_reported":
        raise ValueError("the evidence class must be source_reported")
    if data.get("unit") != "seconds_per_step":
        raise ValueError("the input unit must be seconds_per_step")
    if data.get("source_url") != "https://www.primeintellect.ai/blog/prime-mega-moe":
        raise ValueError("the source URL must identify Prime's Mega MoE article")
    if not isinstance(data.get("author"), str) or not data["author"].strip():
        raise ValueError("the data must name the source author")
    rows = data.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("results must be a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each result must be an object")
        model = row.get("model")
        if not isinstance(model, str) or not model.strip() or model in seen:
            raise ValueError("each result needs a distinct nonempty model name")
        seen.add(model)
        gpus = positive_number(row.get("gpus"), f"{model}: GPU count")
        if not gpus.is_integer():
            raise ValueError(f"{model}: GPU count must be an integer")
        positive_number(row.get("baseline_seconds"), f"{model}: baseline time")
        positive_number(row.get("candidate_seconds"), f"{model}: candidate time")
        positive_number(row.get("reported_throughput_gain_percent"), f"{model}: reported gain")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("prime_results.json"))
    args = parser.parse_args()
    try:
        data = load_results(args.data)
        print("Source-reported training-step timings. This is arithmetic, not a benchmark run.")
        print(f"Authors: {data['author']}")
        print(f"Source: {data['source_url']}")
        print("Model | GPUs | Baseline s | Mega MoE s | Approx. ratio | Approx. time reduction | Source throughput gain")
        for row in data["results"]:
            before = float(row["baseline_seconds"])
            after = float(row["candidate_seconds"])
            ratio = before / after
            reduction = 100 * (1 - after / before)
            print(f"{row['model']} | {row['gpus']} | {before:.2f} | {after:.2f} | {ratio:.3f}x | {reduction:.1f}% | {row['reported_throughput_gain_percent']}%")
        print("Derived ratios use rounded chart values. Source throughput gains are attributed, not recomputed measurements.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot read source timings: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
