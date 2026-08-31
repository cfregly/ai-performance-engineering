#!/usr/bin/env python3
"""Summarise torch.profiler outputs captured by core/scripts/pytorch_profiler_runner.py."""

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, Iterable, List


def _load_json(path: pathlib.Path) -> Dict:
    with path.open() as fh:
        return json.load(fh)


def _discover_modes(directory: pathlib.Path) -> List[str]:
    modes: List[str] = []
    for file in directory.glob("key_averages_*.json"):
        suffix = file.stem.replace("key_averages_", "")
        modes.append(suffix)
    return modes


def collect_metadata(dir_path: pathlib.Path) -> Dict[str, str]:
    metadata_path = dir_path / "metadata.json"
    if not metadata_path.exists():
        return {}
    meta = _load_json(metadata_path)
    return {
        "script": meta.get("script", ""),
        "args": " ".join(meta.get("args", [])),
        "torch_version": meta.get("torch_version", ""),
        "duration_seconds": str(meta.get("duration_seconds", "")),
        "cuda_available": str(meta.get("cuda_available", "")),
        "cuda_device": meta.get("cuda_device", ""),
        "error": meta.get("error", ""),
    }


def collect_key_averages(dir_path: pathlib.Path, mode: str) -> List[Dict[str, str]]:
    json_path = dir_path / f"key_averages_{mode}.json"
    if not json_path.exists():
        return []
    data = _load_json(json_path)
    rows: List[Dict[str, str]] = []
    for entry in data:
        rows.append(
            {
                "mode": mode,
                "name": entry.get("name", ""),
                "count": str(entry.get("count", "")),
                "cpu_time_total_us": str(entry.get("cpu_time_total_us", "")),
                "cuda_time_total_us": str(entry.get("cuda_time_total_us", "")),
                "self_cpu_time_total_us": str(entry.get("self_cpu_time_total_us", "")),
                "self_cuda_time_total_us": str(entry.get("self_cuda_time_total_us", "")),
                "cpu_memory_usage": str(entry.get("cpu_memory_usage", "")),
                "cuda_memory_usage": str(entry.get("cuda_memory_usage", "")),
            }
        )
    return rows


def scan(patterns: Iterable[str]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    result: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    base = pathlib.Path()
    for pat in patterns:
        for directory in base.glob(pat):
            if not directory.is_dir():
                continue
            tag = str(directory)
            meta = collect_metadata(directory)
            modes = _discover_modes(directory)
            entries: List[Dict[str, str]] = []
            for mode in modes:
                entries.extend(collect_key_averages(directory, mode))
            result[tag] = {"metadata": meta, "rows": entries}
    return result


def write_outputs(data: Dict[str, Dict[str, List[Dict[str, str]]]], out_prefix: pathlib.Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_prefix.with_suffix("_metadata.csv")
    rows_path = out_prefix.with_suffix("_operators.csv")

    with meta_path.open("w", newline="") as fh:
        fieldnames = [
            "profile_dir",
            "script",
            "args",
            "torch_version",
            "duration_seconds",
            "cuda_available",
            "cuda_device",
            "error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for directory, payload in data.items():
            row = {"profile_dir": directory}
            row.update(payload.get("metadata", {}))
            writer.writerow(row)

    with rows_path.open("w", newline="") as fh:
        fieldnames = [
            "profile_dir",
            "mode",
            "name",
            "count",
            "cpu_time_total_us",
            "cuda_time_total_us",
            "self_cpu_time_total_us",
            "self_cuda_time_total_us",
            "cpu_memory_usage",
            "cuda_memory_usage",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for directory, payload in data.items():
            for row in payload.get("rows", []):
                record = {"profile_dir": directory}
                record.update(row)
                writer.writerow(record)

    print(f"Wrote {meta_path} and {rows_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarise PyTorch profiler outputs to CSV")
    parser.add_argument("patterns", nargs="+", help="Glob(s) matching profiler output directories")
    parser.add_argument(
        "--output-prefix",
        default="artifacts/runs/analysis/pytorch_profile",
        help="Prefix for generated CSV files (default: artifacts/runs/analysis/pytorch_profile)",
    )
    args = parser.parse_args(argv)

    data = scan(args.patterns)
    if not data:
        print("No PyTorch profiler directories found", file=sys.stderr)
        return 1

    write_outputs(data, pathlib.Path(args.output_prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
