from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

# Find the repo root
CODE_ROOT = Path(__file__).parent.parent.parent
from core.discovery import get_bench_roots


def _best_optimization_entry(bench: Dict[str, Any]) -> Dict[str, Any]:
    """Return the best succeeded optimization entry when available."""
    optimizations = bench.get("optimizations", [])
    if not isinstance(optimizations, list):
        return {}

    succeeded = [
        opt
        for opt in optimizations
        if isinstance(opt, dict) and str(opt.get("status", "")).lower() == "succeeded"
    ]
    candidates = succeeded or [opt for opt in optimizations if isinstance(opt, dict)]
    if not candidates:
        return {}
    return max(candidates, key=lambda opt: float(opt.get("speedup") or 0.0))


def _story_note(story_metadata: Dict[str, Any]) -> str:
    """Build a concise audit/report note from structured story metadata."""
    if not story_metadata:
        return ""

    parts: List[str] = []
    pair_role = str(story_metadata.get("pair_role") or "").strip().lower()
    chapter_alignment = str(story_metadata.get("chapter_alignment") or "").strip().lower()
    comparison_reason = str(story_metadata.get("comparison_reason") or "").strip()
    chapter_native_targets = story_metadata.get("chapter_native_targets") or []

    if pair_role == "comparison" and chapter_alignment == "supplementary":
        parts.append("Supplementary comparison pair.")
    elif pair_role:
        parts.append(f"{pair_role.title()} pair.")

    if comparison_reason:
        parts.append(comparison_reason.rstrip(".") + ".")

    if isinstance(chapter_native_targets, list):
        targets = [str(target).strip() for target in chapter_native_targets if str(target).strip()]
        if targets:
            parts.append(f"Chapter-native targets: {', '.join(targets)}.")

    return " ".join(part for part in parts if part).strip()


def _transform_aggregated_data(all_benchmarks: dict, timestamp: str) -> dict:
    """Transform aggregated benchmark data to the normalized result format."""
    benchmarks: List[Dict[str, Any]] = []
    all_speedups: List[float] = []
    successful = 0
    failed = 0
    failed_regression = 0

    for (chapter, name), bench in all_benchmarks.items():
        baseline_time = bench.get("baseline_time_ms") or 0
        best_speedup = bench.get("best_speedup") or 1.0
        status = bench.get("status", "unknown")
        bench_type = bench.get("type", "python")

        if status == "succeeded":
            successful += 1
        elif "regression" in str(status):
            failed_regression += 1
        elif "failed" in str(status):
            failed += 1

        optimized_time = baseline_time / best_speedup if best_speedup and best_speedup > 0 else baseline_time
        gpu_metrics = bench.get("baseline_gpu_metrics", {})
        baseline_story_metadata = bench.get("baseline_story_metadata")
        if not isinstance(baseline_story_metadata, dict):
            baseline_story_metadata = {}
        best_optimization = _best_optimization_entry(bench)
        best_optimization_story_metadata = best_optimization.get("story_metadata")
        if not isinstance(best_optimization_story_metadata, dict):
            best_optimization_story_metadata = {}
        pair_story_metadata = dict(baseline_story_metadata)
        for key, value in best_optimization_story_metadata.items():
            if key not in pair_story_metadata and value is not None:
                pair_story_metadata[key] = value
        story_note = _story_note(pair_story_metadata)

        optimizations = []
        for opt in bench.get("optimizations", []):
            opt_story_metadata = opt.get("story_metadata")
            if not isinstance(opt_story_metadata, dict):
                opt_story_metadata = {}
            optimizations.append(
                {
                    "technique": opt.get("technique", ""),
                    "speedup": opt.get("speedup", 1.0),
                    "time_ms": opt.get("time_ms", 0),
                    "file": opt.get("file", ""),
                    "story_metadata": opt_story_metadata,
                    "story_note": _story_note(opt_story_metadata),
                }
            )

        baseline_memory_mb = bench.get("baseline_memory_mb")
        best_memory_savings_pct = bench.get("best_memory_savings_pct", 0)
        optimization_goal = bench.get("optimization_goal", "speed")

        optimized_memory_mb = None
        for opt in bench.get("optimizations", []):
            if opt.get("memory_mb"):
                if optimized_memory_mb is None or opt.get("memory_mb") < optimized_memory_mb:
                    optimized_memory_mb = opt.get("memory_mb")

        if optimized_memory_mb is None and baseline_memory_mb and best_memory_savings_pct:
            optimized_memory_mb = baseline_memory_mb * (1 - best_memory_savings_pct / 100)

        benchmarks.append(
            {
                "name": name,
                "chapter": chapter,
                "type": bench_type,
                "baseline_time_ms": baseline_time,
                "optimized_time_ms": optimized_time,
                "speedup": best_speedup,
                "baseline_memory_mb": baseline_memory_mb,
                "optimized_memory_mb": optimized_memory_mb,
                "memory_savings_pct": best_memory_savings_pct,
                "optimization_goal": optimization_goal,
                "status": status,
                "gpu_temp": gpu_metrics.get("temperature_gpu_c"),
                "gpu_power": gpu_metrics.get("power_draw_w"),
                "gpu_util": gpu_metrics.get("utilization_gpu_pct"),
                "optimizations": optimizations,
                "error": bench.get("error"),
                "p75_ms": bench.get("baseline_p75_ms"),
                "baseline_story_metadata": baseline_story_metadata,
                "best_optimization_story_metadata": best_optimization_story_metadata,
                "story_metadata": pair_story_metadata,
                "pair_role": pair_story_metadata.get("pair_role"),
                "chapter_alignment": pair_story_metadata.get("chapter_alignment"),
                "chapter_native_exemplar": pair_story_metadata.get("chapter_native_exemplar"),
                "comparison_reason": pair_story_metadata.get("comparison_reason"),
                "chapter_native_targets": pair_story_metadata.get("chapter_native_targets"),
                "story_note": story_note,
            }
        )

        if best_speedup > 0:
            all_speedups.append(best_speedup)

    benchmarks.sort(key=lambda x: x["speedup"], reverse=True)

    memory_optimizations = [b for b in benchmarks if b.get("optimization_goal") == "memory"]
    speed_optimizations = [b for b in benchmarks if b.get("optimization_goal") != "memory"]

    return {
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": benchmarks,
        "aggregated": True,
        "source_count": len(all_benchmarks),
        "summary": {
            "total_benchmarks": len(benchmarks),
            "avg_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 0,
            "max_speedup": max(all_speedups) if all_speedups else 0,
            "min_speedup": min(all_speedups) if all_speedups else 0,
            "successful": successful,
            "failed": failed,
            "failed_regression": failed_regression,
            "memory_optimizations": len(memory_optimizations),
            "speed_optimizations": len(speed_optimizations),
            "best_memory_savings_pct": max(
                (b.get("memory_savings_pct", 0) for b in memory_optimizations), default=0
            ),
        },
    }


def load_benchmark_data(data_file: Optional[Path] = None, bench_roots: Optional[List[Path]] = None) -> dict:
    """Load benchmark results from disk (single file or aggregated artifacts)."""
    all_benchmarks: Dict[tuple, Dict[str, Any]] = {}
    latest_timestamp: Optional[str] = None
    bench_roots = bench_roots or get_bench_roots(repo_root=CODE_ROOT)

    def _ingest_result_file(path: Path) -> None:
        nonlocal latest_timestamp
        with open(path) as f:
            data = json.load(f)
        timestamp = data.get("timestamp", "")
        if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
            latest_timestamp = timestamp

        for chapter_result in data.get("results", []):
            chapter = chapter_result.get("chapter", "unknown")
            for bench in chapter_result.get("benchmarks", []):
                name = bench.get("example", "unknown")
                key = (chapter, name)
                speedup = bench.get("best_speedup", 0)
                if key not in all_benchmarks or speedup > all_benchmarks[key].get("best_speedup", 0):
                    all_benchmarks[key] = bench
                    all_benchmarks[key]["_chapter"] = chapter

    if data_file:
        path = Path(data_file)
        if path.exists():
            try:
                _ingest_result_file(path)
            except Exception:
                pass
    else:
        for root in bench_roots:
            artifacts_dir = root / "artifacts"
            if artifacts_dir.exists():
                for result_file in sorted(artifacts_dir.rglob("benchmark_test_results.json")):
                    try:
                        _ingest_result_file(result_file)
                    except Exception:
                        pass

            default_path = root / "benchmark_test_results.json"
            if default_path.exists():
                try:
                    _ingest_result_file(default_path)
                except Exception:
                    pass

    if not all_benchmarks:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": [],
            "summary": {"total_benchmarks": 0, "avg_speedup": 0, "max_speedup": 0},
        }

    return _transform_aggregated_data(all_benchmarks, latest_timestamp)


class PerformanceAnalyzer:
    """Stateless benchmark analysis utilities shared by CLI and server."""

    def __init__(self, data_loader: Optional[Callable[[], dict]] = None):
        self._data_loader = data_loader or load_benchmark_data

    def _load_data(self) -> dict:
        try:
            return (self._data_loader or load_benchmark_data)() or {}
        except Exception:
            return {}

    def get_categorized_leaderboards(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        speed_benchmarks = [b for b in benchmarks if b.get("optimization_goal", "speed") == "speed"]
        memory_benchmarks = [b for b in benchmarks if b.get("optimization_goal") == "memory"]
        throughput_benchmarks = [b for b in benchmarks if b.get("optimization_goal") == "throughput"]

        speed_benchmarks.sort(key=lambda x: x.get("speedup", 0), reverse=True)
        memory_benchmarks.sort(key=lambda x: x.get("memory_savings_pct", 0) or 0, reverse=True)
        throughput_benchmarks.sort(key=lambda x: x.get("speedup", 0), reverse=True)

        def format_benchmark(b, category):
            if category == "memory":
                return {
                    "rank": 0,
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "primary_metric": f"-{b.get('memory_savings_pct', 0):.0f}%",
                    "secondary_metric": f"{b.get('speedup', 1):.2f}x speed",
                    "value": b.get("memory_savings_pct", 0) or 0,
                }
            return {
                "rank": 0,
                "name": f"{b.get('chapter')}:{b.get('name')}",
                "primary_metric": f"{b.get('speedup', 1):.2f}x",
                "secondary_metric": f"{b.get('memory_savings_pct', 0):.0f}% mem" if b.get("memory_savings_pct") else "",
                "value": b.get("speedup", 1),
            }

        speed_board = [format_benchmark(b, "speed") for b in speed_benchmarks[:10]]
        for i, b in enumerate(speed_board):
            b["rank"] = i + 1

        memory_board = [format_benchmark(b, "memory") for b in memory_benchmarks[:10]]
        for i, b in enumerate(memory_board):
            b["rank"] = i + 1

        throughput_board = [format_benchmark(b, "throughput") for b in throughput_benchmarks[:10]]
        for i, b in enumerate(throughput_board):
            b["rank"] = i + 1

        return {
            "leaderboards": {
                "speed": {
                    "title": "🚀 Speed Champions",
                    "metric_label": "Speedup",
                    "entries": speed_board,
                    "count": len(speed_benchmarks),
                },
                "memory": {
                    "title": "💾 Memory Champions",
                    "metric_label": "Memory Saved",
                    "entries": memory_board,
                    "count": len(memory_benchmarks),
                },
                "throughput": {
                    "title": "📈 Throughput Champions",
                    "metric_label": "Throughput",
                    "entries": throughput_board,
                    "count": len(throughput_benchmarks),
                },
            },
            "summary": {
                "speed_count": len(speed_benchmarks),
                "memory_count": len(memory_benchmarks),
                "throughput_count": len(throughput_benchmarks),
            },
        }

    def get_pareto_frontier(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        points = []
        for b in benchmarks:
            speedup = b.get("speedup", 1.0)
            memory_savings = b.get("memory_savings_pct", 0) or 0
            points.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": speedup,
                    "memory_savings": memory_savings,
                    "goal": b.get("optimization_goal", "speed"),
                    "is_pareto": False,
                }
            )

        pareto_points: List[Dict[str, Any]] = []
        for i, p in enumerate(points):
            is_dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                if q["speedup"] >= p["speedup"] and q["memory_savings"] >= p["memory_savings"]:
                    if q["speedup"] > p["speedup"] or q["memory_savings"] > p["memory_savings"]:
                        is_dominated = True
                        break
            if not is_dominated:
                p["is_pareto"] = True
                pareto_points.append(p)

        return {
            "pareto_frontier": pareto_points,
            "all_points": points,
            "pareto_count": len(pareto_points),
            "total_count": len(points),
        }

    def get_tradeoff_analysis(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        tradeoffs: List[Dict[str, Any]] = []
        for b in benchmarks:
            speedup = b.get("speedup", 1.0)
            memory_savings = b.get("memory_savings_pct", 0) or 0
            goal = b.get("optimization_goal", "speed")

            if goal == "memory" and memory_savings > 0:
                speed_cost = max(0, 1 - speedup)
                cost_per_percent = (speed_cost * 100) / memory_savings if memory_savings > 0 else 0
                efficiency = memory_savings / (speed_cost * 100 + 1)
                benefit_label = f"-{memory_savings:.0f}% memory"
                cost_label = f"{speed_cost*100:.0f}% slower" if speed_cost > 0 else "No speed loss"
            else:
                speed_gain = max(0, speedup - 1)
                memory_cost = max(0, -memory_savings) if memory_savings else 0
                efficiency = speedup / (memory_cost + 1)
                benefit_label = f"{speedup:.2f}x faster"
                cost_label = f"+{memory_cost:.0f}% memory" if memory_cost > 0 else "No memory increase"
                cost_per_percent = memory_cost / (speed_gain * 100 + 1) if speed_gain > 0 else 0

            tradeoffs.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "goal": goal,
                    "speedup": speedup,
                    "memory_savings": memory_savings,
                    "efficiency_score": round(efficiency, 2),
                    "cost_per_percent": round(cost_per_percent, 3),
                    "benefit": benefit_label,
                    "cost": cost_label,
                }
            )

        tradeoffs.sort(key=lambda x: x["efficiency_score"], reverse=True)

        return {
            "tradeoffs": tradeoffs,
            "best_efficiency": tradeoffs[0] if tradeoffs else None,
            "memory_specialists": [t for t in tradeoffs if t["goal"] == "memory"][:5],
            "speed_specialists": [t for t in tradeoffs if t["goal"] == "speed"][:5],
        }

    def get_constraint_recommendations(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        scenarios: List[Dict[str, Any]] = []

        memory_opts = sorted(
            [b for b in benchmarks if (b.get("memory_savings_pct") or 0) > 10],
            key=lambda x: x.get("memory_savings_pct", 0),
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Memory Constrained",
                "description": "Need to fit larger models/batches in limited VRAM",
                "icon": "💾",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"-{b.get('memory_savings_pct', 0):.0f}% memory",
                        "tradeoff": f"{b.get('speedup', 1):.2f}x speed",
                    }
                    for b in memory_opts[:3]
                ],
            }
        )

        speed_opts = sorted(
            [b for b in benchmarks if b.get("speedup", 1) > 1.5],
            key=lambda x: x.get("speedup", 1),
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Latency Critical",
                "description": "Need lowest possible inference/training time",
                "icon": "⚡",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"{b.get('speedup', 1):.2f}x faster",
                        "tradeoff": f"{b.get('memory_savings_pct', 0):.0f}% mem change"
                        if b.get("memory_savings_pct")
                        else "Minimal memory impact",
                    }
                    for b in speed_opts[:3]
                ],
            }
        )

        balanced = sorted(
            [b for b in benchmarks if b.get("speedup", 1) >= 1.0 and (b.get("memory_savings_pct", 0) or 0) >= 0],
            key=lambda x: x.get("speedup", 1) + (x.get("memory_savings_pct", 0) or 0) / 50,
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Balanced",
                "description": "Want improvements without significant trade-offs",
                "icon": "⚖️",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"{b.get('speedup', 1):.2f}x speed",
                        "tradeoff": f"{b.get('memory_savings_pct', 0):.0f}% mem"
                        if b.get("memory_savings_pct")
                        else "Neutral",
                    }
                    for b in balanced[:3]
                ],
            }
        )

        training_opts = [b for b in benchmarks if "training" in b.get("name", "").lower()]
        if training_opts:
            scenarios.append(
                {
                    "name": "Large Model Training",
                    "description": "Training models that barely fit in memory",
                    "icon": "🏋️",
                    "recommendations": [
                        {
                            "name": f"{b.get('chapter')}:{b.get('name')}",
                            "benefit": f"-{b.get('memory_savings_pct', 0):.0f}% memory"
                            if b.get("memory_savings_pct", 0) > 0
                            else f"{b.get('speedup', 1):.2f}x speed",
                            "tradeoff": b.get("optimization_goal", "speed"),
                        }
                        for b in training_opts[:3]
                    ],
                }
            )

        return {"scenarios": scenarios, "total_benchmarks": len(benchmarks)}

    def get_whatif_recommendations(self, params: dict) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        max_vram_gb = float(params.get("vram", [999999])[0])
        max_latency_ms = float(params.get("latency", [999999])[0])
        min_throughput = float(params.get("throughput", [0])[0])
        max_memory_gb = float(params.get("memory_budget", [999999])[0])

        matching: List[Dict[str, Any]] = []
        for b in benchmarks:
            opt_mem_gb = (b.get("optimized_memory_mb") or b.get("baseline_memory_mb") or 0) / 1024
            if opt_mem_gb > max_memory_gb or opt_mem_gb > max_vram_gb:
                continue

            opt_latency_ms = b.get("optimized_time_ms") or b.get("baseline_time_ms") or 0
            if opt_latency_ms > max_latency_ms:
                continue

            opt_throughput = (b.get("throughput") or b.get("speedup", 1)) * 1000
            if opt_throughput < min_throughput:
                continue

            matching.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": b.get("speedup", 1.0),
                    "memory_savings_pct": b.get("memory_savings_pct", 0),
                    "latency_ms": opt_latency_ms,
                    "throughput": opt_throughput,
                }
            )

        matching.sort(key=lambda x: (x["speedup"], x["memory_savings_pct"]), reverse=True)

        best_for_speed = matching[0] if matching else None
        best_for_memory = (
            max(matching, key=lambda x: x["memory_savings_pct"])
            if matching and any(m.get("memory_savings_pct") for m in matching)
            else None
        )

        return {
            "recommendations": matching[:10],
            "matching_count": len(matching),
            "total_benchmarks": len(benchmarks),
            "constraints": {
                "vram_gb": max_vram_gb if max_vram_gb < 999999 else None,
                "latency_ms": max_latency_ms if max_latency_ms < 999999 else None,
                "min_throughput": min_throughput if min_throughput > 0 else None,
                "memory_budget_gb": max_memory_gb if max_memory_gb < 999999 else None,
            },
            "best_for_speed": best_for_speed,
            "best_for_memory": best_for_memory,
        }

    def get_optimization_stacking(self) -> dict:
        optimization_categories = {
            "memory_format": ["quantization", "fp8", "fp4", "int8", "bf16"],
            "attention": ["flash_attention", "flex_attention", "sdpa", "paged_attention"],
            "parallelism": ["tensor_parallel", "pipeline_parallel", "data_parallel", "fsdp"],
            "caching": ["kv_cache", "cuda_graphs", "compile"],
            "memory_saving": ["checkpointing", "activation_checkpoint", "gradient_checkpoint"],
            "communication": ["nccl", "nvlink", "nvshmem"],
        }

        compatible_pairs = [
            ("flash_attention", "quantization", "High", "5-10x speed + 2-4x memory reduction"),
            ("cuda_graphs", "tensor_cores", "High", "Reduced launch overhead + fast math"),
            ("kv_cache", "quantization", "High", "Compressed cache for longer sequences"),
            ("flash_attention", "tensor_parallel", "High", "Scale attention across GPUs"),
            ("compile", "cuda_graphs", "Medium", "Compiled graphs for best performance"),
            ("fsdp", "bf16", "High", "Sharded training with mixed precision"),
            ("checkpointing", "flash_attention", "Medium", "Memory savings stack"),
        ]

        incompatible_pairs = [
            ("cuda_graphs", "dynamic_shapes", "Graphs require static shapes"),
            ("checkpointing", "cuda_graphs", "Checkpointing needs dynamic execution"),
            ("eager_mode", "compile", "Mutually exclusive execution modes"),
        ]

        return {
            "categories": optimization_categories,
            "compatible_combinations": [
                {"opt1": p[0], "opt2": p[1], "synergy": p[2], "benefit": p[3]} for p in compatible_pairs
            ],
            "incompatible_combinations": [{"opt1": p[0], "opt2": p[1], "reason": p[2]} for p in incompatible_pairs],
            "recommended_stacks": [
                {
                    "name": "Maximum Speed (Inference)",
                    "stack": ["flash_attention", "cuda_graphs", "fp8_quantization", "tensor_cores"],
                    "expected_benefit": "10-50x speedup",
                },
                {
                    "name": "Memory Efficient Training",
                    "stack": ["gradient_checkpointing", "fsdp", "bf16", "flash_attention"],
                    "expected_benefit": "2-4x larger models",
                },
                {
                    "name": "Balanced (Speed + Memory)",
                    "stack": ["flash_attention", "kv_cache_quantization", "compile"],
                    "expected_benefit": "3-5x speed, 30-50% memory reduction",
                },
            ],
        }

    def get_power_efficiency(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        efficiency_data: List[Dict[str, Any]] = []
        for b in benchmarks:
            power_w = b.get("gpu_power") or 0
            speedup = b.get("speedup", 1)
            time_ms = b.get("optimized_time_ms") or b.get("baseline_time_ms") or 1

            if power_w > 0 and time_ms > 0:
                ops_per_watt = speedup / (power_w * time_ms / 1000)
                efficiency_data.append(
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "speedup": speedup,
                        "power_w": power_w,
                        "time_ms": time_ms,
                        "ops_per_watt": round(ops_per_watt, 4),
                        "energy_j": round(power_w * time_ms / 1000, 2),
                    }
                )

        efficiency_data.sort(key=lambda x: x["ops_per_watt"], reverse=True)

        return {
            "efficiency_rankings": efficiency_data[:15],
            "most_efficient": efficiency_data[0] if efficiency_data else None,
            "least_efficient": efficiency_data[-1] if efficiency_data else None,
            "avg_power_w": sum(e["power_w"] for e in efficiency_data) / len(efficiency_data) if efficiency_data else 0,
            "total_benchmarks_with_power": len(efficiency_data),
        }

    def get_scaling_analysis(self) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        scaling_insights = []
        categories: Dict[str, List[Dict[str, Any]]] = {
            "attention": [],
            "matmul": [],
            "memory": [],
            "training": [],
        }

        for b in benchmarks:
            name = b.get("name", "").lower()
            speedup = b.get("speedup", 1)

            if "attention" in name or "flash" in name or "sdpa" in name:
                categories["attention"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "matmul" in name or "gemm" in name:
                categories["matmul"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "memory" in name or "kv" in name or "cache" in name:
                categories["memory"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "training" in name or "grad" in name:
                categories["training"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})

        scaling_recommendations = [
            {
                "factor": "Sequence Length",
                "insight": "Flash Attention scales O(n) vs O(n²) - critical for long sequences",
                "recommendation": "Use Flash Attention for seq_len > 512",
            },
            {
                "factor": "Batch Size",
                "insight": "CUDA Graphs amortize launch overhead - better at larger batches",
                "recommendation": "Use CUDA Graphs for batch_size > 8",
            },
            {
                "factor": "Model Size",
                "insight": "Tensor Parallelism scales linearly with GPU count",
                "recommendation": "Use TP for models > single GPU memory",
            },
            {
                "factor": "Hidden Dimension",
                "insight": "Tensor Cores are most efficient at multiples of 8/16",
                "recommendation": "Pad dimensions to multiples of 16 for Tensor Core utilization",
            },
        ]

        return {
            "categories": {k: sorted(v, key=lambda x: x["speedup"], reverse=True)[:5] for k, v in categories.items()},
            "scaling_recommendations": scaling_recommendations,
            "key_insight": "Optimization impact increases with workload size - small benchmarks may underestimate real-world gains",
        }

    def get_cost_analysis(self, gpu: str = None, custom_rate: float = None) -> dict:
        data = self._load_data()
        benchmarks = data.get("benchmarks", [])

        gpu_pricing = {
            "B200": {"hourly": 5.00, "name": "NVIDIA B200", "tflops": 2250},
            "H100": {"hourly": 3.50, "name": "NVIDIA H100", "tflops": 1979},
            "A100": {"hourly": 2.00, "name": "NVIDIA A100", "tflops": 312},
            "L40S": {"hourly": 1.50, "name": "NVIDIA L40S", "tflops": 362},
            "A10G": {"hourly": 1.00, "name": "NVIDIA A10G", "tflops": 125},
            "T4": {"hourly": 0.50, "name": "NVIDIA T4", "tflops": 65},
        }

        selected_gpu = gpu or "B200"
        if custom_rate is not None:
            hourly_rate = custom_rate
        elif selected_gpu in gpu_pricing:
            hourly_rate = gpu_pricing[selected_gpu]["hourly"]
        else:
            hourly_rate = gpu_pricing["B200"]["hourly"]

        cost_analysis: List[Dict[str, Any]] = []
        for b in benchmarks:
            speedup = b.get("speedup", 1)
            if speedup <= 1:
                continue

            baseline_time_ms = b.get("baseline_time_ms") or 100
            optimized_time_ms = b.get("optimized_time_ms") or baseline_time_ms

            baseline_ops_per_hour = 3600 * 1000 / baseline_time_ms
            optimized_ops_per_hour = 3600 * 1000 / optimized_time_ms

            baseline_cost_per_m = (hourly_rate / baseline_ops_per_hour) * 1_000_000
            optimized_cost_per_m = (hourly_rate / optimized_ops_per_hour) * 1_000_000
            savings_per_m = baseline_cost_per_m - optimized_cost_per_m
            savings_pct = (savings_per_m / baseline_cost_per_m) * 100 if baseline_cost_per_m > 0 else 0

            cost_analysis.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": speedup,
                    "baseline_cost_per_m": round(baseline_cost_per_m, 4),
                    "optimized_cost_per_m": round(optimized_cost_per_m, 4),
                    "savings_per_m": round(savings_per_m, 4),
                    "savings_pct": round(savings_pct, 1),
                }
            )

        cost_analysis.sort(key=lambda x: x["savings_pct"], reverse=True)

        return {
            "gpu_pricing": gpu_pricing,
            "assumed_gpu": selected_gpu,
            "hourly_rate": hourly_rate,
            "available_gpus": list(gpu_pricing.keys()),
            "cost_rankings": cost_analysis[:15],
            "highest_savings": cost_analysis[0] if cost_analysis else None,
            "total_potential_savings": f"Up to {cost_analysis[0]['savings_pct']:.0f}% cost reduction"
            if cost_analysis
            else "N/A",
        }
