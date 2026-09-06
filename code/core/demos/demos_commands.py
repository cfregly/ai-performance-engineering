"""Demos commands mounted by aisp (`aisp demos ...`).

Demos are runnable examples and chapter companions. They are intentionally
separated from:
  - `aisp bench run` (comparable baseline vs optimized benchmarks), and
  - `aisp tools` (utilities / analysis scripts).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    import typer
    from typer import Argument

    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover - Typer is optional for docs builds
    TYPER_AVAILABLE = False
    typer = None  # type: ignore
    Argument = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]

from core.utils.python_entrypoints import (
    build_python_entry_command,
    build_repo_python_env,
    build_torchrun_entry_command,
)


class LaunchVia(str, Enum):
    PYTHON = "python"
    TORCHRUN = "torchrun"


@dataclass(frozen=True)
class DemoSpec:
    name: str
    script_path: Path
    description: str
    launch_via: LaunchVia
    module_name: Optional[str] = None


DEMOS: Dict[str, DemoSpec] = {
    "ch10-tma-multicast": DemoSpec(
        name="ch10-tma-multicast",
        script_path=REPO_ROOT / "ch10" / "tma_multicast_tool.py",
        description="Chapter 10 TMA multicast demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch10-tmem-triple-overlap": DemoSpec(
        name="ch10-tmem-triple-overlap",
        script_path=REPO_ROOT / "ch10" / "tmem_triple_overlap_tool.py",
        description="Chapter 10 TMEM triple-overlap demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch10-warpgroup-specialization": DemoSpec(
        name="ch10-warpgroup-specialization",
        script_path=REPO_ROOT / "ch10" / "warpgroup_specialization_demo.py",
        description="Chapter 10 tcgen05 warpgroup specialization demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch10-dsmem-reduction-fixed": DemoSpec(
        name="ch10-dsmem-reduction-fixed",
        script_path=REPO_ROOT / "ch10" / "dsmem_reduction_fixed_demo.py",
        description="Chapter 10 DSMEM reduction fixed binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch11-stream-overlap": DemoSpec(
        name="ch11-stream-overlap",
        script_path=REPO_ROOT / "ch11" / "stream_overlap_demo.py",
        description="Chapter 11 CUDA stream overlap demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch11.stream_overlap_demo",
    ),
    "ch11-stream-priority": DemoSpec(
        name="ch11-stream-priority",
        script_path=REPO_ROOT / "ch11" / "stream_priority_demo.py",
        description="Chapter 11 CUDA stream priority demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch11.stream_priority_demo",
    ),
    "ch11-memory-async": DemoSpec(
        name="ch11-memory-async",
        script_path=REPO_ROOT / "ch11" / "memory_async_demo.py",
        description="Chapter 11 async copy/compute overlap demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch11.memory_async_demo",
    ),
    "ch11-event-timing": DemoSpec(
        name="ch11-event-timing",
        script_path=REPO_ROOT / "ch11" / "event_timing_demo.py",
        description="Chapter 11 CUDA event timing demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch11.event_timing_demo",
    ),
    "ch11-streams-overlap-cuda": DemoSpec(
        name="ch11-streams-overlap-cuda",
        script_path=REPO_ROOT / "ch11" / "streams_overlap_demo.py",
        description="Chapter 11 CUDA streams overlap binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch11-streams-ordered-cuda": DemoSpec(
        name="ch11-streams-ordered-cuda",
        script_path=REPO_ROOT / "ch11" / "streams_ordered_demo.py",
        description="Chapter 11 stream-ordered allocator binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch11-streams-warp-specialized-cuda": DemoSpec(
        name="ch11-streams-warp-specialized-cuda",
        script_path=REPO_ROOT / "ch11" / "streams_warp_specialized_demo.py",
        description="Chapter 11 warp-specialized streams binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch12-graph-capture": DemoSpec(
        name="ch12-graph-capture",
        script_path=REPO_ROOT / "ch12" / "graph_capture_demo.py",
        description="Chapter 12 CUDA graph capture demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch12.graph_capture_demo",
    ),
    "ch12-graph-replay": DemoSpec(
        name="ch12-graph-replay",
        script_path=REPO_ROOT / "ch12" / "graph_replay_benchmark.py",
        description="Chapter 12 CUDA graph replay microbenchmark demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch12.graph_replay_benchmark",
    ),
    "ch12-instantiation-overhead": DemoSpec(
        name="ch12-instantiation-overhead",
        script_path=REPO_ROOT / "ch12" / "instantiation_overhead_demo.py",
        description="Chapter 12 CUDA graph instantiation overhead demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch12.instantiation_overhead_demo",
    ),
    "ch02-memory-transfer-pcie": DemoSpec(
        name="ch02-memory-transfer-pcie",
        script_path=REPO_ROOT / "ch02" / "memory_transfer_pcie_demo.py",
        description="Chapter 2 PCIe memory transfer binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch02-memory-transfer-nvlink": DemoSpec(
        name="ch02-memory-transfer-nvlink",
        script_path=REPO_ROOT / "ch02" / "memory_transfer_nvlink_demo.py",
        description="Chapter 2 NVLink-C2C memory transfer binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch02-memory-transfer-zero-copy": DemoSpec(
        name="ch02-memory-transfer-zero-copy",
        script_path=REPO_ROOT / "ch02" / "memory_transfer_zero_copy_demo.py",
        description="Chapter 2 zero-copy memory transfer binary demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch03-green-context": DemoSpec(
        name="ch03-green-context",
        script_path=REPO_ROOT / "ch03" / "green_context_demo.py",
        description="Chapter 3 CUDA 13 green-context demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch07-async-prefetch-2d": DemoSpec(
        name="ch07-async-prefetch-2d",
        script_path=REPO_ROOT / "ch07" / "async_prefetch_2d_demo.py",
        description="Chapter 7 async prefetch 2D TMA demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch13-fp8-perchannel": DemoSpec(
        name="ch13-fp8-perchannel",
        script_path=REPO_ROOT / "ch13" / "fp8_perchannel_demo.py",
        description="Chapter 13 FP8 per-channel scaling demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch15-tensor-parallel": DemoSpec(
        name="ch15-tensor-parallel",
        script_path=REPO_ROOT / "ch15" / "tensor_parallel_demo.py",
        description="Chapter 15 tensor-parallel demo (torchrun required).",
        launch_via=LaunchVia.TORCHRUN,
        module_name="ch15.tensor_parallel_demo",
    ),
    "ch15-pipeline-parallel": DemoSpec(
        name="ch15-pipeline-parallel",
        script_path=REPO_ROOT / "ch15" / "pipeline_parallel_demo.py",
        description="Chapter 15 pipeline-parallel demo (torchrun required).",
        launch_via=LaunchVia.TORCHRUN,
        module_name="ch15.pipeline_parallel_demo",
    ),
    "ch15-expert-parallel": DemoSpec(
        name="ch15-expert-parallel",
        script_path=REPO_ROOT / "ch15" / "expert_parallel_demo.py",
        description="Chapter 15 expert-parallel demo (torchrun optional; local mode supported).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch15.expert_parallel_demo",
    ),
    "ch15-context-parallel": DemoSpec(
        name="ch15-context-parallel",
        script_path=REPO_ROOT / "ch15" / "context_parallel_demo.py",
        description="Chapter 15 context-parallel demo (torchrun required).",
        launch_via=LaunchVia.TORCHRUN,
        module_name="ch15.context_parallel_demo",
    ),
    "ch15-speculative-decode": DemoSpec(
        name="ch15-speculative-decode",
        script_path=REPO_ROOT / "ch15" / "speculative_decode_demo.py",
        description="Chapter 15 speculative decoding demo runner (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
        module_name="ch15.speculative_decode_demo",
    ),
    "ch18-v1-engine-loop": DemoSpec(
        name="ch18-v1-engine-loop",
        script_path=REPO_ROOT / "ch18" / "v1_engine_loop.py",
        description="Chapter 18 V1 EngineCore polling-loop demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch17-moe-router-uniform": DemoSpec(
        name="ch17-moe-router-uniform",
        script_path=REPO_ROOT / "ch17" / "moe_router_uniform_demo.py",
        description="Chapter 17 MoE uniform routing demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "ch17-moe-router-topology": DemoSpec(
        name="ch17-moe-router-topology",
        script_path=REPO_ROOT / "ch17" / "moe_router_topology_demo.py",
        description="Chapter 17 MoE topology-aware routing demo (non-benchmark).",
        launch_via=LaunchVia.PYTHON,
    ),
    "labs-decode-multigpu": DemoSpec(
        name="labs-decode-multigpu",
        script_path=REPO_ROOT / "labs" / "decode_optimization" / "decode_multigpu_demo.py",
        description="Lab decode optimization: multi-GPU NVLink-C2C decode stress demo (torchrun required).",
        launch_via=LaunchVia.TORCHRUN,
    ),
}


def _run_demo(
    demo: str,
    demo_args: Optional[List[str]],
    *,
    launch_via: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
) -> int:
    spec = DEMOS.get(demo)
    if spec is None:
        raise ValueError(f"Unknown demo '{demo}'.")
    if not spec.script_path.exists():
        raise FileNotFoundError(f"Demo script not found at {spec.script_path}")

    requested_launch = (launch_via or "").strip().lower()
    resolved_launch = (launch_via or spec.launch_via.value).strip().lower()
    env = build_repo_python_env(REPO_ROOT)

    if spec.launch_via == LaunchVia.TORCHRUN and requested_launch == LaunchVia.PYTHON.value:
        raise ValueError(f"Demo '{demo}' requires torchrun; cannot run with --launch-via=python.")

    if resolved_launch == LaunchVia.PYTHON.value:
        cmd = build_python_entry_command(
            module_name=spec.module_name,
            script_path=None if spec.module_name else spec.script_path,
            argv=demo_args or [],
        )
    elif resolved_launch == LaunchVia.TORCHRUN.value:
        if nproc_per_node is None:
            raise ValueError(
                f"Demo '{demo}' requires torchrun; pass --nproc-per-node <N> (e.g., --nproc-per-node 2)."
            )
        if int(nproc_per_node) <= 0:
            raise ValueError(f"--nproc-per-node must be >= 1, got {nproc_per_node}")

        if nnodes is not None and int(nnodes) <= 0:
            raise ValueError(f"--nnodes must be >= 1, got {nnodes}")
        cmd = build_torchrun_entry_command(
            module_name=spec.module_name,
            script_path=None if spec.module_name else spec.script_path,
            argv=demo_args or [],
            nproc_per_node=int(nproc_per_node),
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
        )
    else:
        raise ValueError(
            f"Invalid --launch-via '{launch_via}'. Expected '{LaunchVia.PYTHON.value}' or '{LaunchVia.TORCHRUN.value}'."
        )
    result = subprocess.run(cmd, env=env)
    return int(result.returncode)


if TYPER_AVAILABLE:
    app = typer.Typer(
        name="demos",
        help="Demos: run runnable examples (non-benchmark)",
        add_completion=False,
    )

    @app.command("list", help="List available demos.")
    def list_demos() -> None:
        for name in sorted(DEMOS):
            typer.echo(f"{name}: {DEMOS[name].description}")

    def _make_demo_command(demo_name: str, description: str):
        def _cmd(
            demo_args: Optional[List[str]] = Argument(
                None,
                help="Arguments forwarded to the demo (use -- to separate).",
            ),
            launch_via: Optional[str] = typer.Option(
                None,
                "--launch-via",
                help="Launcher override: python or torchrun (default depends on the demo).",
            ),
            nproc_per_node: Optional[int] = typer.Option(
                None,
                "--nproc-per-node",
                help="torchrun --nproc_per_node value (required for torchrun demos).",
            ),
            nnodes: Optional[int] = typer.Option(
                None,
                "--nnodes",
                help="torchrun --nnodes value (optional).",
            ),
            rdzv_backend: Optional[str] = typer.Option(
                None,
                "--rdzv-backend",
                help="torchrun --rdzv_backend value (optional).",
            ),
            rdzv_endpoint: Optional[str] = typer.Option(
                None,
                "--rdzv-endpoint",
                help="torchrun --rdzv_endpoint value (optional).",
            ),
        ) -> None:
            try:
                exit_code = _run_demo(
                    demo_name,
                    demo_args,
                    launch_via=launch_via,
                    nproc_per_node=nproc_per_node,
                    nnodes=nnodes,
                    rdzv_backend=rdzv_backend,
                    rdzv_endpoint=rdzv_endpoint,
                )
            except (ValueError, FileNotFoundError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1)
            raise typer.Exit(code=exit_code)

        _cmd.__name__ = demo_name.replace("-", "_")
        _cmd.__doc__ = description
        return _cmd

    for _name, _spec in sorted(DEMOS.items()):
        app.command(_name, help=_spec.description)(_make_demo_command(_name, _spec.description))
else:
    app = None  # type: ignore
