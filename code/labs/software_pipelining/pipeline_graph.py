"""Deterministic schedule graphs for the software-pipelining lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Literal


DependencyKind = Literal["same_iteration", "loop_carried", "anti_dependency"]
PhaseName = Literal["prologue", "steady", "epilogue"]


@dataclass(frozen=True)
class PipelineNode:
    name: str
    unit: str
    latency: int
    description: str


@dataclass(frozen=True)
class DependencyEdge:
    src: str
    dst: str
    kind: DependencyKind
    distance: int
    min_delay: int
    description: str


@dataclass(frozen=True)
class ScheduleSlot:
    node: str
    iteration: int
    time: int
    stage: int
    phase: PhaseName


@dataclass(frozen=True)
class PipelineExample:
    name: str
    description: str
    stage_count: int
    iteration_count: int
    nodes: tuple[PipelineNode, ...]
    edges: tuple[DependencyEdge, ...]
    schedule: tuple[ScheduleSlot, ...]


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...]


def _node_map(example: PipelineExample) -> Dict[str, PipelineNode]:
    return {node.name: node for node in example.nodes}


def _slot_map(example: PipelineExample) -> Dict[tuple[str, int], ScheduleSlot]:
    return {(slot.node, slot.iteration): slot for slot in example.schedule}


def validate_schedule(example: PipelineExample) -> ValidationResult:
    """Validate resource conflicts and dependency timing for a manual schedule."""

    errors: list[str] = []
    nodes = _node_map(example)
    slots = _slot_map(example)

    expected_keys = {
        (node.name, iteration)
        for node in example.nodes
        for iteration in range(example.iteration_count)
    }
    actual_keys = set(slots.keys())

    missing = sorted(expected_keys - actual_keys)
    duplicates = len(actual_keys) != len(example.schedule)
    if missing:
        errors.append(f"Missing schedule slots: {missing}")
    if duplicates:
        errors.append("Schedule contains duplicate (node, iteration) entries.")

    occupancy: dict[tuple[str, int], str] = {}
    for slot in example.schedule:
        node = nodes.get(slot.node)
        if node is None:
            errors.append(f"Unknown node in schedule: {slot.node}")
            continue
        if slot.iteration < 0 or slot.iteration >= example.iteration_count:
            errors.append(f"Invalid iteration for {slot.node}: {slot.iteration}")
            continue
        if slot.stage < 0 or slot.stage >= example.stage_count:
            errors.append(
                f"Invalid stage for {slot.node}[{slot.iteration}]: "
                f"{slot.stage} not in [0, {example.stage_count})"
            )
        if slot.time < 0:
            errors.append(f"Invalid time for {slot.node}[{slot.iteration}]: {slot.time}")
        for cycle in range(slot.time, slot.time + node.latency):
            key = (node.unit, cycle)
            previous = occupancy.get(key)
            if previous is not None:
                errors.append(
                    f"Unit conflict on {node.unit} at t={cycle}: "
                    f"{previous} overlaps {slot.node}[{slot.iteration}]"
                )
            else:
                occupancy[key] = f"{slot.node}[{slot.iteration}]"

    for edge in example.edges:
        if edge.src not in nodes or edge.dst not in nodes:
            errors.append(f"Edge references unknown nodes: {edge}")
            continue
        for src_iteration in range(example.iteration_count):
            dst_iteration = src_iteration + edge.distance
            if dst_iteration < 0 or dst_iteration >= example.iteration_count:
                continue
            src_slot = slots.get((edge.src, src_iteration))
            dst_slot = slots.get((edge.dst, dst_iteration))
            if src_slot is None or dst_slot is None:
                continue
            required_time = src_slot.time + edge.min_delay
            if dst_slot.time < required_time:
                errors.append(
                    f"{edge.kind} violation: {edge.src}[{src_iteration}] at t={src_slot.time} "
                    f"requires {edge.dst}[{dst_iteration}] >= t={required_time}, got t={dst_slot.time}"
                )

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def list_pipeline_examples() -> tuple[str, ...]:
    return tuple(_EXAMPLES.keys())


def get_pipeline_example(name: str) -> PipelineExample:
    try:
        return _EXAMPLES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline example {name!r}. Choices: {sorted(_EXAMPLES)}") from exc


def pipeline_example_to_dict(example: PipelineExample) -> dict:
    validation = validate_schedule(example)
    node_units = {node.name: node.unit for node in example.nodes}
    return {
        "example": example.name,
        "description": example.description,
        "stage_count": example.stage_count,
        "iteration_count": example.iteration_count,
        "nodes": [asdict(node) for node in example.nodes],
        "edges": [asdict(edge) for edge in example.edges],
        "schedule": [
            asdict(slot)
            for slot in sorted(
                example.schedule,
                key=lambda slot: (
                    slot.time,
                    node_units[slot.node],
                    slot.node,
                    slot.iteration,
                ),
            )
        ],
        "validation": asdict(validation),
    }


_EXAMPLES: Dict[str, PipelineExample] = {
    "gemm_mainloop": PipelineExample(
        name="gemm_mainloop",
        description=(
            "Two-stage tiled GEMM mainloop: one load stage feeds one MMA stage, with a small "
            "epilogue and an anti-dependency that protects stage reuse."
        ),
        stage_count=2,
        iteration_count=4,
        nodes=(
            PipelineNode("tma_load", "load", 1, "Load one A/B tile pair into the current stage buffer."),
            PipelineNode("mma", "tc", 1, "Consume the staged tile on tensor cores."),
            PipelineNode("epilogue", "cuda", 1, "Write the accumulator fragment back to global memory."),
        ),
        edges=(
            DependencyEdge("tma_load", "mma", "same_iteration", 0, 1, "Need the tile in shared memory before MMA starts."),
            DependencyEdge("mma", "epilogue", "same_iteration", 0, 1, "Epilogue can only run after MMA finishes."),
            DependencyEdge(
                "mma",
                "tma_load",
                "anti_dependency",
                2,
                1,
                "Do not overwrite a two-stage ring-buffer slot until the matching MMA has consumed it.",
            ),
        ),
        schedule=(
            ScheduleSlot("tma_load", 0, 0, 0, "prologue"),
            ScheduleSlot("tma_load", 1, 1, 1, "prologue"),
            ScheduleSlot("mma", 0, 1, 0, "steady"),
            ScheduleSlot("tma_load", 2, 2, 0, "steady"),
            ScheduleSlot("mma", 1, 2, 1, "steady"),
            ScheduleSlot("epilogue", 0, 2, 0, "steady"),
            ScheduleSlot("tma_load", 3, 3, 1, "steady"),
            ScheduleSlot("mma", 2, 3, 0, "steady"),
            ScheduleSlot("epilogue", 1, 3, 1, "steady"),
            ScheduleSlot("mma", 3, 4, 1, "epilogue"),
            ScheduleSlot("epilogue", 2, 4, 0, "epilogue"),
            ScheduleSlot("epilogue", 3, 5, 1, "epilogue"),
        ),
    ),
    "fa_like_inner_loop": PipelineExample(
        name="fa_like_inner_loop",
        description=(
            "A simplified FlashAttention-style inner loop with staged K/V loads, a QK MMA, "
            "scalar softmax/fixup work, a PV MMA, and explicit loop-carried/anti-dependency edges."
        ),
        stage_count=3,
        iteration_count=4,
        nodes=(
            PipelineNode("load_kv", "load", 1, "Load one K/V tile pair into staged shared memory."),
            PipelineNode("qk_mma", "tc_qk", 1, "Compute QK^T scores for the current tile."),
            PipelineNode("softmax_fixup", "cuda", 1, "Rowmax, exponentiation, and running-softmax fixup."),
            PipelineNode("pv_mma", "tc_pv", 1, "Multiply normalized probabilities by the staged V tile."),
        ),
        edges=(
            DependencyEdge("load_kv", "qk_mma", "same_iteration", 0, 1, "Cannot start the QK matmul before the K/V tile lands."),
            DependencyEdge("qk_mma", "softmax_fixup", "same_iteration", 0, 1, "Softmax/scalar fixup needs the current score tile."),
            DependencyEdge("softmax_fixup", "pv_mma", "same_iteration", 0, 1, "PV matmul uses the normalized probability tile."),
            DependencyEdge(
                "softmax_fixup",
                "softmax_fixup",
                "loop_carried",
                1,
                1,
                "Online softmax carries running max/sum state into the next iteration.",
            ),
            DependencyEdge(
                "pv_mma",
                "load_kv",
                "anti_dependency",
                3,
                1,
                "A stage cannot be reused for a later load until the prior PV matmul is done with that tile.",
            ),
        ),
        schedule=(
            ScheduleSlot("load_kv", 0, 0, 0, "prologue"),
            ScheduleSlot("load_kv", 1, 1, 1, "prologue"),
            ScheduleSlot("qk_mma", 0, 1, 0, "prologue"),
            ScheduleSlot("qk_mma", 1, 2, 1, "steady"),
            ScheduleSlot("softmax_fixup", 0, 2, 0, "steady"),
            ScheduleSlot("softmax_fixup", 1, 3, 1, "steady"),
            ScheduleSlot("pv_mma", 0, 3, 0, "steady"),
            ScheduleSlot("load_kv", 2, 4, 2, "steady"),
            ScheduleSlot("pv_mma", 1, 4, 1, "steady"),
            ScheduleSlot("load_kv", 3, 5, 0, "steady"),
            ScheduleSlot("qk_mma", 2, 5, 2, "steady"),
            ScheduleSlot("qk_mma", 3, 6, 0, "steady"),
            ScheduleSlot("softmax_fixup", 2, 6, 2, "steady"),
            ScheduleSlot("softmax_fixup", 3, 7, 0, "epilogue"),
            ScheduleSlot("pv_mma", 2, 7, 2, "epilogue"),
            ScheduleSlot("pv_mma", 3, 8, 0, "epilogue"),
        ),
    ),
}
