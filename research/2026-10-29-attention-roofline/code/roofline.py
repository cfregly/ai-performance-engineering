#!/usr/bin/env python3
"""Derive one sequence's partial-buffer bound for one attention layer on one device."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Iterator


MIB = 1 << 20
FP32_BYTES = 4
DECIMAL_TB = 10**12


def require_integer(name: str, value: int, minimum: int = 0) -> None:
    """Reject booleans, fractional values, and values below the given minimum."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class AttentionSpec:
    """Describe allocated synthetic partials, without defining an attention kernel.

    Every query head produces one distinct full key-tile dK/dV partial for each
    causal query/key tile pair. Edge key tiles are padded to ``tile_rows``.
    """

    sequence_length: int = 4096
    query_heads: int = 16
    kv_heads: int = 1
    key_dimension: int = 128
    value_dimension: int = 128
    tile_rows: int = 128

    def __post_init__(self) -> None:
        require_integer("sequence_length", self.sequence_length)
        for name in (
            "query_heads",
            "kv_heads",
            "key_dimension",
            "value_dimension",
            "tile_rows",
        ):
            require_integer(name, getattr(self, name), 1)
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")

    @property
    def tiles(self) -> int:
        return (self.sequence_length + self.tile_rows - 1) // self.tile_rows

    @property
    def causal_tile_pairs(self) -> int:
        return self.query_heads * self.tiles * (self.tiles + 1) // 2

    @property
    def partial_bytes_per_pair(self) -> int:
        return self.tile_rows * (self.key_dimension + self.value_dimension) * FP32_BYTES

    @property
    def total_partial_bytes(self) -> int:
        return self.causal_tile_pairs * self.partial_bytes_per_pair

    @property
    def final_dk_dv_bytes(self) -> int:
        """Return logical FP32 final-output bytes, without assigning residency."""
        return (
            self.kv_heads
            * self.sequence_length
            * (self.key_dimension + self.value_dimension)
            * FP32_BYTES
        )


def traffic_floor_bytes(partial_bytes: int, retained_bytes: int) -> int:
    """Count at least one off-chip write and read for bytes not retained.

    This assumes all partials are complete before a separate reducer starts.
    Retention is the total possible on-chip capacity across that boundary.
    Compression, producer/reducer streaming, and recomputation are excluded.
    """
    require_integer("partial_bytes", partial_bytes)
    require_integer("retained_bytes", retained_bytes)
    return 2 * max(0, partial_bytes - retained_bytes)


def parse_bandwidth(value: str | int | Decimal) -> Decimal:
    """Read a positive finite bandwidth in decimal TB/s."""
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("bandwidth must be a positive finite decimal value")
    try:
        bandwidth = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("bandwidth must be a positive finite decimal value") from exc
    if not bandwidth.is_finite() or bandwidth <= 0:
        raise ValueError("bandwidth must be a positive finite decimal value")
    return bandwidth


def time_floor_ms(traffic_bytes: int, bandwidth_tb_per_second: str | int | Decimal) -> Decimal:
    """Convert byte traffic to milliseconds at a hypothetical payload bandwidth."""
    require_integer("traffic_bytes", traffic_bytes)
    bandwidth = parse_bandwidth(bandwidth_tb_per_second)
    with localcontext() as context:
        context.prec = 50
        return Decimal(traffic_bytes) * 1000 / (bandwidth * DECIMAL_TB)


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def batch_partial_bytes(spec: AttentionSpec, query_tiles_per_batch: int) -> list[int]:
    """Return scratch allocation bytes for consecutive query-tile batches.

    Each batch produces its partials and then reduces them. This changes the
    full-producer-then-reducer contract whenever more than one batch is needed.
    Other allocations and output-accumulation traffic are outside this model.
    """
    require_integer("query_tiles_per_batch", query_tiles_per_batch, 1)
    result = []
    for start in range(0, spec.tiles, query_tiles_per_batch):
        stop = min(spec.tiles, start + query_tiles_per_batch)
        pairs_per_head = (stop - start) * (start + stop + 1) // 2
        result.append(pairs_per_head * spec.query_heads * spec.partial_bytes_per_pair)
    return result


ToyScalar = int | float
ToyPartial = tuple[int, int, tuple[tuple[ToyScalar, ...], ...]]


def iter_toy_partials(spec: AttentionSpec, floating: bool) -> Iterator[ToyPartial]:
    """Yield tiny artificial contributions with deterministic values.

    Values have no attention-derivative meaning. Floating values are dyadic
    fractions so this tiny example has exact representable sums. Padded rows
    contain zeros and are omitted from final outputs.
    """
    if spec.sequence_length > 16 or spec.query_heads > 8:
        raise ValueError("the toy is restricted to sequence_length <= 16 and query_heads <= 8")
    if spec.tile_rows > 16 or spec.key_dimension + spec.value_dimension > 16:
        raise ValueError("the toy is restricted to tile_rows <= 16 and combined dimension <= 16")
    group_size = spec.query_heads // spec.kv_heads
    for query_head in range(spec.query_heads):
        kv_head = query_head // group_size
        for query_tile in range(spec.tiles):
            for key_tile in range(query_tile + 1):
                rows = []
                for row in range(spec.tile_rows):
                    position = key_tile * spec.tile_rows + row
                    features = []
                    for feature in range(spec.key_dimension + spec.value_dimension):
                        value = (
                            (query_head + 1) * 11
                            + (query_tile + 1) * 7
                            + (key_tile + 1) * 5
                            + (row + 1) * 3
                            + feature
                        ) % 17 - 8
                        if position >= spec.sequence_length:
                            value = 0
                        features.append(value / 8.0 if floating else value)
                    rows.append(tuple(features))
                yield kv_head, key_tile, tuple(rows)


def reduce_toy(spec: AttentionSpec, partials: Iterator[ToyPartial], floating: bool) -> list:
    """Add contributions in their original order into logical dK/dV outputs."""
    zero = 0.0 if floating else 0
    output = [
        [
            [zero for _ in range(spec.key_dimension + spec.value_dimension)]
            for _ in range(spec.sequence_length)
        ]
        for _ in range(spec.kv_heads)
    ]
    for kv_head, key_tile, rows in partials:
        for row, features in enumerate(rows):
            position = key_tile * spec.tile_rows + row
            if position < spec.sequence_length:
                for feature, value in enumerate(features):
                    output[kv_head][position][feature] += value
    return output


def toy_comparison() -> dict:
    """Check local CPU arithmetic equivalence without collecting timings."""
    spec = AttentionSpec(5, 4, 2, 2, 1, 2)
    results = {}
    for floating in (False, True):
        materialized = list(iter_toy_partials(spec, floating))
        after_materialization = reduce_toy(spec, iter(materialized), floating)
        streamed = reduce_toy(spec, iter_toy_partials(spec, floating), floating)
        if after_materialization != streamed:
            raise AssertionError("toy reductions differ")
        flattened = [value for head in streamed for row in head for value in row]
        results["dyadic_float" if floating else "integer"] = {
            "equal": True,
            "partial_count": len(materialized),
            "final_scalar_count": len(flattened),
            "final_sum": sum(flattened),
            "final_values": streamed,
        }
    return {
        "workload_scope": "one sequence (batch size 1), one attention layer backward, one device",
        "specification": asdict(spec),
        "purpose": "CPU arithmetic illustration with artificial contributions, not an attention kernel",
        "timings_collected": False,
        "reduction_order": "identical in both paths",
        "float_values": "multiples of 1/8 with exactly representable sums in this tiny example",
        "generally_reordered_floating_point_reductions": "may round differently",
        "comparisons": results,
    }


def build_results() -> dict:
    spec = AttentionSpec()
    retained = 128 * MIB
    traffic = traffic_floor_bytes(spec.total_partial_bytes, retained)
    bandwidths = ("8", "6.4")
    cache_rows = []
    for capacity_mib in (0, 16, 32, 64, 128, 256, 512, 1024, 1056, 2048):
        capacity = capacity_mib * MIB
        row = {
            "retained_capacity_mib": capacity_mib,
            "retained_capacity_bytes": capacity,
            "partial_hbm_traffic_floor_bytes": traffic_floor_bytes(spec.total_partial_bytes, capacity),
        }
        for bandwidth in bandwidths:
            row[f"floor_ms_at_{bandwidth}_decimal_tb_per_s"] = decimal_text(
                time_floor_ms(row["partial_hbm_traffic_floor_bytes"], bandwidth)
            )
        cache_rows.append(row)
    batch_rows = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        sizes = batch_partial_bytes(spec, batch_size)
        batch_rows.append(
            {
                "query_tiles_per_batch": batch_size,
                "producer_then_reducer_batches": len(sizes),
                "changes_original_phase_contract": len(sizes) > 1,
                "maximum_partial_scratch_bytes": max(sizes, default=0),
                "maximum_partial_scratch_mib": decimal_text(Decimal(max(sizes, default=0)) / MIB),
                "partial_hbm_traffic_floor_bytes_at_128_mib_retention": sum(
                    traffic_floor_bytes(size, retained) for size in sizes
                ),
                "logical_final_dk_dv_bytes_residency_unspecified": spec.final_dk_dv_bytes,
            }
        )
    return {
        "schema_version": 1,
        "kind": "independently derived synthetic analytical example",
        "measurements": {"gpu_runs": 0, "gpu_timings": [], "cpu_timings": []},
        "specification": asdict(spec),
        "representation": {
            "partial_dtype": "FP32",
            "bytes_per_scalar": FP32_BYTES,
            "edge_key_tiles": "padded to tile_rows",
            "partial_ownership": "unique query-head, query-tile, key-tile",
            "causal_diagonal_tiles": "one full key-tile dK/dV partial, despite masked score elements",
            "head_mapping": "consecutive query-head groups share one KV head",
        },
        "phase_contract": {
            "all_producer_partials_complete_before_separate_reducer": True,
            "maximum_on_chip_retention_bytes_between_phases": retained,
            "compression_allowed": False,
            "producer_reducer_streaming_allowed": False,
            "recomputation_allowed": False,
        },
        "derivation": {
            "tiles": spec.tiles,
            "tile_pairs_per_query_head": spec.tiles * (spec.tiles + 1) // 2,
            "causal_tile_pairs_all_query_heads": spec.causal_tile_pairs,
            "partial_bytes_per_pair": spec.partial_bytes_per_pair,
            "total_partial_bytes": spec.total_partial_bytes,
            "total_partial_mib": decimal_text(Decimal(spec.total_partial_bytes) / MIB),
            "not_retained_bytes_at_phase_boundary": max(0, spec.total_partial_bytes - retained),
            "partial_hbm_traffic_floor_bytes": traffic,
            "traffic_formula": "2 * max(0, total_partial_bytes - retained_capacity_bytes)",
            "time_formula_ms": "traffic_bytes * 1000 / (decimal_TB_per_second * 1000000000000)",
            "hypothetical_bandwidth_floors_ms": {
                bandwidth: decimal_text(time_floor_ms(traffic, bandwidth)) for bandwidth in bandwidths
            },
        },
        "assumptions": [
            "Counts are for one sequence (batch size 1) in one attention layer backward on one device",
            "N counts both query and key/value positions, with aligned causal tiles",
            "Each represented partial byte is independently necessary for the chosen reduction",
            "At most C bytes of these partials can survive on chip across the complete phase boundary",
            "Every unretained partial byte must cross the HBM boundary once in each direction",
            "Bandwidth is a hypothetical aggregate payload ceiling shared by reads and writes",
            "MiB means 1048576 bytes and TB/s means 1000000000000 bytes per second",
        ],
        "excluded_costs": [
            "later writeback of dead partials that remain dirty in a write-back cache",
            "Q/K/V and upstream-gradient reads",
            "attention arithmetic, masks, normalization, and recomputation choices",
            "final dQ/dK/dV writes and any accumulation reads or writes",
            "reducer arithmetic, launch, scheduling, synchronization, and locality costs",
            "other live allocations and competition for cache or shared memory",
            "transaction granularity and bandwidth efficiency",
        ],
        "limitations": [
            "This is a bound for the stated representation and phase contract, not all attention algorithms",
            "The bandwidth values are hypothetical and are not measured hardware performance",
            "This code is not a runnable attention kernel and gives no model or serving speedup",
            "A zero partial-spill lower bound does not prove that a complete attention implementation fits on chip",
            "Batching changes the phase contract and adds unestimated accumulation and scheduling costs",
            "No performance comparison is possible from these calculations alone",
        ],
        "target_screen": {
            "illustrative_target_ms": "0.200",
            "floor_exceeds_target": {
                bandwidth: time_floor_ms(traffic, bandwidth) > Decimal("0.200")
                for bandwidth in bandwidths
            },
            "retention_at_equal_floor_bytes": {
                bandwidth: int(
                    Decimal(spec.total_partial_bytes)
                    - Decimal("0.200") * Decimal(bandwidth) * DECIMAL_TB / 2000
                ) for bandwidth in bandwidths
            },
            "bandwidth_at_equal_floor_decimal_tb_per_s": decimal_text(
                Decimal(traffic) * 1000 / (Decimal("0.200") * DECIMAL_TB)
            ),
            "scope": "A floor above the target rejects this contract. A floor below it proves no attainable runtime.",
        },
        "cache_sensitivity": cache_rows,
        "reduction_batch_sensitivity": batch_rows,
        "toy_correctness_illustration": toy_comparison(),
    }


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def rendered_outputs() -> dict[str, str]:
    results = build_results()
    return {
        "RESULTS.json": json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "cache_sensitivity.csv": csv_text(results["cache_sensitivity"]),
        "reduction_batch_sensitivity.csv": csv_text(results["reduction_batch_sensitivity"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Write deterministic JSON and sensitivity CSV files")
    parser.add_argument("--check", action="store_true", help="Check existing outputs without changing them")
    args = parser.parse_args()
    if args.check and args.output_dir is None:
        parser.error("--check requires --output-dir")
    outputs = rendered_outputs()
    if args.output_dir is None:
        print(outputs["RESULTS.json"], end="")
        return
    if args.check:
        for filename, expected in outputs.items():
            path = args.output_dir / filename
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Reproduction check failed: {path}")
        print(f"Reproduction check passed for {len(outputs)} files")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        (args.output_dir / filename).write_text(content, encoding="utf-8")
    print(f"Wrote {len(outputs)} deterministic files to {args.output_dir}")


if __name__ == "__main__":
    main()
