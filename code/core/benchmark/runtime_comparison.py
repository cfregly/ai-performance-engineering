"""Fail-closed admission for runtime parity between executed benchmark runs."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from core.benchmark.models import BenchmarkRun
from core.benchmark.run_manifest import (
    RunManifest,
    RuntimeParityTarget,
    RuntimeProvenance,
    RuntimeProvenanceParity,
    compare_runtime_provenance,
)

RunSide = Literal["reference", "candidate"]


class RuntimeIntegrityFailure(BaseModel):
    """One reason an executed runtime receipt cannot be admitted."""

    code: str
    detail: str
    run: RunSide | None = None
    local_rank: int | None = None
    schemaVersion: str = "1.0"  # noqa: N815 - repository schema convention


class ExecutedRuntimeComparison(BaseModel):
    """Combined transport-integrity and runtime-version parity verdict."""

    matches: bool
    target: RuntimeParityTarget | None = None
    runtime_parity: RuntimeProvenanceParity | None = None
    integrity_failures: list[RuntimeIntegrityFailure] = Field(default_factory=list)
    schemaVersion: str = "1.0"  # noqa: N815 - repository schema convention


class ExecutedRuntimeComparisonError(RuntimeError):
    """Raised after a structured executed-runtime rejection has been retained."""

    def __init__(self, comparison: ExecutedRuntimeComparison) -> None:
        self.comparison = comparison
        details = [failure.code for failure in comparison.integrity_failures]
        if comparison.runtime_parity is not None:
            details.extend(
                f"runtime_mismatch:{field_name}"
                for field_name in comparison.runtime_parity.mismatched_fields
            )
            details.extend(
                f"runtime_unknown:{field_name}"
                for field_name in comparison.runtime_parity.unknown_fields
            )
        if not details:
            details.append("comparison_unavailable")
        super().__init__(
            "EXECUTED RUNTIME COMPARISON FAILED: " + ", ".join(details)
        )


def _failure(
    failures: list[RuntimeIntegrityFailure],
    code: str,
    detail: str,
    *,
    run: RunSide | None = None,
    local_rank: int | None = None,
) -> None:
    failures.append(
        RuntimeIntegrityFailure(
            code=code,
            detail=detail,
            run=run,
            local_rank=local_rank,
        )
    )


def _render_rank_keys(ranks: set[Any]) -> str:
    """Render possibly-invalid rank keys deterministically without comparing unlike types."""

    return "[" + ", ".join(sorted(repr(rank) for rank in ranks)) + "]"


def _validated_target(
    value: str | None,
    *,
    run: RunSide,
    failures: list[RuntimeIntegrityFailure],
) -> RuntimeParityTarget | None:
    if value is None or not str(value).strip():
        _failure(
            failures,
            "device_missing",
            "The executed benchmark result did not retain a device identity.",
            run=run,
        )
        return None
    try:
        target = torch.device(value).type
    except (TypeError, RuntimeError, ValueError) as exc:
        _failure(
            failures,
            "device_invalid",
            f"The executed benchmark device {value!r} is invalid: {exc}",
            run=run,
        )
        return None
    if target not in {"cpu", "cuda"}:
        _failure(
            failures,
            "device_unsupported",
            f"Runtime comparison supports only CPU or CUDA results, got {value!r}.",
            run=run,
        )
        return None
    return target


def _timing_failures(
    run: BenchmarkRun,
    *,
    side: RunSide,
    failures: list[RuntimeIntegrityFailure],
) -> None:
    result = run.result
    if result.errors:
        rendered = "; ".join(str(error) for error in result.errors[:3])
        if len(result.errors) > 3:
            rendered += f"; and {len(result.errors) - 3} more"
        _failure(
            failures,
            "result_errors",
            f"The benchmark result retained execution errors: {rendered}",
            run=side,
        )

    timing = result.timing
    if isinstance(timing.iterations, bool) or timing.iterations <= 0:
        _failure(
            failures,
            "timing_iterations_invalid",
            f"Timing iterations must be positive, got {timing.iterations!r}.",
            run=side,
        )

    positive_values: dict[str, Any] = {
        "mean_ms": timing.mean_ms,
        "median_ms": timing.median_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
    }
    for name in ("p50_ms", "p90_ms", "p95_ms", "p99_ms"):
        value = getattr(timing, name)
        if value is not None:
            positive_values[name] = value
    positive_values.update(
        {f"percentiles[{key!r}]": value for key, value in timing.percentiles.items()}
    )
    positive_values.update(
        {f"raw_times_ms[{index}]": value for index, value in enumerate(timing.raw_times_ms or [])}
    )

    invalid_values: list[str] = []
    for field_name, value in positive_values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_values.append(f"{field_name}={value!r}")
            continue
        if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
            invalid_values.append(f"{field_name}={value!r}")
    try:
        std_ms = float(timing.std_ms)
    except (TypeError, ValueError):
        invalid_values.append(f"std_ms={timing.std_ms!r}")
    else:
        if isinstance(timing.std_ms, bool) or not math.isfinite(std_ms) or std_ms < 0:
            invalid_values.append(f"std_ms={timing.std_ms!r}")

    if invalid_values:
        rendered_invalid_values = ", ".join(invalid_values[:8])
        if len(invalid_values) > 8:
            rendered_invalid_values += f", and {len(invalid_values) - 8} more"
        _failure(
            failures,
            "timing_values_invalid",
            "Timing values must be finite and positive (standard deviation may be zero): "
            + rendered_invalid_values,
            run=side,
        )


def _runtime_receipt_failures(
    run: BenchmarkRun,
    *,
    side: RunSide,
    target: RuntimeParityTarget | None,
    failures: list[RuntimeIntegrityFailure],
) -> None:
    manifest_runtime = run.manifest.runtime_provenance if run.manifest is not None else None
    result_runtime = run.result.runtime_provenance

    if run.manifest is None:
        _failure(
            failures,
            "manifest_missing",
            "The benchmark run did not retain a run manifest.",
            run=side,
        )
    if manifest_runtime is None:
        _failure(
            failures,
            "manifest_runtime_provenance_missing",
            "The run manifest did not retain executing-process runtime provenance.",
            run=side,
        )
    if result_runtime is None:
        _failure(
            failures,
            "result_runtime_provenance_missing",
            "The benchmark result did not retain executing-process runtime provenance.",
            run=side,
        )
    if (
        manifest_runtime is not None
        and result_runtime is not None
        and manifest_runtime != result_runtime
    ):
        _failure(
            failures,
            "manifest_result_runtime_mismatch",
            "Manifest runtime provenance differs from the executing worker result receipt.",
            run=side,
        )

    local_world_size = getattr(run.result, "local_world_size", None)
    expected_local_ranks: set[int] | None = None
    if local_world_size is None:
        _failure(
            failures,
            "local_world_size_missing",
            "The benchmark result did not retain the local worker count passed to the launcher.",
            run=side,
        )
    elif (
        isinstance(local_world_size, bool)
        or not isinstance(local_world_size, int)
        or local_world_size <= 0
    ):
        _failure(
            failures,
            "local_world_size_invalid",
            f"The retained local worker count must be a positive integer, got {local_world_size!r}.",
            run=side,
        )
    else:
        expected_local_ranks = set(range(local_world_size))

    execution_process_ids = getattr(run.result, "execution_process_ids", None)
    if not isinstance(execution_process_ids, dict) or not execution_process_ids:
        _failure(
            failures,
            "execution_process_ids_missing",
            "No independently observed execution-process IDs were retained.",
            run=side,
        )
        execution_process_ids = {}

    invalid_execution_ids = {
        rank: process_id
        for rank, process_id in execution_process_ids.items()
        if isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    }
    if invalid_execution_ids:
        _failure(
            failures,
            "execution_process_ids_invalid",
            f"Execution-process IDs contain invalid rank/PID entries: {invalid_execution_ids!r}.",
            run=side,
        )

    execution_ranks = set(execution_process_ids)
    if expected_local_ranks is not None and execution_ranks != expected_local_ranks:
        _failure(
            failures,
            "execution_process_ids_incomplete",
            "Execution-process ID ranks differ from the retained local worker count "
            f"(expected={_render_rank_keys(expected_local_ranks)}, "
            f"observed={_render_rank_keys(execution_ranks)}).",
            run=side,
        )

    rank_zero_pid = execution_process_ids.get(0)
    if rank_zero_pid is None:
        _failure(
            failures,
            "rank_zero_execution_process_id_missing",
            "The independently observed execution-process IDs omit local rank 0.",
            run=side,
        )
    elif result_runtime is not None and result_runtime.process_id != rank_zero_pid:
        _failure(
            failures,
            "runtime_process_id_mismatch",
            "The primary runtime snapshot PID does not match the independently observed "
            f"local-rank-0 execution PID ({result_runtime.process_id} != {rank_zero_pid}).",
            run=side,
        )

    per_rank = run.result.runtime_provenance_by_local_rank
    observed_ranks = set(per_rank)
    if len(execution_process_ids) > 1 or per_rank:
        if observed_ranks != execution_ranks:
            _failure(
                failures,
                "runtime_rank_receipts_incomplete",
                "Per-rank runtime receipt ranks differ from independently observed execution "
                f"ranks (runtime={_render_rank_keys(observed_ranks)}, "
                f"execution={_render_rank_keys(execution_ranks)}).",
                run=side,
            )
        valid_execution_ranks = {
            rank
            for rank in execution_ranks
            if not isinstance(rank, bool) and isinstance(rank, int) and rank >= 0
        }
        valid_observed_ranks = {
            rank
            for rank in observed_ranks
            if not isinstance(rank, bool) and isinstance(rank, int) and rank >= 0
        }
        for rank in sorted(valid_observed_ranks & valid_execution_ranks):
            runtime_pid = per_rank[rank].process_id
            execution_pid = execution_process_ids[rank]
            if runtime_pid != execution_pid:
                _failure(
                    failures,
                    "runtime_rank_process_id_mismatch",
                    f"Local rank {rank} runtime snapshot PID does not match its independently "
                    f"observed execution PID ({runtime_pid} != {execution_pid}).",
                    run=side,
                    local_rank=rank,
                )
        rank_zero_runtime: RuntimeProvenance | None = per_rank.get(0)
        if (
            rank_zero_runtime is not None
            and result_runtime is not None
            and rank_zero_runtime != result_runtime
        ):
            _failure(
                failures,
                "primary_rank_zero_runtime_mismatch",
                "The primary runtime snapshot differs from the local-rank-0 runtime receipt.",
                run=side,
                local_rank=0,
            )

        if result_runtime is not None and target is not None:
            primary_manifest = RunManifest.model_construct(
                runtime_provenance=result_runtime,
            )
            for rank in sorted(valid_observed_ranks):
                rank_manifest = RunManifest.model_construct(
                    runtime_provenance=per_rank[rank],
                )
                parity = compare_runtime_provenance(
                    primary_manifest,
                    rank_manifest,
                    target=target,
                )
                if parity.mismatched_fields:
                    _failure(
                        failures,
                        "runtime_rank_provenance_mismatch",
                        f"Local rank {rank} runtime provenance differs from the primary "
                        f"runtime for required {target} fields: "
                        f"{', '.join(parity.mismatched_fields)}.",
                        run=side,
                        local_rank=rank,
                    )
                if parity.unknown_fields:
                    _failure(
                        failures,
                        "runtime_rank_provenance_unknown",
                        f"Local rank {rank} runtime provenance is incomplete for required "
                        f"{target} fields: {', '.join(parity.unknown_fields)}.",
                        run=side,
                        local_rank=rank,
                    )


def compare_executed_runtime_provenance(
    reference: BenchmarkRun,
    candidate: BenchmarkRun,
) -> ExecutedRuntimeComparison:
    """Admit a pair only when execution receipts and required versions match."""

    failures: list[RuntimeIntegrityFailure] = []
    _timing_failures(reference, side="reference", failures=failures)
    _timing_failures(candidate, side="candidate", failures=failures)

    reference_target = _validated_target(
        reference.result.device,
        run="reference",
        failures=failures,
    )
    candidate_target = _validated_target(
        candidate.result.device,
        run="candidate",
        failures=failures,
    )
    _runtime_receipt_failures(
        reference,
        side="reference",
        target=reference_target,
        failures=failures,
    )
    _runtime_receipt_failures(
        candidate,
        side="candidate",
        target=candidate_target,
        failures=failures,
    )
    target: RuntimeParityTarget | None = None
    if reference_target is not None and candidate_target is not None:
        if reference_target == candidate_target:
            target = reference_target
        else:
            _failure(
                failures,
                "device_target_mismatch",
                "Reference and candidate executed different device types "
                f"({reference_target} != {candidate_target}).",
            )

    runtime_parity: RuntimeProvenanceParity | None = None
    if target is not None and reference.manifest is not None and candidate.manifest is not None:
        runtime_parity = compare_runtime_provenance(
            reference.manifest,
            candidate.manifest,
            target=target,
        )

    matches = (
        not failures
        and runtime_parity is not None
        and runtime_parity.matches
    )
    return ExecutedRuntimeComparison(
        matches=matches,
        target=target,
        runtime_parity=runtime_parity,
        integrity_failures=failures,
    )
