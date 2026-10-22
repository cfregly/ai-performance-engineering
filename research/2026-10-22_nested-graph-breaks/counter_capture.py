#!/usr/bin/env python3
"""Preserve and observe native Dynamo counter groups without changing counts.

PyTorch 2.9.1 stores Dynamo counters in a ``defaultdict(Counter)``.  Its
inlining path temporarily patches that top-level mapping.  A group first
created inside the patch is removed when ``patch.dict`` restores the shallow
top-level snapshot.  Materializing selected native groups before tracing makes
those same group objects part of the snapshot, so mutations made by PyTorch
survive restoration.

PyTorch removed that inline ``patch.dict`` wrapper in 2.10.  On the selected
2.14.1 rerun runtime, pre-creation is evidence instrumentation rather than a
workaround for the historical 2.9.1 loss path.

This module never clears a mapping, replaces a group, or writes a counter
entry.  It only performs the native missing-key lookup needed to materialize
empty groups and copies observed values for evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final


SCHEMA: Final = "nested-graph-breaks-counter-lifecycle/v1"
DEFAULT_REQUIRED_GROUPS: Final = ("graph_break", "stats")


class CounterCaptureError(RuntimeError):
    """Raised when the counter mapping cannot be observed transparently."""


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _copy_counts(group_name: str, group: Any) -> dict[str, int]:
    if not isinstance(group, Mapping):
        raise CounterCaptureError(
            f"counter group {group_name!r} is not a mapping: {_type_name(group)}"
        )

    copied: dict[str, int] = {}
    for name, value in group.items():
        if not isinstance(name, str):
            raise CounterCaptureError(
                f"counter group {group_name!r} has non-string key {name!r}"
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise CounterCaptureError(
                f"counter {group_name}.{name} has non-integer value {value!r}"
            )
        copied[name] = value
    return copied


def _entry_sum(counts: Mapping[str, int]) -> int:
    return sum(counts.values())


@dataclass(frozen=True)
class _PreparedGroup:
    name: str
    object_ref: MutableMapping[str, int]
    object_id: int
    object_type: str
    present_before_prepare: bool
    raw_before_prepare: dict[str, int]
    raw_after_prepare: dict[str, int]

    def evidence(self) -> dict[str, Any]:
        return {
            "present_before_prepare": self.present_before_prepare,
            "created_by_native_missing_key_lookup": not self.present_before_prepare,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "raw_before_prepare": dict(self.raw_before_prepare),
            "raw_after_prepare": dict(self.raw_after_prepare),
            "entry_sum_before_prepare": _entry_sum(self.raw_before_prepare),
            "entry_sum_after_prepare": _entry_sum(self.raw_after_prepare),
            "entry_delta_from_preparation": (
                _entry_sum(self.raw_after_prepare)
                - _entry_sum(self.raw_before_prepare)
            ),
            "entries_unchanged_by_preparation": (
                self.raw_after_prepare == self.raw_before_prepare
            ),
        }


@dataclass
class CounterLifecycleCapture:
    """Record lifecycle evidence for pre-created native counter groups."""

    _counters: MutableMapping[str, MutableMapping[str, int]]
    _mapping_id: int
    _mapping_type: str
    _prepared: dict[str, _PreparedGroup]
    _top_level_keys_before_prepare: tuple[str, ...]
    _observations: list[dict[str, Any]] = field(default_factory=list)
    _stage_names: set[str] = field(default_factory=set)

    def observe(self, stage: str) -> dict[str, Any]:
        """Copy raw group state without creating a missing group.

        Observation deliberately uses membership and ``get`` rather than
        ``__getitem__``.  A group lost during restoration therefore remains
        visible as missing instead of being silently recreated by defaultdict.
        """

        if not isinstance(stage, str) or not stage.strip():
            raise CounterCaptureError("observation stage must be a non-empty string")
        if stage in self._stage_names:
            raise CounterCaptureError(f"duplicate observation stage: {stage!r}")

        mapping_identity_preserved = id(self._counters) == self._mapping_id
        groups: dict[str, Any] = {}
        for name, prepared in self._prepared.items():
            present = name in self._counters
            current = self._counters.get(name) if present else None
            same_object = present and current is prepared.object_ref
            raw_counts = _copy_counts(name, current) if present else None
            groups[name] = {
                "present": present,
                "same_object_as_prepared": same_object,
                "binding_replaced": present and not same_object,
                "object_id": id(current) if present else None,
                "object_type": _type_name(current) if present else None,
                "raw_counts": raw_counts,
                "entry_sum": _entry_sum(raw_counts) if raw_counts is not None else None,
                "content_changed_since_preparation": (
                    raw_counts != prepared.raw_after_prepare
                    if raw_counts is not None
                    else None
                ),
            }

        observation = {
            "stage": stage,
            "top_level_mapping_identity_preserved": mapping_identity_preserved,
            "top_level_keys": sorted(str(key) for key in self._counters.keys()),
            "groups": groups,
        }
        self._stage_names.add(stage)
        self._observations.append(observation)
        return observation

    def finish(self) -> dict[str, Any]:
        """Return JSON-compatible evidence and lifecycle validity checks."""

        preparation = {
            name: prepared.evidence() for name, prepared in self._prepared.items()
        }
        observations = list(self._observations)
        all_mapping_identities_preserved = all(
            observation["top_level_mapping_identity_preserved"]
            for observation in observations
        )
        all_groups_present = all(
            group["present"]
            for observation in observations
            for group in observation["groups"].values()
        )
        all_group_identities_preserved = all(
            group["same_object_as_prepared"]
            for observation in observations
            for group in observation["groups"].values()
        )
        preparation_added_zero_entries = all(
            item["entries_unchanged_by_preparation"]
            and item["entry_delta_from_preparation"] == 0
            for item in preparation.values()
        )

        return {
            "schema": SCHEMA,
            "required_groups": list(self._prepared),
            "instrumentation_semantics": (
                "native missing-key group materialization plus read-only snapshots; "
                "no counter entry writes, increments, clears, or group replacement"
            ),
            "top_level_mapping": {
                "object_id": self._mapping_id,
                "object_type": self._mapping_type,
                "keys_before_prepare": list(self._top_level_keys_before_prepare),
            },
            "preparation": preparation,
            "observations": observations,
            "invariants": {
                "preparation_added_zero_entries": preparation_added_zero_entries,
                "all_observed_top_level_mapping_identities_preserved": (
                    all_mapping_identities_preserved
                ),
                "all_observed_required_groups_present": all_groups_present,
                "all_observed_required_group_identities_preserved": (
                    all_group_identities_preserved
                ),
                "lifecycle_valid": (
                    bool(observations)
                    and preparation_added_zero_entries
                    and all_mapping_identities_preserved
                    and all_groups_present
                    and all_group_identities_preserved
                ),
            },
        }


def prepare_counter_capture(
    counters: MutableMapping[str, MutableMapping[str, int]],
    required_groups: Sequence[str] = DEFAULT_REQUIRED_GROUPS,
) -> CounterLifecycleCapture:
    """Pre-create native groups and return a read-only lifecycle observer.

    The supplied mapping must create its native group on missing-key access,
    as ``torch._dynamo.utils.counters`` does.  A plain dict with an absent
    required group is rejected rather than populated with an invented type.
    """

    if not isinstance(counters, MutableMapping):
        raise CounterCaptureError(
            f"counters is not a mutable mapping: {_type_name(counters)}"
        )

    if isinstance(required_groups, (str, bytes)):
        raise CounterCaptureError("required counter groups must be a sequence of names")
    names = tuple(required_groups)
    if not names:
        raise CounterCaptureError("at least one required counter group is needed")
    if any(not isinstance(name, str) or not name for name in names):
        raise CounterCaptureError(
            "required counter group names must be non-empty strings"
        )
    if len(set(names)) != len(names):
        raise CounterCaptureError("required counter group names must be unique")

    keys_before = tuple(sorted(str(key) for key in counters.keys()))
    prepared: dict[str, _PreparedGroup] = {}
    for name in names:
        present_before = name in counters
        raw_before = _copy_counts(name, counters.get(name)) if present_before else {}
        try:
            group = counters[name]
        except KeyError as exc:
            raise CounterCaptureError(
                f"missing group {name!r} was not created by native lookup"
            ) from exc
        if not isinstance(group, MutableMapping):
            raise CounterCaptureError(
                f"counter group {name!r} is not mutable: {_type_name(group)}"
            )
        raw_after = _copy_counts(name, group)
        if raw_after != raw_before:
            raise CounterCaptureError(
                f"materializing counter group {name!r} changed its entries"
            )
        prepared[name] = _PreparedGroup(
            name=name,
            object_ref=group,
            object_id=id(group),
            object_type=_type_name(group),
            present_before_prepare=present_before,
            raw_before_prepare=raw_before,
            raw_after_prepare=raw_after,
        )

    return CounterLifecycleCapture(
        _counters=counters,
        _mapping_id=id(counters),
        _mapping_type=_type_name(counters),
        _prepared=prepared,
        _top_level_keys_before_prepare=keys_before,
    )
