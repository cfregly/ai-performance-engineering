#!/usr/bin/env python3

import json
import unittest
from collections import Counter, defaultdict
from unittest.mock import patch

from counter_capture import (
    CounterCaptureError,
    DEFAULT_REQUIRED_GROUPS,
    prepare_counter_capture,
)


REASON = "user-inserted graph break"


def _reclassify_one_break(counters):
    """Model Unsupported creation followed by graph-break reclassification."""

    counters["unimplemented"][REASON] += 1
    counters["unimplemented"][REASON] -= 1
    if counters["unimplemented"][REASON] <= 0:
        del counters["unimplemented"][REASON]
    counters["graph_break"][REASON] += 1


class CounterCaptureTests(unittest.TestCase):
    def test_native_patch_dict_drops_group_created_inside_scope(self):
        counters = defaultdict(Counter)

        with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
            self.assertNotIn("graph_break", counters)
            _reclassify_one_break(counters)
            self.assertEqual(counters["graph_break"][REASON], 1)

        self.assertNotIn("graph_break", counters)
        self.assertEqual(dict(counters["inline_call"]), {})

    def test_precreated_native_groups_survive_shallow_restoration(self):
        counters = defaultdict(Counter)
        capture = prepare_counter_capture(counters)
        graph_break_group = counters["graph_break"]
        stats_group = counters["stats"]

        with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
            _reclassify_one_break(counters)
            counters["stats"]["unique_graphs"] += 1

        self.assertIs(counters["graph_break"], graph_break_group)
        self.assertIs(counters["stats"], stats_group)
        self.assertEqual(counters["graph_break"][REASON], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

        observation = capture.observe("after_patch_exit")
        self.assertEqual(
            observation["groups"]["graph_break"]["raw_counts"], {REASON: 1}
        )
        self.assertEqual(
            observation["groups"]["stats"]["raw_counts"], {"unique_graphs": 1}
        )
        evidence = capture.finish()
        self.assertTrue(evidence["invariants"]["lifecycle_valid"])

    def test_preparation_creates_native_empty_groups_without_entry_writes(self):
        writes = []

        class AuditedCounter(Counter):
            def __setitem__(self, key, value):
                writes.append((key, value))
                super().__setitem__(key, value)

        counters = defaultdict(AuditedCounter)
        capture = prepare_counter_capture(counters)

        self.assertEqual(tuple(counters), DEFAULT_REQUIRED_GROUPS)
        self.assertIsInstance(counters["graph_break"], AuditedCounter)
        self.assertIsInstance(counters["stats"], AuditedCounter)
        self.assertEqual(writes, [])
        for group in DEFAULT_REQUIRED_GROUPS:
            self.assertEqual(dict(counters[group]), {})

        capture.observe("after_prepare")
        evidence = capture.finish()
        self.assertTrue(evidence["invariants"]["preparation_added_zero_entries"])
        for group in DEFAULT_REQUIRED_GROUPS:
            prepared = evidence["preparation"][group]
            self.assertEqual(prepared["raw_before_prepare"], {})
            self.assertEqual(prepared["raw_after_prepare"], {})
            self.assertEqual(prepared["entry_delta_from_preparation"], 0)

    def test_same_object_content_mutation_is_observed_as_raw_counts(self):
        counters = defaultdict(Counter)
        capture = prepare_counter_capture(counters)
        capture.observe("before_increment")

        counters["graph_break"][REASON] += 2
        observation = capture.observe("after_increment")

        group = observation["groups"]["graph_break"]
        self.assertTrue(group["same_object_as_prepared"])
        self.assertFalse(group["binding_replaced"])
        self.assertTrue(group["content_changed_since_preparation"])
        self.assertEqual(group["raw_counts"], {REASON: 2})
        self.assertEqual(group["entry_sum"], 2)
        self.assertTrue(capture.finish()["invariants"]["lifecycle_valid"])

    def test_group_replacement_is_detected_without_repairing_it(self):
        counters = defaultdict(Counter)
        capture = prepare_counter_capture(counters)
        counters["graph_break"] = Counter({REASON: 1})

        observation = capture.observe("after_replacement")

        group = observation["groups"]["graph_break"]
        self.assertTrue(group["present"])
        self.assertTrue(group["binding_replaced"])
        self.assertFalse(group["same_object_as_prepared"])
        self.assertEqual(group["raw_counts"], {REASON: 1})
        self.assertFalse(capture.finish()["invariants"]["lifecycle_valid"])

    def test_missing_group_is_detected_without_recreating_it(self):
        counters = defaultdict(Counter)
        capture = prepare_counter_capture(counters)
        del counters["graph_break"]

        observation = capture.observe("after_deletion")

        self.assertFalse(observation["groups"]["graph_break"]["present"])
        self.assertNotIn("graph_break", counters)
        self.assertFalse(capture.finish()["invariants"]["lifecycle_valid"])

    def test_existing_counts_are_preserved_exactly(self):
        counters = defaultdict(Counter)
        counters["stats"]["unique_graphs"] = 3

        capture = prepare_counter_capture(counters)
        capture.observe("unchanged")
        evidence = capture.finish()

        prepared = evidence["preparation"]["stats"]
        self.assertEqual(prepared["raw_before_prepare"], {"unique_graphs": 3})
        self.assertEqual(prepared["raw_after_prepare"], {"unique_graphs": 3})
        self.assertEqual(prepared["entry_delta_from_preparation"], 0)

    def test_plain_dict_cannot_invent_a_missing_group_type(self):
        with self.assertRaisesRegex(CounterCaptureError, "native lookup"):
            prepare_counter_capture({})

    def test_single_string_is_not_treated_as_group_name_sequence(self):
        with self.assertRaisesRegex(CounterCaptureError, "sequence of names"):
            prepare_counter_capture(defaultdict(Counter), "graph_break")

    def test_duplicate_observation_stage_is_rejected(self):
        capture = prepare_counter_capture(defaultdict(Counter))
        capture.observe("stage")

        with self.assertRaisesRegex(CounterCaptureError, "duplicate"):
            capture.observe("stage")

    def test_evidence_is_json_serializable(self):
        capture = prepare_counter_capture(defaultdict(Counter))
        capture.observe("after_prepare")

        encoded = json.dumps(capture.finish(), sort_keys=True)

        self.assertIn("nested-graph-breaks-counter-lifecycle/v1", encoded)


if __name__ == "__main__":
    unittest.main()
