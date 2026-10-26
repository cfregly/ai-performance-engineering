"""Regression checks for duration weighting and isolated worker selection."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("render_time_profiles", HERE / "render_time_profiles.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class TimeFlamegraphTests(unittest.TestCase):
    def test_long_event_has_proportionally_more_width(self):
        root = renderer.tree([([("short", "cuda")], 0.001), ([("long", "cuda")], 0.999)])
        output = []
        renderer.bars(root, 0, 0, 1000, 25, 1, output)
        image = "".join(output)
        self.assertIn('width="999.000000"', image)
        self.assertIn('width="1.000000"', image)
        self.assertAlmostEqual(root["weight"], 1.0)

    def test_shared_host_frame_inherits_descendant_gpu_time(self):
        root = renderer.tree([([("python", "cpython"), ("kernel_a", "cuda")], 0.2),
                              ([("python", "cpython"), ("kernel_b", "cuda")], 0.8)])
        self.assertAlmostEqual(root["children"][("python", "cpython")]["weight"], 1.0)

    def test_pid_gpu_stream_and_kind_all_filter(self):
        doc = {"strings": ["cuda", "GPU-A", "cpu", "GPU-B"], "events": []}
        valid = {"pid": 10, "k": 0, "u": 1, "sid": 0, "c": 0.1}
        doc["events"] = [valid, {**valid, "pid": 11}, {**valid, "k": 2},
                         {**valid, "u": 3}, {**valid, "sid": None}]
        self.assertEqual(renderer.select_events(doc, 10, "GPU-A"), [(0, valid)])

    def test_invalid_scoped_time_fails(self):
        for value in [0, -1, float("nan"), float("inf"), True]:
            doc = {"strings": ["cuda", "GPU-A"], "events": [
                {"pid": 10, "k": 0, "u": 1, "sid": 0, "c": value}]}
            with self.assertRaises(ValueError):
                renderer.select_events(doc, 10, "GPU-A")

    def test_retained_enabled_worker_excludes_previous_worker(self):
        folder = HERE.parent / "captures/torch2141-b200-20261026/enabled/zymtrace-export"
        doc = json.loads((folder / "cuda-events.raw.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        selected = renderer.select_events(doc, audit["source_pid"], audit["expected_gpu_uuid"])
        self.assertEqual(len(selected), 188)
        self.assertEqual(len(doc["events"]) - len(selected), 104)
        self.assertEqual({e["pid"] for _, e in selected}, {632770})
        self.assertAlmostEqual(sum(e["c"] for _, e in selected), 0.06432514837848449)


if __name__ == "__main__":
    unittest.main()
