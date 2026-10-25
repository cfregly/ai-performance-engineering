"""Check the published source and actual retained measurements without CUDA."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import analyze
import publish_results

ROOT = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src") and value)


class PublicationTests(unittest.TestCase):
    def test_frozen_source_and_public_protocol_hashes(self):
        manifest = read_json(ROOT / "source-manifest.json")
        for name, entry in manifest["source_files"].items():
            with self.subTest(name=name):
                self.assertEqual(sha256(ROOT / name), entry.get("sha256", entry.get("public_sha256")))
        self.assertTrue((ROOT / "PYTORCH-LICENSE").is_file())

    def test_all_public_result_hashes_and_supporting_files(self):
        manifest = read_json(ROOT / "results/publication-manifest.json")
        self.assertEqual(len(manifest["records"]), 24)
        for record in manifest["records"]:
            path = ROOT / "results" / record["public_relative_path"]
            self.assertEqual(sha256(path), record["public_sha256"])
        for name, entry in manifest["supporting_files"].items():
            self.assertEqual(sha256(ROOT / "results" / name), entry["sha256"])

    def test_actual_retained_matrix_preserves_both_route_verdicts(self):
        records = analyze.discover_records([ROOT / "results"])
        self.assertEqual(len(records), 24)
        actual = analyze.analyze_records(records)
        expected = read_json(ROOT / "results/analyzer-result.json")
        self.assertEqual(actual["validation"]["status"], "PASS")
        self.assertEqual(actual["decision"], expected["decision"])
        self.assertEqual(actual["cases"], expected["cases"])
        self.assertTrue(actual["decision"]["qualified_steady_win"])
        self.assertFalse(actual["decision"]["qualified_startup_win"])
        self.assertEqual(actual["control_validity"]["cold_first_10"]["status"], "FAIL")
        self.assertAlmostEqual(
            actual["cases"]["nested_break_depth8"]["steady_median_invocation"]["geometric_mean_ratio"],
            4.531795,
            places=5,
        )

    def test_actual_record_corruption_is_rejected(self):
        records = analyze.discover_records([ROOT / "results"])
        with self.assertRaises(analyze.ValidationError):
            analyze.analyze_records(records[:-1])
        broken = copy.deepcopy(records)
        broken[0]["treatment"]["actual"] = not broken[0]["treatment"]["actual"]
        with self.assertRaises(analyze.ValidationError):
            analyze.analyze_records(broken)

    def test_projection_changes_only_the_declared_metadata_fields(self):
        manifest = read_json(ROOT / "results/publication-manifest.json")
        entry = manifest["records"][0]
        expected = read_json(ROOT / "results" / entry["public_relative_path"])
        unpublished = copy.deepcopy(expected)
        for change in entry["changes"]:
            parts = re.findall(r"[^.\[\]]+", change["path"])
            current = unpublished
            for part in parts[:-1]:
                current = current[int(part) if isinstance(current, list) else part]
            key = int(parts[-1]) if isinstance(current, list) else parts[-1]
            current[key] = 4321 if change["original_type"] == "int" else "<unpublished-test-metadata>"
        actual, changes = publish_results._public_record(unpublished, entry["record_id"])
        self.assertEqual(actual, expected)
        self.assertEqual({change["path"] for change in changes}, {change["path"] for change in entry["changes"]})
        publish_results._assert_public(actual, ["<unpublished-test-metadata>"])

    def test_projection_rejects_republication_and_unredacted_metadata(self):
        entry = read_json(ROOT / "results/publication-manifest.json")["records"][0]
        public = read_json(ROOT / "results" / entry["public_relative_path"])
        with self.assertRaises(publish_results.PublicationError):
            publish_results._public_record(public, entry["record_id"])
        with self.assertRaises(publish_results.PublicationError):
            publish_results._assert_public({"path": "/tmp/unpublished-test-path"}, [])

    def test_worker_cli_help_needs_no_torch_or_cuda(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "benchmark.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--enabled", result.stdout)
        self.assertIn("--expected-gpu-uuid", result.stdout)

    def test_graph_counts_and_local_html_links(self):
        graphs = ROOT / "graphs"
        data = read_json(graphs / "compiled-graphs.json")
        self.assertEqual({key: len(value) for key, value in data["arms"].items()}, {"0": 16, "1": 2})
        for regions in data["arms"].values():
            self.assertEqual(sum(region["operation_count"] for region in regions), 64)
        parser = LinkCollector()
        parser.feed((graphs / "comparison.html").read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            if not parsed.scheme and parsed.path:
                self.assertTrue((graphs / unquote(parsed.path)).is_file(), link)
        for invocation in range(1, 6):
            self.assertTrue((graphs / f"replay-invocation-{invocation}.svg").is_file())

    def test_graph_manifest_binds_public_data_and_rendered_assets(self):
        root = ROOT / "graphs"
        manifest = read_json(root / "visualization-manifest.json")
        for group in ("public_data_sha256", "rendered_asset_sha256"):
            for name, expected in manifest[group].items():
                self.assertEqual(sha256(root / name), expected, name)
        projection = read_json(root / "public-projection.json")
        self.assertEqual(len(projection["source_wrappers"]), 18)
        for wrapper in projection["source_wrappers"]:
            self.assertEqual(sha256(root / wrapper["public_path"]), wrapper["public_sha256"])

    def test_public_evidence_has_no_private_device_or_host_paths(self):
        forbidden = re.compile(r"GPU-[0-9a-fA-F]{8}-|/Users/|/root/|piq-verda|fregly-dev")
        paths = list((ROOT / "results").rglob("*.json"))
        paths.extend((ROOT / "graphs").rglob("*.json"))
        paths.extend((ROOT / "graphs").rglob("*.html"))
        paths.extend((ROOT / "graphs").rglob("*.svg"))
        for path in paths:
            self.assertIsNone(forbidden.search(path.read_text()), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
