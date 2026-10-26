#!/usr/bin/env python3
"""Mechanical checks for the Whiteboard visual package."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render_whiteboard_visuals as visuals


class WhiteboardVisualTests(unittest.TestCase):
    def test_exact_evidence_values(self) -> None:
        evidence = visuals.load_evidence()
        self.assertEqual(f"{evidence['baseline_steady_us']:.3f}", "407.567")
        self.assertEqual(f"{evidence['candidate_steady_us']:.3f}", "89.935")
        self.assertEqual(f"{evidence['historical_speedup']:.6f}", "4.531795")
        self.assertEqual(f"{evidence['baseline_first10_ms']:.3f}", "2411.177")
        self.assertEqual(f"{evidence['candidate_first10_ms']:.3f}", "3211.677")
        self.assertEqual(f"{evidence['startup_penalty_pct']:.1f}", "33.2")
        self.assertEqual(evidence["baseline_launches"], 16)
        self.assertEqual(evidence["candidate_launches"], 2)
        self.assertEqual(evidence["operations"], 64)
        self.assertEqual(f"{evidence['fresh022_speedup']:.6f}", "4.541758")

    def test_all_families_and_formats_exist(self) -> None:
        expected_dimensions = {
            ".png": (1600, 900),
            "-square.png": (1080, 1080),
            "-400.png": (400, 225),
            "-square-400.png": (400, 400),
        }
        for spec in visuals.SPECS:
            for suffix, dimensions in expected_dimensions.items():
                path = HERE / f"{spec.key}{suffix}"
                self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
                with Image.open(path) as image:
                    self.assertEqual(image.size, dimensions, path)
                    self.assertEqual(image.format, "PNG", path)
            self.assertTrue((HERE / f"{spec.key}.txt").is_file())
            self.assertTrue((HERE / f"{spec.key}.json").is_file())

    def test_landscape_families_are_byte_distinct(self) -> None:
        hashes = []
        for spec in visuals.SPECS:
            path = HERE / f"{spec.key}.png"
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_public_copy_uses_natural_setting_labels(self) -> None:
        for spec in visuals.SPECS:
            sidecar = (HERE / f"{spec.key}.txt").read_text()
            self.assertNotRegex(sidecar, r"nested_graph_breaks\s*=\s*(?:False|True)")
            self.assertNotRegex(sidecar, r"\b(?:Disabled|Enabled)\b")
            self.assertIn("Evidence state: measured historical cohort", sidecar)
            metadata = json.loads((HERE / f"{spec.key}.json").read_text())
            self.assertEqual(
                metadata["provenance_flag_identifier"],
                "torch._dynamo.config.nested_graph_breaks",
            )
            self.assertEqual(metadata["public_setting_labels"], ["Support off", "Support on"])

    def test_titles_are_short(self) -> None:
        for spec in visuals.SPECS:
            words = re.findall(r"[A-Za-z0-9]+", spec.title)
            self.assertLessEqual(len(words), 8, spec.title)

    def test_cohort_boundary_is_explicit(self) -> None:
        provenance = json.loads((HERE / "visual-provenance.json").read_text())
        self.assertEqual(
            provenance["primary_visual_cohort"]["steady_geometric_mean_speedup"],
            4.531795,
        )
        self.assertEqual(
            provenance["excluded_separate_cohort"]["steady_geometric_mean_speedup"],
            4.541758,
        )
        self.assertFalse(
            provenance["excluded_separate_cohort"]["native_profiler_images_used_as_audience_assets"]
        )

    def test_gamma_origin_cover_is_traceable(self) -> None:
        source = visuals.gamma_source_metadata()
        self.assertEqual(source["dimensions"], [2400, 3000])
        self.assertEqual(source["sampled_palette"]["blue"], visuals.BLUE)
        self.assertEqual(source["sampled_palette"]["orange"], visuals.ORANGE)

        status = json.loads((HERE / "gamma-cover-status.json").read_text())
        self.assertEqual(status["state"], "integrated_existing_gamma_export")
        self.assertTrue(status["retained_existing_job_found"])
        self.assertTrue(status["current_cover_is_gamma_origin"])
        self.assertFalse(status["current_cover_is_direct_gamma_export"])
        self.assertEqual(status["export_sha256"], visuals.GAMMA_EXPORT_SHA256)
        self.assertEqual(status["source_card_sha256"], visuals.GAMMA_CARD_1_SHA256)

        derivation = json.loads((HERE / "gamma-cover-derivation.json").read_text())
        self.assertEqual(derivation["state"], "complete")
        self.assertEqual(len(derivation["outputs"]), 4)
        self.assertFalse(derivation["derivation"]["direct_bitmap_crop"])


if __name__ == "__main__":
    unittest.main()

