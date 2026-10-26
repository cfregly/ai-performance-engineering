from __future__ import annotations

import binascii
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from tools.check_package import validate_package


CONCEPTS = ["images/a", "images/b", "images/c"]
COVER = "images/fun"
AUDIENCES = [
    {"key": "curious", "reader": "Curious reader", "stem": "images/aud-curious"},
    {"key": "ml", "reader": "ML engineer", "stem": "images/aud-ml-engineer"},
    {"key": "perf", "reader": "Performance engineer", "stem": "images/aud-performance-engineer"},
    {"key": "kernel", "reader": "Kernel engineer", "stem": "images/aud-kernel-engineer"},
    {"key": "ops", "reader": "Platform operations", "stem": "images/aud-platform-operations"},
    {"key": "business", "reader": "Business and capacity", "stem": "images/aud-business-capacity"},
]


def png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


class PackageGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "posts" / "fixture"
        self.package.mkdir(parents=True)
        for relative, text in {
            "README.md": "# Article\nHypothetical example.\n",
            "tweets.md": "# Social\n",
            "claims-allow.txt": "100 ms | illustrative\n",
            "VERIFICATION.md": "# Verification\n\nHypothetical package.\n\n## GPU stage\n\nNo GPU workload launched.\n",
            "images/README.md": "# Images\n",
            "images/audiences.md": "# Audiences\n",
            "social/x.txt": "Exact X draft.\n",
            "social/linkedin.txt": "Exact LinkedIn draft.\n",
        }.items():
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.package / "sources.json").write_text("[]\n", encoding="utf-8")

        self.stems = [*CONCEPTS, COVER, *(item["stem"] for item in AUDIENCES)]
        for index, stem in enumerate(self.stems, 1):
            color = (index, index + 20, index + 40)
            for suffix, size in (("", (1600, 900)), ("-square", (1080, 1080)), ("-400", (400, 225))):
                path = self.package / f"{stem}{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png_bytes(*size, color))
            (self.package / f"{stem}.txt").write_text("Illustrative fixture.\n", encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "content_state": "hypothetical",
            "concepts": CONCEPTS,
            "cover_stem": COVER,
            "selected_image_stem": COVER,
            "audiences": AUDIENCES,
            "review_receipt": "images/review.json",
            "typefully_receipt": "social/typefully-receipt.json",
        }
        self.write_json("manifest.json", manifest)
        selected = self.package / f"{COVER}.png"
        (self.package / "image.png").write_bytes(selected.read_bytes())
        (self.package / "image-square.png").write_bytes((self.package / f"{COVER}-square.png").read_bytes())
        (self.package / "image.txt").write_bytes((self.package / f"{COVER}.txt").read_bytes())
        self.write_review()
        self.write_typefully()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def read_json(self, relative: str) -> dict:
        return json.loads((self.package / relative).read_text(encoding="utf-8"))

    def write_json(self, relative: str, value: object) -> None:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def write_review(self) -> None:
        images = {
            stem: {"verdict": "PASS", "sha256": hashlib.sha256((self.package / f"{stem}.png").read_bytes()).hexdigest()}
            for stem in self.stems
        }
        self.write_json("images/review.json", {"reviewer": "independent-reviewer", "independent": True, "verdict": "PASS", "images": images})

    def write_typefully(self) -> None:
        self.write_json(
            "social/typefully-receipt.json",
            {
                "status": "draft",
                "scheduled_at": None,
                "platforms": ["x", "linkedin"],
                "image_sha256": hashlib.sha256((self.package / "image.png").read_bytes()).hexdigest(),
                "texts": {
                    "x": (self.package / "social/x.txt").read_text(encoding="utf-8"),
                    "linkedin": (self.package / "social/linkedin.txt").read_text(encoding="utf-8"),
                },
                "content_files": {"x": "social/x.txt", "linkedin": "social/linkedin.txt"},
                "draft_id": "draft-123",
                "editor_url": "https://typefully.example/draft-123",
                "readback_verified": True,
            },
        )

    def hashed_file(self, relative: str, contents: str) -> dict[str, str]:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def make_measured(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest["content_state"] = "measured"
        manifest["measurement_receipt"] = "evidence/measurement-receipt.json"
        self.write_json("manifest.json", manifest)
        (self.package / "VERIFICATION.md").write_text(
            "# Verification\n\nMeasured package.\n\n## GPU stage\n\nMeasured workload with retained evidence.\n",
            encoding="utf-8",
        )
        source = self.hashed_file("code/benchmark.py", "print('benchmark')\n")
        repeats = []
        for repeat_id in ("repeat-01", "repeat-02"):
            raw = self.hashed_file(f"evidence/raw/{repeat_id}.json", json.dumps({"id": repeat_id}) + "\n")
            repeats.append({"id": repeat_id, "accepted": True, "raw_result": raw})
        failed = self.hashed_file("evidence/raw/attempt-warmup.json", '{"status":"failed"}\n')
        guard = self.hashed_file("evidence/guards/guard.json", '{"accepted":true}\n')
        profile = self.hashed_file("evidence/profiles/profile.json", '{"captured":true}\n')
        correctness = self.hashed_file("evidence/correctness.json", '{"accepted":true}\n')
        claim_source = {
            "path": "README.md",
            "sha256": hashlib.sha256((self.package / "README.md").read_bytes()).hexdigest(),
        }
        self.write_json(
            "evidence/measurement-receipt.json",
            {
                "schema_version": 1,
                "state": "measured",
                "workload_scope": "fixture benchmark, fixed input and launch configuration",
                "source_artifacts": [source],
                "accepted_repeats": repeats,
                "failed_attempts": [
                    {
                        "id": "attempt-warmup",
                        "accepted": False,
                        "failure_reason": "warmup was interrupted",
                        "raw_result": failed,
                    }
                ],
                "guard_receipts": [{"name": "execution-guard", "accepted": True, **guard}],
                "profile_evidence": [{"name": "runtime-profile", "accepted": True, **profile}],
                "runtime_identifier": "runtime-image@sha256:fixture",
                "hardware_identifier": "gpu-fixture/serial-fixture",
                "correctness": {"accepted": True, **correctness},
                "expected_power_limit_w": 1000,
                "power_limit_w": 1000,
                "claims": [
                    {
                        "id": "article-latency-claim",
                        "source": claim_source,
                        "accepted_repeat_ids": ["repeat-01", "repeat-02"],
                    }
                ],
            },
        )

    def assert_problem(self, phrase: str) -> None:
        errors = validate_package(self.package)
        self.assertTrue(any(phrase in error for error in errors), f"{phrase!r} absent from {errors!r}")

    def test_valid_fixture_passes(self) -> None:
        self.assertNotIn("measurement_receipt", self.read_json("manifest.json"))
        self.assertEqual(validate_package(self.package), [])

    def test_public_copy_rejects_contextless_boolean_comparisons(self) -> None:
        for phrase in ("with False", "with True"):
            with self.subTest(phrase=phrase):
                (self.package / "README.md").write_text(
                    f"# Article\nCompared the workload {phrase}.\n",
                    encoding="utf-8",
                )
                self.assert_problem(f"contextless comparison '{phrase}'")

    def test_public_copy_rejects_retained_metrics_boilerplate_in_caption(self) -> None:
        (self.package / "images/a.txt").write_text(
            "Raw metrics and units are retained.\n",
            encoding="utf-8",
        )
        self.assert_problem("boilerplate 'Raw metrics and units are retained.'")

    def test_public_copy_accepts_plain_labels_and_ignores_fenced_code(self) -> None:
        (self.package / "README.md").write_text(
            """# Article

The same workload ran with nested graph-break support and without it.

```python
run(with_support=False)  # exact code value, not a public comparison label
note = "with False, with True, Raw metrics and units are retained."
```
""",
            encoding="utf-8",
        )
        self.assertEqual(validate_package(self.package), [])

    def test_valid_measured_fixture_passes(self) -> None:
        self.make_measured()
        self.assertEqual(validate_package(self.package), [])

    def test_measured_manifest_requires_fixed_safe_receipt_path(self) -> None:
        self.make_measured()
        manifest = self.read_json("manifest.json")
        manifest["measurement_receipt"] = "../measurement-receipt.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("must equal 'evidence/measurement-receipt.json'")

    def test_measured_receipt_is_required(self) -> None:
        self.make_measured()
        (self.package / "evidence/measurement-receipt.json").unlink()
        self.assert_problem("missing measurement receipt")

    def test_corrupted_repeat_raw_result_fails_hash_check(self) -> None:
        self.make_measured()
        (self.package / "evidence/raw/repeat-01.json").write_text('{"corrupted":true}\n', encoding="utf-8")
        self.assert_problem("accepted_repeats[0].raw_result.sha256 does not match")

    def test_missing_profile_evidence_file_fails(self) -> None:
        self.make_measured()
        (self.package / "evidence/profiles/profile.json").unlink()
        self.assert_problem("missing measurement receipt profile_evidence[0] file")

    def test_repeated_repeat_id_fails(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["accepted_repeats"][1]["id"] = "repeat-01"
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("accepted repeat IDs must be distinct")

    def test_reused_raw_result_path_fails(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["accepted_repeats"][1]["raw_result"] = receipt["accepted_repeats"][0]["raw_result"]
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("raw-result paths must be distinct")

    def test_source_hash_failure_is_reported(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["source_artifacts"][0]["sha256"] = "0" * 64
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("source_artifacts[0].sha256 does not match")

    def test_source_artifact_symlink_cannot_escape_package(self) -> None:
        self.make_measured()
        outside = Path(self.temp.name) / "outside-source.py"
        outside.write_text("print('outside')\n", encoding="utf-8")
        link = self.package / "code/outside-source.py"
        link.symlink_to(outside)
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["source_artifacts"][0] = {
            "path": "code/outside-source.py",
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("source_artifacts[0].path escapes the package")

    def test_explicit_acceptance_statuses_are_required(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["guard_receipts"][0]["accepted"] = "yes"
        receipt["correctness"]["accepted"] = 1
        self.write_json("evidence/measurement-receipt.json", receipt)
        errors = validate_package(self.package)
        self.assertTrue(any("guard_receipts[0].accepted must be true" in error for error in errors))
        self.assertTrue(any("correctness.accepted must be true" in error for error in errors))

    def test_power_limit_must_match_declared_expected_value(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["expected_power_limit_w"] = 700
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("power_limit_w must equal declared expected_power_limit_w")

    def test_power_limit_uses_declared_value_instead_of_hardcoded_1000(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["expected_power_limit_w"] = 700
        receipt["power_limit_w"] = 700
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assertEqual(validate_package(self.package), [])

    def test_expected_power_limit_rejects_nan(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["expected_power_limit_w"] = float("nan")
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("expected_power_limit_w must be a positive finite number")

    def test_actual_power_limit_rejects_infinity(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["power_limit_w"] = float("inf")
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("power_limit_w must be a positive finite number")

    def test_claim_must_link_to_known_accepted_repeats(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["claims"][0]["accepted_repeat_ids"][1] = "repeat-missing"
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("references unknown repeats: repeat-missing")

    def test_failed_attempt_id_cannot_reuse_accepted_repeat_id(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["failed_attempts"][0]["id"] = "repeat-01"
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("distinct across accepted and failed attempts")

    def test_failed_attempt_raw_result_cannot_reuse_accepted_file(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["failed_attempts"][0]["raw_result"] = receipt["accepted_repeats"][0]["raw_result"]
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("raw-result paths must be distinct across accepted and failed attempts")

    def test_missing_audience_fails(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest["audiences"] = manifest["audiences"][:5]
        self.write_json("manifest.json", manifest)
        self.assert_problem("must contain exactly 6")

    def test_stale_image_review_fails(self) -> None:
        (self.package / "images/a.png").write_bytes(png_bytes(1600, 900, (201, 11, 77)))
        self.assert_problem("stale review hash for images/a")

    def test_missing_typefully_receipt_fails(self) -> None:
        (self.package / "social/typefully-receipt.json").unlink()
        self.assert_problem("missing Typefully receipt")

    def test_missing_independent_review_receipt_fails(self) -> None:
        (self.package / "images/review.json").unlink()
        self.assert_problem("missing independent review receipt")

    def test_scheduled_status_fails(self) -> None:
        receipt = self.read_json("social/typefully-receipt.json")
        receipt["status"] = "scheduled"
        receipt["scheduled_at"] = "2030-01-01T00:00:00Z"
        self.write_json("social/typefully-receipt.json", receipt)
        errors = validate_package(self.package)
        self.assertTrue(any("status must be 'draft'" in e for e in errors))
        self.assertTrue(any("scheduled_at must be null" in e for e in errors))

    def test_typefully_text_mismatch_fails(self) -> None:
        receipt = self.read_json("social/typefully-receipt.json")
        receipt["texts"]["x"] = "Stale text"
        self.write_json("social/typefully-receipt.json", receipt)
        self.assert_problem("texts.x does not exactly match social/x.txt")

    def test_unselected_shipping_file_fails(self) -> None:
        (self.package / "image.png").write_bytes((self.package / "images/a.png").read_bytes())
        self.assert_problem("image.png does not byte-match selected variant images/fun.png")

    def test_missing_manifest_lists_clear_failure(self) -> None:
        (self.package / "manifest.json").unlink()
        self.assert_problem("missing manifest")

    def test_cli_requires_explicit_package(self) -> None:
        script = Path(__file__).with_name("check_package.py")
        result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


    def make_source_reported(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest.update(
            content_state="source_reported",
            source_reported_receipt="evidence/source-reported-receipt.json",
        )
        self.write_json("manifest.json", manifest)
        self.write_json(
            "sources.json",
            {
                "sources": [
                    {
                        "id": "paper",
                        "type": "primary_paper",
                        "url": "https://example.org/paper-v1",
                        "authors": "Named Author et al.",
                    }
                ]
            },
        )
        (self.package / "VERIFICATION.md").write_text(
            "# Source-reported research\n\n## GPU stage\n\nNo local GPU workload was launched.\n"
        )
        raw = self.hashed_file("evidence/paper-results.json", '{"ratio":2.0}\n')
        results = [
            {
                "id": "paper-gain",
                "task": "fixed task",
                "candidate": "candidate",
                "baseline": "standard",
                "hardware": "published GPU",
                "metric": "generation wall time",
                "quality_condition": "matched quality",
                "source_revision": "v1",
                "attribution": "authors",
                "source_id": "paper",
                "speedup": 2.0,
                "retained_result": raw,
            }
        ]
        source_hash = hashlib.sha256((self.package / "sources.json").read_bytes()).hexdigest()
        assertions = hashlib.sha256(json.dumps(results, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        review = self.hashed_file(
            "evidence/source-review.json",
            json.dumps(
                {
                    "reviewer": "independent-source-reviewer",
                    "independent": True,
                    "verdict": "PASS",
                    "sources_sha256": source_hash,
                    "results_sha256": assertions,
                }
            )
            + "\n",
        )
        self.write_json(
            "evidence/source-reported-receipt.json",
            {
                "schema_version": 1,
                "state": "source_reported",
                "locally_measured_performance_claims": False,
                "local_gpu_execution": "not_run",
                "sources": {"path": "sources.json", "sha256": source_hash},
                "results": results,
                "independent_review": review,
            },
        )

    def reseal_source_reported_review(self) -> None:
        receipt = self.read_json("evidence/source-reported-receipt.json")
        source_hash = hashlib.sha256((self.package / "sources.json").read_bytes()).hexdigest()
        receipt["sources"]["sha256"] = source_hash
        assertions = hashlib.sha256(
            json.dumps(receipt["results"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        review = self.read_json("evidence/source-review.json")
        review.update(sources_sha256=source_hash, results_sha256=assertions)
        self.write_json("evidence/source-review.json", review)
        receipt["independent_review"]["sha256"] = hashlib.sha256(
            (self.package / "evidence/source-review.json").read_bytes()
        ).hexdigest()
        self.write_json("evidence/source-reported-receipt.json", receipt)

    def test_source_reported_primary_gain_passes_without_fabricated_local_run(self) -> None:
        self.make_source_reported()
        self.assertEqual(validate_package(self.package), [])

    def test_source_reported_cannot_substitute_for_local_measured_claims(self) -> None:
        self.make_source_reported()
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["locally_measured_performance_claims"] = True
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("exclude locally measured performance claims")

    def test_source_reported_cannot_hide_an_executed_local_workload(self) -> None:
        self.make_source_reported()
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["local_gpu_execution"] = "completed"
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("Executed packages use the measured contract")

    def test_source_reported_loss_cannot_enter_performance_post_pipeline(self) -> None:
        self.make_source_reported()
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["results"][0]["speedup"] = 0.9
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("finite gain greater than 1")

    def test_source_reported_needs_a_named_primary_source_and_baseline(self) -> None:
        self.make_source_reported()
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["results"][0]["source_id"] = "newsletter-only"
        receipt["results"][0]["baseline"] = ""
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("identify a linked primary source")
        self.assert_problem("baseline must be a nonempty string")

    def test_source_reported_claim_edit_invalidates_review(self) -> None:
        self.make_source_reported()
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["results"][0]["speedup"] = 3.0
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("does not bind the current attributed results")

    def test_source_reported_rejects_a_declared_measurement_receipt(self) -> None:
        self.make_measured()
        self.make_source_reported()
        self.assert_problem("must not declare measurement_receipt")

    def test_source_reported_malformed_sources_shape_is_a_validation_error(self) -> None:
        self.make_source_reported()
        self.write_json("sources.json", {"sources": 1})
        self.assert_problem("sources must be a nonempty list")

    def test_source_reported_primary_source_id_must_be_named(self) -> None:
        self.make_source_reported()
        sources = self.read_json("sources.json")
        sources["sources"][0]["id"] = ""
        self.write_json("sources.json", sources)
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["results"][0]["source_id"] = ""
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.reseal_source_reported_review()
        self.assert_problem("id must be a nonempty string")

    def test_source_reported_primary_source_needs_valid_https_host(self) -> None:
        self.make_source_reported()
        sources = self.read_json("sources.json")
        sources["sources"][0]["url"] = "https://"
        self.write_json("sources.json", sources)
        self.reseal_source_reported_review()
        self.assert_problem("valid HTTPS URL with a host")

    def test_source_reported_primary_source_must_name_authors(self) -> None:
        self.make_source_reported()
        sources = self.read_json("sources.json")
        del sources["sources"][0]["authors"]
        self.write_json("sources.json", sources)
        self.reseal_source_reported_review()
        self.assert_problem("must name its author or authors")

    def test_source_reported_review_must_name_reviewer(self) -> None:
        self.make_source_reported()
        review = self.read_json("evidence/source-review.json")
        review["reviewer"] = ""
        self.write_json("evidence/source-review.json", review)
        receipt = self.read_json("evidence/source-reported-receipt.json")
        receipt["independent_review"]["sha256"] = hashlib.sha256(
            (self.package / "evidence/source-review.json").read_bytes()
        ).hexdigest()
        self.write_json("evidence/source-reported-receipt.json", receipt)
        self.assert_problem("review reviewer must be a nonempty string")


if __name__ == "__main__":
    unittest.main()
