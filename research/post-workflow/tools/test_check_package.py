from __future__ import annotations

import binascii
import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from tools.check_package import validate_package
from tools.test_qualified_cpu_study import build_cpu_fixture
from tools.typefully_draft_guard import decide


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
    def test_required_reader_visual_declaration_cannot_be_omitted(self) -> None:
        manifest_path = self.package / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reader_visuals_required"] = True
        manifest.pop("reader_visuals", None)
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(any("reader_visuals_required" in error for error in validate_package(self.package)))

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "posts" / "fixture"
        self.package.mkdir(parents=True)
        for relative, text in {
            "README.md": "# Article\nHypothetical example.\n",
            "tweets.md": "# Social\n",
            "source-comment.md": "# Source-native comment\n\nStatus: draft, unsent\n\n## Source context\n\nFixture source context.\n\n## Comment draft\n\nA useful unsent fixture comment.\n",
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
            "schema_version": 2,
            "content_state": "hypothetical",
            "concepts": CONCEPTS,
            "cover_stem": COVER,
            "selected_image_stem": COVER,
            "audiences": AUDIENCES,
            "review_receipt": "images/review.json",
            "typefully_receipt": "social/typefully-receipt.json",
            "typefully_workflow_receipt": "social/typefully-workflow.json",
            "source_comment": "source-comment.md",
        }
        self.write_json("manifest.json", manifest)
        selected = self.package / f"{COVER}.png"
        (self.package / "image.png").write_bytes(selected.read_bytes())
        (self.package / "image-square.png").write_bytes((self.package / f"{COVER}-square.png").read_bytes())
        (self.package / "image.txt").write_bytes((self.package / f"{COVER}.txt").read_bytes())
        self.write_review()
        self.write_typefully()
        self.write_typefully_workflow()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def read_json(self, relative: str) -> dict:
        return json.loads((self.package / relative).read_text(encoding="utf-8"))

    def write_json(self, relative: str, value: object) -> None:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def prepend_json_member(self, relative: str, key: str, value: object) -> None:
        path = self.package / relative
        document = path.read_text(encoding="utf-8")
        self.assertTrue(document.startswith("{"))
        path.write_text(
            f"{{\n  {json.dumps(key)}: {json.dumps(value)}," + document[1:],
            encoding="utf-8",
        )

    def assert_only_duplicate_json_error(self, label: str, key: str) -> None:
        errors = validate_package(self.package)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(f"invalid {label}", errors[0])
        self.assertIn(f"duplicate JSON key: {key}", errors[0])

    def assert_identical_and_conflicting_duplicates_rejected(
        self,
        relative: str,
        label: str,
        key: str,
        identical: object,
        conflicting: object,
    ) -> None:
        path = self.package / relative
        valid_document = path.read_text(encoding="utf-8")
        self.assertEqual(validate_package(self.package), [])
        for duplicate_kind, value in (
            ("identical", identical),
            ("conflicting", conflicting),
        ):
            with self.subTest(relative=relative, duplicate_kind=duplicate_kind):
                path.write_text(valid_document, encoding="utf-8")
                self.prepend_json_member(relative, key, value)
                self.assert_only_duplicate_json_error(label, key)
        path.write_text(valid_document, encoding="utf-8")

    def assert_duplicate_manifest_schema_version_rejected(self, value: int) -> None:
        self.prepend_json_member("manifest.json", "schema_version", value)
        self.assert_only_duplicate_json_error("manifest", "schema_version")

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
                "published_at": None,
                "publication_authorized": False,
                "platforms": ["x", "linkedin"],
                "image_sha256": hashlib.sha256((self.package / "image.png").read_bytes()).hexdigest(),
                "texts": {
                    "x": (self.package / "social/x.txt").read_text(encoding="utf-8"),
                    "linkedin": (self.package / "social/linkedin.txt").read_text(encoding="utf-8"),
                },
                "content_files": {"x": "social/x.txt", "linkedin": "social/linkedin.txt"},
                "draft_id": "draft-123",
                "social_set_id": 17,
                "media_ids": ["media-x", "media-linkedin"],
                "editor_url": "https://typefully.example/draft-123",
                "readback_verified": True,
            },
        )

    def artifact_ref(self, relative: str) -> dict[str, str]:
        path = self.package / relative
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write_typefully_workflow(self) -> None:
        raw = {
            "id": "draft-123",
            "draft_id": "draft-123",
            "social_set_id": 17,
            "status": "draft",
            "publish_state": None,
            "created_at": "2026-10-26T11:00:00Z",
            "updated_at": "2026-10-26T12:00:00Z",
            "scheduled_date": None,
            "published_at": None,
            "share_url": None,
            "private_url": "https://typefully.example/draft-123",
            "x_published_url": None,
            "linkedin_published_url": None,
            "mastodon_published_url": None,
            "threads_published_url": None,
            "bluesky_published_url": None,
            "substack_published_url": None,
            "x_article_published_url": None,
            "x_post_published_at": None,
            "linkedin_post_published_at": None,
            "mastodon_post_published_at": None,
            "threads_post_published_at": None,
            "bluesky_post_published_at": None,
            "substack_post_published_at": None,
            "x_article_published_at": None,
            "draft_title": "Fixture draft",
            "tags": ["needs-review"],
            "scratchpad_text": "Fixture notes.",
            "platforms": {
                "x": {
                    "enabled": True,
                    "posts": [{
                        "text": (self.package / "social/x.txt").read_text(encoding="utf-8"),
                        "media_ids": ["media-x"],
                    }],
                    "settings": None,
                },
                "linkedin": {
                    "enabled": True,
                    "posts": [{
                        "text": (self.package / "social/linkedin.txt").read_text(encoding="utf-8"),
                        "media_ids": ["media-linkedin"],
                    }],
                    "settings": None,
                },
                "mastodon": {"enabled": False},
                "threads": {"enabled": False},
                "bluesky": {"enabled": False},
                "substack": {"enabled": False},
                "x_article": None,
            },
        }
        target = {"social_set_id": 17}
        payload = {
            key: copy.deepcopy(raw[key])
            for key in ("draft_title", "tags", "scratchpad_text", "platforms")
        }
        guard = decide({
            "desired": {"target": target, "payload": payload},
            "readback_target": target,
            "list_readback": {"results": [copy.deepcopy(raw)]},
            "detail_readback": copy.deepcopy(raw),
            "prior_receipt": None,
        })
        self.assertEqual(guard["decision"], "UNCHANGED")
        self.write_json("social/typefully-readback.json", raw)
        self.write_json("social/final-guard.json", guard)
        analytics = self.hashed_file(
            "social/typefully-analytics-review.md",
            """# Typefully analytics review

Status: completed

- Observed at: 2026-10-26T11:50:00Z
- Reviewer: Fixture analytics reviewer
- Sources checked: Retained Typefully X export and native LinkedIn availability view.
- Evidence limits: One small fixture cohort. Associations are descriptive and no causal result is claimed.
- Access blocker: Native LinkedIn metrics were unavailable in this fixture.
- Decision summary: Keep the direct technical hook and record LinkedIn analytics as unavailable.
""",
        )
        people = self.hashed_file(
            "social/typefully-people-review.md",
            """# Relevant people and conversation review

Status: completed

- Observed at: 2026-10-26T11:51:00Z
- Reviewer: Fixture people reviewer
- Sources checked: Package source list, named maintainers, and directly relevant native conversations.
- Evidence limits: Fixture identities are illustrative. No live mention or relationship is established.
- Access blocker: No additional blocker affected the bounded fixture review.
- Decision summary: Use no native mention and retain no conversation target because none was verified.
""",
        )
        promotion = self.hashed_file(
            "social/promotion-readback.txt", "Observed after final edit and reload.\n"
        )
        self.write_json(
            "social/typefully-workflow.json",
            {
                "schema_version": 1,
                "operation": "create",
                "prior_readback": None,
                "draft_id": "draft-123",
                "social_set_id": 17,
                "verified_at": "2026-10-26T12:05:00Z",
                "status": "draft",
                "scheduled_at": None,
                "published_at": None,
                "publication_authorized": False,
                "platforms": ["x", "linkedin"],
                "lifecycle_tag": "needs-review",
                "tags": ["needs-review"],
                "approval": None,
                "raw_readback": self.artifact_ref("social/typefully-readback.json"),
                "guard_receipt": self.artifact_ref("social/final-guard.json"),
                "media_ids": {"x": [["media-x"]], "linkedin": [["media-linkedin"]]},
                "analytics": {
                    "review": analytics,
                    "x": {"status": "reviewed", "reason": ""},
                    "linkedin": {"status": "unavailable", "reason": "No native fixture analytics."},
                },
                "people": {
                    "review": people,
                    "mentions": {"state": "none", "reason": "No native mention selected.", "verified": []},
                    "opportunities": {"state": "none", "reason": "No native parent selected."},
                },
                "promotion": {
                    name: {
                        "state": "off",
                        "trigger": "",
                        "reason": "Observed off in fixture.",
                        "evidence": promotion,
                        "checked_at": "2026-10-26T12:01:00Z",
                        "dormant": True,
                    }
                    for name in ("auto_retweet", "auto_plug")
                },
                "reviewer": "fixture-reviewer",
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

    def make_legacy_schema_v1(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest["schema_version"] = 1
        manifest.pop("typefully_workflow_receipt", None)
        manifest.pop("source_comment", None)
        self.write_json("manifest.json", manifest)
        (self.package / "source-comment.md").unlink(missing_ok=True)
        (self.package / "social/typefully-workflow.json").unlink(missing_ok=True)

        receipt = self.read_json("social/typefully-receipt.json")
        for field in ("published_at", "publication_authorized", "social_set_id", "media_ids"):
            receipt.pop(field, None)
        self.write_json("social/typefully-receipt.json", receipt)

    def make_cpu_measured(self, *, declare_envelope: bool = True) -> dict:
        receipt = build_cpu_fixture(self.package)
        self.write_json("evidence/measurement-receipt.json", receipt)
        manifest = self.read_json("manifest.json")
        manifest.update(content_state="measured", measurement_receipt="evidence/measurement-receipt.json")
        if declare_envelope:
            manifest["piq_experiment_evidence"] = receipt["piq_envelopes"][0]["path"]
            manifest["diagnostic_finding_required"] = True
            manifest["diagnostic_finding"] = "evidence/diagnostic-finding.json"
            self.write_piq_diagnostic_finding(receipt)
        self.write_json("manifest.json", manifest)
        (self.package / "VERIFICATION.md").write_text(
            "# Verification\n\n## CPU stage\n\nMeasured CPU study with retained qualification evidence.\n",
            encoding="utf-8",
        )
        return receipt

    def write_piq_diagnostic_finding(self, receipt: dict) -> None:
        envelope_ref = receipt["piq_envelopes"][0]
        envelope = self.read_json(envelope_ref["path"])
        attempt_ids = [attempt["id"] for attempt in envelope["attempts"]]
        run_ids = list(dict.fromkeys(attempt["run_id"] for attempt in envelope["attempts"]))
        phase_ref = receipt["raw_census"]
        layers = {
            name: {
                "state": "NOT_APPLICABLE",
                "observations": [],
                "evidence_refs": [],
                "gaps": [f"{name} is outside this synthetic package-gate fixture"],
            }
            for name in (
                "storage_and_input", "process_cpu_and_memory", "accelerator",
                "communication", "inference_or_training_runtime", "application_or_job",
            )
        }
        for name in ("storage_and_input", "application_or_job"):
            layers[name] = {
                "state": "OBSERVED",
                "observations": [f"Synthetic {name} fixture evidence is retained"],
                "evidence_refs": [phase_ref],
                "gaps": [],
            }
        finding = {
            "schema_version": 1,
            "kind": "piq_diagnostic_finding",
            "example_id": "synthetic-cpu-package-gate-fixture",
            "problem": {
                "user_visible_symptom": "Repeated preparation can remain inside a finite job boundary.",
                "reader_decision": "Decide whether to move immutable preparation outside repeated work.",
                "application_or_job_boundary": "Synthetic fixture start through accepted completion.",
            },
            "piq_detection": {
                "detectable_issue": "Repeated immutable preparation remains inside the job boundary.",
                "observed_signal": "Synthetic storage intervals recur inside the retained job census.",
                "affected_layers": ["storage_and_input", "application_or_job"],
                "decision_enabled": "Move immutable preparation to a separately accounted setup phase.",
                "evidence_refs": [phase_ref],
            },
            "predicted_symptom": {
                "hypothesis": "Repeated preparation is charged to every finite job.",
                "expected_observations": ["Preparation appears inside each admitted job boundary."],
                "evidence_refs": [phase_ref],
            },
            "competing_explanation": {
                "hypothesis": "The codec work itself explains the complete elapsed time.",
                "distinguishing_observations": ["Preparation and codec intervals remain separately identifiable."],
                "evidence_refs": [phase_ref],
            },
            "fixed_change": {
                "reference_identity": "synthetic repeated-preparation fixture",
                "candidate_identity": "synthetic separately-accounted-preparation fixture",
                "sole_intended_change": "immutable preparation lifecycle",
                "held_inputs_and_settings": ["same synthetic input and CPU contract"],
                "unavoidable_differences": [],
            },
            "complete_outcome": {
                "start_event": "synthetic finite job start",
                "stop_event": "accepted synthetic finite job completion",
                "setup_accounting": "synthetic setup retained separately and in the complete boundary",
                "cleanup_accounting": "synthetic cleanup retained in the fixture census",
                "failed_work_accounting": "the complete synthetic attempt census is retained",
                "excluded_costs": [],
            },
            "correctness": {
                "independent_oracle": "synthetic exact roundtrip digest",
                "acceptance_rule": "every admitted synthetic attempt passes exact digest equality",
                "covered_work": "every attempt in the declared PIQ envelope",
                "result": "PASS",
                "evidence": receipt["correctness"],
            },
            "layer_evidence": layers,
            "piq_custody": {
                "source_runtime_input_identities": "synthetic source, runtime and input identities retained by the fixture",
                "run_ids": run_ids,
                "attempt_ids": attempt_ids,
                "artifact_refs": [envelope_ref],
                "readback": receipt["raw_census"],
                "export": envelope_ref,
                "cleanup": receipt["registration"],
                "qualification_receipt": self.artifact_ref("evidence/measurement-receipt.json"),
                "report": receipt["qualification_report"],
            },
            "disposition": {
                "value": "SUPPORTED",
                "supported_observations": ["The synthetic fixture retains the issue-first cross-layer contract."],
                "remaining_unknowns": ["Synthetic fixtures do not establish real execution or performance."],
                "evidence_refs": [phase_ref],
            },
            "tradeoff": {
                "measured_benefit": "The synthetic fixture exercises the package contract only.",
                "added_cost_or_responsibility": ["setup accounting and invalidation remain explicit"],
                "applicability": "contract validation for PIQ-backed package tests",
                "simpler_option": "omit PIQ evidence for a non-PIQ historical package",
                "when_to_use_or_skip": "Use for PIQ-backed examples. Skip for packages with no PIQ evidence.",
                "next_measurement_or_action": "Replace synthetic refs with retained real evidence before production.",
            },
        }
        self.write_json("evidence/diagnostic-finding.json", finding)

    def test_valid_qualified_cpu_v2_does_not_require_gpu_placeholders(self) -> None:
        receipt = self.make_cpu_measured(declare_envelope=False)
        self.assertNotIn("power_limit_w", receipt)
        self.assertNotIn("expected_power_limit_w", receipt)
        self.assertEqual(validate_package(self.package), [])

    def test_valid_qualified_cpu_v2_binds_declared_piq_diagnostic(self) -> None:
        self.make_cpu_measured()
        self.assertEqual(validate_package(self.package), [])

    def test_invalid_cpu_v2_cannot_remove_diagnostic_block(self) -> None:
        receipt = self.make_cpu_measured()
        receipt["execution_class"] = "gpu"
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("cpu execution")
        self.assert_problem("PIQ diagnostic evidence cannot establish package readiness")

    def test_cpu_v2_requires_cpu_execution_verification(self) -> None:
        self.make_cpu_measured()
        (self.package / "VERIFICATION.md").write_text("# Verification\n\n## GPU stage\n\nMeasured GPU study.\n")
        self.assert_problem("nonempty '## CPU stage'")
        (self.package / "VERIFICATION.md").write_text("# Verification\n\n## CPU stage\n\nNot run.\n")
        self.assert_problem("CPU stage must explicitly describe itself as measured")

    def test_cpu_v2_wrong_envelope_hash_keeps_diagnostic_block(self) -> None:
        receipt = self.make_cpu_measured()
        receipt["piq_envelopes"][0]["sha256"] = "0" * 64
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("PIQ diagnostic evidence cannot establish package readiness")

    def test_unqualified_cpu_report_keeps_diagnostic_block(self) -> None:
        receipt = self.make_cpu_measured()
        reference = receipt["qualification_report"]
        report = self.read_json(reference["path"])
        report["verdict"] = "INCONCLUSIVE"
        self.write_json(reference["path"], report)
        reference["sha256"] = hashlib.sha256((self.package / reference["path"]).read_bytes()).hexdigest()
        self.write_json("evidence/measurement-receipt.json", receipt)
        errors = validate_package(self.package)
        self.assertTrue(any("report" in error.lower() for error in errors), errors)
        self.assertTrue(any("PIQ diagnostic evidence cannot establish package readiness" in error for error in errors), errors)

    def test_unrelated_diagnostic_cannot_use_valid_cpu_v2_qualification(self) -> None:
        self.make_cpu_measured()
        example = Path(__file__).resolve().parents[1] / "examples/piq-diagnostic"
        shutil.copytree(example, self.package / "evidence/unrelated-diagnostic")
        manifest = self.read_json("manifest.json")
        manifest["piq_experiment_evidence"] = "evidence/unrelated-diagnostic/envelope.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("PIQ diagnostic evidence cannot establish package readiness")

    def test_cpu_label_does_not_waive_legacy_gpu_v1_requirements(self) -> None:
        self.make_measured()
        receipt = self.read_json("evidence/measurement-receipt.json")
        receipt["execution_class"] = "cpu"
        receipt.pop("power_limit_w")
        receipt["profile_evidence"] = []
        self.write_json("evidence/measurement-receipt.json", receipt)
        self.assert_problem("power_limit_w must be a positive finite number")
        self.assert_problem("profile_evidence must be a nonempty list")

    def selected_copy(self, text: str) -> None:
        selected = self.read_json("manifest.json")["selected_image_stem"]
        (self.package / f"{selected}.txt").write_text(text, encoding="utf-8")
        (self.package / "image.txt").write_text(text, encoding="utf-8")

    def source_explanation_basis(self) -> dict:
        self.selected_copy("Choose a cache-miss policy. Valid cached code stays usable.\n")
        self.write_json("sources.json", {"sources": [{
            "id": "compiler-reference", "authors": ["Compiler Contributors"],
            "url": "https://example.org/compiler-reference",
        }]})
        manifest = self.read_json("manifest.json")
        basis = {
            "kind": "source_explanation", "image_stem": COVER,
            "sidecar_sha256": hashlib.sha256((self.package / f"{COVER}.txt").read_bytes()).hexdigest(),
            "sources": {"path": "sources.json", "sha256": hashlib.sha256((self.package / "sources.json").read_bytes()).hexdigest()},
            "source_ids": ["compiler-reference"],
            "scope": "Compiler control flow after representative warmup.",
        }
        manifest["selected_image_basis"] = basis
        self.write_json("manifest.json", manifest)
        return basis

    def test_nonnumeric_source_explanation_cover_uses_private_basis(self) -> None:
        self.source_explanation_basis()
        self.assertNotRegex((self.package / "image.txt").read_text(), r"hypothetical|illustrative")
        self.assertEqual(validate_package(self.package), [])

    def test_source_explanation_cover_needs_private_basis(self) -> None:
        self.selected_copy("Choose a cache-miss policy. Valid cached code stays usable.\n")
        self.assert_problem("manifest.selected_image_basis")

    def test_natural_example_must_state_its_assumptions(self) -> None:
        for text in (
            "Example: assume A is cached and B requires recompilation. Each call is attempted.\n",
            "Example with two workers. Assumptions: BF16 inputs and equal batch sizes.\n",
        ):
            with self.subTest(text=text):
                self.selected_copy(text)
                self.assertEqual(validate_package(self.package), [])
        for text in ("Example: four requests are twice as fast.\n", "Example with no assumptions stated.\n", "Example: assume\n"):
            with self.subTest(text=text):
                self.selected_copy(text)
                self.assert_problem("must identify an example and its assumptions")

    def test_malformed_source_explanation_basis_fails_closed(self) -> None:
        for field, value in (
            ("kind", "measured"), ("image_stem", "images/a"), ("scope", ""),
            ("sidecar_sha256", "bad"), ("sources", {"path": "other.json", "sha256": "0" * 64}),
            ("source_ids", []), ("source_ids", ["unknown"]), ("source_ids", [42]),
        ):
            with self.subTest(field=field, value=value):
                basis = self.source_explanation_basis()
                basis[field] = value
                manifest = self.read_json("manifest.json")
                manifest["selected_image_basis"] = basis
                self.write_json("manifest.json", manifest)
                self.assert_problem("manifest.selected_image_basis")
        for value in (None, "source_explanation", [], 42):
            with self.subTest(value=value):
                self.source_explanation_basis()
                manifest = self.read_json("manifest.json")
                manifest["selected_image_basis"] = value
                self.write_json("manifest.json", manifest)
                self.assert_problem("manifest.selected_image_basis")

    def test_source_explanation_basis_binds_current_sidecar_and_sources(self) -> None:
        self.source_explanation_basis()
        self.selected_copy("Choose how cache misses are handled.\n")
        self.assert_problem("sidecar_sha256 does not match")
        self.source_explanation_basis()
        self.write_json("sources.json", {"sources": []})
        self.assert_problem("sources.sha256 does not match")

    def test_source_explanation_basis_requires_attributed_sources(self) -> None:
        for field, value in (("url", "file:///private/source"), ("authors", [])):
            with self.subTest(field=field):
                self.source_explanation_basis()
                sources = self.read_json("sources.json")
                sources["sources"][0][field] = value
                self.write_json("sources.json", sources)
                manifest = self.read_json("manifest.json")
                manifest["selected_image_basis"]["sources"]["sha256"] = hashlib.sha256((self.package / "sources.json").read_bytes()).hexdigest()
                self.write_json("manifest.json", manifest)
                self.assert_problem("valid HTTPS URL and named attribution")

    def test_source_explanation_exception_does_not_admit_quantitative_covers(self) -> None:
        for text in ("Compile twice as fast at 4 ms.\n", "Faster serving.\n", "Two requests.\n"):
            with self.subTest(text=text):
                self.source_explanation_basis()
                self.selected_copy(text)
                manifest = self.read_json("manifest.json")
                manifest["selected_image_basis"]["sidecar_sha256"] = hashlib.sha256((self.package / f"{COVER}.txt").read_bytes()).hexdigest()
                self.write_json("manifest.json", manifest)
                self.assert_problem("must be nonnumeric and contain no performance claim")

    def test_source_explanation_basis_cannot_replace_private_execution_status(self) -> None:
        self.source_explanation_basis()
        (self.package / "VERIFICATION.md").write_text("# Verification\n\n## GPU stage\n\nNo GPU workload launched.\n")
        self.assert_problem("VERIFICATION.md must state that status explicitly")
        (self.package / "VERIFICATION.md").write_text("# Verification\nHypothetical.\n\n## GPU stage\n\nNot recorded.\n")
        self.assert_problem("GPU stage must explicitly state that no GPU workload or measurement was run")

    def test_source_explanation_basis_cannot_replace_measurement_receipt(self) -> None:
        self.source_explanation_basis()
        self.make_measured()
        self.assertEqual(validate_package(self.package), [])
        (self.package / "evidence/measurement-receipt.json").unlink()
        self.assert_problem("missing measurement receipt")

    def test_source_explanation_basis_cannot_replace_source_reported_receipt(self) -> None:
        self.source_explanation_basis()
        self.make_source_reported()
        self.assertEqual(validate_package(self.package), [])
        (self.package / "evidence/source-reported-receipt.json").unlink()
        self.assert_problem("missing source-reported receipt")

    def test_explicit_review_block_prevents_readiness(self) -> None:
        path = self.package / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["review_status"] = "REVIEW_BLOCKED"
        path.write_text(json.dumps(manifest))
        self.assertTrue(any("private preview only" in error for error in validate_package(self.package)))

    def test_array_manifest_is_a_validation_error(self) -> None:
        (self.package / "manifest.json").write_text("[]")
        self.assertTrue(any("manifest must be a JSON object" in error for error in validate_package(self.package)))

    def test_normal_valid_manifest_json_passes(self) -> None:
        self.assertNotIn("measurement_receipt", self.read_json("manifest.json"))
        self.assertEqual(validate_package(self.package), [])

    def test_manifest_rejects_conflicting_duplicate_key(self) -> None:
        self.assert_duplicate_manifest_schema_version_rejected(1)

    def test_manifest_rejects_identical_duplicate_key(self) -> None:
        self.assert_duplicate_manifest_schema_version_rejected(2)

    def test_sources_reject_identical_and_conflicting_duplicate_keys(self) -> None:
        self.write_json("sources.json", {"sources": []})
        self.assert_identical_and_conflicting_duplicates_rejected(
            "sources.json",
            "sources JSON",
            "sources",
            [],
            [{"id": "shadow-source"}],
        )

    def test_typefully_receipt_rejects_identical_and_conflicting_duplicate_keys(self) -> None:
        self.assert_identical_and_conflicting_duplicates_rejected(
            "social/typefully-receipt.json",
            "Typefully receipt",
            "status",
            "draft",
            "scheduled",
        )

    def test_independent_review_rejects_identical_and_conflicting_duplicate_keys(self) -> None:
        self.assert_identical_and_conflicting_duplicates_rejected(
            "images/review.json",
            "independent review receipt",
            "verdict",
            "PASS",
            "FAIL",
        )

    def test_measurement_receipt_rejects_identical_and_conflicting_duplicate_keys(self) -> None:
        self.make_measured()
        self.assert_identical_and_conflicting_duplicates_rejected(
            "evidence/measurement-receipt.json",
            "measurement receipt",
            "state",
            "measured",
            "source_reported",
        )

    def test_declared_diagnostic_finding_is_part_of_package_gate(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest.update(
            diagnostic_finding_required=True,
            diagnostic_finding="evidence/diagnostic-finding.json",
        )
        self.write_json("manifest.json", manifest)
        self.assert_problem("missing diagnostic finding")

    def test_diagnostic_finding_path_cannot_be_optional(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest["diagnostic_finding"] = "evidence/diagnostic-finding.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("diagnostic_finding_required")

    def indexed_phone_fixture(self, dimensions: tuple[int, int] = (400, 400)) -> dict:
        phone_path = f"{COVER}-400.png"
        (self.package / phone_path).write_bytes(png_bytes(*dimensions, (4, 24, 44)))
        square = {**self.artifact_ref(f"{COVER}-square.png"), "width": 1080, "height": 1080}
        index = {"assets": [{"stem": COVER, "selected": square,
                             "variants": {"square": dict(square), "phone": {
                                 **self.artifact_ref(phone_path), "width": dimensions[0], "height": dimensions[1]}}}]}
        self.write_json("images/asset-index.json", index)
        return index

    def test_indexed_square_phone_current_contract_passes(self) -> None:
        self.indexed_phone_fixture()
        self.assertEqual(validate_package(self.package), [])

    def test_indexed_legacy_phone_current_contract_passes(self) -> None:
        self.indexed_phone_fixture((400, 225))
        self.assertEqual(validate_package(self.package), [])

    def test_unindexed_square_phone_keeps_legacy_dimension_requirement(self) -> None:
        (self.package / f"{COVER}-400.png").write_bytes(png_bytes(400, 400, (4, 24, 44)))
        self.assert_problem(f"wrong dimensions for {COVER}-400.png: expected 400x225")
        self.write_json("images/asset-index.json", {"assets": [{"stem": COVER, "selected": {}}]})
        self.assert_problem(f"wrong dimensions for {COVER}-400.png: expected 400x225")

    def test_indexed_square_phone_rejects_stale_and_mismatched_references(self) -> None:
        original = self.indexed_phone_fixture()
        for failure in ("phone_hash", "selected_hash", "square_hash", "phone_path", "selected_path", "square_path", "selected_size"):
            with self.subTest(failure=failure):
                index = copy.deepcopy(original)
                asset = index["assets"][0]
                if failure.endswith("_hash"):
                    target = asset["selected"] if failure == "selected_hash" else asset["variants"][failure.removesuffix("_hash")]
                    target["sha256"] = "0" * 64
                elif failure.endswith("_path"):
                    target = asset["selected"] if failure == "selected_path" else asset["variants"][failure.removesuffix("_path")]
                    target["path"] = f"{COVER}.png"
                else:
                    asset["selected"]["width"] = 400
                self.write_json("images/asset-index.json", index)
                self.assert_problem("asset index")

    def test_indexed_phone_rejects_unsupported_or_false_geometry(self) -> None:
        for dimensions in ((400, 300), (800, 800), (400, 225)):
            with self.subTest(dimensions=dimensions):
                index = self.indexed_phone_fixture(dimensions)
                if dimensions == (400, 225):
                    path = self.package / f"{COVER}-400.png"
                    path.write_bytes(png_bytes(400, 400, (4, 24, 44)))
                    index["assets"][0]["variants"]["phone"].update(self.artifact_ref(f"{COVER}-400.png"))
                    self.write_json("images/asset-index.json", index)
                self.assert_problem("asset index")

    def test_indexed_phone_rejects_malformed_index_and_duplicate_stem(self) -> None:
        original = self.indexed_phone_fixture()
        cases = [None, [], {}, {"assets": []}, {"assets": [None]},
                 {"assets": original["assets"] * 2}]
        for field, value in (("variants", None), ("phone", None), ("selected", None), ("phone_hash", "invalid"), ("phone_width", "400")):
            index = copy.deepcopy(original)
            asset = index["assets"][0]
            if field == "phone":
                asset["variants"]["phone"] = value
            elif field == "phone_hash":
                asset["variants"]["phone"]["sha256"] = value
            elif field == "phone_width":
                asset["variants"]["phone"]["width"] = value
            else:
                asset[field] = value
            cases.append(index)
        for index in cases:
            with self.subTest(index=index):
                self.write_json("images/asset-index.json", index)
                self.assert_problem("asset index")
        (self.package / "images/asset-index.json").write_text("{invalid")
        self.assert_problem("invalid asset index")

    def test_retained_schema_v1_package_keeps_legacy_contract(self) -> None:
        self.make_legacy_schema_v1()
        self.assertEqual(validate_package(self.package), [])

    def test_schema_v2_upgrade_requires_every_extended_artifact(self) -> None:
        self.make_legacy_schema_v1()
        manifest = self.read_json("manifest.json")
        manifest["schema_version"] = 2
        self.write_json("manifest.json", manifest)
        errors = "\n".join(validate_package(self.package))
        self.assertIn("schema-2 manifest.typefully_workflow_receipt", errors)
        self.assertIn("schema-2 manifest.source_comment", errors)
        self.assertIn("Typefully receipt publication_authorized must be false", errors)

    def test_schema_v1_cannot_partially_adopt_schema_v2_fields(self) -> None:
        self.make_legacy_schema_v1()
        manifest = self.read_json("manifest.json")
        manifest["typefully_workflow_receipt"] = "social/typefully-workflow.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("schema-1 manifest must not declare typefully_workflow_receipt")

    def test_manifest_schema_version_is_explicit_and_bounded(self) -> None:
        manifest = self.read_json("manifest.json")
        for version in (0, 3, True, "2"):
            with self.subTest(version=version):
                manifest["schema_version"] = version
                self.write_json("manifest.json", manifest)
                self.assert_problem("manifest.schema_version must equal 1 or 2")
        template = json.loads(
            (Path(__file__).resolve().parents[1] / "templates/package-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(template["schema_version"], 2)

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

    def test_piq_diagnostic_cannot_replace_measured_receipt_or_package_gate(self) -> None:
        self.make_measured()
        example = Path(__file__).resolve().parents[1] / "examples/piq-diagnostic"
        shutil.copytree(example, self.package / "evidence/piq-experiment")
        manifest = self.read_json("manifest.json")
        manifest["piq_experiment_evidence"] = "evidence/piq-experiment/envelope.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("PIQ diagnostic evidence cannot establish package readiness")
        (self.package / "evidence/measurement-receipt.json").unlink()
        self.assert_problem("measurement receipt")

    def test_piq_import_does_not_enable_hypothetical_package_promotion(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples/piq-diagnostic"
        shutil.copytree(example, self.package / "evidence/piq-experiment")
        manifest = self.read_json("manifest.json")
        manifest["piq_experiment_evidence"] = "evidence/piq-experiment/envelope.json"
        self.write_json("manifest.json", manifest)
        self.assert_problem("PIQ diagnostic evidence cannot establish package readiness")

    def test_declared_two_host_functional_receipt_is_part_of_package_gate(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest["piq_two_host_functional_receipt"] = (
            "evidence/two-host-functional-receipt.json"
        )
        self.write_json("manifest.json", manifest)
        self.assert_problem("missing nonsymlink PIQ two-host functional receipt")
        template = Path(__file__).resolve().parents[1] / "templates/two-host-functional-receipt.json"
        receipt = self.package / "evidence/two-host-functional-receipt.json"
        receipt.parent.mkdir()
        receipt.write_bytes(template.read_bytes())
        self.assert_problem("cannot promote package readiness")

    def test_two_host_functional_receipt_requires_manifest_schema_two(self) -> None:
        self.make_legacy_schema_v1()
        manifest = self.read_json("manifest.json")
        manifest["piq_two_host_functional_receipt"] = (
            "evidence/two-host-functional-receipt.json"
        )
        self.write_json("manifest.json", manifest)
        self.assert_problem("supported only by schema 2")

    def test_missing_typefully_receipt_fails(self) -> None:
        (self.package / "social/typefully-receipt.json").unlink()
        self.assert_problem("missing Typefully receipt")

    def test_typefully_workflow_manifest_contract_is_mandatory(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest.pop("typefully_workflow_receipt")
        self.write_json("manifest.json", manifest)
        self.assert_problem("manifest.typefully_workflow_receipt must equal")

    def test_missing_typefully_workflow_receipt_fails(self) -> None:
        (self.package / "social/typefully-workflow.json").unlink()
        self.assert_problem("Typefully workflow: cannot read receipt")

    def test_source_comment_manifest_and_file_are_mandatory(self) -> None:
        manifest = self.read_json("manifest.json")
        manifest.pop("source_comment")
        self.write_json("manifest.json", manifest)
        self.assert_problem("manifest.source_comment must equal")
        (self.package / "source-comment.md").unlink()
        self.assert_problem("missing required file: source-comment.md")

    def test_unfilled_or_sent_source_comment_fails(self) -> None:
        template = Path(__file__).resolve().parents[1] / "templates/source-comment.md"
        (self.package / "source-comment.md").write_bytes(template.read_bytes())
        self.assert_problem("still matches the unfilled template")
        (self.package / "source-comment.md").write_text(
            "# Source-native comment\n\nStatus: sent\n\n## Source context\n\nFixture.\n\n## Comment draft\n\nFixture.\n"
        )
        self.assert_problem("must declare 'Status: draft, unsent'")

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

    def test_typefully_publication_authorization_fails(self) -> None:
        receipt = self.read_json("social/typefully-receipt.json")
        receipt["publication_authorized"] = True
        receipt["published_at"] = "2030-01-01T00:00:00Z"
        self.write_json("social/typefully-receipt.json", receipt)
        self.assert_problem("publication_authorized must be false")
        self.assert_problem("published_at must be")

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

    def test_cli_help_exits_zero_without_checking_a_package(self) -> None:
        script = Path(__file__).with_name("check_package.py")
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [sys.executable, str(script), flag],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertIn("retained package files", result.stdout)
                self.assertEqual(result.stderr, "")
                self.assertNotIn("FAIL", result.stdout)

    def test_manifest_rejects_nested_duplicate_keys(self) -> None:
        original = json.dumps(self.read_json("manifest.json"))
        duplicates = (
            '"metadata": {"owner": "first", "owner": "second"}',
            '"metadata": {"owner": "same", "owner": "same"}',
        )
        for duplicate in duplicates:
            with self.subTest(duplicate=duplicate):
                (self.package / "manifest.json").write_text(
                    original[:-1] + ", " + duplicate + "}\n",
                    encoding="utf-8",
                )
                errors = validate_package(self.package)
                self.assertEqual(len(errors), 1, errors)
                self.assertIn("invalid manifest", errors[0])
                self.assertIn("duplicate JSON key:", errors[0])

    def test_cli_reports_duplicate_manifest_key_without_traceback(self) -> None:
        original = json.dumps(self.read_json("manifest.json"))
        (self.package / "manifest.json").write_text(
            original[:-1] + ', "schema_version": 2}\n', encoding="utf-8"
        )
        script = Path(__file__).with_name("check_package.py")
        result = subprocess.run(
            [sys.executable, str(script), str(self.package)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("duplicate JSON key: schema_version", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("PASS", result.stdout)


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

    def test_two_host_owner_authority_receipt_is_fail_closed_in_package_gate(
        self,
    ) -> None:
        manifest = self.read_json("manifest.json")
        manifest["piq_two_host_owner_authority_receipt"] = (
            "evidence/two-host-owner-authority-receipt.json"
        )
        self.write_json("manifest.json", manifest)
        self.assert_problem("requires the exact declared producer functional receipt")
        self.assert_problem("missing nonsymlink PIQ two-host owner receipt")


if __name__ == "__main__":
    unittest.main()
