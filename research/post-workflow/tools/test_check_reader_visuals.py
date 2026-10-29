"""Reader-visual contract tests retain local fixtures; no external services."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.check_reader_visuals import (
    ANALYTICAL_REVIEW_CHECKS, ANALYTICAL_STAGES, DERIVATION_REVIEW_CHECKS,
    REVIEW_CHECKS, STAGES, declaration_sha256, main, validate_reader_visuals,
)

COPY_FIXTURE_PATHS = {"article": "README.md", "x": "social/x.txt", "linkedin": "social/linkedin.txt"}


class ReaderVisualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        self.passages = [
            "A router chooses A, B, or C.",
            "Select only allowed weight rows before multiplying.",
            "Output head: dense 5 ms, selected rows 1 ms, 5× faster.",
            "Synthetic output-head stage; not whole-model serving.",
        ]
        self.sequence = [dict(stage=stage, excerpt=excerpt)
                         for stage, excerpt in zip(STAGES, self.passages)]
        self.article = "\n\n".join(self.passages)
        self.x_posts = ["\n\n".join(self.passages[:2]), "\n\n".join(self.passages[2:])]
        self.write("README.md", self.article)
        self.write("social/x.txt", "\n\n---\n\n".join(self.x_posts) + "\n")
        self.write("social/linkedin.txt", self.article + "\n")
        self.write("evidence/measurement-receipt.json", {"schema_version": 1, "state": "measured"})
        self.write("images/comparison.txt", self.passages[2] + "\n" + self.passages[3])
        # Header fixtures exercise the same retained geometry checks as the
        # existing preview gate; no test claims to render or inspect pixels.
        self.png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1200, 800)
        for name in ("images/comparison.png", "evidence/x.png", "evidence/linkedin.png"):
            self.write(name, self.png)
        self.readback = {
            "id": 42, "updated_at": "2026-01-01T10:00:00Z",
            "platforms": {
                "x": {"posts": [{"text": self.x_posts[0], "media_ids": []},
                                  {"text": self.x_posts[1], "media_ids": ["chart-x"]}]},
                "linkedin": {"posts": [{"text": self.article, "media_ids": ["chart-linkedin"]}]},
            },
        }
        self.preview = {"schema_version": 1, "verified": True,
                        "checked_at": "2026-01-01T10:01:00Z", "platforms": {}}
        for platform in ("x", "linkedin"):
            self.preview["platforms"][platform] = {
                "draft_id": 42,
                "screenshot": self.ref(f"evidence/{platform}.png"),
                "attachment_sources": {f"chart-{platform}": dict(self.ref("images/comparison.png"), media_type="image")},
                "images": [{
                    "media_id": f"chart-{platform}", "source_path": "images/comparison.png",
                    "source_sha256": self.ref("images/comparison.png")["sha256"],
                    "natural_width": 1200, "natural_height": 800,
                    "rendered_width": 360, "rendered_height": 240,
                    "object_fit": "contain", "object_position": "50% 50%", "preview_group_size": 1,
                    "visual_review": {"complete_profile": True, "headings_visible": True,
                                      "key_labels_readable": True, "reviewer": "Fixture reviewer"},
                }],
            }
        self.declaration = {
            "schema_version": 1,
            "copy": {surface: dict(self.ref(path), sequence=copy.deepcopy(self.sequence))
                     for surface, path in (("article", "README.md"), ("x", "social/x.txt"),
                                           ("linkedin", "social/linkedin.txt"))},
            "scientific_chart": self.ref("images/comparison.png"),
            "chart_text": self.ref("images/comparison.txt"),
            "measurement_basis": self.ref("evidence/measurement-receipt.json"),
            "review_receipt": "evidence/reader-visual-review.json",
        }
        self.manifest = {"content_state": "measured", "measurement_receipt": "evidence/measurement-receipt.json",
                         "reader_visuals": self.declaration}
        self.retain_preview()
        self.review = {
            "schema_version": 1, "status": "PASS", "reviewer": "Fixture reviewer",
            "reviewed_at": "2026-01-01T10:02:00Z",
            "declaration_sha256": declaration_sha256(self.declaration),
            "checks": {check: {"status": "PASS", "notes": f"Fixture finding for {check}."}
                       for check in REVIEW_CHECKS},
            "platforms": {platform: {"status": "PASS", "notes": "The retained chart labels and scope are visible.",
                                     "screenshot": self.ref(f"evidence/{platform}.png")}
                          for platform in ("x", "linkedin")},
        }
        self.save()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, value: object) -> None:
        path = self.package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")

    def ref(self, name: str) -> dict[str, str]:
        return {"path": name, "sha256": hashlib.sha256((self.package / name).read_bytes()).hexdigest()}

    def retain_preview(self) -> None:
        self.write("social/readback.json", self.readback)
        for spec in self.preview["platforms"].values():
            spec["raw_readback"] = self.ref("social/readback.json")
        self.write("social/profile-preview.json", self.preview)
        self.declaration["preview_receipt"] = self.ref("social/profile-preview.json")

    def save(self, *, renew_review: bool = False) -> list[str]:
        if renew_review:
            self.review["declaration_sha256"] = declaration_sha256(self.declaration)
        self.write("evidence/reader-visual-review.json", self.review)
        if self.declaration.get("schema_version") == 2:
            self.manifest["reader_visuals_review"] = self.ref("evidence/reader-visual-review.json")
        self.write("manifest.json", self.manifest)
        return validate_reader_visuals(self.package)

    def make_analytical(self) -> None:
        self.passages[2:] = [
            "Assuming 8 bytes and 4 bytes/s, the transfer lower bound is 2 s.",
            "This hypothetical arithmetic does not predict a measured serving speedup.",
        ]
        self.sequence = [dict(stage=stage, excerpt=excerpt)
                         for stage, excerpt in zip(ANALYTICAL_STAGES, self.passages)]
        self.article = "\n\n".join(self.passages)
        self.x_posts = ["\n\n".join(self.passages[:2]), "\n\n".join(self.passages[2:])]
        self.write("README.md", self.article)
        self.write("social/x.txt", "\n\n---\n\n".join(self.x_posts) + "\n")
        self.write("social/linkedin.txt", self.article + "\n")
        self.write("images/comparison.txt", "\n".join(self.passages[2:]))
        self.readback["platforms"]["x"]["posts"][0]["text"] = self.x_posts[0]
        self.readback["platforms"]["x"]["posts"][1]["text"] = self.x_posts[1]
        self.readback["platforms"]["linkedin"]["posts"][0]["text"] = self.article
        self.retain_preview()
        self.declaration.update(
            schema_version=2,
            copy={surface: dict(self.ref(path), sequence=copy.deepcopy(self.sequence))
                  for surface, path in COPY_FIXTURE_PATHS.items()},
            chart_text=self.ref("images/comparison.txt"),
        )
        self.declaration.pop("measurement_basis")
        self.manifest.pop("measurement_receipt")
        self.manifest.update(content_state="hypothetical", locally_measured_performance_claims=False,
                             local_gpu_execution="not_run")
        scope = "A synthetic transfer lower bound, with no serving prediction."
        self.authorization = {
            "schema_version": 1, "authorization_kind": "explicit_user_instruction",
            "permission": "single_analytical_package_without_measured_speedup",
            "package_directory": self.package.name, "claimed_scope": scope,
            "authorized_by": "Fixture human", "authorized_at": "2026-01-01T09:00:00Z",
            "source": {"kind": "user_message", "reference": "fixture-conversation/user-message-1",
                       "excerpt": "Prepare this one analytical package without a measured speedup."},
        }
        self.write("code/derive.py", "print(8 / 4)\n")
        self.basis = {
            "schema_version": 1, "result_kind": "synthetic_arithmetic", "author": "Fixture author",
            "package_directory": self.package.name, "claimed_scope": scope,
            "assumptions": ["Transfer size is 8 bytes.", "Bandwidth is an assumed 4 bytes/s."],
            "limitations": ["Launch overhead and contention are excluded."],
            "code": self.ref("code/derive.py"),
        }
        self.results = {key: copy.deepcopy(self.basis[key])
                        for key in ("schema_version", "result_kind", "claimed_scope", "assumptions", "limitations", "code")}
        self.results["values"] = [{"name": "transfer_lower_bound", "value": 2, "unit": "s"}]
        self.independent = {
            "schema_version": 1, "status": "PASS", "reviewer": "Independent fixture reviewer",
            "reviewed_at": "2026-01-01T09:30:00Z", "method": "independent_recalculation",
            "checks": {check: {"status": "PASS", "notes": f"Specific fixture finding for {check}."}
                       for check in DERIVATION_REVIEW_CHECKS},
        }
        self.review["checks"] = {check: {"status": "PASS", "notes": f"Specific fixture finding for {check}."}
                                 for check in ANALYTICAL_REVIEW_CHECKS}
        self.retain_analytical()

    def retain_analytical(self) -> None:
        self.write("evidence/analytical-authorization.json", self.authorization)
        self.write("evidence/analytical-results.json", self.results)
        self.basis["results"] = self.ref("evidence/analytical-results.json")
        self.write("evidence/analytical-basis.json", self.basis)
        self.manifest["analytical_authorization"] = self.ref("evidence/analytical-authorization.json")
        self.declaration["analytical_basis"] = self.ref("evidence/analytical-basis.json")
        self.independent.update(analytical_authorization=self.manifest["analytical_authorization"],
                                analytical_basis=self.declaration["analytical_basis"],
                                code=self.basis["code"], results=self.basis["results"])
        self.write("evidence/independent-review.json", self.independent)
        self.declaration["independent_review"] = self.ref("evidence/independent-review.json")

    def assert_error(self, errors: list[str], expected: str) -> None:
        self.assertTrue(any(expected in error for error in errors), errors)

    def test_valid_sequence_and_chart_in_second_x_post(self) -> None:
        self.assertEqual(self.save(), [])

    def test_legacy_without_declaration_is_unchanged(self) -> None:
        self.write("manifest.json", {"content_state": "measured"})
        self.assertEqual(validate_reader_visuals(self.package), [])

    def test_article_is_private_readme_even_when_public_projection_differs(self) -> None:
        self.manifest["public_projection"] = {"sources": {"README.md": "public-assets/code-README.md"}}
        self.assertEqual(self.save(), [])

    def test_declared_stage_order_cannot_be_skipped_or_reordered(self) -> None:
        self.declaration["copy"]["x"]["sequence"] = list(reversed(self.sequence))
        self.assert_error(self.save(renew_review=True), "copy.x.sequence must declare")

    def test_actual_passages_must_appear_in_declared_order(self) -> None:
        self.write("README.md", "\n\n".join(reversed(self.passages)))
        self.declaration["copy"]["article"].update(self.ref("README.md"))
        self.assert_error(self.save(renew_review=True), "copy.article.mechanism excerpt is absent or out of order")

    def test_selected_social_copy_cannot_be_replaced_with_alternate(self) -> None:
        self.write("social/alternate.txt", self.article)
        self.declaration["copy"]["x"].update(self.ref("social/alternate.txt"))
        self.assert_error(self.save(renew_review=True), "copy.x.path must equal social/x.txt")

    def test_mutated_image_or_copy_invalidates_hash(self) -> None:
        for name, expected in (("images/comparison.png", "scientific_chart: sha256 mismatch"),
                               ("social/linkedin.txt", "copy.linkedin: sha256 mismatch")):
            with self.subTest(name=name):
                original = (self.package / name).read_bytes()
                self.write(name, original + b"changed")
                self.assert_error(validate_reader_visuals(self.package), expected)
                self.write(name, original)

    def test_decorative_cover_and_chart_link_do_not_satisfy_attachment(self) -> None:
        self.write("images/cover.png", self.png + b"cover")
        for platform, spec in self.preview["platforms"].items():
            image = spec["images"][0]
            image["source_path"] = "images/cover.png"
            image["source_sha256"] = self.ref("images/cover.png")["sha256"]
            spec["attachment_sources"][f"chart-{platform}"] = dict(self.ref("images/cover.png"), media_type="image")
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "selected scientific chart is absent")

    def test_receipt_assertion_without_actual_saved_media_fails(self) -> None:
        self.readback["platforms"]["x"]["posts"][1]["media_ids"] = []
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "media_id is missing or stale")

    def test_unrelated_draft_cannot_supply_chart_or_text(self) -> None:
        self.readback["id"] = 99
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "no matching saved draft text")

    def test_saved_text_must_match_current_selected_copy(self) -> None:
        self.readback["platforms"]["linkedin"]["posts"][0]["text"] += " Extra old claim."
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "saved draft text differs")

    def test_screenshot_hash_required_even_when_old_preview_gate_passes(self) -> None:
        self.preview["platforms"]["x"]["screenshot"]["sha256"] = "0" * 64
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "x.screenshot: sha256 mismatch")

    def test_missing_screenshot_or_non_image_rejected(self) -> None:
        del self.preview["platforms"]["x"]["screenshot"]
        self.write("evidence/linkedin.png", b"text is not a screenshot")
        self.preview["platforms"]["linkedin"]["screenshot"] = self.ref("evidence/linkedin.png")
        self.retain_preview()
        errors = self.save(renew_review=True)
        self.assert_error(errors, "x.screenshot requires path and sha256")
        self.assert_error(errors, "linkedin.screenshot must be a retained")

    def test_new_inputs_require_a_new_bound_review(self) -> None:
        self.declaration["chart_text"] = self.ref("README.md")
        self.assert_error(self.save(), "declaration_sha256 is stale or missing")

    def test_semantic_checks_require_review_findings_not_bare_booleans(self) -> None:
        self.review["checks"]["example_before_jargon"] = True
        self.review["checks"]["phone_readable"]["notes"] = " "
        errors = self.save()
        self.assert_error(errors, "checks.example_before_jargon requires PASS and specific notes")
        self.assert_error(errors, "checks.phone_readable requires PASS and specific notes")

    def test_platform_review_must_bind_current_screenshot(self) -> None:
        self.review["platforms"]["x"]["screenshot"] = self.ref("evidence/linkedin.png")
        # Both fixture screenshots initially have identical bytes, but their
        # retained paths still identify different platform evidence.
        self.assert_error(self.save(), "platforms.x must bind the current preview screenshot")

    def test_completed_review_must_follow_preview_and_name_reviewer(self) -> None:
        self.review.update(reviewer=" ", reviewed_at="2026-01-01T09:59:00Z")
        errors = self.save()
        self.assert_error(errors, "requires a named reviewer")
        self.assert_error(errors, "predates its platform preview")

    def test_hypothetical_receipt_cannot_become_a_measured_comparison(self) -> None:
        self.manifest["content_state"] = "hypothetical"
        self.assert_error(self.save(), "measurement_basis must bind")

    def test_measurement_basis_cannot_point_to_an_unrelated_receipt(self) -> None:
        self.write("evidence/unrelated.json", {})
        self.declaration["measurement_basis"] = self.ref("evidence/unrelated.json")
        self.assert_error(self.save(renew_review=True), "measurement_basis must bind")

    def test_source_reported_results_bind_their_existing_evidence_contract(self) -> None:
        self.manifest["content_state"] = "source_reported"
        self.manifest["source_reported_receipt"] = self.manifest.pop("measurement_receipt")
        self.assertEqual(self.save(), [])

    def test_authorized_analytical_contract_preserves_preview_and_reader_gates(self) -> None:
        self.make_analytical()
        self.assertEqual(self.save(renew_review=True), [])
        self.readback["platforms"]["x"]["posts"][1]["media_ids"] = []
        self.retain_preview()
        self.assert_error(self.save(renew_review=True), "media_id is missing or stale")

    def test_schema_two_without_explicit_authorization_cannot_pass(self) -> None:
        self.make_analytical()
        del self.manifest["analytical_authorization"]
        self.assert_error(self.save(renew_review=True), "analytical_authorization requires path and sha256")

    def test_authorization_requires_human_provenance_and_exact_package_scope(self) -> None:
        self.make_analytical()
        self.authorization.update(authorization_kind="inferred", package_directory="another-package",
                                  claimed_scope="all future packages", authorized_by=" ")
        self.authorization["source"] = {"kind": "assistant_message", "reference": "", "excerpt": ""}
        self.retain_analytical()
        errors = self.save(renew_review=True)
        for expected in ("authorization_kind must equal", "package_directory must equal",
                         "claimed_scope must bind", "authorized_by requires", "source.kind must equal user_message",
                         "source.reference requires", "source.excerpt requires"):
            self.assert_error(errors, expected)

    def test_analytical_contract_cannot_relabel_measurements_or_executed_gpu_work(self) -> None:
        self.make_analytical()
        self.manifest.update(content_state="measured", locally_measured_performance_claims=True,
                             local_gpu_execution="completed", measurement_receipt="evidence/measurement-receipt.json",
                             source_reported_receipt="evidence/source.json")
        self.declaration["measurement_basis"] = self.ref("evidence/measurement-receipt.json")
        errors = self.save(renew_review=True)
        for expected in ("content_state hypothetical", "locally_measured_performance_claims false",
                         "local_gpu_execution not_run", "must not declare measurement_receipt",
                         "must not declare source_reported_receipt", "must not declare measurement_basis"):
            self.assert_error(errors, expected)

    def test_schema_two_requires_analytical_progression_and_semantic_review(self) -> None:
        self.make_analytical()
        self.declaration["copy"]["article"]["sequence"][2]["stage"] = "measured_comparison"
        self.review["checks"].pop("no_measured_performance_implication")
        errors = self.save(renew_review=True)
        self.assert_error(errors, "mechanism, analytical_comparison, scope in order")
        self.assert_error(errors, "checks.no_measured_performance_implication requires PASS")

    def test_derivation_requires_explicit_assumptions_limits_and_bound_results(self) -> None:
        self.make_analytical()
        self.basis["assumptions"] = []
        self.basis["limitations"] = ["Repeated", "Repeated"]
        self.results["claimed_scope"] = "Whole-model serving"
        self.results["values"][0].update(value=True, unit="")
        self.retain_analytical()
        errors = self.save(renew_review=True)
        for expected in ("analytical_basis.assumptions requires", "analytical_basis.limitations requires",
                         "results.claimed_scope must bind", "values.value must be a finite number", "values.unit requires"):
            self.assert_error(errors, expected)

    def test_independent_review_must_be_named_separate_and_specific(self) -> None:
        self.make_analytical()
        self.independent.update(reviewer="Fixture author", method="read_only")
        self.independent["checks"]["independent_recalculation"] = True
        self.retain_analytical()
        errors = self.save(renew_review=True)
        for expected in ("must differ from the derivation author", "method must equal independent_recalculation",
                         "checks.independent_recalculation requires PASS"):
            self.assert_error(errors, expected)
        self.independent["reviewer"] = " "
        self.retain_analytical()
        self.assert_error(self.save(renew_review=True), "independent_review.reviewer requires")

    def test_missing_independent_review_or_current_input_binding_cannot_pass(self) -> None:
        self.make_analytical()
        del self.declaration["independent_review"]
        self.assert_error(self.save(renew_review=True), "independent_review requires path and sha256")
        self.retain_analytical()
        self.independent["results"] = self.ref("code/derive.py")
        self.write("evidence/independent-review.json", self.independent)
        self.declaration["independent_review"] = self.ref("evidence/independent-review.json")
        self.assert_error(self.save(renew_review=True), "independent_review.results must bind")

    def test_all_analytical_inputs_are_hash_bound(self) -> None:
        self.make_analytical()
        self.assertEqual(self.save(renew_review=True), [])
        for name, label in (("evidence/analytical-authorization.json", "analytical_authorization"),
                            ("evidence/analytical-basis.json", "analytical_basis"),
                            ("code/derive.py", "analytical_basis.code"),
                            ("evidence/analytical-results.json", "analytical_basis.results"),
                            ("evidence/independent-review.json", "independent_review"),
                            ("evidence/reader-visual-review.json", "reader_visuals_review")):
            with self.subTest(name=name):
                original = (self.package / name).read_bytes()
                self.write(name, original + b" ")
                self.assert_error(validate_reader_visuals(self.package), f"{label}: sha256 mismatch")
                self.write(name, original)

    def test_analytical_receipt_paths_still_reject_symlinks_and_escapes(self) -> None:
        self.make_analytical()
        self.declaration["analytical_basis"]["path"] = "../outside.json"
        self.assert_error(self.save(renew_review=True), "analytical_basis.path must name a package-relative file")
        self.retain_analytical()
        (self.package / "evidence/auth-link.json").symlink_to(self.package / "evidence/analytical-authorization.json")
        self.manifest["analytical_authorization"]["path"] = "evidence/auth-link.json"
        self.assert_error(self.save(renew_review=True), "analytical_authorization.path must not use symlinks")

    def test_malformed_analytical_receipts_fail_without_crashing(self) -> None:
        self.make_analytical()
        self.basis["assumptions"] = [[], {}]
        self.results["values"] = [[], {"name": [], "value": float("inf"), "unit": "s"}]
        self.authorization["source"] = []
        self.independent["reviewed_at"] = []
        self.retain_analytical()
        errors = self.save(renew_review=True)
        self.assert_error(errors, "analytical_basis.assumptions requires")
        self.assert_error(errors, "values.value must be a finite number")
        self.assert_error(errors, "source.kind must equal user_message")
        self.assert_error(errors, "reviewed_at requires a completed ISO time")

    def test_schema_one_cannot_opt_into_analytical_basis(self) -> None:
        self.make_analytical()
        self.declaration["schema_version"] = 1
        self.assert_error(self.save(renew_review=True), "measurement_basis must bind")

    def test_malformed_state_and_receipt_shapes_fail_without_crashing(self) -> None:
        self.manifest["content_state"] = []
        self.declaration["scientific_chart"] = []
        self.review["checks"] = []
        errors = self.save(renew_review=True)
        self.assert_error(errors, "measurement_basis must bind")
        self.assert_error(errors, "scientific_chart requires path and sha256")
        self.assert_error(errors, "checks.example_before_jargon requires PASS")

    def test_package_path_escape_and_symlink_are_rejected(self) -> None:
        self.declaration["scientific_chart"]["path"] = "../outside.png"
        self.assert_error(self.save(renew_review=True), "scientific_chart.path must name a package-relative file")
        (self.package / "images/link.png").symlink_to(self.package / "images/comparison.png")
        self.declaration["scientific_chart"].update(path="images/link.png")
        self.assert_error(self.save(renew_review=True), "scientific_chart.path must not use symlinks")

    def test_declaration_digest_is_stable_across_json_key_order(self) -> None:
        self.assertEqual(declaration_sha256({"a": 1, "b": "×"}), declaration_sha256({"b": "×", "a": 1}))

    def test_cli_usage_and_failure_exit(self) -> None:
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 2)
            self.assertEqual(main([str(self.package / "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
