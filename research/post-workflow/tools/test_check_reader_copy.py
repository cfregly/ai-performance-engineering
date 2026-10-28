"""Focused reader-copy checks use temporary packages and no external services."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_package import validate_package
from tools.check_reader_copy import check_reader_copy, validate_reader_copy, visible_markdown


class ReaderCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, text: str) -> Path:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_internal_process_copy_is_rejected(self) -> None:
        for text in (
            "Illustrative routing.", "No new performance measurement.",
            "No new performance measurements.", "This is not a performance result.",
            "No captured trace.", "REVIEW_BLOCKED", "Review status: pending",
            "Verification: PASS", "Confidence: 97", "Reviewers: Alice",
            "STATUS: incomplete (cover pending)", "For internal review", "QA pending",
            "Status: draft", "STATUS: blocked",
        ):
            with self.subTest(text=text):
                self.assertTrue(check_reader_copy(text, "reader copy"))

    def test_scientific_conditions_and_material_limits_are_preserved(self) -> None:
        text = (
            "Example assumes two workers and BF16 inputs. The arrows show routing order.\n"
            "Assume A is cached and B requires recompilation. Strict errors are caught.\n"
            "Computed from the cited model dimensions, in bytes per token.\n"
            "Measured on one B200 at batch 8. This result does not establish end-to-end serving gains.\n"
            "The source's trace uses a different workload. Source: the authors, Figure 3.\n"
            "Its eager backend exercises routing without generating optimized GPU kernels.\n"
            "GPU time attributed to a host frame is not CPU execution time.\n"
            "Percentages are independently normalized to each worker's captured GPU time."
        )
        self.write("README.md", text)
        self.write("public-assets/captions/a.txt", text)
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_all_declared_reader_surfaces_are_checked(self) -> None:
        for relative in (
            "README.md", "review.md", "social/x.txt", "social/linkedin.txt",
            "public-assets/README.md", "public-assets/gallery.html",
            "public-assets/captions/a.txt", "images/a.txt", "image.txt",
        ):
            with self.subTest(relative=relative):
                path = self.write(relative, "No captured trace")
                self.assertTrue(any(relative in error for error in validate_reader_copy(self.package)))
                path.unlink()

    def test_gallery_labels_fail_but_the_same_link_destinations_pass(self) -> None:
        self.write("public-assets/README.md", (
            "# Routing policy\n![Routing policy](images/a.png)\n"
            "[Input example](images/b-square.png)\n"
            "[Capacity planning](images/aud-business-capacity.png)\n"
            "[Reference][aud-curious.png]\n\n"
            "[aud-curious.png]: images/aud-curious.png \"Routing\"\n"
            "[Compiler reference](https://example.org/no-captured-trace_(notes)/a.png)\n"
        ))
        self.assertEqual(validate_reader_copy(self.package), [])
        for label in (
            "### a: Routing", "### a", "### fun", "Option B", "variant c",
            "a.png", "fun-square.png", "## aud-curious: Newcomers",
            "![a.png](images/a.png)", "[b.png](images/b.png)", "**Option A**",
            "aud-curious", "<h3>a</h3>", "images/a.png", "[images/a.png](images/a.png)",
        ):
            with self.subTest(label=label):
                self.write("public-assets/README.md", label)
                self.assertTrue(any("internal option/filename label" in error
                                    for error in validate_reader_copy(self.package)))

    def test_article_variable_names_and_file_references_are_not_gallery_labels(self) -> None:
        self.write("README.md", "Matrix A and matrix B have the same dimensions. See a.png for the route.")
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_review_gallery_has_label_checks(self) -> None:
        self.write("review.md", "# Policy choice\n![Policy choice](images/a.png)")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("review.md", "### Option A\n![Policy choice](images/a.png)")
        self.assertTrue(any("review.md: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_html_checks_text_and_alt_but_ignores_attributes_and_scripts(self) -> None:
        self.write("public-assets/gallery.html", (
            '<h2>Routing policy</h2><img src="a.png" alt="Routing policy">'
            '<a href="no-captured-trace/a.png">Reference</a>'
            '<script>const status = "REVIEW_BLOCKED";</script>'
            '<!-- Verification: PASS -->'
        ))
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("public-assets/gallery.html", '<img src="a.png" alt="No captured trace">')
        self.assertTrue(validate_reader_copy(self.package))

    def test_svg_checks_actual_text_including_nested_spans_only(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<metadata>REVIEW_BLOCKED</metadata><title>Verification: PASS</title>'
            '<path id="a.png" d="M 0 0"/><image href="aud-curious.png"/>'
            '<text data-label="No captured trace">Cached code stays usable</text></svg>'
        )
        self.write("images/a.svg", svg)
        self.assertEqual(validate_reader_copy(self.package), [])
        for text in (
            'No captured <tspan>trace</tspan>', 'No new <tspan>performance measurement</tspan>',
            'REVIEW_BLOCKED', 'Option A', 'aud-curious.png',
        ):
            with self.subTest(text=text):
                self.write("images/a.svg", svg.replace("Cached code stays usable", text))
                self.assertTrue(validate_reader_copy(self.package))

    def test_svg_and_sidecar_must_both_pass(self) -> None:
        self.write("images/a.txt", "Valid cached code stays usable.")
        self.write("images/a.svg", '<svg><text>No captured trace</text></svg>')
        self.assertTrue(any("images/a.svg" in error for error in validate_reader_copy(self.package)))
        self.write("images/a.svg", '<svg><text>Valid cached code stays usable.</text></svg>')
        self.write("images/a.txt", "No captured trace")
        self.assertTrue(any("images/a.txt" in error for error in validate_reader_copy(self.package)))

    def test_malformed_svg_fails_instead_of_silently_skipping_text(self) -> None:
        self.write("images/a.svg", '<svg><text>No captured trace</svg>')
        self.assertTrue(any("invalid reader copy images/a.svg" in error
                            for error in validate_reader_copy(self.package)))

    def test_private_records_and_raw_captures_are_excluded(self) -> None:
        for relative in (
            "VERIFICATION.md", "manifest.json", "evidence/README.md", "evidence/a.svg",
            "captures/a.svg", "images/README.md", "images/review.json",
            "social/typefully-readback.json", "social/alternate.txt", "claim-scope-review.md",
            "public-assets/evidence/README.md", "public-assets/captures/a.svg",
            "images/superseded/a.txt",
        ):
            self.write(relative, "REVIEW_BLOCKED\nVerification: FAIL\nOption A\n")
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_only_image_gallery_is_checked_in_private_tweet_sheet(self) -> None:
        self.write("tweets.md", "## Private coordination\nREVIEW_BLOCKED\n\n## Image options\n"
                   "### Policy choice\n![Policy choice](images/a.png)\n\n## Private notes\nNo captured trace")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("tweets.md", "## Image options\n### a: Routing\n![Routing](images/a.png)")
        self.assertTrue(any("tweets.md image gallery: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_link_destinations_do_not_hide_visible_process_copy(self) -> None:
        self.assertEqual(visible_markdown("[Routing](https://example.org/(a.png))"), "[Routing]")
        self.assertTrue(check_reader_copy(visible_markdown(
            "[No captured trace](https://example.org/(a.png))"), "reader copy"))
        self.assertTrue(check_reader_copy(visible_markdown(
            "[Routing](unterminated No captured trace"), "reader copy"))

    def test_package_gate_rejects_reader_copy_and_keeps_other_errors(self) -> None:
        self.write("sources.json", "[]")
        self.write("manifest.json", "{}")
        self.write("README.md", "No captured trace")
        errors = validate_package(self.package)
        self.assertTrue(any("README.md: internal review/process disclaimer" in error for error in errors))
        self.assertTrue(any("manifest.schema_version" in error for error in errors))
        self.assertTrue(any("missing required file" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
