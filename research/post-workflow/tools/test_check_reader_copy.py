"""Focused reader-copy checks use temporary packages and no external services."""

from __future__ import annotations

import json
import sys
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
                if relative == "images/a.txt":
                    self.write("README.md", "![Routing](images/a.png)")
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
        self.write("README.md", "![Routing](images/a.png)")
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
        self.write("README.md", "![Routing](images/a.png)")
        self.write("images/a.txt", "Valid cached code stays usable.")
        self.write("images/a.svg", '<svg><text>No captured trace</text></svg>')
        self.assertTrue(any("images/a.svg" in error for error in validate_reader_copy(self.package)))
        self.write("images/a.svg", '<svg><text>Valid cached code stays usable.</text></svg>')
        self.write("images/a.txt", "No captured trace")
        self.assertTrue(any("images/a.txt" in error for error in validate_reader_copy(self.package)))

    def test_malformed_svg_fails_instead_of_silently_skipping_text(self) -> None:
        self.write("README.md", "![Routing](images/a.png)")
        self.write("images/a.svg", '<svg><text>No captured trace</svg>')
        self.assertTrue(any("invalid reader copy images/a.svg" in error
                            for error in validate_reader_copy(self.package)))

    def test_private_records_and_raw_captures_are_excluded(self) -> None:
        for relative in (
            "VERIFICATION.md", "evidence/README.md", "evidence/a.svg",
            "captures/a.svg", "images/README.md", "images/review.json",
            "social/typefully-readback.json", "social/alternate.txt", "claim-scope-review.md",
            "public-assets/evidence/README.md", "public-assets/captures/a.svg",
            "images/superseded/a.txt",
        ):
            self.write(relative, "REVIEW_BLOCKED\nVerification: FAIL\nOption A\n")
        self.write("manifest.json", json.dumps({"private_notes": "REVIEW_BLOCKED, Option A"}))
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_unlinked_support_sheet_is_not_selected_by_its_directory(self) -> None:
        self.write("images/contact-sheet.svg", '<svg><text>aud-ml-engineer</text></svg>')
        self.write("images/contact-sheet.txt", "OptionA")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("README.md", "[Compare diagrams](images/contact-sheet.svg)")
        errors = validate_reader_copy(self.package)
        self.assertTrue(any("contact-sheet.svg: internal option/filename label" in error for error in errors))
        self.assertTrue(any("contact-sheet.txt: internal option/filename label" in error for error in errors))

    def test_public_mapping_activates_an_unlinked_support_sheet(self) -> None:
        self.write("images/contact-sheet.svg", '<svg><text>aud-ml-engineer</text></svg>')
        self.write("manifest.json", json.dumps({"public_projection": {
            "directory": "_public", "sources": {"images/contact-sheet.svg": "images/contact-sheet.svg"}}}))
        self.assertTrue(any("contact-sheet.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_linked_historical_folder_figures_cannot_hide_labels(self) -> None:
        for folder in ("images/superseded", "evidence/history"):
            for suffix, text in (("svg", '<svg><text>aud-ml-engineer</text></svg>'),
                                 ("txt", "OptionA")):
                with self.subTest(folder=folder, suffix=suffix):
                    relative = folder + '/route.' + suffix
                    self.write(relative, text)
                    self.write("README.md", f'[Current figure]({relative})')
                    self.assertTrue(any(relative + ': internal option/filename label' in error
                                        for error in validate_reader_copy(self.package)))

    def test_balanced_parenthesis_destinations_activate_historical_figures(self) -> None:
        for folder in ("images/superseded", "evidence/history"):
            with self.subTest(folder=folder):
                relative = folder + '/chart(v2).svg'
                self.write(relative, '<svg><text>aud-ml-engineer</text></svg>')
                self.write("README.md", f'![Comparison]({relative})')
                self.assertTrue(any(relative + ': internal option/filename label' in error
                                    for error in validate_reader_copy(self.package)))
                self.write("README.md", '![Comparison](' + relative.replace('(', r'\(').replace(')', r'\)') + ')')
                self.assertTrue(any(relative + ': internal option/filename label' in error
                                    for error in validate_reader_copy(self.package)))
                self.write(relative, '<svg><text>A: original · B: moved</text><text>H: 8 × 2048</text></svg>')
                self.assertEqual(validate_reader_copy(self.package), [])

    def test_nested_link_label_activates_svg_and_sidecar_checks(self) -> None:
        self.write("README.md", "[![Figure](images/route.svg)](https://example.org)")
        self.write("images/route.svg", '<svg><text>aud-curious</text></svg>')
        self.assertIn("images/route.svg: internal option/filename label: 'aud-curious'",
                      validate_reader_copy(self.package))
        self.write("images/route.svg", '<svg><text>Compare complete steps</text></svg>')
        self.write("images/route.txt", "aud-curious")
        self.assertIn("images/route.txt: internal option/filename label: 'aud-curious'",
                      validate_reader_copy(self.package))
        self.write("images/route.txt", "Compare complete steps")
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_deeply_nested_link_labels_do_not_use_python_recursion(self) -> None:
        text = "![Figure](images/route.svg)"
        for _ in range(sys.getrecursionlimit() + 1):
            text = f"[{text}](https://example.org)"
        self.write("README.md", text)
        self.write("images/route.svg", '<svg><text>aud-curious</text></svg>')
        self.assertIn("images/route.svg: internal option/filename label: 'aud-curious'",
                      validate_reader_copy(self.package))

    def test_nested_link_labels_do_not_hide_unsafe_destinations(self) -> None:
        for relative, text in (
            ("README.md", "[See [x](file:///a.svg)](https://example.org)"),
            ("README.md", "[See [person](urn:li:person:1)](https://example.org)"),
            ("social/linkedin.txt", "[@[Name](urn:li:person:1)](https://example.org)"),
        ):
            with self.subTest(relative=relative, text=text):
                path = self.write(relative, text)
                self.assertIn(f"invalid reader reference {relative}: unsupported local scheme",
                              validate_reader_copy(self.package))
                path.unlink()

    def test_nested_bracket_labels_activate_historical_figures(self) -> None:
        for folder in ("images/superseded", "evidence/history"):
            with self.subTest(folder=folder):
                relative = folder + '/chart.svg'
                self.write(relative, '<svg><text>aud-ml-engineer</text></svg>')
                for markup in (f'![Comparison [detail]]({relative})',
                               f'[Comparison [detail]]({relative} "Comparison title")'):
                    self.write("README.md", markup)
                    self.assertTrue(any(relative + ': internal option/filename label' in error
                                        for error in validate_reader_copy(self.package)))
                self.write(relative, '<svg><text>A: original · B: moved</text><text>H: 8 × 2048</text></svg>')
                self.assertEqual(validate_reader_copy(self.package), [])

    def test_publicly_mapped_historical_folder_figures_cannot_hide_labels(self) -> None:
        for folder in ("images/superseded", "evidence/history"):
            for suffix, text in (("svg", '<svg><text>aud-ml-engineer</text></svg>'),
                                 ("txt", "OptionA")):
                with self.subTest(folder=folder, suffix=suffix):
                    relative = folder + '/route.' + suffix
                    self.write(relative, text)
                    self.write("manifest.json", json.dumps({"public_projection": {
                        "directory": "_public", "sources": {relative: relative}}}))
                    self.assertTrue(any(relative + ': internal option/filename label' in error
                                        for error in validate_reader_copy(self.package)))

    def test_machine_evidence_is_not_prose_but_its_figure_links_are_followed(self) -> None:
        self.write("images/superseded/route.svg", '<svg><text>aud-curious</text></svg>')
        self.write("evidence/record.json", json.dumps({"status": "REVIEW_BLOCKED", "private_label": "OptionA",
            "artifact_path": "../images/superseded/route.svg"}))
        self.write("README.md", "[Evidence record](evidence/record.json)")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("evidence/record.json", json.dumps({"status": "REVIEW_BLOCKED", "private_label": "OptionA",
            "figure": "![Current figure](../images/superseded/route.svg)"}))
        errors = validate_reader_copy(self.package)
        self.assertTrue(any("images/superseded/route.svg: internal option/filename label" in error for error in errors))
        self.assertFalse(any("evidence/record.json: internal" in error for error in errors))
        self.write("README.md", "No figure is linked here.")
        self.write("manifest.json", json.dumps({"public_projection": {
            "directory": "_public", "sources": {"evidence/record.json": "evidence/record.json"}}}))
        self.assertTrue(any("images/superseded/route.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_unsupported_scheme_with_netloc_is_not_a_safe_external_url(self) -> None:
        for scheme in ("file", "javascript", "custom", "ftp"):
            with self.subTest(scheme=scheme):
                self.write("README.md", f'[Figure]({scheme}://host/route.svg)')
                self.assertTrue(any("unsupported local scheme" in error
                                    for error in validate_reader_copy(self.package)))
        self.write("README.md", "[External figure](https://example.org/route.svg)\n"
                   "[Network figure](//example.org/route.svg)")
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_mapped_companion_uses_its_public_link_base(self) -> None:
        self.write("public-assets/README.md", "![Dataflow](images/route.svg)")
        self.write("source/designed.svg", '<svg><text>H: 8 × 2048</text></svg>')
        self.write("images/contact-sheet.svg", '<svg><text>aud-ml-engineer</text></svg>')
        self.write("manifest.json", json.dumps({"public_projection": {"directory": "_public", "sources": {
            "README.md": "public-assets/README.md", "images/route.svg": "source/designed.svg"}}}))
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("source/designed.svg", '<svg><text>aud-ml-engineer</text></svg>')
        self.assertTrue(any("source/designed.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_explicit_selected_roles_activate_figure_text(self) -> None:
        for declaration in ({"cover_stem": "images/route"}, {"selected_image_stem": "images/route"},
                            {"concepts": ["images/route"]}, {"audiences": [{"stem": "images/route"}]},
                            {"reader_visuals": {"scientific_chart": {"path": "images/route.png"}}}):
            with self.subTest(declaration=declaration):
                self.write("manifest.json", json.dumps(declaration))
                self.write("images/route.svg", '<svg><text>OptionA</text></svg>')
                self.assertTrue(any("images/route.svg" in error for error in validate_reader_copy(self.package)))

    def test_scientific_legends_and_tensor_variables_are_not_file_labels(self) -> None:
        self.write("README.md", "![Scientific comparison](images/route.png)")
        labels = "A: original · B: moved\nH: 8 × 2048\nK: 2048 × 8"
        self.write("images/route.svg", '<svg>' + ''.join(f'<text>{line}</text>' for line in labels.splitlines()) + '</svg>')
        self.write("images/route.txt", labels)
        self.assertEqual(validate_reader_copy(self.package), [])
        for internal in ("Choose OptionA", "variant B", "aud-ml-engineer", "a.png", "fun-square.png"):
            with self.subTest(internal=internal):
                self.write("images/route.svg", f'<svg><text>H: 8 × 2048</text><text>{internal}</text></svg>')
                self.assertTrue(any("internal option/filename label" in error for error in validate_reader_copy(self.package)))

    def test_plain_single_letter_gallery_label_still_fails(self) -> None:
        self.write("review.md", "a: A choice for the reader")
        self.assertTrue(any("internal option/filename label" in error for error in validate_reader_copy(self.package)))

    def test_plural_variants_are_not_a_concatenated_option_label(self) -> None:
        self.write("public-assets/README.md", "Both variants preserve the scientific legend. More options remain possible.")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("public-assets/README.md", "Choose OptionA")
        self.assertTrue(any("internal option/filename label" in error for error in validate_reader_copy(self.package)))

    def test_linked_methods_file_references_are_not_gallery_titles(self) -> None:
        self.write("README.md", "[Reproduction methods](code/README.md)")
        self.write("code/README.md", "The aud-performance-engineer variants use the retained renderer. See a.png.")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("code/README.md", "No captured trace")
        self.assertTrue(any("code/README.md: internal review/process disclaimer" in error
                            for error in validate_reader_copy(self.package)))

    def test_referenced_nested_svg_cannot_hide_internal_labels(self) -> None:
        self.write("README.md", "![Dataflow](images/outer.svg)")
        self.write("images/outer.svg", '<svg><image href="inner.svg"/></svg>')
        self.write("images/inner.svg", '<svg><text>aud-curious</text></svg>')
        self.assertTrue(any("images/inner.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_guide_and_tweet_gallery_references_activate_reader_figures(self) -> None:
        self.write("images/route.svg", '<svg><text>aud-curious</text></svg>')
        self.write("tweets.md", "## Private drafts\n![Route](images/route.svg)\n")
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("images/README.md", "Coordination: REVIEW_BLOCKED\n![Route](route.svg)")
        self.assertTrue(any("images/route.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))
        (self.package / "images/README.md").unlink()
        self.write("tweets.md", "## Image options\n![Route](images/route.svg)\n")
        self.assertTrue(any("images/route.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_only_used_markdown_references_activate_support_artifacts(self) -> None:
        self.write("images/route.svg", '<svg><text>aud-curious</text></svg>')
        self.write("README.md", "[route]: images/route.svg\n\n````python\n[Route](images/route.svg)\n````\n"
                   '<script>const example = "[Route](images/route.svg)";</script>')
        self.assertEqual(validate_reader_copy(self.package), [])
        self.write("README.md", "![route]\n\n[route]: images/route.svg\n")
        self.assertTrue(any("images/route.svg: internal option/filename label" in error
                            for error in validate_reader_copy(self.package)))

    def test_local_reader_paths_fail_closed(self) -> None:
        for reference in ("../escape.svg", "%2e%2e/escape.svg", "/absolute.svg", "file:///absolute.svg"):
            with self.subTest(reference=reference):
                self.write("README.md", f'![Route]({reference})')
                self.assertTrue(any("invalid reader" in error for error in validate_reader_copy(self.package)))
        self.write("README.md", "![Route](images/missing.svg)")
        self.assertTrue(any("declared reader file is missing" in error for error in validate_reader_copy(self.package)))

    def test_typefully_linkedin_organization_mentions_are_allowed(self) -> None:
        self.write("social/linkedin.txt", (
            "Read @[Prime Intellect](urn:li:organization:100845452) and "
            "@[Another organization](urn:li:organization:1).\n"
            "Compare complete training steps on the intended workload."
        ))
        self.assertEqual(validate_reader_copy(self.package), [])

    def test_linkedin_mention_labels_still_receive_visible_copy_checks(self) -> None:
        self.write("social/linkedin.txt", "@[No captured trace](urn:li:organization:100845452)")
        errors = validate_reader_copy(self.package)
        self.assertIn("social/linkedin.txt: internal review/process disclaimer: 'No captured trace'", errors)
        self.assertFalse(any("unsupported local scheme" in error for error in errors))

    def test_linkedin_mention_exception_does_not_apply_to_other_files_or_aliases(self) -> None:
        mention = "@[Prime Intellect](urn:li:organization:100845452)"
        for relative in ("README.md", "social/x.txt", "public-assets/README.md"):
            with self.subTest(relative=relative):
                path = self.write(relative, mention)
                self.assertIn(f"invalid reader reference {relative}: unsupported local scheme",
                              validate_reader_copy(self.package))
                path.unlink()
        self.write("public-assets/mention.txt", mention)
        self.write("manifest.json", json.dumps({"public_projection": {
            "directory": "_public", "sources": {"social/linkedin.txt": "public-assets/mention.txt"}}}))
        self.assertIn("invalid reader reference social/linkedin.txt: unsupported local scheme",
                      validate_reader_copy(self.package))

    def test_linkedin_urns_require_the_exact_native_mention_shape(self) -> None:
        for text in (
            "[Prime Intellect](urn:li:organization:100845452)",
            "@[Prime Intellect](urn:li:person:100845452)",
            "@[Prime Intellect](URN:li:organization:100845452)",
            "@[Prime Intellect](urn:li:organization:0)",
            "@[Prime Intellect](urn:li:organization:01)",
            "@[Prime Intellect](urn:li:organization:-1)",
            "@[Prime Intellect](urn:li:organization:+1)",
            "@[Prime Intellect](urn:li:organization:1.5)",
            "@[Prime Intellect](urn:li:organization:١)",
            "@[Prime Intellect](urn:li:organization:)",
            "@[Prime Intellect](urn:li:organization:100845452abc)",
            "@[Prime Intellect](urn:li:organization:100845452#fragment)",
            "@[Prime Intellect](urn:li:organization:100845452?query=value)",
            "@[Prime Intellect](urn:li:organization:100845452/extra)",
            "@[Prime Intellect]( urn:li:organization:100845452)",
            "@[Prime Intellect](<urn:li:organization:100845452>)",
            '@[Prime Intellect](urn:li:organization:100845452 "title")',
            "@[ ](urn:li:organization:100845452)",
            "@[Prime\nIntellect](urn:li:organization:100845452)",
            r"\@[Prime Intellect](urn:li:organization:100845452)",
        ):
            with self.subTest(text=text):
                self.write("social/linkedin.txt", text)
                self.assertIn("invalid reader reference social/linkedin.txt: unsupported local scheme",
                              validate_reader_copy(self.package))
        for text in (
            "@[Prime Intellect](urn:li:organization:100845452",
            "@[Prime Intellect](<urn:li:organization:100845452)",
        ):
            with self.subTest(text=text):
                self.write("social/linkedin.txt", text)
                self.assertIn("invalid reader reference social/linkedin.txt: "
                              "LinkedIn URN outside an exact native organization mention",
                              validate_reader_copy(self.package))

    def test_markdown_cleanup_cannot_synthesize_a_native_linkedin_mention(self) -> None:
        for text in (
            "@<!-- -->[Prime Intellect](urn:li:organization:100845452)",
            "@`x`[Prime Intellect](urn:li:organization:100845452)",
            "@<script></script>[Prime Intellect](urn:li:organization:100845452)",
            "@<style></style>[Prime Intellect](urn:li:organization:100845452)",
            r"@\[Prime Intellect](urn:li:organization:100845452)",
            r"@[Prime Intellect\](urn:li:organization:100845452)",
            r"@[Prime Intellect]\(urn:li:organization:100845452)",
        ):
            with self.subTest(text=text):
                self.write("social/linkedin.txt", text)
                self.assertIn("invalid reader reference social/linkedin.txt: "
                              "LinkedIn URN outside an exact native organization mention",
                              validate_reader_copy(self.package))

    def test_stray_linkedin_urn_does_not_abort_neighbor_checks(self) -> None:
        self.write("social/linkedin.txt", (
            "urn:li:organization:100845452\n"
            "[Reference](../../escape.svg)"
        ))
        errors = validate_reader_copy(self.package)
        self.assertIn("invalid reader reference social/linkedin.txt: "
                      "LinkedIn URN outside an exact native organization mention", errors)
        self.assertIn("invalid reader path reference from social/linkedin.txt: escapes the package", errors)

    def test_linked_linkedin_candidate_does_not_gain_the_native_exception(self) -> None:
        self.write("README.md", "[Draft](social/v8-candidate/linkedin.txt)")
        self.write("social/v8-candidate/linkedin.txt",
                   "@[Prime Intellect](urn:li:organization:100845452)")
        self.assertIn("invalid reader reference social/v8-candidate/linkedin.txt: unsupported local scheme",
                      validate_reader_copy(self.package))

    def test_public_mapping_of_canonical_linkedin_copy_does_not_gain_the_native_exception(self) -> None:
        self.write("social/linkedin.txt", "@[Prime Intellect](urn:li:organization:100845452)")
        self.write("manifest.json", json.dumps({"public_projection": {
            "directory": "_public", "sources": {"images/post.txt": "social/linkedin.txt"}}}))
        self.assertIn("invalid reader reference images/post.txt: unsupported local scheme",
                      validate_reader_copy(self.package))

    def test_linkedin_html_and_reference_links_do_not_gain_a_urn_exception(self) -> None:
        for text in (
            '<a href="urn:li:organization:100845452">Prime Intellect</a>',
            "@[Prime Intellect][prime]\n\n[prime]: urn:li:organization:100845452",
        ):
            with self.subTest(text=text):
                self.write("social/linkedin.txt", text)
                self.assertIn("invalid reader reference social/linkedin.txt: unsupported local scheme",
                              validate_reader_copy(self.package))

    def test_valid_linkedin_mention_does_not_hide_an_unsafe_reference(self) -> None:
        mention = "@[Prime Intellect](urn:li:organization:100845452)\n"
        for reference, reason in (
            ("file:///absolute.svg", "invalid reader reference social/linkedin.txt: unsupported local scheme"),
            ("../../escape.svg", "invalid reader path reference from social/linkedin.txt: escapes the package"),
            ("%2e%2e/%2e%2e/escape.svg", "invalid reader path reference from social/linkedin.txt: escapes the package"),
            ("/absolute.svg", "invalid reader path reference from social/linkedin.txt: must stay within the package"),
            ("urn:example:other", "invalid reader reference social/linkedin.txt: unsupported local scheme"),
            ("urn:li:organization:100845452", "invalid reader reference social/linkedin.txt: unsupported local scheme"),
        ):
            with self.subTest(reference=reference):
                self.write("social/linkedin.txt", mention + f"[Reference]({reference})")
                self.assertIn(reason, validate_reader_copy(self.package))

    def test_valid_linkedin_mention_does_not_hide_a_symlink_neighbor(self) -> None:
        target = self.write("retained.svg", '<svg><text>Compare complete steps</text></svg>')
        (self.package / "images").mkdir()
        (self.package / "images/route.svg").symlink_to(target)
        self.write("social/linkedin.txt", (
            "@[Prime Intellect](urn:li:organization:100845452)\n"
            "[Figure](../images/route.svg)"
        ))
        self.assertIn("invalid reader path reference from social/linkedin.txt: symlinks are not allowed",
                      validate_reader_copy(self.package))

    def test_valid_linkedin_mention_still_activates_an_existing_in_package_neighbor(self) -> None:
        self.write("escape.svg", '<svg><text>aud-curious</text></svg>')
        self.write("social/linkedin.txt", (
            "@[Prime Intellect](urn:li:organization:100845452)\n"
            "[Figure](../escape.svg)"
        ))
        errors = validate_reader_copy(self.package)
        self.assertIn("escape.svg: internal option/filename label: 'aud-curious'", errors)
        self.assertFalse(any("escapes the package" in error for error in errors))

    def test_package_gate_accepts_native_linkedin_mention_without_hiding_other_errors(self) -> None:
        self.write("sources.json", "[]")
        self.write("manifest.json", "{}")
        self.write("social/linkedin.txt", "@[Prime Intellect](urn:li:organization:100845452)")
        errors = validate_package(self.package)
        self.assertFalse(any("invalid reader reference social/linkedin.txt" in error for error in errors))
        self.assertTrue(any("manifest.schema_version" in error for error in errors))
        self.assertTrue(any("missing required file" in error for error in errors))

    def test_symlinked_active_figure_is_rejected(self) -> None:
        target = self.write("retained.svg", '<svg><text>A: original · B: moved</text></svg>')
        (self.package / "images").mkdir()
        (self.package / "images/route.svg").symlink_to(target)
        self.write("README.md", "![Route](images/route.svg)")
        self.assertTrue(any("symlinks are not allowed" in error for error in validate_reader_copy(self.package)))

    def test_invalid_manifest_roles_do_not_skip_article_checks(self) -> None:
        self.write("README.md", "No captured trace")
        for source in ("../outside.svg", "/outside.svg", "images/../outside.svg"):
            with self.subTest(source=source):
                self.write("manifest.json", json.dumps({"public_projection": {
                    "directory": "_public", "sources": {"images/route.svg": source}}}))
                errors = validate_reader_copy(self.package)
                self.assertTrue(any("invalid reader path" in error for error in errors))
                self.assertTrue(any("README.md: internal review/process disclaimer" in error for error in errors))
        self.write("manifest.json", "not JSON")
        errors = validate_reader_copy(self.package)
        self.assertTrue(any("invalid reader selection manifest.json" in error for error in errors))
        self.assertTrue(any("README.md: internal review/process disclaimer" in error for error in errors))

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
