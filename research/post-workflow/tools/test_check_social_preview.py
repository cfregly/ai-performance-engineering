import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock

from tools.check_social_preview import _pdf_page_count, validate_social_preview


class SocialPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        (self.package / "social").mkdir()
        (self.package / "profiles").mkdir()
        self.write("profiles/profile-semantics.json", {})
        # The guard needs only the PNG signature and IHDR dimensions.
        (self.package / "profile.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1200, 747))
        self.readback = {"id": 123, "updated_at": "2026-10-26T10:00:00Z", "platforms": {platform: {"posts": [{"media_ids": [f"{platform}-image"]}]} for platform in ("x", "linkedin")}}
        self.receipt = {"schema_version": 1, "verified": True, "checked_at": "2026-10-26T10:05:00Z", "platforms": {}}
        for platform in ("x", "linkedin"):
            self.receipt["platforms"][platform] = {"draft_id": 123, "raw_readback": {"path": "social/readback.json"}, "images": [{"media_id": f"{platform}-image", "source_path": "profile.png", "source_sha256": self.digest("profile.png"), "natural_width": 1200, "natural_height": 747, "rendered_width": 510, "rendered_height": 317.475, "object_fit": "cover", "object_position": "50% 50%", "preview_group_size": 1, "visual_review": {"complete_profile": True, "headings_visible": True, "key_labels_readable": True, "reviewer": "rendered image review"}}]}
            self.receipt["platforms"][platform]["attachment_sources"] = {f"{platform}-image": {"path": "profile.png", "sha256": self.digest("profile.png"), "media_type": "image"}}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        (self.package / path).write_text(json.dumps(value), encoding="utf-8")

    def digest(self, path):
        return hashlib.sha256((self.package / path).read_bytes()).hexdigest()

    def image(self):
        return self.receipt["platforms"]["x"]["images"][0]

    def validate(self):
        self.write("social/readback.json", self.readback)
        for spec in [*self.receipt["platforms"].values(), *self.receipt.get("additional_drafts", [])]:
            spec["raw_readback"]["sha256"] = self.digest("social/readback.json")
        self.write("social/profile-preview.json", self.receipt)
        return validate_social_preview(self.package)

    def test_valid_single_image_preview(self):
        self.assertEqual(self.validate(), [])

    def test_cover_crop_of_old_multi_image_layout_rejected(self):
        self.image().update(rendered_width=259, rendered_height=291)
        self.assertTrue(any("cover crops" in error for error in self.validate()))

    def test_rounding_noise_allowed(self):
        self.image()["rendered_height"] = 318
        self.assertEqual(self.validate(), [])

    def test_contain_allows_different_box_ratio(self):
        self.image().update(object_fit="contain", rendered_width=259, rendered_height=291)
        self.assertEqual(self.validate(), [])

    def test_distorted_fill_rejected(self):
        self.image().update(object_fit="fill", rendered_width=259, rendered_height=291)
        self.assertTrue(any("fill distorts" in error for error in self.validate()))

    def test_png_prevents_fabricated_natural_ratio(self):
        self.image().update(natural_width=259, natural_height=291, rendered_width=259, rendered_height=291)
        self.assertTrue(any("retained PNG" in error for error in self.validate()))

    def test_source_hash_mismatch_rejected(self):
        self.image()["source_sha256"] = "0" * 64
        self.assertTrue(any("sha256 mismatch" in error for error in self.validate()))

    def test_clipping_or_unreadable_labels_rejected(self):
        for field in ("complete_profile", "headings_visible", "key_labels_readable"):
            with self.subTest(field=field):
                self.image()["visual_review"][field] = False
                self.assertTrue(any("visual_review" in error for error in self.validate()))
                self.image()["visual_review"][field] = True

    def test_visual_reviewer_required(self):
        self.image()["visual_review"]["reviewer"] = " "
        self.assertTrue(any("visual_review" in error for error in self.validate()))

    def test_missing_visual_review_rejected(self):
        del self.image()["visual_review"]
        self.assertTrue(any("visual_review" in error for error in self.validate()))

    def test_stale_media_id_rejected(self):
        self.image()["media_id"] = "old-upload"
        self.assertTrue(any("missing or stale" in error for error in self.validate()))

    def test_unrelated_draft_cannot_supply_media(self):
        self.readback["id"] = 999
        self.assertTrue(any("matching draft_id" in error for error in self.validate()))

    def test_multi_image_post_rejected_even_if_receipt_claims_single(self):
        self.readback["platforms"]["x"]["posts"][0]["media_ids"].append("second-image")
        self.assertTrue(any("multiple images" in error for error in self.validate()))

    def test_preview_group_size_required(self):
        del self.image()["preview_group_size"]
        self.assertTrue(any("preview_group_size=1" in error for error in self.validate()))

    def test_every_attached_media_requires_review_including_other_posts(self):
        self.readback["platforms"]["x"]["posts"].append({"media_ids": ["cover", "other"]})
        self.assertTrue(any("missing rendered image review" in error for error in self.validate()))

    def test_additional_draft_crop_rejected(self):
        additional = json.loads(json.dumps(self.receipt["platforms"]["x"]))
        additional.update(platform="x", draft_id=456)
        additional["images"][0].update(rendered_width=259, rendered_height=291)
        self.receipt["additional_drafts"] = [additional]
        additional_readback = json.loads(json.dumps(self.readback))
        additional_readback["id"] = 456
        self.readback = {"drafts": [self.readback, additional_readback]}
        errors = self.validate()
        self.assertTrue(any("additional draft 1 image 1: cover crops" in error for error in errors))

    def test_review_before_latest_draft_edit_is_stale(self):
        self.readback["updated_at"] = "2026-10-26T10:06:00Z"
        self.assertTrue(any("stale relative" in error for error in self.validate()))

    def test_missing_date_rejected(self):
        del self.receipt["checked_at"]
        self.assertTrue(any("checked_at" in error for error in self.validate()))

    def test_required_receipt_cannot_be_missing(self):
        self.assertTrue(validate_social_preview(self.package))

    def test_legacy_without_profiles_is_unaffected(self):
        self.assertEqual(validate_social_preview(self.package / "other"), [])

    def test_raw_readback_is_required_and_hashed(self):
        self.validate()
        self.receipt["platforms"]["x"]["raw_readback"]["sha256"] = "0" * 64
        self.write("social/profile-preview.json", self.receipt)
        self.assertTrue(any("raw_readback: sha256 mismatch" in error for error in validate_social_preview(self.package)))

    def test_nonfinite_dimensions_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.image()["rendered_width"] = value
                self.assertTrue(any("positive finite" in error for error in self.validate()))

    def document_pages(self):
        # Real two-page PDF with a page tree, cross-reference table and trailer.
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1400 1400] >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1400 1400] >>"]
        pdf = b"%PDF-1.4\n"
        offsets = [0]
        for index, value in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += f"{index} 0 obj\n".encode() + value + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 5\n0000000000 65535 f \n" + b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
        pdf += f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        (self.package / "comparison.pdf").write_bytes(pdf)
        # Exercise receipt logic without requiring optional PDF tools in the test environment.
        page_count = mock.patch("tools.check_social_preview._pdf_page_count", return_value=2)
        page_count.start()
        self.addCleanup(page_count.stop)
        images = self.receipt["platforms"]["linkedin"]["images"]
        self.receipt["platforms"]["linkedin"]["attachment_sources"][images[0]["media_id"]] = {"path": "comparison.pdf", "sha256": self.digest("comparison.pdf"), "media_type": "pdf", "page_count": 2}
        images[0].update(page_index=1, document_source_path="comparison.pdf", document_sha256=self.digest("comparison.pdf"), safe_content_box=[0, 0, 1200, 747], overlay_insets={"top": 0, "bottom": 0, "left": 0, "right": 0})
        images[0]["visual_review"]["controls_clear"] = True
        images.append({**images[0], "page_index": 2})
        return images

    def test_pdf_carousel_allows_one_media_id_with_distinct_reviewed_pages(self):
        images = self.document_pages()
        self.assertEqual(images[0]["media_id"], images[1]["media_id"])
        self.assertEqual(self.validate(), [])

    def test_pdf_carousel_rejects_duplicate_page_or_missing_pdf_hash(self):
        images = self.document_pages()
        images[1]["page_index"] = 1
        self.assertTrue(any("duplicate page_index" in error for error in self.validate()))
        images[1]["page_index"] = 2
        del images[1]["document_sha256"]
        self.assertTrue(any("document: sha256 mismatch" in error for error in self.validate()))
        (self.package / "comparison.pdf").write_bytes(b"not a PDF")
        for image in images:
            image["document_sha256"] = self.digest("comparison.pdf")
        self.assertTrue(any("%PDF magic" in error for error in self.validate()))

    def test_pdf_overlay_rejects_full_bleed_and_accepts_padded_content(self):
        images = self.document_pages()
        page = self.package / "document-page.png"
        for height, content in ((1400, [0, 0, 1400, 1400]), (1750, [0, 175, 1400, 1400])):
            page.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1400, height))
            for image in images:
                image.update(source_path="document-page.png", source_sha256=self.digest("document-page.png"), natural_width=1400, natural_height=height, rendered_width=560, rendered_height=height * 0.4, safe_content_box=content, overlay_insets={"top": 46, "bottom": 44, "left": 0, "right": 0})
            errors = self.validate()
            if height == 1400:
                self.assertTrue(any("intersects preview overlay controls" in error for error in errors))
            else:
                self.assertEqual(errors, [])

    def test_pdf_requires_every_page_review(self):
        self.document_pages().pop()
        self.assertTrue(any("every PDF page 1..2" in error for error in self.validate()))

    def test_pdf_overlay_rejects_ambiguous_contain_box_and_position(self):
        images = self.document_pages()
        for image in images:
            image.update(object_fit="contain", object_position="0% 0%", rendered_width=510, rendered_height=747, safe_content_box=[0, 60, 1200, 600], overlay_insets={"top": 40, "bottom": 0, "left": 0, "right": 0})
        errors = self.validate()
        self.assertTrue(any("matching source and rendered aspect ratios" in error for error in errors))
        self.assertTrue(any("centered object_position" in error for error in errors))

    def test_pdf_overlay_uses_uniform_contain_scale_with_rounding(self):
        images = self.document_pages()
        for image in images:
            image.update(object_fit="contain", rendered_width=510, rendered_height=318.5, safe_content_box=[0, 600, 1200, 100], overlay_insets={"top": 255.6, "bottom": 0, "left": 0, "right": 0})
        # Uniform contain puts content at 255.5125, while separate axis scales give 255.823.
        self.assertTrue(any("intersects preview overlay controls" in error for error in self.validate()))

    def test_pdf_cannot_masquerade_as_image_or_omit_document_fields(self):
        images = self.document_pages()
        images[:] = [images[0]]
        for field in ("page_index", "document_source_path", "document_sha256"):
            del images[0][field]
        errors = self.validate()
        self.assertTrue(any("PDF attachment requires matching" in error for error in errors))
        self.receipt["platforms"]["linkedin"]["attachment_sources"][images[0]["media_id"]]["media_type"] = "image"
        self.assertTrue(any("image attachment must have PNG signature" in error for error in self.validate()))

    def test_declared_pdf_page_count_must_match_retained_pdf(self):
        images = self.document_pages()
        self.receipt["platforms"]["linkedin"]["attachment_sources"][images[0]["media_id"]]["page_count"] = 1
        self.assertTrue(any("differs from actual retained PDF (2)" in error for error in self.validate()))

    def test_attachment_source_required_for_every_media(self):
        del self.receipt["platforms"]["x"]["attachment_sources"]
        self.assertTrue(any("attachment_sources" in error for error in self.validate()))

    def test_generic_manifest_requires_preview_receipt(self):
        (self.package / "profiles/profile-semantics.json").unlink()
        self.write("manifest.json", {"scientific_profiles": True})
        self.assertTrue(validate_social_preview(self.package))

    def test_generic_profiles_semantics_requires_preview_receipt(self):
        self.assertTrue(validate_social_preview(self.package))

    def internal_layout(self):
        internal = self.package / "_internal"
        internal.mkdir()
        (self.package / "social").rename(internal / "social")
        (self.package / "profiles").rename(internal / "profiles")
        (internal / "images").mkdir()
        (self.package / "profile.png").rename(internal / "images/profile.png")
        for spec in self.receipt["platforms"].values():
            spec["raw_readback"]["path"] = "_internal/social/readback.json"
            for image in spec["images"]:
                image["source_path"] = "_internal/images/profile.png"
            for source in spec["attachment_sources"].values():
                source["path"] = "_internal/images/profile.png"
        self.write("_internal/social/readback.json", self.readback)
        for spec in self.receipt["platforms"].values():
            spec["raw_readback"]["sha256"] = self.digest("_internal/social/readback.json")
        self.write("_internal/social/profile-preview.json", self.receipt)

    def test_internal_metadata_and_private_evidence(self):
        self.internal_layout()
        self.assertEqual(validate_social_preview(self.package), [])

    def test_internal_readback_can_verify_public_png(self):
        self.internal_layout()
        (self.package / "images").mkdir()
        (self.package / "_internal/images/profile.png").rename(self.package / "images/profile.png")
        for spec in self.receipt["platforms"].values():
            for image in spec["images"]:
                image["source_path"] = "images/profile.png"
            for source in spec["attachment_sources"].values():
                source["path"] = "images/profile.png"
        self.write("_internal/social/profile-preview.json", self.receipt)
        self.assertEqual(validate_social_preview(self.package), [])

    def test_internal_manifest_requires_receipt(self):
        (self.package / "profiles/profile-semantics.json").unlink()
        (self.package / "_internal").mkdir()
        self.write("_internal/manifest.json", {"scientific_profiles": True})
        errors = validate_social_preview(self.package)
        self.assertTrue(any("_internal/social/profile-preview.json" in error for error in errors))

    def test_internal_receipt_takes_precedence_over_legacy_receipt(self):
        self.assertEqual(self.validate(), [])
        (self.package / "_internal/profiles").mkdir(parents=True)
        self.write("_internal/profiles/profile-semantics.json", {})
        self.assertTrue(any("_internal/social/profile-preview.json" in error for error in validate_social_preview(self.package)))

    def test_explicit_receipt_override_and_path_boundary(self):
        self.assertEqual(self.validate(), [])
        (self.package / "social/profile-preview.json").rename(self.package / "social/custom.json")
        self.assertEqual(validate_social_preview(self.package, Path("social/custom.json")), [])
        self.assertTrue(validate_social_preview(self.package, Path("../outside.json")))

    def test_explicit_receipt_requires_file_without_marker(self):
        (self.package / "profiles/profile-semantics.json").unlink()
        self.assertTrue(validate_social_preview(self.package, Path("social/custom.json")))

    def test_default_receipt_symlink_cannot_escape_package(self):
        self.assertEqual(self.validate(), [])
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "receipt.json"
            external.write_text(json.dumps(self.receipt), encoding="utf-8")
            receipt = self.package / "social/profile-preview.json"
            receipt.unlink()
            receipt.symlink_to(external)
            self.assertTrue(any("receipt must stay within the package" in error for error in validate_social_preview(self.package)))

    def test_default_metadata_directory_symlink_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.package / "_internal").symlink_to(outside, target_is_directory=True)
            self.assertTrue(any("metadata must stay within the package" in error for error in validate_social_preview(self.package)))

    def test_default_metadata_file_symlinks_cannot_escape_package(self):
        for relative, label in (("manifest.json", "manifest"), ("profiles/profile-semantics.json", "semantics")):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as outside:
                path = self.package / relative
                original = path.read_bytes() if path.exists() else None
                path.unlink(missing_ok=True)
                external = Path(outside) / "metadata.json"
                external.write_text("{}", encoding="utf-8")
                path.symlink_to(external)
                self.assertTrue(any(f"{label} must stay within the package" in error for error in validate_social_preview(self.package)))
                path.unlink()
                if original is not None:
                    path.write_bytes(original)

    @unittest.skipUnless(shutil.which("pdfinfo") or importlib.util.find_spec("pypdf"), "optional PDF parser is not installed")
    def test_installed_parser_reads_real_two_page_pdf(self):
        self.document_pages()
        self.assertEqual(_pdf_page_count((self.package / "comparison.pdf").read_bytes()), 2)

    def test_missing_pdf_parser_fails_with_actionable_error(self):
        with mock.patch("tools.check_social_preview.shutil.which", return_value=None), mock.patch.dict("sys.modules", {"pypdf": None}):
            with self.assertRaisesRegex(ValueError, "existing pdfinfo or pypdf"):
                _pdf_page_count(b"%PDF-1.4\n")


if __name__ == "__main__":
    unittest.main()
