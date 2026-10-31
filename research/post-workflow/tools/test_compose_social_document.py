"""Verify generated PDFs and failure behavior with small, distinct RGB fixtures."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, RectangleObject

from tools.compose_social_document import _sources, _verify_pdf, compose


class ComposeSocialDocumentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = Path(self.temporary.name)
        (self.package / "images").mkdir()
        self.original_pixels = []
        self.assets = []
        for number, side in enumerate((3, 5), 1):
            pixels = bytes((position*17+number*29) % 256 for position in range(side*side*3))
            image = Image.frombytes("RGB", (side, side), pixels)
            relative = f"images/card-{number}.png"
            path = self.package / relative
            image.save(path)
            self.original_pixels.append(pixels)
            self.assets.append({"index": number, "title": f"Card {number}", "selected": {
                "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": side, "height": side,
            }})
        self.save_index()

    def save_index(self):
        (self.package / "images/asset-index.json").write_text(json.dumps({"assets": self.assets}))

    def test_selected_card_index_preserves_full_gallery(self):
        original = (self.package / "images/asset-index.json").read_bytes()
        selected = dict(self.assets[1], index=1)
        relative = "images/social-card-index.json"
        (self.package / relative).write_text(json.dumps({"assets": [selected]}))
        receipt = compose(self.package, "images/short-guide.pdf", asset_index=relative)
        self.assertEqual(receipt["document"]["page_count"], 1)
        self.assertEqual(receipt["asset_index"]["path"], relative)
        self.assertEqual(receipt["pages"][0]["source_png"], selected["selected"])
        self.assertEqual((self.package / "images/asset-index.json").read_bytes(), original)

    def test_selected_index_cannot_escape_package(self):
        with self.assertRaises(ValueError):
            compose(self.package, "images/short-guide.pdf", asset_index="../outside.json")
        self.assertFalse((self.package / "images/short-guide.pdf").exists())

    def test_actual_pdf_page_order_pixels_transform_and_receipt(self):
        receipt = compose(self.package, "images/resource-guide.pdf")
        path = self.package / "images/resource-guide.pdf"
        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 2)
        self.assertEqual(receipt["document"]["page_count"], 2)
        self.assertEqual(receipt["document"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.package / "images/resource-guide.composition.json").read_text()), receipt)
        for number, page in enumerate(reader.pages):
            self.assertEqual(list(page.mediabox), [0, 0, 540, 604.8])
            embedded_images = [value.get_object() for value in page["/Resources"]["/XObject"].values()
                               if value.get_object().get("/Subtype") == "/Image"]
            self.assertEqual(len(embedded_images), 1)
            image = embedded_images[0]
            self.assertEqual(image["/ColorSpace"], "/DeviceRGB")
            self.assertEqual(image["/BitsPerComponent"], 8)
            self.assertEqual(image.get_data(), self.original_pixels[number])
            # Confirm the pixels through pypdf's image decoding as well.
            self.assertEqual(list(page.images)[0].image.tobytes(), self.original_pixels[number])
            operations = ContentStream(page.get_contents(), reader).operations
            transforms = [[float(value) for value in args] for args, op in operations if op == b"cm"]
            self.assertIn([421.2, 0.0, 0.0, 421.2, 59.4, 59.4], transforms)
            self.assertEqual(sum(op == b"Do" for _, op in operations), 1)
            self.assertFalse(any(op in (b"Tj", b"TJ") for _, op in operations))
            page_receipt = receipt["pages"][number]
            self.assertEqual(page_receipt["page_index"], number+1)
            self.assertEqual(page_receipt["source_png"], self.assets[number]["selected"])
            self.assertEqual(page_receipt["image_box_points"], [59.400000000000006, 59.400000000000006, 421.2, 421.2])
            self.assertAlmostEqual(page_receipt["image_box_points"][0]*2+421.2, 540)
            self.assertAlmostEqual(page_receipt["image_box_points"][1]+421.2
                                   + receipt["outer_band_points"]["top"], 604.8)
        self.assertEqual(receipt["extra_top_clearance_ratio"], .12)
        self.assertAlmostEqual(receipt["outer_band_points"]["top"], 124.2)
        self.assertAlmostEqual(receipt["outer_band_points"]["bottom"], 59.4)
        self.assertTrue(receipt["validation"]["all_image_transforms_horizontally_centered"])
        self.assertFalse(receipt["validation"]["all_image_transforms_centered"])
        self.assertTrue(receipt["top_clearance_source"].startswith("style/visual-tokens.json:"))

    def test_custom_scale_changes_placement_without_changing_pixels(self):
        receipt = compose(self.package, "images/smaller-guide.pdf", .5)
        self.assertEqual(receipt["pages"][0]["image_box_points"], [135.0, 135.0, 270.0, 270.0])
        reader = PdfReader(self.package / "images/smaller-guide.pdf")
        self.assertEqual(list(reader.pages[1].images)[0].image.tobytes(), self.original_pixels[1])

    def test_zero_clearance_retains_square_page_and_identical_image_scale(self):
        rectangular = compose(self.package, "images/rectangular.pdf")
        square = compose(self.package, "images/square.pdf", .78, top_clearance=0)
        self.assertEqual(square["page_size_points"], {"width": 540, "height": 540})
        self.assertEqual(square["pages"][0]["image_transform"], rectangular["pages"][0]["image_transform"])
        self.assertEqual(square["pages"][0]["decoded_rgb_sha256"], rectangular["pages"][0]["decoded_rgb_sha256"])
        self.assertTrue(square["validation"]["all_image_transforms_centered"])
        self.assertEqual(square["top_clearance_source"], "explicit_argument")
        self.assertIsNone(square["shared_visual_tokens"])

    def test_actual_page_box_and_image_placement_are_verified(self):
        compose(self.package, "images/verified.pdf")
        data = (self.package / "images/verified.pdf").read_bytes()
        _, sources = _sources(self.package.resolve())
        for failure in ("height", "crop", "wrong_clearance", "offset"):
            with self.subTest(failure=failure):
                reader = PdfReader(io.BytesIO(data))
                writer = PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                if failure == "height":
                    writer.pages[0].mediabox = RectangleObject([0, 0, 540, 600])
                elif failure == "crop":
                    writer.pages[0].cropbox = RectangleObject([0, 10, 540, 604.8])
                elif failure == "offset":
                    writer.pages[0].add_transformation((1, 0, 0, 1, 1, 0))
                stream = io.BytesIO()
                writer.write(stream)
                with self.assertRaisesRegex(ValueError, "image placement" if failure == "offset" else "unexpected page box"):
                    _verify_pdf(stream.getvalue(), sources, .78, 0 if failure == "wrong_clearance" else .12)

    def test_bad_later_hash_fails_before_creating_outputs(self):
        self.assets[1]["selected"]["sha256"] = "0"*64
        self.save_index()
        with self.assertRaisesRegex(ValueError, "hash/signature mismatch"):
            compose(self.package, "images/new-directory/must-not-exist.pdf")
        self.assertFalse((self.package / "images/new-directory").exists())

    def test_missing_nonsquare_and_wrong_dimensions_fail_before_write(self):
        for failure in ("missing", "nonsquare", "wrong_dimensions"):
            with self.subTest(failure=failure):
                selected = self.assets[0]["selected"]
                path = self.package / selected["path"]
                original = path.read_bytes()
                previous = dict(selected)
                try:
                    if failure == "missing":
                        path.unlink()
                    elif failure == "nonsquare":
                        Image.new("RGB", (3, 4), "red").save(path)
                        selected.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), height=4)
                    else:
                        selected["width"] = 9
                    self.save_index()
                    with self.assertRaises((ValueError, OSError)):
                        compose(self.package, f"images/{failure}.pdf")
                    self.assertFalse((self.package / f"images/{failure}.pdf").exists())
                    self.assertFalse((self.package / f"images/{failure}.composition.json").exists())
                finally:
                    path.write_bytes(original)
                    selected.clear()
                    selected.update(previous)
                    self.save_index()

    def test_existing_pdf_and_receipt_are_never_overwritten(self):
        path = self.package / "images/gamma-original.pdf"
        path.write_bytes(b"%PDF-original-Gamma-content")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            compose(self.package, "images/gamma-original.pdf")
        self.assertEqual(path.read_bytes(), b"%PDF-original-Gamma-content")
        receipt_path = self.package / "images/already-owned.composition.json"
        receipt_path.write_text("existing receipt")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            compose(self.package, "images/already-owned.pdf")
        self.assertEqual(receipt_path.read_text(), "existing receipt")
        self.assertFalse((self.package / "images/already-owned.pdf").exists())

    def test_invalid_scale_and_output_paths_are_rejected(self):
        for scale in (0, -1, 1, 2, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                compose(self.package, "images/invalid.pdf", scale)
        for clearance in (-1, True, "0.12", float("nan"), float("inf"), 1e308):
            with self.subTest(top_clearance=clearance), self.assertRaises(ValueError):
                compose(self.package, "images/invalid.pdf", top_clearance=clearance)
        for output in ("../outside.pdf", "images/../escape.pdf", "/tmp/outside.pdf", "evidence/guide.pdf", "images/source.png"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                compose(self.package, output)


if __name__ == "__main__":
    unittest.main()
