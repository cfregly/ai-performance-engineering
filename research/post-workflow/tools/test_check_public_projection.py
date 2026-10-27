from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.check_public_projection import validate_public_projection


class PublicProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        self.package.mkdir()
        self.sources = {
            "README.md": "article.md",
            "images/mechanism.png": "images/mechanism.png",
            "code/run.py": "code/run.py",
            "captures/run.nsys-rep": "captures/run.nsys-rep",
            "evidence/hashes.txt": "evidence/hashes.txt",
        }
        self.write("article.md", "# One mechanism\n\n![Mechanism](images/mechanism.png)\n")
        self.write("images/mechanism.png", b"\x89PNG\x00synthetic fixture")
        self.write("code/run.py", "print('at::native::binary_internal::MulFunctor')\n")
        self.write("captures/run.nsys-rep", b"\x00\xffnative capture fixture")
        self.write("evidence/hashes.txt", "run.nsys-rep: retained native capture\n")
        self.stage()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, content: str | bytes) -> None:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def manifest(self, declaration: object | None = None) -> None:
        if declaration is None:
            declaration = {"directory": "public", "sources": self.sources}
        self.write("manifest.json", json.dumps({"public_projection": declaration}))

    def stage(self) -> None:
        for public, source in self.sources.items():
            target = self.package / "public" / public
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.package / source, target)
        self.manifest()

    def errors(self) -> list[str]:
        return validate_public_projection(self.package)

    def assert_error(self, text: str) -> None:
        self.assertTrue(any(text in error for error in self.errors()), self.errors())

    def test_clean_projection_with_native_captures_and_evidence(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_historical_package_is_unchanged(self) -> None:
        self.write("manifest.json", '{"content_state": "hypothetical"}')
        self.write("public/private.txt", "Typefully")
        self.assertEqual(self.errors(), [])
        (self.package / "manifest.json").unlink()
        self.assertEqual(self.errors(), [])

    def test_internal_directory_even_when_empty(self) -> None:
        (self.package / "public/code/__INTERNAL").mkdir()
        self.assert_error("internal paths")

    def test_private_text_is_rejected(self) -> None:
        for text in ("../_internal/note", "Typefully draft", "VERIFICATION.md", "## post this", "<!-- private note -->", "images/review.json"):
            with self.subTest(text=text):
                self.write("article.md", f"# Article\n{text}\n![a](images/mechanism.png)\n")
                self.stage()
                self.assertTrue(self.errors())

    def test_code_reference_is_checked(self) -> None:
        self.write("code/run.py", "# Read sources.json\n")
        self.stage()
        self.assert_error("pipeline file reference")

    def test_binary_internal_symbol_is_valid(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_missing_source_is_rejected(self) -> None:
        (self.package / "images/mechanism.png").unlink()
        self.assert_error("retained source file is missing")

    def test_changed_staged_bytes_are_rejected(self) -> None:
        self.write("public/code/run.py", "print('changed')\n")
        self.assert_error("sha256 mismatch")

    def test_missing_and_extra_staged_files_are_rejected(self) -> None:
        (self.package / "public/code/run.py").unlink()
        self.write("public/code/extra.py", "print(1)\n")
        self.assert_error("declared public file is missing")
        self.assert_error("not declared")

    def test_missing_shown_image_is_rejected(self) -> None:
        self.write("article.md", "# Article\n![missing](images/missing.png)\n")
        self.stage()
        self.assert_error("shown image is missing")

    def test_unshown_png_is_rejected(self) -> None:
        self.write("article.md", "# Article\n")
        self.stage()
        self.assert_error("PNG is not shown")

    def test_nested_html_and_reference_images_are_recognized(self) -> None:
        self.write("article.md", '# Article\n<img src="images/mechanism.png">\n')
        self.stage()
        self.assertEqual(self.errors(), [])
        self.write("article.md", "# Article\n![Mechanism][view]\n\n[view]: images/mechanism.png\n")
        self.stage()
        self.assertEqual(self.errors(), [])

    def test_empty_article_is_rejected(self) -> None:
        self.write("article.md", " \n")
        self.stage()
        self.assert_error("must be nonempty")

    def test_source_and_output_symlinks_are_rejected(self) -> None:
        source = self.package / "code/run.py"
        source.unlink()
        source.symlink_to(self.package / "article.md")
        self.assert_error("symlinks are not allowed")
        source.unlink()
        self.write("code/run.py", "print('restored')\n")
        self.stage()
        output = self.package / "public/code/run.py"
        output.unlink()
        output.symlink_to(source)
        self.assert_error("symlinks are not allowed")

    def test_symlinked_source_parent_is_rejected(self) -> None:
        (self.package / "source-link").symlink_to(self.package / "code", target_is_directory=True)
        self.sources["code/run.py"] = "source-link/run.py"
        self.manifest()
        self.assert_error("symlinks are not allowed")

    def test_symlinked_output_parent_is_rejected(self) -> None:
        (self.package / "stage-link").symlink_to(self.package / "public", target_is_directory=True)
        self.manifest({"directory": "stage-link", "sources": self.sources})
        self.assert_error("symlinks are not allowed")

    def test_traversal_is_rejected_for_every_declared_path(self) -> None:
        for field in ("directory", "source", "public"):
            with self.subTest(field=field):
                declaration = {"directory": "public", "sources": dict(self.sources)}
                if field == "directory":
                    declaration["directory"] = "../outside"
                elif field == "source":
                    declaration["sources"]["code/run.py"] = "../outside.py"
                else:
                    declaration["sources"]["../outside.py"] = "code/run.py"
                self.manifest(declaration)
                self.assert_error("safe")

    def test_source_inside_staged_output_is_rejected(self) -> None:
        self.sources["code/run.py"] = "public/code/run.py"
        self.manifest()
        self.assert_error("source must not be inside")

    def test_unsupported_public_top_level_file_is_rejected(self) -> None:
        self.write("private.md", "not public")
        self.sources["private.md"] = "private.md"
        self.stage()
        self.assert_error("public files must be README.md")

    def test_image_path_traversal_is_rejected(self) -> None:
        self.write("article.md", "# Article\n![bad](../images/mechanism.png)\n")
        self.stage()
        self.assert_error("image path escapes")

    def test_cli_checks_only_the_given_package(self) -> None:
        script = Path(__file__).with_name("check_public_projection.py")
        result = subprocess.run([sys.executable, str(script), str(self.package)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write("public/code/unlisted.py", "print(1)\n")
        result = subprocess.run([sys.executable, str(script), str(self.package)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not declared", result.stdout)


if __name__ == "__main__":
    unittest.main()
