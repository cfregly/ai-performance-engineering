from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.check_public_projection import (
    _PRIVATE_IDENTITY_SCAN_CHUNK_BYTES,
    validate_public_projection,
)
from tools.qualified_gpu_training import build_gpu_public_summary, canonical_json
from tools.test_qualified_gpu_training import build_gpu_fixture


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

    def make_gpu_v3_projection(self) -> dict:
        receipt = build_gpu_fixture(self.package, synthetic=False)
        summary_path = self.package / "evidence/gpu-training-public-summary.json"
        summary_path.write_bytes(canonical_json(build_gpu_public_summary(self.package, receipt)))
        self.sources["evidence/gpu-training-summary.json"] = "evidence/gpu-training-public-summary.json"
        self.stage()
        manifest = {
            "content_state": "measured",
            "measurement_receipt": "evidence/measurement-receipt.json",
            "measurement_contract": "qualified_gpu_training_v3",
            "gpu_training_public_summary": "evidence/gpu-training-public-summary.json",
            "public_projection": {"directory": "public", "sources": self.sources},
        }
        self.write("manifest.json", json.dumps(manifest, indent=2) + "\n")
        return receipt

    def assert_error(self, text: str) -> None:
        self.assertTrue(any(text in error for error in self.errors()), self.errors())

    def test_clean_projection_with_native_captures_and_evidence(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_gpu_v3_allows_only_deterministic_public_summary(self) -> None:
        self.make_gpu_v3_projection()
        self.assertEqual([], self.errors())

    def test_legacy_measured_projection_contracts_are_unsupported(self) -> None:
        for version, contract in ((1, None), (2, "qualified_cpu_v2")):
            with self.subTest(version=version):
                manifest = {
                    "content_state": "measured",
                    "measurement_receipt": "evidence/measurement-receipt.json",
                    "public_projection": {"directory": "public", "sources": self.sources},
                }
                if contract is not None:
                    manifest["measurement_contract"] = contract
                self.write("evidence/measurement-receipt.json", json.dumps({"schema_version": version}))
                self.write("manifest.json", json.dumps(manifest))
                self.assert_error("only qualified_gpu_training_v3 is active")

    def test_gpu_v3_rejects_private_evidence_mapping(self) -> None:
        receipt = self.make_gpu_v3_projection()
        private_sources = {
            "receipt": "evidence/measurement-receipt.json",
            "design": receipt["design"]["path"],
            "qualification-report": receipt["qualification_report"]["path"],
            "raw-census": receipt["raw_census"]["path"],
        }
        for name, source in private_sources.items():
            with self.subTest(name=name):
                public = f"evidence/private-{name}.json"
                self.sources[public] = source
                self.stage()
                manifest = {
                    "content_state": "measured",
                    "measurement_receipt": "evidence/measurement-receipt.json",
                    "measurement_contract": "qualified_gpu_training_v3",
                    "gpu_training_public_summary": "evidence/gpu-training-public-summary.json",
                    "public_projection": {"directory": "public", "sources": self.sources},
                }
                self.write("manifest.json", json.dumps(manifest))
                self.assert_error("forbidden private evidence")
                del self.sources[public]
                (self.package / "public" / public).unlink()

    def test_gpu_v3_rejects_private_identifier_leak(self) -> None:
        receipt = self.make_gpu_v3_projection()
        summary_path = self.package / "evidence/gpu-training-public-summary.json"
        summary = json.loads(summary_path.read_text())
        envelope = json.loads((self.package / receipt["piq_envelope"]["path"]).read_text())
        summary["scope_and_limits"]["scope"] += " " + envelope["attempts"][0]["run_id"]
        summary_path.write_bytes(canonical_json(summary))
        shutil.copyfile(summary_path, self.package / "public/evidence/gpu-training-summary.json")
        self.assert_error("leaks a private execution identity")

    def test_gpu_v3_rejects_private_run_id_in_readme(self) -> None:
        receipt = self.make_gpu_v3_projection()
        envelope = json.loads((self.package / receipt["piq_envelope"]["path"]).read_text())
        run_id = envelope["attempts"][0]["run_id"]
        article = f"# One mechanism\n\nRun: {run_id}\n\n![Mechanism](images/mechanism.png)\n"
        self.write("article.md", article)
        self.write("public/README.md", article)
        self.assert_error("README.md: staged public file leaks a private execution identity")

    def test_gpu_v3_rejects_private_assignment_id_in_staged_text(self) -> None:
        receipt = self.make_gpu_v3_projection()
        envelope_path = self.package / receipt["piq_envelope"]["path"]
        envelope = json.loads(envelope_path.read_text())
        assignment_ref = envelope["executions"][0]["assignment_evidence"]["path"]
        assignment = json.loads((envelope_path.parent / assignment_ref).read_text())
        text = f"retained capture job: {assignment['scheduler_job_id']}\n"
        self.write("evidence/hashes.txt", text)
        self.write("public/evidence/hashes.txt", text)
        self.assert_error(
            "evidence/hashes.txt: staged public file leaks a private execution identity"
        )

    def test_gpu_v3_scans_binary_metadata_across_chunk_boundaries(self) -> None:
        receipt = self.make_gpu_v3_projection()
        envelope = json.loads((self.package / receipt["piq_envelope"]["path"]).read_text())
        run_id = envelope["attempts"][0]["run_id"].encode()
        prefix = b"\x00" + b"x" * (
            _PRIVATE_IDENTITY_SCAN_CHUNK_BYTES - len(run_id) // 2 - 1
        )
        capture = prefix + run_id + b"\xff"
        self.write("captures/run.nsys-rep", capture)
        self.write("public/captures/run.nsys-rep", capture)
        self.assert_error(
            "captures/run.nsys-rep: staged public file leaks a private execution identity"
        )

    def test_gpu_v3_wrong_private_report_hash_and_nonqualified_summary_fail(self) -> None:
        receipt = self.make_gpu_v3_projection()
        receipt["qualification_report"]["sha256"] = "0" * 64
        self.write("evidence/measurement-receipt.json", json.dumps(receipt, indent=2) + "\n")
        self.assert_error("requires a qualified receipt")

    def test_manifest_duplicate_key_is_rejected_strictly(self) -> None:
        self.write(
            "manifest.json",
            '{"public_projection": {"directory": "public", "sources": {"README.md": "article.md"}}, '
            '"public_projection": {"directory": "public", "sources": {"README.md": "article.md"}}}',
        )
        self.assert_error("duplicate JSON key")

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
