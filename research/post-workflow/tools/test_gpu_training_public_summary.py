from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.project_gpu_training_public_summary import project
from tools.qualified_gpu_training import (
    EvidenceError, build_gpu_public_summary, canonical_json, strict_load_json,
)
from tools.test_qualified_gpu_training import build_gpu_fixture


class GpuTrainingPublicSummaryTests(unittest.TestCase):
    def test_new_schemas_parse_and_every_declared_object_is_closed(self) -> None:
        root = Path(__file__).resolve().parent.parent
        schema_paths = [
            root / "schemas/measurement-receipt-v3.schema.json",
            root / "schemas/piq-experiment-evidence-v2.schema.json",
            root / "schemas/gpu-training-public-summary-v1.schema.json",
        ]

        def open_objects(value: object, location: str = "$") -> list[str]:
            found: list[str] = []
            if isinstance(value, dict):
                if (value.get("type") == "object"
                        and value.get("additionalProperties") is not False):
                    found.append(location)
                for key, child in value.items():
                    found.extend(open_objects(child, f"{location}.{key}"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    found.extend(open_objects(child, f"{location}[{index}]"))
            return found

        for path in schema_paths:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual([], open_objects(schema))

    def test_summary_is_deterministic_and_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package, synthetic=False)
            first = build_gpu_public_summary(package, receipt)
            second = build_gpu_public_summary(package, copy.deepcopy(receipt))
            self.assertEqual(canonical_json(first), canonical_json(second))
            self.assertEqual(
                {"schema_version", "kind", "contract_id", "qualification", "method",
                 "results", "evidence", "scope_and_limits"},
                set(first),
            )

    def test_private_identities_do_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package, synthetic=False)
            envelope = strict_load_json(package / receipt["piq_envelope"]["path"])
            summary_text = json.dumps(build_gpu_public_summary(package, receipt), sort_keys=True)
            private_values = [
                *(item["id"] for item in envelope["attempts"]),
                *(item["run_id"] for item in envelope["attempts"]),
                *(item["observed_resources"]["host"] for item in envelope["executions"]),
                *(uuid for item in envelope["executions"] for uuid in item["observed_resources"]["device_uuids"]),
            ]
            for private in private_values:
                self.assertNotIn(private, summary_text)
            self.assertNotIn("cgroup", summary_text.lower())
            self.assertNotIn("owner", summary_text.lower())

    def test_projector_writes_canonical_bytes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            receipt = build_gpu_fixture(package, synthetic=False)
            output = package / "public-summary.json"
            project(package, output)
            self.assertEqual(
                canonical_json(build_gpu_public_summary(package, receipt)),
                output.read_bytes(),
            )
            with self.assertRaisesRegex(EvidenceError, "already exists"):
                project(package, output)

    def test_nonqualified_or_wrong_report_binding_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package, synthetic=False)
            receipt["qualification_report"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(EvidenceError, "non-qualified"):
                build_gpu_public_summary(package, receipt)

    def test_synthetic_fixture_cannot_be_projected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            with self.assertRaisesRegex(EvidenceError, "synthetic"):
                build_gpu_public_summary(package, receipt)


if __name__ == "__main__":
    unittest.main()
