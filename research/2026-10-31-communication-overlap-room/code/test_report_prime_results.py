"""Check the offline report's arithmetic, setup and input errors."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parent
PROGRAM = DIRECTORY / "report_prime_results.py"


class ReportTests(unittest.TestCase):
    def invoke(self, *arguments: str, cwd: Path | None = None):
        return subprocess.run(
            [sys.executable, str(PROGRAM), *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def reject_edited_data(self, edit):
        data = json.loads((DIRECTORY / "prime_results.json").read_text())
        edit(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edited.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = self.invoke("--data", str(path))
        self.assertEqual(result.returncode, 1, result)
        self.assertEqual(result.stdout, "")
        self.assertIn("Cannot read source timings:", result.stderr)
        return result

    def test_report_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(cwd=Path(directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("Authors: Mario Sieg / Prime Intellect", result.stdout)
        self.assertIn("Qwen3-30B-A3B | 32 | 2.78 | 2.08 | 1.337x | 25.2% | 34%", result.stdout)
        self.assertIn("Qwen3-235B-A22B | 128 | 12.72 | 9.65 | 1.318x | 24.1% | 32%", result.stdout)
        self.assertIn("not a benchmark run", result.stdout)

    def test_help(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--data", result.stdout)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke("--data", str(Path(directory) / "absent.json"))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Cannot read source timings:", result.stderr)

    def test_wrong_units(self):
        result = self.reject_edited_data(lambda data: data.update(unit="milliseconds"))
        self.assertIn("seconds_per_step", result.stderr)

    def test_zero_time(self):
        result = self.reject_edited_data(
            lambda data: data["results"][0].update(candidate_seconds=0)
        )
        self.assertIn("positive finite number", result.stderr)

    def test_overflow_time(self):
        result = self.reject_edited_data(
            lambda data: data["results"][0].update(baseline_seconds=10**400)
        )
        self.assertIn("finite numeric value", result.stderr)

    def test_nonfinite_time(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                result = self.reject_edited_data(
                    lambda data: data["results"][0].update(candidate_seconds=value)
                )
                self.assertIn("positive finite number", result.stderr)

    def test_missing_author(self):
        result = self.reject_edited_data(lambda data: data.pop("author"))
        self.assertIn("name the source author", result.stderr)


if __name__ == "__main__":
    unittest.main()
