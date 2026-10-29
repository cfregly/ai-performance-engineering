"""Control-plane checks for the capture-first qualified CPU producer."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("run_cpu_study.py")
SPEC = importlib.util.spec_from_file_location("_run_cpu_study_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeError(Exception):
    def __init__(self, status: int):
        super().__init__(f"status {status}")
        self.status = status


class FakeClient:
    def __init__(self, outcome: object):
        self.outcome = outcome
        self.calls: list[str] = []

    def get_run(self, run_id: str) -> object:
        self.calls.append(run_id)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class CaptureFirstProducerTest(unittest.TestCase):
    def test_scientific_method_ast_matches_configured_api_main(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        expected = {
            "toolchain_input": "4b3f0d11cd9f263c408e33453cc3aabcc1acc9f747242e20f99d08fd97bdfdb1",
            "prepare": "7f4262187c0e8cf0622aebfb162cffb7cb3717970a16d6a5a4990c0fc88f8bfd",
        }
        observed = {
            node.name: hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in expected
        }
        self.assertEqual(observed, expected)

    def test_capture_has_no_client_or_network_boundary(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        capture = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "capture"
        )
        names = {node.id for node in ast.walk(capture) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(capture) if isinstance(node, ast.Attribute)
        }
        self.assertIn("capture_manifest", attributes)
        self.assertTrue(
            {"PIQClient", "socket", "urlopen", "requests"}.isdisjoint(
                names | attributes
            )
        )
        retain = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "retain"
        )
        retain_attributes = {
            node.attr for node in ast.walk(retain) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("capture_manifest", retain_attributes)
        self.assertTrue(
            {"resume_capture", "complete_capture_custody"} <= retain_attributes
        )

    def test_capture_rejects_api_credentials(self) -> None:
        MODULE.require_capture_environment({})
        with self.assertRaisesRegex(ValueError, "capture refuses PIQ API credentials"):
            MODULE.require_capture_environment({"PIQ_API_TOKEN": "secret"})
        with self.assertRaisesRegex(ValueError, "PIQ_API_TOKEN, PIQ_DECISION_TOKEN"):
            MODULE.require_capture_environment(
                {"PIQ_API_TOKEN": "secret", "PIQ_DECISION_TOKEN": "secret"}
            )

    def test_retention_resumes_only_after_exact_404(self) -> None:
        missing = FakeClient(FakeError(404))
        self.assertEqual(
            MODULE.retention_action(missing, "run-1"), "resume-absent-capture"
        )
        self.assertEqual(missing.calls, ["run-1"])
        existing = FakeClient({"run": {"runId": "run-1"}})
        self.assertEqual(
            MODULE.retention_action(existing, "run-1"), "complete-existing-custody"
        )
        unavailable = FakeClient(FakeError(503))
        with self.assertRaisesRegex(
            RuntimeError, "outcome is unknown; refuse retention write"
        ):
            MODULE.retention_action(unavailable, "run-1")

    def test_only_current_three_stage_modes_are_exposed(self) -> None:
        parser = MODULE.build_parser()
        command = next(action for action in parser._actions if action.dest == "command")
        api_url = next(action for action in parser._actions if action.dest == "api_url")
        self.assertEqual(tuple(command.choices), ("prepare", "capture", "retain"))
        self.assertTrue(api_url.required)
        source = SOURCE.read_text(encoding="utf-8")
        for obsolete in ("from piq_api", "import uvicorn", "Settings(", "create_app("):
            self.assertNotIn(obsolete, source)

    def test_retention_io_uses_known_write_budget_without_retries(self) -> None:
        self.assertEqual(MODULE.RETENTION_IO_TIMEOUT_SECONDS, 120)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        retain = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "retain"
        )
        constructor = next(
            node
            for node in ast.walk(retain)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PIQClient"
        )
        keywords = {item.arg: item.value for item in constructor.keywords}
        self.assertIsInstance(keywords["timeout"], ast.Name)
        self.assertEqual(keywords["timeout"].id, "RETENTION_IO_TIMEOUT_SECONDS")
        self.assertEqual(ast.literal_eval(keywords["max_retries"]), 0)


if __name__ == "__main__":
    unittest.main()
