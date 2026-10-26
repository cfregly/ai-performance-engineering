from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_capture.py")
SPEC = importlib.util.spec_from_file_location("prepare_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_capture)


class PrepareCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        wrapper = self.repo / "code/core/scripts/profiling/profile.sh"
        benchmark = (
            self.repo / "research/2026-10-22_nested-graph-breaks/benchmark.py"
        )
        wrapper.parent.mkdir(parents=True)
        benchmark.parent.mkdir(parents=True)
        wrapper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        wrapper.chmod(0o755)
        benchmark.write_text("print('fixture')\n", encoding="utf-8")
        self.injection = self.root / "libzymtracecudaprofiler.so"
        self.injection.write_bytes(b"synthetic injection library")
        self.owner_checkpoint = self.root / "OWNER-CHECKPOINT.json"
        self.owner_checkpoint.write_text('{"owner":"fixture"}\n', encoding="utf-8")
        self.run_root = self.root / "new-run"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "repo_root": self.repo,
            "run_root": self.run_root,
            "python": Path(sys.executable),
            "injection_library": self.injection,
            "runtime_owner_checkpoint": self.owner_checkpoint,
            "collection_agent": "zymtrace.internal.example:443",
            "zymtrace_project": "nested-graph-breaks",
            "expected_gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "tls_mode": "tls",
            "timeout_seconds": 240,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_builds_two_nonexecuting_repository_wrapper_commands(self) -> None:
        plan = prepare_capture.build_plan(self._args())

        self.assertEqual(plan["status"], "dry_run_only")
        self.assertTrue(plan["not_executed"])
        self.assertEqual(
            plan["zymtrace"]["backend_and_profiler_version_pin"], "26.10.1"
        )
        self.assertEqual([arm["enabled"] for arm in plan["arms"]], [0, 1])
        self.assertFalse(self.run_root.exists())
        for arm in plan["arms"]:
            command = arm["argv"]
            self.assertEqual(command[0], "timeout")
            self.assertIn("--tool", command)
            self.assertIn("zymtrace", command)
            self.assertIn(
                "ZYMTRACE_CUDAPROFILER__ENABLE_PC_SAMPLING=false", command
            )
            self.assertIn(
                "ZYMTRACE_CUDAPROFILER__STACK_TRACE_SAMPLING__ENABLED=true",
                command,
            )
            self.assertIn(
                "ZYMTRACE_CUDAPROFILER__STACK_TRACE_SAMPLING__TARGET=200",
                command,
            )
            self.assertFalse(
                any("GRAPH_SAMPLING" in argument for argument in command)
            )
            self.assertNotIn("ZYMTRACE_AUTH_TOKEN", arm["shell_command"])
            self.assertNotIn("--profile", command)
            self.assertNotIn("zymtrace.internal.example:443", command)
            self.assertNotIn("nested-graph-breaks", command)

        self.assertEqual(
            plan["capture_policy"]["experiment_matrix_coverage"]["cells"],
            "2 of 24",
        )
        self.assertTrue(plan["capture_policy"]["qualification_use_prohibited"])
        self.assertEqual(
            plan["capture_policy"]["maximum_total_wall_seconds_including_kill_grace"],
            540,
        )
        self.assertEqual(
            plan["capture_policy"]["kernel_launch_sampling_target_per_thread_per_second"],
            200,
        )

    def test_endpoint_and_project_are_host_profiler_preconditions(self) -> None:
        plan = prepare_capture.build_plan(self._args())

        endpoint = plan["zymtrace"]["collection_agent_precondition"]
        project = plan["zymtrace"]["project_precondition"]
        self.assertEqual(endpoint["value"], "zymtrace.internal.example:443")
        self.assertFalse(endpoint["passed_to_workload"])
        self.assertEqual(project["value"], "nested-graph-breaks")
        self.assertFalse(project["passed_to_workload"])

    def test_records_distinct_source_and_owner_hashes(self) -> None:
        plan = prepare_capture.build_plan(self._args())

        self.assertEqual(len(plan["source"]["benchmark"]["sha256"]), 64)
        self.assertEqual(len(plan["source"]["profile_wrapper"]["sha256"]), 64)
        self.assertEqual(
            len(plan["runtime_ownership"]["checkpoint"]["sha256"]), 64
        )
        self.assertNotEqual(
            plan["arms"][0]["environment"]["TORCHINDUCTOR_CACHE_DIR"],
            plan["arms"][1]["environment"]["TORCHINDUCTOR_CACHE_DIR"],
        )

    def test_preserves_virtualenv_python_invocation_symlink(self) -> None:
        target = self.root / "python3.12"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
        invocation = self.root / "venv/bin/python"
        invocation.parent.mkdir(parents=True)
        invocation.symlink_to(target)

        plan = prepare_capture.build_plan(self._args(python=invocation))

        self.assertEqual(plan["source"]["python"], str(invocation))
        for arm in plan["arms"]:
            python_index = arm["argv"].index("--python") + 1
            self.assertEqual(arm["argv"][python_index], str(invocation))

    def test_rejects_placeholder_or_url_collection_agent(self) -> None:
        for endpoint in ("ZYMTRACE_URL", "https://zymtrace.example:443/mcp"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(prepare_capture.PlanError):
                    prepare_capture.build_plan(
                        self._args(collection_agent=endpoint)
                    )

    def test_rejects_short_or_noncanonical_gpu_uuid(self) -> None:
        invalid_uuids = (
            "GPU-12345678",
            "GPU-12345678-1234-1234-1234-123456789ab",
            "GPU-12345678-1234-1234-1234-123456789abc-extra",
        )
        for gpu_uuid in invalid_uuids:
            with self.subTest(gpu_uuid=gpu_uuid):
                with self.assertRaisesRegex(
                    prepare_capture.PlanError, "full GPU-... UUID"
                ):
                    prepare_capture.build_plan(
                        self._args(expected_gpu_uuid=gpu_uuid)
                    )

    def test_rejects_existing_run_root(self) -> None:
        self.run_root.mkdir()
        with self.assertRaisesRegex(
            prepare_capture.PlanError, "must not already exist"
        ):
            prepare_capture.build_plan(self._args())

    def test_rejects_missing_owner_checkpoint(self) -> None:
        missing = self.root / "missing-owner-checkpoint"
        with self.assertRaisesRegex(
            prepare_capture.PlanError, "runtime owner checkpoint"
        ):
            prepare_capture.build_plan(
                self._args(runtime_owner_checkpoint=missing)
            )

    def test_rejects_timeout_beyond_visibility_smoke_bound(self) -> None:
        with self.assertRaisesRegex(prepare_capture.PlanError, "between 60 and 300"):
            prepare_capture.build_plan(self._args(timeout_seconds=301))


if __name__ == "__main__":
    unittest.main()
