"""Repository-root CI, submodule, setup, and public entrypoint contracts."""

from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
EXPECTED_WORKFLOWS = {
    "benchmark-validation.yml",
    "dual-arch-compare.yml",
    "tier1-nightly.yml",
}


def _load_workflow(name: str) -> dict[str, object]:
    workflow_path = WORKFLOW_ROOT / name
    return yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _gitlink_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.split("\t", maxsplit=1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("160000 ")
    }


def test_active_workflows_live_at_git_root() -> None:
    assert {path.name for path in WORKFLOW_ROOT.glob("*.yml")} >= EXPECTED_WORKFLOWS

    for workflow_name in EXPECTED_WORKFLOWS:
        payload = _load_workflow(workflow_name)
        defaults = payload["defaults"]
        assert isinstance(defaults, dict)
        assert defaults["run"]["working-directory"] == "code"


def test_workflows_use_current_node24_action_refs() -> None:
    expected_actions = {
        "actions/checkout": {
            "v7",
            "3d3c42e5aac5ba805825da76410c181273ba90b1",
        },
        "actions/download-artifact": {"v7"},
        "actions/setup-node": {"v7"},
        "actions/setup-python": {
            "v7",
            "5fda3b95a4ea91299a34e894583c3862153e4b97",
        },
        "actions/upload-artifact": {
            "v7",
            "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        },
    }
    observed_actions: set[str] = set()

    workflow_paths = sorted({*WORKFLOW_ROOT.glob("*.yml"), *WORKFLOW_ROOT.glob("*.yaml")})
    assert workflow_paths

    for workflow_path in workflow_paths:
        payload = _load_workflow(workflow_path.name)
        jobs = payload["jobs"]
        assert isinstance(jobs, dict)
        for job in jobs.values():
            for step in job.get("steps", []):
                action_ref = step.get("uses")
                if not isinstance(action_ref, str) or not action_ref.startswith("actions/"):
                    continue
                action_name, version = action_ref.split("@", maxsplit=1)
                assert version in expected_actions[action_name]
                observed_actions.add(action_name)

    assert observed_actions == set(expected_actions)


def test_dual_arch_workflow_is_gpu_independent_and_cuda_13_bounded() -> None:
    payload = _load_workflow("dual-arch-compare.yml")
    triggers = payload["on"]
    assert triggers["push"]["branches"] == ["main", "develop"]
    assert triggers["pull_request"]["branches"] == ["main", "develop"]
    jobs = payload["jobs"]
    assert isinstance(jobs, dict)
    compare_job = jobs["compare-builds"]
    assert compare_job["runs-on"] == "ubuntu-latest"
    assert compare_job["env"] == {
        "ARCH_LIST": "sm_100 sm_103 sm_120 sm_121",
        "AUTO_ARCH_DETECTION": "0",
        "COMPARE_BUILD_JOBS": "2",
    }
    steps = compare_job["steps"]
    bootstrap_step = next(
        step for step in steps if step["name"] == "Install checkout prerequisites"
    )
    checkout_index = next(
        index for index, step in enumerate(steps) if step["name"] == "Checkout repository"
    )
    bootstrap_index = steps.index(bootstrap_step)
    assert bootstrap_index < checkout_index
    assert bootstrap_step["working-directory"] == "/"
    for package in ("build-essential", "ca-certificates", "git", "make", "python3"):
        assert package in bootstrap_step["run"]
    assert not any(step["name"] == "Install build dependencies" for step in steps)
    checkout_step = steps[checkout_index]
    assert "with" not in checkout_step
    verify_step = next(
        step for step in steps if step["name"] == "Verify configured CUDA architecture targets"
    )
    assert "nvcc --list-gpu-code" not in verify_step["run"]
    assert "make --no-print-directory" in verify_step["run"]
    assert 'ARCH="${architecture}"' in verify_step["run"]
    assert "verify-cuda-arch-target" in verify_step["run"]

    arch_makefile = (CODE_ROOT / "core" / "common" / "cuda_arch.mk").read_text(encoding="utf-8")
    assert "CUDA_13_ARCH_LIST := sm_100 sm_103 sm_120 sm_121" in arch_makefile
    assert "ARCH_LIST ?= $(CUDA_13_ARCH_LIST)" in arch_makefile
    assert "CUDA_ARCH_PROBE_SOURCE" in arch_makefile
    assert "verify-cuda-arch-target:" in arch_makefile
    assert "$(CUDA_NVCC_ARCH_FLAGS) -c" in arch_makefile
    assert "sm_122" not in arch_makefile
    assert "sm_123" not in arch_makefile


def test_tier1_runner_is_attested_and_hardware_pinned() -> None:
    payload = _load_workflow("tier1-nightly.yml")
    assert payload["permissions"] == {"actions": "read", "contents": "read"}
    bootstrap_input = payload["on"]["workflow_dispatch"]["inputs"]["bootstrap_history"]
    assert bootstrap_input["type"] == "boolean"
    assert bootstrap_input["default"] == "false"
    acceptance_input = payload["on"]["workflow_dispatch"]["inputs"]["accept_history_anchor"]
    assert acceptance_input["type"] == "boolean"
    assert acceptance_input["default"] == "false"
    note_input = payload["on"]["workflow_dispatch"]["inputs"]["acceptance_note"]
    assert note_input["type"] == "string"
    assert note_input["default"] == ""
    tier1_job = payload["jobs"]["tier1"]
    assert "concurrency" not in payload
    assert tier1_job["concurrency"] == {
        "group": "tier1-nightly-producer",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert payload["defaults"]["run"]["shell"] == "bash"
    assert tier1_job["runs-on"] == [
        "self-hosted",
        "linux",
        "x64",
        "gpu",
        "b200",
        "node24-actions",
    ]
    assert tier1_job["env"] == {
        "TIER1_EXPECTED_GPU_NAME": "NVIDIA B200",
        "TIER1_ARTIFACT_SUFFIX": "${{ github.run_id }}-${{ github.run_attempt }}",
        "TIER1_ACCEPT_HISTORY_ANCHOR": "${{ inputs.accept_history_anchor || false }}",
        "TIER1_ACCEPTANCE_NOTE": "${{ inputs.acceptance_note || '' }}",
    }
    assert "needs" not in tier1_job
    assert tier1_job["outputs"] == {
        "run_id": "${{ steps.compute_run_id.outputs.run_id }}",
        "artifact_suffix": "${{ steps.compute_run_id.outputs.artifact_suffix }}",
        "candidate_history_artifact_id": (
            "${{ steps.upload_tier1_candidate_history.outputs.artifact-id }}"
        ),
        "evidence_artifact_id": "${{ steps.upload_tier1_evidence.outputs.artifact-id }}",
        "evidence_artifact_name": (
            "tier1-evidence-${{ steps.compute_run_id.outputs.artifact_suffix }}"
        ),
        "evidence_artifact_digest": ("${{ steps.upload_tier1_evidence.outputs.artifact-digest }}"),
        "evidence_identity_bound": "${{ steps.bind_evidence.outcome }}",
    }
    preflight = next(step for step in tier1_job["steps"] if step["name"] == "GPU preflight")["run"]
    assert '"--query-gpu=name,mig.mode.current"' in preflight
    assert 'name != expected or mig_mode != "Disabled"' in preflight
    steps = tier1_job["steps"]
    run_id_index = next(
        index for index, step in enumerate(steps) if step["name"] == "Compute run id"
    )
    preflight_index = next(
        index for index, step in enumerate(steps) if step["name"] == "GPU preflight"
    )
    assert run_id_index == 0
    assert steps[run_id_index]["id"] == "compute_run_id"
    assert steps[run_id_index]["working-directory"] == "/"
    assert "GITHUB_OUTPUT" in steps[run_id_index]["run"]
    assert run_id_index < preflight_index
    branch_step = steps[1]
    assert branch_step["name"] == "Validate canonical request"
    assert branch_step["working-directory"] == "/"
    assert "refs/heads/main" in branch_step["run"]
    assert "Scheduled Tier-1 runs cannot accept a new history anchor" in branch_step["run"]
    assert "acceptance_note requires accept_history_anchor" in branch_step["run"]
    assert "bootstrap_history requires protected anchor acceptance" in branch_step["run"]

    setup_index = next(index for index, step in enumerate(steps) if step["name"] == "Set up Python")
    restore_index = next(
        index for index, step in enumerate(steps) if step["name"] == "Restore latest Tier-1 history"
    )
    dependency_index = next(
        index for index, step in enumerate(steps) if step["name"] == "Install Python dependencies"
    )
    assert setup_index < restore_index < dependency_index
    restore_step = steps[restore_index]
    assert restore_step["env"] == {
        "GITHUB_TOKEN": "${{ github.token }}",
        "BOOTSTRAP_HISTORY": "${{ inputs.bootstrap_history || false }}",
    }
    assert "python -m core.scripts.ci.restore_tier1_history" in restore_step["run"]
    assert "--destination artifacts/history/tier1" in restore_step["run"]
    assert "--allow-bootstrap" in restore_step["run"]
    assert "--allow-anchor-renewal" in restore_step["run"]

    benchmark_step = next(step for step in steps if step["name"] == "Run canonical tier-1 suite")
    assert benchmark_step["id"] == "run_tier1"
    assert benchmark_step["continue-on-error"] == (
        "${{ env.TIER1_ACCEPT_HISTORY_ANCHOR == 'true' }}"
    )
    assert benchmark_step["env"] == {
        "AISP_TIER1_EVIDENCE_ARTIFACT_NAME": ("tier1-evidence-${{ env.TIER1_ARTIFACT_SUFFIX }}")
    }
    assert "--history-anchor-candidate" in benchmark_step["run"]
    assert "--accept-history-anchor" not in benchmark_step["run"]
    assert "--acceptance-note" not in benchmark_step["run"]
    assert "--update-expectations" not in benchmark_step["run"]
    assert "| tee /tmp/tier1_run.json" in benchmark_step["run"]

    identity_step = next(
        step for step in steps if step["name"] == "Bind immutable Tier-1 evidence identity"
    )
    assert identity_step["id"] == "bind_evidence"
    assert identity_step["if"] == ("${{ always() && steps.run_tier1.outcome != 'skipped' }}")
    assert 'payload["run_id"] = run_id' in identity_step["run"]
    assert "source_manifest_json" in identity_step["run"]
    assert "source_result_json" in identity_step["run"]
    assert 'payload["git"] = manifest_git' in identity_step["run"]
    assert 'summary["source_manifest_git_commit"] = expected_commit' in identity_step["run"]
    assert 'summary["source_git_dirty"] = False' in identity_step["run"]

    summary_step = next(step for step in steps if step["name"] == "Summarize tier-1 results")
    assert summary_step["working-directory"] == "/"
    assert "History root: `artifacts/history/tier1`" in summary_step["run"]

    evidence_step = next(
        step for step in steps if step["name"] == "Upload Tier-1 evidence artifact"
    )
    candidate_step = next(
        step for step in steps if step["name"] == "Upload Tier-1 candidate history artifact"
    )
    assert steps.index(evidence_step) < steps.index(candidate_step)
    assert evidence_step["id"] == "upload_tier1_evidence"
    assert evidence_step["if"] == "${{ always() && steps.run_tier1.outcome != 'skipped' }}"
    assert "code/artifacts/runs/${{ env.RUN_ID }}" in evidence_step["with"]["path"]
    assert "code/artifacts/runs/${{ env.RUN_ID }}__recheck__*" in evidence_step["with"]["path"]
    assert evidence_step["with"]["retention-days"] == "90"
    assert evidence_step["with"]["name"] == ("tier1-evidence-${{ env.TIER1_ARTIFACT_SUFFIX }}")
    assert candidate_step["if"] == (
        "${{ always() && steps.run_tier1.outcome != 'skipped' && "
        "steps.upload_tier1_evidence.outcome == 'success' }}"
    )
    assert candidate_step["with"]["name"] == (
        "tier1-candidate-history-${{ env.TIER1_ARTIFACT_SUFFIX }}"
    )
    assert candidate_step["with"]["retention-days"] == "90"

    publisher_job = payload["jobs"]["publish-history"]
    assert publisher_job["needs"] == "tier1"
    assert publisher_job["concurrency"] == {
        "group": "tier1-history-publication",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert "needs.tier1.outputs.candidate_history_artifact_id != ''" in publisher_job["if"]
    assert "needs.tier1.outputs.evidence_identity_bound == 'success'" in publisher_job["if"]
    publisher_steps = publisher_job["steps"]
    publisher_restore = next(
        step for step in publisher_steps if step["name"] == "Restore live Tier-1 canonical history"
    )
    assert "python -m core.scripts.ci.restore_tier1_history" in publisher_restore["run"]
    publisher_merge = next(
        step for step in publisher_steps if step["name"] == "Merge immutable Tier-1 evidence row"
    )
    assert "python -m core.scripts.benchmarks.merge_tier1_history" in publisher_merge["run"]
    assert "--canonical-history-root /tmp/tier1-live" in publisher_merge["run"]
    publisher_upload = next(
        step for step in publisher_steps if step["name"] == "Upload Tier-1 history artifact"
    )
    assert publisher_upload["with"]["path"] == "/tmp/tier1-published"
    assert publisher_upload["with"]["retention-days"] == "90"

    authorization_job = payload["jobs"]["authorize-history-anchor"]
    assert authorization_job["needs"] == "tier1"
    assert authorization_job["environment"] == "tier1-canonical-acceptance"
    assert "needs.tier1.result == 'success'" in authorization_job["if"]
    assert authorization_job["steps"][0]["working-directory"] == "/"

    promotion_job = payload["jobs"]["promote-history-anchor"]
    assert promotion_job["needs"] == ["tier1", "authorize-history-anchor"]
    assert promotion_job["concurrency"] == {
        "group": "tier1-history-publication",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert "environment" not in promotion_job
    assert "needs.tier1.result == 'success'" in promotion_job["if"]
    assert "needs['authorize-history-anchor'].result == 'success'" in promotion_job["if"]
    promotion_steps = promotion_job["steps"]
    download_step = next(
        step
        for step in promotion_steps
        if step["name"] == "Download immutable Tier-1 candidate history"
    )
    assert download_step["uses"] == "actions/download-artifact@v7"
    assert download_step["with"] == {
        "artifact-ids": "${{ env.TIER1_CANDIDATE_HISTORY_ARTIFACT_ID }}",
        "merge-multiple": "true",
        "path": "/tmp/tier1-candidate",
    }
    evidence_download_step = next(
        step
        for step in promotion_steps
        if step["name"] == "Download immutable Tier-1 benchmark evidence"
    )
    assert evidence_download_step["with"] == {
        "artifact-ids": "${{ env.TIER1_EVIDENCE_ARTIFACT_ID }}",
        "merge-multiple": "true",
        "path": "/tmp/tier1-evidence",
    }
    live_restore_step = next(
        step for step in promotion_steps if step["name"] == "Restore live Tier-1 canonical history"
    )
    assert live_restore_step["env"] == {
        "GITHUB_TOKEN": "${{ github.token }}",
        "BOOTSTRAP_HISTORY": "${{ inputs.bootstrap_history || false }}",
    }
    assert "python -m core.scripts.ci.restore_tier1_history" in live_restore_step["run"]
    assert "--destination /tmp/tier1-live" in live_restore_step["run"]
    assert "--allow-anchor-renewal" in live_restore_step["run"]
    checkout_step = next(step for step in promotion_steps if step["name"] == "Checkout repository")
    assert checkout_step["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    ratify_step = next(
        step for step in promotion_steps if step["name"] == "Ratify immutable Tier-1 candidate"
    )
    assert "python -m core.scripts.benchmarks.promote_tier1_history" in ratify_step["run"]
    assert "--canonical-history-root /tmp/tier1-live" in ratify_step["run"]
    assert "--output-history-root /tmp/tier1-ratified" in ratify_step["run"]
    assert "--requester" in ratify_step["run"]
    assert "--note" in ratify_step["run"]
    assert "--workflow-run" in ratify_step["run"]
    assert "--expected-git-commit" in ratify_step["run"]
    assert "--expected-evidence-artifact" in ratify_step["run"]
    assert "--expected-evidence-digest" in ratify_step["run"]
    final_history_step = next(
        step
        for step in promotion_steps
        if step["name"] == "Upload ratified Tier-1 history artifact"
    )
    assert final_history_step["with"]["name"] == ("tier1-history-${{ env.TIER1_ARTIFACT_SUFFIX }}")
    assert final_history_step["with"]["path"] == "/tmp/tier1-ratified"
    assert final_history_step["with"]["retention-days"] == "90"


def test_dual_arch_compare_script_resolves_chapters_from_code_root(
    tmp_path: Path,
) -> None:
    script = CODE_ROOT / "core" / "scripts" / "ci" / "run_compare_builds.sh"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    make_stub = stub_bin / "make"
    make_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s|%s\\n\' "$PWD" "$*" >> "${COMPARE_BUILD_LOG:?}"\n',
        encoding="utf-8",
    )
    make_stub.chmod(0o755)
    build_log = tmp_path / "compare-build-directories.txt"
    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env['PATH']}"
    env["COMPARE_BUILD_LOG"] = str(build_log)
    env["COMPARE_BUILD_JOBS"] = "2"

    subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_chapters = [
        "ch01",
        "ch02",
        "ch04",
        "ch06",
        "ch07",
        "ch08",
        "ch09",
        "ch10",
        "ch11",
        "ch12",
        "ch16",
        "ch18",
        "ch19",
        "ch20",
    ]
    assert build_log.read_text(encoding="utf-8").splitlines() == [
        f"{CODE_ROOT / chapter}|--jobs=2 compare" for chapter in expected_chapters
    ]


def test_dual_arch_compare_script_rejects_invalid_parallelism(tmp_path: Path) -> None:
    script = CODE_ROOT / "core" / "scripts" / "ci" / "run_compare_builds.sh"
    env = os.environ.copy()
    env["COMPARE_BUILD_JOBS"] = "0"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "COMPARE_BUILD_JOBS must be a positive integer" in result.stderr


def test_nested_workflows_are_non_executable_pointers() -> None:
    nested_root = CODE_ROOT / ".github" / "workflows"

    for workflow_path in nested_root.glob("*.yml"):
        text = workflow_path.read_text(encoding="utf-8")
        assert "repository-root .github/workflows" in text or "no bootcamp tree" in text
        assert "jobs:" not in text


def test_benchmark_validation_has_blocking_campaign_checks() -> None:
    payload = _load_workflow("benchmark-validation.yml")
    jobs = payload["jobs"]
    assert isinstance(jobs, dict)
    validate_steps = jobs["validate"]["steps"]
    contract_step = next(
        step for step in validate_steps if step["name"] == "Run core contract tests"
    )
    contract_command = contract_step["run"]

    assert "tests/test_optimization_campaign.py" in contract_command
    assert "tests/test_campaign_evidence.py" in contract_command
    assert "tests/test_llm_patch_worker.py" in contract_command
    assert "tests/test_mcts_optimizer.py" in contract_command
    assert "tests/test_ch10_makefile_contract.py" in contract_command
    assert "tests/test_tma_multicast_tool.py" in contract_command
    assert "tests/test_dual_arch_make_contract.py" in contract_command
    assert "tests/test_run_benchmarks_cuda_wrapper_regression.py" in contract_command
    assert "test_ch04_nvshmem_torchrun_specs_preserve_variant_contracts" in contract_command
    assert "tests/test_llm_patch_promotion.py" in contract_command
    assert "tests/test_repository_configuration.py" in contract_command
    assert "continue-on-error" not in contract_step

    shell_step = next(
        step for step in validate_steps if step["name"] == "Validate shell entrypoints"
    )
    assert "bash -n core/scripts/ci/run_compare_builds.sh" in shell_step["run"]

    audit_step = next(
        step
        for step in validate_steps
        if step["name"] == "Enforce repository-wide benchmark contracts"
    )
    assert "continue-on-error" not in audit_step


def test_cpu_ci_uses_exact_coverage_pins() -> None:
    workflow = (WORKFLOW_ROOT / "benchmark-validation.yml").read_text(encoding="utf-8")
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")

    for pin in ("pytest-cov==7.1.0", "coverage[toml]==7.15.2"):
        assert pin in requirements
        assert pin in workflow


def test_cuda13_base_requirements_exclude_conflicting_vllm_dependencies() -> None:
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    active_lines = {
        line.split("#", 1)[0].strip()
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    }
    vllm_pin_lines = {
        line.split("#", 1)[0].strip()
        for line in (CODE_ROOT / "vllm_no_deps.pin").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.split("#", 1)[0].strip()
    }

    assert "vllm==0.16.0+cu130" not in active_lines
    assert not any(line.startswith("vllm==") for line in active_lines)
    assert not any(line.startswith("cupy-cuda12x==") for line in active_lines)
    assert not any(line.startswith("typer-slim") for line in active_lines)
    assert "cupy-cuda13x==14.2.0" in active_lines
    assert "fastapi[standard-no-fastapi-cloud-cli]==0.121.3" in active_lines
    assert vllm_pin_lines == {"vllm==0.16.0+cu130"}


def test_tier1_installs_the_cuda13_vllm_pin_as_a_separate_no_deps_phase() -> None:
    tier1_job = _load_workflow("tier1-nightly.yml")["jobs"]["tier1"]
    setup_python = next(
        step for step in tier1_job["steps"] if step["name"] == "Set up Python"
    )
    install = next(
        step["run"]
        for step in tier1_job["steps"]
        if step["name"] == "Install Python dependencies"
    )

    assert setup_python["with"]["cache-dependency-path"].splitlines() == [
        "code/requirements_latest.txt",
        "code/vllm_no_deps.pin",
    ]
    assert "-r requirements_latest.txt" in install
    assert 'vllm_spec="$(grep -E \'^vllm==\' vllm_no_deps.pin)"' in install
    assert 'test -n "${vllm_spec}"' in install
    assert "--no-deps" in install
    assert "--index-url https://wheels.vllm.ai/0.16.0/cu130" in install
    assert "--extra-index-url https://wheels.vllm.ai" not in install
    assert '"${vllm_spec}"' in install
    assert install.index("-r requirements_latest.txt") < install.index("vllm_spec=")
    assert install.index("vllm_spec=") < install.index("--no-deps")



def test_cpu_ci_installs_direct_collection_dependencies() -> None:
    workflow = _load_workflow("benchmark-validation.yml")
    install = next(step["run"] for step in workflow["jobs"]["validate"]["steps"]
                   if step["name"] == "Install CPU test dependencies")
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    # HTTP tests import requests; nanochat.engine imports tokenizers transitively
    # through checkpoint_manager. CLI tests also need the same Typer/Click/Rich
    # versions as the shared environment, including help and argument parsing.
    for package in ("requests", "tokenizers", "typer", "click", "rich"):
        pins = [line.split("#", 1)[0].strip() for line in requirements.splitlines()
                if line.strip().startswith(f"{package}==")]
        assert len(pins) == 1, f"The full CPU suite directly imports {package}"
        assert pins[0] in install.split(), f"Clean CPU CI must install {pins[0]} before collection"


def test_full_test_collection_gates_cpu_and_attested_gpu_workflows() -> None:
    cpu_job = _load_workflow("benchmark-validation.yml")["jobs"]["validate"]
    cpu_step = next(step for step in cpu_job["steps"] if step["name"] == "Run full CPU test suite")
    gpu_job = _load_workflow("tier1-nightly.yml")["jobs"]["tier1"]
    gpu_steps = gpu_job["steps"]
    gpu_step = next(step for step in gpu_steps if step["name"] == "Run verification tests on the attested GPU runner")
    for step, report in ((cpu_step, "pytest-cpu.xml"), (gpu_step, "pytest-gpu.xml")):
        assert "python -m pytest tests " in step["run"]
        assert "--ignore" not in step["run"]
        assert "-k " not in step["run"]
        assert "-o timeout=" in step["run"]
        assert "--timeout=" not in step["run"]  # external plugins are deliberately disabled
        assert report in step["run"]
        assert "continue-on-error" not in step
    preflight = next(step for step in gpu_steps if step["name"] == "GPU preflight")
    performance = next(step for step in gpu_steps if step["name"] == "Run canonical tier-1 suite")
    assert gpu_steps.index(preflight) < gpu_steps.index(gpu_step) < gpu_steps.index(performance)
    assert gpu_job["concurrency"]["group"] == "tier1-nightly-producer"
    assert "b200" in gpu_job["runs-on"]


def test_dashboard_ci_uses_blocking_node_24_gates() -> None:
    payload = _load_workflow("benchmark-validation.yml")
    jobs = payload["jobs"]
    assert isinstance(jobs, dict)
    dashboard = jobs["dashboard"]
    dashboard_steps = dashboard["steps"]

    setup_node = next(step for step in dashboard_steps if step["name"] == "Set up Node")
    assert setup_node["with"]["node-version"] == "24"
    expected_commands = {
        "npm ci",
        "npm audit --audit-level=moderate",
        "npm run lint",
        "npm test -- --runInBand",
        "npm run build",
    }
    actual_commands = {
        step["run"] for step in dashboard_steps if isinstance(step, dict) and "run" in step
    }
    assert expected_commands <= actual_commands
    assert all("continue-on-error" not in step for step in dashboard_steps)


def test_workflow_path_filters_are_root_relative() -> None:
    payload = _load_workflow("benchmark-validation.yml")
    triggers = payload["on"]
    assert isinstance(triggers, dict)

    for event_name in ("push", "pull_request"):
        paths = triggers[event_name]["paths"]
        assert {
            "code/Makefile",
            "code/dashboard/**",
            "code/requirements_latest.txt",
            "code/setup.sh",
            "code/uv.lock",
            ".gitmodules",
            ".github/workflows/**",
        } <= set(paths)
        for path_filter in paths:
            assert path_filter.startswith(("code/", ".github/", ".gitmodules"))


def test_gitmodules_matches_every_tracked_gitlink() -> None:
    parser = configparser.ConfigParser()
    parser.read(REPOSITORY_ROOT / ".gitmodules")
    declared_paths = {parser[section]["path"] for section in parser.sections()}

    assert declared_paths == _gitlink_paths()
    assert parser['submodule "code/third_party/cutlass"']["url"] == (
        "https://github.com/NVIDIA/cutlass.git"
    )
    for path in ("code/tools/nccl-tests", "code/cluster/tools/nccl-tests"):
        assert parser[f'submodule "{path}"']["url"] == ("https://github.com/NVIDIA/nccl-tests.git")


def test_cutlass_dsl_and_source_pins_match_validated_release() -> None:
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    setup = (CODE_ROOT / "setup.sh").read_text(encoding="utf-8")
    installer = (CODE_ROOT / "core" / "scripts" / "install_cutlass.sh").read_text(encoding="utf-8")

    assert re.search(r"^nvidia-cutlass-dsl\[cu13\]==4\.5\.2$", requirements, re.MULTILINE)
    assert 'CUTLASS_REF="${CUTLASS_REF:-v4.5.2}"' in setup
    assert 'CUTLASS_TARGET_VERSION="${CUTLASS_TARGET_VERSION:-4.5.2}"' in setup
    assert '"nvidia-cutlass-dsl[cu13]==4.5.2"' in setup
    assert "CUTLASS_REF:-v4.5.2" in installer


def test_uv_metadata_matches_supported_python_contract() -> None:
    lock_metadata = (CODE_ROOT / "uv.lock").read_text(encoding="utf-8")
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in lock_metadata
    assert "not a resolved dependency lock" in requirements


def test_setup_uses_git_root_and_does_not_ignore_submodule_failure() -> None:
    setup = (CODE_ROOT / "setup.sh").read_text(encoding="utf-8")

    root_discovery = setup.index('REPOSITORY_ROOT="$(discover_repository_root')
    safe_directory = setup.index('git config --global --add safe.directory "${REPOSITORY_ROOT}"')
    first_repository_git = setup.index('git -C "${REPOSITORY_ROOT}"')
    assert root_discovery < safe_directory < first_repository_git
    assert "rev-parse --show-toplevel" not in setup
    assert 'git -C "${REPOSITORY_ROOT}" submodule sync --recursive' in setup
    assert 'git -C "${REPOSITORY_ROOT}" submodule update --init --recursive' in setup
    assert "submodule update --init --recursive >/dev/null 2>&1 || true" not in setup


def test_dashboard_make_and_node_contracts_match_documented_ports() -> None:
    makefile = (CODE_ROOT / "Makefile").read_text(encoding="utf-8")
    package = json.loads(
        (CODE_ROOT / "dashboard" / "web" / "package.json").read_text(encoding="utf-8")
    )
    readme = (CODE_ROOT / "dashboard" / "web" / "README.md").read_text(encoding="utf-8")

    phony_line = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert {"dashboard", "dashboard-api", "dashboard-web"} <= set(phony_line.split())
    assert "DASHBOARD_API_PORT ?= 6970" in makefile
    assert "DASHBOARD_WEB_PORT ?= 3000" in makefile
    assert "$(PYTHON) -m dashboard.api.server serve --port $(DASHBOARD_API_PORT)" in makefile
    assert "npm --prefix dashboard/web run dev -- --port $(DASHBOARD_WEB_PORT)" in makefile
    assert package["engines"]["node"] == ">=24.0.0"
    assert package["devDependencies"]["concurrently"] == "10.0.5"
    assert "make dashboard" in readme
    assert "127.0.0.1:6970" in readme
    assert "127.0.0.1:3000" in readme


def test_makefile_and_examples_use_campaign_entrypoint() -> None:
    makefile = (CODE_ROOT / "Makefile").read_text(encoding="utf-8")
    example = (CODE_ROOT / "examples" / "optimize_examples.py").read_text(encoding="utf-8")

    assert "campaign-init:" in makefile
    assert "campaign-gate:" in makefile
    assert 'test -n "$(CONTROL_COMMIT)"' in makefile
    assert '--initial-control-commit "$(CONTROL_COMMIT)"' in makefile
    assert "core.optimization.campaign" in makefile
    assert "core.optimization.auto" not in makefile
    assert "auto-optimize" not in makefile
    assert "cli.advanced_cli" not in makefile
    assert "core.optimization.campaign" in example
    assert 'parser.add_argument("--initial-control-commit", required=True)' in example
    assert "AutoOptimizer" not in example
    assert "\n\t-flake8" not in makefile
    assert "\n\t-mypy" not in makefile


def test_default_lint_gate_is_strict_and_legacy_debt_is_explicit() -> None:
    makefile = (CODE_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "lint: lint-trusted" in makefile
    assert "lint-legacy-debt:" in makefile
    assert "check_benchmarks --include-unpaired --fail-on-warnings" in makefile
    assert "ruff check . --select E9,F63,F7,F82,B006,B023" in makefile
    for path in (
        "core/optimization/campaign.py",
        "core/optimization/campaign_evidence.py",
        "core/optimization/evidence_validation.py",
        "core/harness/llm_patch_worker.py",
        "core/optimization/search/mcts_optimizer.py",
        "core/analysis/llm_patch_promotion.py",
        "tests/test_optimization_campaign.py",
        "tests/test_campaign_evidence.py",
        "tests/test_llm_patch_worker.py",
        "tests/test_mcts_optimizer.py",
        "tests/test_llm_patch_promotion.py",
    ):
        assert path in makefile

    legacy_recipe = makefile.split("lint-legacy-debt:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "flake8 core/" in legacy_recipe
    assert "core/benchmark/metrics.py core/profiling/profiler_config.py" in legacy_recipe
    assert "exit $$status" in legacy_recipe


def test_ci_blocks_on_repository_wide_firstparty_correctness() -> None:
    workflow = _load_workflow("benchmark-validation.yml")
    steps = workflow["jobs"]["static-analysis"]["steps"]

    matching_steps = [
        step
        for step in steps
        if step.get("name") == "Run repository-wide first-party Ruff correctness checks"
    ]
    assert len(matching_steps) == 1
    assert matching_steps[0]["working-directory"] == "code"
    assert matching_steps[0]["run"] == "ruff check . --select E9,F63,F7,F82,B006,B023"


def test_make_mypy_gates_use_configured_python_interpreter() -> None:
    makefile = (CODE_ROOT / "Makefile").read_text(encoding="utf-8")
    trusted_recipe = makefile.split("lint-trusted:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    legacy_recipe = makefile.split("lint-legacy-debt:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]

    assert "$(PYTHON) -m mypy $(AUTONOMOUS_MYPY_PATHS)" in trusted_recipe
    assert (
        "$(PYTHON) -m mypy core/benchmark/metrics.py "
        "core/profiling/profiler_config.py --ignore-missing-imports || status=1" in legacy_recipe
    )
    assert "\n\tmypy " not in trusted_recipe
    assert "\n\tmypy " not in legacy_recipe
