"""Repository-root CI, submodule, setup, and public entrypoint contracts."""

from __future__ import annotations

import configparser
import json
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


def test_dual_arch_workflow_is_gpu_independent_and_cuda_13_bounded() -> None:
    payload = _load_workflow("dual-arch-compare.yml")
    jobs = payload["jobs"]
    assert isinstance(jobs, dict)
    compare_job = jobs["compare-builds"]
    assert compare_job["runs-on"] == "ubuntu-latest"
    assert compare_job["env"] == {
        "ARCH_LIST": "sm_100 sm_103 sm_120 sm_121",
        "AUTO_ARCH_DETECTION": "0",
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
    assert "ca-certificates git" in bootstrap_step["run"]
    verify_step = next(
        step for step in steps if step["name"] == "Verify configured CUDA architecture targets"
    )
    assert "nvcc --list-gpu-code" in verify_step["run"]

    arch_makefile = (CODE_ROOT / "core" / "common" / "cuda_arch.mk").read_text(encoding="utf-8")
    assert "CUDA_13_ARCH_LIST := sm_100 sm_103 sm_120 sm_121" in arch_makefile
    assert "ARCH_LIST ?= $(CUDA_13_ARCH_LIST)" in arch_makefile
    assert "sm_122" not in arch_makefile
    assert "sm_123" not in arch_makefile


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
    assert "test_ch04_nvshmem_torchrun_specs_preserve_variant_contracts" in contract_command
    assert "tests/test_llm_patch_promotion.py" in contract_command
    assert "tests/test_repository_configuration.py" in contract_command
    assert "continue-on-error" not in contract_step

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
