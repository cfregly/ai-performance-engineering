import subprocess
import sys
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args, *, timeout=30, env=None):
    return subprocess.run(
        [sys.executable, "-m", "cli.aisp", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=env,
    )


def test_help_exits_cleanly():
    result = _run_cli(["help"])
    assert result.returncode == 0
    assert "aisp" in result.stdout.lower()


def test_mcp_server_import():
    import mcp.mcp_server as mcp_server

    assert isinstance(mcp_server.TOOLS, dict)
    # ensure harness tools are registered
    assert "run_benchmarks" in mcp_server.TOOLS


def test_bench_list_targets_help():
    result = _run_cli(["bench", "list-targets", "--help"])
    assert result.returncode == 0
    assert "list-targets" in result.stdout


def test_bench_run_e2e_help():
    result = _run_cli(["bench", "run-e2e", "--help"])
    assert result.returncode == 0
    assert "run-e2e" in result.stdout


def test_bench_run_e2e_dry_run_returns_structured_payload():
    result = _run_cli(
        [
            "bench",
            "run-e2e",
            "--dry-run",
            "--no-run-full-sweep",
            "--no-run-cluster",
            "--no-run-fabric",
        ],
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert payload["overall_status"] == "dry_run"
    assert payload["run_id"]
    assert payload["stages"][0]["name"] == "tier1"


def test_bench_run_e2e_resume_requires_run_id():
    result = _run_cli(
        [
            "bench",
            "run-e2e",
            "--resume",
            "--no-run-tier1",
            "--no-run-full-sweep",
            "--no-run-cluster",
            "--no-run-fabric",
        ],
        timeout=60,
    )
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["overall_status"] == "failed"
    assert "requires an explicit run_id" in payload["error"]


def test_bench_analyze_help():
    result = _run_cli(["bench", "analyze", "--help"])
    assert result.returncode == 0
    assert "analyze" in result.stdout.lower()


def test_bench_whatif_help():
    result = _run_cli(["bench", "whatif", "--help"])
    assert result.returncode == 0
    assert "whatif" in result.stdout.lower()


def test_distributed_plan_outputs_json_serializable_payload():
    result = _run_cli(
        ["--json", "distributed", "plan", "--model-size", "7", "--gpus", "8", "--nodes", "1"],
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert "best" in payload


def test_distributed_topology_returns_structured_payload():
    result = _run_cli(["--json", "distributed", "topology"], timeout=60)
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert "available" in payload


def test_inference_vllm_outputs_json_serializable_payload():
    result = _run_cli(["inference", "vllm", "--model-size", "7", "--target", "throughput"], timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert isinstance(payload["vllm_config"], dict)
    assert payload["vllm_config"]["model"] == "model"
    assert payload["vllm_config"]["tensor_parallel_size"] >= 1
    assert payload["vllm_config"]["launch_command"]


def test_cluster_nmx_partition_lab_requires_explicit_nmx_url():
    env = os.environ.copy()
    env["AISP_FABRIC_NMX_URL"] = "https://ignored.example"
    result = _run_cli(["cluster", "nmx-partition-lab", "--alpha-name", "AlphaPartition"], env=env)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error"] == "NMX URL is required. Pass --nmx-url to target the fabric management plane."


def test_cluster_common_eval_rejects_unknown_preset_via_real_cli():
    result = _run_cli(["cluster", "common-eval", "--preset", "not-a-preset", "--hosts", "localhost"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert "Unknown preset" in payload["error"]


def test_cluster_eval_suite_smoke_writes_real_manifest_and_meta():
    run_id = f"test_cli_smoke_eval_suite_{uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "cluster" / "runs" / run_id
    try:
        result = _run_cli(["cluster", "eval-suite", "--mode", "smoke", "--run-id", run_id], timeout=60)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        assert Path(payload["run_dir"]) == run_dir
        assert Path(payload["manifest_path"]).exists()
        assert Path(payload["meta_path"]).exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_cluster_promote_run_reports_missing_run_dir_via_real_cli():
    result = _run_cli(["cluster", "promote-run", "--run-id", "does-not-exist"])
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["published"] is False
    assert payload["run_id"] == "does-not-exist"
    assert "Missing or invalid run directory" in payload["error"]


def test_cluster_validate_field_report_accepts_candidate_package(tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(REPO_ROOT / "cluster/published/current", candidate)
    result = _run_cli([
        "--json", "cluster", "validate-field-report", "--published-root", str(candidate), "--package-only",
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True, payload
    assert "Stale-artifact hygiene: not requested" in payload["stdout"]

    (candidate / "manifest.json").unlink()
    missing = _run_cli([
        "--json", "cluster", "validate-field-report", "--published-root", str(candidate), "--package-only",
    ])
    payload = json.loads(missing.stdout)
    assert payload["success"] is False
    assert "Invalid published manifest" in payload["stderr"]


def test_cluster_common_eval_multinode_readiness_writes_real_manifest():
    run_id = f"test_cli_common_eval_{uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "cluster" / "runs" / run_id
    try:
        result = _run_cli(
            ["cluster", "common-eval", "--preset", "multinode-readiness", "--hosts", "localhost", "--run-id", run_id],
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        assert payload["preset"] == "multinode-readiness"
        assert Path(payload["manifest_path"]).exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_cluster_common_eval_two_node_readiness_dry_run_succeeds_without_workloads():
    run_id = f"test_cli_2node_ready_{uuid4().hex[:8]}"
    run_dir = REPO_ROOT / "cluster" / "runs" / run_id
    try:
        result = _run_cli(
            [
                "cluster",
                "common-eval",
                "--preset",
                "multinode-readiness",
                "--run-id",
                run_id,
                "--hosts",
                "node-a,node-b",
                "--labels",
                "node-a,node-b",
                "--ssh-user",
                "tester",
                "--oob-if",
                "bond0",
                "--socket-ifname",
                "bond0",
            ],
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        readiness_path = run_dir / "structured" / f"{run_id}_multinode_readiness.json"
        assert readiness_path.exists()
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        assert readiness["status"] == "ok"
        assert readiness["inputs"]["hosts"] == ["node-a", "node-b"]
        assert readiness["inputs"]["oob_if"] == "bond0"
        assert readiness["inputs"]["socket_ifname"] == "bond0"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_cluster_watch_promote_reports_missing_run_dir_via_real_cli():
    result = _run_cli(["cluster", "watch-promote", "--run-id", "does-not-exist", "--pid", "999999"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert "Missing or invalid run directory" in payload["error"]
