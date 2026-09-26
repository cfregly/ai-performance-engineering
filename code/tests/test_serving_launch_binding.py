"""Verify P/D launch bytes, declared roles, and observed process placement."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from labs.serving_comparison.provenance import bind_pd_launch, verify_pd_allocation
from labs.serving_comparison.schema import ConfigError

EXAMPLES = Path(__file__).resolve().parents[1] / "labs/serving_comparison/examples"


def setup_launch(tmp_path, engine="vllm"):
    spec = json.loads((EXAMPLES / f"{engine}-pd-launch.json").read_text())
    provenance = json.loads((EXAMPLES / f"{engine}-pd-provenance.json").read_text())
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(spec))
    provenance["manifest_digest"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = {
        "engine": engine,
        "endpoint": provenance["endpoints"]["router"],
        "lifecycle": {
            "start_command": [
                sys.executable,
                "-m",
                "labs.serving_comparison.process_group_launcher",
                "--spec",
                str(path),
            ],
            "working_directory": str(tmp_path),
        },
        "provenance": provenance,
        "model": provenance["workload"]["model"],
    }
    return path, spec, arguments


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_launch_binding_matches_native_template_roles(tmp_path, engine):
    _, _, arguments = setup_launch(tmp_path, engine)
    binding = bind_pd_launch(**arguments)
    assert binding["roles"]["prefill"]["gpu_ids"] == ["0"]
    assert binding["roles"]["decode"]["gpu_ids"] == ["1"]
    assert binding["roles"]["router"]["gpu_ids"] == []
    assert binding["launcher_argv"][-2:] == ["--expected-sha256", binding["sha256"]]
    assert binding["runtime_pool_allocation_verified"] is False


def test_digest_mismatch_and_post_validation_edit_fail_before_children_start(tmp_path):
    path, _, arguments = setup_launch(tmp_path)
    binding = bind_pd_launch(**arguments)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ConfigError, match="manifest_digest"):
        bind_pd_launch(**arguments)
    result = subprocess.run(
        binding["launcher_argv"],
        cwd=EXAMPLES.parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "changed after provenance validation" in result.stderr
    assert "process_started" not in result.stdout


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("gpu", "GPU pool"),
        ("route", "router prefill URL"),
        ("connector", "connector"),
        ("model", "served model"),
    ],
)
def test_fresh_digest_does_not_hide_semantic_launch_drift(tmp_path, mutation, error):
    path, spec, arguments = setup_launch(tmp_path)
    if mutation == "gpu":
        spec["processes"][0]["environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        command = spec["processes"][2 if mutation == "route" else 0]["command"]
        flag, value = {
            "route": ("--prefill-url", "http://127.0.0.1:9999"),
            "connector": ("--kv-transfer-config", '{"kv_connector":"Other","kv_role":"kv_both"}'),
            "model": ("--served-model-name", "wrong-model"),
        }[mutation]
        command[command.index(flag) + 1] = value
    path.write_text(json.dumps(spec))
    arguments["provenance"]["manifest_digest"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ConfigError, match=error):
        bind_pd_launch(**arguments)


def test_observed_children_must_allocate_their_own_disjoint_gpu_pools(tmp_path):
    _, _, arguments = setup_launch(tmp_path)
    binding = bind_pd_launch(**arguments)
    roles = binding["roles"]
    log = tmp_path / "server.stdout.log"
    log.write_text(
        "".join(
            json.dumps(
                {"event": "process_started", "name": roles[role]["process_name"], "pid": pid}
            )
            + "\n"
            for role, pid in [("prefill", 100), ("decode", 200), ("router", 300)]
        )
    )
    allocation = {
        "compute_processes": [{"pid": 101, "gpu_uuid": "gpu-a"}, {"pid": 201, "gpu_uuid": "gpu-b"}]
    }
    kwargs = {
        "stdout_path": log,
        "allocation": allocation,
        "gpu_uuid_map": {"0": "gpu-a", "1": "gpu-b"},
        "descendants": lambda pid: {pid + 1},
    }
    observed = verify_pd_allocation(binding, **kwargs)
    assert observed["runtime_pool_allocation_verified"] is True
    allocation["compute_processes"][0]["gpu_uuid"] = "gpu-b"
    with pytest.raises(ConfigError, match="prefill observed GPU contexts"):
        verify_pd_allocation(binding, **kwargs)


@pytest.mark.parametrize("mutation", ["engine", "router", "router_mode", "proxy_identity"])
def test_native_entrypoint_drift_rejected_with_fresh_digest(tmp_path, mutation):
    path, spec, arguments = setup_launch(tmp_path, "sglang")
    if mutation == "engine":
        spec["processes"][0]["command"][2] = "unrelated.server"
    elif mutation == "router":
        spec["processes"][2]["command"][2] = "unrelated.router"
    elif mutation == "router_mode":
        spec["processes"][2]["command"].remove("--pd-disaggregation")
    else:
        arguments["provenance"]["proxy"]["implementation"] = "unrelated_proxy"
    path.write_text(json.dumps(spec))
    arguments["provenance"]["manifest_digest"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ConfigError, match="entrypoint|pd-disaggregation"):
        bind_pd_launch(**arguments)


def test_launcher_uses_current_interpreter_and_rejects_arbitrary_wrapper(tmp_path):
    _, _, arguments = setup_launch(tmp_path)
    arguments["lifecycle"]["start_command"][0] = "python3"
    assert bind_pd_launch(**arguments)["launcher_argv"][0] == sys.executable
    arguments["lifecycle"]["start_command"][0] = "/tmp/untrusted-wrapper"
    with pytest.raises(ConfigError, match="bundled process_group_launcher"):
        bind_pd_launch(**arguments)


def test_sif_native_entrypoint_and_case_insensitive_backend(tmp_path):
    path, spec, arguments = setup_launch(tmp_path)
    for child in spec["processes"][:2]:
        child["command"] = [
            "singularity",
            "exec",
            "--nv",
            "--bind",
            "/models:/models",
            "/images/vllm.sif",
            *child["command"],
        ]
    path.write_text(json.dumps(spec))
    arguments["provenance"]["manifest_digest"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert bind_pd_launch(**arguments)["roles"]["prefill"]["entrypoint"] == (
        "vllm.entrypoints.openai.api_server"
    )
    _, _, sglang = setup_launch(tmp_path, "sglang")
    sglang["provenance"]["connector"]["backend"] = "NIXL"
    assert bind_pd_launch(**sglang)["roles"]["prefill"]["entrypoint"] == "sglang.launch_server"
