"""Current-stack contracts for the checked serving comparison example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CURRENT_VLLM_VERSION = "0.33.0.dev0+cu130.torch2141"


def _read_json(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def test_checked_profile_targets_current_vllm_build() -> None:
    profile = _read_json("profile.engine.template.json")
    vllm_arms = [arm for arm in profile["arms"] if arm["engine"] == "vllm"]

    assert {arm["runtime"]["version"] for arm in vllm_arms} == {CURRENT_VLLM_VERSION}
    for arm in vllm_arms:
        version_assertions = [
            probe["assertions"]["version"]
            for probe in arm["identity"]["probes"]
            if probe["proves"] == ["runtime_version"]
        ]
        assert version_assertions
        assert set(version_assertions) == {CURRENT_VLLM_VERSION}

    provenance = _read_json("vllm-pd-provenance.json")
    assert provenance["runtime"]["version"] == CURRENT_VLLM_VERSION


def test_vllm_pd_provenance_still_binds_exact_launch_bytes() -> None:
    provenance = _read_json("vllm-pd-provenance.json")
    launch_bytes = (EXAMPLES / "vllm-pd-launch.json").read_bytes()

    assert provenance["manifest_digest"] == (f"sha256:{hashlib.sha256(launch_bytes).hexdigest()}")
