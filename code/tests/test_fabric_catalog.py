from __future__ import annotations

from pathlib import Path

import pytest

from cluster.fabric.catalog import generate_catalog_payload


def test_generate_catalog_payload_extracts_key_commands_and_lab_only_markers() -> None:
    payload = generate_catalog_payload(run_id="2026-03-16_fabric_catalog")
    assert payload["run_id"] == "2026-03-16_fabric_catalog"
    assert payload["catalog_source"] == "committed_snapshot"
    entries = payload["entries"]
    assert entries

    commands = {entry["command"]: entry for entry in entries if entry["entry_type"] == "command"}
    assert "curl -k <nmx-base>/gpus | jq" in commands
    assert "curl -k <nmx-base>/switch-nodes | jq" in commands
    assert "curl -k <nmx-base>/chassis | jq" in commands
    assert "curl -k <nmx-base>/ports | jq" in commands
    assert 'curl -sk <nmx-base>/metrics | grep "^switch_temperature" | head -5' in commands
    assert "nv show qos roce" in commands
    assert any(command.startswith("ibdiagnet --get_phy_info") for command in commands)

    delete_partition = commands["curl -k -X DELETE <nmx-base>/partitions/<ID>"]
    assert delete_partition["mutates_state"] is True
    assert delete_partition["lab_only"] is True
    update_partition = commands[
        "curl -k -X PUT <nmx-base>/partitions/<partition-id> -H \"Content-Type: application/json\" -d '{\"DomainUUID\":\"<DomainUUID>\",\"Members\":{\"locations\":[\"<chassis.slot.host.gpu>\"]}}' | jq"
    ]
    assert update_partition["mutates_state"] is True
    assert update_partition["lab_only"] is True
    assert delete_partition["preconditions"] == "--nmx-url provided and endpoint reachable"

    disable_port = commands["ibportstate 218 1 disable"]
    assert disable_port["mutates_state"] is True
    assert disable_port["lab_only"] is True

    assert len(entries) == len({entry["id"] for entry in entries})


def test_generate_catalog_payload_covers_all_fabric_families() -> None:
    payload = generate_catalog_payload(run_id="2026-03-16_fabric_catalog")
    families = {entry["fabric_family"] for entry in payload["entries"]}
    assert {"nvlink", "infiniband", "spectrum-x"} <= families


def test_generate_catalog_payload_rejects_an_explicit_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fabric source root not found"):
        generate_catalog_payload(tmp_path / "missing", run_id="explicit-missing-source")
