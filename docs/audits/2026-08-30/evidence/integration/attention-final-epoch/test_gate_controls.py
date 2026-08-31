"""Real filesystem/report-parser controls; none execute or simulate a GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("attention_final_epoch_driver", HERE / "run_cuda_acceptance.py")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
NODE = "tests/test_audit_wave1_attention_regressions.py::test_real_cuda_reduction_repeated_multwarp_reuse"


def _junit(path, *, tag=None, tests=1, classname=None, name=None):
    suite = ET.Element("testsuite", tests=str(tests), failures="0", errors="0", skipped="0")
    case = ET.SubElement(suite, "testcase", classname=classname or "tests.test_audit_wave1_attention_regressions",
                         name=name or NODE.split("::")[1])
    if tag:
        ET.SubElement(case, tag)
    ET.ElementTree(suite).write(path)


def test_driver_root_and_actual_collected_selection():
    assert (gate.ROOT / "code/tests/test_audit_wave1_attention_regressions.py").is_file()
    cases = gate.read_cases(HERE / "expected_cuda_cases.json")
    assert len(cases) == 32
    assert cases[-1].startswith("tests/test_audit_wave1_prefill_full_output.py::")


def test_source_gate_uses_real_bytes_and_rejects_changed_or_missing_file(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    manifest = [{"path": "source.py", "sha256": gate.hashlib.sha256(source.read_bytes()).hexdigest()}]
    assert gate.inspect_epoch(tmp_path, manifest)[1] == []
    source.write_text("value = 2\n")
    assert gate.inspect_epoch(tmp_path, manifest)[1] == ["source.py"]
    source.unlink()
    assert gate.inspect_epoch(tmp_path, manifest)[1] == ["source.py"]


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py"])
def test_source_gate_rejects_outside_paths(tmp_path, path):
    with pytest.raises(ValueError):
        gate.inspect_epoch(tmp_path, [{"path": path, "sha256": "0" * 64}])


def test_source_gate_rejects_duplicate_manifest_entries(tmp_path):
    item = {"path": "source.py", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="duplicate"):
        gate.inspect_epoch(tmp_path, [item, item])


@pytest.mark.parametrize("tag", ["skipped", "failure", "error"])
def test_report_tags_reject_even_with_forged_zero_summary_counts(tmp_path, tag):
    report = tmp_path / "parser-fixture.xml"
    _junit(report, tag=tag)
    assert not gate.assess_junit(report, 0, [NODE])["accepted"]


@pytest.mark.parametrize("fault", ["wrong_name", "wrong_class", "wrong_count", "nonzero_exit", "missing", "malformed"])
def test_report_identity_counts_process_and_parse_failures_reject(tmp_path, fault):
    report = tmp_path / "parser-fixture.xml"
    _junit(report, tests=2 if fault == "wrong_count" else 1,
           name="test_real_unreviewed" if fault == "wrong_name" else None,
           classname="tests.unreviewed" if fault == "wrong_class" else None)
    if fault == "missing":
        report.unlink()
    elif fault == "malformed":
        report.write_text("<unfinished>")
    assert not gate.assess_junit(report, 1 if fault == "nonzero_exit" else 0, [NODE])["accepted"]


def test_parser_accepts_exact_clean_control_without_claiming_gpu_execution(tmp_path):
    report = tmp_path / "parser-fixture.xml"
    _junit(report)
    assessment = gate.assess_junit(report, 0, [NODE])
    assert assessment["accepted"] and assessment["exact_case_identities"]
    assert "gpu_executed" not in assessment and "status" not in assessment


def test_selection_rejects_replacement_by_duplicate(tmp_path):
    data = json.loads((HERE / "expected_cuda_cases.json").read_text())
    data["nodeids"][-1] = data["nodeids"][0]
    path = tmp_path / "selection-fixture.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="distinct"):
        gate.read_cases(path)
