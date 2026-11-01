"""CPU-only failure fixtures. CUDA/tool calls are replaced; these are not GPU results."""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from divide_not_reciprocal import __main__ as runner
from divide_not_reciprocal import exhaustive, report, timing, triton_div




PROFILE_CONFIGS = {m: (4096, 8) for m in range(3)}


def synthetic_probe(case=0, **changes):
    """Synthetic contract data, not a captured report or GPU receipt."""
    data = {"schema_version": 1, "state": "COMPLETE", "pid": 4321,
            "case_index": case, "case_label": timing.CASES[case], "n": timing.NCU_N,
            "warmup_n": timing.NCU_WARM_N, "dtype": "torch.float32",
            "configs": {str(k): list(v) for k, v in PROFILE_CONFIGS.items()},
            "nvtx_range": timing._profile_range(case), "calls_completed": 2,
            "lowering_verification": "NOT_PERFORMED"}
    data.update(changes)
    return data


def synthetic_raw_csv(case=0, units=None, changes=None, launch_count=2):
    """Handwritten raw-page wide/units shape; values are explicitly synthetic.

    Not an official verbatim example and never a retained real ncu capture.
    """
    header = list(timing.LAUNCH_COLUMNS) + list(timing.METRICS)
    kernels = ["hardsigmoid_kernel"] * 3 + [
        "at::native::vectorized_elementwise_kernel<4, hardsigmoid_kernel_cuda>",
        "at::native::vectorized_elementwise_kernel<4, MulFunctor<float>>",
        "at::native::vectorized_elementwise_kernel<4, DivFunctor<float>>",
        "at::native::vectorized_elementwise_kernel<4, MulFunctor<float>>"]
    sizes = [timing.NCU_WARM_N, timing.NCU_N]
    rows = []
    for index, n in enumerate(sizes):
        elements, threads = (PROFILE_CONFIGS[case][0], PROFILE_CONFIGS[case][1] * 32) if case < 3 else (512, 128)
        row = {"ID": str(index), "Process ID": "4321", "Kernel Name": kernels[case], "Context": "1", "Stream": "7",
               "Block Size": f"({threads}, 1, 1)", "Grid Size": f"({n // elements}, 1, 1)", "Device": "0", "CC": "10.0"}
        row.update(dict(zip(timing.METRICS, map(str, (1000 if index == 0 else 1000000, n * 2, n * 4, n * 4)))))
        if changes:
            row.update(changes.get(index, {}))
        rows.append(row)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerow({**dict.fromkeys(timing.LAUNCH_COLUMNS, ""), **dict(zip(timing.METRICS, units or timing.BASE_UNITS))})
    writer.writerows((rows + [rows[-1]])[:launch_count])
    return "==PROF== synthetic fixture only\n" + stream.getvalue()


def parse_synthetic(text=None, case=0, probe=None, configs=None):
    return timing.parse_ncu_csv(synthetic_raw_csv(case) if text is None else text, timing.NCU_N,
                                case=case, configs=PROFILE_CONFIGS if configs is None else configs,
                                probe=synthetic_probe(case) if probe is None else probe)


def synthetic_tool_success(cmd, *, env, **kwargs):
    """Mock the native bytes and child's manifest; executes no external tool."""
    assert "timeout" not in kwargs and "text" not in kwargs
    assert cmd[cmd.index("--config-file") + 1] == "off"
    assert cmd[cmd.index("--rename-kernels") + 1] == "off"
    assert cmd[cmd.index("--page") + 1] == "raw"
    assert cmd[cmd.index("--print-units") + 1] == "base"
    assert cmd[cmd.index("--print-kernel-base") + 1] == "demangled"
    if "-o" in cmd:
        assert cmd[cmd.index("--cache-control") + 1] == "all"
        assert "--launch-skip" not in cmd and "--launch-count" not in cmd and "-k" not in cmd
        assert cmd[cmd.index("--nvtx-include") + 1] == timing._profile_range(0) + "/"
        native = Path(cmd[cmd.index("-o") + 1])
        native.write_bytes(b"SYNTHETIC native bytes: not an ncu report")
        Path(env["DNR_PROBE_PATH"]).write_text(json.dumps(synthetic_probe()))
    else:
        assert "--import" in cmd
    return subprocess.CompletedProcess(cmd, 0, synthetic_raw_csv().encode(), b"")


def test_unique_external_runs_and_collision_refusal(tmp_path):
    first, n1 = report.reserve_run(tmp_path, None)
    second, n2 = report.reserve_run(tmp_path, None)
    assert first != second and n1 != n2
    (first / "keep.txt").write_text("source evidence")
    with pytest.raises(FileExistsError):
        report.reserve_run(tmp_path, n1)
    assert (first / "keep.txt").read_text() == "source evidence"
    with pytest.raises(ValueError):
        report.reserve_run(tmp_path, "../escape")


def test_unwritable_output_is_rejected_before_preflight(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "preflight", lambda *a: calls.append(True))
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert runner.main(["--results", str(blocked), "--name", "new"]) == 2
    assert not calls


def test_json_survives_markdown_failure(tmp_path, monkeypatch):
    d, name = report.reserve_run(tmp_path, "journal")
    monkeypatch.setattr(report, "render", lambda *a: (_ for _ in ()).throw(ValueError("render failure")))
    with pytest.raises(ValueError, match="render failure"):
        report.write(d, name, {"state": "PARTIAL", "completed": [123]})
    assert json.loads((d / "journal.json").read_text())["completed"] == [123]


def test_preflight_failure_leaves_partial_without_experiment(tmp_path, monkeypatch):
    def fail(*a):
        raise RuntimeError("missing dependency")
    monkeypatch.setattr(runner, "preflight", fail)
    monkeypatch.setattr(exhaustive, "build", lambda *a: pytest.fail("experiment started"))
    assert runner.main(["--results", str(tmp_path), "--name", "preflight"]) == 1
    data = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert data["state"] == "PARTIAL" and data["errors"][0]["stage"] == "preflight"
    assert "missing dependency" in data["errors"][0]["message"]


def fake_environment(monkeypatch):
    # torch.device construction in the runner is harmless; all tensor/tool work is replaced.
    pytest.importorskip("torch")
    monkeypatch.setattr(runner, "preflight", lambda *a: {"gpu": "fixture"})
    monkeypatch.setattr(report, "smi", lambda *a: "fixture")
    monkeypatch.setattr(report, "conditions", lambda *a: "fixture conditions")


def minimal_triton():
    return {"reference": "fixture", "coverage": "one fixture chunk", "chunks": 1,
            "modes": {}, "pairs_differ": {"fixture pair": 17}}


def test_completed_chunk_survives_later_stage_failure(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    def fail(device, quick=False, progress=None):
        progress(minimal_triton())
        # Check that the callback really wrote the completed chunk before failure.
        disk = json.loads((tmp_path / "chunk/chunk.json").read_text())
        assert disk["triton"]["pairs_differ"]["fixture pair"] == 17
        raise RuntimeError("injected next-chunk failure")
    monkeypatch.setattr(triton_div, "run", fail)
    code = runner.main(["--results", str(tmp_path), "--name", "chunk", "--skip", "exhaustive,hardsigmoid,timing,ncu"])
    data = json.loads((tmp_path / "chunk/chunk.json").read_text())
    assert code == 1 and data["state"] == "PARTIAL"
    assert data["stages"]["triton"] == "FAILED" and data["triton"]["chunks"] == 1
    assert "injected next-chunk failure" in data["errors"][0]["traceback"]


def test_late_metadata_failure_keeps_completed_results(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(triton_div, "run", lambda *a, **k: minimal_triton())
    calls = []
    def conditions(*a):
        calls.append(True)
        if len(calls) > 1:
            raise OSError("injected metadata error")
        return "fixture start metadata"
    monkeypatch.setattr(report, "conditions", conditions)
    assert runner.main(["--results", str(tmp_path), "--name", "late", "--skip", "exhaustive,hardsigmoid,timing,ncu"]) == 1
    data = json.loads((tmp_path / "late/late.json").read_text())
    assert data["state"] == "PARTIAL" and data["stages"]["triton"] == "COMPLETE"
    assert data["triton"]["pairs_differ"]["fixture pair"] == 17
    assert data["errors"][0]["stage"] == "end_metadata"


def test_nvcc_fallback_is_shared_and_version_parser_is_safe(monkeypatch):
    monkeypatch.setattr(exhaustive.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda p: str(p) in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda/bin/cuobjdump"))
    assert exhaustive.nvcc() == "/usr/local/cuda/bin/nvcc"
    assert exhaustive.cuobjdump() == "/usr/local/cuda/bin/cuobjdump"
    monkeypatch.setattr(report, "smi", lambda *a: "fixture driver")
    # Metadata receives the exact preflight version, never invokes bare nvcc, and tolerates unusual version text.
    pf = {"gpu": "fixture", "tools": {"nvcc_version": "unusual but successful version text"},
          "torch_cuda": "fixture", "torch": "fixture", "triton": "not requested"}
    assert "unusual but successful version text" in report.conditions(None, {}, {}, "fixture", pf)


def test_compiler_failure_surfaces_complete_stdout_stderr(monkeypatch):
    def failed(cmd, **kwargs):
        assert "timeout" not in kwargs
        raise subprocess.CalledProcessError(9, cmd, output="compiler stdout", stderr="specific compiler error")
    monkeypatch.setattr(exhaustive.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="specific compiler error") as error:
        exhaustive.checked_command(["nvcc", "fixture"])
    assert "compiler stdout" in str(error.value) and "9" in str(error.value)










def test_timing_failure_keeps_completed_samples_without_extra_launches(monkeypatch):
    class Event:
        def __init__(self, **kw):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 0.25
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    calls = []
    def first():
        calls.append("first")
    def broken():
        calls.append("broken")
        raise RuntimeError("measurement failure")
    with pytest.raises(RuntimeError, match="measurement failure") as error:
        timing._time([first, broken], 0, 2)
    assert error.value.partial_results["samples_ms"] == [[0.25], []]
    assert calls == ["first", "broken"]


def test_success_marks_only_requested_stages_complete(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(triton_div, "run", lambda *a, **k: minimal_triton())
    assert runner.main(["--results", str(tmp_path), "--name", "success", "--skip", "exhaustive,hardsigmoid,timing,ncu"]) == 0
    data = json.loads((tmp_path / "success/success.json").read_text())
    assert data["state"] == "COMPLETE" and data["stages"] == {"triton": "COMPLETE"}
    assert data["implementation_sha256"]["count.cu"]


def test_requested_profile_errors_make_full_run_partial(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(timing, "time_cases", lambda *a, **k: {"n": 32, "bytes_moved": 256, "warmup": 20, "iters": 100,
                                                            "cases": {}, "best": {}})
    monkeypatch.setattr(timing, "ncu_counts", lambda *a, **k: {"n": 32, "launches": [], "artifacts": [],
                                                           "errors": [{"case": "fixture", "message": "denied"}]})
    assert runner.main(["--results", str(tmp_path), "--name", "profile", "--skip", "exhaustive,triton,hardsigmoid"]) == 1
    data = json.loads((tmp_path / "profile/profile.json").read_text())
    assert data["state"] == "PARTIAL" and data["stages"]["ncu"] == "FAILED"
    assert data["stages"]["timing"] == "COMPLETE" and data["ncu"]["errors"][0]["message"] == "denied"
    assert (tmp_path / "profile/ncu/preflight.txt").is_file()


def test_profile_artifact_collision_refused_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture",))
    directory = tmp_path / "ncu"
    timing.prepare_profile_dir(directory)
    (directory / "case-0.csv").write_text("existing evidence")
    monkeypatch.setattr(timing.subprocess, "run", lambda *a, **k: pytest.fail("capture started"))
    with pytest.raises(FileExistsError):
        timing.ncu_counts({}, {}, directory)
    assert (directory / "case-0.csv").read_text() == "existing evidence"


def test_exhaustive_requires_complete_histogram(monkeypatch):
    class Function:
        def __call__(self, *a):
            return 0  # No counters written; deliberately missing coverage.
    monkeypatch.setattr(exhaustive.ctypes, "CDLL", lambda *a: SimpleNamespace(run_count=Function()))
    build = exhaustive.Build("fixture", [], Path("not-loaded.so"), "", "")
    with pytest.raises(RuntimeError, match="Histogram coverage mismatch"):
        exhaustive.run(build, 0, 0, 8)






def test_final_markdown_failure_keeps_json_partial(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(triton_div, "run", lambda *a, **k: minimal_triton())
    render = report.render
    def fail_final(name, data):
        if data["state"] == "COMPLETE":
            raise ValueError("final markdown failure")
        return render(name, data)
    monkeypatch.setattr(report, "render", fail_final)
    assert runner.main(["--results", str(tmp_path), "--name", "final", "--skip", "exhaustive,hardsigmoid,timing,ncu"]) == 1
    data = json.loads((tmp_path / "final/final.json").read_text())
    assert data["state"] == "PARTIAL" and data["stages"]["triton"] == "COMPLETE"
    assert data["errors"][-1]["stage"] == "final_checkpoint"




@pytest.mark.parametrize("case", range(7))
def test_ncu_raw_wide_units_identity_and_large_only_normalization(case):
    data = parse_synthetic(case=case)
    assert data["time_s"] == pytest.approx(1e-3)
    assert data["inst_executed"] == 2 * timing.NCU_N
    assert data["warp_instructions_per_32_elements"] == 64
    assert data["dram_bytes"] == 8 * timing.NCU_N
    assert data["profiled_elements"] == timing.NCU_N
    assert data["warm_launch"]["metrics_si"][timing.METRICS[1]] == 2 * timing.NCU_WARM_N
    assert data["launch_identity"]["ID"] == "1"
    assert data["lowering_verification"] == "NOT_PERFORMED"


@pytest.mark.parametrize("text", [
    "", '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n',
    synthetic_raw_csv(units=("usecond", "inst", "byte", "byte")),
    synthetic_raw_csv(units=("nsecond", "inst", "Kbyte", "byte")),
    synthetic_raw_csv(changes={1: {timing.METRICS[0]: "nan"}}),
    synthetic_raw_csv(changes={1: {timing.METRICS[1]: "-1"}}),
    synthetic_raw_csv(changes={1: {timing.METRICS[1]: "1.234,5"}}),
    synthetic_raw_csv(changes={1: {timing.METRICS[1]: "1,23"}}),
    synthetic_raw_csv(changes={1: {timing.METRICS[1]: "0"}}),
    synthetic_raw_csv(changes={1: {"Kernel Name": "unrelated_kernel"}}),
    synthetic_raw_csv(changes={1: {"Process ID": "777"}}),
    synthetic_raw_csv(changes={1: {"ID": "0"}}),
    synthetic_raw_csv(changes={1: {"Grid Size": "(4, 1, 1)"}}),  # two tiny launches
    synthetic_raw_csv(changes={1: {"Block Size": "(128, 1, 1)"}}),
    synthetic_raw_csv(changes={0: {"Block Size": "(128, 1, 1)"}, 1: {"Block Size": "(128, 1, 1)"}}),
    synthetic_raw_csv(changes={1: {"Grid Size": "(16384, 2, 1)"}}),
    synthetic_raw_csv(changes={1: {"Grid Size": "garbled"}}),
    synthetic_raw_csv(changes={1: {"Stream": ""}}),
    synthetic_raw_csv(launch_count=1),  # warm-only capture
    synthetic_raw_csv(launch_count=3),  # a composite/extra launch cannot be hidden
])
def test_ncu_rejects_wrong_schema_units_numbers_identity_or_geometry(text):
    with pytest.raises(ValueError):
        parse_synthetic(text)


@pytest.mark.parametrize("changes", [{"n": timing.NCU_WARM_N}, {"case_index": 1}, {"case_label": "other"},
    {"nvtx_range": "other"}, {"calls_completed": 1}, {"state": "PARTIAL"}, {"pid": 0},
    {"configs": {}}, {"dtype": "torch.float16"}])
def test_ncu_requires_completed_actual_probe_binding(changes):
    with pytest.raises(ValueError, match="probe"):
        parse_synthetic(probe=synthetic_probe(**changes))


def test_ncu_valid_number_grouping_is_not_locale_guessing():
    text = synthetic_raw_csv(changes={1: {timing.METRICS[1]: "134,217,728"}})
    assert parse_synthetic(text)["warp_instructions_per_32_elements"] == 64


def test_ncu_eager_tiny_capture_cannot_use_large_normalization():
    text = synthetic_raw_csv(case=3, changes={1: {"Grid Size": "(32, 1, 1)"}})
    with pytest.raises(ValueError, match="geometry"):
        parse_synthetic(text, case=3)


def test_denied_profile_is_error_with_raw_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def denied(cmd, **kwargs):
        assert "timeout" not in kwargs and "text" not in kwargs
        return subprocess.CompletedProcess(cmd, 1, b"capture stdout\xff", b"ERR_NVGPUCTRPERM: denied\xfe")
    monkeypatch.setattr(timing.subprocess, "run", denied)
    checkpoints = []
    out = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu", lambda v: checkpoints.append(json.loads(json.dumps(v))))
    assert not out["launches"] and len(out["errors"]) == 1 and checkpoints
    assert out["artifacts"][0]["returncode"] == 1
    assert (tmp_path / "ncu/case-0.csv").read_bytes() == b"capture stdout\xff"
    assert (tmp_path / "ncu/case-0.stderr.txt").read_bytes() == b"ERR_NVGPUCTRPERM: denied\xfe"
    assert len(out["artifacts"][0]["files"]) == 2


def test_zero_exit_without_native_profile_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    monkeypatch.setattr(timing.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, synthetic_raw_csv().encode(), b""))
    out = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not out["launches"] and "nonempty native" in out["errors"][0]["message"]


def test_native_export_and_import_are_both_required_for_profile_success(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    calls = []
    def captured(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["cwd"] == (tmp_path / "ncu").resolve()
        return synthetic_tool_success(cmd, **kwargs)
    monkeypatch.setattr(timing.subprocess, "run", captured)
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not result["errors"] and len(result["launches"]) == 1 and len(calls) == 2
    assert len(result["artifacts"][0]["files"]) == 6
    assert result["configs"] == {str(k): list(v) for k, v in PROFILE_CONFIGS.items()}
    assert result["launches"][0]["profiled_elements"] == timing.NCU_N
    for f in result["artifacts"][0]["files"]:
        import hashlib
        assert hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"]


def test_native_import_error_is_partial_profile_data(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def failed_import(cmd, **kwargs):
        if "-o" in cmd:
            return synthetic_tool_success(cmd, **kwargs)
        return subprocess.CompletedProcess(cmd, 4, b"", b"invalid native report")
    monkeypatch.setattr(timing.subprocess, "run", failed_import)
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not result["launches"] and "import exited 4" in result["errors"][0]["message"]
    assert (tmp_path / "ncu/case-0.import.stderr.txt").read_bytes() == b"invalid native report"


def test_native_metric_mismatch_is_rejected_even_if_dram_sum_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def swapped(cmd, **kwargs):
        r = synthetic_tool_success(cmd, **kwargs)
        if "--import" in cmd:
            r.stdout = synthetic_raw_csv(changes={1: {timing.METRICS[2]: str(4 * timing.NCU_N - 1), timing.METRICS[3]: str(4 * timing.NCU_N + 1)}}).encode()
        return r
    monkeypatch.setattr(timing.subprocess, "run", swapped)
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not result["launches"] and "disagree" in result["errors"][0]["message"]


def test_ncu_decode_failure_keeps_exact_raw_bytes_native_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def bad_utf8(cmd, **kwargs):
        r = synthetic_tool_success(cmd, **kwargs)
        if "-o" in cmd:
            r.stdout = b"\xff" + r.stdout
        return r
    monkeypatch.setattr(timing.subprocess, "run", bad_utf8)
    checkpoints = []
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu", lambda v: checkpoints.append(v.copy()))
    assert not result["launches"] and result["errors"][0]["type"] == "UnicodeDecodeError"
    assert (tmp_path / "ncu/case-0.csv").read_bytes().startswith(b"\xff")
    assert len(result["artifacts"][0]["files"]) == 6 and checkpoints


def test_ncu_missing_probe_is_not_accepted_from_good_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def missing(cmd, **kwargs):
        r = synthetic_tool_success(cmd, **kwargs)
        if "-o" in cmd:
            Path(kwargs["env"]["DNR_PROBE_PATH"]).unlink()
        return r
    monkeypatch.setattr(timing.subprocess, "run", missing)
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not result["launches"] and result["errors"][0]["type"] == "FileNotFoundError"
    assert len(result["artifacts"][0]["files"]) == 5


def test_ncu_missing_sweep_configs_never_start_a_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing.subprocess, "run", lambda *a, **k: pytest.fail("capture started"))
    with pytest.raises(ValueError, match="three"):
        timing.ncu_counts({}, {}, tmp_path / "ncu")


class SyntheticTensor:
    dtype = "torch.float32"
    def __init__(self, n):
        self.n = n
    def numel(self):
        return self.n
    def uniform_(self, *args):
        return self


@pytest.mark.parametrize("fail_large", [False, True])
def test_ncu_probe_range_and_manifest_use_actual_tensors(tmp_path, monkeypatch, fail_large):
    calls, nvtx, events = [], [], []
    fake = SimpleNamespace(float32="torch.float32", empty=lambda n, **kw: SyntheticTensor(n),
                           empty_like=lambda t: SyntheticTensor(t.numel()),
                           cuda=SimpleNamespace(synchronize=lambda: events.append("sync"),
                             nvtx=SimpleNamespace(
                                 range_push=lambda name: (nvtx.append(("push", name)), events.append("push")),
                                 range_pop=lambda: (nvtx.append(("pop", None)), events.append("pop")))))
    monkeypatch.setitem(sys.modules, "torch", fake)
    def cases(x, out, configs):
        def call():
            calls.append(x.numel())
            events.append(x.numel())
            if fail_large and x.numel() == timing.NCU_N:
                raise RuntimeError("synthetic large-call failure")
        return [call]
    monkeypatch.setattr(timing, "make_cases", cases)
    monkeypatch.setenv("DNR_CONFIGS", json.dumps(PROFILE_CONFIGS))
    path = tmp_path / "probe.json";monkeypatch.setenv("DNR_PROBE_PATH", str(path))
    if fail_large:
        with pytest.raises(RuntimeError, match="synthetic"):
            timing.ncu_probe(0)
    else:
        timing.ncu_probe(0)
    data = json.loads(path.read_text())
    assert calls == [timing.NCU_WARM_N, timing.NCU_N]
    assert nvtx == [("push", timing._profile_range(0)), ("pop", None)]
    assert events == ["sync", "push", timing.NCU_WARM_N, "sync", timing.NCU_N,
                      *([] if fail_large else ["sync"]), "pop"]
    assert data["n"] == timing.NCU_N and data["warmup_n"] == timing.NCU_WARM_N
    assert data["state"] == ("PARTIAL" if fail_large else "COMPLETE")
    assert data["calls_completed"] == (1 if fail_large else 2)


def test_timing_statistics_failure_retains_completed_raw_samples_upstream(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    import torch
    monkeypatch.setattr(torch, "empty", lambda n, **kw: SyntheticTensor(n))
    monkeypatch.setattr(torch, "empty_like", lambda t: SyntheticTensor(t.numel()))
    monkeypatch.setattr(timing, "sweep", lambda *a: (PROFILE_CONFIGS, {"synthetic sweep": 1.0}))
    monkeypatch.setattr(timing, "make_cases", lambda *a: [None] * len(timing.CASES))
    samples = [[0.25, 0.5, 0.75] for _ in timing.CASES]
    rotations = []
    def complete(cases, warmup, iters):
        rotations.append((len(cases), warmup, iters))
        return samples
    monkeypatch.setattr(timing, "_time", complete)
    monkeypatch.setattr(timing.statistics, "median", lambda *a: (_ for _ in ()).throw(ValueError("synthetic row-statistics failure")))
    assert runner.main(["--results", str(tmp_path), "--name", "stats", "--skip", "exhaustive,triton,hardsigmoid,ncu"]) == 1
    data = json.loads((tmp_path / "stats/stats.json").read_text())
    assert data["state"] == "PARTIAL" and data["stages"]["timing"] == "FAILED"
    assert data["timing_partial"]["completed_samples_ms"] == samples
    assert rotations == [(7, timing.WARMUP, timing.ITERS)]


@pytest.mark.parametrize("column", ["ID", "Device", "Kernel Name"])
def test_ncu_native_import_launch_identity_must_match_capture(tmp_path, monkeypatch, column):
    monkeypatch.setattr(timing.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(timing, "CASES", ("fixture case",))
    def changed(cmd, **kwargs):
        result = synthetic_tool_success(cmd, **kwargs)
        if "--import" in cmd:
            replacement = {"ID": ["2", "3"], "Device": ["1", "1"], "Kernel Name": ["hardsigmoid_kernel(float*)"] * 2}[column]
            result.stdout = synthetic_raw_csv(changes={0: {column: replacement[0]}, 1: {column: replacement[1]}}).encode()
        return result
    monkeypatch.setattr(timing.subprocess, "run", changed)
    result = timing.ncu_counts({}, PROFILE_CONFIGS, tmp_path / "ncu")
    assert not result["launches"] and "disagree" in result["errors"][0]["message"]


def test_ncu_populated_large_row_preserves_report_interface():
    row = {"case": timing.CASES[0], **parse_synthetic()}
    text = "\n".join(report.table5({"n": timing.NCU_N, "launches": [row], "errors": []}))
    assert timing.CASES[0] in text and "64.000" in text
    assert f"{2 * timing.NCU_N:,}" in text and f"{8 * timing.NCU_N:,}" in text
    assert "0.001" in text and "warp instructions" in text


@pytest.mark.parametrize("extra", ["unexpected application output", "2,4321,hardsigmoid_kernel,1,7,(256 1 1),(4 1 1),0,10.0,1,1,1,1"])
def test_ncu_does_not_discard_unquoted_extra_output_or_launches(extra):
    with pytest.raises(ValueError):
        parse_synthetic(synthetic_raw_csv() + extra + "\n")


def load_gpu_checks():
    import importlib.util
    path = Path(__file__).with_name("test_gpu.py")
    spec = importlib.util.spec_from_file_location("dnr_gpu_contract_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timed_contract_distinguishes_modes_and_composed_bound(monkeypatch):
    torch = pytest.importorskip("torch")
    checks = load_gpu_checks()
    monkeypatch.setattr(checks, "torch", torch)
    # Exact ordered encodings: source is one step above divide. A valid full
    # mode two steps below divide is three below source; the old bound rejected it.
    bits = torch.tensor([0x3F000000], dtype=torch.int32)
    divide, source, full = bits.view(torch.float32), (bits + 1).view(torch.float32), (bits - 2).view(torch.float32)
    checks.assert_timed_expression(0, full, source, divide)
    checks.assert_timed_expression(1, source, source, divide)
    checks.assert_timed_expression(2, divide, source, divide)
    for mode, wrong in ((0, (bits - 3).view(torch.float32)), (1, divide), (2, source)):
        with pytest.raises(AssertionError):
            checks.assert_timed_expression(mode, wrong, source, divide)


def test_required_gpu_dependency_refusal_is_a_failure(monkeypatch):
    checks = load_gpu_checks()
    monkeypatch.setenv("DNR_REQUIRE_GPU_TESTS", "1")
    with pytest.raises(pytest.fail.Exception, match="missing test dependency"):
        checks.unavailable("missing test dependency")
    monkeypatch.delenv("DNR_REQUIRE_GPU_TESTS")
    with pytest.raises(pytest.skip.Exception, match="missing test dependency"):
        checks.unavailable("missing test dependency")
