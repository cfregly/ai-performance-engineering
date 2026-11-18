"""CPU contract and static source checks for the native NCCL symmetric-memory pair."""

from __future__ import annotations

import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from ch04 import nccl_symmetric_memory_build_receipt as build_receipt
from ch04 import run_nccl_symmetric_memory as runner
from ch04.baseline_nccl_symmetric_memory import get_benchmark as get_baseline
from ch04.optimized_nccl_symmetric_memory import get_benchmark as get_optimized

CODE_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_ROOT = CODE_ROOT / "ch04"


@pytest.mark.parametrize("arm", runner.ARMS)
def test_workload_arguments_cover_each_real_arm(arm: str) -> None:
    workload = runner.NativeWorkload(
        arm=arm, ranks=4, elements=4096, iterations=7, warmup=2
    )
    workload.validate()
    arguments = workload.arguments()
    assert arguments == [
        "--arm",
        arm,
        "--elements",
        "4096",
        "--iterations",
        "7",
        "--warmup",
        "2",
    ]


@pytest.mark.parametrize(
    "workload,match",
    [
        (runner.NativeWorkload("unknown", 2), "arm"),
        (runner.NativeWorkload("cuda_malloc", 1), "two ranks"),
        (runner.NativeWorkload("cuda_malloc", 2, elements=0), "elements"),
        (runner.NativeWorkload("cuda_malloc", 2, iterations=0), "iterations"),
        (runner.NativeWorkload("cuda_malloc", 2, warmup=-1), "warmup"),
        (runner.NativeWorkload("cuda_malloc", True), "must be integers"),
    ],
)
def test_workload_rejects_invalid_native_contract(
    workload: runner.NativeWorkload, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        workload.validate()


def test_runner_help_exposes_attribution_and_artifact_controls() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHAPTER_ROOT / "run_nccl_symmetric_memory.py"), "--help"],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for option in (
        "--all-arms",
        "--ranks",
        "--elements",
        "--iterations",
        "--warmup",
        "--json-out",
        "--dump-output",
        "--launcher-arg",
    ):
        assert option in completed.stdout
    assert "Launch four sequential MPI runs" in completed.stdout


def test_complete_output_validator_checks_every_float(tmp_path: Path) -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=3, elements=2051, iterations=2, warmup=1
    )
    output = tmp_path / "output.f32"
    values = array(
        "f", (runner._expected_value(workload.ranks, index) for index in range(workload.elements))
    )
    with output.open("wb") as stream:
        values.tofile(stream)
    receipt = runner.validate_full_output(output, workload)
    assert receipt == {"checked_elements": workload.elements, "max_abs_error": 0.0}

    values[2049] += 5.0
    corrupt = tmp_path / "corrupt.f32"
    with corrupt.open("wb") as stream:
        values.tofile(stream)
    with pytest.raises(RuntimeError, match="element 2049"):
        runner.validate_full_output(corrupt, workload)


def _valid_result(workload: runner.NativeWorkload) -> dict:
    samples = [1.0, 1.25, 1.5][: workload.iterations]
    statistics_ms = runner.expected_statistics(samples)
    return {
        "schema": runner.SCHEMA,
        "status": "passed",
        "arm": workload.arm,
        "dtype": "float32",
        "reduction": "sum",
        "out_of_place": True,
        "single_node": True,
        "p2p_gate_passed": True,
        "elements": workload.elements,
        "bytes_per_buffer": workload.elements * 4,
        "ranks": workload.ranks,
        "iterations": workload.iterations,
        "warmup": workload.warmup,
        "nccl_header_version_code": 22703,
        "nccl_runtime_version_code": 22803,
        "minimum_nccl_version_code": 22703,
        "timing_scope": "MPI_MAX CUDA event elapsed around one ncclAllReduce",
        "setup_ms": 5.0,
        "topology_gate_ms": 0.5,
        "communicator_ms": 1.0,
        "allocation_ms": 1.0,
        "registration_ms": 1.0,
        "initialization_ms": 1.0,
        "warmup_ms": 2.0,
        "time_per_iteration_ms": statistics_ms["mean"],
        "verification_ms": 1.0,
        "output_write_ms": 0.5,
        "cleanup_ms": 1.0,
        "rtol": 1.0e-5,
        "atol": 1.0e-5,
        "samples_ms": samples,
        "statistics_ms": statistics_ms,
        "verified_ranks": workload.ranks,
        "checked_elements_per_rank": workload.elements,
        "checked_elements_total": workload.elements * workload.ranks,
        "max_abs_error": 0.0,
        "rank_verification": [
            {
                "rank": rank,
                "checked_elements": workload.elements,
                "bad_elements": 0,
                "max_abs_error": 0.0,
            }
            for rank in range(workload.ranks)
        ],
        "rank_mapping": [
            {"rank": rank, "device_ordinal": rank} for rank in range(workload.ranks)
        ],
    }


def test_result_validator_recomputes_raw_mpi_max_statistics() -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=2, elements=128, iterations=3, warmup=1
    )
    result = _valid_result(workload)
    runner.validate_result(result, workload)
    result["statistics_ms"]["p95"] = 0.01
    with pytest.raises(RuntimeError, match="p95.*disagrees"):
        runner.validate_result(result, workload)


def test_result_validator_requires_every_rank_and_unique_gpu() -> None:
    workload = runner.NativeWorkload(
        arm="cuda_malloc", ranks=2, elements=128, iterations=2, warmup=0
    )
    result = _valid_result(workload)
    result["rank_verification"][1]["bad_elements"] = 1
    with pytest.raises(RuntimeError, match="every output element"):
        runner.validate_result(result, workload)
    result = _valid_result(workload)
    result["rank_mapping"][1]["device_ordinal"] = 0
    with pytest.raises(RuntimeError, match="distinct GPU"):
        runner.validate_result(result, workload)


def test_nccl_header_macro_parser_enforces_real_numeric_version() -> None:
    output = "\n".join(
        (
            "#define NCCL_MAJOR 2",
            "#define NCCL_MINOR 27",
            "#define NCCL_PATCH 3",
        )
    )
    assert build_receipt.parse_nccl_version_macros(output) == (22703, "2.27.3")
    with pytest.raises(RuntimeError, match="NCCL_PATCH"):
        build_receipt.parse_nccl_version_macros(output.replace("#define NCCL_PATCH 3", ""))


def test_native_source_has_four_real_paths_and_hard_gates() -> None:
    source = (CHAPTER_ROOT / "nccl_symmetric_memory_native.cu").read_text(encoding="utf-8")
    assert "#if NCCL_VERSION_CODE < 22703" in source
    assert "nccl_runtime_version" in source
    assert "NCCL_WIN_COLL_SYMMETRIC" in source
    assert "ncclCommWindowRegister" in source
    assert "ncclCommWindowDeregister" in source
    assert "ncclCommRegister" in source
    assert "ncclCommDeregister" in source
    assert "ncclMemAlloc" in source
    assert "cudaMalloc" in source
    assert "ncclAllReduce(\n            send_buffer, receive_buffer" in source
    assert "cudaEventRecord(sample_start, stream)" in source
    assert "cudaEventRecord(sample_stop, stream)" in source
    assert "cudaEventElapsedTime(&local_ms, sample_start, sample_stop)" in source
    assert "MPI_DOUBLE, MPI_MAX" in source
    assert "verify:complete_allreduce_output" in source
    assert "Every selected GPU pair must support directed CUDA peer access" in source
    for arm in runner.ARMS:
        assert f'"{arm}"' in source


def test_child_runtime_environment_uses_receipted_nccl_path_without_mutating_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "nccl" / "lib"
    secondary = tmp_path / "nccl" / "lib64"
    primary.mkdir(parents=True)
    secondary.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited/runtime")
    build = {
        "build_options": {
            "nccl_libs": f"-L{primary} -L{secondary} -lnccl",
        }
    }

    environment, directories = runner.child_runtime_environment(build)

    assert directories == [str(primary.resolve()), str(secondary.resolve())]
    assert environment["LD_LIBRARY_PATH"] == (
        f"{primary.resolve()}:{secondary.resolve()}:/inherited/runtime"
    )
    assert runner.os.environ["LD_LIBRARY_PATH"] == "/inherited/runtime"


def test_native_cleanup_is_exception_safe_and_ordered() -> None:
    source = (CHAPTER_ROOT / "nccl_symmetric_memory_native.cu").read_text(encoding="utf-8")
    cleanup = source.split("std::string cleanup_resources() noexcept", 1)[1].split(
        "void cleanup()", 1
    )[0]
    deregister = cleanup.index("ncclCommWindowDeregister")
    free = cleanup.index("ncclMemFree")
    communicator = cleanup.index("ncclCommDestroy")
    event = cleanup.index("cudaEventDestroy")
    stream = cleanup.index("cudaStreamDestroy")
    assert deregister < free < communicator < event < stream
    assert "~NativeBenchmark()" in source
    assert "cleanup_noexcept();" in source
    assert "require_collective_cuda_success(send_status" in source
    assert "require_collective_nccl_success(send_status" in source


def test_makefile_builds_receipted_native_target_without_fallback() -> None:
    makefile = (CHAPTER_ROOT / "Makefile").read_text(encoding="utf-8")
    assert ".PHONY: all help nvshmem nccl_symmetric_memory_native" in makefile
    assert "nccl_symmetric_memory_build_receipt.py --binary $@" in makefile
    assert "-lnccl" in makefile
    assert "educational mode" not in makefile.split("nccl_symmetric_memory_native:", 1)[1].split(
        "nvshmem_multigpu_examples", 1
    )[0]


def test_public_wrappers_fix_only_the_requested_control_and_candidate() -> None:
    baseline = get_baseline()
    optimized = get_optimized()
    assert baseline.arm == "cuda_malloc"
    assert optimized.arm == "window_register"
    assert baseline.elements == optimized.elements
    assert baseline.native_iterations == optimized.native_iterations
    assert baseline.native_warmup == optimized.native_warmup


def test_workload_and_intake_files_are_machine_readable() -> None:
    yaml = pytest.importorskip("yaml")
    workload = yaml.safe_load(
        (CHAPTER_ROOT / "nccl_symmetric_memory_workload_spec.yaml").read_text(encoding="utf-8")
    )
    intake = yaml.safe_load(
        (CHAPTER_ROOT / "nccl_symmetric_memory_performance_intake.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert workload["comparison_axis"]["control"] == "cuda_malloc"
    assert workload["comparison_axis"]["candidate"] == "window_register"
    assert workload["measurement_policy"]["outlier_policy"]["keep_raw_samples"] is True
    assert intake["evidence"]["static_review_findings"][-1].startswith("Static inspection")


def test_readme_labels_the_small_run_as_a_smoke() -> None:
    readme = (CHAPTER_ROOT / "README.md").read_text(encoding="utf-8")
    assert "--all-arms --ranks 2 --elements 1048576 --iterations 2 --warmup 1" in readme
    assert "The smoke proves only" in readme
    assert "--suite-timeout 0 --execution-timeout-seconds 0" in readme
