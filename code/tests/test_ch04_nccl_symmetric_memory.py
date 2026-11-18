"""CPU contract and static source checks for the native NCCL symmetric-memory pair."""

from __future__ import annotations

import json
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


def _native_source() -> str:
    return (CHAPTER_ROOT / "nccl_symmetric_memory_native.cu").read_text(encoding="utf-8")


@pytest.mark.parametrize("arm", runner.ARMS)
def test_workload_arguments_cover_each_real_arm(arm: str) -> None:
    workload = runner.NativeWorkload(arm=arm, ranks=4, elements=4096, iterations=7, warmup=2)
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


def test_complete_output_validator_rejects_one_unit_error(tmp_path: Path) -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=2, elements=1, iterations=2, warmup=1
    )
    output = tmp_path / "one-unit-error.f32"
    values = array("f", [runner._expected_value(workload.ranks, 0) + 1.0])
    with output.open("wb") as stream:
        values.tofile(stream)

    with pytest.raises(RuntimeError, match="element 0"):
        runner.validate_full_output(output, workload)


def test_rank_keyed_oracle_rejects_repeated_middle_rank_buffer(tmp_path: Path) -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=3, elements=32, iterations=2, warmup=1
    )
    output = tmp_path / "repeated-rank.f32"
    rank_one = array(
        "f",
        (float(runner._input_value(1, index)) for index in range(workload.elements)),
    )
    repeated = array("f", (3.0 * value for value in rank_one))
    with output.open("wb") as stream:
        repeated.tofile(stream)

    with pytest.raises(RuntimeError, match="element"):
        runner.validate_full_output(output, workload)


def _valid_result(workload: runner.NativeWorkload) -> dict:
    samples = [1.0, 1.25, 1.5][: workload.iterations]
    statistics_ms = runner.expected_statistics(samples)
    registration_ranks = (
        workload.ranks if workload.arm in {"comm_register", "window_register"} else 0
    )
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
        "nccl_runtime_version_code": 22703,
        "nccl_runtime_library_path": "/opt/nccl/lib/libnccl.so.2",
        "nccl_runtime_library_verified_ranks": workload.ranks,
        "profiler_preload_allowed": False,
        "cuda_visible_devices": ",".join(str(rank) for rank in range(workload.ranks)),
        "minimum_nccl_version_code": 22703,
        "timing_scope": "MPI_MAX CUDA event elapsed around one ncclAllReduce",
        "setup_ms": 5.0,
        "topology_gate_ms": 0.5,
        "communicator_ms": 1.0,
        "allocation_ms": 1.0,
        "registration_ms": 1.0,
        "send_registration_handle_verified_ranks": registration_ranks,
        "receive_registration_handle_verified_ranks": registration_ranks,
        "initialization_ms": 1.0,
        "warmup_ms": 2.0,
        "time_per_iteration_ms": statistics_ms["mean"],
        "verification_ms": 1.0,
        "output_write_ms": 0.5,
        "cleanup_ms": 1.0,
        "rtol": 0.0,
        "atol": 0.0,
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
            {
                "rank": rank,
                "device_ordinal": rank,
                "gpu_uuid": f"GPU-00000000-0000-0000-0000-{rank + 1:012x}",
                "pci_bus_id": f"0000:{rank + 1:02x}:00.0",
            }
            for rank in range(workload.ranks)
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
    result = _valid_result(workload)
    result["rank_mapping"][1]["gpu_uuid"] = result["rank_mapping"][0]["gpu_uuid"]
    with pytest.raises(RuntimeError, match="physical GPU evidence"):
        runner.validate_result(result, workload)
    result = _valid_result(workload)
    result["rank_mapping"][1]["pci_bus_id"] = result["rank_mapping"][0]["pci_bus_id"]
    with pytest.raises(RuntimeError, match="physical GPU evidence"):
        runner.validate_result(result, workload)


def test_result_validator_binds_ordered_visibility_to_the_launcher() -> None:
    workload = runner.NativeWorkload(
        arm="cuda_malloc", ranks=2, elements=128, iterations=2, warmup=0
    )
    result = _valid_result(workload)
    runner.validate_result(result, workload, expected_cuda_visibility="0,1")
    with pytest.raises(RuntimeError, match="launcher environment"):
        runner.validate_result(result, workload, expected_cuda_visibility="1,0")


@pytest.mark.parametrize("arm", runner.ARMS)
def test_result_validator_requires_registration_handle_rank_evidence(arm: str) -> None:
    workload = runner.NativeWorkload(arm=arm, ranks=2, elements=128, iterations=2, warmup=0)
    result = _valid_result(workload)
    runner.validate_result(result, workload)

    expected = workload.ranks if arm in {"comm_register", "window_register"} else 0
    invalid = 0 if expected else workload.ranks
    for field in (
        "send_registration_handle_verified_ranks",
        "receive_registration_handle_verified_ranks",
    ):
        corrupted = {**result, field: invalid}
        with pytest.raises(RuntimeError, match="registration handle"):
            runner.validate_result(corrupted, workload)


def test_result_validator_binds_preload_permission_to_profile_mode() -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=2, elements=128, iterations=2, warmup=0
    )
    result = _valid_result(workload)
    runner.validate_result(result, workload)

    result["profiler_preload_allowed"] = True
    with pytest.raises(RuntimeError, match="profiler_preload_allowed"):
        runner.validate_result(result, workload)
    runner.validate_result(result, workload, profiler_preload_allowed=True)


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
    source = _native_source()
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


def test_native_source_uses_the_repository_nvtx_header() -> None:
    source = _native_source()
    include = '#include "../core/common/nvtx_utils.cuh"'
    assert include in source
    assert (CHAPTER_ROOT / "../core/common/nvtx_utils.cuh").resolve().is_file()


def test_native_runtime_library_binding_is_collective_and_fail_closed() -> None:
    source = _native_source()
    assert "#include <dlfcn.h>" in source
    assert 'key == "--expected-nccl-library"' in source
    assert "--expected-nccl-library is required" in source

    binding = source.split("NcclRuntimeBinding verify_nccl_runtime_binding", 1)[1].split(
        "struct Statistics", 1
    )[0]
    assert 'std::getenv("LD_PRELOAD")' in binding
    assert "!allow_profiler_preload" in binding
    assert "dlsym(RTLD_DEFAULT, symbol_name)" in binding
    assert "dladdr" in binding
    assert "dladdr(resolved_symbol" in binding
    assert "dladdr(reinterpret_cast<const void*>(&ncclGetVersion)" not in binding
    assert "resolved_get_version(&local_runtime_version)" in binding
    assert "realpath" in binding
    assert "symbol_path != expected_path" in binding
    assert "MPI_Allreduce" in binding
    assert "minimum_runtime_version != maximum_runtime_version" in binding
    assert "minimum_runtime_version != NCCL_VERSION_CODE" in binding
    assert '"NCCL runtime binding: "' in binding
    for symbol in (
        "ncclGetVersion",
        "ncclGetErrorString",
        "ncclGetUniqueId",
        "ncclCommInitRank",
        "ncclMemAlloc",
        "ncclMemFree",
        "ncclCommRegister",
        "ncclCommDeregister",
        "ncclCommWindowRegister",
        "ncclCommWindowDeregister",
        "ncclAllReduce",
        "ncclCommDestroy",
    ):
        assert f'"{symbol}"' in binding

    main = source.split("int main(int argc, char** argv)", 1)[1]
    gate = main.index("verify_nccl_runtime_binding")
    setup = main.index("benchmark.setup()")
    assert gate < setup
    assert '\\"nccl_runtime_library_path\\":' in source
    assert '\\"nccl_runtime_library_verified_ranks\\":' in source


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


def test_mechanism_environment_retains_forced_settings_without_secret_values() -> None:
    captured = runner.mechanism_environment(
        {
            "CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a",
            "NCCL_ALGO": "Tree",
            "NCCL_AUTH_TOKEN": "do-not-retain-this-token",
            "OMPI_MCA_btl": "self,vader,tcp",
            "UCX_TLS": "cuda_copy,cuda_ipc",
            "AWS_SECRET_ACCESS_KEY": "outside-filter",
        }
    )
    assert captured["cuda_visible_devices"] == "GPU-b,GPU-a"
    assert captured["variables"] == {
        "CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a",
        "NCCL_ALGO": "Tree",
        "NCCL_AUTH_TOKEN": "<redacted>",
        "OMPI_MCA_btl": "self,vader,tcp",
        "UCX_TLS": "cuda_copy,cuda_ipc",
    }
    assert captured["redacted_variables"] == ["NCCL_AUTH_TOKEN"]
    assert captured["nccl_environment_overrides"] == ["NCCL_ALGO", "NCCL_AUTH_TOKEN"]
    assert captured["nccl_attribution"] == "forced_environment"
    assert "do-not-retain-this-token" not in json.dumps(captured)
    assert "outside-filter" not in json.dumps(captured)


def test_native_nccl_runtime_failures_are_errors_not_capability_skips() -> None:
    source = _native_source()
    collective_check = source.split("void require_collective_nccl_success", 1)[1].split(
        "void require_collective_cuda_success", 1
    )[0]
    assert "throw Unsupported" not in collective_check
    assert "throw std::runtime_error" in collective_check
    assert 'throw Unsupported("NCCL runtime' not in source
    communicator_setup = source.split("void initialize_communicator()", 1)[1].split(
        "void allocate_buffers()", 1
    )[0]
    assert "ncclGetVersion" not in communicator_setup


def test_native_cleanup_is_explicit_fail_closed_and_ordered() -> None:
    source = _native_source()
    cleanup = source.split("void cleanup_resources()", 1)[1].split("void cleanup()", 1)[0]
    synchronize = cleanup.index("CUDA_CHECK(cudaStreamSynchronize(stream))")
    deregister = cleanup.index("ncclCommWindowDeregister")
    communicator = cleanup.index("ncclCommDestroy")
    free = cleanup.index("ncclMemFree")
    event = cleanup.index("cudaEventDestroy")
    stream = cleanup.index("cudaStreamDestroy")
    assert synchronize < deregister < communicator < free < event < stream
    assert "~NativeBenchmark() = default;" in source
    assert "cleanup_noexcept" not in source
    assert "benchmark.cleanup();" in source
    assert "require_collective_cuda_success(send_status" in source
    assert "require_collective_nccl_success(send_status" in source


def test_native_failure_unwind_reaches_mpi_abort_without_collective_cleanup() -> None:
    source = _native_source()
    main = source.split("int main(int argc, char** argv)", 1)[1]
    unsupported_catch = main.split("catch (const Unsupported& error)", 1)[1].split(
        "catch (const std::exception& error)", 1
    )[0]
    error_catch = main.split("catch (const std::exception& error)", 1)[1]
    assert "MPI_Abort(MPI_COMM_WORLD, 3)" in unsupported_catch
    assert "MPI_Abort(MPI_COMM_WORLD, 1)" in error_catch
    assert "cleanup" not in unsupported_catch
    assert "cleanup" not in error_catch


def test_native_topology_rejections_are_collectively_ordered() -> None:
    source = _native_source()
    gate = source.split("void require_single_node_unique_p2p_mapping()", 1)[1].split(
        "void initialize_communicator()", 1
    )[0]
    for local_flag, all_flag in (
        ("local_visibility_match", "all_visibility_match"),
        ("local_device_count_valid", "all_device_counts_valid"),
        ("local_unique", "all_unique"),
    ):
        local = gate.index(local_flag)
        reduction = gate.index("MPI_Allreduce", local)
        collective = gate.index(all_flag, reduction)
        rejection = gate.index("throw Unsupported", collective)
        assert local < reduction < collective < rejection
    assert "require_collective_cuda_success(cudaSetDevice" in gate
    assert 'cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties"' in gate
    assert "cudaDeviceGetPCIBusId" in gate
    assert "format_cuda_uuid" in gate
    assert "pci_bus_ids_by_rank" in gate
    assert 'require_collective_cuda_success(p2p_status, "cudaDeviceCanAccessPeer")' in gate


def test_native_registration_requires_handles_without_rechecking_window_status() -> None:
    source = _native_source()
    allocation = source.split("void allocate_buffers()", 1)[1].split("void initialize_inputs()", 1)[
        0
    ]
    comm_registration = allocation.split("if (options.arm == Arm::kCommRegister)", 1)[1].split(
        "} else if (options.arm == Arm::kWindowRegister)", 1
    )[0]
    assert "require_collective_handle" in comm_registration
    assert "send_registration" in comm_registration
    assert "receive_registration" in comm_registration

    window_registration = allocation.split("} else if (options.arm == Arm::kWindowRegister)", 1)[1]
    assert "if (send_status != ncclSuccess)" in window_registration
    assert "if (receive_status != ncclSuccess)" in window_registration
    assert "require_collective_handle" in window_registration
    assert "send_window" in window_registration
    assert "receive_window" in window_registration
    assert "require_collective_nccl_success" not in window_registration
    assert '\\"send_registration_handle_verified_ranks\\":' in source
    assert '\\"receive_registration_handle_verified_ranks\\":' in source


def test_native_outputs_use_exclusive_creation() -> None:
    source = _native_source()
    writer = source.split("void write_exclusive_output", 1)[1].split("void write_float_output", 1)[
        0
    ]
    assert "O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC" in writer
    assert "::write(" in writer
    assert "::fsync(" in writer

    float_writer = source.split("void write_float_output", 1)[1].split("}  // namespace", 1)[0]
    assert "write_exclusive_output(" in float_writer
    main = source.split("int main(int argc, char** argv)", 1)[1]
    assert "write_exclusive_output(" in main
    assert "std::ofstream" not in source


def test_native_verification_uses_a_fresh_nonperiodic_final_output() -> None:
    source = _native_source()
    assert "% 1024U" not in source
    pattern = source.split("std::uint32_t rank_index_pattern", 1)[1].split("float input_value", 1)[
        0
    ]
    assert "static_cast<std::uint64_t>(index)" in pattern
    assert "static_cast<std::uint64_t>(rank + 1)" in pattern
    assert "0xbf58476d1ce4e5b9ULL" in pattern
    assert "0x94d049bb133111ebULL" in pattern
    assert "mixed & 0xffU" in pattern

    poison = source.split("void poison_final_timed_destination()", 1)[1].split(
        "std::vector<double> measure()", 1
    )[0]
    assert "cudaMemsetAsync" in poison
    assert "receive_buffer, 0xFF" in poison
    assert "CUDA_CHECK(cudaStreamSynchronize(stream))" in poison

    measure = source.split("std::vector<double> measure()", 1)[1].split("struct Verification", 1)[0]
    poison_call = measure.index("poison_final_timed_destination();")
    barrier = measure.index("MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));", poison_call)
    timer_start = measure.index("cudaEventRecord(sample_start, stream)")
    allreduce = measure.index("enqueue_allreduce();", timer_start)
    timer_stop = measure.index("cudaEventRecord(sample_stop, stream)", allreduce)
    assert poison_call < barrier < timer_start < allreduce < timer_stop


def test_makefile_builds_receipted_native_target_without_fallback() -> None:
    makefile = (CHAPTER_ROOT / "Makefile").read_text(encoding="utf-8")
    assert ".PHONY: all help nvshmem nccl_symmetric_memory_force" in makefile
    assert "nccl_symmetric_memory_native" not in makefile.split(".PHONY:", 1)[1].split("\n", 1)[0]
    assert "nccl_symmetric_memory_native: nccl_symmetric_memory_force" in makefile
    assert "nccl_symmetric_memory_build_receipt.py --binary $@" in makefile
    assert "-lnccl" in makefile
    assert "-ldl" in makefile
    assert "NCCL_MULTIARCH" in makefile
    assert "-print-multiarch" in makefile
    assert ".DELETE_ON_ERROR:" in makefile
    for dependency in (
        "../core/common/cuda_arch.mk",
        "../core/common/nvtx_utils.cuh",
        "../core/common/nvtx_label_utils.cuh",
    ):
        assert dependency in makefile
        assert dependency in build_receipt.SOURCE_FILES
        assert dependency in runner.BUILD_SOURCE_FILES
    assert (
        "educational mode"
        not in makefile.split("nccl_symmetric_memory_native:", 1)[1].split(
            "nvshmem_multigpu_examples", 1
        )[0]
    )


def test_public_wrappers_fix_only_the_requested_control_and_candidate() -> None:
    baseline = get_baseline()
    optimized = get_optimized()
    assert baseline.arm == "cuda_malloc"
    assert optimized.arm == "window_register"
    assert baseline.elements == optimized.elements
    assert baseline.native_iterations == optimized.native_iterations
    assert baseline.native_warmup == optimized.native_warmup
    assert baseline.get_output_tolerance() == (0.0, 0.0)
    assert optimized.get_output_tolerance() == (0.0, 0.0)


def test_workload_and_intake_files_are_machine_readable() -> None:
    yaml = pytest.importorskip("yaml")
    workload = yaml.safe_load(
        (CHAPTER_ROOT / "nccl_symmetric_memory_workload_spec.yaml").read_text(encoding="utf-8")
    )
    intake = yaml.safe_load(
        (CHAPTER_ROOT / "nccl_symmetric_memory_performance_intake.yaml").read_text(encoding="utf-8")
    )
    assert workload["comparison_axis"]["control"] == "cuda_malloc"
    assert workload["comparison_axis"]["candidate"] == "window_register"
    assert workload["measurement_policy"]["outlier_policy"]["keep_raw_samples"] is True
    assert intake["evidence"]["static_review_findings"][-1].startswith("Static inspection")


def test_readme_labels_the_small_run_as_a_smoke() -> None:
    readme = (CHAPTER_ROOT / "README.md").read_text(encoding="utf-8")
    assert "--all-arms --ranks 2 --elements 1048576 --iterations 2 --warmup 1" in readme
    assert "A passing smoke proves only" in readme
    assert "--suite-timeout 0 --execution-timeout-seconds 0" in readme


def test_result_reservation_prevents_a_loser_from_poisoning_the_owner(tmp_path: Path) -> None:
    result_path = tmp_path / "owned.json"
    owner = runner.JsonResultReservation.acquire(result_path)
    with pytest.raises(FileExistsError, match="existing result artifact"):
        runner.JsonResultReservation.acquire(result_path)

    duplicate_rc = runner.main(
        [
            "--ranks",
            "2",
            "--binary",
            str(tmp_path / "missing-native"),
            "--json-out",
            str(result_path),
        ]
    )
    assert duplicate_rc == 1
    owner.write_json({"owner": "first"})
    assert result_path.read_text(encoding="utf-8") == '{\n  "owner": "first"\n}\n'
