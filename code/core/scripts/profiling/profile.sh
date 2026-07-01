#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_RUNNER="${SCRIPT_DIR}/pytorch_profiler_runner.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/artifacts/runs/profile_manual"
PYTHON_DEFAULT="${PYTHON:-python}"
HARNESS_MODULE="core.scripts.harness.profile_harness"

DEFAULT_NCU_METRICS_RAW="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_DEFAULT" -c "from core.scripts.harness.metrics_config import BASE_NCU_METRICS; print(','.join(BASE_NCU_METRICS))" 2>/dev/null || true)"
DEFAULT_NCU_METRICS="${DEFAULT_NCU_METRICS_RAW//$'\n'/}"
if [[ -z "$DEFAULT_NCU_METRICS" ]]; then
    DEFAULT_NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_divergence.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,shared_load_sectors,shared_store_sectors,flop_count_sp,flop_count_hp,gpu__time_elapsed.avg"
fi

DEFAULT_NSYS_TRACE_RAW="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_DEFAULT" -c "from core.scripts.harness.metrics_config import BASE_NSYS_TRACE_MODULES; print(','.join(BASE_NSYS_TRACE_MODULES))" 2>/dev/null || true)"
DEFAULT_NSYS_TRACE="${DEFAULT_NSYS_TRACE_RAW//$'\n'/}"
if [[ -z "$DEFAULT_NSYS_TRACE" ]]; then
    DEFAULT_NSYS_TRACE="cuda,nvtx,osrt,cublas,cudnn,nvlink"
fi

DEFAULT_NSYS_EXTRA_RAW="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_DEFAULT" -c "from core.scripts.harness.metrics_config import BASE_NSYS_EXTRA_ARGS; print(' '.join(BASE_NSYS_EXTRA_ARGS))" 2>/dev/null || true)"
DEFAULT_NSYS_EXTRA_RAW="${DEFAULT_NSYS_EXTRA_RAW//$'\n'/}"
DEFAULT_NSYS_EXTRA_OPTS=()
if [[ -n "$DEFAULT_NSYS_EXTRA_RAW" ]]; then
    read -r -a DEFAULT_NSYS_EXTRA_OPTS <<< "$DEFAULT_NSYS_EXTRA_RAW"
fi

print_usage() {
    cat <<'USAGE'
Usage:
  profile.sh --list
  profile.sh [HARNESS_FLAGS...]
  profile.sh <script.py> [--arch sm_100] [--tool nsys|ncu|pytorch|hta|perf|zymtrace|all]
                         [--pytorch-mode full] [--output-root DIR] [--python PYTHON]
                         [-- script-args ...]

Harness mode (preferred):
  For registered examples, forward directly to the module-backed profile harness. Any
  of the harness arguments (--examples, --tags, --profile, --profile-mode,
  --dry-run, --skip-existing, --force-build, etc.) can be passed through.

Direct mode:
  Executes a specific script path with Nsight Systems, Nsight Compute,
  PyTorch profiler, HTA, perf, and/or Zymtrace CUDA injection. Optional script arguments can be
  provided after a literal "--".

Examples:
  profile.sh --profile nsys --examples ch14_triton_examples
  profile.sh code/ch07/memory_access_pytorch.py --tool ncu
  profile.sh code/ch15/speculative_decoding_benchmarks.py --tool zymtrace
  profile.sh code/ch09/fusion_pytorch.py --tool pytorch --pytorch-mode memory -- --batch-size 4
USAGE
}

require_command() {
    local cmd="$1"
    local label="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "⚠ Skipping ${label}; '${cmd}' not found in PATH" >&2
        return 1
    fi
    return 0
}

print_command() {
    local parts=("$@")
    printf '→ %s\n' "$(printf '%q ' "${parts[@]}")"
}

timestamp() {
    date +%Y%m%d_%H%M%S
}

resolve_path() {
    local target="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath "$target"
    else
        python - "${target}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
    fi
}

resolve_architecture() {
    local requested="$1"
    if [[ "$requested" != "auto" ]]; then
        echo "$requested"
        return
    fi

    local detected="sm_100"
    # Prefer the shared SM detector; it maps CC 10.3 -> sm_103 (Blackwell Ultra /
    # GB300) and CC 10.0 -> sm_100 (B200/GB200), so it is GB300-correct.
    local probed
    probed=$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_DEFAULT" -m core.benchmark.detect_sm 2>/dev/null || true)
    if [[ "$probed" =~ ^sm_[0-9]+a?$ ]]; then
        echo "$probed"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 || true)
        if [[ -n "$gpu_name" && "$gpu_name" =~ (B300|GB300) ]]; then
            echo "sm_103"
            return
        fi
        if [[ -n "$gpu_name" && ! "$gpu_name" =~ (B200|B300) ]]; then
            echo "sm_100"
            echo "⚠ Non-Blackwell GPU detected (${gpu_name}); defaulting to sm_100" >&2
            return
        fi
    else
        echo "⚠ Unable to query GPU via nvidia-smi; assuming sm_100" >&2
    fi
    echo "$detected"
}

run_nsys() {
    require_command nsys "Nsight Systems" || return
    local base="${SESSION_DIR}/nsys_${ARCH_VALUE}_$(timestamp)"
    local cmd=(
        nsys profile
        --force-overwrite=true
        -o "$base"
        -t "$DEFAULT_NSYS_TRACE"
        -s cpu
        --python-sampling=true
        --python-sampling-frequency=1000
        --cudabacktrace=true
        --cudabacktrace-threshold=0
        --stats=true
    )
    if ((${#DEFAULT_NSYS_EXTRA_OPTS[@]})); then
        cmd+=("${DEFAULT_NSYS_EXTRA_OPTS[@]}")
    fi
    cmd+=("$PYTHON_BIN" "$SCRIPT_PATH")
    if [[ -n "${SCRIPT_ARGS[*]-}" ]]; then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Nsight Systems report: ${base}.nsys-rep"
}

run_ncu() {
    require_command ncu "Nsight Compute" || return
    pkill -f nsys >/dev/null 2>&1 || true
    local base="${SESSION_DIR}/ncu_${ARCH_VALUE}_$(timestamp)"
    local metrics="$DEFAULT_NCU_METRICS"
    if [[ -n "${NCU_EXTRA_METRICS:-}" ]]; then
        metrics="${metrics},${NCU_EXTRA_METRICS}"
    fi
    local cmd=(
        ncu --set full
        --metrics "$metrics"
        --import-source yes
        -o "$base"
        "$PYTHON_BIN" "$SCRIPT_PATH"
    )
    if [[ -n "${SCRIPT_ARGS[*]-}" ]]; then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Nsight Compute report: ${base}.ncu-rep"
}

run_hta() {
    require_command nsys "Nsight Systems" || return
    local base="${SESSION_DIR}/hta_${ARCH_VALUE}_$(timestamp)"
    local cmd=(nsys profile --force-overwrite=true -o "$base" -t cuda,nvtx,osrt,cudnn,cublas,nccl \
        -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true \
        --stats=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop --capture-range-op=both \
        --multi-gpu=all "$PYTHON_BIN" "$SCRIPT_PATH")
    if [[ -n "${SCRIPT_ARGS[*]-}" ]]; then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ HTA report: ${base}.nsys-rep"
}

run_perf() {
    require_command perf "perf" || return
    local data_file="${SESSION_DIR}/perf_${ARCH_VALUE}_$(timestamp).data"
    local cmd=(perf record --call-graph dwarf -o "$data_file" "$PYTHON_BIN" "$SCRIPT_PATH")
    if [[ -n "${SCRIPT_ARGS[*]-}" ]]; then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Perf data captured: ${data_file}"
    echo "     View with: perf report -i ${data_file}"
}

resolve_zymtrace_injection() {
    local candidate
    if [[ -n "${CUDA_INJECTION64_PATH:-}" ]]; then
        candidate="${CUDA_INJECTION64_PATH}"
        if [[ -r "$candidate" ]]; then
            resolve_path "$candidate"
            return 0
        fi
        echo "✗ CUDA_INJECTION64_PATH is set but does not point to a file or is not readable: ${candidate}" >&2
        return 1
    fi
    if [[ -n "${ZYMTRACE_CUDA_INJECTION64_PATH:-}" ]]; then
        candidate="${ZYMTRACE_CUDA_INJECTION64_PATH}"
        if [[ -r "$candidate" ]]; then
            resolve_path "$candidate"
            return 0
        fi
        echo "✗ ZYMTRACE_CUDA_INJECTION64_PATH is set but does not point to a file or is not readable: ${candidate}" >&2
        return 1
    fi
    candidate="/var/lib/zymtrace/profiler/libzymtracecudaprofiler.so"
    if [[ -r "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi
    echo "✗ Zymtrace CUDA injection library not found. Set CUDA_INJECTION64_PATH or ZYMTRACE_CUDA_INJECTION64_PATH." >&2
    return 1
}

write_zymtrace_manifest() {
    local manifest_path="$1"
    local injection_lib="$2"
    shift 2
    "$PYTHON_BIN" - "$manifest_path" "$injection_lib" "$PYTHON_BIN" "$SCRIPT_PATH" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
payload = {
    "tool": "zymtrace",
    "cuda_injection64_path": sys.argv[2],
    "python": sys.argv[3],
    "script": sys.argv[4],
    "script_args": sys.argv[5:],
    "environment": {
        "CUDA_INJECTION64_PATH": sys.argv[2],
        "ZYMTRACE_CUDA_INJECTION64_PATH": sys.argv[2],
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
    },
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

run_zymtrace() {
    local injection_lib
    injection_lib="$(resolve_zymtrace_injection)"
    local manifest="${SESSION_DIR}/zymtrace_launch_manifest.json"
    if ((${#SCRIPT_ARGS[@]})); then
        write_zymtrace_manifest "$manifest" "$injection_lib" "${SCRIPT_ARGS[@]}"
    else
        write_zymtrace_manifest "$manifest" "$injection_lib"
    fi

    local cmd=(
        env
        "CUDA_INJECTION64_PATH=${injection_lib}"
        "ZYMTRACE_CUDA_INJECTION64_PATH=${injection_lib}"
        "$PYTHON_BIN"
        "$SCRIPT_PATH"
    )
    if ((${#SCRIPT_ARGS[@]})); then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Zymtrace launch manifest: ${manifest}"
}

run_pytorch() {
    local modes=("${PYTORCH_MODES[@]}")
    if ((${#modes[@]} == 0)); then
        modes=(full)
    fi
    for mode in "${modes[@]}"; do
        local out_dir="${SESSION_DIR}/pytorch_${mode}"
        local cmd=("$PYTHON_BIN" "$PYTORCH_RUNNER" "$SCRIPT_PATH" --output-dir "$out_dir" --profile-mode "$mode")
        if ((${#SCRIPT_ARGS[@]})); then
            cmd+=(--script-args "${SCRIPT_ARGS[@]}")
        fi
        print_command "${cmd[@]}"
        "${cmd[@]}"
        echo "  ↳ PyTorch profiler output: ${out_dir}"
    done
}

run_comprehensive() {
    run_nsys
    run_ncu
    run_pytorch
    run_hta
    run_perf
    if resolve_zymtrace_injection >/dev/null 2>&1; then
        run_zymtrace
    fi
}

if (( $# == 0 )); then
    print_usage
    exit 1
fi

case "$1" in
    --help|-h)
        print_usage
        exit 0
        ;;
    --list)
        env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "$PYTHON_DEFAULT" -m "$HARNESS_MODULE" --list
        exit $?
        ;;
    --examples|--example|--tags|--tag|--profile|--profile-mode|--output-root|--dry-run|--skip-existing|--max-examples|--force-build)
        env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "$PYTHON_DEFAULT" -m "$HARNESS_MODULE" "$@"
        exit $?
        ;;
esac

SCRIPT_PATH="$1"
shift

if [[ ! -f "$SCRIPT_PATH" ]]; then
    if [[ -f "${REPO_ROOT}/${SCRIPT_PATH}" ]]; then
        SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_PATH}"
    else
        echo "✗ Unable to locate script: ${SCRIPT_PATH}" >&2
        exit 1
    fi
fi

SCRIPT_PATH="$(resolve_path "$SCRIPT_PATH")"
SCRIPT_BASENAME="$(basename "$SCRIPT_PATH")"
SCRIPT_ARGS=()
ARCH="auto"
PROFILE_SPEC="all"
PYTORCH_MODES=()
PYTHON_BIN="${PYTHON:-python}"
# Allow overriding via environment but default to earlier value
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
ARCH_SET=false
PROFILE_SET=false

while (( $# > 0 )); do
    case "$1" in
        --arch)
            ARCH="$2"
            ARCH_SET=true
            shift 2
            ;;
        --tool|--profile-type|--profile)
            if $PROFILE_SET; then
                PROFILE_SPEC+="${PROFILE_SPEC:+,}$2"
            else
                PROFILE_SPEC="$2"
                PROFILE_SET=true
            fi
            shift 2
            ;;
        --pytorch-mode)
            PYTORCH_MODES+=("$2")
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --)
            shift
            SCRIPT_ARGS=("$@")
            break
            ;;
        *)
            if ! $ARCH_SET; then
                ARCH="$1"
                ARCH_SET=true
            elif ! $PROFILE_SET; then
                PROFILE_SPEC="$1"
                PROFILE_SET=true
            else
                echo "✗ Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
    [[ $# -eq 0 ]] && break
done

ARCH_VALUE="$(resolve_architecture "$ARCH")"
mkdir -p "$OUTPUT_ROOT"
SESSION_DIR="${OUTPUT_ROOT}/$(timestamp)_${SCRIPT_BASENAME%.*}"
mkdir -p "$SESSION_DIR"

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

if [[ "$SCRIPT_PATH" =~ /code/ch01/ || "$SCRIPT_PATH" =~ /code/ch02/ || "$SCRIPT_PATH" =~ /code/ch01[3-9]/ || "$SCRIPT_PATH" =~ /code/ch20/ ]]; then
    export TORCHINDUCTOR_AUTOTUNE=${TORCHINDUCTOR_AUTOTUNE:-0}
    export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
fi


PROFILE_SPEC="$(printf '%s' "$PROFILE_SPEC" | tr '[:upper:]' '[:lower:]')"
IFS="," read -r -a RAW_TOOLS <<< "$PROFILE_SPEC"
if ((${#RAW_TOOLS[@]} == 0)); then
    RAW_TOOLS=("all")
fi
TOOLS=()
for tool in "${RAW_TOOLS[@]}"; do
    case "$tool" in
        all)
            TOOLS=(nsys ncu pytorch hta perf)
            if resolve_zymtrace_injection >/dev/null 2>&1; then
                TOOLS+=(zymtrace)
            fi
            break
            ;;
        nsys|ncu|hta|perf|pytorch|torch|zymtrace)
            norm="$tool"
            [[ "$norm" == "torch" ]] && norm="pytorch"
            TOOLS+=("${norm}")
            ;;
        *)
            echo "✗ Unknown profile tool: ${tool}" >&2
            exit 1
            ;;
    esac
done

# Deduplicate while preserving order
DEDUP_TOOLS=()
for tool in "${TOOLS[@]:-}"; do
    [[ -z "$tool" ]] && continue
    seen=false
    for existing in "${DEDUP_TOOLS[@]:-}"; do
        [[ -z "$existing" ]] && continue
        if [[ "$existing" == "$tool" ]]; then
            seen=true
            break
        fi
    done
    if [[ "$seen" == false ]]; then
        DEDUP_TOOLS+=("$tool")
    fi
done
TOOLS=("${DEDUP_TOOLS[@]}")

cat <<SUMMARY
=== Profiling Session ===
Script       : ${SCRIPT_PATH}
Python       : ${PYTHON_BIN}
Architecture : ${ARCH_VALUE}
Tools        : ${TOOLS[*]}
Output Dir   : ${SESSION_DIR}
SUMMARY
if [[ -n "${SCRIPT_ARGS[*]-}" ]]; then
    printf 'Arguments    : %s\n' "$(printf '%q ' "${SCRIPT_ARGS[@]}")"
fi

for tool in "${TOOLS[@]:-}"; do
    case "$tool" in
        nsys)
            run_nsys
            ;;
        ncu)
            run_ncu
            ;;
        hta)
            run_hta
            ;;
        perf)
            run_perf
            ;;
        pytorch)
            run_pytorch
            ;;
        zymtrace)
            run_zymtrace
            ;;
        *)
            echo "✗ Unknown profile tool after normalization: ${tool}" >&2
            exit 1
            ;;
    esac
done

echo
echo "Profiling complete. Artifacts available under: ${SESSION_DIR}"
