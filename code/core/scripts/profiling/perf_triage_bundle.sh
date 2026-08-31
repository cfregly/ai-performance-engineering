#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Capture a lightweight performance triage bundle (snapshots + short run).

Usage:
  perf_triage_bundle.sh [--output-root DIR] [--tag LABEL] [--duration SEC]
                        [--nsys | --no-nsys]
                        [--] <command to profile>

Examples:
  perf_triage_bundle.sh --output-root ./artifacts/runs
  perf_triage_bundle.sh --output-root ./artifacts/runs --tag baseline -- \
    python ch01/baseline_matmul.py --batch-size 32
  perf_triage_bundle.sh --output-root ./artifacts/runs --nsys --duration 90 -- \
    python your_script.py --arg foo

Behavior:
  - Always writes hardware/software snapshots to PERF_TRIAGE_*/ (GPU/CPU/mem/storage/network, CUDA/PyTorch versions).
  - If Nsight Systems is available (default), profiles the command with CUDA/NVTX/OSRT/cuDNN/cuBLAS traces and emits .nsys-rep + summary.
  - Otherwise, runs nvidia-smi dmon while the command executes and saves a CSV timeseries.
  - If `timeout` is available, wraps the command with the requested duration.
  - Bundles everything into PERF_TRIAGE_*.tgz for easy sharing.
USAGE
}

OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/artifacts/runs}"
TAG=""
DURATION=60
NSYS_MODE="auto"  # auto | on | off
CMD=()
CAPTURE_MODE="snapshots_only"
CMD_EXIT="not_run"
TIMEOUT_BIN="$(command -v timeout || true)"
TIMEOUT_USED="no"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"; shift 2;;
        --tag)
            TAG="$2"; shift 2;;
        --duration)
            DURATION="$2"; shift 2;;
        --nsys)
            NSYS_MODE="on"; shift;;
        --no-nsys)
            NSYS_MODE="off"; shift;;
        -h|--help)
            usage; exit 0;;
        --)
            shift
            CMD=("$@")
            break;;
        *)
            CMD+=("$1"); shift;;
    esac
done

timestamp() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$OUTPUT_ROOT"
HOST=$(hostname || echo "unknown_host")
TS=$(timestamp)
DIR_NAME="perf_triage_${HOST}_${TS}"
if [[ -n "$TAG" ]]; then
    DIR_NAME="${DIR_NAME}_${TAG}"
fi
OUT_DIR="${OUTPUT_ROOT%/}/${DIR_NAME}"
mkdir -p "$OUT_DIR"

cleanup_pids=()
cleanup() {
    for pid in "${cleanup_pids[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

run_or_note() {
    local cmd="$1" output="$2"
    if command -v ${cmd%% *} >/dev/null 2>&1; then
        eval "$cmd" >"$output" 2>/dev/null || true
    else
        echo "${cmd%% *} not found in PATH" >"$output"
    fi
}

log "Writing system snapshots to $OUT_DIR"
run_or_note "nvidia-smi -L" "$OUT_DIR/gpu_list.txt"
run_or_note "nvidia-smi -q" "$OUT_DIR/gpu_diagnostics.txt"
run_or_note "nvidia-smi topo -m" "$OUT_DIR/nvlink_topology.txt"
run_or_note "lscpu" "$OUT_DIR/cpu.txt"
run_or_note "free -h" "$OUT_DIR/memory.txt"
run_or_note "lsblk -o NAME,SIZE,MODEL,TYPE,MOUNTPOINT" "$OUT_DIR/storage.txt"
run_or_note "which nvcc && nvcc --version" "$OUT_DIR/cuda_version.txt"
run_or_note "ibv_devinfo" "$OUT_DIR/ibverbs.txt"

PYTHON_BIN="${PYTHON:-python}"
PY_SCRIPT="$OUT_DIR/python_env_probe.py"
cat >"$PY_SCRIPT" <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"torch import failed: {exc}")
    sys.exit(0)
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "sm_arch", torch.cuda.get_device_capability())
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(idx)
        name = torch.cuda.get_device_name(idx)
        print(f"gpu{idx}", name, cap)
PY

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" "$PY_SCRIPT" >"$OUT_DIR/python_env.txt" 2>"$OUT_DIR/python_env.err" || true
else
    echo "$PYTHON_BIN not found in PATH" >"$OUT_DIR/python_env.txt"
fi

# Comprehensive capability report
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    (cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && "$PYTHON_BIN" -m scripts.utilities.dump_hardware_capabilities) \
        >"$OUT_DIR/hardware_capabilities.txt" 2>"$OUT_DIR/hardware_capabilities.err" || true
else
    echo "$PYTHON_BIN not found in PATH" >"$OUT_DIR/hardware_capabilities.txt"
fi

run_command() {
    if [[ ${#CMD[@]} -eq 0 ]]; then
        log "No command provided; skipping runtime capture."
        return
    fi

    local runner=("${CMD[@]}")
    if [[ -n "$TIMEOUT_BIN" ]]; then
        runner=("$TIMEOUT_BIN" "${DURATION}s" "${CMD[@]}")
        TIMEOUT_USED="yes"
    fi

    if [[ "$NSYS_MODE" != "off" ]] && command -v nsys >/dev/null 2>&1; then
        log "Running command under Nsight Systems"
        local base="$OUT_DIR/run"
        CAPTURE_MODE="nsys"
        set +e
        nsys profile -o "$base" --force-overwrite=true \
            -t cuda,nvtx,osrt,cudnn,cublas \
            --capture-range=nvtx --stop-on-exit=true \
            "${runner[@]}"
        local status=$?
        set -e
        CMD_EXIT=$status
        nsys stats "${base}.nsys-rep" >"$OUT_DIR/run.nsys.txt" 2>/dev/null || true
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi not available; skipping runtime capture."
        return
    fi

    log "Running command with nvidia-smi dmon sampling (duration ~${DURATION}s)"
    CAPTURE_MODE="dmon"
    nvidia-smi dmon -s pucm -d 1 -o DT -f "$OUT_DIR/gpu_timeseries.csv" >/dev/null 2>&1 &
    cleanup_pids+=($!)
    local status=0
    set +e
    "${runner[@]}"
    status=$?
    set -e
    CMD_EXIT=$status
    if ((${#cleanup_pids[@]})); then
        cleanup
        cleanup_pids=()
    fi
}

run_command

cat >"$OUT_DIR/manifest.txt" <<EOF
host: $HOST
timestamp: $TS
tag: ${TAG:-none}
nsys_mode: $NSYS_MODE
duration_seconds: $DURATION
command: ${CMD[*]:-none}
capture_mode: $CAPTURE_MODE
exit_code: $CMD_EXIT
timeout_used: $TIMEOUT_USED
EOF

log "Packaging bundle"
tar -czf "${OUT_DIR}.tgz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
log "Done. Bundle: ${OUT_DIR}.tgz"
