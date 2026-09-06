#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run a strict DeepGEMM FP8xFP4 smoke/perf check and write:
  - raw log
  - structured JSON
  - clock-lock metadata

Usage:
  scripts/run_cluster_perf_fp4_smoke.sh [options]

Options:
  --run-id <id>     RUN_ID prefix (default: YYYY-MM-DD_fp4_smoke)
  --label <label>   Label used in output filenames (default: hostname)
  --runtime <mode>  host|container (default: host)
  --stack-profile <name>
                    Stack profile: old_container|orig_parity_container|new_container|host_only
                    (default: runtime-specific from configs/cluster_perf_stack_profiles.json)
  --image <image>   Container image for runtime=container
                    (default: stack-profile image_ref or $CONTAINER_IMAGE)
  --m <int>         M dimension (default: 4096)
  --n <int>         N dimension (default: 4096)
  --k <int>         K dimension (default: 4096)
  --warmup <n>      Warmup iterations (default: 10)
  --iters <n>       Measured iterations (default: 30)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" ]]; then
  echo "ERROR: missing stack profile helper: ${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" >&2
  exit 1
fi
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
# shellcheck source=scripts/cluster_perf_stack_profiles.sh
source "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh"

RUN_ID="$(date +%Y-%m-%d)_fp4_smoke"
LABEL="$(hostname)"
RUNTIME="host"
STACK_PROFILE=""
IMAGE="${CONTAINER_IMAGE:-}"
M="4096"
N="4096"
K="4096"
WARMUP="10"
ITERS="30"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --stack-profile) STACK_PROFILE="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --m) M="${2:-}"; shift 2 ;;
    --n) N="${2:-}"; shift 2 ;;
    --k) K="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  exit 2
fi
if [[ -z "$STACK_PROFILE" ]]; then
  STACK_PROFILE="$(cluster_perf_default_profile_for_runtime "$ROOT_DIR" "$RUNTIME")"
fi
if ! cluster_perf_profile_exists "$ROOT_DIR" "$STACK_PROFILE"; then
  echo "ERROR: unknown --stack-profile: ${STACK_PROFILE}" >&2
  exit 2
fi
if ! cluster_perf_profile_runtime_allowed "$ROOT_DIR" "$STACK_PROFILE" "$RUNTIME"; then
  echo "ERROR: --stack-profile ${STACK_PROFILE} does not allow runtime=${RUNTIME}" >&2
  exit 2
fi
if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  IMAGE="$(cluster_perf_profile_image_ref "$ROOT_DIR" "$STACK_PROFILE")"
fi
if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  echo "ERROR: no container image resolved for runtime=container/profile=${STACK_PROFILE}" >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "${OUT_RAW_DIR}" "${OUT_STRUCT_DIR}"

OUT_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke.log"
OUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke.json"
LOCK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke_clock_lock.json"
PREFLIGHT_STACK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke_preflight_stack.json"
PREFLIGHT_CLOCK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke_preflight_clock_lock.json"
CONTAINER_STRUCTURED_DIR="/cluster-results/structured"
OUT_STRUCT_DIR_MOUNT_SOURCE="$(cd "${OUT_STRUCT_DIR}" && pwd -P)"
OUT_JSON_IN_CONTAINER="${CONTAINER_STRUCTURED_DIR}/$(basename "${OUT_JSON}")"
MATH_ALLOW_TF32="$(cluster_perf_profile_math_allow_tf32 "$ROOT_DIR" "$STACK_PROFILE")"
MATH_FP32_MATMUL_PRECISION="$(cluster_perf_profile_math_precision "$ROOT_DIR" "$STACK_PROFILE")"
HOST_CUDA_HOME=""

resolve_host_cuda_home() {
  local resolved="${CUDA_HOME:-}"
  if [[ -z "$resolved" ]]; then
    if [[ -d /usr/local/cuda ]]; then
      resolved="/usr/local/cuda"
    elif command -v nvcc >/dev/null 2>&1; then
      local nvcc_path
      nvcc_path="$(command -v nvcc)"
      resolved="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
    fi
  fi
  if [[ -z "$resolved" || ! -f "${resolved}/include/cuda_runtime.h" ]]; then
    echo "ERROR: CUDA toolkit headers not found for host runtime (resolved CUDA_HOME=${resolved:-<empty>})." >&2
    exit 2
  fi
  printf '%s\n' "$resolved"
}

run_clocked_with_tee() {
  local lock_meta="$1"
  local out_log="$2"
  shift 2

  # Positional parameters are intentionally expanded by the child shell.
  # shellcheck disable=SC2016
  "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$lock_meta" \
    -- bash -o pipefail -c 'out_log="$1"; shift; "$@" 2>&1 | tee "$out_log"' \
    bash "$out_log" "$@"
}

echo "== Cluster Perf FP4 Smoke =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "RUNTIME=${RUNTIME}"
echo "STACK_PROFILE=${STACK_PROFILE}"
if [[ "$RUNTIME" == "container" ]]; then
  echo "IMAGE=${IMAGE}"
fi
echo "MATH_POLICY=allow_tf32:${MATH_ALLOW_TF32},float32_matmul_precision:${MATH_FP32_MATMUL_PRECISION}"
echo "SHAPE=${M}x${N}x${K}"
echo "WARMUP=${WARMUP}"
echo "ITERS=${ITERS}"
echo

if [[ ! -x "${ROOT_DIR}/env/venv/bin/python" ]]; then
  echo "ERROR: venv python not found: ${ROOT_DIR}/env/venv/bin/python" >&2
  exit 2
fi
if [[ "$RUNTIME" == "host" ]]; then
  if [[ "$STACK_PROFILE" == "host_only" ]]; then
    HOST_RUNTIME_ENV_FILE="${ROOT_DIR}/env/venv/orig_parity_runtime_env.sh"
    if [[ ! -f "$HOST_RUNTIME_ENV_FILE" ]]; then
      echo "ERROR: missing host runtime env file for host_only parity: ${HOST_RUNTIME_ENV_FILE}" >&2
      echo "Run scripts/bootstrap_cluster_nodes.sh (or run_fp4_checks_all_nodes.sh with bootstrap enabled) first." >&2
      exit 2
    fi
    # shellcheck disable=SC1090
    source "$HOST_RUNTIME_ENV_FILE"
  fi
  HOST_CUDA_HOME="$(resolve_host_cuda_home)"
fi

preflight_args=(
  --runtime "${RUNTIME}"
  --stack-profile "${STACK_PROFILE}"
  --profiles-json "${ROOT_DIR}/configs/cluster_perf_stack_profiles.json"
  --out-json "${PREFLIGHT_STACK_META}"
  --host-python "${ROOT_DIR}/env/venv/bin/python"
)
if [[ "$RUNTIME" == "container" ]]; then
  preflight_args+=(--image "${IMAGE}")
fi
"${ROOT_DIR}/env/venv/bin/python" "${ROOT_DIR}/scripts/preflight_cluster_perf_runtime.py" "${preflight_args[@]}"

# Fail fast before long benchmark runs if clocks cannot be locked.
"${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
  --lock-meta-out "${PREFLIGHT_CLOCK_META}" \
  -- bash -lc "set -euo pipefail; nvidia-smi -L >/dev/null"

if [[ "$RUNTIME" == "host" ]]; then
  echo "HOST_CUDA_HOME=${HOST_CUDA_HOME}"
  host_cmd=(
    env
    "CUDA_HOME=${HOST_CUDA_HOME}"
    "CUDACXX=${HOST_CUDA_HOME}/bin/nvcc"
    "CLUSTER_PERF_ALLOW_TF32=${MATH_ALLOW_TF32}"
    "CLUSTER_PERF_FLOAT32_MATMUL_PRECISION=${MATH_FP32_MATMUL_PRECISION}"
    "${ROOT_DIR}/env/venv/bin/python" -u
    "${ROOT_DIR}/analysis/smoke_deepgemm_fp8_fp4.py"
    --m "${M}"
    --n "${N}"
    --k "${K}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --out-json "${OUT_JSON}"
  )
  run_clocked_with_tee "$LOCK_META" "$OUT_LOG" "${host_cmd[@]}"
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found for --runtime container." >&2
    exit 2
  fi
  container_cmd=(
    docker run --rm --gpus all --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -e "CLUSTER_PERF_ALLOW_TF32=${MATH_ALLOW_TF32}"
    -e "CLUSTER_PERF_FLOAT32_MATMUL_PRECISION=${MATH_FP32_MATMUL_PRECISION}"
    --volume "${ROOT_DIR}:/workspace"
    --volume "${OUT_STRUCT_DIR_MOUNT_SOURCE}:${CONTAINER_STRUCTURED_DIR}"
    --workdir /workspace
    "${IMAGE}" python -u analysis/smoke_deepgemm_fp8_fp4.py
    --m "${M}"
    --n "${N}"
    --k "${K}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --out-json "${OUT_JSON_IN_CONTAINER}"
  )
  run_clocked_with_tee "$LOCK_META" "$OUT_LOG" "${container_cmd[@]}"
fi

echo
echo "Outputs:"
echo "  - $OUT_LOG"
echo "  - $OUT_JSON"
echo "  - $LOCK_META"
echo "  - $PREFLIGHT_STACK_META"
echo "  - $PREFLIGHT_CLOCK_META"
