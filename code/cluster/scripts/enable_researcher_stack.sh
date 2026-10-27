#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="${LABEL:-$(hostname)}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"

APPLY=0
INSTALL_BASE_PKGS=0
INSTALL_CUDA_TOOLKIT=0
INSTALL_PYTHON_STACK=0
ENABLE_SERVICES=1
ENSURE_DOCKER_GROUP=1

usage() {
  cat <<'USAGE'
Usage: enable_researcher_stack.sh [options]

Default behavior is dry-run. Use --apply to make changes.

Core toggles:
  --apply                    Apply changes (otherwise dry-run)
  --no-enable-services       Skip enabling NVIDIA services
  --no-docker-group          Skip adding user to docker group

Optional installs:
  --install-base-pkgs        Install common tooling (git, curl, jq, build tools, python venv)
  --install-cuda-toolkit     Install CUDA toolkit via apt (driver must already be installed)
  --install-python-stack     Create a fresh venv from the repository runtime lock

Environment overrides:
  RUN_ID=YYYY-MM-DD          Output file prefix
  LABEL=<label>              Label for logs (default: hostname)
  VENV_DIR=<path>            Virtualenv path (default: /opt/cluster-torch2141)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift 1
      ;;
    --no-enable-services)
      ENABLE_SERVICES=0
      shift 1
      ;;
    --no-docker-group)
      ENSURE_DOCKER_GROUP=0
      shift 1
      ;;
    --install-base-pkgs)
      INSTALL_BASE_PKGS=1
      shift 1
      ;;
    --install-cuda-toolkit)
      INSTALL_CUDA_TOOLKIT=1
      shift 1
      ;;
    --install-python-stack)
      INSTALL_PYTHON_STACK=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCTURED="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW" "$OUT_STRUCTURED"
LOG_FILE="${OUT_RAW}/${RUN_ID}_${LABEL}_enable_stack.log"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -Iseconds
}

run_or_echo() {
  local cmd="$*"
  if [[ "$APPLY" -eq 1 ]]; then
    eval "$cmd"
  else
    echo "[dry-run] $cmd"
  fi
}

echo "=== enable_researcher_stack.sh ==="
echo "Run ID: ${RUN_ID}"
echo "Label: ${LABEL}"
echo "Apply: ${APPLY}"
echo "Log: ${LOG_FILE}"

echo "[$(timestamp)] Checking NVIDIA container toolkit version"
if command -v nvidia-container-toolkit >/dev/null 2>&1; then
  nvidia-container-toolkit --version || true
elif command -v dpkg >/dev/null 2>&1; then
  dpkg -l | grep -E 'nvidia-container-toolkit|nvidia-container-runtime' || true
fi

if [[ "$ENABLE_SERVICES" -eq 1 ]]; then
  echo "[$(timestamp)] Enabling NVIDIA services"
  run_or_echo "sudo systemctl enable --now nvidia-persistenced"
  run_or_echo "sudo systemctl enable --now nvidia-fabricmanager"
  run_or_echo "sudo systemctl enable --now nvidia-dcgm"
fi

if [[ "$ENSURE_DOCKER_GROUP" -eq 1 ]]; then
  echo "[$(timestamp)] Ensuring docker group access for $USER"
  if id -nG "$USER" | tr ' ' '\n' | grep -q '^docker$'; then
    echo "User ${USER} already in docker group"
  else
    run_or_echo "sudo usermod -aG docker ${USER}"
    echo "Note: re-login required for docker group to take effect"
  fi
fi

if [[ "$INSTALL_BASE_PKGS" -eq 1 ]]; then
  echo "[$(timestamp)] Installing base packages"
  run_or_echo "sudo apt-get update"
  run_or_echo "sudo apt-get install -y git curl jq build-essential python3-venv python3-pip"
fi

if [[ "$INSTALL_CUDA_TOOLKIT" -eq 1 ]]; then
  echo "[$(timestamp)] Installing CUDA toolkit (apt)"
  run_or_echo "sudo apt-get update"
  run_or_echo "sudo apt-get install -y cuda-toolkit-13-0"
fi

if [[ "$INSTALL_PYTHON_STACK" -eq 1 ]]; then
  VENV_DIR="${VENV_DIR:-/opt/cluster-torch2141}"
  CODE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
  STACK_LOCK="${CODE_ROOT}/requirements_torch2141_linux_x86_64.txt"
  if [[ ! -f "${STACK_LOCK}" ]]; then
    echo "ERROR: Runtime lock is missing: ${STACK_LOCK}" >&2
    exit 1
  fi
  if [[ -e "${VENV_DIR}" ]]; then
    echo "ERROR: Choose a new VENV_DIR to preserve the existing environment: ${VENV_DIR}" >&2
    exit 1
  fi
  run_stack_command() {
    if [[ "$APPLY" -eq 1 ]]; then
      "$@"
    else
      printf '[dry-run] '
      printf '%q ' "$@"
      printf '\n'
    fi
  }
  echo "[$(timestamp)] Creating isolated Python 3.12 runtime at ${VENV_DIR}"
  run_stack_command sudo mkdir -p "${VENV_DIR}"
  run_stack_command sudo chown "${USER}:${USER}" "${VENV_DIR}"
  run_stack_command python3.12 -m venv "${VENV_DIR}"
  run_stack_command "${VENV_DIR}/bin/python" -m pip install --require-hashes --no-deps -r "${STACK_LOCK}"
  run_stack_command "${VENV_DIR}/bin/python" -m pip check
  echo "Activate via: source ${VENV_DIR}/bin/activate"
  echo "The base graph does not include the ABI-bound TE and vLLM bindings."
  echo "Build and validate those source wheels using docs/environment.md before serving."
fi

echo "[$(timestamp)] enable_researcher_stack.sh complete"
