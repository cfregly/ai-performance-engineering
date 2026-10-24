#!/bin/bash

#
# AI Performance Engineering Setup Script
# ========================================
#
# This script installs EVERYTHING you need:
#   1. NVIDIA Driver 580.126.09 (auto-upgrades if needed; open kernel modules for B200/GB200)
#   2. Python 3.12
#   3. CUDA 13.0.2 toolkit + cuBLAS 13.1.0.3 (Update 2) repository
#   4. The canonical PyTorch/Triton stack pinned in requirements_latest.txt
#      (torch==2.14.1+cu130, triton==3.8.0) for CUDA 13.0.2. The script requires
#      these exact pins in requirements_latest.txt and the hash lock.
#   5. NVIDIA Nsight Systems 2026.5.1 CLI (for timeline profiling)
#   6. NVIDIA Nsight Compute 2026.2.1 (for kernel metrics), the version that
#      core/profiling/ncu_torchrun_capture.py accepts
#   7. CMake and ninja from the pip pins in requirements_latest.txt
#   8. All Python dependencies from requirements_latest.txt
#   9. System tools (numactl, perf, htop, etc.)
#  10. Configures NVIDIA drivers for profiling
#
# Requirements:
#   - Ubuntu 22.04 or 24.04 on x86_64 (tested on 22.04). NVIDIA publishes the
#     CUDA 13.0.2 and Nsight packages below for both releases, not for 20.04.
#   - NVIDIA B200 or GB200 GPU with compute capability 10.0
#   - sudo/root access
#   - Internet connection
#
# Usage:
#   sudo ./setup.sh
#   (logs to ./setup.log by default; override with SETUP_LOG_FILE)
#
# Duration: 10-20 minutes (first run may require reboot for driver upgrade)
#
# What it does:
#   - Adds official NVIDIA CUDA 13.0 (Update 2) repository
#   - Configures APT to prefer official NVIDIA packages
#   - Disables problematic command-not-found APT hook
#   - Removes duplicate deadsnakes repository entries
#   - Creates a fresh isolated Python 3.12 runtime for the pinned PyTorch stack
#   - Auto-upgrades NVIDIA driver to 580+ if needed (will prompt reboot)
#   - Installs CUDA 13.0.2 toolkit and libraries
#   - Installs pinned Nsight Systems CLI 2026.5.1 and Nsight Compute 2026.2.1
#     and puts them ahead of the CUDA toolkit's bundled 2025.3.x copies on PATH
#   - Installs the pinned CMake and ninja wheels and stops if CMake is older
#     than the newest cmake_minimum_required in first-party CMakeLists.txt files
#   - Installs pinned PyTorch CUDA 13 wheels (no source build)
#   - Installs nvidia-ml-py (replaces deprecated pynvml)
#   - Configures NVIDIA kernel modules for profiling
#   - Fixes hardware info script compatibility
#   - Runs validation tests
#
# Notes:
#   - If driver upgrade is needed, script will exit and ask you to reboot
#   - After reboot, simply re-run: sudo ./setup.sh
#   - Every run requires a fresh AISP_VENV_DIR so an earlier runtime cannot
#     contaminate the hash-locked environment
#   - Disk cleanup is enabled by default (CLEAN_APT_CACHE/CLEAN_PIP_CACHE/CLEAN_BUILD_ARTIFACTS)
#   - Minimum free space thresholds: SETUP_MIN_FREE_GB / SETUP_MIN_TE_BUILD_FREE_GB
#
# After running this script, you can:
#   - Run examples: python3 ch01/performance_basics.py
#   - Drive the benchmark suite: python -m cli.aisp bench run
#   - Capture peak performance: python core/benchmark/benchmark_peak.py
#

set -e  # Exit on any error

resolve_setup_directory() {
    local source_path="${BASH_SOURCE[0]}"
    local source_dir
    while [ -L "${source_path}" ]; do
        source_dir="$(cd -P "$(dirname "${source_path}")" && pwd -P)"
        source_path="$(readlink "${source_path}")"
        if [[ "${source_path}" != /* ]]; then
            source_path="${source_dir}/${source_path}"
        fi
    done
    cd -P "$(dirname "${source_path}")" && pwd -P
}

discover_repository_root() {
    local candidate="$1"
    while [ "${candidate}" != "/" ]; do
        if [ -e "${candidate}/.git" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        candidate="$(dirname "${candidate}")"
    done
    return 1
}

SCRIPT_DIR="$(resolve_setup_directory)"
PROJECT_ROOT="${SCRIPT_DIR}"
if ! REPOSITORY_ROOT="$(discover_repository_root "${PROJECT_ROOT}")"; then
    REPOSITORY_ROOT="${PROJECT_ROOT}"
fi
CUDA_FULL_VERSION="13.0.2"
SUPPORTED_NATIVE_GPU_COMPUTE_CAPABILITY="10.0"
GPU_COMPUTE_CAP_RAW=""
GPU_COMPUTE_SM_NUM=""
GPU_CUDA_ARCH_LIST=""

require_supported_native_gpu_compute_capability() {
    local detected_capability="$1"
    if [ -z "${detected_capability}" ]; then
        echo "ERROR: Could not detect the GPU compute capability." >&2
        return 1
    fi
    if [ "${detected_capability}" != "${SUPPORTED_NATIVE_GPU_COMPUTE_CAPABILITY}" ]; then
        echo "ERROR: This CUDA 13.0 and PyTorch 2.14.1 native setup supports only NVIDIA B200/GB200 compute capability ${SUPPORTED_NATIVE_GPU_COMPUTE_CAPABILITY}." >&2
        if [ "${detected_capability}" = "10.3" ]; then
            echo "       NVIDIA B300/GB300 compute capability 10.3 requires a separate SM103 native build contract and qualification." >&2
        else
            echo "       Detected unsupported compute capability ${detected_capability}." >&2
        fi
        return 1
    fi

    GPU_COMPUTE_CAP_RAW="${detected_capability}"
    GPU_COMPUTE_SM_NUM="${detected_capability//./}"
    GPU_CUDA_ARCH_LIST="${detected_capability}"
    export GPU_COMPUTE_CAP_RAW GPU_COMPUTE_SM_NUM GPU_CUDA_ARCH_LIST
}

detect_supported_native_gpu() {
    local detected_capabilities detected_capability detected_count
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: NVIDIA GPU not detected. Ensure the NVIDIA driver is installed before running setup." >&2
        return 1
    fi
    if ! detected_capabilities="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '[:blank:]\r')"; then
        echo "ERROR: Could not query GPU compute capability with nvidia-smi." >&2
        return 1
    fi

    detected_count=0
    while IFS= read -r detected_capability; do
        if [ -z "${detected_capability}" ]; then
            continue
        fi
        detected_count=$((detected_count + 1))
        require_supported_native_gpu_compute_capability "${detected_capability}" || return 1
    done <<< "${detected_capabilities}"
    if [ "${detected_count}" -eq 0 ]; then
        echo "ERROR: Could not detect the GPU compute capability." >&2
        return 1
    fi
    echo "Detected ${detected_count} supported GPU(s) at compute capability ${GPU_COMPUTE_CAP_RAW} (sm_${GPU_COMPUTE_SM_NUM})."
}

bootstrap_repository_checkout() {
    if [ ! -e "${REPOSITORY_ROOT}/.git" ]; then
        return 0
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: Git is required to prepare repository submodules." >&2
        return 1
    fi

    # Under sudo, Git sees the checkout as owned by another user. Mark the
    # physical repository root safe before any command reads repository data.
    git config --global --add safe.directory "${REPOSITORY_ROOT}"
    if [ -d "${PROJECT_ROOT}/vendor/pytorch-src" ]; then
        git config --global --add safe.directory "${PROJECT_ROOT}/vendor/pytorch-src"
    fi
    if [ -f "${REPOSITORY_ROOT}/.gitmodules" ]; then
        echo "Synchronizing submodules from ${REPOSITORY_ROOT}/.gitmodules..."
        git -C "${REPOSITORY_ROOT}" submodule sync --recursive
        git -C "${REPOSITORY_ROOT}" submodule update --init --recursive
    fi
}

if [ "${AISP_SETUP_BOOTSTRAP_ONLY:-0}" = "1" ]; then
    bootstrap_repository_checkout
    printf 'Repository bootstrap complete: %s\n' "${REPOSITORY_ROOT}"
    exit 0
fi

require_supported_host() {
    local architecture distributor ID VERSION_ID
    architecture="$(uname -m)"
    if [ "${architecture}" != "x86_64" ]; then
        echo "ERROR: Architecture ${architecture} is not supported. This setup requires Ubuntu 22.04 or 24.04 on x86_64." >&2
        return 1
    fi

    if command -v lsb_release >/dev/null 2>&1; then
        if ! distributor="$(lsb_release -is)" || ! UBUNTU_VERSION="$(lsb_release -rs)"; then
            echo "ERROR: Could not read the Linux distribution with lsb_release." >&2
            return 1
        fi
    elif [ -r /etc/os-release ]; then
        # The fixed host file is the fallback when lsb_release is unavailable.
        # shellcheck disable=SC1091
        . /etc/os-release
        distributor="${ID:-}"
        UBUNTU_VERSION="${VERSION_ID:-}"
    else
        echo "ERROR: Cannot identify this host. Expected Ubuntu 22.04 or 24.04 on x86_64." >&2
        return 1
    fi

    case "${distributor}" in
        Ubuntu|ubuntu) ;;
        *)
            echo "ERROR: ${distributor:-Unknown Linux} is not supported. This setup requires Ubuntu 22.04 or 24.04 on x86_64." >&2
            return 1
            ;;
    esac

    echo "Detected Ubuntu version: ${UBUNTU_VERSION}"
    case "${UBUNTU_VERSION}" in
        22.04) ;;
        24.04)
            echo "Warning: This script is tested on Ubuntu 22.04. Ubuntu 24.04 uses NVIDIA's ubuntu2404 CUDA repository and is not yet validated here."
            ;;
        *)
            echo "ERROR: Ubuntu ${UBUNTU_VERSION} is not supported. This script installs CUDA ${CUDA_FULL_VERSION} from NVIDIA's Ubuntu 22.04 or 24.04 repository." >&2
            return 1
            ;;
    esac
    CUDA_REPO_DISTRO="ubuntu${UBUNTU_VERSION//./}"
}

require_supported_host

# Reject an existing runtime before setup changes the host. A failed or
# completed run is retained for inspection and never mutated in place.
AISP_VENV_DIR="${AISP_VENV_DIR:-${PROJECT_ROOT}/.venv-torch2141}"
if [[ "${AISP_VENV_DIR}" != /* ]]; then
    echo "ERROR: AISP_VENV_DIR must be an absolute fresh path: ${AISP_VENV_DIR}" >&2
    exit 1
fi
if [ -e "${AISP_VENV_DIR}" ] || [ -L "${AISP_VENV_DIR}" ]; then
    echo "ERROR: AISP_VENV_DIR must be a fresh path: ${AISP_VENV_DIR}" >&2
    echo "       Preserve the existing runtime and choose a new AISP_VENV_DIR." >&2
    exit 1
fi
detect_supported_native_gpu
bootstrap_repository_checkout

# =============================================================================
# Logging + Disk Hygiene (always-on logging, keep disk usage tidy)
# =============================================================================
LOG_FILE="${SETUP_LOG_FILE:-${PROJECT_ROOT}/setup.log}"
LOG_MAX_BYTES="${SETUP_LOG_MAX_BYTES:-104857600}"  # 100 MB
LOG_COMPRESS="${SETUP_LOG_COMPRESS:-1}"
SETUP_MIN_FREE_GB="${SETUP_MIN_FREE_GB:-6}"
SETUP_MIN_TE_BUILD_FREE_GB="${SETUP_MIN_TE_BUILD_FREE_GB:-8}"
CLEAN_APT_CACHE="${CLEAN_APT_CACHE:-1}"
CLEAN_PIP_CACHE="${CLEAN_PIP_CACHE:-0}"
CLEAN_BUILD_ARTIFACTS="${CLEAN_BUILD_ARTIFACTS:-1}"

rotate_setup_log() {
    if [ -f "${LOG_FILE}" ]; then
        local log_size
        log_size="$(stat -c %s "${LOG_FILE}" 2>/dev/null || echo 0)"
        if [ "${log_size}" -ge "${LOG_MAX_BYTES}" ]; then
            local ts rotated
            ts="$(date +%Y%m%d_%H%M%S)"
            rotated="${LOG_FILE}.${ts}"
            mv "${LOG_FILE}" "${rotated}" 2>/dev/null || true
            if [ "${LOG_COMPRESS}" -eq 1 ] && command -v gzip >/dev/null 2>&1; then
                gzip -f "${rotated}" >/dev/null 2>&1 || true
            fi
        fi
    fi
}

start_setup_logging() {
    local log_dir
    log_dir="$(dirname "${LOG_FILE}")"
    mkdir -p "${log_dir}"
    rotate_setup_log
    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        touch "${LOG_FILE}" 2>/dev/null || true
        chown "${SUDO_USER}":"${SUDO_USER}" "${LOG_FILE}" 2>/dev/null || true
    fi
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "Logging to ${LOG_FILE}"
}

get_free_kb() {
    df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}'
}

reclaim_disk_space_basic() {
    echo "Reclaiming disk space (APT + pip caches)..."
    if [ "${CLEAN_APT_CACHE}" -eq 1 ] && [ "$(id -u)" -eq 0 ]; then
        apt-get clean >/dev/null 2>&1 || true
        rm -rf /var/lib/apt/lists/* >/dev/null 2>&1 || true
    fi
    if [ "${CLEAN_PIP_CACHE}" -eq 1 ] && [ -x "${AISP_VENV_DIR}/bin/python" ]; then
        "${AISP_VENV_DIR}/bin/python" -m pip cache purge >/dev/null 2>&1 || true
    fi
}

ensure_free_space_gb() {
    local min_gb="$1"
    local reason="$2"
    local free_kb free_gb
    free_kb="$(get_free_kb)"
    free_gb=$((free_kb / 1024 / 1024))
    if [ "${free_gb}" -lt "${min_gb}" ]; then
        echo "Low disk space: ${free_gb} GB free (need ${min_gb} GB) for ${reason}."
        reclaim_disk_space_basic
        free_kb="$(get_free_kb)"
        free_gb=$((free_kb / 1024 / 1024))
        if [ "${free_gb}" -lt "${min_gb}" ]; then
            echo "ERROR: Insufficient disk space after cleanup (${free_gb} GB free)."
            echo "       Free up space or set SETUP_MIN_FREE_GB/SETUP_MIN_TE_BUILD_FREE_GB."
            exit 1
        fi
    fi
}

start_setup_logging
ensure_free_space_gb "${SETUP_MIN_FREE_GB}" "setup start"

echo "AI Performance Engineering Setup Script"
echo "=========================================="
echo "This script will install:"
echo "  • NVIDIA Driver 580.126.09 (auto-upgrade if needed)"
echo "  • Python 3.12 for the pinned PyTorch stack"
echo "  • CUDA 13.0.2 toolkit + cuBLAS 13.1.0.3 (Update 2) repository"
echo "  • PyTorch CUDA 13 wheels pinned by requirements_latest.txt"
echo "  • NVIDIA Nsight Systems 2026.5.1"
echo "  • NVIDIA Nsight Compute 2026.2.1"
echo "  • CMake and ninja pinned by requirements_latest.txt"
echo "  • All project dependencies"
echo "  • System tools (numactl, perf, etc.)"
echo ""
echo "Note: If driver upgrade is needed, you'll be prompted to reboot."
echo ""

REQUIRED_DRIVER_VERSION="580.126.09"
PYTHON_TARGET_VERSION="3.12"
PYTHON_TARGET_BIN="python${PYTHON_TARGET_VERSION}"
CUDA_SHORT_VERSION="13.0"
# The CUDA toolkit meta-package is pinned to the Update 2 repository version.
CUDA_HOME_DIR="/usr/local/cuda-${CUDA_SHORT_VERSION}"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
# Optional GPT-OSS dependencies use a separately pinned runtime.
if [ "${INSTALL_GPT_OSS:-0}" = "1" ]; then
    echo "ERROR: GPT-OSS requires a separately pinned runtime outside the shared torch 2.14 environment."
    exit 1
fi
VLLM_PIN_FILE="${VLLM_PIN_FILE:-${PROJECT_ROOT}/vllm_no_deps.pin}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements_latest.txt"
VLLM_SOURCE_BUILD_MANIFEST="${PROJECT_ROOT}/vllm_source_build.json"
VLLM_PIP_SPEC=""
FLASHINFER_EXPECTED_VERSION=""

requirements_pin() {
    local package="$1"
    local version
    version="$(sed -n -E "s/^${package}==([^[:space:]#]+).*/\1/p" "${REQUIREMENTS_FILE}" | head -n 1)"
    if [ -z "${version}" ]; then
        echo "ERROR: ${REQUIREMENTS_FILE} must pin ${package} with an exact == version." >&2
        return 1
    fi
    printf '%s\n' "${version}"
}
# =============================================================================
# DEPENDENCY VERSION PINS (update together, test after changes)
# Run: python core/scripts/check_upstream_versions.py --check-te-cutlass
# Source of truth for pinned versions lives here (dependency_versions.json removed)
# =============================================================================
#
# CUTLASS 4.5.2 - Validated for SM100a (Blackwell) and SM103a support
#   - Provides: tmem_allocator_sm100.hpp, mma_sm100_umma.hpp, copy_traits_sm100.hpp
#   - Uses the release tag with corrected version metadata
#
# Transformer Engine's exact source archive and build contract live in
# native_torch_extensions.json. The torch binding is rebuilt for this ABI.
#
# MLPerf v6 source trees
#   - Inference README explicitly recommends the master branch for v6.0 submissions.
#   - Training README carries a v6.0 section but does not publish a GitHub release tag,
#     so we track its official branch ref and record the resolved commit at install time.
#
CUTLASS_REPO_URL="${CUTLASS_REPO_URL:-https://github.com/NVIDIA/cutlass.git}"
# Immutable commit behind the official CUTLASS 4.5.2 release.
CUTLASS_REF="${CUTLASS_REF:-db1c288993354c88e551c40c19a8fb93a774a241}"
CUTLASS_RELEASE_TAG="v4.5.2"
CUTLASS_TARGET_VERSION="${CUTLASS_TARGET_VERSION:-4.5.2}"
CUTLASS_SRC_DIR="${CUTLASS_SRC_DIR:-${THIRD_PARTY_DIR}/cutlass}"
MLPERF_INFERENCE_REPO_URL="${MLPERF_INFERENCE_REPO_URL:-https://github.com/mlcommons/inference.git}"
MLPERF_INFERENCE_GIT_REF="${MLPERF_INFERENCE_GIT_REF:-master}"
MLPERF_INFERENCE_SRC_DIR="${MLPERF_INFERENCE_SRC_DIR:-${THIRD_PARTY_DIR}/mlperf_inference}"
MLPERF_TRAINING_REPO_URL="${MLPERF_TRAINING_REPO_URL:-https://github.com/mlcommons/training.git}"
MLPERF_TRAINING_GIT_REF="${MLPERF_TRAINING_GIT_REF:-master}"
MLPERF_TRAINING_SRC_DIR="${MLPERF_TRAINING_SRC_DIR:-${THIRD_PARTY_DIR}/mlperf_training}"
PIP_ROOT_USER_ACTION="ignore"
export PROJECT_ROOT REQUIRED_DRIVER_VERSION PYTHON_TARGET_VERSION PYTHON_TARGET_BIN PIP_ROOT_USER_ACTION
# These values must be populated from exact requirements_latest.txt pins.
PYTORCH_TORCH_VERSION=""
PYTORCH_TORCHAUDIO_VERSION=""
PYTORCH_TORCHVISION_VERSION=""
PYTORCH_TORCHAO_VERSION=""
PYTORCH_TRITON_VERSION=""
PYTORCH_STABLE_INDEX="https://download.pytorch.org/whl"
PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "ERROR: Required pin file is missing: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi
PYTORCH_TORCH_VERSION="$(requirements_pin torch)"
PYTORCH_TORCHAUDIO_VERSION="$(requirements_pin torchaudio)"
PYTORCH_TORCHVISION_VERSION="$(requirements_pin torchvision)"
PYTORCH_TORCHAO_VERSION="$(requirements_pin torchao)"
PYTORCH_TRITON_VERSION="$(requirements_pin triton)"
FLASHINFER_EXPECTED_VERSION="$(requirements_pin flashinfer-python)"
REQ_CMAKE_VERSION="$(requirements_pin cmake)"
REQ_NINJA_VERSION="$(requirements_pin ninja)"
PYTORCH_STACK_LOCK="${PROJECT_ROOT}/requirements_torch2141_linux_x86_64.txt"
if [ ! -f "${PYTORCH_STACK_LOCK}" ]; then
    echo "ERROR: The PyTorch runtime lock is missing: ${PYTORCH_STACK_LOCK}" >&2
    exit 1
fi
python3 - "${PYTORCH_STACK_LOCK}" \
    "torch=${PYTORCH_TORCH_VERSION}" \
    "torchaudio=${PYTORCH_TORCHAUDIO_VERSION}" \
    "torchvision=${PYTORCH_TORCHVISION_VERSION}" \
    "torchao=${PYTORCH_TORCHAO_VERSION}" \
    "triton=${PYTORCH_TRITON_VERSION}" \
    "flashinfer-python=${FLASHINFER_EXPECTED_VERSION}" \
    "cmake=${REQ_CMAKE_VERSION}" \
    "ninja=${REQ_NINJA_VERSION}" <<'PYLOCKPINS'
from pathlib import Path
import re
import sys


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


locked = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s\\]+)", line)
    if match:
        locked[canonical(match.group(1))] = match.group(2)

for pair in sys.argv[2:]:
    name, expected = pair.split("=", 1)
    actual = locked.get(canonical(name))
    if actual != expected:
        raise SystemExit(
            f"Lock mismatch for {name}: requirements_latest.txt has {expected}, lock has {actual}"
        )
print("Core requirements pins match the hash lock")
PYLOCKPINS

pytorch_package_index() {
    local version="$1"
    local cuda_variant="${2:-}"
    local index="${PYTORCH_STABLE_INDEX}"
    case "${version}" in
        *.dev*|*+git*) index="${PYTORCH_NIGHTLY_INDEX}" ;;
    esac
    if [ -n "${cuda_variant}" ]; then
        index="${index}/${cuda_variant}"
    fi
    printf '%s\n' "${index}"
}

require_torchao_cuda_wheel_support() {
    local architecture="$1"
    if [ "${architecture}" = "aarch64" ] && [ "${PYTORCH_TORCHAO_VERSION}" = "0.18.0+cu130" ]; then
        echo "ERROR: No official aarch64 CUDA wheel is published for torchao==${PYTORCH_TORCHAO_VERSION}." >&2
        echo "       This binary-wheel setup cannot complete on that platform; no CPU substitute will be installed." >&2
        echo "       A separate torchao v0.18.0 source build against torch 2.14.1 and CUDA 13 requires target validation." >&2
        echo "       Upstream build instructions: https://github.com/pytorch/ao/tree/v0.18.0#installation" >&2
        return 1
    fi
}

GPU_CLOCK_SERVICE_PATH="/etc/systemd/system/gpu-clock-pin.service"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Running as root. This is fine for containerized environments."
else
   echo "This script requires root privileges. Please run with sudo."
   exit 1
fi
mkdir -p "${THIRD_PARTY_DIR}"

lock_gpu_clocks_if_supported() {
    # Best-effort: lock SM clocks to the max supported value to reduce run-to-run noise.
    # Skips silently if not supported or if nvidia-smi is unavailable.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    local gpu_name sm_max mem_max
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r')"
    sm_max="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null | head -n 1 | awk '{print $1}')"
    mem_max="$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader 2>/dev/null | head -n 1 | awk '{print $1}')"

    if [ -z "${sm_max}" ]; then
        echo "Skipping clock lock: sm_max not available from nvidia-smi."
        return
    fi

    echo "Attempting to lock SM clocks for ${gpu_name:-GPU} to ${sm_max} MHz (best-effort; may require admin privileges)..."
    if nvidia-smi -lgc "${sm_max},${sm_max}" >/dev/null 2>&1; then
        echo "  Locked SM clocks to ${sm_max} MHz."
    else
        echo "  SM clock lock not supported on this GPU/driver; continuing without lock."
    fi

    if [ -n "${mem_max}" ]; then
        if nvidia-smi --lock-memory-clocks="${mem_max},${mem_max}" >/dev/null 2>&1; then
            echo "  Locked memory clocks to ${mem_max} MHz."
        elif nvidia-smi --lock-memory-clocks="${mem_max}" >/dev/null 2>&1; then
            echo "  Locked memory clocks to ${mem_max} MHz (single-value interface)."
        elif nvidia-smi --lock-memory-clocks-deferred="${mem_max}" >/dev/null 2>&1; then
            echo "  Deferred lock of memory clocks to ${mem_max} MHz (takes effect after driver reload)."
        else
            echo "  Memory clock lock not supported here; continuing without lock."
        fi
    fi
}

install_gpu_clock_service() {
    # Install a systemd unit to reapply clock locks on boot.
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl not available; skipping GPU clock service install."
        return
    fi

    cat > "${GPU_CLOCK_SERVICE_PATH}" <<'EOF'
[Unit]
Description=Pin GPU clocks for stability
After=multi-user.target
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -lc '\
  sm=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader | head -n1 | awk "{print \$1}") || exit 0; \
  mem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader | head -n1 | awk "{print \$1}") || exit 0; \
  nvidia-smi -pm 1 || true; \
  nvidia-smi -lgc $${sm},$${sm} || true; \
  nvidia-smi -ac $${mem},$${sm} || true; \
  nvidia-smi --lock-memory-clocks-deferred=$${mem} || true; \
'

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload || true
    systemctl enable --now gpu-clock-pin.service || true
}

apply_deferred_memory_lock_now() {
    # Try to apply a deferred memory clock lock without a reboot.
    # This is best-effort and will fail if GPU reset isn't supported or processes are running.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    if nvidia-smi --gpu-reset >/dev/null 2>&1; then
        echo "Applied deferred memory clock lock via nvidia-smi --gpu-reset."
    else
        echo "INFO: Could not trigger deferred memory lock without reboot; it will apply on next driver reload/boot."
    fi
}

# Lock GPU clocks (best-effort) to reduce perf variance; safe to skip if unsupported.
lock_gpu_clocks_if_supported
install_gpu_clock_service
apply_deferred_memory_lock_now

pip_cmd() {
    python3 -m pip "$@"
}

pip_install() {
    pip_cmd install "$@"
}

pip_wheel() {
    pip_cmd wheel "$@"
}

pip_show() {
    pip_cmd show "$@"
}

# The floor is the newest cmake_minimum_required in tracked CMakeLists.txt files
# outside third_party. The pin check proves the pip CMake wins the PATH lookup.
require_cmake_toolchain() {
    local pinned_version="$1"
    local cmake_bin
    if ! cmake_bin="$(command -v cmake)"; then
        echo "ERROR: cmake is not on PATH after installing cmake==${pinned_version}." >&2
        exit 1
    fi
    if ! AISP_CMAKE_BIN="${cmake_bin}" \
        AISP_CMAKE_PINNED_VERSION="${pinned_version}" \
        AISP_REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
        python3 - <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

root = Path(os.environ["AISP_REPOSITORY_ROOT"])
cmake_bin = os.environ["AISP_CMAKE_BIN"]
pinned = os.environ["AISP_CMAKE_PINNED_VERSION"]


def version_key(text):
    return tuple(int(part) for part in text.split("."))


listing = subprocess.run(
    ["git", "-C", str(root), "ls-files", "-z", "--", ":(glob)**/CMakeLists.txt"],
    capture_output=True, text=True, check=False,
)
if listing.returncode != 0:
    sys.exit(
        f"ERROR: Cannot list tracked CMakeLists.txt files in {root}: {listing.stderr.strip()}\n"
        "       Run setup.sh from a git checkout so it can find the first-party CMake floor."
    )
declaration = re.compile(
    r"^\s*cmake_minimum_required\s*\(\s*VERSION\s+([0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE | re.MULTILINE,
)
floor = None
for relative in filter(None, listing.stdout.split("\0")):
    if "third_party" in PurePosixPath(relative).parts:
        continue
    for match in declaration.finditer((root / relative).read_text(encoding="utf-8")):
        if floor is None or version_key(match.group(1)) > version_key(floor[0]):
            floor = (match.group(1), relative)
if floor is None:
    sys.exit(f"ERROR: No first-party cmake_minimum_required declaration found in {root}.")
required, source = floor

probe = subprocess.run([cmake_bin, "--version"], capture_output=True, text=True, check=False)
found = re.match(r"cmake version ([0-9]+(?:\.[0-9]+)*)", probe.stdout)
if probe.returncode != 0 or found is None:
    sys.exit(f"ERROR: Cannot read the version of {cmake_bin}: {(probe.stdout + probe.stderr).strip()}")
installed = found.group(1)
if version_key(installed) < version_key(required):
    sys.exit(
        f"ERROR: {cmake_bin} is CMake {installed}, but {source} requires "
        f"cmake_minimum_required {required}."
    )
if version_key(installed)[:3] != version_key(pinned)[:3]:
    sys.exit(
        f"ERROR: {cmake_bin} is CMake {installed}, not the cmake=={pinned} pin from "
        "requirements_latest.txt. Put the pinned CMake first on PATH."
    )
print(f"CMake {installed} at {cmake_bin} meets cmake_minimum_required {required} ({source}).")
PY
    then
        exit 1
    fi
}

# core/profiling/ncu_torchrun_capture.py accepts only its qualified NCU version,
# and the harness runs nsys and ncu by name, so PATH must resolve the pinned copies.
require_pinned_nsight_tools() {
    local nsys_bin ncu_bin nsys_output ncu_output
    local nsys_resolved ncu_resolved expected_nsys_resolved expected_ncu_resolved
    if ! nsys_bin="$(command -v nsys)" || ! ncu_bin="$(command -v ncu)"; then
        echo "ERROR: nsys and ncu must be on PATH after installing Nsight Systems ${NSYS_VERSION} and Nsight Compute ${NCU_VERSION}." >&2
        exit 1
    fi
    if ! nsys_resolved="$(realpath "${nsys_bin}")" \
        || ! expected_nsys_resolved="$(realpath "${NSYS_BIN}")" \
        || [ "${nsys_resolved}" != "${expected_nsys_resolved}" ]; then
        echo "ERROR: ${nsys_bin} does not resolve to the pinned Nsight Systems binary ${NSYS_BIN}." >&2
        exit 1
    fi
    if ! ncu_resolved="$(realpath "${ncu_bin}")" \
        || ! expected_ncu_resolved="$(realpath "${NCU_BIN}")" \
        || [ "${ncu_resolved}" != "${expected_ncu_resolved}" ]; then
        echo "ERROR: ${ncu_bin} does not resolve to the pinned Nsight Compute binary ${NCU_BIN}." >&2
        exit 1
    fi
    nsys_output="$("${nsys_bin}" --version 2>&1 | head -n 1 || true)"
    ncu_output="$("${ncu_bin}" --version 2>&1 | grep -m 1 '^Version' || true)"
    if [[ "${nsys_output}" != *" ${NSYS_VERSION}"* ]]; then
        echo "ERROR: ${nsys_bin} reports '${nsys_output}', not the pinned Nsight Systems ${NSYS_VERSION}." >&2
        exit 1
    fi
    if [[ "${ncu_output}" != "Version ${NCU_VERSION}"* ]]; then
        echo "ERROR: ${ncu_bin} reports '${ncu_output}', not the pinned Nsight Compute ${NCU_VERSION}." >&2
        exit 1
    fi
    echo "Nsight Systems ${NSYS_VERSION} (${nsys_bin}) and Nsight Compute ${NCU_VERSION} (${ncu_bin}) are first on PATH."
}

# Ensure a tool is reachable by adding a symlink in /usr/local/bin if found elsewhere.
ensure_tool_on_path() {
    local tool_name="$1"
    shift
    local patterns=("$@")
    local nullglob_state
    nullglob_state="$(shopt -p nullglob || true)"
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for candidate in $pattern; do
            local target="$candidate"
            if [ -d "$target" ] && [ -x "${target}/${tool_name}" ]; then
                target="${target}/${tool_name}"
            fi
            if [ -x "$target" ]; then
                ln -sf "$target" "/usr/local/bin/${tool_name}"
                echo "Ensured ${tool_name} is on PATH via /usr/local/bin/${tool_name} -> ${target}"
                if [ -n "$nullglob_state" ]; then
                    eval "$nullglob_state"
                else
                    shopt -u nullglob
                fi
                return 0
            fi
        done
    done
    if [ -n "$nullglob_state" ]; then
        eval "$nullglob_state"
    else
        shopt -u nullglob
    fi
    return 1
}

# Verify CUDA availability without changing the selected runtime
verify_pytorch_cuda() {
    local context="$1"
    if ! python3 <<'PY'; then
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: PyTorch CUDA not available")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    sys.exit(1)
PY
        echo "CRITICAL: PyTorch CUDA was not available during ${context}!"
        return 1
    fi
    return 0
}

install_aisp_cli_wrapper() {
    local target="/usr/local/bin/aisp"
    cat > "${target}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${AISP_VENV_DIR}/bin/python" "${PROJECT_ROOT}/cli/aisp.py" "\$@"
EOF
    chmod 755 "${target}"
    echo "Installed aisp CLI wrapper at ${target}"
}

ensure_codex_cli() {
    if command -v codex >/dev/null 2>&1; then
        CODEX_BIN="$(command -v codex)"
        export CODEX_BIN
        echo "Codex CLI detected at ${CODEX_BIN}"
        return 0
    fi

    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
        local user_home
        user_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
        if [ -n "${user_home}" ] && [ -x "${user_home}/.local/bin/codex" ]; then
            CODEX_BIN="${user_home}/.local/bin/codex"
            export CODEX_BIN
            echo "Codex CLI detected at ${CODEX_BIN}"
            return 0
        fi
    fi

    echo "ERROR: codex CLI not found. Install it separately before running MCP setup." >&2
    exit 1
}

verify_fp8_functionality() {
    python3 <<'PY'
import torch
status = {
    "torchao": {"ok": False, "error": ""},
    "transformer_engine": {"ok": False, "error": ""},
}

def torchao_fp8_check():
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except Exception as exc:
        return False, f"torchao import failed: {exc}"
    try:
        model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False)).cuda().half()
        model = convert_to_float8_training(model, config=Float8LinearConfig())
        x = torch.randn(32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        y = model(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)

def te_fp8_check():
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:
        return False, f"Transformer Engine import failed: {exc}"
    try:
        layer = te.Linear(128, 128, bias=False).to(torch.bfloat16).cuda()
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with te.fp8_autocast(enabled=True):
            y = layer(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)

status["torchao"]["ok"], status["torchao"]["error"] = torchao_fp8_check()
status["transformer_engine"]["ok"], status["transformer_engine"]["error"] = te_fp8_check()

for name, result in status.items():
    if result["ok"] is True:
        print(f"[setup] ✓ {name} FP8 runtime check passed")
    else:
        print(f"[setup] ERROR: {name} FP8 runtime check failed: {result['error']}")

if not all(entry["ok"] for entry in status.values()):
    raise SystemExit(1)
PY
}

echo ""
echo "Configuring inotify watch limit for large workspaces..."
TARGET_INOTIFY_WATCHES=524288
CURRENT_INOTIFY_WATCHES=0
if [ -r /proc/sys/fs/inotify/max_user_watches ]; then
    CURRENT_INOTIFY_WATCHES=$(cat /proc/sys/fs/inotify/max_user_watches)
fi

if [ "$CURRENT_INOTIFY_WATCHES" -lt "$TARGET_INOTIFY_WATCHES" ]; then
    if grep -q '^fs.inotify.max_user_watches' /etc/sysctl.conf 2>/dev/null; then
        sed -i "s/^fs\.inotify\.max_user_watches=.*/fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES}/" /etc/sysctl.conf
    else
        echo "fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES}" >> /etc/sysctl.conf
    fi

    if sysctl -w fs.inotify.max_user_watches="${TARGET_INOTIFY_WATCHES}" >/dev/null 2>&1; then
        echo "Set fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES} (consumes up to ~540 MiB if fully utilized)."
    else
        echo "Warning: Failed to apply inotify watch limit via sysctl; please verify manually."
    fi
else
    echo "fs.inotify.max_user_watches already set to ${CURRENT_INOTIFY_WATCHES}."
fi

# Display the GPU and validate its installed driver. Compute capability was
# already gated before repository checkout and every host mutation.
echo ""
echo "Checking the validated NVIDIA GPU driver..."
nvidia-smi
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [[ -n "$DRIVER_VERSION" ]]; then
    if ! dpkg --compare-versions "$DRIVER_VERSION" ge "$REQUIRED_DRIVER_VERSION"; then
        echo "Current NVIDIA driver: $DRIVER_VERSION"
        echo "CUDA ${CUDA_SHORT_VERSION} Update 2 requires driver ${REQUIRED_DRIVER_VERSION}+."
        echo "This script will upgrade it automatically."
    else
        echo "NVIDIA driver version: $DRIVER_VERSION (compatible with CUDA ${CUDA_SHORT_VERSION} Update 2)"
    fi
fi

# Ensure open kernel modules are enabled for Blackwell GPUs
MODPROBE_CONF="/etc/modprobe.d/nvidia-open.conf"
if [[ ! -f "$MODPROBE_CONF" ]] || ! grep -q "NVreg_OpenRmEnableUnsupportedGpus=1" "$MODPROBE_CONF"; then
    echo "Configuring NVIDIA open kernel modules for Blackwell GPUs..."
    cat <<'EOF' > "$MODPROBE_CONF"
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0
EOF
    update-initramfs -u
    if lsmod | grep -q "^nvidia"; then
        echo "Reloading NVIDIA kernel modules to enable profiling counters..."
        systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
        for module in nvidia_uvm nvidia_peermem nvidia_modeset nvidia_drm nvidia; do
            if lsmod | grep -q "^${module}"; then
                modprobe -r "${module}" >/dev/null 2>&1 || true
            fi
        done
        modprobe nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0 >/dev/null 2>&1 || true
        for module in nvidia_modeset nvidia_uvm nvidia_peermem; do
            modprobe "${module}" >/dev/null 2>&1 || true
        done
        systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
fi

# Update system packages
echo ""
echo "Updating system packages..."

# Disable command-not-found APT hook if it's causing issues with Python upgrade
if [ -f /etc/apt/apt.conf.d/50command-not-found ] && ! /usr/lib/cnf-update-db 2>/dev/null; then
    echo "Disabling problematic command-not-found APT hook..."
    rm -f /etc/apt/apt.conf.d/50command-not-found
fi

# Clean up duplicate deadsnakes repository if it exists
if [ -f /etc/apt/sources.list.d/deadsnakes.list ] && [ -f /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list ]; then
    echo "Removing duplicate deadsnakes repository..."
    rm -f /etc/apt/sources.list.d/deadsnakes.list
fi

apt update || {
    echo "apt update had errors, but continuing..."
}

# Install required packages for adding repositories
apt install -y wget curl gnupg software-properties-common

# Install NVIDIA DCGM (Data Center GPU Manager) for monitoring stack
echo ""
echo "Installing NVIDIA DCGM components..."
if ! dpkg -s datacenter-gpu-manager >/dev/null 2>&1; then
    apt install -y datacenter-gpu-manager
    echo "NVIDIA DCGM installed"
else
    echo "NVIDIA DCGM already present"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^nvidia-dcgm.service'; then
    systemctl enable nvidia-dcgm >/dev/null 2>&1 || true
    systemctl restart nvidia-dcgm >/dev/null 2>&1 || systemctl start nvidia-dcgm >/dev/null 2>&1 || true
    echo "nvidia-dcgm service enabled"
else
    echo "nvidia-dcgm systemd unit not found; ensure the DCGM daemon is running on this host."
fi

# Add NVIDIA CUDA ${CUDA_SHORT_VERSION} repository
echo ""
echo "Adding NVIDIA CUDA ${CUDA_SHORT_VERSION} repository..."

# Check if CUDA repository is already configured
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && [ ! -f "/etc/apt/sources.list.d/cuda-${CUDA_REPO_DISTRO}-x86_64.list" ]; then
    CUDA_REPO_PKG="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/tmp/$CUDA_REPO_PKG" ]; then
        wget -P /tmp "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DISTRO}/x86_64/$CUDA_REPO_PKG"
    fi
    dpkg -i /tmp/$CUDA_REPO_PKG
    echo "NVIDIA CUDA repository added"
else
    echo "NVIDIA CUDA repository already configured"
fi

# Configure APT to prefer official NVIDIA packages over Lambda Labs or other repos
echo ""
echo "Configuring APT preferences for NVIDIA packages..."
cat > /etc/apt/preferences.d/nvidia-official <<'APT_PREF'
# Prefer official NVIDIA packages over Lambda Labs
Package: nvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: nvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: nvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600

# Also apply to libnvidia packages
Package: libnvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: libnvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: libnvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600
APT_PREF

apt update

# Install target Python runtime
echo ""
echo "🐍 Installing Python ${PYTHON_TARGET_VERSION}..."

# Check if target Python is already installed
if ! command -v "${PYTHON_TARGET_BIN}" &> /dev/null; then
    apt install -y software-properties-common
    
    # Check if deadsnakes PPA is already added
    if ! grep -q "deadsnakes/ppa" /etc/apt/sources.list.d/*.list 2>/dev/null; then
        add-apt-repository -y ppa:deadsnakes/ppa
    else
        echo "deadsnakes PPA already configured"
    fi
    
    apt update || true
    apt install -y "${PYTHON_TARGET_BIN}" "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv"
    echo "Python ${PYTHON_TARGET_VERSION} installed"
else
    CURRENT_TARGET_VERSION=$("${PYTHON_TARGET_BIN}" --version 2>&1 | awk '{print $2}')
    echo "Python ${PYTHON_TARGET_VERSION} already installed (version $CURRENT_TARGET_VERSION)"
    # Still ensure dev packages are present
    apt install -y "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv"
fi

# The base interpreter remains untouched. The isolated Python 3.12 runtime is
# created after any driver upgrade has completed, so a reboot does not leave a
# partially populated environment behind.

# Upgrade NVIDIA driver to required version if needed (required for CUDA ${CUDA_SHORT_VERSION} Update 2)
echo ""
echo "Checking NVIDIA driver version..."
if command -v nvidia-smi &> /dev/null; then
    CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    
    if ! dpkg --compare-versions "$CURRENT_DRIVER" ge "$REQUIRED_DRIVER_VERSION"; then
        echo "Current driver ($CURRENT_DRIVER) is too old for CUDA ${CUDA_SHORT_VERSION} Update 2"
        echo "Upgrading to NVIDIA driver 580 (open kernel modules)..."
        
        # Unhold any NVIDIA packages to avoid resolver conflicts
        mapfile -t held_nvidia_packages < <(
            apt-mark showhold 2>/dev/null | grep -E '^(nvidia|libnvidia|cuda)-' || true
        )
        if [ "${#held_nvidia_packages[@]}" -gt 0 ]; then
            echo "Releasing held NVIDIA/CUDA packages..."
            apt-mark unhold "${held_nvidia_packages[@]}" || true
        fi

        # Remove old driver packages that might conflict
        echo "Removing old NVIDIA driver packages..."
        apt remove -y nvidia-driver-* nvidia-dkms-* nvidia-kernel-common nvidia-kernel-common-* \
            libnvidia-compute-* libnvidia-extra-* nvidia-utils-* \
            python3-jax-cuda* python3-torch-cuda python3-torchvision-cuda 2>/dev/null || true
        apt autoremove -y
        
        # Install new driver
        echo "Installing NVIDIA driver 580..."
        if apt install -y nvidia-driver-580-open; then
            echo "NVIDIA driver 580 installed successfully"
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║  REBOOT REQUIRED                                       ║"
            echo "║                                                            ║"
            echo "║  The NVIDIA driver has been upgraded to version 580.      ║"
            echo "║  Please reboot your system and re-run this script.        ║"
            echo "║                                                            ║"
            echo "║  After reboot:                                             ║"
            echo "║    1. Run: nvidia-smi                                      ║"
            echo "║    2. Verify driver version is ${REQUIRED_DRIVER_VERSION}+                 ║"
            echo "║    3. Re-run: sudo ./setup.sh                              ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            exit 0
        else
            echo "Failed to install NVIDIA driver 580"
            echo "Please manually install the driver and reboot before continuing"
            exit 1
        fi
    else
        echo "NVIDIA driver $CURRENT_DRIVER is compatible with CUDA ${CUDA_SHORT_VERSION} Update 2"
    fi
fi

# Install CUDA toolchain (13.0.2 Update 2)
echo ""
echo "Installing CUDA ${CUDA_FULL_VERSION} toolchain..."
if ! apt install -y "cuda-toolkit-13-0=13.0.2-1"; then
    echo "ERROR: cuda-toolkit-13-0=13.0.2-1 is unavailable. Refusing to build with a different toolkit." >&2
    exit 1
fi
INSTALLED_CUDA_TOOLKIT_VERSION="$(dpkg-query -W -f='${Version}' cuda-toolkit-13-0 2>/dev/null || true)"
if [ "${INSTALLED_CUDA_TOOLKIT_VERSION}" != "13.0.2-1" ]; then
    echo "ERROR: cuda-toolkit-13-0 resolved to ${INSTALLED_CUDA_TOOLKIT_VERSION:-missing}, expected 13.0.2-1." >&2
    exit 1
fi

# NCCL, cuDNN, and NVSHMEM are part of the hash-locked Python graph. Their
# wheel libraries and headers are selected after that graph is installed. Do
# not replace host packages or let an independently managed system copy define
# this runtime.

# Install TensorRT runtime libraries for standalone TensorRT workflows.
echo ""
echo "Installing TensorRT runtime libraries (CUDA 13)..."
apt install -y libnvinfer10 libnvinfer-plugin10 libnvinfer-vc-plugin10 libnvonnxparsers10

# TensorRT-LLM runtime helpers (MPI + audio deps)
echo ""
echo "Installing TensorRT-LLM system dependencies (OpenMPI, libsndfile)..."
apt install -y openmpi-bin libsndfile1

# Install GPUDirect Storage (GDS) for high-performance I/O
echo ""
echo "Installing GPUDirect Storage (GDS)..."
if ! dpkg -s gds-tools-13-0 >/dev/null 2>&1; then
    apt install -y gds-tools-13-0
    echo "GDS tools installed"
else
    echo "GDS tools already installed"
fi

# Ensure the nvidia-fs kernel module package is present so modprobe succeeds
if ! dpkg -s nvidia-fs >/dev/null 2>&1 && ! dpkg -s nvidia-fs-dkms >/dev/null 2>&1; then
    echo "Installing nvidia-fs-dkms (GDS kernel module)..."
    apt install -y nvidia-fs-dkms
else
    echo "nvidia-fs package already installed"
fi

# Load nvidia-fs kernel module for GDS
echo "Loading nvidia-fs kernel module..."
if ! lsmod | grep -q nvidia_fs; then
    if modprobe nvidia-fs 2>/dev/null; then
        echo "nvidia-fs module loaded"
    else
        echo "Could not load nvidia-fs module (requires root)"
        echo "   Load it manually with: sudo modprobe nvidia-fs"
        echo "   Or run: sudo core/scripts/setup/load_gds_module.sh"
    fi
else
    echo "nvidia-fs module already loaded"
fi

# Persist nvidia-fs so it loads on boot (avoids manual modprobe after reboot)
echo "Configuring nvidia-fs to load on boot..."
modules_load_conf="/etc/modules-load.d/nvidia-fs.conf"
if [ -f "${modules_load_conf}" ]; then
    if grep -Fxq "nvidia-fs" "${modules_load_conf}"; then
        echo "nvidia-fs already listed in ${modules_load_conf}"
    else
        echo "nvidia-fs" >> "${modules_load_conf}"
        echo "Added nvidia-fs to ${modules_load_conf}"
    fi
else
    cat <<'EOF' > "${modules_load_conf}"
# Load NVIDIA GPUDirect Storage kernel module at boot
nvidia-fs
EOF
    echo "Created ${modules_load_conf} for boot-time nvidia-fs loading"
fi
echo "GDS requires a supported storage backend (NVMe, BeeGFS, NFS, WekaFS, etc.). Configure storage and rerun /usr/local/cuda/gds/tools/gdscheck -p to confirm."

# Ensure NVMe driver is loaded for GDS
echo "Checking NVMe kernel modules..."
modprobe nvme-core >/dev/null 2>&1 || true
modprobe nvme >/dev/null 2>&1 || true
if lsblk -d -o NAME,TRAN 2>/dev/null | grep -qi nvme; then
    echo "NVMe devices detected (required for optimal GDS)."
else
    echo "WARNING: No NVMe devices detected; GDS performance may be limited until NVMe storage is available."
fi

# The hash-bound Python graph supplies KvikIO for the GDS examples.

# Install the CUDA 13.0 command-line tools from the same pinned toolkit release.
echo ""
echo "Installing CUDA sanitizers and debugging tools..."
if ! apt install -y "cuda-command-line-tools-13-0=13.0.2-1"; then
    echo "ERROR: cuda-command-line-tools-13-0=13.0.2-1 is unavailable." >&2
    exit 1
fi
INSTALLED_CUDA_TOOLS_VERSION="$(dpkg-query -W -f='${Version}' cuda-command-line-tools-13-0 2>/dev/null || true)"
if [ "${INSTALLED_CUDA_TOOLS_VERSION}" != "13.0.2-1" ]; then
    echo "ERROR: cuda-command-line-tools-13-0 resolved to ${INSTALLED_CUDA_TOOLS_VERSION:-missing}, expected 13.0.2-1." >&2
    exit 1
fi
if ! command -v compute-sanitizer &> /dev/null; then
    echo "ERROR: compute-sanitizer is missing after installing the pinned CUDA command-line tools." >&2
    exit 1
fi
echo "CUDA command-line tools 13.0.2 installed"

# Install pinned NVIDIA Nsight Systems and Compute
echo ""
echo "Installing pinned NVIDIA Nsight Systems and Compute..."

# Create temporary directory for downloads
TEMP_DIR="$(mktemp -d /tmp/aisp_nsight_install.XXXXXX)"
cd "$TEMP_DIR"

# Add the signed NVIDIA DEVTOOLS repository for the CLI-only Nsight Systems
# package. Keep this key scoped to that repository and reject key drift before
# changing APT configuration.
NSYS_VERSION="2026.5.1"
NSYS_PACKAGE="nsight-systems-cli-${NSYS_VERSION}"
NSYS_PACKAGE_VERSION="2026.5.1.161-265138896106v0"
DEVTOOLS_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
DEVTOOLS_KEY_SHA256="47217c49dcb9e47a8728b354450f694c9898cd4a126173044a69b1e9ac0fba96"
DEVTOOLS_KEYRING_SHA256="0d4c5cbe167a020fc2978ea28eaea0e797f89fdee696488e93bf446dd868486f"
DEVTOOLS_KEY_FINGERPRINT="AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80"
DEVTOOLS_KEYRING="/usr/share/keyrings/nvidia-devtools-keyring.gpg"
DEVTOOLS_SOURCE_LIST="/etc/apt/sources.list.d/nvidia-devtools.list"
DEVTOOLS_REPO_URL="https://developer.download.nvidia.com/devtools/repos/${CUDA_REPO_DISTRO}/amd64"
DEVTOOLS_GNUPG_HOME="${TEMP_DIR}/gnupg"
DEVTOOLS_RAW_KEY="${TEMP_DIR}/7fa2af80.pub"
DEVTOOLS_KEYRING_TEMP="${TEMP_DIR}/nvidia-devtools-keyring.gpg"
install -d -m 0700 "${DEVTOOLS_GNUPG_HOME}"
curl --disable --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    --output "${DEVTOOLS_RAW_KEY}" "${DEVTOOLS_KEY_URL}"
if ! printf '%s  %s\n' "${DEVTOOLS_KEY_SHA256}" "${DEVTOOLS_RAW_KEY}" | sha256sum --check --status; then
    echo "ERROR: NVIDIA DEVTOOLS signing key SHA-256 mismatch." >&2
    exit 1
fi
mapfile -t DEVTOOLS_KEY_FINGERPRINTS < <(
    gpg --no-options --homedir "${DEVTOOLS_GNUPG_HOME}" --batch --show-keys \
        --with-colons --fingerprint "${DEVTOOLS_RAW_KEY}" \
        | awk -F: '$1 == "fpr" { print $10 }'
)
if [ "${#DEVTOOLS_KEY_FINGERPRINTS[@]}" -ne 1 ] || [ "${DEVTOOLS_KEY_FINGERPRINTS[0]}" != "${DEVTOOLS_KEY_FINGERPRINT}" ]; then
    echo "ERROR: NVIDIA DEVTOOLS signing key fingerprint mismatch." >&2
    exit 1
fi
gpg --no-options --homedir "${DEVTOOLS_GNUPG_HOME}" --batch --yes --dearmor \
    --output "${DEVTOOLS_KEYRING_TEMP}" "${DEVTOOLS_RAW_KEY}"
if ! printf '%s  %s\n' "${DEVTOOLS_KEYRING_SHA256}" "${DEVTOOLS_KEYRING_TEMP}" | sha256sum --check --status; then
    echo "ERROR: NVIDIA DEVTOOLS dearmored key SHA-256 mismatch." >&2
    exit 1
fi
install -m 0644 "${DEVTOOLS_KEYRING_TEMP}" "${DEVTOOLS_KEYRING}"
printf 'deb [arch=amd64 signed-by=%s] %s/ /\n' \
    "${DEVTOOLS_KEYRING}" "${DEVTOOLS_REPO_URL}" > "${TEMP_DIR}/nvidia-devtools.list"
install -m 0644 "${TEMP_DIR}/nvidia-devtools.list" "${DEVTOOLS_SOURCE_LIST}"

# Refresh package metadata only after both signed NVIDIA repositories are ready.
apt update -qq

# Install the exact CLI-only Nsight Systems build and select its concrete target
# executable instead of the full GUI package or a version-selector link.
echo "Installing Nsight Systems CLI (pinned ${NSYS_PACKAGE}=${NSYS_PACKAGE_VERSION})..."
if ! apt install -y --upgrade "${NSYS_PACKAGE}=${NSYS_PACKAGE_VERSION}"; then
    echo "ERROR: ${NSYS_PACKAGE}=${NSYS_PACKAGE_VERSION} is unavailable from ${DEVTOOLS_REPO_URL}." >&2
    exit 1
fi
INSTALLED_NSYS_PACKAGE_VERSION="$(dpkg-query -W -f='${Version}' "${NSYS_PACKAGE}" 2>/dev/null || true)"
if [ "${INSTALLED_NSYS_PACKAGE_VERSION}" != "${NSYS_PACKAGE_VERSION}" ]; then
    echo "ERROR: ${NSYS_PACKAGE} resolved to ${INSTALLED_NSYS_PACKAGE_VERSION:-missing}, expected ${NSYS_PACKAGE_VERSION}." >&2
    exit 1
fi
NSYS_BIN="/opt/nvidia/nsight-systems-cli/${NSYS_VERSION}/target-linux-x64/nsys"
if [[ ! -x "${NSYS_BIN}" ]]; then
    echo "ERROR: Nsight Systems CLI ${NSYS_VERSION} binary not found at ${NSYS_BIN}." >&2
    exit 1
fi
update-alternatives --install /usr/local/bin/nsys nsys "$NSYS_BIN" 50
update-alternatives --set nsys "$NSYS_BIN"
echo "Nsight Systems CLI pinned to ${NSYS_PACKAGE_VERSION} (${NSYS_BIN})"

# Install Nsight Compute and set binary alternative
NCU_VERSION="2026.2.1"
echo "Installing Nsight Compute (pinned ${NCU_VERSION})..."
apt install -y --upgrade "nsight-compute-${NCU_VERSION}"
NCU_BIN="/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu"
if [[ ! -x "$NCU_BIN" ]]; then
    echo "ERROR: Nsight Compute ${NCU_VERSION} binary not found at ${NCU_BIN}." >&2
    exit 1
fi
update-alternatives --install /usr/local/bin/ncu ncu "$NCU_BIN" 50
update-alternatives --set ncu "$NCU_BIN"
echo "Nsight Compute pinned to ${NCU_VERSION} (${NCU_BIN})"

# The CUDA 13.0 toolkit packages also install Nsight Compute 2025.3.1 and Nsight
# Systems 2025.3.2. Every PATH entry that adds ${CUDA_HOME_DIR}/bin must put the
# pinned Nsight directories ahead of it.
NSIGHT_BIN_PATH="$(dirname "${NSYS_BIN}"):$(dirname "${NCU_BIN}")"

# Create the runtime only after any driver-triggered reboot exit. The path was
# rejected before setup made host changes.
"${PYTHON_TARGET_BIN}" -m venv "${AISP_VENV_DIR}"
export VIRTUAL_ENV="${AISP_VENV_DIR}"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR="${AISP_VENV_DIR}/torch_extensions"
export TRITON_CACHE_DIR="${AISP_VENV_DIR}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${AISP_VENV_DIR}/inductor_cache"
export PATH="${AISP_VENV_DIR}/bin:${PATH}"
echo "Python runtime: ${AISP_VENV_DIR}"

# Configure PATH and LD_LIBRARY_PATH for CUDA environment
echo ""
echo "Configuring CUDA ${CUDA_SHORT_VERSION} environment..."
# Keep the runtime paths local to this environment. Existing shells and
# independently pinned benchmark environments retain their configuration.
CUDA_PROFILE_SCRIPT="${AISP_VENV_DIR}/cuda-env.sh"
cat > "${CUDA_PROFILE_SCRIPT}" <<PROFILE_EOF
export CUDA_HOME="${CUDA_HOME_DIR}"
export CUDA_PATH="${CUDA_HOME_DIR}"
export PATH="${AISP_VENV_DIR}/bin:${NSIGHT_BIN_PATH}:${CUDA_HOME_DIR}/bin:\$PATH"
export AISP_CUDNN_RUNTIME_POLICY=torch
PROFILE_EOF

# Persist CUDA toolchain settings for the remainder of the script
export CUDA_HOME="${CUDA_HOME_DIR}"
export CUDA_PATH="${CUDA_HOME_DIR}"
export PATH="${NSIGHT_BIN_PATH}:$CUDA_HOME/bin:$PATH"
export AISP_CUDNN_RUNTIME_POLICY=torch
require_pinned_nsight_tools

export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"

echo ""
echo "Verifying CUDA toolchain after installation..."
if ! nvcc --version >/tmp/nvcc_version.txt 2>&1; then
    cat /tmp/nvcc_version.txt
    echo "nvcc not available even after CUDA install. Aborting."
    exit 1
fi
cat /tmp/nvcc_version.txt
rm -f /tmp/nvcc_version.txt

# CUDA-dependent Python dependencies are installed from the same locked graph.

# Clean up
cd /
rm -rf "$TEMP_DIR"
cd "$PROJECT_ROOT"

# Install system tools for performance testing
echo ""
echo "Installing system performance tools..."
KERNEL_RELEASE=$(uname -r)

apt install -y \
    numactl \
    linux-tools-common \
    linux-tools-generic \
    "linux-tools-${KERNEL_RELEASE}" \
    gdb \
    ffmpeg \
    libdw-dev \
    libelf-dev \
    perf-tools-unstable \
    infiniband-diags \
    perftest \
    htop \
    iotop \
    libjemalloc2 \
    libtcmalloc-minimal4 \
    ripgrep \
    sysstat

echo ""
echo "Installing the isolated Rust 1.95.0 toolchain..."
export RUSTUP_HOME="${THIRD_PARTY_DIR}/rustup-torch2141"
export CARGO_HOME="${THIRD_PARTY_DIR}/cargo-torch2141"
export PATH="${CARGO_HOME}/bin:${PATH}"
if [ ! -x "${CARGO_HOME}/bin/rustup" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --no-modify-path --default-toolchain 1.95.0
fi
rustup toolchain install 1.95.0
export RUSTUP_TOOLCHAIN=1.95.0

echo ""
# The dependency graph below supplies CMake, Ninja and Python tooling.

# Existing system/user torch installations belong to other environments.
# The dedicated runtime receives the full pinned graph below.

echo ""
echo "The hash-bound graph includes the complete CUDA torch runtime."
echo ""

# Install project dependencies
echo ""
echo "Installing project dependencies..."

# Single-source serving stack pins from requirements_latest.txt and the
# separately installed ABI-bound vLLM pin manifest.
# This keeps setup-time ABI checks aligned with runtime benchmark checks.
if [ ! -f "$VLLM_PIN_FILE" ]; then
    echo "ERROR: ABI-bound vLLM pin manifest is missing: $VLLM_PIN_FILE"
    exit 1
fi
VLLM_PIN_COUNT="$(grep -Evc '^[[:space:]]*(#|$)' "$VLLM_PIN_FILE" || true)"
VLLM_PIP_SPEC="$(grep -Ev '^[[:space:]]*(#|$)' "$VLLM_PIN_FILE" | head -n 1 || true)"
if [ "$VLLM_PIN_COUNT" -ne 1 ] || [[ "$VLLM_PIP_SPEC" != vllm==* ]]; then
    echo "ERROR: $VLLM_PIN_FILE must contain exactly one active vllm== pin"
    exit 1
fi
REQ_VLLM_VERSION="${VLLM_PIP_SPEC#vllm==}"
if [ ! -f "${VLLM_SOURCE_BUILD_MANIFEST}" ]; then
    echo "ERROR: vLLM source-build contract is missing: ${VLLM_SOURCE_BUILD_MANIFEST}"
    exit 1
fi
python3 - "${VLLM_SOURCE_BUILD_MANIFEST}" "${REQ_VLLM_VERSION}" <<'PYCONTRACT'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    contract = json.load(stream)
if contract["package"]["version"] != sys.argv[2]:
    raise SystemExit("vLLM source-build version differs from vllm_no_deps.pin")
PYCONTRACT

echo "Serving stack pins (from requirements_latest.txt and vllm_no_deps.pin):"
echo "  torch==${PYTORCH_TORCH_VERSION}"
echo "  torchaudio==${PYTORCH_TORCHAUDIO_VERSION}"
echo "  torchvision==${PYTORCH_TORCHVISION_VERSION}"
echo "  torchao==${PYTORCH_TORCHAO_VERSION}"
echo "  triton==${PYTORCH_TRITON_VERSION}"
echo "  ${VLLM_PIP_SPEC}"
echo "  flashinfer-python==${FLASHINFER_EXPECTED_VERSION}"

# TorchAudio 2.11 publishes +cu130 wheels for x86_64 and aarch64.

# Resolve sources after the requirements pins above have replaced the defaults.
# Each package chooses its own channel, including Triton's optional +git builds.
PYTORCH_CU130_INDEX="$(pytorch_package_index "${PYTORCH_TORCH_VERSION}" cu130)"
PYTORCH_TORCH_FIND_LINKS="${PYTORCH_TORCH_FIND_LINKS:-${PYTORCH_CU130_INDEX}/torch/}"
PYTORCH_TORCHAUDIO_INDEX="$(pytorch_package_index "${PYTORCH_TORCHAUDIO_VERSION}" cu130)"
PYTORCH_TORCHAO_INDEX="$(pytorch_package_index "${PYTORCH_TORCHAO_VERSION}" cu130)"
PYTORCH_TRITON_INDEX="$(pytorch_package_index "${PYTORCH_TRITON_VERSION}")"
echo "Resolved package indexes:"
echo "  torch: ${PYTORCH_CU130_INDEX}"
echo "  torchaudio: ${PYTORCH_TORCHAUDIO_INDEX}"
echo "  torchao: ${PYTORCH_TORCHAO_INDEX}"
echo "  triton: ${PYTORCH_TRITON_INDEX}"

# This lock targets Linux x86_64 and Python 3.12. Reject unsupported hosts.
HOST_ARCH="$(uname -m)"
if [ "${HOST_ARCH}" = "arm64" ]; then HOST_ARCH="aarch64"; fi
require_torchao_cuda_wheel_support "${HOST_ARCH}"
if [ "${HOST_ARCH}" != "x86_64" ]; then
    echo "ERROR: The shared runtime lock requires Linux x86_64. Resolve and qualify a separate platform lock."
    exit 1
fi

# Install the preflight-validated Linux/Python 3.12 graph.
if ! pip_install --require-hashes --no-deps \
    --find-links "${PROJECT_ROOT}/vendor/python-wheels" \
    --report "${AISP_VENV_DIR}/pytorch-stack-install-report.json" \
    -r "${PYTORCH_STACK_LOCK}"; then
    echo "ERROR: The locked PyTorch runtime installation failed."
    exit 1
fi

# The graph includes the exact CUDA 13 torch family. Do not uninstall or
# re-resolve those packages after the hash-checked installation.
CMAKE_PIP_BIN_DIR="$(python3 -c 'import cmake; print(cmake.CMAKE_BIN_DIR)')"
export PATH="${CMAKE_PIP_BIN_DIR}:${PATH}"
require_cmake_toolchain "${REQ_CMAKE_VERSION}"

# Select wheel runtime libraries without preloading a system NCCL build.
if [[ "${LD_PRELOAD:-}" == *libnccl* || "${LD_PRELOAD:-}" == *libcudnn* || "${LD_PRELOAD:-}" == *libcublas* ]]; then
    echo "ERROR: An external CUDA/NCCL preload would override the locked runtime. Launch setup without that preload."
    exit 1
fi
RUNTIME_WHEEL_LIB_DIRS="$(PYTHONNOUSERSITE=1 python3 - <<'PYLIBS'
from core.harness.serving_stack import configure_serving_stack_runtime_env
print(":".join(configure_serving_stack_runtime_env()))
PYLIBS
)"
if [ -z "${RUNTIME_WHEEL_LIB_DIRS}" ]; then
    echo "ERROR: The locked CUDA wheel library directories were not found." >&2
    exit 1
fi
export LD_LIBRARY_PATH="${RUNTIME_WHEEL_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export AISP_CUDNN_RUNTIME_POLICY=torch
mapfile -t CUDA_WHEEL_SDK_PATHS < <(python3 - <<'PYSDK'
from importlib import metadata
from pathlib import Path

components = (
    ("nvidia-nccl-cu13", "nccl", "nccl.h", "libnccl.so.2"),
    ("nvidia-cudnn-cu13", "cudnn", "cudnn.h", "libcudnn.so.9"),
    ("nvidia-nvshmem-cu13", "nvshmem", "nvshmem.h", "libnvshmem_host.so.3"),
)
for distribution_name, component, header_name, library_name in components:
    distribution = metadata.distribution(distribution_name)
    root = Path(distribution.locate_file(f"nvidia/{component}")).resolve()
    include = root / "include"
    library = root / "lib" / library_name
    header = include / header_name
    if not header.is_file() or not library.is_file():
        raise SystemExit(
            f"Locked {distribution_name} is missing {header.relative_to(root)} or "
            f"{library.relative_to(root)}"
        )
    print(distribution.version)
    print(root)
    print(include)
    print(library)
PYSDK
)
if [ "${#CUDA_WHEEL_SDK_PATHS[@]}" -ne 12 ]; then
    echo "ERROR: Could not resolve the locked NCCL, cuDNN, and NVSHMEM wheel assets." >&2
    exit 1
fi
NCCL_WHEEL_VERSION="${CUDA_WHEEL_SDK_PATHS[0]}"
NCCL_WHEEL_ROOT="${CUDA_WHEEL_SDK_PATHS[1]}"
NCCL_INCLUDE_DIR="${CUDA_WHEEL_SDK_PATHS[2]}"
NCCL_LIB_PATH="${CUDA_WHEEL_SDK_PATHS[3]}"
CUDNN_WHEEL_VERSION="${CUDA_WHEEL_SDK_PATHS[4]}"
CUDNN_WHEEL_ROOT="${CUDA_WHEEL_SDK_PATHS[5]}"
CUDNN_INCLUDE_DIR="${CUDA_WHEEL_SDK_PATHS[6]}"
CUDNN_LIB_PATH="${CUDA_WHEEL_SDK_PATHS[7]}"
NVSHMEM_WHEEL_VERSION="${CUDA_WHEEL_SDK_PATHS[8]}"
NVSHMEM_WHEEL_ROOT="${CUDA_WHEEL_SDK_PATHS[9]}"
NVSHMEM_INCLUDE_DIR="${CUDA_WHEEL_SDK_PATHS[10]}"
NVSHMEM_LIB_PATH="${CUDA_WHEEL_SDK_PATHS[11]}"
# Keep source-build helpers from inheriting SDK roots. They resolve build inputs
# from their pinned contracts. The activation file exports these wheel roots for
# direct extension builds after setup succeeds.
unset NCCL_HOME CUDNN_HOME NVSHMEM_HOME
export CPATH="${NCCL_INCLUDE_DIR}:${CUDNN_INCLUDE_DIR}:${NVSHMEM_INCLUDE_DIR}${CPATH:+:${CPATH}}"
export NCCL_LIB_PATH
RUNTIME_ENV_FILE="${AISP_VENV_DIR}/runtime-env.sh"
cat > "${RUNTIME_ENV_FILE}" <<ENVEOF
source "${AISP_VENV_DIR}/bin/activate"
source "${CUDA_PROFILE_SCRIPT}"
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"
export LD_LIBRARY_PATH="${RUNTIME_WHEEL_LIB_DIRS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export NCCL_HOME="${NCCL_WHEEL_ROOT}"
export CUDNN_HOME="${CUDNN_WHEEL_ROOT}"
export NVSHMEM_HOME="${NVSHMEM_WHEEL_ROOT}"
export CPATH="${NCCL_INCLUDE_DIR}:${CUDNN_INCLUDE_DIR}:${NVSHMEM_INCLUDE_DIR}\${CPATH:+:\${CPATH}}"
export NCCL_LIB_PATH="${NCCL_LIB_PATH}"
ENVEOF
echo "Activate the pinned runtime with: source ${RUNTIME_ENV_FILE}"

if ! PYTORCH_EXPECTED_VERSION="${PYTORCH_TORCH_VERSION}" python3 - <<'PY'
import sys
import torch
import os

cuda_ver = getattr(torch.version, "cuda", None) or ""
if not cuda_ver.startswith("13."):
    print(f"ERROR: Expected cu13 torch wheel, got torch.version.cuda={cuda_ver!r}")
    sys.exit(1)
expected = os.environ.get("PYTORCH_EXPECTED_VERSION", "")
if expected and torch.__version__ != expected:
    print(f"ERROR: Expected torch=={expected}, got {torch.__version__}")
    sys.exit(1)
print(f"[setup] torch cu13 confirmed: {torch.__version__} (cuda {cuda_ver})")
PY
then
    echo "ERROR: cu130 torch install did not deliver the expected pinned version."
    exit 1
fi

echo ""
echo "Verifying PyTorch CUDA installation..."
if ! python3 <<'PY'; then
import os
import sys
import torch

print("PyTorch CUDA wheel installed")
print(f"  version: {torch.__version__}")
print(f"  location: {torch.__file__}")

if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
    print("ERROR: torch.version.cuda is None - CUDA not enabled in this build")
    sys.exit(1)

print(f"  cuda: {torch.version.cuda}")
if not str(torch.version.cuda).startswith("13."):
    print(f"ERROR: Expected torch.version.cuda to start with '13.' but got {torch.version.cuda!r}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False after install")
    sys.exit(1)

arch_list = torch.cuda.get_arch_list()
print(f"  arch list: {arch_list}")
required_sm = os.environ.get("GPU_COMPUTE_SM_NUM")
if required_sm:
    target = f"sm_{required_sm}"
    if target not in arch_list:
        print(f"ERROR: Expected PyTorch arch list to include {target}, got {arch_list}")
        sys.exit(1)

print("✓ PyTorch CUDA build verified successfully")
print(f"✓ GPU count: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 0:
    print(f"✓ GPU 0: {torch.cuda.get_device_name(0)}")
PY

    echo ""
    cat <<'EOF'
CRITICAL ERROR: PyTorch CUDA verification failed!
This should not happen after installing the CUDA wheel.
Please check:
  1. NVIDIA driver is loaded: nvidia-smi
  2. CUDA libraries are accessible: echo $LD_LIBRARY_PATH
  3. PyTorch wheel was built correctly
EOF
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ PyTorch CUDA installation and verification complete"
echo "============================================================================"
echo ""

# Accelerate and TorchTitan are already installed by the hash-bound graph.
# Verify their public imports without resolving or changing that graph.
python3 <<'PYTRAINING'
import accelerate
import torchtitan
from torchtitan.config import CompileConfig, ParallelismConfig

compile_config = CompileConfig()
if not hasattr(compile_config, "enable_async_tensor_parallel"):
    raise SystemExit("TorchTitan CompileConfig lacks enable_async_tensor_parallel")
print(f"Accelerate: {accelerate.__version__}")
print(f"TorchTitan: {torchtitan.__file__}")
print(f"CompileConfig: {CompileConfig.__module__}")
print(f"ParallelismConfig: {ParallelismConfig.__module__}")
print(
    "Async tensor parallelism default: "
    f"{compile_config.enable_async_tensor_parallel}"
)
PYTRAINING

# TensorRT-LLM 1.1 was tied to the historical runtime. It is not part of this
# shared torch 2.14 stack and must use a separately qualified environment.
if [ "${INSTALL_TRTLLM:-0}" = "1" ]; then
    echo "ERROR: TensorRT-LLM needs a separately pinned runtime. It cannot be installed into this torch 2.14 environment."
    exit 1
fi

# Triton is installed by the hash-bound graph and must match the torch build.
python3 <<'PYTRITON'
from importlib import metadata
import os
import triton

expected = "3.8.0"
if metadata.version("triton") != expected:
    raise SystemExit(f"Expected Triton {expected}, got {triton.__version__}")
print(f"Triton: {triton.__version__}")
PYTRITON

# Verify the matched torchvision wheel did not replace the pinned CUDA torch.
if ! verify_pytorch_cuda "torchvision installation"; then
    echo "ERROR: PyTorch CUDA missing after torchvision!"
    exit 1
fi

echo ""
echo "Resolving immutable CUTLASS source (${CUTLASS_REF}) for local CUDA builds..."
validate_cutlass_source() {
    local provenance
    if ! provenance="$(
        CUTLASS_SRC_DIR="${CUTLASS_SRC_DIR}" \
        CUTLASS_REPO_URL="${CUTLASS_REPO_URL}" \
        CUTLASS_REF="${CUTLASS_REF}" \
        CUTLASS_RELEASE_TAG="${CUTLASS_RELEASE_TAG}" \
        CUTLASS_TARGET_VERSION="${CUTLASS_TARGET_VERSION}" \
        python3 <<'PYCUTLASS'
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(os.environ["CUTLASS_SRC_DIR"])
receipt_path = root / ".git" / "aisp-source.json"
version_path = root / "include" / "cutlass" / "version.h"
if not receipt_path.is_file() or not version_path.is_file():
    raise SystemExit("CUTLASS source lacks its Git provenance or version header")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


commit = git("rev-parse", "HEAD")
tree = git("rev-parse", "HEAD^{tree}")
origin = git("remote", "get-url", "origin")
dirty = git("status", "--porcelain", "--untracked-files=all")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
text = version_path.read_text(encoding="utf-8")


def value(name: str) -> str:
    match = re.search(rf"^#define\s+{name}\s+([0-9]+)\s*$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"CUTLASS version header is missing {name}")
    return match.group(1)


version = ".".join(
    value(name) for name in ("CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH")
)
expected = {
    "schema_version": 1,
    "repository": os.environ["CUTLASS_REPO_URL"],
    "requested_commit": os.environ["CUTLASS_REF"],
    "resolved_commit": commit,
    "tree": tree,
    "release_tag": os.environ["CUTLASS_RELEASE_TAG"],
    "source_version": version,
}
if commit != os.environ["CUTLASS_REF"]:
    raise SystemExit(f"CUTLASS commit is {commit}, expected {os.environ['CUTLASS_REF']}")
if origin != os.environ["CUTLASS_REPO_URL"]:
    raise SystemExit(f"CUTLASS origin is {origin}, expected {os.environ['CUTLASS_REPO_URL']}")
if dirty:
    raise SystemExit("CUTLASS source has tracked or untracked changes")
if version != os.environ["CUTLASS_TARGET_VERSION"]:
    raise SystemExit(
        f"CUTLASS source version is {version}, expected {os.environ['CUTLASS_TARGET_VERSION']}"
    )
if receipt != expected:
    raise SystemExit("CUTLASS source receipt does not match the checked-out tree")
print(f"{commit}\t{tree}\t{version}")
PYCUTLASS
    )"; then
        return 1
    fi
    IFS=$'\t' read -r CUTLASS_RESOLVED_COMMIT CUTLASS_RESOLVED_TREE CURRENT_CUTLASS_VERSION <<< "${provenance}"
    [ -n "${CUTLASS_RESOLVED_COMMIT}" ] && \
        [ -n "${CUTLASS_RESOLVED_TREE}" ] && \
        [ "${CURRENT_CUTLASS_VERSION}" = "${CUTLASS_TARGET_VERSION}" ]
}

if ! validate_cutlass_source; then
    if [ -d "${CUTLASS_SRC_DIR}/.git" ] && \
        [ -n "$(git -C "${CUTLASS_SRC_DIR}" status --porcelain --untracked-files=all 2>/dev/null)" ]; then
        echo "ERROR: Existing CUTLASS source has local changes: ${CUTLASS_SRC_DIR}" >&2
        echo "       Preserve it and choose a fresh CUTLASS_SRC_DIR." >&2
        exit 1
    fi
    echo "Installing a fresh verified CUTLASS source tree..."
    CUTLASS_REPO="${CUTLASS_REPO_URL}" \
        CUTLASS_REF="${CUTLASS_REF}" \
        CUTLASS_RELEASE_TAG="${CUTLASS_RELEASE_TAG}" \
        CUTLASS_EXPECTED_VERSION="${CUTLASS_TARGET_VERSION}" \
        CUTLASS_SRC_DIR="${CUTLASS_SRC_DIR}" \
        "${PROJECT_ROOT}/core/scripts/install_cutlass.sh"
    if ! validate_cutlass_source; then
        echo "ERROR: Installed CUTLASS source failed immutable provenance validation." >&2
        exit 1
    fi
fi
echo "  CUTLASS ${CURRENT_CUTLASS_VERSION} commit ${CUTLASS_RESOLVED_COMMIT}"
echo "  CUTLASS tree ${CUTLASS_RESOLVED_TREE}"
echo "  CUTLASS headers available at ${CUTLASS_SRC_DIR}/include"

# Build the exact Transformer Engine binding against this runtime. The core
# library and build dependencies were installed from the hash-bound graph.
if [ -z "${NATIVE_BUILD_ROOT:-}" ]; then
    NATIVE_BUILD_PARENT="$(mktemp -d "${THIRD_PARTY_DIR}/torch2141-native-build.XXXXXX")"
    NATIVE_BUILD_ROOT="${NATIVE_BUILD_PARENT}/build"
fi
if ! CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
    python3 "${PROJECT_ROOT}/scripts/build_torch_extensions.py" \
    --contract "${PROJECT_ROOT}/native_torch_extensions.json" \
    --output-dir "${NATIVE_BUILD_ROOT}" \
    --package transformer-engine-torch \
    --cuda-arch-list "${GPU_CUDA_ARCH_LIST}" \
    --max-jobs "${MAX_JOBS:-8}" --nvcc-threads "${NVCC_THREADS:-2}"; then
    echo "ERROR: The pinned Transformer Engine binding build failed."
    exit 1
fi
TE_WHEEL_PATH="$(python3 - "${NATIVE_BUILD_ROOT}/native-torch-extensions.manifest.json" <<'PYNATIVE'
import hashlib
import json
from pathlib import Path
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
wheel = receipt["packages"]["transformer-engine-torch"]["wheel"]
path = Path(wheel["path"])
if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != wheel["sha256"]:
    raise SystemExit("Transformer Engine wheel is missing or its hash does not match")
print(path)
PYNATIVE
)"
pip_install --no-deps "${TE_WHEEL_PATH}"

# CRITICAL: Verify PyTorch CUDA after Transformer Engine installation
echo ""
echo "Verifying PyTorch CUDA after Transformer Engine installation..."
if ! verify_pytorch_cuda "Transformer Engine installation"; then
    echo "ERROR: PyTorch CUDA verification failed after Transformer Engine!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after Transformer Engine"

echo ""
echo "Running FP8 runtime checks (torchao + Transformer Engine)..."
if ! verify_fp8_functionality; then
    echo "ERROR: FP8 runtime checks failed"
    exit 1
fi
echo "✓ FP8 runtime checks passed"

# Snapshot Transformer Engine capability (NVFP4 / MXFP8) after installation
echo ""
echo "Capturing Transformer Engine capability snapshot..."

if ! python3 <<'PY'; then
import warnings

warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*")
warnings.filterwarnings("ignore", message=".*Warning only once for all operators.*")

import torch
import triton
import transformer_engine.pytorch as te

if not torch.cuda.is_available():
    raise SystemExit("Transformer Engine capability check requires PyTorch CUDA")

nvfp4_available, nvfp4_reason = te.is_nvfp4_available(return_reason=True)
mxfp8_available, mxfp8_reason = te.is_mxfp8_available(return_reason=True)


def format_status(flag, reason):
    return "yes" if flag else f"no ({reason or 'not supported on this stack'})"


print(f"  Triton version: {triton.__version__}")
print("Transformer Engine capability snapshot:")
print(f"  NVFP4 available: {format_status(nvfp4_available, nvfp4_reason)}")
print(f"  MXFP8 available: {format_status(mxfp8_available, mxfp8_reason)}")
PY
    echo "ERROR: Transformer Engine capability verification failed."
    exit 1
fi

# Ensure artifacts and wheel caches are writable by the invoking user for future runs.
if [ -n "${SUDO_USER:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/artifacts"
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PROJECT_ROOT}/artifacts" 2>/dev/null || true
    chown -R "${SUDO_USER}:${SUDO_USER}" "${THIRD_PARTY_DIR}/wheels" 2>/dev/null || true
fi

# Verify the selected PyTorch runtime after native package installation
echo ""
echo "============================================================================"
echo "Final verification: PyTorch CUDA installation"
echo "============================================================================"

if ! verify_pytorch_cuda "final verification (after all installations)"; then
    echo "ERROR: PyTorch CUDA verification failed during final verification!"
    exit 1
fi

# Print final status
python3 <<'PY'
import sys
import torch
print(f"✓ Final PyTorch CUDA verification: {torch.__version__} with CUDA {torch.version.cuda}")
PY

# Shared serving uses FA4. FA2 builds belong in a separate training runtime
# because the two distributions own the same flash_attn namespace.
python3 <<'PYATTENTION'
from importlib import metadata
try:
    metadata.distribution("flash-attn")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("FlashAttention-2 must use a separate environment from shared FA4")
from flash_attn.cute import flash_attn_func
if metadata.version("flash-attn-4") != "4.0.0b36":
    raise SystemExit("The shared FlashAttention-4 version differs from the locked graph")
print("Shared FlashAttention-4 frontend import verified")
PYATTENTION

# Install pinned vLLM serving stack and verify ABI against torch.
echo ""
echo "Installing pinned vLLM serving stack (${VLLM_PIP_SPEC}) for this CUDA/toolchain..."
VLLM_ARCH_LIST="${GPU_CUDA_ARCH_LIST}"
VLLM_EXPECTED_VERSION="${VLLM_PIP_SPEC#vllm==}"

# No published wheel targets this torch ABI. Rebuild the fixed source, retain
# its source/metadata/hash receipt, then install only the verified local wheel.
if [ -z "${VLLM_BUILD_ROOT:-}" ]; then
    VLLM_BUILD_PARENT="$(mktemp -d "${THIRD_PARTY_DIR}/vllm-torch2141-build.XXXXXX")"
    VLLM_BUILD_ROOT="${VLLM_BUILD_PARENT}/build"
fi
if ! CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
    python3 "${PROJECT_ROOT}/scripts/build_vllm_for_torch.py" \
    --contract "${VLLM_SOURCE_BUILD_MANIFEST}" \
    --output-dir "${VLLM_BUILD_ROOT}" \
    --cuda-arch-list "${VLLM_ARCH_LIST}" \
    --max-jobs "${MAX_JOBS:-8}" \
    --nvcc-threads "${NVCC_THREADS:-2}"; then
    echo "ERROR: The pinned vLLM source build failed. No binary substitute will be installed."
    exit 1
fi
VLLM_WHEEL_PATH="$(python3 - "${VLLM_BUILD_ROOT}/vllm-source-wheel.manifest.json" <<'PYWHEEL'
import hashlib
import json
from pathlib import Path
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
wheel = Path(receipt["wheel"]["path"])
if not wheel.is_file():
    raise SystemExit("The vLLM build receipt wheel is missing")
if hashlib.sha256(wheel.read_bytes()).hexdigest() != receipt["wheel"]["sha256"]:
    raise SystemExit("The vLLM build receipt wheel hash does not match")
print(wheel)
PYWHEEL
)"
if [ ! -f "${VLLM_WHEEL_PATH}" ]; then
    echo "ERROR: The verified vLLM source wheel is missing."
    exit 1
fi
if ! pip_install --no-deps "${VLLM_WHEEL_PATH}"; then
    echo "ERROR: The verified vLLM source wheel could not be installed."
    exit 1
fi

# The full vLLM runtime dependency graph was installed with hashes above.
# Install no second dependency graph after building the native wheel.
python3 <<PY
import importlib
import importlib.metadata
import os
from pathlib import Path
try:
    import torch
    import vllm  # type: ignore
except Exception as exc:
    print(f"ERROR: vLLM import failed: {exc}")
    raise SystemExit(1)

expected_torch = "${PYTORCH_TORCH_VERSION}"
expected_vllm = "${VLLM_EXPECTED_VERSION}"
expected_flashinfer = "${FLASHINFER_EXPECTED_VERSION}"

torch_version = torch.__version__
def dist_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        print(f"ERROR: required package not installed: {dist_name}")
        raise SystemExit(1)

vllm_dist_version = dist_version("vllm")
flashinfer_dist_version = dist_version("flashinfer-python")

if torch_version != expected_torch:
    print(f"ERROR: torch version mismatch: expected {expected_torch}, got {torch_version}")
    raise SystemExit(1)
if vllm_dist_version != expected_vllm:
    print(f"ERROR: vLLM version mismatch: expected {expected_vllm}, got {vllm_dist_version}")
    raise SystemExit(1)
if flashinfer_dist_version != expected_flashinfer:
    print(
        f"ERROR: flashinfer-python version mismatch: "
        f"expected {expected_flashinfer}, got {flashinfer_dist_version}"
    )
    raise SystemExit(1)

try:
    importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm._moe_C_stable_libtorch")
except Exception as exc:
    print("ERROR: vLLM extension import failed (ABI mismatch likely):")
    print(exc)
    raise SystemExit(1)

print(f"✓ torch installed: {torch_version}")
print(f"✓ vLLM installed: {vllm_dist_version} (module reports {vllm.__version__})")
print(f"✓ flashinfer-python installed: {flashinfer_dist_version}")
print("✓ vLLM stable native extension imports succeeded")

rust_frontend = Path(vllm.__file__).resolve().parent / "vllm-rs"
if not rust_frontend.is_file() or not os.access(rust_frontend, os.X_OK):
    raise SystemExit(f"ERROR: vLLM Rust frontend is missing or not executable: {rust_frontend}")
print(f"✓ vLLM Rust frontend: {rust_frontend}")
PY

echo ""
echo "Verifying PyTorch CUDA after vLLM installation..."
if ! verify_pytorch_cuda "vLLM installation"; then
    echo "ERROR: PyTorch CUDA verification failed after vLLM!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after vLLM"

# Verify installation
echo ""
echo "Verifying installation..."

# Re-export NCCL tuning for verification steps
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

# Check PyTorch
echo "Checking PyTorch installation..."
python3 - "$REQUIRED_DRIVER_VERSION" <<'PY'
import os
import sys
import textwrap
from packaging import version

required_driver = version.parse(sys.argv[1])

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"PyTorch import failed: {exc}")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    try:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except Exception:  # pragma: no cover
        pass
else:
    driver_version = None
    try:
        from torch._C import _cuda_getDriverVersion  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _cuda_getDriverVersion = None

    if _cuda_getDriverVersion is not None:
        try:
            driver_version = _cuda_getDriverVersion()
        except Exception:
            driver_version = None

    if driver_version:
        current = version.parse(str(driver_version))
        if current < required_driver:
            print(
                textwrap.dedent(
                    f"""
                    ⚠ NVIDIA driver {current} is older than required {required_driver}.
                    → Install a newer driver (e.g., nvidia-driver-580) and reboot, then rerun setup.sh.
                    """
                ).strip()
            )
    else:
        print("CUDA runtime not available. Ensure the NVIDIA driver meets CUDA 13.0 Update 2 requirements and reboot if this is a fresh install.")
    raise SystemExit("ERROR: PyTorch CUDA is unavailable during installation verification")
PY

# Check CUDA tools
echo ""
echo "Checking CUDA tools..."
if command -v nvcc &> /dev/null; then
    echo "NVCC: $(nvcc --version | head -1)"
else
    echo "ERROR: NVCC not found"
    exit 1
fi

# Check Nsight tools
echo ""
echo "Checking Nsight tools..."
if command -v nsys &> /dev/null; then
    NSYS_VERSION_OUTPUT="$(nsys --version 2>/dev/null | head -1)"
    echo "Nsight Systems: ${NSYS_VERSION_OUTPUT}"
    if [[ "${NSYS_VERSION_OUTPUT}" == *" ${NSYS_VERSION}"* ]]; then
        echo "  Matches the pinned ${NSYS_VERSION} install"
    else
        echo "  WARNING: expected the pinned Nsight Systems ${NSYS_VERSION} on PATH"
    fi
else
    echo "Nsight Systems not found"
fi

if command -v ncu &> /dev/null; then
    NCU_VERSION_OUTPUT="$(ncu --version 2>/dev/null | grep -m 1 '^Version' || true)"
    echo "Nsight Compute: ${NCU_VERSION_OUTPUT}"
    if [[ "${NCU_VERSION_OUTPUT}" == "Version ${NCU_VERSION}"* ]]; then
        echo "  Matches the pinned ${NCU_VERSION} install"
    else
        echo "  WARNING: expected the pinned Nsight Compute ${NCU_VERSION} on PATH"
    fi
else
    echo "Nsight Compute not found"
fi

# Check system tools
echo ""
echo "Checking system tools..."
tools=("numactl" "perf" "htop" "iostat" "ibstat")
for tool in "${tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo "$tool: installed"
    else
        echo "$tool: not found"
    fi
done

# Run basic performance test
echo ""
echo "Running basic performance test..."
if ! python3 <<'PY'; then
import sys
import time

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: failed to import torch for performance test: {exc}") from exc

if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False for performance test")

device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

for _ in range(10):
    torch.mm(x, y)

start = time.time()
for _ in range(100):
    torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"Matrix multiplication (1000x1000): {(end - start) * 1000 / 100:.2f} ms per operation")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
PY
    echo "ERROR: Basic GPU performance verification failed."
    exit 1
fi

# Test example scripts
echo ""
echo "Testing example scripts..."

# Test Chapter 1
echo "Testing Chapter 1 (Performance Basics)..."
if [ -f "$PROJECT_ROOT/ch01/performance_basics.py" ]; then
    if python3 "$PROJECT_ROOT/ch01/performance_basics.py" > /dev/null 2>&1; then
        echo "Chapter 1: Performance basics working"
    else
        echo "Chapter 1: Some issues detected (check output above)"
    fi
else
    echo "Chapter 1 example not present, skipping."
fi

# Test Chapter 2
echo "Testing Chapter 2 (Hardware Info)..."
if [ -f "$PROJECT_ROOT/ch02/hardware_info.py" ]; then
    if python3 "$PROJECT_ROOT/ch02/hardware_info.py" > /dev/null 2>&1; then
        echo "Chapter 2: Hardware info working"
    else
        echo "Chapter 2: Some issues detected (check output above)"
    fi
else
    echo "Chapter 2 example not present, skipping."
fi

# Test Chapter 3
echo "Testing Chapter 3 (NUMA Affinity)..."
if [ -f "$PROJECT_ROOT/ch03/bind_numa_affinity.py" ]; then
    if python3 "$PROJECT_ROOT/ch03/bind_numa_affinity.py" > /dev/null 2>&1; then
        echo "Chapter 3: NUMA affinity working"
    else
        echo "Chapter 3: Some issues detected (check output above)"
    fi
else
    echo "Chapter 3 example not present, skipping."
fi

# Comprehensive setup verification
echo ""
echo "Running comprehensive setup verification..."
echo "=============================================="

# Test 1: PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
if ! python3 <<'PY'; then
import os
import ctypes
import sys

import torch

print(f"  PyTorch version: {torch.__version__}")
print(f"  PyTorch location: {torch.__file__}")

# Verify CUDA is enabled in the build
if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
    print("ERROR: PyTorch build does not have CUDA enabled!")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    sys.exit(1)

print(f"  CUDA version in PyTorch: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA runtime is not available")
    sys.exit(1)

print(f"  GPU count: {torch.cuda.device_count()}")
print(f"  GPU name: {torch.cuda.get_device_name(0)}")

nccl_path = os.environ.get("NCCL_LIB_PATH")
try:
    if not nccl_path or not os.path.exists(nccl_path):
        raise RuntimeError(f"locked NCCL library is unavailable: {nccl_path!r}")
    libnccl = ctypes.CDLL(nccl_path)
    nv = ctypes.c_int()
    libnccl.ncclGetVersion(ctypes.byref(nv))
    print(f"  NCCL version: {nv.value}")
except Exception as exc:
    raise SystemExit(f"ERROR: NCCL runtime verification failed: {exc}") from exc

print("PyTorch and CUDA working correctly")
PY
    echo "ERROR: PyTorch/CUDA runtime verification failed."
    exit 1
fi

# Test 2: Performance test
echo ""
echo "Testing GPU performance..."
if ! python3 <<'PY'; then
import sys
import time

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA runtime is unavailable for the GPU performance test")
    sys.exit(1)

device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

for _ in range(10):
    torch.mm(x, y)

start = time.time()
for _ in range(100):
    torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"  Matrix multiplication: {(end - start) * 1000 / 100:.2f} ms per operation")
print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
print("GPU performance test passed")
PY
    echo "ERROR: GPU performance verification failed."
    exit 1
fi

# Test 3: torch.compile test
echo ""
echo "Testing torch.compile..."
if ! python3 <<'PY'; then
import sys
import time
import traceback

import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA runtime is unavailable for torch.compile verification")
device = torch.device("cuda")

def simple_model(x):
    return torch.mm(x, x.t())

x = torch.randn(1000, 1000, device=device)

start = time.time()
for _ in range(10):
    simple_model(x)
torch.cuda.synchronize()
uncompiled_time = time.time() - start

try:
    compiled_model = torch.compile(simple_model, mode="reduce-overhead")
except AssertionError as exc:
    print("torch.compile failed with assertion error:")
    print(exc)
    sys.exit(1)
except Exception:
    print("torch.compile failed with an unexpected exception:")
    traceback.print_exc()
    sys.exit(1)

# Warm up compiled model (includes compile time, but we don't measure this)
for _ in range(5):
    compiled_model(x)
torch.cuda.synchronize()

# Now measure compiled performance (compile time excluded)
start = time.time()
for _ in range(20):
    compiled_model(x)
torch.cuda.synchronize()
compiled_time = time.time() - start

print(f"  Uncompiled time: {uncompiled_time * 1000 / 10:.2f} ms per run")
print(f"  Compiled time:   {compiled_time * 1000 / 20:.2f} ms per run (warmed up, compile time excluded)")
if compiled_time < uncompiled_time:
    speedup = uncompiled_time / (compiled_time * 10 / 20)
    print(f"  Speedup: {speedup:.2f}x")
print("torch.compile test passed")
PY
    echo "ERROR: torch.compile verification failed."
    exit 1
fi

# Verify the locked CUTLASS DSL backend for torch.compile
echo ""
echo "Verifying CUTLASS DSL 4.8.0 backend..."
echo "================================================================="

# The graph already installed CUTLASS DSL and CUDA Python bindings.
# Verify availability without resolving a second dependency graph.
if python3 -c 'import cutlass, cuda.bindings'; then
    echo "CUTLASS backend packages installed (pinned versions)"
    echo "   - nvidia-cutlass-dsl[cu13]==4.8.0: CUTLASS kernels for torch.compile"
    
    # Verify PyTorch CUDA after CUTLASS installation
    echo ""
    echo "Verifying PyTorch CUDA after CUTLASS installation..."
    if ! verify_pytorch_cuda "CUTLASS installation"; then
        echo "ERROR: PyTorch CUDA verification failed after CUTLASS!"
        exit 1
    fi
    echo "✓ PyTorch CUDA verified after CUTLASS"
    echo "   - cuda-python==13.0.3: CUDA runtime bindings"
    echo ""
    
    # Detect CUTLASS C++ header location
    CUTLASS_INCLUDE=$(python3 core/scripts/utilities/detect_cutlass_info.py 2>/dev/null | head -1)
    if [ -n "$CUTLASS_INCLUDE" ] && [ -d "$CUTLASS_INCLUDE/include" ]; then
        echo "CUTLASS C++ headers available at: $CUTLASS_INCLUDE/include"
        echo "   Use with: nvcc -I$CUTLASS_INCLUDE/include ..."
        echo ""
        echo "The Python package includes both Python API and C++ headers."
        echo "   No source build needed - ready for both Python and CUDA C++ usage!"
    else
        # CUTLASS headers are in site-packages - this is normal and expected
        # The Python API works fine without explicit header path
        echo "CUTLASS C++ headers are in site-packages (normal for Python package installation)"
    fi
else
    echo "ERROR: Locked CUTLASS backend import failed."
    exit 1
fi

echo ""

echo "Capturing hardware capabilities..."
if python3 "${PROJECT_ROOT}/core/scripts/utilities/probe_hardware_capabilities.py" 2>/tmp/probe_hardware.log; then
    cat /tmp/probe_hardware.log
    echo ""
    echo "Hardware capabilities summary:"
    if [ -f "${PROJECT_ROOT}/artifacts/hardware_capabilities.json" ]; then
        python3 <<PY
import json
import sys
try:
        with open('${PROJECT_ROOT}/artifacts/hardware_capabilities.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            devices = data.get('devices', [])
        elif isinstance(data, list):
            devices = data
        else:
            devices = []
        if not devices:
            raise RuntimeError("capability artifact contains no GPU devices")
        for device in devices:
            name = device.get('name', 'Unknown')
            device_index = device.get('device_index', '?')
            arch = device.get('architecture', 'Unknown')
            compute = device.get('compute_capability', '?')
            print(f"  GPU {device_index}: {name}")
            print(f"    Architecture: {arch} (SM {compute})")
            print(f"    Memory: {device.get('total_memory_gb', 0):.2f} GB")
            print(f"    SMs: {device.get('num_sms', '?')}")
            print(f"    Tensor Cores: {device.get('tensor_cores', 'Unknown')}")
            features = device.get('features', {})
            if isinstance(features, dict):
                print(f"    Features: FP4={features.get('fp4', False)}, FP6={features.get('fp6', False)}, FP8={features.get('fp8', False)}, BF16={features.get('bf16', False)}")
            elif isinstance(features, list):
                joined = ', '.join(features)
                if joined:
                    print(f"    Features: {joined}")
except Exception as e:
    raise SystemExit(f"Error reading capabilities: {e}") from e
PY
    else
        echo "ERROR: hardware_capabilities.json was not created." >&2
        exit 1
    fi
else
    cat /tmp/probe_hardware.log
    echo "ERROR: Hardware capability probe failed"
    exit 1
fi
rm -f /tmp/probe_hardware.log

# Ensure artifacts and wheel caches are writable by the invoking user for future runs.
if [ -n "${SUDO_USER:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/artifacts"
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PROJECT_ROOT}/artifacts" 2>/dev/null || true
    chown -R "${SUDO_USER}:${SUDO_USER}" "${THIRD_PARTY_DIR}/wheels" 2>/dev/null || true
fi

echo ""
echo "Regenerating CUDA arch detection header..."
if python3 "$PROJECT_ROOT/core/scripts/utilities/generate_arch_detection_header.py"; then
    echo "arch_detection.cuh regenerated from live hardware probe"
else
    echo "ERROR: Failed to regenerate arch_detection.cuh from the live GPU."
    exit 1
fi

echo ""

# Test 4: Hardware info script
echo ""
echo "Testing hardware detection..."
if [ -f "$PROJECT_ROOT/ch02/hardware_info.py" ]; then
    if python3 "$PROJECT_ROOT/ch02/hardware_info.py" > /dev/null 2>&1; then
        echo "Hardware detection working"
    else
        echo "ERROR: Hardware detection failed"
        exit 1
    fi
else
    echo "Hardware detection script not present, skipping."
fi

# Test 5: NUMA binding script
echo ""
echo "Testing NUMA binding..."
if [ -f "$PROJECT_ROOT/ch03/bind_numa_affinity.py" ]; then
    # Temporarily disable exit-on-error for this test
    set +e
    python3 "$PROJECT_ROOT/ch03/bind_numa_affinity.py" > /dev/null 2>&1
    NUMA_EXIT_CODE=$?
    set -e
    
    if [ $NUMA_EXIT_CODE -eq 0 ]; then
        echo "NUMA binding working"
    else
        echo "NUMA binding had issues (expected in containers or without distributed launch)"
    fi
else
    echo "NUMA binding script not present, skipping."
fi

# Test 6: Codex MCP configuration
echo ""
echo "Testing Codex MCP configuration..."
if [ -f "$PROJECT_ROOT/scripts/setup_mcp.sh" ]; then
    ensure_codex_cli
    if [ -n "${CODEX_BIN:-}" ]; then
        "$PROJECT_ROOT/scripts/setup_mcp.sh" --codex-bin "${CODEX_BIN}"
    else
        "$PROJECT_ROOT/scripts/setup_mcp.sh"
    fi
    echo "Codex MCP configuration working"
else
    echo "ERROR: MCP setup script not found at $PROJECT_ROOT/scripts/setup_mcp.sh"
    exit 1
fi

echo ""
echo "All critical tests passed! Setup is working correctly."

# Ensure non-root GPU reset capability via nvidia-smi
echo ""
echo "Configuring nvidia-smi for non-root GPU resets..."
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! command -v setcap >/dev/null 2>&1; then
        echo "Installing libcap2-bin (provides setcap)..."
        apt-get update >/dev/null 2>&1 || true
        apt-get install -y libcap2-bin >/dev/null 2>&1 || true
    fi

    if command -v setcap >/dev/null 2>&1; then
        if ! getcap /usr/bin/nvidia-smi 2>/dev/null | grep -q "cap_sys_admin"; then
            if setcap cap_sys_admin+ep /usr/bin/nvidia-smi 2>/dev/null; then
                echo "Granted cap_sys_admin capability to /usr/bin/nvidia-smi"
            else
                echo "WARNING: Failed to grant cap_sys_admin to /usr/bin/nvidia-smi (continuing)"
            fi
        else
            echo "/usr/bin/nvidia-smi already has cap_sys_admin capability"
        fi
        getcap /usr/bin/nvidia-smi 2>/dev/null || true
    else
        echo "WARNING: setcap not available; skipping capability grant."
    fi
else
    echo "nvidia-smi not found; skipping GPU reset capability grant."
fi

# Restart services impacted by NVSHMEM/CUDA installs (e.g., glances monitoring)
echo ""
# Restart background services if needed (silently skip if not present)
if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-units --type=service --all 2>/dev/null | grep -q "^glances.service"; then
        if systemctl is-active --quiet glances.service 2>/dev/null; then
            systemctl restart glances.service >/dev/null 2>&1 && echo "Restarted glances.service" || true
        else
            systemctl restart glances.service >/dev/null 2>&1 || true
        fi
    fi
fi

# Run verification scripts
echo ""
echo "Running Verification Checks..."
echo "=================================="
echo ""

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"
if python3 core/verification/verify_pytorch.py; then
    echo "PyTorch verification passed"
else
    echo "ERROR: PyTorch verification failed"
    exit 1
fi

echo ""

# Verify NVLink connectivity (only if multiple GPUs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Verifying NVLink connectivity..."
    # Enable NVLink counters so downstream telemetry can read per-link utilization
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi nvlink -gt d >/dev/null 2>&1 || true
    fi
    if python3 core/verification/verify_nvlink.py; then
        echo "NVLink verification passed"
    else
        echo "NVLink verification had warnings (review output above)"
    fi
else
    echo "Single GPU detected, skipping NVLink verification"
fi

echo ""

# Verify CUTLASS backend
echo "Verifying CUTLASS backend..."
if python3 core/verification/verify_cutlass.py 2>/dev/null; then
    echo "CUTLASS verification passed"
else
    echo "ERROR: CUTLASS verification failed"
    exit 1
fi

echo ""

# Verify GPUDirect Storage (GDS)
echo "Verifying GPUDirect Storage (GDS)..."
if lsblk -d -o NAME,TRAN 2>/dev/null | grep -qi nvme; then
    if python3 core/verification/verify_gds.py; then
        echo "GDS verification passed"
    else
        echo "GDS verification had issues (may need to load nvidia-fs module)"
        echo "   Load module with: sudo modprobe nvidia-fs"
        echo "   Ensure a supported storage backend (NVMe, BeeGFS, NFS, WekaFS, etc.) is configured and rerun /usr/local/cuda/gds/tools/gdscheck -p"
    fi
else
    echo "WARNING: No NVMe devices detected; skipping GDS verification (requires NVMe or supported backend)."
fi

echo ""

echo "All critical verifications passed!"

# Run peak performance benchmark
echo ""
echo "Running Peak Performance Benchmark..."
echo "======================================"
if python3 <<'PY'
import json
import os
import runpy
from datetime import datetime
from pathlib import Path

from core.harness.benchmark_harness import lock_gpu_clocks

try:
    import pynvml
except ImportError as exc:
    raise SystemExit(f"pynvml required for clock verification: {exc}")

output_dir = Path(os.environ["PROJECT_ROOT"])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
manifest_path = output_dir / f"benchmark_peak_manifest_{timestamp}.json"

with lock_gpu_clocks():
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
    finally:
        pynvml.nvmlShutdown()

    print(f"App clocks locked: SM={app_sm}MHz Mem={app_mem}MHz")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "device_index": 0,
                "app_clock": {"sm_mhz": app_sm, "mem_mhz": app_mem},
                "notes": "benchmark_peak run with lock_gpu_clocks wrapper",
            },
            indent=2,
        )
        + "\n"
    )
    import sys
    sys.argv = ["benchmark_peak.py", "--output-dir", str(output_dir)]
    runpy.run_path(str(output_dir / "core/benchmark/benchmark_peak.py"), run_name="__main__")

print(f"Peak run manifest saved to: {manifest_path}")
PY
then
    echo "Peak performance benchmark completed successfully"
else
    echo "ERROR: Peak performance benchmark failed"
    exit 1
fi

# Reject dependency drift after every native build and optional setup path.
python3 -m pip check
python3 - "${PYTORCH_STACK_LOCK}" <<'PYLOCK'
from importlib import metadata
from packaging.requirements import Requirement
from pathlib import Path
import re
import sys
mismatches = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not re.match(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==", line):
        continue
    requirement = Requirement(line.removesuffix("\\").strip())
    actual = metadata.version(requirement.name)
    if not requirement.specifier.contains(actual, prereleases=True):
        mismatches.append(f"{requirement.name}: expected {requirement.specifier}, found {actual}")
if mismatches:
    raise SystemExit("Locked runtime drift:\n" + "\n".join(mismatches))
print("Every locked runtime version matches")
PYLOCK


echo ""
if [[ -n "${GPU_COMPUTE_SM_NUM:-}" && "${GPU_COMPUTE_SM_NUM}" -ge 100 ]]; then
    echo "Running comprehensive Blackwell feature verification..."
    echo "(TMEM, TMA, CTA Clusters, DSMEM, FP8, FP4, Warp Features)"
    if python3 "${PROJECT_ROOT}/core/verification/verify_all_blackwell_features.py" --json; then
        echo "✓ All Blackwell features verified successfully!"
        echo "  Results saved to: artifacts/blackwell_verification/verification_results.json"
    else
        echo "ERROR: Blackwell feature verification failed."
        echo "  Review: artifacts/blackwell_verification/verification_results.json"
        exit 1
    fi
else
    echo "Skipping Blackwell verification (non-Blackwell GPU detected)."
fi

# Point the global CLI at this runtime only after every installation, benchmark,
# lock, and architecture validation has succeeded.
install_aisp_cli_wrapper

echo ""
echo "Final cleanup (APT/pip caches)..."
reclaim_disk_space_basic

# Final summary
echo ""
echo "Setup Complete!"
echo "=================="
echo ""
echo "Installed:"
echo "  • PyTorch ${PYTORCH_TORCH_VERSION} CUDA wheel from ${PYTORCH_CU130_INDEX}"
echo "  • CUDA ${CUDA_FULL_VERSION} toolchain and development tools"
echo "  • NCCL ${NCCL_WHEEL_VERSION} from the locked nvidia-nccl-cu13 wheel"
echo "  • cuDNN ${CUDNN_WHEEL_VERSION} from the locked nvidia-cudnn-cu13 wheel"
echo "  • NVSHMEM ${NVSHMEM_WHEEL_VERSION} from the locked nvidia-nvshmem-cu13 wheel"
echo "  • CUTLASS ${CURRENT_CUTLASS_VERSION} commit ${CUTLASS_RESOLVED_COMMIT} tree ${CUTLASS_RESOLVED_TREE}"
echo "  • GPUDirect Storage (GDS) tools, drivers, kvikio library, and boot-time nvidia-fs load config"
echo "  • NVIDIA Nsight Systems ${NSYS_VERSION}"
echo "  • NVIDIA Nsight Compute ${NCU_VERSION}"
echo "  • CMake ${REQ_CMAKE_VERSION} and ninja ${REQ_NINJA_VERSION} (pip wheels pinned in requirements_latest.txt)"
echo "  • All project dependencies"
echo "  • System performance tools (numactl, perf, etc.)"
echo ""
echo "Verified:"
echo "  • PyTorch installation and CUDA functionality"
echo "  • Locked NCCL, cuDNN, and NVSHMEM wheel libraries and headers"
echo "  • NVLink check executed when multiple GPUs were present"
echo "  • CUTLASS source provenance and backend configuration"
echo "  • GDS checks executed when an NVMe device was present"
echo "  • Blackwell features (if SM 10.0+): TMEM, TMA, Clusters, DSMEM, FP8, FP4"
echo ""
echo "Post-Run Checklist:"
echo "  • Verify Python runtime:          python --version  (expect 3.12.x)"
echo "  • Verify CUDA compiler:           nvcc --version    (expect release 13.0.88 / Update 2)"
echo "  • Verify CMake:                   cmake --version   (expect ${REQ_CMAKE_VERSION})"
echo "  • Verify driver & GPU status:     nvidia-smi        (expect driver ${REQUIRED_DRIVER_VERSION})"
echo "  • Storage: Configure GDS-supported storage (NVMe, BeeGFS, NFS, WekaFS, etc.) and validate with /usr/local/cuda/gds/tools/gdscheck -p"
echo "  • Verify PyTorch arch coverage:   python - <<'PY'"
echo "                                      import torch"
echo "                                      print(torch.cuda.get_arch_list())"
echo "                                    PY"
echo "                                    (must include sm_${GPU_COMPUTE_SM_NUM})"
echo "  • NCCL library:                   ${NCCL_LIB_PATH}"
echo "  • cuDNN library:                  ${CUDNN_LIB_PATH}"
echo "  • NVSHMEM library:                ${NVSHMEM_LIB_PATH}"
echo "  • Before additional builds: source ${AISP_VENV_DIR}/bin/activate and ${AISP_VENV_DIR}/runtime-env.sh"
echo "For more information, see the README.md file and chapter-specific documentation."
echo ""
echo "Happy performance engineering!"
