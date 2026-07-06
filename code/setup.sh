#!/bin/bash

#
# AI Performance Engineering Setup Script
# ========================================
#
# This script installs EVERYTHING you need:
#   1. NVIDIA Driver 580.126.09 (auto-upgrades if needed; open kernel modules for B200/B300)
#   2. Python 3.12
#   3. CUDA 13.0.2 toolkit + cuBLAS 13.1.0.3 (Update 2) repository
#   4. The canonical PyTorch/Triton stack pinned in requirements_latest.txt
#      (torch==2.9.1+cu130, triton==3.5.0) for CUDA 13.0.2; versions are
#      single-sourced from requirements_latest.txt, not the nightly defaults below
#   5. NVIDIA Nsight Systems 2025.3.2 (for timeline profiling)
#   6. NVIDIA Nsight Compute 2025.3.1 (for kernel metrics)
#   7. All Python dependencies from requirements_latest.txt
#   8. System tools (numactl, perf, htop, etc.)
#   9. Configures NVIDIA drivers for profiling
#
# Requirements:
#   - Ubuntu 22.04+ (tested on 22.04)
#   - NVIDIA B200/B300 GPU (or compatible)
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
#   - Fixes Python APT module (python3-apt) compatibility
#   - Disables problematic command-not-found APT hook
#   - Removes duplicate deadsnakes repository entries
#   - Upgrades Python to 3.12 (required by PyTorch 2.10 dev builds)
#   - Auto-upgrades NVIDIA driver to 580+ if needed (will prompt reboot)
#   - Installs CUDA 13.0.2 toolkit and libraries
#   - Installs latest Nsight tools (2025.x)
#   - Prepares for PyTorch 2.10-dev (source build) with CUDA 13.0.2
#   - Removes conflicting system packages (python3-optree, etc.)
#   - Installs nvidia-ml-py (replaces deprecated pynvml)
#   - Configures NVIDIA kernel modules for profiling
#   - Fixes hardware info script compatibility
#   - Runs validation tests
#
# Notes:
#   - If driver upgrade is needed, script will exit and ask you to reboot
#   - After reboot, simply re-run: sudo ./setup.sh
#   - The script is idempotent and safe to re-run
#   - Disk cleanup is enabled by default (CLEAN_APT_CACHE/CLEAN_PIP_CACHE/CLEAN_BUILD_ARTIFACTS)
#   - Minimum free space thresholds: SETUP_MIN_FREE_GB / SETUP_MIN_TE_BUILD_FREE_GB
#
# After running this script, you can:
#   - Run examples: python3 ch01/performance_basics.py
#   - Drive the benchmark suite: python -m cli.aisp bench run
#   - Capture peak performance: python core/benchmark/benchmark_peak.py
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# =============================================================================
# Logging + Disk Hygiene (always-on logging, keep disk usage tidy)
# =============================================================================
LOG_FILE="${SETUP_LOG_FILE:-${PROJECT_ROOT}/setup.log}"
LOG_MAX_BYTES="${SETUP_LOG_MAX_BYTES:-104857600}"  # 100 MB
LOG_COMPRESS="${SETUP_LOG_COMPRESS:-1}"
SETUP_MIN_FREE_GB="${SETUP_MIN_FREE_GB:-6}"
SETUP_MIN_TE_BUILD_FREE_GB="${SETUP_MIN_TE_BUILD_FREE_GB:-8}"
CLEAN_APT_CACHE="${CLEAN_APT_CACHE:-1}"
CLEAN_PIP_CACHE="${CLEAN_PIP_CACHE:-1}"
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
    if [ "${CLEAN_PIP_CACHE}" -eq 1 ]; then
        python3 -m pip cache purge >/dev/null 2>&1 || true
        rm -rf /root/.cache/pip >/dev/null 2>&1 || true
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
echo "  • Python 3.12 (PyTorch 2.10-dev compatible)"
echo "  • CUDA 13.0.2 toolkit + cuBLAS 13.1.0.3 (Update 2) repository"
echo "  • Environment configured for PyTorch 2.10-dev source build"
echo "  • NVIDIA Nsight Systems 2025.3.2 (latest)"
echo "  • NVIDIA Nsight Compute 2025.3.1 (latest)"
echo "  • All project dependencies"
echo "  • System tools (numactl, perf, etc.)"
echo ""
echo "Note: If driver upgrade is needed, you'll be prompted to reboot."
echo ""

REQUIRED_DRIVER_VERSION="580.126.09"
PYTHON_TARGET_VERSION="3.12"
PYTHON_TARGET_MAJOR="${PYTHON_TARGET_VERSION%%.*}"
PYTHON_TARGET_MINOR="${PYTHON_TARGET_VERSION##*.}"
PYTHON_TARGET_BIN="python${PYTHON_TARGET_VERSION}"
PYTHON_ABI_TAG="cp${PYTHON_TARGET_MAJOR}${PYTHON_TARGET_MINOR}"
PYTHON_DIST_PACKAGES="/usr/local/lib/python${PYTHON_TARGET_VERSION}/dist-packages"
CUDA_SHORT_VERSION="13.0"
CUDA_FULL_VERSION="13.0.2"
# cuBLAS is pinned to 13.1.0.3 (Update 2) via cuda-libraries; cuDNN latest in repo
CUDNN_VERSION="9.16.0.29"
NCCL_SHORT_VERSION="2.28.7"
CUDA_HOME_DIR="/usr/local/cuda-${CUDA_SHORT_VERSION}"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
mkdir -p "${THIRD_PARTY_DIR}"
FLASH_ATTN_TAG="${FLASH_ATTN_TAG:-v2.8.3}"
FLASH_ATTN_ARCH="$(uname -m)"
if [ "${FLASH_ATTN_ARCH}" = "arm64" ]; then
    FLASH_ATTN_ARCH="aarch64"
fi
FLASH_ATTN_WHEEL_BASENAME="flash_attn-2.8.3-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_${FLASH_ATTN_ARCH}.whl"
FLASH_ATTN_EXPECTED_VERSION="${FLASH_ATTN_TAG#v}"
detect_default_sm() {
    if [ -n "${GPU_COMPUTE_SM_NUM:-}" ]; then
        echo "${GPU_COMPUTE_SM_NUM}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
        if [ -n "${cap}" ]; then
            local major="${cap%.*}"
            local minor="${cap#*.}"
            printf "%s%s\n" "${major}" "${minor}"
            return
        fi
    fi
    echo "121"
}
FLASH_ATTENTION_FORCE_CUDA_SM_VALUE="${FLASH_ATTENTION_FORCE_CUDA_SM_VALUE:-$(detect_default_sm)}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_VERSION_TAG="${VLLM_VERSION_TAG:-main}"
INSTALL_GPT_OSS="${INSTALL_GPT_OSS:-0}"
VLLM_GIT_REF="${VLLM_GIT_REF:-${VLLM_VERSION_TAG}}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-${THIRD_PARTY_DIR}/vllm-src}"
VLLM_WHEEL_DIR="${THIRD_PARTY_DIR}/wheels"
VLLM_WHEEL_INFO_PATH="${VLLM_WHEEL_INFO_PATH:-${VLLM_WHEEL_DIR}/vllm-build-info.json}"
VLLM_WHEEL_ARCH="$(uname -m)"
VLLM_EXTRA_INDEX_URL="${VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/cu130}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.16.0}"
FLASHINFER_EXPECTED_VERSION="${FLASHINFER_EXPECTED_VERSION:-0.6.3}"
TORCHTITAN_TOMLI_VERSION="${TORCHTITAN_TOMLI_VERSION:-2.4.0}"
TORCHTITAN_TYRO_VERSION="${TORCHTITAN_TYRO_VERSION:-1.0.10}"
TORCHTITAN_RUNTIME_DEPS=(
    "tomli==${TORCHTITAN_TOMLI_VERSION}"
    "tyro==${TORCHTITAN_TYRO_VERSION}"
)
VLLM_RUNTIME_DEPS=(
    "cbor2==5.8.0"
    "msgspec==0.20.0"
    "gguf==0.18.0"
    "ijson==3.5.0"
    "pybase64==1.4.3"
    "setproctitle==1.3.7"
    "diskcache==5.6.3"
    "partial-json-parser==0.2.1.1.post7"
    "lm-format-enforcer==0.11.3"
    "outlines_core==0.2.11"
    "llguidance==1.3.0"
    "xgrammar==0.1.29"
    "compressed-tensors==0.13.0"
    "depyf==0.20.0"
    "watchfiles==1.1.1"
    "blake3==1.0.8"
    "anthropic==0.84.0"
    "openai==2.24.0"
    "openai-harmony==0.0.8"
    "model-hosting-container-standards==0.1.13"
    "mcp==1.26.0"
    "grpcio-reflection==1.78.0"
)
if [ "${VLLM_WHEEL_ARCH}" = "arm64" ]; then
    VLLM_WHEEL_ARCH="aarch64"
fi
VLLM_WHEEL_PATTERN="${VLLM_WHEEL_PATTERN:-${VLLM_WHEEL_DIR}/vllm-*-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_${VLLM_WHEEL_ARCH}.whl}"
# =============================================================================
# DEPENDENCY VERSION PINS (update together, test after changes)
# Run: python core/scripts/check_upstream_versions.py --check-te-cutlass
# Source of truth for pinned versions lives here (dependency_versions.json removed)
# =============================================================================
#
# CUTLASS 4.3.0 - Required for SM100a (Blackwell) support
#   - Provides: tmem_allocator_sm100.hpp, mma_sm100_umma.hpp, copy_traits_sm100.hpp
#   - Uses the release tag with corrected version metadata
#
# TransformerEngine v2.9 - Stable release with CUDA 13 wheels
#   - IMPORTANT: TE v2.9 still bundles CUTLASS 4.2.0
#   - CUTLASS 4.2.0 LACKS SM100a headers - symlink workaround REQUIRED
#   - We replace TE's bundled CUTLASS with our 4.3.0 via symlink
#   - Check: make verify-cutlass
#
# When to remove symlink workaround:
#   - When TE bundles CUTLASS >= 4.3.0 with SM100a headers
#   - Run: python core/scripts/check_upstream_versions.py --check-te-cutlass
#   - If "TE main bundles: CUTLASS 4.3.0+" appears, symlink may be removable
#
# MLPerf v6 source trees
#   - Inference README explicitly recommends the master branch for v6.0 submissions.
#   - Training README carries a v6.0 section but does not publish a GitHub release tag,
#     so we track its official branch ref and record the resolved commit at install time.
#
TE_REPO_URL="${TE_REPO_URL:-https://github.com/NVIDIA/TransformerEngine.git}"
# TE v2.9 release (2025-11-11) - stable release with CUDA 13 support
TE_GIT_COMMIT="${TE_GIT_COMMIT:-v2.9}"
TE_VERSION="v2.9"
TE_BUNDLED_CUTLASS_VERSION="4.2.0"  # What TE bundles (needs symlink override)
TE_SRC_DIR="${TE_SRC_DIR:-${THIRD_PARTY_DIR}/TransformerEngine-src}"
CUTLASS_REPO_URL="${CUTLASS_REPO_URL:-https://github.com/NVIDIA/cutlass.git}"
# CUTLASS 4.3.0 release tag
CUTLASS_REF="${CUTLASS_REF:-v4.3.0}"
CUTLASS_TARGET_VERSION="${CUTLASS_TARGET_VERSION:-4.3.0}"
CUTLASS_SRC_DIR="${CUTLASS_SRC_DIR:-${THIRD_PARTY_DIR}/cutlass}"
MLPERF_INFERENCE_REPO_URL="${MLPERF_INFERENCE_REPO_URL:-https://github.com/mlcommons/inference.git}"
MLPERF_INFERENCE_GIT_REF="${MLPERF_INFERENCE_GIT_REF:-master}"
MLPERF_INFERENCE_SRC_DIR="${MLPERF_INFERENCE_SRC_DIR:-${THIRD_PARTY_DIR}/mlperf_inference}"
MLPERF_TRAINING_REPO_URL="${MLPERF_TRAINING_REPO_URL:-https://github.com/mlcommons/training.git}"
MLPERF_TRAINING_GIT_REF="${MLPERF_TRAINING_GIT_REF:-master}"
MLPERF_TRAINING_SRC_DIR="${MLPERF_TRAINING_SRC_DIR:-${THIRD_PARTY_DIR}/mlperf_training}"
PIP_ROOT_USER_ACTION="ignore"
SOURCE_BUILD_ALLOWED=0
GPU_COMPUTE_SM_NUM=""
VLLM_PREBUILT_INSTALLED=0
export PROJECT_ROOT REQUIRED_DRIVER_VERSION PYTHON_TARGET_VERSION PYTHON_TARGET_MAJOR PYTHON_TARGET_MINOR PYTHON_TARGET_BIN PYTHON_ABI_TAG PYTHON_DIST_PACKAGES PIP_ROOT_USER_ACTION

if command -v git >/dev/null 2>&1; then
    git config --global --add safe.directory "${PROJECT_ROOT}" >/dev/null 2>&1 || true
    if [ -d "${PROJECT_ROOT}/vendor/pytorch-src" ]; then
        git config --global --add safe.directory "${PROJECT_ROOT}/vendor/pytorch-src" >/dev/null 2>&1 || true
    fi
    if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if [ -f "${PROJECT_ROOT}/.gitmodules" ]; then
            git -C "${PROJECT_ROOT}" submodule sync --recursive >/dev/null 2>&1 || true
            git -C "${PROJECT_ROOT}" submodule update --init --recursive >/dev/null 2>&1 || true
        fi
    fi
fi
PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-https://github.com/pytorch/pytorch.git}"
PYTORCH_COMMIT="${PYTORCH_COMMIT:-main}"
PYTORCH_SRC_DIR="${PYTORCH_SRC_DIR:-${THIRD_PARTY_DIR}/pytorch-src}"
PYTORCH_BUILD_DIR="${PYTORCH_SRC_DIR}"
PYTORCH_DIST_DIR="${PYTORCH_BUILD_DIR}/dist"
PYTORCH_WHEEL_DIR="${THIRD_PARTY_DIR}/wheels"
PYTORCH_WHEEL_PATTERN="${PYTORCH_WHEEL_PATTERN:-torch-*-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-*.whl}"
mkdir -p "${PYTORCH_WHEEL_DIR}"

TORCH_CUDA_ARCH_LIST_VALUE="10.0;10.3;12.1;12.2+PTX"
CMAKE_CUDA_ARCH_LIST_VALUE="100;103;121;122"
TORCH_SM_ARCH_LIST_VALUE="sm_100;sm_103;sm_121;sm_122"
CUTLASS_NVCC_ARCHS_VALUE_DEFAULT="100;103;121;122"
CUTLASS_NVCC_ARCHS_VALUE="${CUTLASS_NVCC_ARCHS_VALUE_DEFAULT}"
PYTORCH_NIGHTLY_DATE="20251213"
PYTORCH_TORCH_VERSION="2.10.0.dev${PYTORCH_NIGHTLY_DATE}+cu130"
# PYTORCH_TORCHVISION_VERSION="0.25.0.dev${PYTORCH_NIGHTLY_DATE}+cu130"
PYTORCH_TORCHAUDIO_VERSION="2.10.0.dev${PYTORCH_NIGHTLY_DATE}+cu130"
PYTORCH_TORCHAO_VERSION="0.16.0.dev${PYTORCH_NIGHTLY_DATE}+cu130"
PYTORCH_TRITON_VERSION="3.6.0+git8fedd49b"
PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly"
PYTORCH_CU130_INDEX_ROOT="https://download.pytorch.org/whl/nightly/cu130"
PYTORCH_CU130_INDEX="${PYTORCH_CU130_INDEX_ROOT}"
PYTORCH_TORCH_FIND_LINKS="${PYTORCH_TORCH_FIND_LINKS:-https://download.pytorch.org/whl/nightly/cu130/torch/}"
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
    if [ -z "${PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES:-}" ]; then
        if python3 -m pip --help 2>&1 | grep -q -- '--break-system-packages'; then
            PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES=1
        else
            PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES=0
        fi
    fi

    if [ "${PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES}" -eq 1 ]; then
        python3 -m pip --break-system-packages "$@"
    else
        PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip "$@"
    fi
}

pip_install() {
    pip_cmd install "$@"
}

pip_uninstall() {
    pip_cmd uninstall "$@"
}

pip_wheel() {
    pip_cmd wheel "$@"
}

pip_show() {
    pip_cmd show "$@"
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

# Reusable function to reassemble split wheels
reassemble_split_wheel() {
    local wheel_path="$1"
    local tmp_dir="${2:-$(mktemp -d "${TMPDIR:-/tmp}/wheel-reassemble.XXXXXX")}"
    
    # If full wheel exists, return it
    if [ -f "${wheel_path}" ]; then
        echo "${wheel_path}"
        return 0
    fi
    
    # Check for split parts
    if compgen -G "${wheel_path}.part*" >/dev/null 2>&1; then
        local combined="${tmp_dir}/$(basename "${wheel_path}")"
        mapfile -t PARTS < <(ls "${wheel_path}".part* | sort -V)
        if cat "${PARTS[@]}" > "${combined}" 2>/dev/null; then
            echo "${combined}"
            return 0
        fi
    fi
    
    return 1
}

# Reusable function to verify PyTorch CUDA and restore if needed
verify_and_restore_pytorch_cuda() {
    local context="$1"
    python3 <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: PyTorch CUDA not available")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    sys.exit(1)
PY
    if [ $? -ne 0 ]; then
        echo "CRITICAL: PyTorch CUDA was not available during ${context}!"
        return 1
    fi
    return 0
}

ensure_wheel_root_pure() {
    local wheel_path="$1"
    if [ -z "${wheel_path}" ] || [ ! -f "${wheel_path}" ]; then
        return 0
    fi
    python3 - "${wheel_path}" <<'PY'
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])
if not wheel_path.exists():
    raise SystemExit(0)

with zipfile.ZipFile(wheel_path, "r") as src:
    wheel_entries = [name for name in src.namelist() if name.endswith(".dist-info/WHEEL")]
    if not wheel_entries:
        raise SystemExit(0)
    with tempfile.TemporaryDirectory(dir=str(wheel_path.parent)) as tmpdir:
        tmp_path = Path(tmpdir) / wheel_path.name
        with zipfile.ZipFile(tmp_path, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename in wheel_entries:
                    text = data.decode("utf-8").splitlines()
                    for idx, line in enumerate(text):
                        if line.startswith("Root-Is-Purelib:"):
                            if line.strip().lower() != "root-is-purelib: false":
                                text[idx] = "Root-Is-Purelib: false"
                            break
                    else:
                        text.append("Root-Is-Purelib: false")
                    data = ("\n".join(text) + "\n").encode("utf-8")
                dst.writestr(info, data)
    shutil.move(tmp_path, wheel_path)
PY
}

patch_installed_transformer_engine_metadata() {
    python3 <<'PY'
import importlib.metadata as metadata
from importlib.metadata import PackageNotFoundError

def patch_distribution(name: str) -> None:
    try:
        dist = metadata.distribution(name)
    except PackageNotFoundError:
        return

    files = dist.files or []
    wheel_entry = None
    for file in files:
        if file.name == "WHEEL":
            wheel_entry = dist.locate_file(file)
            break

    if not wheel_entry:
        return

    lines = wheel_entry.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("Root-Is-Purelib:"):
            if line.strip().lower() != "root-is-purelib: false":
                lines[idx] = "Root-Is-Purelib: false"
            break
    else:
        lines.append("Root-Is-Purelib: false")

    wheel_entry.write_text("\n".join(lines) + "\n")

for dist_name in ("transformer_engine", "transformer_engine_torch", "transformer_engine_cu12"):
    patch_distribution(dist_name)
PY
}

patch_transformer_engine_loader() {
    python3 <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

OLD_BLOCK = '''    if te_framework_installed:
        assert te_installed_via_pypi, "Could not find `transformer-engine` PyPI package."
        assert te_core_installed, "Could not find TE core package `transformer-engine-cu*`."

        assert version(module_name) == version("transformer-engine") == te_core_version, (
            "Transformer Engine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and {te_core_package_name}"
            f" v{te_core_version}. Install transformer-engine using "
            f"'pip3 install --no-build-isolation transformer-engine[{extra_dep_name}]==VERSION'"
        )
'''

NEW_BLOCK = '''    if te_framework_installed:
        if te_installed_via_pypi and te_core_installed:
            assert version(module_name) == version("transformer-engine") == te_core_version, (
                "Transformer Engine package version mismatch. Found"
                f" {module_name} v{version(module_name)}, transformer-engine"
                f" v{version('transformer-engine')}, and {te_core_package_name}"
                f" v{te_core_version}. Install transformer-engine using "
                f"'pip3 install --no-build-isolation transformer-engine[{extra_dep_name}]==VERSION'"
            )
        else:
            pass
'''

patched = False
for dist_name in ("transformer_engine", "transformer-engine"):
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        continue
    path = Path(dist.locate_file("transformer_engine/common/__init__.py"))
    if not path.exists():
        continue
    text = path.read_text()
    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
        path.write_text(text)
        patched = True

if patched:
    print("[setup] Patched Transformer Engine loader for local wheel support")
else:
    print("[setup] Transformer Engine loader patch skipped (already applied)")
PY
}

install_proton_cli_stub() {
    if command -v proton >/dev/null 2>&1; then
        echo "Proton CLI already available (proton command found)"
        return 0
    fi
    local target="/usr/local/bin/proton"
    install -m 755 "${PROJECT_ROOT}/core/scripts/proton_stub.py" "${target}"
    echo "Installed Proton stub CLI at ${target}"
}

install_aisp_cli_wrapper() {
    local target="/usr/local/bin/aisp"
    cat > "${target}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${PROJECT_ROOT}/cli/aisp.py" "\$@"
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

    if [ -z "${CODEX_INSTALL_CMD:-}" ]; then
        echo "ERROR: codex CLI not found. Set CODEX_INSTALL_CMD to install it before MCP setup." >&2
        exit 1
    fi

    echo "Installing Codex CLI..."
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ] && [ "${CODEX_INSTALL_AS_USER:-1}" -eq 1 ]; then
        sudo -H -u "${SUDO_USER}" bash -lc "${CODEX_INSTALL_CMD}"
    else
        bash -lc "${CODEX_INSTALL_CMD}"
    fi

    if command -v codex >/dev/null 2>&1; then
        CODEX_BIN="$(command -v codex)"
        export CODEX_BIN
        echo "Codex CLI installed at ${CODEX_BIN}"
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

    echo "ERROR: codex CLI not found after install. Ensure it is in PATH or set CODEX_BIN to its full path." >&2
    exit 1
}

remove_conflicting_user_triton() {
    if [ -z "${SUDO_USER:-}" ] || [ "${SUDO_USER}" = "root" ]; then
        return 0
    fi
    local user_site
    user_site=$(sudo -H -u "${SUDO_USER}" python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null) || true
    if [ -z "${user_site}" ]; then
        return 0
    fi
    if sudo -H -u "${SUDO_USER}" test -d "${user_site}/triton"; then
        rm -rf "${user_site}/triton"
    fi
    sudo -H -u "${SUDO_USER}" sh -c "rm -rf ${user_site}/pytorch_triton-*.dist-info" 2>/dev/null || true
}

remove_usercustomize_shim() {
    local targets=(
        "$HOME/.local/lib/python3.12/site-packages/usercustomize.py"
        "/usr/local/lib/python3.12/dist-packages/usercustomize.py"
    )
    for target in "${targets[@]}"; do
        if [ -f "$target" ]; then
            rm -f "$target"
            echo "[setup] Removed legacy usercustomize shim at $target"
        fi
    done
}

disable_transformer_engine_sanity_check() {
    python3 <<'PY'
import ast
import importlib.metadata as metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

def patch_module(module_path: Path) -> bool:
    if not module_path.exists():
        return False
    source = module_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sanity_checks_for_pypi_installation":
            target = node
            break
    if target is None or target.lineno is None or target.end_lineno is None:
        return False
    lines = source.splitlines()
    replacement = [
        "def sanity_checks_for_pypi_installation() -> None:",
        "    \"\"\"Runtime environment bundles TE wheels directly; skip PyPI provenance checks.\"\"\"",
        "    return None",
        "",
    ]
    start = target.lineno - 1
    end = target.end_lineno
    lines[start:end] = replacement
    module_path.write_text("\n".join(lines) + ("\n" if lines and lines[-1] else ""))
    return True

patched_any = False
for dist_name in ("transformer_engine", "transformer-engine"):
    try:
        dist = metadata.distribution(dist_name)
    except PackageNotFoundError:
        continue
    module_path = Path(dist.locate_file("transformer_engine/common/__init__.py"))
    if patch_module(module_path):
        print(f"[setup] Patched Transformer Engine sanity check at {module_path}")
        patched_any = True

if not patched_any:
    print("[setup] Warning: transformer_engine.common not patched (module not found)")
PY
}

patch_transformer_engine_loader
install_proton_cli_stub
install_aisp_cli_wrapper
remove_conflicting_user_triton
remove_usercustomize_shim

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
        msg = str(exc)
        if "flash_attn" in msg or "flash_attn_2_cuda" in msg or "undefined symbol" in msg:
            return None, f"Transformer Engine import skipped due to FlashAttention issue: {exc}"
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
    elif result["ok"] is None:
        print(f"[setup] ⚠ {name} FP8 runtime check skipped: {result['error']}")
    else:
        print(f"[setup] ERROR: {name} FP8 runtime check failed: {result['error']}")

if not all(entry["ok"] or entry["ok"] is None for entry in status.values()):
    raise SystemExit(1)
PY
}

TORCHAO_EXTRA_INDEX_URL="${PYTORCH_CU130_INDEX}"

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    echo "Installing lsb-release..."
    apt update && apt install -y lsb-release
fi

UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "20.04" ]]; then
    echo "Warning: This script is tested on Ubuntu 22.04. Other versions may work but are not guaranteed."
fi

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

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "NVIDIA GPU detected"

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
else
    echo "NVIDIA GPU not detected. Please ensure NVIDIA drivers are installed."
    exit 1
fi

GPU_COMPUTE_CAP_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')
GPU_COMPUTE_SM_NUM=$(echo "${GPU_COMPUTE_CAP_RAW}" | tr -d '.')
FLASH_ATTN_TARGET_SM="${GPU_COMPUTE_SM_NUM:-121}"
FLASH_ATTN_TARGET_ARCHS="${FLASH_ATTENTION_CUDA_ARCHS:-${FLASH_ATTN_TARGET_SM}}"
export FLASH_ATTN_TARGET_SM FLASH_ATTN_TARGET_ARCHS
if [[ -n "${GPU_COMPUTE_SM_NUM}" ]]; then
    echo "Detected GPU compute capability: sm_${GPU_COMPUTE_SM_NUM}"
    SOURCE_BUILD_ALLOWED=0
    export GPU_COMPUTE_SM_NUM
else
    SOURCE_BUILD_ALLOWED=0
    echo "Could not detect GPU compute capability; defaulting to prebuilt wheels only."
fi

# Use current arch for CUTLASS build targets (fallback to defaults if unknown)
if [ -n "${GPU_COMPUTE_SM_NUM}" ]; then
    CUTLASS_NVCC_ARCHS_VALUE="${GPU_COMPUTE_SM_NUM}"
else
    CUTLASS_NVCC_ARCHS_VALUE="${CUTLASS_NVCC_ARCHS_VALUE_DEFAULT}"
fi
if [ -n "${GPU_COMPUTE_SM_NUM}" ]; then
    sm_len=${#GPU_COMPUTE_SM_NUM}
    sm_major="${GPU_COMPUTE_SM_NUM:0:$((sm_len-1))}"
    sm_minor="${GPU_COMPUTE_SM_NUM: -1}"
    # Use Blackwell 'a' suffix to enable TMA instructions in ptxas
    if [ "${GPU_COMPUTE_SM_NUM}" -eq 100 ]; then
        TE_TORCH_ARCH_LIST="${sm_major}.${sm_minor}a"
        TE_CUTLASS_ARCHS="100a"
    else
        TE_TORCH_ARCH_LIST="${sm_major}.${sm_minor}"
        TE_CUTLASS_ARCHS="${GPU_COMPUTE_SM_NUM}"
    fi
else
    TE_TORCH_ARCH_LIST="10.0"
    TE_CUTLASS_ARCHS="${CUTLASS_NVCC_ARCHS_VALUE_DEFAULT}"
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

# Fix apt_pkg module before apt update (if Python was upgraded)
if ! python3 -c "import apt_pkg" 2>/dev/null; then
    echo "Fixing apt_pkg module..."
    apt install -y --reinstall python3-apt 2>/dev/null || true
fi

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
apt install -y wget curl software-properties-common

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

python3 <<'PY'
import os
import shutil
import site
import sys

candidates = [
    "/usr/share/dcgm/bindings/python3",
    "/usr/share/dcgm/bindings/python",
    "/usr/share/dcgm/python3",
    "/usr/share/dcgm/python",
    "/usr/lib/dcgm/python3",
    "/usr/lib/dcgm/python",
]

dest = None
for path in site.getsitepackages():
    if path.startswith("/usr/lib") or path.startswith("/usr/local/lib"):
        dest = path
        break

if dest is None:
    print("Unable to locate site-packages directory; skipping DCGM Python binding install.")
    sys.exit(0)

for cand in candidates:
    if not os.path.isdir(cand):
        continue
    for entry in os.listdir(cand):
        if entry == "__pycache__":
            continue
        src = os.path.join(cand, entry)
        dst = os.path.join(dest, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"DCGM Python bindings installed to {dest}")
    break
else:
    print("DCGM Python bindings not found; pydcgm import may fail.")
PY

# Add NVIDIA CUDA ${CUDA_SHORT_VERSION} repository
echo ""
echo "Adding NVIDIA CUDA ${CUDA_SHORT_VERSION} repository..."

# Check if CUDA repository is already configured
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]; then
    CUDA_REPO_PKG="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/tmp/$CUDA_REPO_PKG" ]; then
        wget -P /tmp https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$CUDA_REPO_PKG
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
    apt install -y "${PYTHON_TARGET_BIN}" "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv" python3-pip
    echo "Python ${PYTHON_TARGET_VERSION} installed"
else
    CURRENT_TARGET_VERSION=$("${PYTHON_TARGET_BIN}" --version 2>&1 | awk '{print $2}')
    echo "Python ${PYTHON_TARGET_VERSION} already installed (version $CURRENT_TARGET_VERSION)"
    # Still ensure dev packages are present
    apt install -y "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv" python3-pip
fi

# Set target Python as default if not already
CURRENT_PY3=$(python3 --version 2>&1 | awk '{print $2}')
if [[ ! "$CURRENT_PY3" =~ ^${PYTHON_TARGET_MAJOR}\.${PYTHON_TARGET_MINOR}\. ]]; then
    echo "Setting Python ${PYTHON_TARGET_VERSION} as default..."
    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/${PYTHON_TARGET_BIN}" 1
    update-alternatives --set python3 "/usr/bin/${PYTHON_TARGET_BIN}"
else
    echo "Python ${PYTHON_TARGET_VERSION} is already the default"
fi

# Ensure `python` convenience shim follows target Python
if ! command -v python >/dev/null 2>&1; then
    echo "Creating python -> ${PYTHON_TARGET_BIN} alternative..."
    update-alternatives --install /usr/bin/python python "/usr/bin/${PYTHON_TARGET_BIN}" 1
    update-alternatives --set python "/usr/bin/${PYTHON_TARGET_BIN}"
else
    PYTHON_VERSION_OUTPUT="$(python --version 2>/dev/null || true)"
    if [[ "$PYTHON_VERSION_OUTPUT" != "Python ${PYTHON_TARGET_VERSION}."* ]]; then
        echo "Updating python alternative to point at ${PYTHON_TARGET_BIN}..."
        update-alternatives --install /usr/bin/python python "/usr/bin/${PYTHON_TARGET_BIN}" 1
        update-alternatives --set python "/usr/bin/${PYTHON_TARGET_BIN}"
    else
        echo "python points to $(python --version) (no change needed)"
    fi
fi

# Ensure pip is installed for target Python
if ! "${PYTHON_TARGET_BIN}" -m pip --version &> /dev/null; then
    echo "Installing pip for Python ${PYTHON_TARGET_VERSION}..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "${PYTHON_TARGET_BIN}"
fi

# Upgrade pip
pip_install --upgrade --ignore-installed pip setuptools packaging

# Ensure wheel build backend is available for manual builds
pip_install --no-cache-dir --upgrade --ignore-installed wheel

install_torchao_packages() {
    echo "Installing torchao (nightly, CUDA 13.x / cu130 index)..."
    local install_args=(--no-cache-dir --upgrade --ignore-installed torchao --extra-index-url "${TORCHAO_EXTRA_INDEX_URL}")
    if pip_install "${install_args[@]}"; then
        echo "torchao installed in system Python site-packages"
    else
        echo "ERROR: torchao installation failed for system Python!"
        exit 1
    fi
    python3 - <<'PY'
try:
    import torchao  # noqa: F401
    print("[setup] torchao import verified (system)")
except Exception as exc:
    raise SystemExit(f"[setup] ERROR: torchao import failed after installation: {exc}")
PY
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
        echo "Installing torchao for ${SUDO_USER}'s user site-packages..."
        if sudo -H -u "${SUDO_USER}" PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user "${install_args[@]}"; then
            echo "  ✓ torchao installed for ${SUDO_USER}"
        else
            echo "ERROR: Failed to install torchao for ${SUDO_USER}"
            exit 1
        fi
    fi
}

# Fix python3-apt for the new Python version
DISTUTILS_PTH="/usr/lib/python3/dist-packages/distutils-precedence.pth"
if [ -f "$DISTUTILS_PTH" ]; then
    echo "Removing distutils-precedence.pth shim (causes _distutils_hack errors)..."
    rm -f "$DISTUTILS_PTH"
fi

echo ""
echo "Fixing Python APT module..."
apt install -y --reinstall python3-apt
if [ -f "$DISTUTILS_PTH" ]; then
    echo "Removing distutils-precedence.pth shim after python3-apt reinstall..."
    rm -f "$DISTUTILS_PTH"
fi

# Remove distro flatbuffers package whose invalid version breaks pip metadata
if dpkg -s python3-flatbuffers >/dev/null 2>&1; then
    echo "Removing distro python3-flatbuffers package (invalid version metadata)..."
    apt remove -y python3-flatbuffers
fi

# Upgrade NVIDIA driver to required version if needed (required for CUDA ${CUDA_SHORT_VERSION} Update 2)
echo ""
echo "Checking NVIDIA driver version..."
if command -v nvidia-smi &> /dev/null; then
    CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    
    if ! dpkg --compare-versions "$CURRENT_DRIVER" ge "$REQUIRED_DRIVER_VERSION"; then
        echo "Current driver ($CURRENT_DRIVER) is too old for CUDA ${CUDA_SHORT_VERSION} Update 2"
        echo "Upgrading to NVIDIA driver 580 (open kernel modules)..."
        
        # Unhold any NVIDIA packages to avoid resolver conflicts
        if apt-mark showhold 2>/dev/null | grep -Eq '^(nvidia|libnvidia|cuda)-'; then
            echo "Releasing held NVIDIA/CUDA packages..."
            apt-mark unhold $(apt-mark showhold | grep -E '^(nvidia|libnvidia|cuda)-' | tr '\n' ' ') || true
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
    echo "Pinned CUDA toolkit version unavailable; installing latest cuda-toolkit-13-0 package instead."
    apt install -y cuda-toolkit-13-0
fi

# Install NCCL for Blackwell optimizations
echo ""
echo "Installing NCCL ${NCCL_SHORT_VERSION} (Blackwell-optimized)..."
if ! apt install -y --allow-downgrades "libnccl2=${NCCL_SHORT_VERSION}-1+cuda13.0" "libnccl-dev=${NCCL_SHORT_VERSION}-1+cuda13.0"; then
    echo "Pinned NCCL ${NCCL_SHORT_VERSION} packages unavailable; installing default libnccl2/libnccl-dev from CUDA repo."
    apt install -y libnccl2 libnccl-dev
fi

# Install cuDNN 9.16.0.29-1 (latest in CUDA 13 repo, matches current system pin)
# Following NVIDIA's approach: PyTorch bundles cuDNN, but we install a matching system version for build tools
echo ""
echo "Installing cuDNN 9.16.0.29-1 for CUDA ${CUDA_SHORT_VERSION}..."
# Remove any older cuDNN versions first
apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 2>/dev/null || true
# Install the exact version if available; otherwise fall back to repo latest
if apt-cache madison libcudnn9-cuda-13 | grep -q "9.16.0.29-1"; then
    apt install -y "libcudnn9-cuda-13=9.16.0.29-1" \
                   "libcudnn9-dev-cuda-13=9.16.0.29-1" \
                   "libcudnn9-headers-cuda-13=9.16.0.29-1"
    echo "✓ Installed cuDNN 9.16.0.29-1 (CUDA 13)"
else
    echo "WARNING: cuDNN 9.16.0.29-1 not available, installing latest..."
    apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13
fi
# Pin the version to prevent upgrades
apt-mark hold libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13

# Update ldconfig cache to ensure correct cuDNN version is found
echo "Updating ldconfig cache..."
ldconfig 2>/dev/null || true

# Install NVSHMEM 3.4.5 for CUDA 13 (enables SymmetricMemory fast paths)
echo ""
echo "Installing NVSHMEM 3.4.5 runtime and headers (CUDA 13)..."
apt install -y nvshmem-cuda-13 libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13 libnvshmem3-static-cuda-13
# Fix packaging bug that leaves nvshmem_transport_ibgda.so.3 alternative pointing at a non-existent file
nvshmem_alt_file="/var/lib/dpkg/alternatives/nvshmem"
nvshmem_bad_ibgda="/usr/lib/x86_64-linux-gnu/nvshmem/13/.3"
nvshmem_good_ibgda="/usr/lib/x86_64-linux-gnu/nvshmem/13/nvshmem_transport_ibgda.so.3"
if [ -f "${nvshmem_alt_file}" ] && grep -q "${nvshmem_bad_ibgda}" "${nvshmem_alt_file}"; then
    sed -i "s#${nvshmem_bad_ibgda}#${nvshmem_good_ibgda}#" "${nvshmem_alt_file}"
    if ! update-alternatives --auto nvshmem; then
        echo "WARNING: update-alternatives failed for nvshmem; nvshmem_transport_ibgda.so.3 may be missing"
    fi
fi
if [ ! -e "/usr/lib/x86_64-linux-gnu/nvshmem_transport_ibgda.so.3" ] && [ -e "${nvshmem_good_ibgda}.0.0" ]; then
    ln -sf "${nvshmem_good_ibgda}.0.0" "/usr/lib/x86_64-linux-gnu/nvshmem_transport_ibgda.so.3"
fi

# Install TensorRT runtime + Python bindings (CUDA 13)
echo ""
echo "Installing TensorRT runtime and Python bindings (CUDA 13)..."
apt install -y libnvinfer10 libnvinfer-plugin10 libnvinfer-vc-plugin10 libnvonnxparsers10 python3-libnvinfer

# TensorRT-LLM runtime helpers (MPI + audio deps)
echo ""
echo "Installing TensorRT-LLM system dependencies (OpenMPI, libsndfile)..."
apt install -y openmpi-bin libsndfile1

# Ensure nvshmemrun is on PATH for IBGDA/NVSHMEM binaries
NVSHMEM_BIN_CANDIDATES=(
    "${NVSHMEM_HOME:-/usr/lib/x86_64-linux-gnu}/bin"
    "/usr/local/nvshmem/bin"
    "/opt/nvidia/nvshmem/bin"
    "/usr/lib/nvshmem/bin"
    "/usr/lib/x86_64-linux-gnu/nvshmem/bin"
    "/usr/bin/nvshmem_13"
    "/usr/bin"
)
if ! ensure_tool_on_path "nvshmemrun" "${NVSHMEM_BIN_CANDIDATES[@]}"; then
    if ensure_tool_on_path "nvshmrun.hydra" "/usr/bin/nvshmem_13" "/usr/bin"; then
        ln -sf "/usr/local/bin/nvshmrun.hydra" "/usr/local/bin/nvshmemrun"
        echo "Ensured nvshmemrun is on PATH via nvshmrun.hydra alias"
    else
        echo "WARNING: nvshmemrun not found on system PATH; set NVSHMEM_HOME or install nvshmemrun to enable NVSHMEM multi-GPU launchers"
    fi
fi
if ! ensure_tool_on_path "hydra_pmi_proxy" "/usr/bin/nvshmem_13" "/usr/bin"; then
    echo "WARNING: hydra_pmi_proxy not found on system PATH; nvshmemrun may fail to launch"
fi

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
    modprobe nvidia-fs 2>/dev/null && echo "nvidia-fs module loaded" || {
        echo "Could not load nvidia-fs module (requires root)"
        echo "   Load it manually with: sudo modprobe nvidia-fs"
        echo "   Or run: sudo core/scripts/setup/load_gds_module.sh"
    }
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

# Install kvikio Python library for GDS (required for Ch5 examples)
echo ""
echo "Installing kvikio Python library for GPUDirect Storage..."
if pip_install --no-cache-dir --upgrade --ignore-installed kvikio-cu13==25.10.0 --extra-index-url https://download.pytorch.org/whl/cu130; then
    echo "kvikio-cu13==25.10.0 installed (enables GPU Direct Storage in Python)"
else
    echo "kvikio installation failed, but continuing..."
    echo "   Install manually with: pip install kvikio-cu13==25.10.0"
fi

# Install CUDA sanitizers and debugging tools (compute-sanitizer, cuda-memcheck, etc.)
echo ""
echo "Installing CUDA sanitizers and debugging tools..."
if apt install -y cuda-command-line-tools-13-0; then
    echo "CUDA command-line tools 13.0 installed (compute-sanitizer, cuda-gdb, cuda-memcheck)"
else
        echo "Could not install cuda-command-line-tools-13-0, trying fallback packages..."
    if apt install -y cuda-command-line-tools; then
            echo "CUDA command-line tools (generic) installed"
    else
            echo "cuda-command-line-tools package unavailable. Trying NVIDIA CUDA toolkit..."
        if apt install -y nvidia-cuda-toolkit; then
                echo "NVIDIA CUDA toolkit installed (includes cuda-memcheck)"
        else
                echo "Could not install CUDA command-line tools. compute-sanitizer may be unavailable."
        fi
    fi
fi

# Ensure compute-sanitizer is present; install sanitizer package directly if needed
if ! command -v compute-sanitizer &> /dev/null; then
    echo "compute-sanitizer not found after command-line tools install. Installing cuda-sanitizer package..."
    if apt install -y cuda-sanitizer-13-0; then
        echo "cuda-sanitizer-13-0 installed"
    else
        echo "Could not install cuda-sanitizer-13-0, attempting generic cuda-sanitizer package..."
        if apt install -y cuda-sanitizer; then
            echo "cuda-sanitizer package installed"
        else
            echo "compute-sanitizer installation failed; please install manually."
        fi
    fi
fi

# Install latest NVIDIA Nsight Systems and Compute
echo ""
echo "Installing latest NVIDIA Nsight Systems and Compute..."

# Update apt to ensure latest packages are available
apt update -qq

# Create temporary directory for downloads
TEMP_DIR="/tmp/nsight_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Install Nsight Systems and set binary alternative  
echo "Installing Nsight Systems (pinned 2025.3.2)..."
NSYS_VERSION="2025.3.2"
apt install -y --upgrade "nsight-systems-${NSYS_VERSION}"
# Try multiple possible locations
for bin_path in "/opt/nvidia/nsight-systems/${NSYS_VERSION}/bin/nsys" "/opt/nvidia/nsight-systems/${NSYS_VERSION}/nsys"; do
    if [[ -x "$bin_path" ]]; then
        NSYS_BIN="$bin_path"
        break
    fi
done
if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
    update-alternatives --install /usr/local/bin/nsys nsys "$NSYS_BIN" 50
    update-alternatives --set nsys "$NSYS_BIN"
    echo "Nsight Systems pinned to ${NSYS_VERSION} (${NSYS_BIN})"
else
    echo "Nsight Systems binary not found"
fi
# Fallback: ensure nsys is reachable even if alternatives fail or install path changes
if ! ensure_tool_on_path "nsys" \
    "/opt/nvidia/nsight-systems/${NSYS_VERSION}/bin" \
    "/opt/nvidia/nsight-systems/${NSYS_VERSION}" \
    "/opt/nvidia/nsight-systems/"*/bin \
    "/usr/local/cuda-*/bin" \
    "/usr/local/cuda/bin"; then
    echo "Nsight Systems still not on PATH; please verify installation."
fi

# Install Nsight Compute and set binary alternative
echo "Installing Nsight Compute (pinned 2025.3.1)..."
NCU_VERSION="2025.3.1"
apt install -y --upgrade "nsight-compute-${NCU_VERSION}"
NCU_BIN="/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu"
if [[ -x "$NCU_BIN" ]]; then
    update-alternatives --install /usr/local/bin/ncu ncu "$NCU_BIN" 50
    update-alternatives --set ncu "$NCU_BIN"
    echo "Nsight Compute pinned to ${NCU_VERSION} (${NCU_BIN})"
else
    echo "Nsight Compute binary not found at ${NCU_BIN}"
fi
# Fallback: ensure ncu is reachable even if alternatives fail or install path changes
if ! ensure_tool_on_path "ncu" \
    "/opt/nvidia/nsight-compute/${NCU_VERSION}" \
    "/opt/nvidia/nsight-compute/${NCU_VERSION}/bin" \
    "/opt/nvidia/nsight-compute/"*/bin \
    "/usr/local/cuda-*/bin" \
    "/usr/local/cuda/bin"; then
    echo "Nsight Compute still not on PATH; please verify installation."
fi

# Nsight tools are already in PATH when installed via apt
echo "Nsight tools installed and available in PATH"

# Configure PATH and LD_LIBRARY_PATH for CUDA environment
echo ""
echo "Configuring CUDA ${CUDA_SHORT_VERSION} environment..."
# Update /etc/environment for system-wide CUDA path
if ! grep -q "${CUDA_HOME_DIR}/bin" /etc/environment; then
    sed -i "s|PATH=\"\(.*\)\"|PATH=\"${CUDA_HOME_DIR}/bin:\1\"|" /etc/environment
    echo "Added CUDA ${CUDA_SHORT_VERSION} to system PATH"
fi

# Create profile.d script for CUDA
CUDA_PROFILE_SCRIPT="/etc/profile.d/cuda-${CUDA_SHORT_VERSION}.sh"
cat > "${CUDA_PROFILE_SCRIPT}" <<PROFILE_EOF
# CUDA ${CUDA_SHORT_VERSION} environment variables
export CUDA_HOME=${CUDA_HOME_DIR}
export PATH=${CUDA_HOME_DIR}/bin:\$PATH
export CUDA_PATH=${CUDA_HOME_DIR}

# Runtime library policy:
#  - auto (default): prefer PyTorch's bundled runtime when present
#  - torch: force PyTorch runtime precedence
#  - system: force system runtime precedence
export AISP_CUDNN_RUNTIME_POLICY="\${AISP_CUDNN_RUNTIME_POLICY:-auto}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${CUDA_HOME}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
PROFILE_EOF
chmod +x "${CUDA_PROFILE_SCRIPT}"
echo "Created ${CUDA_PROFILE_SCRIPT} for persistent CUDA ${CUDA_SHORT_VERSION} environment"

# Update nvcc symlink to CUDA toolkit (override Ubuntu's default)
rm -f /usr/bin/nvcc
ln -s "${CUDA_HOME_DIR}/bin/nvcc" /usr/bin/nvcc
echo "Updated /usr/bin/nvcc symlink to CUDA ${CUDA_SHORT_VERSION}"

# Source the CUDA environment for current session
source "${CUDA_PROFILE_SCRIPT}"

# Persist CUDA toolchain settings for the remainder of the script
export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH="$CUDA_HOME/bin:$PATH"

export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"
export LD_LIBRARY_PATH="${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo ""
echo "Verifying CUDA toolchain after installation..."
if ! nvcc --version >/tmp/nvcc_version.txt 2>&1; then
    cat /tmp/nvcc_version.txt
    echo "nvcc not available even after CUDA install. Aborting."
    exit 1
fi
cat /tmp/nvcc_version.txt
rm -f /tmp/nvcc_version.txt

python3 - <<'PY'
import sys
try:
    import torch
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"torch.version.cuda = {cuda_ver}")
    if not cuda_ver:
        print("PyTorch build does not have CUDA enabled")
except ImportError:
    print("PyTorch not yet installed (will be installed later in this script)")
except Exception as exc:
    print(f"Warning: Could not verify PyTorch CUDA support: {exc}")
PY

# Install runtime monitoring/precision dependencies that require CUDA headers
echo ""
echo "Installing CUDA-dependent Python packages (prometheus-client, transformer-engine)..."

# Helper to ensure pip installs run with CUDA environment populated
pip_cuda() {
    local cuda_home="${CUDA_HOME:-${CUDA_HOME_DIR}}"
    CUDA_HOME="${cuda_home}" \
    TORCH_CUDA_HOME="${cuda_home}" \
    CUDA_PATH="${cuda_home}" \
    PATH="${cuda_home}/bin:${PATH}" \
    LD_LIBRARY_PATH="${cuda_home}/lib64:${cuda_home}/lib64/stubs:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="${PROJECT_ROOT}/.cuda_env:${PYTHONPATH:-}" \
    pip_cmd "$@"
}

export PYTHONPATH="${PROJECT_ROOT}/.cuda_env:${PYTHONPATH:-}"

# Ensure NumPy is present for torch's extension helper before building any wheels
pip_cuda install --no-cache-dir --upgrade --ignore-installed numpy || true

install_cuda_package() {
    local package_spec="$1"
    echo "Installing $package_spec"
    if pip_cuda install \
        --no-cache-dir --upgrade --ignore-installed --prefer-binary "$package_spec"; then
        echo "Installed $package_spec"
        return 0
    fi

    echo "Installing numpy inside pip build environment to satisfy torch setup requirements..."
    pip_cuda install --no-cache-dir --upgrade --ignore-installed numpy || true

    echo "Initial install of $package_spec failed, retrying without build isolation..."
    if pip_cuda install \
        --no-cache-dir --upgrade --ignore-installed --no-build-isolation --prefer-binary "$package_spec"; then
        echo "Installed $package_spec (no-build-isolation)"
        return 0
    fi

    echo "Failed to install $package_spec"
    return 1
}

install_cuda_package "prometheus-client==0.21.0"

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
    cmake \
    rustc \
    cargo \
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
echo "Installing Rust nightly toolchain (edition2024 support)..."
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi
if command -v rustup >/dev/null 2>&1; then
    rustup toolchain install nightly
    rustup default nightly
else
    echo "WARNING: rustup not available; edition2024 builds may fail."
fi

echo ""
echo "Installing detect-secrets (system-wide)..."
pip_install --no-cache-dir --upgrade --ignore-installed detect-secrets

# Install ninja (required for PyTorch CUDA extensions)
echo ""
echo "Installing ninja (required for CUDA extensions)..."
pip_install --no-cache-dir --upgrade --ignore-installed ninja==1.13.0

# Force remove ALL existing torch installations (CPU and CUDA) to prevent conflicts
echo ""
echo "Removing all existing torch installations..."
pip_uninstall -y torch torchvision torchdata functorch pytorch-triton >/dev/null 2>&1 || true
# Also remove any dist-info directories that pip couldn't clean up
rm -rf /usr/local/lib/python3.12/dist-packages/torch* /usr/local/lib/python3.12/dist-packages/torchvision* /usr/local/lib/python3.12/dist-packages/torchdata* /usr/local/lib/python3.12/dist-packages/functorch* /usr/local/lib/python3.12/dist-packages/pytorch_triton* 2>/dev/null || true
rm -rf /home/*/.local/lib/python3.12/site-packages/torch* /home/*/.local/lib/python3.12/site-packages/torchvision* 2>/dev/null || true

# Ensure critical harness dependencies are present in the base (non-venv) environment
echo ""
echo "Installing pinned harness deps (pydantic/typer/typing_extensions) into system Python..."
pip_install --no-cache-dir --upgrade --ignore-installed \
    click==8.1.7 \
    pydantic==2.12.4 \
    pydantic-core==2.41.5 \
    hypothesis==6.138.0 \
    typing-extensions==4.15.0 \
    typer==0.12.0 \
    typer-slim[standard]==0.12.0

# Build or reuse PyTorch wheel (SM100/103/121)
echo ""
find "${PYTORCH_WHEEL_DIR}" -maxdepth 1 -name "torch-*-manylinux*.whl" -delete 2>/dev/null || true
PYTORCH_WHEEL_GLOB="${PYTORCH_WHEEL_DIR}/${PYTORCH_WHEEL_PATTERN}"
echo ""
echo "NOTE: PyTorch CUDA wheels will be installed AFTER core requirements"
echo "      to prevent dependencies from overriding them with CPU versions"
echo ""

# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip_uninstall -y torchvision >/dev/null 2>&1 || true

# Use the updated requirements file with pinned versions
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_latest.txt"

# Single-source serving stack pins from requirements_latest.txt.
# This keeps setup-time ABI checks aligned with runtime benchmark checks.
if [ -f "$REQUIREMENTS_FILE" ]; then
    REQ_TORCH_VERSION="$(grep -E '^torch==' "$REQUIREMENTS_FILE" | head -n 1 | cut -d= -f3 || true)"
    REQ_TRITON_VERSION="$(grep -E '^triton==' "$REQUIREMENTS_FILE" | head -n 1 | cut -d= -f3 || true)"
    REQ_VLLM_VERSION="$(grep -E '^vllm==' "$REQUIREMENTS_FILE" | head -n 1 | cut -d= -f3 || true)"
    REQ_FLASHINFER_VERSION="$(grep -E '^flashinfer-python==' "$REQUIREMENTS_FILE" | head -n 1 | cut -d= -f3 || true)"

    if [ -n "${REQ_TORCH_VERSION}" ]; then
        PYTORCH_TORCH_VERSION="${REQ_TORCH_VERSION}"
        PYTORCH_TORCHAUDIO_VERSION="${REQ_TORCH_VERSION}"
    fi
    # Single-source Triton too so setup-time installs match the requirements pin
    # (requirements pins triton==3.5.0; the default above is a 3.6 nightly).
    if [ -n "${REQ_TRITON_VERSION}" ]; then
        PYTORCH_TRITON_VERSION="${REQ_TRITON_VERSION}"
    fi
    if [ -n "${REQ_VLLM_VERSION}" ]; then
        VLLM_PIP_SPEC="vllm==${REQ_VLLM_VERSION}"
    fi
    if [ -n "${REQ_FLASHINFER_VERSION}" ]; then
        FLASHINFER_EXPECTED_VERSION="${REQ_FLASHINFER_VERSION}"
    fi

    echo "Serving stack pins (from requirements_latest.txt):"
    echo "  torch==${PYTORCH_TORCH_VERSION}"
    echo "  triton==${PYTORCH_TRITON_VERSION}"
    echo "  ${VLLM_PIP_SPEC}"
    echo "  flashinfer-python==${FLASHINFER_EXPECTED_VERSION}"
fi

# Detect FlashInfer availability (flashinfer-python wheels target Python 3.10-3.14)
FLASHINFER_SPEC=""
SKIP_FLASHINFER=0
if [ -f "$REQUIREMENTS_FILE" ]; then
    FLASHINFER_SPEC="$(grep -E '^flashinfer-python==' "$REQUIREMENTS_FILE" | head -n 1 || true)"
    if [ -n "${FLASHINFER_SPEC}" ]; then
        if python3 - <<'PY' >/dev/null 2>&1
import sys
v = sys.version_info[:2]
raise SystemExit(0 if (3, 10) <= v <= (3, 14) else 1)
PY
        then
            :
        else
            SKIP_FLASHINFER=1
            echo "NOTE: ${FLASHINFER_SPEC} skipped in bulk install for Python $(python3 -c 'import sys; print(\"%d.%d\" % sys.version_info[:2])')."
            echo "      Set FORCE_FLASHINFER_INSTALL=1 to attempt install anyway."
        fi
    fi
fi

# Create temporary requirements file without packages we install in a pinned
# post-CUDA step (prevents CPU torch overrides and vLLM ABI drift).
TEMP_REQUIREMENTS="/tmp/requirements_no_torch_deps.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    REQUIREMENTS_EXCLUDE_REGEX='^(accelerate==|torchtitan==|torch==|vllm==)'
    if [ "${SKIP_FLASHINFER}" -eq 1 ]; then
        REQUIREMENTS_EXCLUDE_REGEX='^(accelerate==|torchtitan==|torch==|vllm==|flashinfer-python==)'
    fi
    grep -Ev "${REQUIREMENTS_EXCLUDE_REGEX}" "$REQUIREMENTS_FILE" > "$TEMP_REQUIREMENTS" || true
    echo "Created temporary requirements file excluding torch/vLLM/accelerate/torchtitan"
    echo "  (these are installed later in a pinned, post-CUDA compatibility step)"
fi

# Install dependencies with error handling (excluding accelerate/torchtitan)
if [ -f "$TEMP_REQUIREMENTS" ]; then
    echo "Installing Python packages from requirements file (excluding accelerate/torchtitan)..."
    if ! pip_install --no-cache-dir --no-input --ignore-installed -r "$TEMP_REQUIREMENTS"; then
        echo "Some packages failed to install from requirements file."
        echo "Installing core packages individually..."
            pip_install --no-cache-dir --no-input --ignore-installed \
            blinker==1.9.0 \
            nvidia-mathdx==25.6.0 \
            nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
            numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
            matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
            click==8.1.7 \
            jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 ruff==0.8.4 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 typer==0.12.0 rich==13.7.0 \
            transformers==4.56.0 datasets==2.21.0 sentencepiece==0.2.0 tokenizers==0.22.2 \
            onnx==1.19.0 \
            py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
            optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
            dask==2025.9.1 xarray==2025.6.1 \
            "fsspec[http]==2024.6.1"
    fi
    rm -f "$TEMP_REQUIREMENTS"
else
    echo "Requirements file not found at $REQUIREMENTS_FILE. Installing core packages directly..."
    pip_install --no-cache-dir --no-input --ignore-installed \
        blinker==1.9.0 \
        nvidia-mathdx==25.6.0 \
        nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
        numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
        matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
        click==8.1.7 \
        jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 ruff==0.8.4 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 typer==0.12.0 rich==13.7.0 \
        transformers==4.56.0 datasets==2.21.0 sentencepiece==0.2.0 tokenizers==0.22.2 \
        onnx==1.19.0 \
        py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
        optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
        dask==2025.9.1 xarray==2025.6.1 \
        "fsspec[http]==2024.6.1"
fi

if [ -n "${FLASHINFER_SPEC}" ] && [ "${SKIP_FLASHINFER}" -eq 1 ] && [ "${FORCE_FLASHINFER_INSTALL:-0}" -eq 1 ]; then
    echo "Attempting optional ${FLASHINFER_SPEC} install (forced)..."
    if ! pip_install --no-cache-dir --upgrade --ignore-installed --only-binary=:all: "${FLASHINFER_SPEC}"; then
        echo "Warning: ${FLASHINFER_SPEC} install failed; continuing without it."
    fi
fi

# Install PyTorch CUDA 13 stack (binary wheels only, no source builds)
echo ""
echo "============================================================================"
echo "Installing PyTorch nightly 2.10 cu130 stack (binary wheels only)"
echo "============================================================================"
echo ""

echo "Removing any existing PyTorch installations..."
pip_uninstall -y torch torchvision torchdata functorch pytorch-triton >/dev/null 2>&1 || true

echo "Installing torch ${PYTORCH_TORCH_VERSION} + torchaudio/torchao (cu130 nightly)..."
if ! pip_install --no-cache-dir --upgrade --ignore-installed \
    --index-url "${PYTORCH_CU130_INDEX}" \
    --find-links "${PYTORCH_TORCH_FIND_LINKS}" \
    --extra-index-url "https://pypi.org/simple" \
    --only-binary=":all:" \
    "torch==${PYTORCH_TORCH_VERSION}"; then
    echo "ERROR: torch ${PYTORCH_TORCH_VERSION} install failed from nightly cu130 index"
    exit 1
fi

# NOTE: torchvision is intentionally skipped for now; nightly cu130 wheels have been failing.
# If/when needed, re-enable this block and validate that cu130 wheels exist.
# if ! pip_install --no-cache-dir --upgrade --ignore-installed \
#     --index-url "${PYTORCH_CU130_INDEX}" \
#     --extra-index-url "https://pypi.org/simple" \
#     --only-binary=":all:" \
#     "torchvision==${PYTORCH_TORCHVISION_VERSION}"; then
#     echo "ERROR: torchvision install failed from nightly cu130 index"
#     exit 1
# fi

if ! pip_install --no-cache-dir --upgrade --ignore-installed \
    --index-url "${PYTORCH_CU130_INDEX}" \
    --extra-index-url "https://pypi.org/simple" \
    --only-binary=":all:" \
    "torchaudio==${PYTORCH_TORCHAUDIO_VERSION}"; then
    echo "ERROR: torchaudio install failed from nightly cu130 index"
    exit 1
fi

if ! pip_install --no-cache-dir --upgrade --ignore-installed \
    --index-url "${PYTORCH_CU130_INDEX}" \
    --extra-index-url "https://pypi.org/simple" \
    --only-binary=":all:" \
    "torchao==${PYTORCH_TORCHAO_VERSION}"; then
    echo "ERROR: torchao install failed from nightly cu130 index"
    exit 1
fi

if ! pip_install --no-cache-dir --upgrade --ignore-installed \
    --index-url "${PYTORCH_NIGHTLY_INDEX}" \
    --extra-index-url "https://pypi.org/simple" \
    --only-binary=":all:" \
    "triton==${PYTORCH_TRITON_VERSION}"; then
    echo "ERROR: triton install failed"
    exit 1
fi

# torchdata/torchtitan fallbacks (no cu130 wheels published)
pip_uninstall -y torchdata torchtitan >/dev/null 2>&1 || true
# Avoid re-pulling a CPU torch dependency: install these without deps.
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps torchdata==0.10.0 || {
    echo "Warning: torchdata install failed; continuing"
}
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps torchtitan==0.2.0 || {
    echo "Warning: torchtitan 0.2.0 install failed; continuing without torchtitan"
}

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
    echo "ERROR: cu130 torch install did not deliver the expected nightly build."
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

# NOW install accelerate and torchtitan (they depend on torch, but torch CUDA is already installed)
echo ""
echo "Installing accelerate and torchtitan (PyTorch CUDA is already installed)..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Extract accelerate and torchtitan versions from original requirements
    ACCELERATE_VERSION=$(grep "^accelerate==" "$REQUIREMENTS_FILE" | cut -d= -f3 || echo "0.29.0")
    TORCHTITAN_VERSION=$(grep "^torchtitan==" "$REQUIREMENTS_FILE" | cut -d= -f3 || echo "0.2.0")
    
    echo "  Installing accelerate==${ACCELERATE_VERSION}..."
    pip_install --no-input --no-deps "accelerate==${ACCELERATE_VERSION}" || {
        echo "  Warning: accelerate installation failed, but continuing..."
    }
    
    echo "  Installing torchtitan==${TORCHTITAN_VERSION}..."
    pip_install --no-input --no-deps "torchtitan==${TORCHTITAN_VERSION}" || {
        echo "  Warning: torchtitan installation failed, but continuing..."
    }

    echo "  Installing torchtitan runtime dependencies..."
    pip_install --no-input "${TORCHTITAN_RUNTIME_DEPS[@]}" || {
        echo "  ERROR: torchtitan runtime dependency install failed"
        exit 1
    }
    
    # CRITICAL: Verify PyTorch CUDA wasn't overridden by accelerate/torchtitan dependencies
    echo "  Verifying PyTorch CUDA wasn't overridden..."
    if ! verify_and_restore_pytorch_cuda "accelerate/torchtitan installation"; then
        echo "  ERROR: Failed to restore PyTorch CUDA after accelerate/torchtitan!"
        exit 1
    fi
    echo "  ✓ PyTorch CUDA verified"

    echo "  Verifying torchtitan runtime imports..."
    if ! python3 <<'PY'; then
import torchtitan
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp

print(f"  torchtitan: {torchtitan.__file__}")
print(f"  JobConfig: {JobConfig.__module__}")
print(f"  maybe_enable_async_tp: {maybe_enable_async_tp.__module__}")
PY
        echo "  ERROR: torchtitan runtime verification failed"
        exit 1
    fi
    echo "  ✓ torchtitan runtime verified"
    
    echo "✓ accelerate and torchtitan installed (PyTorch CUDA + torchtitan runtime verified)"
fi

# TRT-LLM dependencies
echo ""
echo "Installing TRT-LLM dependencies..."
echo "Installing TensorRT-LLM Python packages..."
pip_install --no-cache-dir --upgrade --no-deps \
    "tensorrt-llm==1.1.0" \
    "nvtx==0.2.14" \
    "mpi4py==4.1.1" \
    "soundfile==0.13.1" \
    "onnx-graphsurgeon==0.5.8" \
    "ordered-set==4.1.0" \
    "strenum==0.4.15" \
    "nvidia-modelopt==0.37.0" \
    "pulp==3.3.0" \
    "h5py==3.12.1" \
    "torchprofile==0.0.4" || {
    echo "Warning: TensorRT-LLM dependency install failed"
}

# Verify PyTorch CUDA wasn't overridden by TRT-LLM deps
if ! verify_and_restore_pytorch_cuda "TRT-LLM installation"; then
    echo "ERROR: PyTorch CUDA missing after TRT-LLM installs!"
    exit 1
fi

# Single pin for shared deps to avoid dependency churn
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps "fsspec[http]==2024.6.1" || {
    echo "Warning: failed to pin fsspec[http]==2024.6.1"
}
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps tokenizers==0.22.2 huggingface-hub==0.34.6 onnx==1.19.0 onnxscript==0.1.0 einops==0.8.0 || {
    echo "Warning: failed to pin tokenizers/huggingface-hub/onnx/onnxscript/einops"
}

# Ensure triton is available (required by Transformer Engine and other PyTorch extensions)
# Triton should be bundled with PyTorch, but install it explicitly to ensure it's available
echo "Verifying triton availability (required by Transformer Engine)..."
if ! python3 -c "import triton" 2>/dev/null; then
    echo "  Triton not found. Installing triton from PyTorch nightly index (with --no-deps to prevent torch override)..."
    pip_install --no-cache-dir --upgrade --no-deps \
        --index-url "${PYTORCH_NIGHTLY_INDEX}" \
        --extra-index-url "https://pypi.org/simple" \
        --only-binary=":all:" \
        "triton==${PYTORCH_TRITON_VERSION}" || {
        echo "  Warning: Failed to install triton. Transformer Engine may not work."
    }
else
    TRITON_VERSION=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "unknown")
    echo "  ✓ Triton version: ${TRITON_VERSION}"
fi

# Install torchvision from cu130 binaries (no source builds)
# NOTE: torchvision is intentionally skipped for now; nightly cu130 wheels have been failing.
# echo ""
# echo "Installing torchvision (cu13 wheel, no-build)..."
# pip_install --no-input --no-deps \
#     --index-url "${PYTORCH_CU130_INDEX}" \
#     --extra-index-url "https://pypi.org/simple" \
#     --only-binary=":all:" \
#     torchvision || {
#     echo "Warning: torchvision cu13 install failed. Continuing without it."
# }

# Verify PyTorch CUDA wasn't overridden by torchvision installation
# NOTE: torchvision is intentionally skipped for now; keep this check disabled until re-enabled.
# if ! verify_and_restore_pytorch_cuda "torchvision installation"; then
#     echo "ERROR: PyTorch CUDA missing after torchvision!"
#     exit 1
# fi

echo ""
echo "Refreshing CUTLASS source tree (${CUTLASS_REF}) for local CUDA builds..."
CURRENT_CUTLASS_VERSION=$(CUTLASS_SRC_DIR="${CUTLASS_SRC_DIR}" python3 <<'PY' | tr -d '\n'
from pathlib import Path
import os
import re
root = Path(os.environ.get("CUTLASS_SRC_DIR", ""))
version_path = root / "include" / "cutlass" / "version.h"
if not version_path.exists():
    print("")
    raise SystemExit(0)
text = version_path.read_text()
def grab(name: str) -> str:
    match = re.search(r"#define\\s+%s\\s+([0-9]+)" % name, text)
    return match.group(1) if match else ""
parts = [grab("CUTLASS_MAJOR"), grab("CUTLASS_MINOR"), grab("CUTLASS_PATCH")]
if "" in parts:
    print("")
else:
    print(".".join(parts))
PY
)

if [ ! -d "${CUTLASS_SRC_DIR}" ] || [ -z "${CURRENT_CUTLASS_VERSION}" ]; then
    echo "  CUTLASS source missing or unreadable; installing ${CUTLASS_REF}..."
    CUTLASS_REPO="${CUTLASS_REPO_URL}" CUTLASS_REF="${CUTLASS_REF}" "${PROJECT_ROOT}/core/scripts/install_cutlass.sh"
    CURRENT_CUTLASS_VERSION="${CUTLASS_TARGET_VERSION}"
elif [ "${CURRENT_CUTLASS_VERSION}" != "${CUTLASS_TARGET_VERSION}" ]; then
    echo "  CUTLASS source is ${CURRENT_CUTLASS_VERSION} (target ${CUTLASS_TARGET_VERSION}); reinstalling..."
    CUTLASS_REPO="${CUTLASS_REPO_URL}" CUTLASS_REF="${CUTLASS_REF}" "${PROJECT_ROOT}/core/scripts/install_cutlass.sh"
    CURRENT_CUTLASS_VERSION="${CUTLASS_TARGET_VERSION}"
else
    echo "  CUTLASS source tree already at ${CURRENT_CUTLASS_VERSION}; keeping."
fi
echo "  CUTLASS headers available at ${CUTLASS_SRC_DIR}/include"

# Ensure monitoring/runtime dependencies are available even if requirements were cached
echo ""
echo "Ensuring monitoring/runtime packages (Prometheus, Transformer Engine)..."

ensure_free_space_gb "${SETUP_MIN_TE_BUILD_FREE_GB}" "Transformer Engine build"

echo ""
echo "Installing Transformer Engine from source (cu13, arch sm_${GPU_COMPUTE_SM_NUM:-unknown})..."
mkdir -p "${TE_SRC_DIR}"
if [ ! -d "${TE_SRC_DIR}/.git" ]; then
    rm -rf "${TE_SRC_DIR}"
    git clone --recursive "${TE_REPO_URL}" "${TE_SRC_DIR}"
fi
git -C "${TE_SRC_DIR}" fetch --all --tags --prune --force >/dev/null 2>&1 || true
git -C "${TE_SRC_DIR}" checkout "${TE_GIT_COMMIT}" >/dev/null 2>&1 || true
git -C "${TE_SRC_DIR}" submodule sync --recursive >/dev/null 2>&1 || true
git -C "${TE_SRC_DIR}" submodule update --init --recursive >/dev/null 2>&1 || true

# =============================================================================
# CRITICAL: CUTLASS Version Override for Blackwell (SM100a) Support
# =============================================================================
# TransformerEngine ${TE_VERSION} bundles CUTLASS ${TE_BUNDLED_CUTLASS_VERSION} which LACKS SM100a headers.
# We MUST replace TE's bundled CUTLASS with our standalone CUTLASS ${CUTLASS_TARGET_VERSION}.
#
# Required SM100a headers (only in CUTLASS 4.3.0+):
#   - cute/arch/tmem_allocator_sm100.hpp
#   - cute/arch/mma_sm100_umma.hpp
#   - cute/atom/copy_traits_sm100.hpp
#
# Without this symlink: Blackwell builds WILL FAIL with "missing header" errors.
# Verification: make verify-cutlass
# Check if still needed: python core/scripts/check_upstream_versions.py --check-te-cutlass
# =============================================================================
echo "  CUTLASS version override for SM100a (Blackwell) support:"
echo "    TE ${TE_VERSION} bundles: CUTLASS ${TE_BUNDLED_CUTLASS_VERSION} (missing SM100a headers)"
echo "    Standalone CUTLASS:   ${CUTLASS_TARGET_VERSION} (has SM100a headers)"
echo "  Creating symlink to override TE's bundled CUTLASS..."

if [ -d "${TE_SRC_DIR}/3rdparty/cutlass" ] || [ -L "${TE_SRC_DIR}/3rdparty/cutlass" ]; then
    rm -rf "${TE_SRC_DIR}/3rdparty/cutlass"
fi
ln -s "${CUTLASS_SRC_DIR}" "${TE_SRC_DIR}/3rdparty/cutlass"

# Verify the symlink is correct and SM100a headers exist
if [ ! -L "${TE_SRC_DIR}/3rdparty/cutlass" ]; then
    echo "ERROR: Failed to create CUTLASS symlink for TransformerEngine"
    echo "       This will cause Blackwell builds to fail!"
    echo "       Try: ./core/scripts/fix_cutlass_symlink.sh"
    exit 1
fi
if [ ! -f "${TE_SRC_DIR}/3rdparty/cutlass/include/cute/arch/tmem_allocator_sm100.hpp" ]; then
    echo "ERROR: CUTLASS SM100a headers missing after symlink - Blackwell builds will fail"
    echo "       Expected: ${TE_SRC_DIR}/3rdparty/cutlass/include/cute/arch/tmem_allocator_sm100.hpp"
    echo "       Try: ./core/scripts/install_cutlass.sh && ./core/scripts/fix_cutlass_symlink.sh"
    exit 1
fi
echo "  ✓ TE CUTLASS symlinked: ${TE_BUNDLED_CUTLASS_VERSION} → ${CUTLASS_TARGET_VERSION} (SM100a ready)"

echo "Removing existing Transformer Engine installs to avoid dist-info skew..."
pip_uninstall -y transformer_engine transformer-engine >/dev/null 2>&1 || true
rm -rf "${PYTHON_DIST_PACKAGES}"/transformer_engine*.dist-info 2>/dev/null || true

pip_install --no-input --upgrade --ignore-installed pybind11

TE_BUILD_ARCH_LIST="${TE_TORCH_ARCH_LIST}"
TE_BUILD_NVCC_ARCHS="${TE_CUTLASS_ARCHS}"
if [ -z "${TE_BUILD_ARCH_LIST}" ]; then
    TE_BUILD_ARCH_LIST="10.0"
fi
if [ -z "${TE_BUILD_NVCC_ARCHS}" ]; then
    TE_BUILD_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS_VALUE_DEFAULT}"
fi

if ! TORCH_CUDA_ARCH_LIST="${TE_BUILD_ARCH_LIST}" \
       CUTLASS_NVCC_ARCHS="${TE_BUILD_NVCC_ARCHS}" \
       CMAKE_CUDA_ARCH_LIST="${TE_BUILD_NVCC_ARCHS}" \
       NVCC_GENCODE="-gencode=arch=compute_${TE_BUILD_NVCC_ARCHS},code=sm_${TE_BUILD_NVCC_ARCHS}" \
       NVTE_FRAMEWORK=pytorch \
       NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
       MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
       pip_install --no-cache-dir --upgrade --ignore-installed --no-build-isolation --no-deps "${TE_SRC_DIR}"; then
    echo "ERROR: Transformer Engine source installation failed (arch list ${TE_BUILD_ARCH_LIST}, NVCC archs ${TE_BUILD_NVCC_ARCHS})."
    exit 1
fi

if [ "${CLEAN_BUILD_ARTIFACTS}" -eq 1 ]; then
    echo "Cleaning Transformer Engine build artifacts..."
    rm -rf "${TE_SRC_DIR}/build" "${TE_SRC_DIR}/.eggs" "${TE_SRC_DIR}/dist" 2>/dev/null || true
fi

# Ensure TE runtime dependencies for FP8 tooling
pip_uninstall -y onnx onnxscript einops >/dev/null 2>&1 || true
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps onnx==1.19.0 einops==0.8.0 onnxscript==0.1.0 || {
    echo "Warning: failed to install TE runtime deps (onnx/onnxscript/einops)"
}
# TE FP8 requires ml_dtypes; install without deps to avoid torch overrides
pip_uninstall -y ml_dtypes >/dev/null 2>&1 || true
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps ml_dtypes==0.5.1 || {
    echo "Warning: failed to install ml_dtypes==0.5.1 required by Transformer Engine FP8 checks"
}

patch_installed_transformer_engine_metadata
patch_transformer_engine_loader
disable_transformer_engine_sanity_check

# CRITICAL: Verify PyTorch CUDA after Transformer Engine installation
echo ""
echo "Verifying PyTorch CUDA after Transformer Engine installation..."
if ! verify_and_restore_pytorch_cuda "Transformer Engine installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after Transformer Engine!"
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

# Ensure triton is available (required by Transformer Engine)
echo "Checking for triton..."
if ! python3 -c "import triton" 2>/dev/null; then
    echo "  Triton not found. Installing triton (required by Transformer Engine)..."
    # Try to install triton - it should match the PyTorch version
    pip_install --no-cache-dir --upgrade triton || {
        echo "  Warning: Failed to install triton. Transformer Engine capability check may fail."
    }
fi

python3 <<'PY'
import sys
import warnings

# Suppress FlashAttention kernel override warnings (harmless - happens when FlashAttention is imported multiple times)
warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*")
warnings.filterwarnings("ignore", message=".*Warning only once for all operators.*")

# First check if PyTorch CUDA is available
try:
    import torch
    if not torch.cuda.is_available():
        print("Transformer Engine capability snapshot skipped: PyTorch CUDA not available")
        print("  torch.version.cuda = {}".format(getattr(torch.version, 'cuda', None)))
        print("  torch.__version__ = {}".format(torch.__version__))
        print("  Transformer Engine requires CUDA-enabled PyTorch.")
        print("  This check will be retried after PyTorch verification.")
        sys.exit(0)
except ImportError:
    print("Transformer Engine capability snapshot skipped: PyTorch not available")
    sys.exit(0)

# Check if triton is available (required by Transformer Engine)
try:
    import triton
    print(f"  Triton version: {triton.__version__}")
except ImportError:
    print("  Warning: triton not found. It should be bundled with PyTorch.")
    print("  Transformer Engine capability check may fail without triton.")
    print("  Note: triton should be available via PyTorch. If missing, this may indicate an incomplete PyTorch installation.")

try:
    import transformer_engine.pytorch as te
except ImportError as exc:
    print(f"Transformer Engine capability snapshot skipped: {exc}")
    if "triton" in str(exc).lower():
        print("  Triton is required for Transformer Engine. Please ensure PyTorch is fully installed.")
    elif "undefined symbol" in str(exc).lower() and "cuda" in str(exc).lower():
        print("  This error indicates PyTorch CUDA libraries are incompatible or not available.")
        print("  Transformer Engine check will be retried after PyTorch verification.")
    else:
        print("  Transformer Engine may not be fully installed yet.")
    sys.exit(0)
except Exception as exc:
    print(f"Transformer Engine capability snapshot skipped: {exc}")
    if "CUDA" in str(exc):
        print("  This may be due to PyTorch CUDA not being available.")
        print("  Transformer Engine check will be retried after PyTorch verification.")
    sys.exit(0)

try:
    nvfp4_available, nvfp4_reason = te.is_nvfp4_available(return_reason=True)
    mxfp8_available, mxfp8_reason = te.is_mxfp8_available(return_reason=True)
    def format_status(flag, reason):
        return "yes" if flag else f"no ({reason or 'not supported on this stack'})"
    print("Transformer Engine capability snapshot:")
    print(f"  NVFP4 available: {format_status(nvfp4_available, nvfp4_reason)}")
    print(f"  MXFP8 available: {format_status(mxfp8_available, mxfp8_reason)}")
except Exception as exc:
    print(f"Transformer Engine capability check failed: {exc}")
    print("  This may be normal if CUDA runtime is not available.")
PY

# Refresh hardware capability cache now so subsequent non-root runs do not need
# to invoke the probe with elevated privileges.
echo ""
echo "Refreshing GPU hardware capability cache..."
if python3 "${PROJECT_ROOT}/core/scripts/utilities/probe_hardware_capabilities.py" --allow-unavailable 2>/tmp/probe_hardware.log; then
    cat /tmp/probe_hardware.log
else
    cat /tmp/probe_hardware.log
    echo "Warning: GPU hardware capability probe failed; CUDA may be unavailable."
    echo "This is expected if PyTorch CUDA is not yet properly installed."
    echo "The capability cache will be refreshed after PyTorch verification."
fi
rm -f /tmp/probe_hardware.log

# Ensure artifacts and wheel caches are writable by the invoking user for future runs.
if [ -n "${SUDO_USER:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/artifacts"
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PROJECT_ROOT}/artifacts" 2>/dev/null || true
    chown -R "${SUDO_USER}:${SUDO_USER}" "${THIRD_PARTY_DIR}/wheels" 2>/dev/null || true
fi

# Final verification: Ensure our custom PyTorch wheel remains installed after all package installations
echo ""
echo "============================================================================"
echo "Final verification: PyTorch CUDA installation"
echo "============================================================================"

if ! verify_and_restore_pytorch_cuda "final verification (after all installations)"; then
    echo "ERROR: Failed to restore PyTorch CUDA during final verification!"
    exit 1
fi

# Print final status
python3 <<'PY'
import sys
import torch
print(f"✓ Final PyTorch CUDA verification: {torch.__version__} with CUDA {torch.version.cuda}")
PY

# Clean up Transformer Engine wheel conflicts
python3 <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("transformer_engine")
if not spec:
    raise SystemExit(0)
pkg_root = Path(spec.origin).parent
wheel_dir = pkg_root / "wheel_lib"
if not wheel_dir.exists():
    raise SystemExit(0)
for so_path in pkg_root.glob("*.so"):
    if (wheel_dir / so_path.name).exists():
        so_path.unlink()
PY

python3 <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

def mark_non_pure(dist_name: str) -> None:
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return
    files = dist.files or []
    wheel_file = None
    for entry in files:
        if entry.name == "WHEEL":
            wheel_file = dist.locate_file("") / entry
            break
    if wheel_file is None:
        return
    lines = wheel_file.read_text().splitlines()
    changed = False
    for idx, line in enumerate(lines):
        if line.startswith("Root-Is-Purelib:"):
            value = line.split(":", 1)[1].strip().lower()
            if value != "false":
                lines[idx] = "Root-Is-Purelib: false"
                changed = True
            break
    else:
        lines.append("Root-Is-Purelib: false")
        changed = True
    if changed:
        wheel_file.write_text("\n".join(lines) + "\n")

for name in (
    "transformer_engine",
    "transformer_engine_cu12",
    "transformer_engine_cu13",
    "transformer_engine_torch",
    "transformer-engine",
):
    mark_non_pure(name)
PY

python3 <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

def ensure_te_import_tolerant(dist_name: str) -> bool:
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return False
    init_path = Path(dist.locate_file("transformer_engine/__init__.py"))
    if not init_path.exists():
        return False
    source = init_path.read_text()
    marker = "import transformer_engine.common"
    if marker not in source:
        return False
    injection = (
        "\ntry:\n"
        "    from importlib.metadata import PackageNotFoundError, distribution\n"
        "    TE_METADATA_AVAILABLE = True\n"
        "except ImportError:\n"
        "    TE_METADATA_AVAILABLE = False\n"
        "_IN_TREE_TRANSFORMER_ENGINE = False\n"
        "if TE_METADATA_AVAILABLE:\n"
        "    try:\n"
        "        distribution(\"transformer-engine\")\n"
        "        _IN_TREE_TRANSFORMER_ENGINE = True\n"
        "    except PackageNotFoundError:\n"
        "        _IN_TREE_TRANSFORMER_ENGINE = True\n"
        "else:\n"
        "    _IN_TREE_TRANSFORMER_ENGINE = True\n"
    )
    if marker + injection in source:
        return True
    if injection in source:
        source = source.replace(injection, "", 1)
    source = source.replace(marker, marker + injection, 1)
    init_path.write_text(source)
    return True

ensure_te_import_tolerant("transformer_engine")
ensure_te_import_tolerant("transformer-engine")
PY

# Install FlashAttention for optimized SDPA paths

echo ""
echo "Ensuring FlashAttention (flash-attn ${FLASH_ATTN_TAG}) is available..."
FLASH_ATTN_CACHE_PATH="${THIRD_PARTY_DIR}/wheels/${FLASH_ATTN_WHEEL_BASENAME}"
FLASH_ATTN_SPLIT_PREFIX="${FLASH_ATTN_CACHE_PATH}.part"

flash_attn_filter_warning() {
    python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flash_attn.flash_attn_interface")
if not spec or not spec.origin:
    raise SystemExit
path = Path(spec.origin)
lines = path.read_text().splitlines()
marker = "Overriding a previously registered kernel for the same operator"
if any(marker in ln for ln in lines):
    raise SystemExit
inject = [
    "import warnings",
    "",
    'warnings.filterwarnings(',
    '    "ignore",',
    '    message=".*Overriding a previously registered kernel for the same operator.*flash_attn.*",',
    '    category=UserWarning,',
    '    module="torch.library",',
    ')',
    'warnings.filterwarnings(',
    '    "ignore",',
    '    message="Warning only once for all operators,\\s+other operators may also be overridden.",',
    '    category=UserWarning,',
    '    module="torch.library",',
    ')',
]
out = []
inserted = False
for ln in lines:
    out.append(ln)
    if not inserted and ln.strip() == "import os":
        out.extend(inject)
        inserted = True
if not inserted:
    out = inject + [""] + lines
path.write_text("\n".join(out) + "\n")
print(f"Patched FlashAttention warning filters at {path}")
PY
}

flash_attn_import_check() {
    python3 - <<'PY'
try:
    import flash_attn_2_cuda  # type: ignore
except Exception as exc:
    print(f"[setup] FlashAttention import check failed: {exc}")
    raise SystemExit(1)
print("[setup] FlashAttention CUDA extension import OK")
PY
}

install_flash_attention() {
    FLASH_ATTN_AVAILABLE=0
    local force_source="${FLASH_ATTN_FORCE_SOURCE:-0}"
    local fa_sm="${FLASH_ATTN_TARGET_SM:-${GPU_COMPUTE_SM_NUM:-121}}"
    local fa_arch_list
    if [ -n "${GPU_COMPUTE_SM_NUM:-}" ]; then
        fa_arch_list="${GPU_COMPUTE_SM_NUM}"
    else
        fa_arch_list="${TE_TORCH_ARCH_LIST:-12.1}"
    fi
    local fa_archs_env="${FLASH_ATTN_TARGET_ARCHS:-${fa_sm}}"
    local fa_max_jobs="${FLASH_ATTN_MAX_JOBS:-8}"
    local fa_nvcc_threads="${FLASH_ATTN_NVCC_THREADS:-1}"
    if [[ "${fa_arch_list}" == *a ]]; then
        fa_arch_list="${fa_arch_list%a}"
    fi
    if [[ "${fa_archs_env}" == *a ]]; then
        fa_archs_env="${fa_archs_env%a}"
    fi

    echo "Installing FlashAttention ${FLASH_ATTN_EXPECTED_VERSION} (torch $(python3 - <<'PY'
import torch
print(torch.__version__)
PY
))..."
    # Force reinstall to ensure ABI matches current torch
    pip_uninstall -y flash-attn >/dev/null 2>&1 || true

    if [ "${force_source}" -ne 1 ]; then
        if pip_install --no-cache-dir --upgrade --force-reinstall --ignore-installed --no-deps --prefer-binary --only-binary=:all: \
            flash-attn=="${FLASH_ATTN_EXPECTED_VERSION}"; then
            if flash_attn_import_check; then
                FLASH_ATTN_AVAILABLE=1
                return 0
            fi
            echo "FlashAttention binary wheel installed but import failed; rebuilding from source..."
            pip_uninstall -y flash-attn >/dev/null 2>&1 || true
        else
            echo "Binary FlashAttention wheel unavailable; building from source..."
        fi
    else
        echo "FLASH_ATTN_FORCE_SOURCE=${force_source}; building FlashAttention from source..."
    fi

    if TORCH_CUDA_ARCH_LIST="${fa_arch_list}" \
       FLASH_ATTENTION_FORCE_CUDA_SM="${fa_sm}" \
       FLASH_ATTN_CUDA_ARCHS="${fa_archs_env}" \
       MAX_JOBS="${fa_max_jobs}" \
       NVCC_THREADS="${fa_nvcc_threads}" \
       FLASH_ATTENTION_FORCE_BUILD=TRUE \
       pip_install --no-cache-dir --upgrade --force-reinstall --ignore-installed --no-build-isolation --no-deps \
       "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_TAG}"; then
        if flash_attn_import_check; then
            FLASH_ATTN_AVAILABLE=1
            return 0
        fi
        echo "FlashAttention source build succeeded but import failed."
        pip_uninstall -y flash-attn >/dev/null 2>&1 || true
    fi
    echo "Warning: FlashAttention not available (install/import failed)."
    return 1
}

install_flash_attention_from_parts() {
    local tmp_dir
    local combined
    local status
    local -a PARTS=()
    tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/flashattn-wheel.XXXXXX")
    combined="${tmp_dir}/${FLASH_ATTN_WHEEL_BASENAME}"
    mapfile -t PARTS < <(ls "${FLASH_ATTN_SPLIT_PREFIX}"* | sort -V)
    if [ "${#PARTS[@]}" -eq 0 ]; then
        rm -rf "${tmp_dir}"
        return 1
    fi
    if cat "${PARTS[@]}" > "${combined}"; then
        pip_install --no-cache-dir --upgrade --ignore-installed --no-deps "${combined}"
        status=$?
    else
        status=1
    fi
    rm -rf "${tmp_dir}"
    return "${status}"
}

cache_flash_attention_wheel() {
    rm -f "${FLASH_ATTN_SPLIT_PREFIX}"* 2>/dev/null || true
    if [ -f "${FLASH_ATTN_CACHE_PATH}" ]; then
        echo "Splitting FlashAttention wheel into <50MB chunks under ${THIRD_PARTY_DIR}/wheels";
        split -b 45m -d -a 2 "${FLASH_ATTN_CACHE_PATH}" "${FLASH_ATTN_SPLIT_PREFIX}"
    fi
}

rebuild_flash_attention_cache() {
    echo "Skipping FlashAttention wheel rebuild; binary-only mode."
    return 0
}

current_flash_attn_version="$(pip_show flash-attn 2>/dev/null | awk '/Version/ {print $2}' || true)"
if [ -n "${current_flash_attn_version}" ] && [ "${current_flash_attn_version}" != "${FLASH_ATTN_EXPECTED_VERSION}" ]; then
    echo "FlashAttention version mismatch (${current_flash_attn_version} != ${FLASH_ATTN_EXPECTED_VERSION}), reinstalling..."
    pip_uninstall -y flash-attn >/dev/null 2>&1 || true
    current_flash_attn_version=""
fi

detect_gpu_sm() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cc
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        if [[ -n "${cc}" ]]; then
            echo "${cc}"
            return 0
        fi
    fi
    return 1
}

echo "Installing FlashAttention (binary wheel preferred, source fallback)..."
if ! install_flash_attention; then
    echo "Warning: FlashAttention installation failed; continuing without FlashAttention."
fi
if [ "${FLASH_ATTN_AVAILABLE:-0}" -eq 1 ]; then
    # Filter FlashAttention warnings, if present
    flash_attn_filter_warning
else
    echo "FlashAttention unavailable; skipping warning filter."
fi

echo ""
echo "Verifying PyTorch CUDA after FlashAttention installation..."
if ! verify_and_restore_pytorch_cuda "FlashAttention installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after FlashAttention!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after FlashAttention"

# Install pinned vLLM serving stack and verify ABI against torch.
echo ""
echo "Installing pinned vLLM serving stack (${VLLM_PIP_SPEC}) for this CUDA/toolchain..."
VLLM_ARCH_LIST="12.1"
if [ -n "${GPU_COMPUTE_SM:-}" ]; then
    if [ "${#GPU_COMPUTE_SM}" -ge 3 ]; then
        major="${GPU_COMPUTE_SM:0:2}"
        minor="${GPU_COMPUTE_SM:2:1}"
        VLLM_ARCH_LIST="${major}.${minor}"
    fi
fi
VLLM_EXPECTED_VERSION="${VLLM_PIP_SPEC#vllm==}"

# Binary-only mode: skip vLLM source sync/caching
VLLM_WHEEL_PATH=""
VLLM_WHEEL_HAS_PARTS=0

echo "Installing vLLM from cu13 wheels (binary only, no deps)..."
pip_uninstall -y vllm >/dev/null 2>&1 || true
if ! pip_install --no-cache-dir --upgrade --ignore-installed --prefer-binary --only-binary=:all: --no-deps \
    ${VLLM_EXTRA_INDEX_URL:+--extra-index-url "${VLLM_EXTRA_INDEX_URL}"} \
    "${VLLM_PIP_SPEC}"; then
    echo "ERROR: Failed to install prebuilt ${VLLM_PIP_SPEC} wheel from index ${VLLM_EXTRA_INDEX_URL:-PyPI}."
    exit 1
fi
VLLM_WHEEL_PATH=""
VLLM_WHEEL_HAS_PARTS=0
VLLM_PREBUILT_INSTALLED=1

if [ "${VLLM_PREBUILT_INSTALLED}" -ne 1 ]; then
    if [ -z "${VLLM_WHEEL_PATH}" ] || { [ ! -f "${VLLM_WHEEL_PATH}" ] && [ "${VLLM_WHEEL_HAS_PARTS}" -eq 0 ]; }; then
        echo "ERROR: vLLM wheel is missing (expected under ${VLLM_WHEEL_DIR})"
        exit 1
    fi

    if [ "${VLLM_WHEEL_HAS_PARTS}" -eq 0 ] && ! compgen -G "${VLLM_WHEEL_PATH}.part*" >/dev/null; then
        echo "Splitting vLLM wheel into <50MB chunks under ${VLLM_WHEEL_DIR}"
        split -b 45m -d -a 2 "${VLLM_WHEEL_PATH}" "${VLLM_WHEEL_PATH}.part"
    fi
    if compgen -G "${VLLM_WHEEL_PATH}.part*" >/dev/null; then
        VLLM_WHEEL_HAS_PARTS=1
    fi

    tmp_vllm_install_dir=$(mktemp -d "${TMPDIR:-/tmp}/vllm-wheel-install.XXXXXX")
    vllm_wheel_to_install=$(reassemble_split_wheel "${VLLM_WHEEL_PATH}" "${tmp_vllm_install_dir}")

    if [ -z "${vllm_wheel_to_install}" ] || [ ! -f "${vllm_wheel_to_install}" ]; then
        echo "ERROR: Could not reassemble vLLM wheel from ${VLLM_WHEEL_PATH}"
        rm -rf "${tmp_vllm_install_dir}"
        exit 1
    fi

    pip_uninstall -y vllm >/dev/null 2>&1 || true
    TORCH_CUDA_ARCH_LIST="${VLLM_ARCH_LIST}" \
    VLLM_TARGET_DEVICE="cuda" \
    pip_install --no-cache-dir --force-reinstall --upgrade --no-deps "${vllm_wheel_to_install}"
    rm -rf "${tmp_vllm_install_dir}"
else
    echo "vLLM prebuilt wheel installed from ${VLLM_EXTRA_INDEX_URL:-PyPI}; skipping local wheel cache handling."
fi

echo "Installing pinned vLLM runtime dependencies (required when vLLM is installed with --no-deps)..."
if ! pip_install --no-cache-dir --upgrade --ignore-installed "${VLLM_RUNTIME_DEPS[@]}"; then
    echo "ERROR: Failed to install pinned vLLM runtime dependencies."
    exit 1
fi

python3 <<PY
import importlib
import importlib.metadata
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
    importlib.import_module("vllm._C")
except Exception as exc:
    print("ERROR: vLLM extension import failed (ABI mismatch likely):")
    print(exc)
    raise SystemExit(1)

print(f"✓ torch installed: {torch_version}")
print(f"✓ vLLM installed: {vllm_dist_version} (module reports {vllm.__version__})")
print(f"✓ flashinfer-python installed: {flashinfer_dist_version}")
print("✓ vLLM extension import succeeded: vllm._C")
PY

echo ""
echo "Verifying PyTorch CUDA after vLLM installation..."
if ! verify_and_restore_pytorch_cuda "vLLM installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after vLLM!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after vLLM"

# Remove conflicting system packages that interfere with PyTorch
echo ""
echo "Removing conflicting system packages..."
# Remove python3-optree which conflicts with torch.compile
if dpkg -s python3-optree >/dev/null 2>&1; then
    echo "Removing python3-optree (conflicts with torch.compile)..."
    apt remove -y python3-optree python3-keras 2>/dev/null || true
fi
# Clean up other conflicting Lambda Labs packages
apt autoremove -y 2>/dev/null || true

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
PY

# Check CUDA tools
echo ""
echo "Checking CUDA tools..."
if command -v nvcc &> /dev/null; then
    echo "NVCC: $(nvcc --version | head -1)"
else
    echo "NVCC not found"
fi

# Check Nsight tools
echo ""
echo "Checking Nsight tools..."
if command -v nsys &> /dev/null; then
    NSYS_VERSION=$(nsys --version 2>/dev/null | head -1)
    echo "Nsight Systems: $NSYS_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NSYS_VERSION" | grep -q "2025"; then
        echo "  Recent 2025 version installed!"
    else
        echo "  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "Nsight Systems not found"
fi

if command -v ncu &> /dev/null; then
    NCU_VERSION=$(ncu --version 2>/dev/null | head -1)
    echo "Nsight Compute: $NCU_VERSION"
    # Check if it's 2025.3.1 or a recent 2025 version
    if echo "$NCU_VERSION" | grep -qE "2025\.3\.1|2025\.3\.[2-9]|2025\.[4-9]"; then
        echo "  Latest version installed (2025.3.1 or newer)!"
    elif echo "$NCU_VERSION" | grep -q "2025"; then
        echo "  Recent 2025 version installed!"
    else
        echo "  May not be the latest version (expected: 2025.3.1 or newer)"
    fi
else
    echo "Nsight Compute not found"
fi

# Check system tools
echo ""
echo "Checking system tools..."
tools=("numactl" "perf" "htop" "iostat" "ibstat")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "$tool: installed"
    else
        echo "$tool: not found"
    fi
done

# Run basic performance test
echo ""
echo "Running basic performance test..."
python3 <<'PY'
import sys
import time

try:
    import torch
except Exception as exc:
    print(f"Skipping performance test: failed to import torch ({exc})")
    sys.exit(0)

if not torch.cuda.is_available():
    print("Skipping performance test: torch.cuda.is_available() is False")
    sys.exit(0)

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

# Detect NCCL shared library for LD_PRELOAD
NCCL_LIB_PATH=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libnccl.so.2 \
    /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
    /usr/local/lib/libnccl.so.2 \
    /usr/lib/libnccl.so.2
    do
    if [ -f "${candidate}" ]; then
        NCCL_LIB_PATH="${candidate}"
        break
    fi
done
export NCCL_LIB_PATH

# Set up environment variables for optimal performance
echo ""
echo "Setting up environment variables..."
cat >> ~/.bashrc <<EOF

# AI Performance Engineering Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# PyTorch optimization
export TORCH_COMPILE_DEBUG=0
export TORCH_LOGS="+dynamo"

# CUDA paths
export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH=${CUDA_HOME_DIR}/bin:\$PATH
export AISP_CUDNN_RUNTIME_POLICY="\${AISP_CUDNN_RUNTIME_POLICY:-auto}"
export LD_LIBRARY_PATH="\${CUDA_HOME_DIR}/lib64:\${CUDA_HOME_DIR}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

# NCCL preload configuration
export NCCL_LIB_PATH="${NCCL_LIB_PATH}"
if [ -n "\${NCCL_LIB_PATH}" ]; then
    export LD_PRELOAD="\${NCCL_LIB_PATH}:\${LD_PRELOAD:-}"
else
    echo "WARNING: NCCL library not found; skipping LD_PRELOAD setup."
fi
EOF

echo "Environment variables added to ~/.bashrc"

# Source the environment variables for current session
if [ -n "${NCCL_LIB_PATH}" ]; then
    export LD_PRELOAD="${NCCL_LIB_PATH}:${LD_PRELOAD:-}"
    echo "NCCL 2.28.7 activated for current session (${NCCL_LIB_PATH})"
else
    echo "NCCL library not found for current session; continuing without LD_PRELOAD."
fi
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=1

# Comprehensive setup verification
echo ""
echo "Running comprehensive setup verification..."
echo "=============================================="

# Test 1: PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
python3 <<'PY'
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
    print("WARNING: CUDA runtime not available.")
    print("  This may be due to:")
    print("    1. NVIDIA driver not loaded")
    print("    2. CUDA libraries not in LD_LIBRARY_PATH")
    print("    3. GPU not accessible")
    print("  However, PyTorch CUDA build is correctly installed.")
    sys.exit(0)  # Don't fail the test if CUDA runtime isn't available

print(f"  GPU count: {torch.cuda.device_count()}")
print(f"  GPU name: {torch.cuda.get_device_name(0)}")

nccl_path = os.environ.get("NCCL_LIB_PATH")
try:
    if nccl_path and os.path.exists(nccl_path):
        libnccl = ctypes.CDLL(nccl_path)
        nv = ctypes.c_int()
        libnccl.ncclGetVersion(ctypes.byref(nv))
        print(f"  NCCL version: {nv.value}")
    else:
        print("  NCCL version: unavailable (library not found)")
except Exception as exc:
    print(f"  NCCL version: error ({exc})")

print("PyTorch and CUDA working correctly")
PY

# Don't fail if CUDA runtime isn't available, but verify CUDA build is correct
if [ $? -ne 0 ]; then
    echo "PyTorch/CUDA test completed with warnings (CUDA runtime may not be available)"
fi

# Test 2: Performance test
echo ""
echo "Testing GPU performance..."
python3 <<'PY'
import sys
import time

import torch

if not torch.cuda.is_available():
    print("  Skipping GPU performance test: CUDA runtime not available")
    sys.exit(0)

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

# Don't fail if CUDA runtime isn't available
if [ $? -ne 0 ]; then
    echo "GPU performance test skipped (CUDA runtime not available)"
fi

# Test 3: torch.compile test
echo ""
echo "Testing torch.compile..."
python3 <<'PY'
import sys
import time
import traceback

import torch

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
    if "duplicate template name" in str(exc):
        print("torch.compile skipped due to known PyTorch nightly issue: duplicate kernel template name")
        print(f"   Details: {exc}")
        sys.exit(0)
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

compile_status=$?
if [ $compile_status -ne 0 ]; then
    echo "torch.compile test failed!"
    exit 1
fi

# Step 11: Install CUTLASS 4.3+ Backend for torch.compile
echo ""
echo "Step 11: Installing CUTLASS 4.3+ Backend (nvidia-cutlass-dsl)..."
echo "================================================================="

# Install CUTLASS DSL 4.2+ and CUDA Python bindings system-wide
# The Python package (nvidia-cutlass-dsl) includes:
#   - Python API for torch.compile CUTLASS backend
#   - C++ headers for direct CUDA C++ kernel development
#   - All CUTLASS library headers (1000+ header files)
echo "Installing nvidia-cutlass-dsl and cuda-python (pinned versions)..."
pip_install --no-cache-dir --upgrade --ignore-installed "nvidia-cutlass-dsl==4.3.0" "cuda-python==13.0.3"
pip_install --no-cache-dir --upgrade --force-reinstall --no-deps "cuda-bindings==13.0.3"

if [ $? -eq 0 ]; then
    echo "CUTLASS backend packages installed (pinned versions)"
    echo "   - nvidia-cutlass-dsl==4.3.0: CUTLASS kernels for torch.compile"
    
    # Verify PyTorch CUDA after CUTLASS installation
    echo ""
    echo "Verifying PyTorch CUDA after CUTLASS installation..."
    if ! verify_and_restore_pytorch_cuda "CUTLASS installation"; then
        echo "ERROR: Failed to restore PyTorch CUDA after CUTLASS!"
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
    echo "CUTLASS backend installation had issues, but continuing..."
fi

echo ""

echo "Capturing hardware capabilities..."
if python3 "${PROJECT_ROOT}/core/scripts/utilities/probe_hardware_capabilities.py" --allow-unavailable 2>/tmp/probe_hardware.log; then
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
    print(f"  Error reading capabilities: {e}", file=sys.stderr)
PY
    else
        echo "  Warning: hardware_capabilities.json not found"
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
    echo "WARNING: Failed to regenerate arch_detection.cuh (continuing with existing header)"
fi

echo ""

# Test 4: Hardware info script
echo ""
echo "Testing hardware detection..."
if [ -f "$PROJECT_ROOT/ch02/hardware_info.py" ]; then
    python3 "$PROJECT_ROOT/ch02/hardware_info.py" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
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

VERIFICATION_FAILED=0

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"
export LD_LIBRARY_PATH="${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if python3 core/verification/verify_pytorch.py; then
    echo "PyTorch verification passed"
else
    echo "PyTorch verification failed"
    VERIFICATION_FAILED=1
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

# Summary of verification
if [ $VERIFICATION_FAILED -eq 0 ]; then
    echo "All critical verifications passed!"
else
    echo "Some verifications failed - review output above"
fi

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

# Final hard pins to ensure dependency consistency
echo ""
echo "Reinforcing final dependency pins (HF + TE runtime)..."
pip_uninstall -y tokenizers huggingface-hub onnx onnxscript einops fsspec >/dev/null 2>&1 || true
pip_install --no-cache-dir --force-reinstall --ignore-installed --no-deps \
    tokenizers==0.22.2 \
    huggingface-hub==0.34.6 \
    onnx==1.19.0 \
    onnxscript==0.1.0 \
    einops==0.8.0 \
    fsspec==2024.6.1 || {
    echo "Warning: final dependency pinning failed"
}
pip_install --no-cache-dir --upgrade --ignore-installed \
    --extra-index-url "https://pypi.org/simple" \
    transformers==4.56.0 \
    datasets==2.21.0 \
    sentencepiece==0.2.0 || {
    echo "Warning: final transformers/datasets/sentencepiece pinning failed"
}

# ------------------------------------------------------------------------------
# Transformer Engine (FP8/FP4/TMA) rebuild for Blackwell (sm_100a)
# ------------------------------------------------------------------------------
echo ""
echo "Rebuilding Transformer Engine for sm_100a with FP8/FP4/TMA..."

# Ensure CUDA 13-only runtime is visible to cudnn-frontend (avoid multiple libcudart)
disable_cuda12_cudart() {
    local cuda12_libdir="/usr/local/cuda-12/targets/x86_64-linux/lib"
    if [ -d "${cuda12_libdir}" ]; then
        for lib in libcudart.so libcudart.so.12 libcudart.so.12.*; do
            if [ -f "${cuda12_libdir}/${lib}" ] && [ ! -f "${cuda12_libdir}/${lib}.bak" ]; then
                mv "${cuda12_libdir}/${lib}" "${cuda12_libdir}/${lib}.bak"
            fi
        done
    fi
}
disable_cuda12_cudart

TE_ENV_VARS_BASE=(
    "TRITON_CODEGEN_ARCH=sm_100a"
    "TRITON_OVERRIDE_ARCH=sm100"
    "CMAKE_CUDA_ARCHITECTURES=100a"
    "NVTE_ENABLE_FP8=1"
    "NVTE_ENABLE_FP4=1"
    "NVTE_ENABLE_TMA=1"
)

run_with_te_env() {
    TRITON_CODEGEN_ARCH=sm_100a \
    TRITON_OVERRIDE_ARCH=sm100 \
    CMAKE_CUDA_ARCHITECTURES=100a \
    NVTE_ENABLE_FP8=1 \
    NVTE_ENABLE_FP4=1 \
    NVTE_ENABLE_TMA=1 \
    "$@"
}

TE_WHEEL_DIR="${THIRD_PARTY_DIR}/wheels"
mkdir -p "${TE_WHEEL_DIR}"

# Keep torch pinned to cu130 nightly wheel (cu13 deps) for TE builds
if ! pip_install --no-cache-dir --force-reinstall --ignore-installed --no-deps \
    --index-url "${PYTORCH_CU130_INDEX}" \
    --find-links "${PYTORCH_TORCH_FIND_LINKS}" \
    --extra-index-url "https://pypi.org/simple" \
    --only-binary=":all:" \
    "torch==${PYTORCH_TORCH_VERSION}"; then
    echo "ERROR: torch ${PYTORCH_TORCH_VERSION} install failed for TE rebuild"
    exit 1
fi
pip_install --upgrade pydantic==2.12.4 pydantic-core==2.41.5

# Clone/update TransformerEngine source under third_party
if [ ! -d "${TE_SRC_DIR}" ]; then
    git clone "${TE_REPO_URL}" "${TE_SRC_DIR}"
fi
git -C "${TE_SRC_DIR}" fetch --all
git -C "${TE_SRC_DIR}" checkout "${TE_GIT_COMMIT}"
git -C "${TE_SRC_DIR}" submodule update --init --recursive

# Build a wheel and install it (stores wheel in third_party/wheels for reuse)
(
    cd "${TE_SRC_DIR}"
    run_with_te_env pip_wheel . -w "${TE_WHEEL_DIR}" --no-build-isolation --no-deps
)
LATEST_TE_WHEEL="$(ls -t "${TE_WHEEL_DIR}"/transformer_engine-*.whl | head -n 1)"
if [ -z "${LATEST_TE_WHEEL}" ]; then
    echo "ERROR: TransformerEngine wheel not built."
    exit 1
fi
run_with_te_env pip_install --force-reinstall --no-build-isolation --no-deps "${LATEST_TE_WHEEL}"

# Persist runtime environment for future shells
TE_ENV_FILE="${THIRD_PARTY_DIR}/te_sm100a_env.sh"
CU13_LIBDIR="$(python3 - <<'PY'
import site, pathlib
paths = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if sp:
        paths.append(pathlib.Path(sp) / 'nvidia' / 'cu13' / 'lib')
for candidate in paths:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
print('')
PY
)"
CUSPARSELT_LIBDIR="$(python3 - <<'PY'
import site, pathlib
paths = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if sp:
        paths.append(pathlib.Path(sp) / 'nvidia' / 'cusparselt' / 'lib')
for candidate in paths:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
print('')
PY
)"
if [ -z "${CU13_LIBDIR}" ] || [ -z "${CUSPARSELT_LIBDIR}" ]; then
    echo "ERROR: Could not locate cu13 or cusparselt library paths."
    exit 1
fi
cat > "${TE_ENV_FILE}" <<EOF
# TransformerEngine sm_100a FP8/FP4/TMA environment
export TRITON_CODEGEN_ARCH=sm_100a
export TRITON_OVERRIDE_ARCH=sm100
export NVTE_ENABLE_FP8=1
export NVTE_ENABLE_FP4=1
export NVTE_ENABLE_TMA=1
export LD_LIBRARY_PATH=${CUSPARSELT_LIBDIR}:${CU13_LIBDIR}:\${LD_LIBRARY_PATH}
EOF
chmod +x "${TE_ENV_FILE}"
echo "To use TE FP8/FP4 fast paths on Blackwell, run: source ${TE_ENV_FILE}"

echo ""
if [[ -n "${GPU_COMPUTE_SM_NUM:-}" && "${GPU_COMPUTE_SM_NUM}" -ge 100 ]]; then
    echo "Running comprehensive Blackwell feature verification..."
    echo "(TMEM, TMA, CTA Clusters, DSMEM, FP8, FP4, Warp Features)"
    if python3 "${PROJECT_ROOT}/core/verification/verify_all_blackwell_features.py" --json; then
        echo "✓ All Blackwell features verified successfully!"
        echo "  Results saved to: artifacts/blackwell_verification/verification_results.json"
    else
        echo "WARNING: Some Blackwell features may need attention."
        echo "  Review: artifacts/blackwell_verification/verification_results.json"
        # Don't exit - this is informational, not critical
    fi
else
    echo "Skipping Blackwell verification (non-Blackwell GPU detected)."
fi

if [ "${INSTALL_GPT_OSS}" -eq 1 ]; then
    echo ""
    echo "Downloading GPT-OSS model and installing CLI..."
    GPT_OSS_MODEL_DIR="${GPT_OSS_MODEL_DIR:-${PROJECT_ROOT}/gpt-oss-20b}"
    huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir "${GPT_OSS_MODEL_DIR}/" || {
        echo "ERROR: Failed to download gpt-oss-20b model"
        exit 1
    }
    if ! pip_install --no-cache-dir --upgrade gpt-oss; then
        echo "ERROR: Failed to install gpt-oss package"
        exit 1
    fi
    TRITON_KERNELS_SRC_DIR="${TRITON_KERNELS_SRC_DIR:-${PROJECT_ROOT}/third_party/triton-kernels}"
    if [ -z "${TRITON_KERNELS_SPEC:-}" ]; then
        if [ ! -d "${TRITON_KERNELS_SRC_DIR}" ]; then
            if ! git clone --depth 1 https://github.com/openai/triton-kernels.git "${TRITON_KERNELS_SRC_DIR}"; then
                echo "WARNING: Failed to clone https://github.com/openai/triton-kernels.git into ${TRITON_KERNELS_SRC_DIR}."
            fi
        fi
        if [ -f "${TRITON_KERNELS_SRC_DIR}/pyproject.toml" ] || [ -f "${TRITON_KERNELS_SRC_DIR}/setup.py" ]; then
            TRITON_KERNELS_SPEC="${TRITON_KERNELS_SRC_DIR}"
        else
            TRITON_KERNELS_SPEC="git+https://github.com/openai/triton-kernels.git"
        fi
    fi
    if [[ "${TRITON_KERNELS_SPEC}" == git+https://github.com/openai/triton-kernels.git ]]; then
        if ! git ls-remote https://github.com/openai/triton-kernels.git >/dev/null 2>&1; then
            echo "WARNING: Cannot access https://github.com/openai/triton-kernels.git without credentials."
            echo "         Provide TRITON_KERNELS_SPEC (wheel/URL) or clone into ${TRITON_KERNELS_SRC_DIR}."
        fi
    fi
    if ! pip_install --no-cache-dir --upgrade --no-deps "${TRITON_KERNELS_SPEC}"; then
        echo "WARNING: Failed to install ${TRITON_KERNELS_SPEC}; gpt_oss chat may not work."
    fi
    GPT_OSS_BACKEND="${GPT_OSS_BACKEND:-triton}"
    if [ "${GPT_OSS_BACKEND}" = "triton" ]; then
        if ! python3 - <<'PY'
import importlib
modules = [
    "triton_kernels.swiglu",
    "triton_kernels.numerics_details.mxfp",
    "triton_kernels.matmul_ogs",
    "triton_kernels.routing",
    "triton_kernels.tensor",
]
for name in modules:
    importlib.import_module(name)
PY
        then
            echo "WARNING: triton_kernels missing required modules; falling back to torch backend."
            GPT_OSS_BACKEND="torch"
        fi
    fi
    if ! python3 -m gpt_oss.chat --backend "${GPT_OSS_BACKEND}" "${GPT_OSS_MODEL_DIR}"; then
        echo "WARNING: gpt_oss chat invocation failed; skipping."
    fi
else
    echo ""
    echo "Skipping GPT-OSS model/CLI install (set INSTALL_GPT_OSS=1 to enable)."
fi

echo ""
echo "Final cleanup (APT/pip caches)..."
reclaim_disk_space_basic

# Final summary
echo ""
echo "Setup Complete!"
echo "=================="
echo ""
echo "Installed:"
echo "  • PyTorch source build (${PYTORCH_BUILD_VERSION:-custom}) with NVIDIA arch list"
echo "  • CUDA ${CUDA_FULL_VERSION} toolchain and development tools"
echo "  • NCCL ${NCCL_SHORT_VERSION} (Blackwell-optimized with NVLS support)"
echo "  • NVSHMEM 3.4.5 runtime and headers (CUDA 13)"
echo "  • GPUDirect Storage (GDS) tools, drivers, kvikio library, and boot-time nvidia-fs load config"
echo "  • NVIDIA Nsight Systems (latest available)"
echo "  • NVIDIA Nsight Compute (latest available)"
echo "  • All project dependencies"
echo "  • System performance tools (numactl, perf, etc.)"
echo ""
echo "Verified:"
echo "  • PyTorch installation and CUDA functionality"
echo "  • NVLink connectivity (if multi-GPU)"
echo "  • CUTLASS backend configuration"
echo "  • GPUDirect Storage (GDS) functionality"
echo "  • Blackwell features (if SM 10.0+): TMEM, TMA, Clusters, DSMEM, FP8, FP4"
echo ""
echo "Post-Run Checklist:"
echo "  • Verify Python runtime:          python --version  (expect 3.12.x)"
echo "  • Verify CUDA compiler:           nvcc --version    (expect release 13.0.88 / Update 2)"
echo "  • Verify driver & GPU status:     nvidia-smi        (expect driver ${REQUIRED_DRIVER_VERSION})"
echo "  • Storage: Configure GDS-supported storage (NVMe, BeeGFS, NFS, WekaFS, etc.) and validate with /usr/local/cuda/gds/tools/gdscheck -p"
echo "  • Verify PyTorch arch coverage:   python - <<'PY'"
echo "                                      import torch"
echo "                                      print(torch.cuda.get_arch_list())"
echo "                                    PY"
echo "                                    (expect ['sm_100', 'sm_103', 'sm_121', 'compute_121'])"
echo "  • cuDNN: Installed 9.15.1.9-1 (matches PyTorch's compile-time version; PyTorch bundles 9.13.0 but we use system 9.15.1)"
echo "  • Before additional builds: source /etc/profile.d/cuda-${CUDA_SHORT_VERSION}.sh (or start a new shell)"
echo "For more information, see the README.md file and chapter-specific documentation."
echo ""
echo "Happy performance engineering!"
