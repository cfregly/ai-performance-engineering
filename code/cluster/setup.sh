#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup.sh [options]

Options:
  --run-id <id>            RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>          Node label for discovery metadata (default: node1)
  --venv <path>            Python venv path (default: cluster/env/venv)
  --nccl-tests-dir <path>  nccl-tests checkout path (default: cluster/tools/nccl-tests)
  --skip-discovery         Skip discovery metadata capture
  --skip-apt               Skip apt package installs
  --skip-python            Skip the locked Python 3.12 runtime install
  --skip-nccl-tests         Skip nccl-tests clone/build
  --install-transformer-engine
                           Build the pinned Transformer Engine Torch binding
  --install-vllm           Build pinned vLLM and its required TE binding
  -h, --help               Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="node1"
VENV_DIR="${ROOT_DIR}/env/venv"
VENV_PY="${VENV_DIR}/bin/python"
NCCL_TESTS_DIR="${ROOT_DIR}/tools/nccl-tests"
PYTHON_STACK_LOCK="${CODE_ROOT}/requirements_torch2141_linux_x86_64.txt"
VLLM_SOURCE_CONTRACT="${CODE_ROOT}/vllm_source_build.json"
NATIVE_EXTENSION_CONTRACT="${CODE_ROOT}/native_torch_extensions.json"
VLLM_RUST_TOOLCHAIN="1.95.0"
SKIP_DISCOVERY=0
SKIP_APT=0
SKIP_PYTHON=0
SKIP_NCCL_TESTS=0
INSTALL_TRANSFORMER_ENGINE=0
INSTALL_VLLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --nccl-tests-dir)
      NCCL_TESTS_DIR="$2"
      shift 2
      ;;
    --skip-discovery)
      SKIP_DISCOVERY=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --skip-nccl-tests)
      SKIP_NCCL_TESTS=1
      shift
      ;;
    --install-transformer-engine)
      INSTALL_TRANSFORMER_ENGINE=1
      shift
      ;;
    --install-vllm)
      INSTALL_VLLM=1
      INSTALL_TRANSFORMER_ENGINE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$SKIP_PYTHON" -eq 1 ]] && \
   [[ "$INSTALL_TRANSFORMER_ENGINE" -eq 1 || "$INSTALL_VLLM" -eq 1 ]]; then
  echo "--skip-python cannot be combined with source-build install options." >&2
  exit 1
fi

require_exact_rust_toolchain() {
  local expected="$1"
  local rustc_version=""
  local cargo_version=""

  if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
    echo "Rust ${expected} requires both rustc and cargo on PATH." >&2
    return 1
  fi
  rustc_version="$(rustc --version)"
  cargo_version="$(cargo --version)"
  if [[ "$rustc_version" != "rustc ${expected} "* ]]; then
    echo "Expected rustc ${expected}, found ${rustc_version}." >&2
    return 1
  fi
  if [[ "$cargo_version" != "cargo ${expected} "* ]]; then
    echo "Expected cargo ${expected}, found ${cargo_version}." >&2
    return 1
  fi
  echo "Using ${rustc_version}"
  echo "Using ${cargo_version}"
}

provision_vllm_rust_toolchain() {
  local expected="$1"
  local rustup_home="$2"
  local bootstrap_cargo_home="$3"

  if [[ -e "$rustup_home" || -L "$rustup_home" ]]; then
    echo "Choose a new --venv path to preserve the existing Rust toolchain: $rustup_home" >&2
    return 1
  fi
  if [[ -e "$bootstrap_cargo_home" || -L "$bootstrap_cargo_home" ]]; then
    echo "Choose a new --venv path to preserve the existing Rust bootstrap: $bootstrap_cargo_home" >&2
    return 1
  fi

  mkdir -p "$rustup_home" "$bootstrap_cargo_home"
  export RUSTUP_HOME="$rustup_home"
  export CARGO_HOME="$bootstrap_cargo_home"
  export PATH="${CARGO_HOME}/bin:${PATH}"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
  rustup toolchain install "$expected" --profile minimal
  export RUSTUP_TOOLCHAIN="$expected"
  require_exact_rust_toolchain "$expected"
}

resolve_torch_lib_dir() {
  local py_bin="$1"
  if [[ ! -x "$py_bin" ]]; then
    return 0
  fi

  "$py_bin" - <<'PY'
import os
import site
import sys
from pathlib import Path

roots = []
py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
roots.append(Path(sys.prefix) / "lib" / py_version / "site-packages")

venv = os.environ.get("VIRTUAL_ENV")
if venv:
    roots.append(Path(venv) / "lib" / py_version / "site-packages")

try:
    roots.extend(Path(p) for p in site.getsitepackages())
except Exception:
    pass

try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass

seen = set()
for root in roots:
    candidate = root / "torch" / "lib"
    key = str(candidate)
    if key in seen:
        continue
    seen.add(key)
    if candidate.is_dir():
        print(candidate)
        raise SystemExit(0)

print("")
PY
}

write_runtime_env_file() {
  local runtime_env_file="$1"
  local torch_lib_dir="$2"
  mkdir -p "$(dirname "$runtime_env_file")"

  if [[ -n "$torch_lib_dir" ]]; then
    cat >"${runtime_env_file}" <<EOF
#!/usr/bin/env bash
export AISP_CUDNN_RUNTIME_POLICY="\${AISP_CUDNN_RUNTIME_POLICY:-auto}"
if [[ -d "${torch_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${torch_lib_dir}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
EOF
  else
    cat >"${runtime_env_file}" <<'EOF'
#!/usr/bin/env bash
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"
EOF
  fi

  chmod 0755 "${runtime_env_file}"
}

OUT_STRUCT_DIR="${ROOT_DIR}/results/structured"
OUT_RAW_DIR="${ROOT_DIR}/results/raw/setup"
DISCOVERY_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_meta.json"
SETUP_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_setup.json"
SETUP_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_setup.log"

mkdir -p "$OUT_STRUCT_DIR" "$OUT_RAW_DIR"

if [[ -e "$DISCOVERY_OUT" && "$SKIP_DISCOVERY" -eq 0 ]]; then
  echo "Discovery output already exists: $DISCOVERY_OUT" >&2
  echo "Pick a different --run-id or use --skip-discovery." >&2
  exit 1
fi

if [[ -e "$SETUP_OUT" ]]; then
  echo "Setup output already exists: $SETUP_OUT" >&2
  echo "Pick a different --run-id before re-running." >&2
  exit 1
fi

exec > >(tee -a "$SETUP_LOG") 2>&1

echo "== Cluster setup starting ($(date -Iseconds)) =="
echo "ROOT_DIR=$ROOT_DIR"
echo "RUN_ID=$RUN_ID"
echo "LABEL=$LABEL"
echo "VENV_DIR=$VENV_DIR"
echo "NCCL_TESTS_DIR=$NCCL_TESTS_DIR"
echo "PYTHON_STACK_LOCK=$PYTHON_STACK_LOCK"
echo "VLLM_SOURCE_CONTRACT=$VLLM_SOURCE_CONTRACT"
echo "NATIVE_EXTENSION_CONTRACT=$NATIVE_EXTENSION_CONTRACT"

if [[ "$SKIP_DISCOVERY" -eq 0 ]]; then
  echo "== Discovery =="
  "${ROOT_DIR}/scripts/collect_system_info.sh" --output "$DISCOVERY_OUT" --label "$LABEL"
fi

APT_PACKAGES=(
  build-essential
  cmake
  ninja-build
  git
  pkg-config
  python3-venv
  python3-pip
  python3-dev
  numactl
  hwloc
  pciutils
  ethtool
  iproute2
  iputils-ping
  iperf3
  rdma-core
  infiniband-diags
  ibverbs-utils
  libibverbs-dev
  libnuma-dev
  libopenmpi-dev
  openmpi-bin
  libdw-dev
  libelf-dev
  libssl-dev
  libffi-dev
  patchelf
  curl
  wget
  libnccl2
  libnccl-dev
  cuda-toolkit-13-0
)

if [[ "$SKIP_APT" -eq 0 ]]; then
  echo "== Apt install =="
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
fi

CUDA_HOME="/usr/local/cuda"
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "CUDA_HOME missing at $CUDA_HOME; nvcc install may have failed." >&2
  exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"

MPI_HOME=""
if command -v mpicc >/dev/null 2>&1; then
  MPI_INCDIRS="$(mpicc --showme:incdirs 2>/dev/null | awk '{print $1}')"
  if [[ -n "$MPI_INCDIRS" ]]; then
    MPI_HOME="$(dirname "$MPI_INCDIRS")"
  fi
fi
if [[ -z "$MPI_HOME" && -d /usr/lib/aarch64-linux-gnu/openmpi ]]; then
  MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
fi

if [[ "$SKIP_PYTHON" -eq 0 ]]; then
  echo "== Locked Python 3.12 runtime =="
  if [[ ! -f "$PYTHON_STACK_LOCK" ]]; then
    echo "PyTorch 2.14.1 runtime lock is missing: $PYTHON_STACK_LOCK" >&2
    exit 1
  fi
  if [[ -e "$VENV_DIR" ]]; then
    echo "Choose a new --venv path to preserve the existing environment: $VENV_DIR" >&2
    exit 1
  fi
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12 is required for the locked PyTorch 2.14.1 runtime." >&2
    exit 1
  fi
  python3.12 -m venv "$VENV_DIR"
  VENV_PY="${VENV_DIR}/bin/python"
  echo "== Installing the hash-locked benchmark runtime =="
  (
    cd "$CODE_ROOT"
    "$VENV_PY" -m pip install --require-hashes --no-deps -r "$PYTHON_STACK_LOCK"
  )
  export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"

  if [[ "$INSTALL_TRANSFORMER_ENGINE" -eq 1 ]]; then
    if [[ ! -f "$NATIVE_EXTENSION_CONTRACT" ]]; then
      echo "Pinned native-extension contract is missing: $NATIVE_EXTENSION_CONTRACT" >&2
      exit 1
    fi
    NATIVE_BUILD_DIR="${VENV_DIR}.native-build"
    echo "== Building pinned Transformer Engine Torch binding =="
    CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
      "$VENV_PY" "${CODE_ROOT}/scripts/build_torch_extensions.py" \
      --contract "$NATIVE_EXTENSION_CONTRACT" \
      --output-dir "$NATIVE_BUILD_DIR" \
      --package transformer-engine-torch \
      --cuda-arch-list "${VLLM_ARCH_LIST:-10.0}" \
      --max-jobs "${MAX_JOBS:-8}" \
      --nvcc-threads "${NVCC_THREADS:-2}"
    TE_WHEEL="$("$VENV_PY" - "$NATIVE_BUILD_DIR/native-torch-extensions.manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    wheel = json.load(handle)["packages"]["transformer-engine-torch"]["wheel"]
path = Path(wheel["path"])
if not path.is_file():
    raise SystemExit("Transformer Engine build receipt wheel is missing")
if hashlib.sha256(path.read_bytes()).hexdigest() != wheel["sha256"]:
    raise SystemExit("Transformer Engine build receipt wheel hash does not match")
print(path)
PY
)"
    "$VENV_PY" -m pip install --no-deps "$TE_WHEEL"
    "$VENV_PY" - "$NATIVE_EXTENSION_CONTRACT" <<'PY'
import json
from importlib import metadata
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    contract = json.load(handle)
package = next(
    item for item in contract["packages"] if item["id"] == "transformer-engine-torch"
)
expected = dict(package["preinstalled_distributions"])
expected[package["id"]] = package["expected_metadata"]["version"]
for distribution, version in expected.items():
    actual = metadata.version(distribution)
    if actual != version:
        raise SystemExit(f"Expected {distribution}=={version}, found {actual}")
import transformer_engine.pytorch
import transformer_engine_torch
print("Transformer Engine Torch binding", expected["transformer-engine-torch"])
PY
  fi

  if [[ "$INSTALL_VLLM" -eq 1 ]]; then
    if [[ ! -f "$VLLM_SOURCE_CONTRACT" ]]; then
      echo "Pinned vLLM source contract is missing: $VLLM_SOURCE_CONTRACT" >&2
      exit 1
    fi
    CONTRACT_RUST_TOOLCHAIN="$("$VENV_PY" - "$VLLM_SOURCE_CONTRACT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["build"]["rust_toolchain"])
PY
)"
    if [[ "$CONTRACT_RUST_TOOLCHAIN" != "$VLLM_RUST_TOOLCHAIN" ]]; then
      echo "Expected vLLM Rust contract ${VLLM_RUST_TOOLCHAIN}, found ${CONTRACT_RUST_TOOLCHAIN}." >&2
      exit 1
    fi
    echo "== Provisioning isolated Rust ${VLLM_RUST_TOOLCHAIN} toolchain =="
    provision_vllm_rust_toolchain \
      "$VLLM_RUST_TOOLCHAIN" \
      "${VENV_DIR}.rustup" \
      "${VENV_DIR}.rust-bootstrap"
    VLLM_BUILD_DIR="${VENV_DIR}.vllm-build"
    echo "== Building pinned vLLM source contract =="
    CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
      "$VENV_PY" "${CODE_ROOT}/scripts/build_vllm_for_torch.py" \
      --contract "$VLLM_SOURCE_CONTRACT" \
      --output-dir "$VLLM_BUILD_DIR" \
      --cuda-arch-list "${VLLM_ARCH_LIST:-10.0}" \
      --max-jobs "${MAX_JOBS:-8}" \
      --nvcc-threads "${NVCC_THREADS:-2}"
    VLLM_WHEEL="$("$VENV_PY" - "$VLLM_BUILD_DIR/vllm-source-wheel.manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    wheel = json.load(handle)["wheel"]
path = Path(wheel["path"])
if not path.is_file():
    raise SystemExit("vLLM build receipt wheel is missing")
if hashlib.sha256(path.read_bytes()).hexdigest() != wheel["sha256"]:
    raise SystemExit("vLLM build receipt wheel hash does not match")
print(path)
PY
)"
    "$VENV_PY" -m pip install --no-deps "$VLLM_WHEEL"
  fi

  TORCH_LIB_DIR="$(resolve_torch_lib_dir "$VENV_PY")"
  if [[ -n "$TORCH_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH:-}"
    echo "Using torch runtime libraries from: $TORCH_LIB_DIR"
  else
    echo "WARNING: torch/lib was not found under ${VENV_DIR}; setup sanity checks may use system runtime libs."
  fi

  echo "== Python sanity =="
  "$VENV_DIR/bin/python" - <<'PY'
import os
import torch
import transformers
import datasets
import tokenizers
import huggingface_hub
import sentencepiece
import fsspec
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
print("torch.version.git", torch.version.git_version)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("sentencepiece", sentencepiece.__version__)
print("fsspec", fsspec.__version__)
print("AISP_CUDNN_RUNTIME_POLICY", os.environ.get("AISP_CUDNN_RUNTIME_POLICY", ""))
if torch.__version__ != "2.14.1+cu130":
    raise SystemExit(f"Unexpected torch version: {torch.__version__}")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("PyTorch 2.14.1 runtime must use the CXX11 ABI")
if torch.cuda.is_available():
    print("torch.backends.cudnn.version", torch.backends.cudnn.version())
    nccl_v = torch.cuda.nccl.version()
    if nccl_v:
        print("torch.cuda.nccl.version", ".".join(str(x) for x in nccl_v))
    else:
        print("torch.cuda.nccl.version", "")
PY

  if [[ "$INSTALL_VLLM" -eq 1 ]]; then
    "$VENV_DIR/bin/python" - <<'PY'
import vllm
print("vllm", vllm.__version__)
PY
  fi
  echo "== pip check =="
  "$VENV_PY" -m pip check
fi

TORCH_LIB_DIR="${TORCH_LIB_DIR:-$(resolve_torch_lib_dir "$VENV_PY")}"
RUNTIME_ENV_FILE="${VENV_DIR}/orig_parity_runtime_env.sh"
write_runtime_env_file "$RUNTIME_ENV_FILE" "$TORCH_LIB_DIR"
if [[ -n "$TORCH_LIB_DIR" ]]; then
  echo "Wrote runtime env helper: ${RUNTIME_ENV_FILE} (torch/lib=${TORCH_LIB_DIR})"
else
  echo "Wrote runtime env helper: ${RUNTIME_ENV_FILE} (torch/lib unresolved)"
fi

if [[ "$SKIP_NCCL_TESTS" -eq 0 ]]; then
  echo "== nccl-tests =="
  mkdir -p "$(dirname "$NCCL_TESTS_DIR")"
  if [[ ! -d "$NCCL_TESTS_DIR/.git" ]]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_DIR"
  fi
  if [[ -z "$MPI_HOME" ]]; then
    echo "MPI_HOME not detected; set --skip-nccl-tests or install OpenMPI dev headers." >&2
    exit 1
  fi
  make -C "$NCCL_TESTS_DIR" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME=/usr CUDA_HOME="$CUDA_HOME"
fi

echo "== Writing setup metadata =="
python3 - <<'PY' "$SETUP_OUT" "$RUN_ID" "$LABEL" "$VENV_DIR" "$NCCL_TESTS_DIR"
import json
import os
import subprocess
import sys
import time

out_path, run_id, label, venv_dir, nccl_tests_dir = sys.argv[1:6]

def run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def run_lines(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return [], proc.stderr.strip()
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()], ""

apt_packages = [
    "build-essential","cmake","ninja-build","git","pkg-config","python3-venv","python3-pip","python3-dev",
    "numactl","hwloc","pciutils","ethtool","iproute2","iputils-ping","iperf3","rdma-core","infiniband-diags",
    "ibverbs-utils","libibverbs-dev","libnuma-dev","libopenmpi-dev","openmpi-bin","libdw-dev","libelf-dev",
    "libssl-dev","libffi-dev","patchelf","curl","wget","libnccl2","libnccl-dev","cuda-toolkit-13-0",
]

versions = {}
lines, err = run_lines(
    "dpkg-query -W -f='${Package} ${Version}\\n' " + " ".join(apt_packages)
)
if err:
    versions["dpkg_query_error"] = err
else:
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            versions[parts[0]] = parts[1]

torch_info = {}
if os.path.exists(os.path.join(venv_dir, "bin", "python")):
    torch_info = run(
        f"{os.path.join(venv_dir, 'bin', 'python')} - <<'PY'\n"
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    info = {\n"
        "        'version': torch.__version__,\n"
        "        'cuda_available': torch.cuda.is_available(),\n"
        "        'cuda_version': torch.version.cuda,\n"
        "        'git_version': torch.version.git_version,\n"
        "    }\n"
        "except Exception as exc:\n"
        "    info = {'error': str(exc)}\n"
        "print(json.dumps(info))\n"
        "PY"
    )

pip_freeze = []
pip_freeze_err = ""
if os.path.exists(os.path.join(venv_dir, "bin", "pip")):
    pip_freeze, pip_freeze_err = run_lines(f"{os.path.join(venv_dir, 'bin', 'pip')} freeze")

nccl_commit = ""
if os.path.exists(os.path.join(nccl_tests_dir, ".git")):
    nccl_commit = run(f"git -C {nccl_tests_dir} rev-parse HEAD")

payload = {
    "run_id": run_id,
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "apt_versions": versions,
    "torch_info_cmd": torch_info,
    "pip_freeze": pip_freeze,
    "pip_freeze_error": pip_freeze_err,
    "pip_check": run(f"{os.path.join(venv_dir, 'bin', 'pip')} check") if os.path.exists(os.path.join(venv_dir, 'bin', 'pip')) else {},
    "nvcc": run("nvcc --version"),
    "mpirun": run("mpirun --version"),
    "nvidia_smi": run("nvidia-smi"),
    "nccl_tests_commit": nccl_commit,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY

echo "== Setup complete ($(date -Iseconds)) =="
