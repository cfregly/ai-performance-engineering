#!/usr/bin/env bash
# Fetch and install the requested CUTLASS release into third_party/cutlass.
set -euo pipefail

REF="${1:-${CUTLASS_REF:-v4.5.2}}"
REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${CUTLASS_SRC_DIR:-${PROJECT_ROOT}/third_party/cutlass}"

echo "Installing CUTLASS (${REF}) into ${DEST_DIR}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/cutlass.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if git clone --filter=blob:none "${REPO}" "${tmp_dir}/cutlass-src"; then
    if git -C "${tmp_dir}/cutlass-src" checkout "${REF}"; then
        :
    else
        echo "ERROR: Failed to checkout CUTLASS ref ${REF}" >&2
        exit 1
    fi
else
    echo "ERROR: Failed to clone CUTLASS repo (${REPO} @ ${REF})" >&2
    exit 1
fi

mkdir -p "$(dirname "${DEST_DIR}")"
rm -rf "${DEST_DIR}"
mv "${tmp_dir}/cutlass-src" "${DEST_DIR}"
rm -rf "${DEST_DIR}/.git"

echo "CUTLASS installed at ${DEST_DIR}"
