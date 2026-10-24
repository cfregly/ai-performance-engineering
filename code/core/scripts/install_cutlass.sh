#!/usr/bin/env bash
# Fetch and install one immutable CUTLASS source revision.
set -euo pipefail

DEFAULT_COMMIT="db1c288993354c88e551c40c19a8fb93a774a241"
REF="${1:-${CUTLASS_REF:-${DEFAULT_COMMIT}}}"
REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
EXPECTED_VERSION="${CUTLASS_EXPECTED_VERSION:-4.5.2}"
RELEASE_TAG="${CUTLASS_RELEASE_TAG:-v4.5.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${CUTLASS_SRC_DIR:-${PROJECT_ROOT}/third_party/cutlass}"

if [[ ! "${REF}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: CUTLASS_REF must be a full lowercase 40-character commit, got ${REF}." >&2
    exit 1
fi

echo "Installing CUTLASS (${REF}) into ${DEST_DIR}"
mkdir -p "$(dirname "${DEST_DIR}")"
tmp_dir="$(mktemp -d "$(dirname "${DEST_DIR}")/.cutlass-install.XXXXXX")"
staged_source="${tmp_dir}/cutlass-src"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if ! git clone --filter=blob:none --no-checkout "${REPO}" "${staged_source}"; then
    echo "ERROR: Failed to clone CUTLASS repository ${REPO}." >&2
    exit 1
fi
if ! git -C "${staged_source}" switch --detach "${REF}"; then
    echo "ERROR: Failed to select immutable CUTLASS commit ${REF}." >&2
    exit 1
fi

resolved_commit="$(git -C "${staged_source}" rev-parse HEAD)"
resolved_tree="$(git -C "${staged_source}" rev-parse 'HEAD^{tree}')"
if [ "${resolved_commit}" != "${REF}" ]; then
    echo "ERROR: CUTLASS resolved to ${resolved_commit}, expected ${REF}." >&2
    exit 1
fi
if [ -n "$(git -C "${staged_source}" status --porcelain --untracked-files=all)" ]; then
    echo "ERROR: Fresh CUTLASS source is unexpectedly dirty." >&2
    exit 1
fi

python3 - "${staged_source}" "${EXPECTED_VERSION}" <<'PYVERSION'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
text = (root / "include" / "cutlass" / "version.h").read_text(encoding="utf-8")


def value(name: str) -> str:
    match = re.search(rf"^#define\s+{name}\s+([0-9]+)\s*$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"CUTLASS version header is missing {name}")
    return match.group(1)


actual = ".".join(
    value(name) for name in ("CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH")
)
if actual != expected:
    raise SystemExit(f"CUTLASS source version is {actual}, expected {expected}")
PYVERSION

python3 - \
    "${staged_source}/.git/aisp-source.json" \
    "${REPO}" \
    "${REF}" \
    "${resolved_commit}" \
    "${resolved_tree}" \
    "${RELEASE_TAG}" \
    "${EXPECTED_VERSION}" <<'PYRECEIPT'
import json
from pathlib import Path
import sys

receipt = {
    "schema_version": 1,
    "repository": sys.argv[2],
    "requested_commit": sys.argv[3],
    "resolved_commit": sys.argv[4],
    "tree": sys.argv[5],
    "release_tag": sys.argv[6],
    "source_version": sys.argv[7],
}
Path(sys.argv[1]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PYRECEIPT

existing_git_root=""
if [ -d "${DEST_DIR}" ]; then
    existing_git_root="$(git -C "${DEST_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -e "${DEST_DIR}/.git" ] && [ -z "${existing_git_root}" ]; then
        echo "ERROR: Existing CUTLASS source has invalid Git metadata: ${DEST_DIR}" >&2
        exit 1
    fi
    if [ -n "${existing_git_root}" ]; then
        destination_root="$(cd "${DEST_DIR}" && pwd -P)"
        existing_git_root="$(cd "${existing_git_root}" && pwd -P)"
        if [ "${existing_git_root}" = "${destination_root}" ] && \
            [ -n "$(git -C "${DEST_DIR}" status --porcelain --untracked-files=all)" ]; then
            echo "ERROR: Existing CUTLASS source has local changes: ${DEST_DIR}" >&2
            echo "       Preserve it and choose a fresh CUTLASS_SRC_DIR." >&2
            exit 1
        fi
    fi
fi

backup_source=""
if [ -e "${DEST_DIR}" ] || [ -L "${DEST_DIR}" ]; then
    backup_source="${DEST_DIR}.backup.$(date -u +%Y%m%dT%H%M%SZ).$$"
    while [ -e "${backup_source}" ] || [ -L "${backup_source}" ]; do
        backup_source="${backup_source}.1"
    done
    mv "${DEST_DIR}" "${backup_source}"
fi
if ! mv "${staged_source}" "${DEST_DIR}"; then
    if [ -n "${backup_source}" ] && \
        { [ -e "${backup_source}" ] || [ -L "${backup_source}" ]; }; then
        mv "${backup_source}" "${DEST_DIR}"
    fi
    echo "ERROR: Failed to install the verified CUTLASS source." >&2
    exit 1
fi

echo "CUTLASS installed at ${DEST_DIR}"
echo "CUTLASS commit ${resolved_commit} tree ${resolved_tree}"
if [ -n "${backup_source}" ]; then
    echo "Previous CUTLASS source preserved at ${backup_source}"
fi
