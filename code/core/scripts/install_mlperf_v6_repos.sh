#!/usr/bin/env bash
# Fetch official MLPerf v6 source trees into third_party/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read_setup_default() {
    local var_name="$1"
    python - "${PROJECT_ROOT}/setup.sh" "${var_name}" <<'PY'
import sys
from pathlib import Path

setup_path = Path(sys.argv[1])
var_name = sys.argv[2]
prefix = f'{var_name}="${{{var_name}:-'

for raw_line in setup_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith(prefix) and line.endswith('}"'):
        print(line[len(prefix):-2])
        raise SystemExit(0)

raise SystemExit(1)
PY
}

THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"

MLPERF_INFERENCE_REPO="${MLPERF_INFERENCE_REPO_URL:-$(read_setup_default MLPERF_INFERENCE_REPO_URL)}"
MLPERF_INFERENCE_REF="${MLPERF_INFERENCE_GIT_REF:-$(read_setup_default MLPERF_INFERENCE_GIT_REF)}"
MLPERF_INFERENCE_DEST_RAW="${MLPERF_INFERENCE_SRC_DIR:-$(read_setup_default MLPERF_INFERENCE_SRC_DIR)}"
MLPERF_INFERENCE_DEST="${MLPERF_INFERENCE_DEST_RAW//\$\{THIRD_PARTY_DIR\}/${THIRD_PARTY_DIR}}"
MLPERF_INFERENCE_EXPECTED_LABEL="${MLPERF_INFERENCE_EXPECTED_LABEL:-MLPerf Inference v6.0}"

MLPERF_TRAINING_REPO="${MLPERF_TRAINING_REPO_URL:-$(read_setup_default MLPERF_TRAINING_REPO_URL)}"
MLPERF_TRAINING_REF="${MLPERF_TRAINING_GIT_REF:-$(read_setup_default MLPERF_TRAINING_GIT_REF)}"
MLPERF_TRAINING_DEST_RAW="${MLPERF_TRAINING_SRC_DIR:-$(read_setup_default MLPERF_TRAINING_SRC_DIR)}"
MLPERF_TRAINING_DEST="${MLPERF_TRAINING_DEST_RAW//\$\{THIRD_PARTY_DIR\}/${THIRD_PARTY_DIR}}"
MLPERF_TRAINING_EXPECTED_LABEL="${MLPERF_TRAINING_EXPECTED_LABEL:-MLPerf Training v6.0}"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/mlperf-v6.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

install_repo() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    local dest="$4"
    local expected_label="$5"

    local src_dir="${tmp_dir}/${name}"
    echo "Installing ${name} (${ref}) into ${dest}"

    git clone --filter=blob:none "${repo}" "${src_dir}" >/dev/null 2>&1
    git -C "${src_dir}" checkout "${ref}" >/dev/null 2>&1

    local actual_commit
    actual_commit="$(git -C "${src_dir}" rev-parse HEAD)"
    local committed_at
    committed_at="$(git -C "${src_dir}" show -s --format=%cI HEAD)"

    mkdir -p "$(dirname "${dest}")"
    rm -rf "${dest}"
    mv "${src_dir}" "${dest}"
    rm -rf "${dest}/.git"

    python - "$dest" "$name" "$repo" "$ref" "$actual_commit" "$committed_at" "$expected_label" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
name = sys.argv[2]
repo = sys.argv[3]
ref = sys.argv[4]
actual_commit = sys.argv[5]
committed_at = sys.argv[6]
expected_label = sys.argv[7]
payload = {
    "name": name,
    "repo": repo,
    "requested_ref": ref,
    "resolved_commit": actual_commit,
    "committed_at": committed_at,
    "expected_label": expected_label,
}
(dest / "VENDORED_FROM.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

install_repo \
    "mlperf_inference" \
    "${MLPERF_INFERENCE_REPO}" \
    "${MLPERF_INFERENCE_REF}" \
    "${MLPERF_INFERENCE_DEST}" \
    "${MLPERF_INFERENCE_EXPECTED_LABEL}"

install_repo \
    "mlperf_training" \
    "${MLPERF_TRAINING_REPO}" \
    "${MLPERF_TRAINING_REF}" \
    "${MLPERF_TRAINING_DEST}" \
    "${MLPERF_TRAINING_EXPECTED_LABEL}"

echo "MLPerf v6 source trees installed:"
echo "  - ${MLPERF_INFERENCE_DEST}"
echo "  - ${MLPERF_TRAINING_DEST}"
