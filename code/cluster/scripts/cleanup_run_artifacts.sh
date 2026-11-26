#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python3 "$SCRIPT_DIR/cleanup_run_artifacts.py" "$@"
