#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  log_activity.sh cmd [--run-id <id>] [--label <label>] [--note <note>] -- <command...>
  log_activity.sh decision [--run-id <id>] --note <note>

Notes:
- Writes the activity table and command log under runs/<RUN_ID>/progress/activity/.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODE="$1"
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL=""
NOTE=""

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
    --note)
      NOTE="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
ACTIVITY_DIR="${CLUSTER_RUN_DIR_EFFECTIVE}/progress/activity"
ACTIVITY_LOG="${ACTIVITY_DIR}/${RUN_ID}_activity.log"
ACTIVITY_REPORT="${ACTIVITY_DIR}/${RUN_ID}_activity.md"
mkdir -p "$ACTIVITY_DIR"

timestamp="$(date -Iseconds)"

escape_md_cell() {
  local s="$1"
  s="${s//$'\n'/ }"
  s="${s//|/\\|}"
  printf '%s' "$s"
}

append_entry() {
  local date_cell="$1"
  local update_cell="$2"
  local row
  row="| ${date_cell} | $(escape_md_cell "$update_cell") |"

  if [[ ! -e "$ACTIVITY_REPORT" ]]; then
    (set -o noclobber; printf '## Activity Log\n\n| Date | Update |\n| --- | --- |\n' > "$ACTIVITY_REPORT") 2>/dev/null \
      || [[ -f "$ACTIVITY_REPORT" ]]
  fi
  printf '%s\n' "$row" >> "$ACTIVITY_REPORT"
}

case "$MODE" in
  cmd)
    if [[ $# -lt 1 ]]; then
      echo "cmd mode requires a command after --" >&2
      usage >&2
      exit 1
    fi
    cmd_str="$*"
    {
      echo "== ${timestamp} [cmd] label=${LABEL:-n/a} note=${NOTE:-} =="
      echo "\$ ${cmd_str}"
    } >> "$ACTIVITY_LOG"
    set +e
    "$@" 2>&1 | tee -a "$ACTIVITY_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    echo "rc=${rc}" >> "$ACTIVITY_LOG"
    entry_date="${timestamp%%T*}"
    entry="[cmd] label=${LABEL:-n/a} rc=${rc} cmd=\`$cmd_str\`${NOTE:+ note=\"$NOTE\"}"
    append_entry "$entry_date" "$entry"
    exit "$rc"
    ;;
  decision)
    if [[ -z "$NOTE" ]]; then
      echo "decision mode requires --note" >&2
      usage >&2
      exit 1
    fi
    echo "== ${timestamp} [decision] note=${NOTE} ==" >> "$ACTIVITY_LOG"
    entry_date="${timestamp%%T*}"
    entry="[decision] note=\"$NOTE\""
    append_entry "$entry_date" "$entry"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac
