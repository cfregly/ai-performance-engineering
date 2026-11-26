#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source scripts/enable_activity_log.sh [--run-id <id>] [--label <label>] [--log-file <path>]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname -s)"
LOG_FILE=""

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
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      return 1
      ;;
  esac
done

if [[ -z "$LOG_FILE" ]]; then
  resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
  LOG_FILE="${CLUSTER_RUN_DIR_EFFECTIVE}/progress/activity/${RUN_ID}_shell.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"

export ACTIVITY_LOG_RUN_ID="$RUN_ID"
export ACTIVITY_LOG_LABEL="$LABEL"
export ACTIVITY_LOG_FILE="$LOG_FILE"

__activity_log_last_cmd=""
__activity_log_original_prompt="${PROMPT_COMMAND:-}"

__activity_log_prompt() {
  local rc=$?
  history -a
  local cmd
  cmd="$(history 1 | sed 's/^[ ]*[0-9]\+[ ]*//')"
  if [[ -z "$cmd" ]]; then
    return
  fi
  if [[ "$cmd" == "$__activity_log_last_cmd" ]]; then
    return
  fi
  case "$cmd" in
    "history "*|history|*log_activity.sh*|*enable_activity_log.sh*)
      __activity_log_last_cmd="$cmd"
      return
      ;;
  esac
  __activity_log_last_cmd="$cmd"
  printf "%s [cmd] label=%s rc=%s cmd=%s\n" "$(date -Iseconds)" "$ACTIVITY_LOG_LABEL" "$rc" "$cmd" >> "$ACTIVITY_LOG_FILE"
}

if [[ -n "$__activity_log_original_prompt" ]]; then
  PROMPT_COMMAND="__activity_log_prompt; $__activity_log_original_prompt"
else
  PROMPT_COMMAND="__activity_log_prompt"
fi

echo "Activity logging enabled -> ${ACTIVITY_LOG_FILE}"
echo "Decision logging: scripts/log_activity.sh decision --run-id ${RUN_ID} --note \"<decision>\""
