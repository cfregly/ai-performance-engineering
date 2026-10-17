#!/usr/bin/env bash

# Automated secure cleanup script based on the hardening checklist provided
# by cfregly. Run in dry-run mode first (default) to review actions, then
# re-run with --execute once you're confident and have revoked credentials.

set -uo pipefail
IFS=$'\n\t'
shopt -s nullglob

DRY_RUN=1
ASSUME_YES=0
BACKUP_ENABLED=1
REBOOT_AFTER=0
REQUIRE_VERIFY_TOOLS=0
HOME_DIR="${HOME:-$PWD}"
AI_REPO="${AI_REPO:-$HOME_DIR/ai-performance-engineering}"
PPLX_REPO="${PPLX_REPO:-$HOME_DIR/pplx-garden}"
BACKUP_ROOT=""
SHRED_BIN="$(command -v shred || true)"
FAILURES=0
TOTAL_SHREDS=0
TOTAL_DELETES=0
RG_AVAILABLE=0
DETECT_AVAILABLE=0

usage() {
  cat <<'EOF'
Usage: system_wipe.sh [options]

Options:
  --execute             Apply destructive changes (default is dry-run preview).
  -y, --yes             Assume "yes" for interactive confirmations.
  --no-backup           Skip creation of sanitized .bashrc/.profile backups.
  --reboot              Reboot automatically after cleanup (default: no reboot).
  --ai-root PATH        Override ai-performance-engineering repo path.
  --pplx-root PATH      Override pplx-garden repo path.
  --require-verify-tools  Exit early if ripgrep/detect-secrets are missing.
  -h, --help            Show this message.

The script is intentionally conservative: it previews all destructive steps
unless --execute is provided. Even in execute mode, it pauses so you can verify
that credentials/tokens have been rotated before wiping local material.
EOF
}

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_error() {
  printf '[ERROR] %s\n' "$*" >&2
}

log_section() {
  printf '\n== %s ==\n' "$*"
}

run_cmd() {
  local desc="$1"
  shift
  if [[ $DRY_RUN -eq 1 ]]; then
    log_info "[dry-run] $desc: $*"
    return 0
  fi
  log_info "$desc"
  if "$@"; then
    return 0
  fi
  local code=$?
  log_warn "$desc failed with exit code $code"
  FAILURES=1
  return "$code"
}

run_safe_cmd() {
  local desc="$1"
  shift
  log_info "$desc"
  if "$@"; then
    return 0
  fi
  local code=$?
  log_warn "$desc failed with exit code $code"
  FAILURES=1
  return "$code"
}

maybe_sudo() {
  local desc="$1"
  shift
  if [[ $EUID -eq 0 ]]; then
    run_cmd "$desc" "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    run_cmd "$desc" sudo "$@"
  else
    log_warn "Skipping '$desc' (sudo not available)"
  fi
}

setup_backup_root() {
  [[ $BACKUP_ENABLED -eq 1 ]] || { BACKUP_ROOT=""; return; }
  BACKUP_ROOT="$HOME_DIR/wipe_backups/$(date +%Y%m%d_%H%M%S)"
  if [[ $DRY_RUN -eq 1 ]]; then
    log_info "[dry-run] would create backup directory $BACKUP_ROOT"
  else
    mkdir -p "$BACKUP_ROOT"
  fi
}

backup_file() {
  local file
  for file in "$@"; do
    [[ -e "$file" ]] || continue
    [[ $BACKUP_ENABLED -eq 1 ]] || continue
    if [[ $DRY_RUN -eq 1 ]]; then
      log_info "[dry-run] backup $file -> $BACKUP_ROOT"
      continue
    fi
    if [[ -z "${BACKUP_ROOT:-}" ]]; then
      log_error "Backup root not set while backing up $file"
      exit 1
    fi
    local rel="${file#$HOME_DIR/}"
    local dest="$BACKUP_ROOT/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -a "$file" "$dest"
  done
}

remove_path() {
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    TOTAL_DELETES=$((TOTAL_DELETES + 1))
    if [[ $DRY_RUN -eq 1 ]]; then
      log_info "[dry-run][rm] $path"
      continue
    fi
    if rm -rf -- "$path"; then
      log_info "[removed] $path"
    else
      log_warn "Failed to remove $path"
      FAILURES=1
    fi
  done
}

shred_file() {
  local file
  for file in "$@"; do
    [[ -e "$file" ]] || continue
    TOTAL_SHREDS=$((TOTAL_SHREDS + 1))
    if [[ -z "$SHRED_BIN" ]]; then
      log_warn "shred not available, falling back to rm for $file"
      remove_path "$file"
      continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      log_info "[dry-run][shred] $file"
      continue
    fi
    if "$SHRED_BIN" -u "$file"; then
      log_info "[shredded] $file"
    else
      log_warn "Failed to shred $file"
      FAILURES=1
    fi
  done
}

remove_dirs_matching() {
  local root="$1"
  shift
  [[ -d "$root" ]] || return
  local name
  for name in "$@"; do
    while IFS= read -r -d '' dir; do
      remove_path "$dir"
    done < <(find "$root" -type d -name "$name" -print0 2>/dev/null)
  done
}

have_tool() {
  command -v "$1" >/dev/null 2>&1
}

check_verification_tools() {
  local warn="${1:-1}"
  local context="${2:-preflight}"
  local missing=0

  if have_tool rg; then
    RG_AVAILABLE=1
  else
    RG_AVAILABLE=0
    if [[ $warn -eq 1 ]]; then
      log_warn "[$context] ripgrep (rg) not found. Install via: sudo apt install ripgrep"
    fi
    missing=1
  fi

  if have_tool detect-secrets; then
    DETECT_AVAILABLE=1
  else
    DETECT_AVAILABLE=0
    if [[ $warn -eq 1 ]]; then
      log_warn "[$context] detect-secrets not found. Install via: pip install detect-secrets --break-system-packages"
    fi
    missing=1
  fi

  if [[ $missing -eq 1 && $REQUIRE_VERIFY_TOOLS -eq 1 ]]; then
    log_error "Verification tools are required but missing. Install ripgrep/detect-secrets or rerun without --require-verify-tools."
    exit 1
  fi
}

manual_preflight() {
  log_section "Manual prerequisites"
  log_info "1. Revoke GitHub, HuggingFace, OpenAI, Docker Hub, and Cursor tokens."
  log_info "2. Sign out of Cursor/OpenAI everywhere and invalidate API keys."
  log_info "3. Ensure SSH key rotations are complete before files are shredded."
  if [[ $DRY_RUN -eq 1 ]]; then
    log_info "Dry-run mode: manual confirmation skipped."
    return
  fi
  if [[ $ASSUME_YES -eq 1 ]]; then
    log_info "Assuming prerequisites are satisfied (--yes)."
    return
  fi
  if [[ ! -t 0 ]]; then
    log_error "Interactive confirmation required but stdin is non-interactive."
    log_error "Re-run with --yes or provide a TTY."
    exit 1
  fi
  read -r -p "Have you revoked/rotated the remote credentials listed above? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES)
      log_info "Proceeding with destructive operations."
      ;;
    *)
      log_warn "Aborting until credential rotation is complete."
      exit 1
      ;;
  esac
}

reset_shell_files() {
  log_info "Resetting ~/.bashrc and ~/.profile to minimal stubs"
  backup_file "$HOME_DIR/.bashrc" "$HOME_DIR/.profile"
  if [[ $DRY_RUN -eq 1 ]]; then
    log_info "[dry-run] would rewrite ~/.bashrc and ~/.profile"
    return
  fi
  cat <<'EOF' > "$HOME_DIR/.bashrc"
# Minimal sanitized .bashrc generated by system_wipe.sh
if [ -f /etc/bash.bashrc ]; then
  . /etc/bash.bashrc
fi

PS1='\u@\h:\w\$ '
EOF

  cat <<'EOF' > "$HOME_DIR/.profile"
# Minimal sanitized .profile generated by system_wipe.sh
if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi
EOF
}

wipe_histories() {
  log_info "Wiping shell/editor histories"
  shred_file \
    "$HOME_DIR/.bash_history" \
    "$HOME_DIR/.lesshst" \
    "$HOME_DIR/.viminfo" \
    "$HOME_DIR/.wget-hsts" \
    "$HOME_DIR/.python_history"
}

cleanup_credentials() {
  log_section "Credentials & dotfiles"
  shred_file \
    "$HOME_DIR/.gitconfig" \
    "$HOME_DIR/.git-credentials" \
    "$HOME_DIR/.config/gh/hosts.yml" \
    "$HOME_DIR/.netrc"

  shred_file \
    "$HOME_DIR/.ssh/github_rsa" \
    "$HOME_DIR/.ssh/github_rsa.pub" \
    "$HOME_DIR/.ssh/known_hosts" \
    "$HOME_DIR/.ssh/known_hosts.old"

  remove_path \
    "$HOME_DIR/.cursor" \
    "$HOME_DIR/.cursor-server" \
    "$HOME_DIR/.vscode/extensions/openai.chatgpt-0.4.38-universal"

  shred_file "$HOME_DIR/.codex/auth.json"
  remove_path \
    "$HOME_DIR/.codex/sessions" \
    "$HOME_DIR/.codex/archived_sessions"

  reset_shell_files
  remove_path "$HOME_DIR/.bash_aliases"
  wipe_histories

  remove_path \
    "$HOME_DIR/.sudo_as_admin_successful" \
    "$HOME_DIR/.pkgconfig/python3.pc"
  remove_path "$HOME_DIR/.pkgconfig"

  remove_path \
    "$HOME_DIR/Desktop/dgx-spark-developer-site.desktop" \
    "$HOME_DIR/.config/autostart/spark.desktop"
}

reset_repo() {
  local repo="$1"
  local branch="${2:-main}"
  [[ -d "$repo" ]] || { log_info "Skipping reset: $repo not found"; return; }
  if [[ -d "$repo/.git" ]]; then
    run_cmd "git fetch origin ($repo)" git -C "$repo" fetch origin
    run_cmd "git reset --hard origin/$branch ($repo)" git -C "$repo" reset --hard "origin/$branch"
    run_cmd "git clean -fdx ($repo)" git -C "$repo" clean -fdx
    remove_path "$repo/.git"
  else
    log_info "No .git directory in $repo; skipping git reset"
  fi
}

purge_ai_repo_artifacts() {
  [[ -d "$AI_REPO" ]] || return
  log_info "Removing generated artifacts from $AI_REPO"
  remove_path \
    "$AI_REPO/artifacts/runs" \
    "$AI_REPO/profiles" \
    "$AI_REPO/profiling_results" \
    "$AI_REPO/legacy_artifacts" \
    "$AI_REPO/vendor/wheels" \
    "$AI_REPO/.pytest_cache" \
    "$AI_REPO/__pycache__" \
    "$AI_REPO/.torch_extensions" \
    "$AI_REPO/.torch_inductor" \
    "$AI_REPO/a.out" \
    "$AI_REPO/pack.o" \
    "$AI_REPO/te_build.log" \
    "$AI_REPO/reports/proof_of_benefit.csv"

  remove_path \
    "$AI_REPO"/benchmark_peak_results_*.json \
    "$AI_REPO"/benchmark_test_results.* \
    "$AI_REPO"/benchmark_alignment_report.* \
    "$AI_REPO"/artifacts/*/*.log

  for lab_dir in \
    "$AI_REPO/labs/fullstack_cluster" \
    "$AI_REPO/labs/blackwell_matmul" \
    "$AI_REPO/labs/moe_cuda" \
    "$AI_REPO/labs/flexattention"
  do
    if [[ -d "$lab_dir" ]]; then
      while IFS= read -r -d '' path; do
        remove_path "$path"
      done < <(find "$lab_dir" -name 'expectations_*.json.bak' -print0 2>/dev/null)
    fi
  done

  remove_dirs_matching "$AI_REPO" "__pycache__" ".pytest_cache" ".torch_inductor" ".torch_extensions"
}

purge_pplx_repo_artifacts() {
  [[ -d "$PPLX_REPO" ]] || return
  log_info "Removing generated artifacts from $PPLX_REPO"
  remove_path \
    "$PPLX_REPO/target" \
    "$PPLX_REPO/docker" \
    "$PPLX_REPO/python/python_ext/build" \
    "$PPLX_REPO/python/python_ext/dist" \
    "$PPLX_REPO/python/python_ext"/*.egg-info \
    "$PPLX_REPO/.pytest_cache" \
    "$PPLX_REPO/__pycache__"

  remove_dirs_matching "$PPLX_REPO" "__pycache__" ".pytest_cache"
}

purge_additional_sources() {
  log_info "Removing deprecated source trees"
  remove_path \
    "$HOME_DIR/gdrcopy-2.5.1" \
    "$HOME_DIR/libfabric-1.21.0" \
    "$PPLX_REPO/sm121_story"
}

cleanup_repositories() {
  log_section "Repositories & generated artifacts"
  reset_repo "$AI_REPO" "main"
  purge_ai_repo_artifacts

  reset_repo "$PPLX_REPO" "main"
  purge_pplx_repo_artifacts

  purge_additional_sources
}

cleanup_caches_toolchains() {
  log_section "Caches & toolchains"
  remove_path \
    "$HOME_DIR/.cache/pip" \
    "$HOME_DIR/.cache/torch_extensions" \
    "$HOME_DIR/.cache/torch_extensions/py311_cu130" \
    "$HOME_DIR/.nv/ComputeCache" \
    "$HOME_DIR/.torch_extensions" \
    "$HOME_DIR/.torch_inductor" \
    "$HOME_DIR/.cache/torch_extensions"/warp_specialized_* \
    "$HOME_DIR/Documents/NVIDIA Nsight Compute"/*/Sections

  remove_path \
    "$HOME_DIR/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-base" \
    "$HOME_DIR/.cache/huggingface/hub/models--openai--gpt-oss-20b" \
    "$HOME_DIR/.cache/huggingface/xet"

  remove_path \
    "$HOME_DIR/.cargo" \
    "$HOME_DIR/.rustup"
  shred_file "$HOME_DIR/.cargo/env"

  remove_path \
    "$HOME_DIR/.config/autostart" \
    "$HOME_DIR/.cache/matplotlib" \
    "$HOME_DIR/.cache/Microsoft" \
    "$HOME_DIR/.cache/motd.legal-displayed"

  remove_path "$HOME_DIR/.docker"
}

cleanup_editors() {
  log_section "Editors & assistant caches"
  remove_dirs_matching "$HOME_DIR" ".cursor" ".cursor-server"
  remove_dirs_matching "$HOME_DIR" ".vscode" ".idea"
  remove_dirs_matching "$AI_REPO" ".cursor" ".vscode" ".idea"
  run_safe_cmd "Listing dot directories (depth<=3) for manual review" \
    find "$HOME_DIR" -maxdepth 3 -type d -name '.??*'
}

clean_tmp_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || return
  maybe_sudo "Clear $dir" find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
}

cleanup_temp_and_logs() {
  log_section "Temp files & logs"
  clean_tmp_dir "/tmp"
  clean_tmp_dir "/var/tmp"

  maybe_sudo "Rotate journalctl" journalctl --rotate
  maybe_sudo "Vacuum journalctl" journalctl --vacuum-time=1s
  maybe_sudo "Purge /var/log" rm -rf /var/log/* /var/log/journal
}

docker_cleanup() {
  log_section "Docker & GPU artifacts"
  if command -v docker >/dev/null 2>&1; then
    maybe_sudo "docker system prune -af" docker system prune -af
    maybe_sudo "docker builder prune -af" docker builder prune -af
  else
    log_info "docker not installed; skipping daemon prune"
  fi
  remove_path "$HOME_DIR/.docker"
}

verification_scans() {
  log_section "Verification sweeps"
  check_verification_tools 0 "verification"
  if [[ $RG_AVAILABLE -eq 1 ]]; then
    run_safe_cmd "ripgrep potential secrets" \
      rg -n --hidden --no-messages "(password|secret|token|key)" "$HOME_DIR"
  else
    log_warn "Skipping ripgrep scan (rg unavailable)."
  fi

  run_safe_cmd "Listing *.pem/.key/.env files" \
    find "$HOME_DIR" \( -name '*.pem' -o -name '*.key' -o -name '.env*' \) -print

  if [[ $DETECT_AVAILABLE -eq 1 ]]; then
    run_safe_cmd "detect-secrets scan" detect-secrets scan "$HOME_DIR"
  else
    log_warn "Skipping detect-secrets scan (tool unavailable)."
  fi
}




cleanup_local_tree_final() {
  log_section "Final .local cleanup"
  remove_path \
    "$HOME_DIR/.local/lib/python3.11" \
    "$HOME_DIR/.local/bin" \
    "$HOME_DIR/.local/state/wireplumber" \
    "$HOME_DIR/.local/share/applications"
}

print_summary() {
  log_section "Summary"
  log_info "Shred operations queued: $TOTAL_SHREDS"
  log_info "Delete operations queued: $TOTAL_DELETES"
  if [[ $BACKUP_ENABLED -eq 1 ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      log_info "Backups would be stored under $BACKUP_ROOT"
    else
      log_info "Backups stored under $BACKUP_ROOT"
    fi
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    log_info "Dry-run complete. Re-run with --execute when ready."
  else
    if [[ $FAILURES -eq 1 ]]; then
      log_warn "Some commands reported failures; review logs above."
    else
      log_info "Cleanup completed without command failures."
    fi
    if [[ $REBOOT_AFTER -eq 1 ]]; then
      maybe_sudo "Reboot system" reboot
    else
      log_info "Manual reboot recommended after you verify everything."
    fi
  fi
  if [[ $RG_AVAILABLE -eq 0 ]]; then
    log_warn "ripgrep unavailable -> pattern scan skipped. Install via: sudo apt install ripgrep"
  fi
  if [[ $DETECT_AVAILABLE -eq 0 ]]; then
    log_warn "detect-secrets unavailable -> plugin scan skipped. Install via: pip install detect-secrets --break-system-packages"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --execute)
        DRY_RUN=0
        ;;
      -y|--yes|--assume-yes)
        ASSUME_YES=1
        ;;
      --no-backup)
        BACKUP_ENABLED=0
        ;;
      --reboot)
        REBOOT_AFTER=1
        ;;
      --ai-root)
        shift
        [[ $# -gt 0 ]] || { log_error "--ai-root requires a path"; exit 1; }
        AI_REPO="$1"
        ;;
      --pplx-root)
        shift
        [[ $# -gt 0 ]] || { log_error "--pplx-root requires a path"; exit 1; }
        PPLX_REPO="$1"
        ;;
      --require-verify-tools)
        REQUIRE_VERIFY_TOOLS=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        log_error "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift || break
  done
}

main() {
  parse_args "$@"
  setup_backup_root
  manual_preflight
  check_verification_tools 1 "preflight"
  cleanup_credentials
  cleanup_repositories
  cleanup_caches_toolchains
  cleanup_editors
  cleanup_temp_and_logs
  docker_cleanup
  verification_scans
  cleanup_local_tree_final
  print_summary
}

main "$@"
