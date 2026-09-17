#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage: run_bench_modes.sh MODE [M N K]

MODE: current | clock-pinned | sustained | triton | all
Defaults: M=N=K=8192, CUDA_VISIBLE_DEVICES=0
EOF
}

mode=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
M=${1:-8192}
N=${2:-8192}
K=${3:-8192}
if [[ $# -gt 3 ]]; then usage >&2; exit 2; fi

case "$mode" in
    current|clock-pinned|sustained|triton|all) ;;
    *) usage >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
if [[ ! "$gpu_id" =~ ^[0-9A-Za-z:.-]+$ ]]; then
    echo "invalid CUDA_VISIBLE_DEVICES entry: $gpu_id" >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
make nvfp4-r9
bin=./out/nvfp4-r9

pin_active=0
reset_clock() {
    if [[ $pin_active == 1 ]]; then
        sudo -n nvidia-smi --id="$gpu_id" -rgc >/dev/null
        pin_active=0
    fi
}
trap reset_clock EXIT
trap 'exit 130' INT TERM

run_current() {
    "$bin" "$M" "$N" "$K" \
        --bench-mode current --rounds 5 --cooldown-sec 8
}

run_clock_pinned() {
    sudo -n nvidia-smi --id="$gpu_id" -lgc 1305,1305 >/dev/null
    pin_active=1
    local rc=0
    "$bin" "$M" "$N" "$K" \
        --bench-mode clock-pinned --rounds 5 --cooldown-sec 8 || rc=$?
    reset_clock
    return "$rc"
}

run_sustained() {
    "$bin" "$M" "$N" "$K" \
        --bench-mode sustained --rounds 3 --cooldown-sec 0
}

run_triton() {
    "$bin" "$M" "$N" "$K" \
        --bench-mode triton --rounds 1 --cooldown-sec 0
}

case "$mode" in
    current)      run_current ;;
    clock-pinned) run_clock_pinned ;;
    sustained)    run_sustained ;;
    triton)       run_triton ;;
    all)
        run_current
        run_clock_pinned
        run_sustained
        run_triton
        ;;
esac
