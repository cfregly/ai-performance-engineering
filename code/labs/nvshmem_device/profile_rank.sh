#!/bin/bash
set -eu

profile_rank="${1:?Pass the rank to profile}"
report="${2:?Pass a fresh NCU report path}"
kernel_pattern="${3:?Pass the native kernel regex}"
native_binary="${4:?Pass the native binary path}"
shift 4
case "$profile_rank" in
    0|1) ;;
    *) printf '%s\n' 'Profile rank must be zero or one' >&2; exit 2 ;;
esac
if [[ "${OMPI_COMM_WORLD_RANK:?This selector expects Open MPI rank metadata}" == "$profile_rank" ]]; then
    exec ncu --target-processes application-only \
        --replay-mode kernel --cache-control none --clock-control none \
        --metrics gpu__time_duration.sum,profiler__replayer_passes,profiler__replayer_passes_type_warmup,profiler__replayer_bytes_mem_backed_up.sum \
        --launch-count 1 --kill no --kernel-name-base demangled \
        --kernel-name "regex:$kernel_pattern" --nvtx \
        --nvtx-include 'compute_graph:nvshmem_device_steady_state/' \
        -o "$report" "$native_binary" "$@"
fi
exec "$native_binary" "$@"
