#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_cluster_eval_suite.sh --hosts <h1,h2,...> [options]

Runs a reusable "field report" eval suite:
  1) Discovery + TCP sysctl + storage layout + manifest
  2) Optional: quick friction battery (install/pull/import/download/IP/speed) on all nodes
  3) Optional: monitoring expectations snapshot (control-plane/GPU/system signals) on all nodes
  4) Required: hang-triage readiness bundle (py-spy + strace) on all nodes
  5) Required: fast torchrun NCCL connectivity probe (all nodes)
  6) Benchmark A: NCCL all_reduce_perf (single-node + multi-node)
  7) Required: NCCL env sensitivity sweep (CROSS_NIC / QPS / CTAs)
  8) Optional: cluster health suite (iperf/IB/NCCL/torchdist, with optional GDR)
  9) Benchmark B: vLLM online serving sweep (containerized, single-node; optional request-rate sweep)
  10) Optional: vLLM multinode serving benchmark (Ray, 2-node; auto-on for multi-node runs)
  11) Benchmark C: BF16 GEMM per-GPU sanity (all nodes)
  12) Benchmark D: GPU STREAM-style memory behavior probe (all nodes)
  13) Optional FP4 checks: DeepGEMM FP8xFP4 smoke + grouped GEMM (all nodes)
  14) Optional high-impact extras (ml-engineering parity):
     MAMF, all-reduce stability, all-reduce latency comp,
     all_gather_object control-plane comparison, NCCL all-to-all, NCCL algo comparison
  15) Optional: CPU<->GPU C2C memcpy benchmark (local)
  16) Optional: NUMA memory-bandwidth probe (all nodes)
  17) Optional: end-to-end transformer train-step benchmark (single-node + multi-node)
  18) Optional: checkpoint-like I/O benchmark (all nodes)
  19) Storage: fio (all nodes)
  20) Optional: nvbandwidth bundle (all nodes; auto-on for multi-node runs)
  21) Plots (includes NVLink topology) + SLO-aware vLLM goodput analysis + scorecard + MLPerf-alignment summary + manifest refresh + artifact validation

Notes:
  - GPU benchmarks are strict: they FAIL if GPU clock lock cannot be acquired.
  - Preflight is strict: it FAILS if `nvidia-persistenced`, `nvidia-dcgm`, or
    (multi-node) IMEX domain health is not ready.
  - Step execution metadata is written under
    `runs/<run_id>/structured/<run_id>_suite_steps.json`.
  - For multi-node runs, explicitly pin OOB and NCCL socket interfaces.

Options:
  --resume                  Resume from existing `suite_steps` for this run id;
                            completed steps (exit_code=0) are skipped.
  --run-id <id>            Base RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>      Comma-separated host list (required)
  --labels <l1,l2,...>     Optional labels (must match host count)
  --ssh-user <user>        SSH user (default: ubuntu)
  --ssh-key <path>         SSH key path (default: $SSH_KEY)
  --oob-if <iface>         Interface for OpenMPI OOB (required for multi-node NCCL)
  --socket-ifname <iface>  NCCL socket bootstrap interface (default: --oob-if)
  --nccl-ib-hca <list>     NCCL_IB_HCA allowlist (optional)
  --nccl-nvls-enable <0|1|2> Export NCCL_NVLS_ENABLE to NCCL ranks (health suite only; default: unset)

  --primary-label <label>  Label for single-node/local steps (default: hostname -s or first --labels entry)

  --connectivity-probe-master-port <port>  Torchrun connectivity probe rendezvous port (default: 29504)
  --connectivity-probe-barrier-iters <n>   Connectivity probe barrier iterations (default: 5)
  --connectivity-probe-payload-bytes <n>   Connectivity probe all-reduce payload bytes (default: 8388608)
  --connectivity-probe-timeout-sec <n>     Connectivity probe distributed timeout in seconds (default: 120)

  --nccl-env-min-bytes <size>   NCCL env sensitivity min bytes (default: 1M)
  --nccl-env-max-bytes <size>   NCCL env sensitivity max bytes (default: 64M)
  --nccl-env-warmup <n>         NCCL env sensitivity warmup iterations (default: 5)
  --nccl-env-iters <n>          NCCL env sensitivity measured iterations (default: 20)
  --run-nccl-env-sensitivity    Force-enable NCCL env sensitivity sweep (default: on)
  --skip-nccl-env-sensitivity   Skip NCCL env sensitivity sweep

  --run-quick-friction          Run quick friction battery on all nodes (default: on)
  --skip-quick-friction         Skip quick friction battery
  --quick-friction-strict       Fail suite if any quick friction host result is non-ok
  --quick-friction-checks <csv> Quick friction checks (default: uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest)
  --quick-friction-timeout-sec <n>  Per-check timeout for quick friction (default: 900)
  --quick-friction-torch-version <ver>  Torch version for install checks (default: 2.5.1)
  --quick-friction-torch-index-url <url>  Torch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --quick-friction-ngc-image <ref>  Container image for pull timing check (default: nvcr.io/nvidia/pytorch:24.05-py3)
  --quick-friction-hf-model <id>    HF model for download timing check (default: openai-community/gpt2)
  --quick-friction-hf-local-dir-base <path>  Base temp dir for HF download check (default: /tmp)
  --quick-friction-allow-failed-checks <csv>  Classify these quick-friction check failures as expected (default: auto for localhost, none otherwise)

  --render-localhost-report      Force rendering run-local localhost reports plus published `cluster/field-report-localhost.md` + notes when localhost package is detected
  --skip-render-localhost-report Disable localhost field-report package rendering

  --run-monitoring-expectations      Run monitoring expectations snapshot on all nodes (default: on)
  --skip-monitoring-expectations     Skip monitoring expectations snapshot
  --monitoring-expectations-strict   Fail suite if any monitoring snapshot host result is non-ok
  --monitoring-checks <csv>          Monitoring checks (default: kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail)
  --monitoring-k8s-mode <mode>       Control-plane expectations mode: auto|expect|skip (default: auto)
  --monitoring-sample-count <n>      Sample count for dmon checks (default: 20)
  --monitoring-dmesg-lines <n>       dmesg tail lines to capture (default: 400)
  --monitoring-timeout-sec <n>       Per-check timeout for monitoring checks (default: 180)

  --model <hf_model_id>    vLLM model id (default: openai/gpt-oss-120b)
  --tp <n>                 vLLM tensor parallel (default: all visible GPUs)
  --isl <n>                vLLM input seq len (default: 1024)
  --osl <n>                vLLM output seq len (default: 1024)
  --concurrency-range "…"  vLLM concurrencies (default: "32 64 128 256 512")
  --vllm-repeats <n>       Number of vLLM concurrency sweep repetitions (default: 1)
  --vllm-max-points-per-run <n>
                           Resume-segmentation budget for vLLM sweep points per invocation.
                           0 runs all points in one invocation (default: 0).
  --run-vllm-request-rate-sweep    Run single-node vLLM request-rate sweep (default: off)
  --skip-vllm-request-rate-sweep   Skip single-node vLLM request-rate sweep
  --vllm-request-rate-range "..."  Request-rate sweep values (default: "1 2 4 8 16")
  --vllm-request-rate-repeats <n>  Number of vLLM request-rate sweep repetitions (default: 1)
  --vllm-request-rate-max-concurrency <n>  Max concurrency cap for request-rate sweep (default: 256)
  --vllm-request-rate-num-prompts <n>      Prompts per request-rate point (default: max_concurrency*20)
  --vllm-max-conc-cv-p95-pct <pct>         Optional fail threshold for concurrency sweep tok/s CV p95
  --vllm-max-rate-cv-p95-pct <pct>         Optional fail threshold for request-rate sweep tok/s CV p95
  --port <port>            vLLM server port (default: 8888)
  --vllm-slo-p99-ttft-ms <ms>  SLO threshold for vLLM p99 TTFT (default: 2000)
  --vllm-slo-p99-tpot-ms <ms>  SLO threshold for vLLM p99 TPOT (default: 200)
  --run-vllm-multinode     Force-enable 2-node vLLM serving benchmark via Ray
  --skip-vllm-multinode    Force-disable 2-node vLLM serving benchmark via Ray
  --vllm-multinode-concurrency <n>  Multinode vLLM max concurrency (single-point mode, default: 64)
  --vllm-multinode-concurrency-range "..."
                           Multinode vLLM concurrency sweep values (space/comma-separated).
                           When set, runs one multinode serve pass per value.
  --vllm-multinode-num-prompts <n>  Multinode vLLM prompt count (default: concurrency*10)
  --vllm-multinode-ray-port <port>  Multinode vLLM Ray head port (default: 6379)
  --vllm-multinode-image <image>    Multinode vLLM container image (default: auto by arch)
  --vllm-multinode-ray-timeout <sec>     Timeout waiting for Ray cluster (default: 300)
  --vllm-multinode-server-timeout <sec>  Timeout waiting for vLLM health (default: 1200)
  --vllm-multinode-worker-startup-wait <sec>  Delay before leader launch (default: 10)

  --enable-fp4             Run FP4 checks on all nodes (default: on)
  --disable-fp4            Disable FP4 checks
  --fp4-runtime <mode>     FP4 execution mode: host|container (default: host)
  --fp4-stack-profile <name>
                           FP4 stack profile: old_container|orig_parity_container|new_container|host_only
                           (default: runtime-specific from configs/cluster_perf_stack_profiles.json)
  --fp4-image <image>      FP4 container image for --fp4-runtime container
                           (default: fp4-stack-profile image_ref or $CONTAINER_IMAGE)
  --fp4-preset <name>      Grouped-GEMM preset for FP4 check (default: auto; GB-family uses all)
  --fp4-warmup <n>         Grouped-GEMM warmup for FP4 check (default: 5)
  --fp4-iters <n>          Grouped-GEMM measured iterations for FP4 check (default: 30)
  --fp4-smoke-m <int>      FP4 smoke shape M (default: 4096)
  --fp4-smoke-n <int>      FP4 smoke shape N (default: 4096)
  --fp4-smoke-k <int>      FP4 smoke shape K (default: 4096)
  --fp4-smoke-warmup <n>   FP4 smoke warmup iterations (default: 10)
  --fp4-smoke-iters <n>    FP4 smoke measured iterations (default: 30)
  --fp4-smoke-rounds <n>   FP4 paired smoke rounds per host (default: 3)
  --fp4-smoke-skew-threshold-pct <pct>
                           Fail FP4 smoke guard when max pairwise median gap exceeds this percent (default: 5)

  --bootstrap-nodes                Bootstrap all nodes before checks (default: on)
  --skip-bootstrap-nodes           Skip node bootstrap
  --bootstrap-install-system-packages   Install missing system deps during bootstrap (default: on)
  --bootstrap-skip-system-packages      Skip system package install during bootstrap
  --bootstrap-sync-code            Sync scripts/analysis/env requirements to remote nodes (default: on)
  --bootstrap-skip-sync-code       Skip code sync during bootstrap
  --bootstrap-install-python-deps  Ensure env/venv + Python deps during bootstrap (default: on)
  --bootstrap-skip-python-deps     Skip python dep install during bootstrap
  --bootstrap-host-parity-image <ref>  Source image for host-only parity install
                                        (default: cluster_perf_orig_parity:latest)
  --bootstrap-torch-index-url <url>  Legacy fallback torch index (default: https://pypi.ngc.nvidia.com)
  --bootstrap-torch-version <ver>    Expected torch version after bootstrap parity install
                                     (default: 2.10.0a0+a36e1d39eb.nv26.01.42222806)

  --fio-test-dir <path>    fio directory (default: /tmp)
  --fio-file-size <size>   fio size per job (default: 8G)
  --fio-runtime <sec>      fio runtime per test (default: 30)
  --fio-repeats <n>        fio repetitions per host (default: 1)
  --fio-max-seq-bw-cv-pct <pct>  Optional fail threshold for fio seq read/write BW CV
  --run-nvbandwidth        Force-enable nvbandwidth bundle (all nodes)
  --skip-nvbandwidth       Force-disable nvbandwidth bundle
  --nvbandwidth-runtime <host|container>  nvbandwidth runtime (default: host)
  --nvbandwidth-image <image>             nvbandwidth image for runtime=container
                                          (default: cluster_perf_orig_parity:latest)
  --nvbandwidth-bin <path>                nvbandwidth executable for runtime=host (default: nvbandwidth)
  --nvbandwidth-quick      Use reduced nvbandwidth testcase subset

  --run-gpu-stream         Force-enable GPU STREAM-style benchmark (all nodes; default: on)
  --skip-gpu-stream        Force-disable GPU STREAM-style benchmark
  --gpu-stream-device <n>  CUDA device index for STREAM benchmark (default: 0)
  --gpu-stream-size-mb <n> STREAM benchmark vector size in MB (default: 1024)
  --gpu-stream-iters <n>   STREAM benchmark measured iterations per op (default: 40)
  --gpu-stream-warmup <n>  STREAM benchmark warmup iterations per op (default: 10)
  --gpu-stream-dtype <d>   STREAM benchmark dtype: fp32|fp16|bf16 (default: fp32)

  --health-suite <mode>    Optional multi-node diagnostics:
                           off|collectives|base|extended (default: collectives)
  --health-gdr             Enable GPUDirect RDMA checks inside health suite (default: off)
  --health-gdr-gpu <id>    CUDA device id for health-suite GDR checks (default: 0)
  --health-gdr-mem-types <csv>  CUDA mem types for GDR checks (default: 0)
  --health-gdr-use-dmabuf  Also run health-suite GDR checks with --use_cuda_dmabuf

  --enable-mamf            Run MAMF compute-depth scan (default: off)
  --mamf-mode <mode>       MAMF mode: quick|medium|thorough (default: quick)
  --mamf-concurrent        Run MAMF on all GPUs concurrently (straggler focus)

  --enable-allreduce-stability  Run all-reduce jitter/stability profile (default: off)
  --disable-allreduce-stability Force-disable all-reduce jitter/stability (overrides modern profile)
  --allreduce-payload-gib <f>   Stability payload GiB (default: 2.0)
  --allreduce-iters <n>         Stability iterations (default: 200)
  --allreduce-warmup <n>        Stability warmup iterations (default: 20)

  --enable-allreduce-latency-comp  Run 1x-large vs many-small all-reduce comparison (default: off)
  --allreduce-latency-payload-gib <f>  Latency comparison total payload GiB (default: 4.0)
  --allreduce-latency-chunks <n>       Latency comparison chunk count (default: 1000)
  --allreduce-latency-iters <n>        Latency comparison iterations (default: 5)
  --allreduce-latency-warmup <n>       Latency comparison warmup iterations (default: 1)

  --enable-allgather-control-plane  Run all_gather_object vs tensor collectives benchmark (default: off)
  --allgather-control-iters <n>      Control-plane benchmark iterations (default: 2000)
  --allgather-control-warmup <n>     Control-plane benchmark warmup iterations (default: 200)
  --enable-nccl-alltoall         Run NCCL alltoall_perf benchmark (default: off)
  --disable-nccl-alltoall        Force-disable NCCL alltoall_perf (overrides modern profile)
  --nccl-alltoall-min-bytes <size>  NCCL alltoall min bytes (default: 1M)
  --nccl-alltoall-max-bytes <size>  NCCL alltoall max bytes (default: 64M)
  --nccl-alltoall-warmup <n>        NCCL alltoall warmup iterations (default: 5)
  --nccl-alltoall-iters <n>         NCCL alltoall measured iterations (default: 20)

  --enable-nccl-algo-comparison  Run NCCL Ring/Tree/NVLS/auto comparison (default: off)
  --disable-nccl-algo-comparison Force-disable NCCL algo comparison (overrides modern profile)
  --nccl-algos <list>            NCCL algorithms for comparison (default: Ring,Tree,NVLS,auto)
  --run-fabric-eval              Run the fabric verification ladder and scorecard (default: off)
  --skip-fabric-eval             Skip the fabric verification ladder
  --nmx-url <url>                NMX base URL for fabric/NVLink management-plane checks
  --nmx-token <token>            Optional NMX bearer token for management-plane checks
  --ib-mgmt-host <host>          InfiniBand management host for fabric CLI verification
  --ib-mgmt-user <user>          SSH user for the InfiniBand management host
  --ib-mgmt-ssh-key <path>       SSH key path for the InfiniBand management host
  --cumulus-hosts <hosts>        Comma-separated Cumulus/Spectrum-X switch hosts
  --cumulus-user <user>          SSH user for Cumulus/Spectrum-X switches
  --cumulus-ssh-key <path>       SSH key path for Cumulus/Spectrum-X switches
  --require-management-plane     Require IB/NMX/Cumulus management-plane coverage for publish-grade fabric runs
  --modern-llm-profile           Enable a modern high-signal LLM profile:
                                 request-rate sweep on, nvbandwidth on,
                                 allreduce stability on, NCCL all-to-all on, NCCL algo comparison on,
                                 allreduce latency-comp on, allgather control-plane on,
                                 torchrun train-step on,
                                 strict canonical completeness gating on,
                                 vLLM/fio repeats promoted to at least 3

  --gpu-hourly-cost-usd <usd>    Optional single-GPU hourly cost for scorecard cost-per-1M-token metrics
  --coverage-baseline-run-id <id>
                                  Optional baseline run ID to emit coverage/scorecard delta artifact
  --strict-multinode-readiness   Enforce fail-fast multi-node readiness invariants when hosts>1 (default: on)
  --no-strict-multinode-readiness
                                  Allow warning/skip behavior for missing multi-node prerequisites
  --multinode-readiness-check-only
                                  Validate multi-node readiness + emit readiness artifact, then exit (no workloads)
  --strict-canonical-completeness  Require semantic completeness thresholds for canonical output artifacts
  --no-strict-canonical-completeness
                                  Disable semantic completeness thresholds (artifacts still validated)
  --canonical-min-coverage-pct <pct>
                                  Minimum subsystem coverage score when strict completeness is enabled (default: 100)
  --canonical-min-advanced-coverage-pct <pct>
                                  Minimum advanced coverage score when strict completeness is enabled (default: 85)
  --partial-resume-max-attempts <n>
                                  Max auto-resume attempts for rc=75 resumable steps within one suite invocation (default: 64)
  --disable-auto-partial-resume  Disable automatic in-process retry for rc=75 resumable steps

  --check-ib-sharp         Optional multi-node check: SHARP userspace + forced NCCL CollNet all-reduce
                           (runs scripts/check_ib_sharp.sh; default: off)
  --ib-sharp-attempt-start-sharp-am  Best-effort: attempt to install/start sharp_am before rerunning the forced CollNet check
  --ib-sharp-am-host <h>   Host to start sharp_am on (default: first host in --hosts)

  --run-c2c                Run CPU<->GPU C2C memcpy benchmark on primary node (default: off)
  --c2c-device <id>        C2C benchmark GPU device (default: 0)
  --c2c-bw-sizes <csv>     C2C bandwidth sizes in bytes (default: 4194304,67108864,1073741824)
  --c2c-lat-sizes <csv>    C2C latency sizes in bytes (default: 4,4096,65536)
  --c2c-bw-iters <n>       C2C bandwidth iterations (default: 20)
  --c2c-lat-iters <n>      C2C latency iterations (default: 20000)
  --c2c-warmup <n>         C2C warmup iterations (default: 5)

  --run-numa-mem-bw        Run NUMA memory-bandwidth probe on all nodes (default: off)
  --numa-bytes <n>         NUMA probe bytes per iteration (default: 1073741824)
  --numa-iters <n>         NUMA probe iterations (default: 10)
  --numa-threads <n>       NUMA probe thread count (default: 16)
  --numa-warmup <n>        NUMA probe warmup iterations (default: 2)
  --numa-nodes <csv>       Optional explicit NUMA nodes to probe (default: auto)
  --numa-cpu-node <n>      Optional CPU NUMA node to bind (default: auto)

  --run-train-step         Run torchrun transformer train-step benchmark (default: off)
  --train-step-single-node Also run a single-node train-step baseline on primary host (default: on when --run-train-step)
  --train-step-multi-node  Run multi-node train-step on --hosts (default: on when --run-train-step)
  --train-master-port <p>  Train-step torchrun rendezvous port (default: 29510)
  --train-steps <n>        Train-step measured steps (default: 30)
  --train-warmup-steps <n> Train-step warmup steps (default: 5)
  --train-batch-size <n>   Train-step per-rank batch size (default: 2)
  --train-seq-len <n>      Train-step sequence length (default: 2048)
  --train-hidden <n>       Train-step hidden size (default: 4096)
  --train-layers <n>       Train-step layers (default: 24)
  --train-heads <n>        Train-step attention heads (default: 32)
  --train-mlp-ratio <n>    Train-step MLP ratio (default: 4)
  --train-precision <bf16|fp16>  Train-step precision (default: bf16)
  --train-fsdp <0|1>       Train-step FSDP FULL_SHARD enable (default: 1)
  --train-lr <float>       Train-step AdamW learning rate (default: 1e-4)

  --run-checkpoint-io      Run checkpoint-like write/read benchmark on all nodes (default: off)
  --checkpoint-test-dir <path>  Checkpoint benchmark directory (default: /tmp)
  --checkpoint-bytes <size>     Bytes per checkpoint file (default: 4G)
  --checkpoint-block-size <size>  Block size (default: 4M)
  --checkpoint-files <n>        Number of files (default: 1)
  --checkpoint-fsync <0|1>      fsync after write (default: 1)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -f "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" ]]; then
  echo "ERROR: missing stack profile helper: ${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" >&2
  exit 1
fi
# shellcheck source=scripts/cluster_perf_stack_profiles.sh
source "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh"

QUEUE_RUNNER_LOCK_PATH="${AISP_CLUSTER_SUITE_QUEUE_LOCK_PATH:-${CODE_ROOT}/artifacts/parallel_runs/queue.runner.lock}"
QUEUE_RUNNER_LOCK_TIMEOUT_SEC="${AISP_CLUSTER_SUITE_QUEUE_LOCK_TIMEOUT_SEC:-0}"
QUEUE_RUNNER_LOCK_FD=""
QUEUE_RUNNER_LOCK_HELD=0

acquire_queue_runner_lock() {
  local lock_dir timeout_sec
  timeout_sec="${QUEUE_RUNNER_LOCK_TIMEOUT_SEC}"
  if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]]; then
    echo "ERROR: AISP_CLUSTER_SUITE_QUEUE_LOCK_TIMEOUT_SEC must be a non-negative integer (got: ${timeout_sec})" >&2
    exit 2
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required for strict suite/benchmark mutual exclusion." >&2
    exit 1
  fi
  lock_dir="$(dirname "${QUEUE_RUNNER_LOCK_PATH}")"
  mkdir -p "$lock_dir"
  touch "${QUEUE_RUNNER_LOCK_PATH}"
  exec {QUEUE_RUNNER_LOCK_FD}>"${QUEUE_RUNNER_LOCK_PATH}"
  if [[ "$timeout_sec" -eq 0 ]]; then
    if ! flock -n "$QUEUE_RUNNER_LOCK_FD"; then
      echo "ERROR: queue lock busy at ${QUEUE_RUNNER_LOCK_PATH}; refusing overlapping run." >&2
      echo "Remediation: stop active bench/profile workloads, then rerun cluster eval suite." >&2
      exit 1
    fi
  else
    if ! flock -w "$timeout_sec" "$QUEUE_RUNNER_LOCK_FD"; then
      echo "ERROR: timed out after ${timeout_sec}s waiting for queue lock ${QUEUE_RUNNER_LOCK_PATH}." >&2
      echo "Remediation: stop active bench/profile workloads, then rerun cluster eval suite." >&2
      exit 1
    fi
  fi
  QUEUE_RUNNER_LOCK_HELD=1
  export AISP_CLUSTER_SUITE_QUEUE_LOCK_HELD=1
  printf '{"owner":"run_cluster_eval_suite.sh","pid":%s,"ts":"%s"}\n' "$$" "$(date -Iseconds)" >"${QUEUE_RUNNER_LOCK_PATH}" || true
}

release_queue_runner_lock() {
  if [[ "$QUEUE_RUNNER_LOCK_HELD" -ne 1 || -z "${QUEUE_RUNNER_LOCK_FD}" ]]; then
    return 0
  fi
  flock -u "$QUEUE_RUNNER_LOCK_FD" || true
  eval "exec ${QUEUE_RUNNER_LOCK_FD}>&-"
  QUEUE_RUNNER_LOCK_HELD=0
  QUEUE_RUNNER_LOCK_FD=""
  unset AISP_CLUSTER_SUITE_QUEUE_LOCK_HELD || true
}

RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
SOCKET_IFNAME=""
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_NVLS_ENABLE=""

PRIMARY_LABEL=""

CONNECTIVITY_PROBE_MASTER_PORT="29504"
CONNECTIVITY_PROBE_BARRIER_ITERS="5"
CONNECTIVITY_PROBE_PAYLOAD_BYTES="8388608"
CONNECTIVITY_PROBE_TIMEOUT_SEC="120"

NCCL_ENV_MIN_BYTES="1M"
NCCL_ENV_MAX_BYTES="64M"
NCCL_ENV_WARMUP="5"
NCCL_ENV_ITERS="20"
RUN_NCCL_ENV_SENSITIVITY=1

RUN_QUICK_FRICTION=1
QUICK_FRICTION_STRICT=0
QUICK_FRICTION_CHECKS="uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest"
QUICK_FRICTION_TIMEOUT_SEC="900"
QUICK_FRICTION_TORCH_VERSION="2.5.1"
QUICK_FRICTION_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
QUICK_FRICTION_NGC_IMAGE="nvcr.io/nvidia/pytorch:24.05-py3"
QUICK_FRICTION_HF_MODEL="openai-community/gpt2"
QUICK_FRICTION_HF_LOCAL_DIR_BASE="/tmp"
QUICK_FRICTION_ALLOW_FAILED_CHECKS=""

RUN_MONITORING_EXPECTATIONS=1
MONITORING_EXPECTATIONS_STRICT=0
MONITORING_CHECKS="kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail"
MONITORING_K8S_MODE="auto"
MONITORING_SAMPLE_COUNT="20"
MONITORING_DMESG_LINES="400"
MONITORING_TIMEOUT_SEC="180"

MODEL="openai/gpt-oss-120b"
TP=""
ISL="1024"
OSL="1024"
CONCURRENCY_RANGE="32 64 128 256 512"
VLLM_REPEATS="1"
VLLM_MAX_POINTS_PER_RUN="0"
PORT="8888"
VLLM_SLO_P99_TTFT_MS="2000"
VLLM_SLO_P99_TPOT_MS="200"
RUN_VLLM_REQUEST_RATE_SWEEP=0
VLLM_REQUEST_RATE_RANGE="1 2 4 8 16"
VLLM_REQUEST_RATE_REPEATS="1"
VLLM_REQUEST_RATE_MAX_CONCURRENCY="256"
VLLM_REQUEST_RATE_NUM_PROMPTS=""
VLLM_MAX_CONC_CV_P95_PCT=""
VLLM_MAX_RATE_CV_P95_PCT=""
RUN_VLLM_MULTINODE_MODE="auto"
RUN_VLLM_MULTINODE=0
VLLM_MULTINODE_CONCURRENCY="64"
VLLM_MULTINODE_CONCURRENCY_RANGE=""
VLLM_MULTINODE_NUM_PROMPTS=""
VLLM_MULTINODE_RAY_PORT="6379"
VLLM_MULTINODE_IMAGE=""
VLLM_MULTINODE_RAY_TIMEOUT="300"
VLLM_MULTINODE_SERVER_TIMEOUT="1200"
VLLM_MULTINODE_WORKER_STARTUP_WAIT="10"
VLLM_MULTINODE_CONCURRENCY_VALUES=()

ENABLE_FP4=1
FP4_RUNTIME="host"
FP4_STACK_PROFILE=""
FP4_IMAGE="${CONTAINER_IMAGE:-}"
FP4_PRESET="auto"
FP4_WARMUP="5"
FP4_ITERS="30"
FP4_SMOKE_M="4096"
FP4_SMOKE_N="4096"
FP4_SMOKE_K="4096"
FP4_SMOKE_WARMUP="10"
FP4_SMOKE_ITERS="30"
FP4_SMOKE_ROUNDS="3"
FP4_SMOKE_SKEW_THRESHOLD_PCT="5"

BOOTSTRAP_NODES=1
BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1
BOOTSTRAP_SYNC_CODE=1
BOOTSTRAP_INSTALL_PYTHON_DEPS=1
BOOTSTRAP_HOST_PARITY_IMAGE="${BOOTSTRAP_HOST_PARITY_IMAGE:-cluster_perf_orig_parity:latest}"
BOOTSTRAP_TORCH_INDEX_URL="https://pypi.ngc.nvidia.com"
BOOTSTRAP_TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"

RENDER_LOCALHOST_REPORT_MODE="auto"

FIO_TEST_DIR="/tmp"
FIO_FILE_SIZE="8G"
FIO_RUNTIME="30"
FIO_REPEATS="1"
FIO_MAX_SEQ_BW_CV_PCT=""
RUN_NVBANDWIDTH_MODE="auto"
RUN_NVBANDWIDTH=0
NVBANDWIDTH_RUNTIME="host"
NVBANDWIDTH_IMAGE="cluster_perf_orig_parity:latest"
NVBANDWIDTH_BIN="nvbandwidth"
NVBANDWIDTH_QUICK=0
RUN_GPU_STREAM_MODE="on"
RUN_GPU_STREAM=1
GPU_STREAM_DEVICE="0"
GPU_STREAM_SIZE_MB="1024"
GPU_STREAM_ITERS="40"
GPU_STREAM_WARMUP="10"
GPU_STREAM_DTYPE="fp32"

HEALTH_SUITE_MODE="collectives"
HEALTH_GDR=0
HEALTH_GDR_GPU="0"
HEALTH_GDR_MEM_TYPES="0"
HEALTH_GDR_USE_DMABUF=0
CHECK_IB_SHARP=0
IB_SHARP_ATTEMPT_START_SHARP_AM=0
IB_SHARP_AM_HOST=""

RUN_C2C=0
C2C_DEVICE="0"
C2C_BW_SIZES="4194304,67108864,1073741824"
C2C_LAT_SIZES="4,4096,65536"
C2C_BW_ITERS="20"
C2C_LAT_ITERS="20000"
C2C_WARMUP="5"

RUN_NUMA_MEM_BW=0
NUMA_BYTES="1073741824"
NUMA_ITERS="10"
NUMA_THREADS="16"
NUMA_WARMUP="2"
NUMA_NODES=""
NUMA_CPU_NODE=""

RUN_TRAIN_STEP=0
TRAIN_STEP_SINGLE_NODE=1
TRAIN_STEP_MULTI_NODE=1
TRAIN_MASTER_PORT="29510"
TRAIN_STEPS="30"
TRAIN_WARMUP_STEPS="5"
TRAIN_BATCH_SIZE="2"
TRAIN_SEQ_LEN="2048"
TRAIN_HIDDEN="4096"
TRAIN_LAYERS="24"
TRAIN_HEADS="32"
TRAIN_MLP_RATIO="4"
TRAIN_PRECISION="bf16"
TRAIN_FSDP="1"
TRAIN_LR="1e-4"

RUN_CHECKPOINT_IO=0
CHECKPOINT_TEST_DIR="/tmp"
CHECKPOINT_BYTES="4G"
CHECKPOINT_BLOCK_SIZE="4M"
CHECKPOINT_FILES="1"
CHECKPOINT_FSYNC="1"

ENABLE_MAMF=0
MAMF_MODE="quick"
MAMF_CONCURRENT=0
ENABLE_ALLREDUCE_STABILITY=0
ALLREDUCE_PAYLOAD_GIB="2.0"
ALLREDUCE_ITERS="200"
ALLREDUCE_WARMUP="20"
ENABLE_ALLREDUCE_LATENCY_COMP=0
ALLREDUCE_LATENCY_PAYLOAD_GIB="4.0"
ALLREDUCE_LATENCY_CHUNKS="1000"
ALLREDUCE_LATENCY_ITERS="5"
ALLREDUCE_LATENCY_WARMUP="1"
ENABLE_ALLGATHER_CONTROL_PLANE=0
ALLGATHER_CONTROL_ITERS="2000"
ALLGATHER_CONTROL_WARMUP="200"
ENABLE_NCCL_ALLTOALL=0
NCCL_ALLTOALL_MIN_BYTES="1M"
NCCL_ALLTOALL_MAX_BYTES="64M"
NCCL_ALLTOALL_WARMUP="5"
NCCL_ALLTOALL_ITERS="20"
ENABLE_NCCL_ALGO_COMPARISON=0
NCCL_ALGOS="Ring,Tree,NVLS,auto"
RUN_FABRIC_EVAL=0
REQUIRE_MANAGEMENT_PLANE=0
FABRIC_NMX_URL=""
FABRIC_NMX_TOKEN=""
FABRIC_IB_MGMT_HOST=""
FABRIC_IB_MGMT_USER=""
FABRIC_IB_MGMT_SSH_KEY=""
FABRIC_CUMULUS_HOSTS=""
FABRIC_CUMULUS_USER=""
FABRIC_CUMULUS_SSH_KEY=""
MODERN_LLM_PROFILE=0
MODERN_LLM_PROFILE_EXPLICIT=0
GPU_HOURLY_COST_USD=""
COVERAGE_BASELINE_RUN_ID=""
STRICT_MULTINODE_READINESS=1
MULTINODE_READINESS_CHECK_ONLY=0
STRICT_CANONICAL_COMPLETENESS=0
CANONICAL_MIN_COVERAGE_PCT="100"
CANONICAL_MIN_ADVANCED_COVERAGE_PCT="85"
AUTO_RESUME_PARTIAL_STEPS=1
PARTIAL_RESUME_MAX_ATTEMPTS="64"
RESUME_RUN=0
FORCE_DISABLE_ALLREDUCE_STABILITY=0
FORCE_DISABLE_NCCL_ALLTOALL=0
FORCE_DISABLE_NCCL_ALGO_COMPARISON=0
FORCE_DISABLE_VLLM_REQUEST_RATE_SWEEP=0
FORCE_DISABLE_STRICT_CANONICAL_COMPLETENESS=0
RUN_TRAIN_STEP_EXPLICIT=0
VLLM_REPEATS_EXPLICIT=0
VLLM_REQUEST_RATE_REPEATS_EXPLICIT=0
FIO_REPEATS_EXPLICIT=0
MODEL_EXPLICIT=0
TP_EXPLICIT=0
ISL_EXPLICIT=0
OSL_EXPLICIT=0
CONCURRENCY_RANGE_EXPLICIT=0
RUN_VLLM_REQUEST_RATE_SWEEP_EXPLICIT=0
VLLM_REQUEST_RATE_RANGE_EXPLICIT=0
VLLM_REQUEST_RATE_MAX_CONCURRENCY_EXPLICIT=0
VLLM_REQUEST_RATE_NUM_PROMPTS_EXPLICIT=0
VLLM_PROFILE_CLASS="suite_default"
VLLM_PROFILE_SELECTION_REASON="default suite vLLM contract"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --resume) RESUME_RUN=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --nccl-nvls-enable) NCCL_NVLS_ENABLE="$2"; shift 2 ;;

    --primary-label) PRIMARY_LABEL="$2"; shift 2 ;;
    --connectivity-probe-master-port) CONNECTIVITY_PROBE_MASTER_PORT="$2"; shift 2 ;;
    --connectivity-probe-barrier-iters) CONNECTIVITY_PROBE_BARRIER_ITERS="$2"; shift 2 ;;
    --connectivity-probe-payload-bytes) CONNECTIVITY_PROBE_PAYLOAD_BYTES="$2"; shift 2 ;;
    --connectivity-probe-timeout-sec) CONNECTIVITY_PROBE_TIMEOUT_SEC="$2"; shift 2 ;;
    --nccl-env-min-bytes) NCCL_ENV_MIN_BYTES="$2"; shift 2 ;;
    --nccl-env-max-bytes) NCCL_ENV_MAX_BYTES="$2"; shift 2 ;;
    --nccl-env-warmup) NCCL_ENV_WARMUP="$2"; shift 2 ;;
    --nccl-env-iters) NCCL_ENV_ITERS="$2"; shift 2 ;;
    --run-nccl-env-sensitivity) RUN_NCCL_ENV_SENSITIVITY=1; shift ;;
    --skip-nccl-env-sensitivity) RUN_NCCL_ENV_SENSITIVITY=0; shift ;;

    --run-quick-friction) RUN_QUICK_FRICTION=1; shift ;;
    --skip-quick-friction) RUN_QUICK_FRICTION=0; shift ;;
    --quick-friction-strict) QUICK_FRICTION_STRICT=1; shift ;;
    --quick-friction-checks) QUICK_FRICTION_CHECKS="$2"; shift 2 ;;
    --quick-friction-timeout-sec) QUICK_FRICTION_TIMEOUT_SEC="$2"; shift 2 ;;
    --quick-friction-torch-version) QUICK_FRICTION_TORCH_VERSION="$2"; shift 2 ;;
    --quick-friction-torch-index-url) QUICK_FRICTION_TORCH_INDEX_URL="$2"; shift 2 ;;
    --quick-friction-ngc-image) QUICK_FRICTION_NGC_IMAGE="$2"; shift 2 ;;
    --quick-friction-hf-model) QUICK_FRICTION_HF_MODEL="$2"; shift 2 ;;
    --quick-friction-hf-local-dir-base) QUICK_FRICTION_HF_LOCAL_DIR_BASE="$2"; shift 2 ;;
    --quick-friction-allow-failed-checks) QUICK_FRICTION_ALLOW_FAILED_CHECKS="$2"; shift 2 ;;
    --render-localhost-report) RENDER_LOCALHOST_REPORT_MODE="on"; shift ;;
    --skip-render-localhost-report) RENDER_LOCALHOST_REPORT_MODE="off"; shift ;;

    --run-monitoring-expectations) RUN_MONITORING_EXPECTATIONS=1; shift ;;
    --skip-monitoring-expectations) RUN_MONITORING_EXPECTATIONS=0; shift ;;
    --monitoring-expectations-strict) MONITORING_EXPECTATIONS_STRICT=1; shift ;;
    --monitoring-checks) MONITORING_CHECKS="$2"; shift 2 ;;
    --monitoring-k8s-mode) MONITORING_K8S_MODE="$2"; shift 2 ;;
    --monitoring-sample-count) MONITORING_SAMPLE_COUNT="$2"; shift 2 ;;
    --monitoring-dmesg-lines) MONITORING_DMESG_LINES="$2"; shift 2 ;;
    --monitoring-timeout-sec) MONITORING_TIMEOUT_SEC="$2"; shift 2 ;;

    --model) MODEL="$2"; MODEL_EXPLICIT=1; shift 2 ;;
    --tp) TP="$2"; TP_EXPLICIT=1; shift 2 ;;
    --isl) ISL="$2"; ISL_EXPLICIT=1; shift 2 ;;
    --osl) OSL="$2"; OSL_EXPLICIT=1; shift 2 ;;
    --concurrency-range) CONCURRENCY_RANGE="$2"; CONCURRENCY_RANGE_EXPLICIT=1; shift 2 ;;
    --vllm-repeats) VLLM_REPEATS="$2"; VLLM_REPEATS_EXPLICIT=1; shift 2 ;;
    --vllm-max-points-per-run) VLLM_MAX_POINTS_PER_RUN="$2"; shift 2 ;;
    --run-vllm-request-rate-sweep) RUN_VLLM_REQUEST_RATE_SWEEP=1; RUN_VLLM_REQUEST_RATE_SWEEP_EXPLICIT=1; FORCE_DISABLE_VLLM_REQUEST_RATE_SWEEP=0; shift ;;
    --skip-vllm-request-rate-sweep) RUN_VLLM_REQUEST_RATE_SWEEP=0; RUN_VLLM_REQUEST_RATE_SWEEP_EXPLICIT=1; FORCE_DISABLE_VLLM_REQUEST_RATE_SWEEP=1; shift ;;
    --vllm-request-rate-range) VLLM_REQUEST_RATE_RANGE="$2"; VLLM_REQUEST_RATE_RANGE_EXPLICIT=1; shift 2 ;;
    --vllm-request-rate-repeats) VLLM_REQUEST_RATE_REPEATS="$2"; VLLM_REQUEST_RATE_REPEATS_EXPLICIT=1; shift 2 ;;
    --vllm-request-rate-max-concurrency) VLLM_REQUEST_RATE_MAX_CONCURRENCY="$2"; VLLM_REQUEST_RATE_MAX_CONCURRENCY_EXPLICIT=1; shift 2 ;;
    --vllm-request-rate-num-prompts) VLLM_REQUEST_RATE_NUM_PROMPTS="$2"; VLLM_REQUEST_RATE_NUM_PROMPTS_EXPLICIT=1; shift 2 ;;
    --vllm-max-conc-cv-p95-pct) VLLM_MAX_CONC_CV_P95_PCT="$2"; shift 2 ;;
    --vllm-max-rate-cv-p95-pct) VLLM_MAX_RATE_CV_P95_PCT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --vllm-slo-p99-ttft-ms) VLLM_SLO_P99_TTFT_MS="$2"; shift 2 ;;
    --vllm-slo-p99-tpot-ms) VLLM_SLO_P99_TPOT_MS="$2"; shift 2 ;;
    --run-vllm-multinode) RUN_VLLM_MULTINODE_MODE="on"; shift ;;
    --skip-vllm-multinode) RUN_VLLM_MULTINODE_MODE="off"; shift ;;
    --vllm-multinode-concurrency) VLLM_MULTINODE_CONCURRENCY="$2"; shift 2 ;;
    --vllm-multinode-concurrency-range) VLLM_MULTINODE_CONCURRENCY_RANGE="$2"; shift 2 ;;
    --vllm-multinode-num-prompts) VLLM_MULTINODE_NUM_PROMPTS="$2"; shift 2 ;;
    --vllm-multinode-ray-port) VLLM_MULTINODE_RAY_PORT="$2"; shift 2 ;;
    --vllm-multinode-image) VLLM_MULTINODE_IMAGE="$2"; shift 2 ;;
    --vllm-multinode-ray-timeout) VLLM_MULTINODE_RAY_TIMEOUT="$2"; shift 2 ;;
    --vllm-multinode-server-timeout) VLLM_MULTINODE_SERVER_TIMEOUT="$2"; shift 2 ;;
    --vllm-multinode-worker-startup-wait) VLLM_MULTINODE_WORKER_STARTUP_WAIT="$2"; shift 2 ;;

    --enable-fp4) ENABLE_FP4=1; shift ;;
    --disable-fp4) ENABLE_FP4=0; shift ;;
    --fp4-runtime) FP4_RUNTIME="$2"; shift 2 ;;
    --fp4-stack-profile) FP4_STACK_PROFILE="$2"; shift 2 ;;
    --fp4-image) FP4_IMAGE="$2"; shift 2 ;;
    --fp4-preset) FP4_PRESET="$2"; shift 2 ;;
    --fp4-warmup) FP4_WARMUP="$2"; shift 2 ;;
    --fp4-iters) FP4_ITERS="$2"; shift 2 ;;
    --fp4-smoke-m) FP4_SMOKE_M="$2"; shift 2 ;;
    --fp4-smoke-n) FP4_SMOKE_N="$2"; shift 2 ;;
    --fp4-smoke-k) FP4_SMOKE_K="$2"; shift 2 ;;
    --fp4-smoke-warmup) FP4_SMOKE_WARMUP="$2"; shift 2 ;;
    --fp4-smoke-iters) FP4_SMOKE_ITERS="$2"; shift 2 ;;
    --fp4-smoke-rounds) FP4_SMOKE_ROUNDS="$2"; shift 2 ;;
    --fp4-smoke-skew-threshold-pct) FP4_SMOKE_SKEW_THRESHOLD_PCT="$2"; shift 2 ;;

    --bootstrap-nodes) BOOTSTRAP_NODES=1; shift ;;
    --skip-bootstrap-nodes) BOOTSTRAP_NODES=0; shift ;;
    --bootstrap-install-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --bootstrap-skip-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --bootstrap-sync-code) BOOTSTRAP_SYNC_CODE=1; shift ;;
    --bootstrap-skip-sync-code) BOOTSTRAP_SYNC_CODE=0; shift ;;
    --bootstrap-install-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=1; shift ;;
    --bootstrap-skip-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=0; shift ;;
    --bootstrap-host-parity-image) BOOTSTRAP_HOST_PARITY_IMAGE="$2"; shift 2 ;;
    --bootstrap-torch-index-url) BOOTSTRAP_TORCH_INDEX_URL="$2"; shift 2 ;;
    --bootstrap-torch-version) BOOTSTRAP_TORCH_VERSION="$2"; shift 2 ;;

    --fio-test-dir) FIO_TEST_DIR="$2"; shift 2 ;;
    --fio-file-size) FIO_FILE_SIZE="$2"; shift 2 ;;
    --fio-runtime) FIO_RUNTIME="$2"; shift 2 ;;
    --fio-repeats) FIO_REPEATS="$2"; FIO_REPEATS_EXPLICIT=1; shift 2 ;;
    --fio-max-seq-bw-cv-pct) FIO_MAX_SEQ_BW_CV_PCT="$2"; shift 2 ;;
    --run-nvbandwidth) RUN_NVBANDWIDTH_MODE="on"; shift ;;
    --skip-nvbandwidth) RUN_NVBANDWIDTH_MODE="off"; shift ;;
    --nvbandwidth-runtime) NVBANDWIDTH_RUNTIME="$2"; shift 2 ;;
    --nvbandwidth-image) NVBANDWIDTH_IMAGE="$2"; shift 2 ;;
    --nvbandwidth-bin) NVBANDWIDTH_BIN="$2"; shift 2 ;;
    --nvbandwidth-quick) NVBANDWIDTH_QUICK=1; shift ;;
    --run-gpu-stream) RUN_GPU_STREAM_MODE="on"; shift ;;
    --skip-gpu-stream) RUN_GPU_STREAM_MODE="off"; shift ;;
    --gpu-stream-device) GPU_STREAM_DEVICE="$2"; shift 2 ;;
    --gpu-stream-size-mb) GPU_STREAM_SIZE_MB="$2"; shift 2 ;;
    --gpu-stream-iters) GPU_STREAM_ITERS="$2"; shift 2 ;;
    --gpu-stream-warmup) GPU_STREAM_WARMUP="$2"; shift 2 ;;
    --gpu-stream-dtype) GPU_STREAM_DTYPE="$2"; shift 2 ;;

    --health-suite) HEALTH_SUITE_MODE="$2"; shift 2 ;;
    --health-gdr) HEALTH_GDR=1; shift ;;
    --health-gdr-gpu) HEALTH_GDR_GPU="$2"; shift 2 ;;
    --health-gdr-mem-types) HEALTH_GDR_MEM_TYPES="$2"; shift 2 ;;
    --health-gdr-use-dmabuf) HEALTH_GDR_USE_DMABUF=1; shift ;;
    --enable-mamf) ENABLE_MAMF=1; shift ;;
    --mamf-mode) MAMF_MODE="$2"; shift 2 ;;
    --mamf-concurrent) MAMF_CONCURRENT=1; shift ;;
    --enable-allreduce-stability) ENABLE_ALLREDUCE_STABILITY=1; shift ;;
    --disable-allreduce-stability) ENABLE_ALLREDUCE_STABILITY=0; FORCE_DISABLE_ALLREDUCE_STABILITY=1; shift ;;
    --allreduce-payload-gib) ALLREDUCE_PAYLOAD_GIB="$2"; shift 2 ;;
    --allreduce-iters) ALLREDUCE_ITERS="$2"; shift 2 ;;
    --allreduce-warmup) ALLREDUCE_WARMUP="$2"; shift 2 ;;
    --enable-allreduce-latency-comp) ENABLE_ALLREDUCE_LATENCY_COMP=1; shift ;;
    --allreduce-latency-payload-gib) ALLREDUCE_LATENCY_PAYLOAD_GIB="$2"; shift 2 ;;
    --allreduce-latency-chunks) ALLREDUCE_LATENCY_CHUNKS="$2"; shift 2 ;;
    --allreduce-latency-iters) ALLREDUCE_LATENCY_ITERS="$2"; shift 2 ;;
    --allreduce-latency-warmup) ALLREDUCE_LATENCY_WARMUP="$2"; shift 2 ;;
    --enable-allgather-control-plane) ENABLE_ALLGATHER_CONTROL_PLANE=1; shift ;;
    --allgather-control-iters) ALLGATHER_CONTROL_ITERS="$2"; shift 2 ;;
    --allgather-control-warmup) ALLGATHER_CONTROL_WARMUP="$2"; shift 2 ;;
    --enable-nccl-alltoall) ENABLE_NCCL_ALLTOALL=1; shift ;;
    --disable-nccl-alltoall) ENABLE_NCCL_ALLTOALL=0; FORCE_DISABLE_NCCL_ALLTOALL=1; shift ;;
    --nccl-alltoall-min-bytes) NCCL_ALLTOALL_MIN_BYTES="$2"; shift 2 ;;
    --nccl-alltoall-max-bytes) NCCL_ALLTOALL_MAX_BYTES="$2"; shift 2 ;;
    --nccl-alltoall-warmup) NCCL_ALLTOALL_WARMUP="$2"; shift 2 ;;
    --nccl-alltoall-iters) NCCL_ALLTOALL_ITERS="$2"; shift 2 ;;
    --enable-nccl-algo-comparison) ENABLE_NCCL_ALGO_COMPARISON=1; shift ;;
    --disable-nccl-algo-comparison) ENABLE_NCCL_ALGO_COMPARISON=0; FORCE_DISABLE_NCCL_ALGO_COMPARISON=1; shift ;;
    --nccl-algos) NCCL_ALGOS="$2"; shift 2 ;;
    --run-fabric-eval) RUN_FABRIC_EVAL=1; shift ;;
    --skip-fabric-eval) RUN_FABRIC_EVAL=0; shift ;;
    --nmx-url) FABRIC_NMX_URL="$2"; shift 2 ;;
    --nmx-token) FABRIC_NMX_TOKEN="$2"; shift 2 ;;
    --ib-mgmt-host) FABRIC_IB_MGMT_HOST="$2"; shift 2 ;;
    --ib-mgmt-user) FABRIC_IB_MGMT_USER="$2"; shift 2 ;;
    --ib-mgmt-ssh-key) FABRIC_IB_MGMT_SSH_KEY="$2"; shift 2 ;;
    --cumulus-hosts) FABRIC_CUMULUS_HOSTS="$2"; shift 2 ;;
    --cumulus-user) FABRIC_CUMULUS_USER="$2"; shift 2 ;;
    --cumulus-ssh-key) FABRIC_CUMULUS_SSH_KEY="$2"; shift 2 ;;
    --require-management-plane) REQUIRE_MANAGEMENT_PLANE=1; shift ;;
    --modern-llm-profile) MODERN_LLM_PROFILE=1; MODERN_LLM_PROFILE_EXPLICIT=1; shift ;;
    --gpu-hourly-cost-usd) GPU_HOURLY_COST_USD="$2"; shift 2 ;;
    --coverage-baseline-run-id) COVERAGE_BASELINE_RUN_ID="$2"; shift 2 ;;
    --strict-multinode-readiness) STRICT_MULTINODE_READINESS=1; shift ;;
    --no-strict-multinode-readiness) STRICT_MULTINODE_READINESS=0; shift ;;
    --multinode-readiness-check-only) MULTINODE_READINESS_CHECK_ONLY=1; shift ;;
    --strict-canonical-completeness) STRICT_CANONICAL_COMPLETENESS=1; FORCE_DISABLE_STRICT_CANONICAL_COMPLETENESS=0; shift ;;
    --no-strict-canonical-completeness) STRICT_CANONICAL_COMPLETENESS=0; FORCE_DISABLE_STRICT_CANONICAL_COMPLETENESS=1; shift ;;
    --canonical-min-coverage-pct) CANONICAL_MIN_COVERAGE_PCT="$2"; shift 2 ;;
    --canonical-min-advanced-coverage-pct) CANONICAL_MIN_ADVANCED_COVERAGE_PCT="$2"; shift 2 ;;
    --partial-resume-max-attempts) PARTIAL_RESUME_MAX_ATTEMPTS="$2"; shift 2 ;;
    --disable-auto-partial-resume) AUTO_RESUME_PARTIAL_STEPS=0; shift ;;
    --check-ib-sharp) CHECK_IB_SHARP=1; shift ;;
    --ib-sharp-attempt-start-sharp-am) IB_SHARP_ATTEMPT_START_SHARP_AM=1; shift ;;
    --ib-sharp-am-host) IB_SHARP_AM_HOST="$2"; shift 2 ;;

    --run-c2c) RUN_C2C=1; shift ;;
    --c2c-device) C2C_DEVICE="$2"; shift 2 ;;
    --c2c-bw-sizes) C2C_BW_SIZES="$2"; shift 2 ;;
    --c2c-lat-sizes) C2C_LAT_SIZES="$2"; shift 2 ;;
    --c2c-bw-iters) C2C_BW_ITERS="$2"; shift 2 ;;
    --c2c-lat-iters) C2C_LAT_ITERS="$2"; shift 2 ;;
    --c2c-warmup) C2C_WARMUP="$2"; shift 2 ;;

    --run-numa-mem-bw) RUN_NUMA_MEM_BW=1; shift ;;
    --numa-bytes) NUMA_BYTES="$2"; shift 2 ;;
    --numa-iters) NUMA_ITERS="$2"; shift 2 ;;
    --numa-threads) NUMA_THREADS="$2"; shift 2 ;;
    --numa-warmup) NUMA_WARMUP="$2"; shift 2 ;;
    --numa-nodes) NUMA_NODES="$2"; shift 2 ;;
    --numa-cpu-node) NUMA_CPU_NODE="$2"; shift 2 ;;

    --run-train-step) RUN_TRAIN_STEP=1; RUN_TRAIN_STEP_EXPLICIT=1; shift ;;
    --train-step-single-node) TRAIN_STEP_SINGLE_NODE=1; shift ;;
    --train-step-multi-node) TRAIN_STEP_MULTI_NODE=1; shift ;;
    --train-master-port) TRAIN_MASTER_PORT="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --train-warmup-steps) TRAIN_WARMUP_STEPS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --train-seq-len) TRAIN_SEQ_LEN="$2"; shift 2 ;;
    --train-hidden) TRAIN_HIDDEN="$2"; shift 2 ;;
    --train-layers) TRAIN_LAYERS="$2"; shift 2 ;;
    --train-heads) TRAIN_HEADS="$2"; shift 2 ;;
    --train-mlp-ratio) TRAIN_MLP_RATIO="$2"; shift 2 ;;
    --train-precision) TRAIN_PRECISION="$2"; shift 2 ;;
    --train-fsdp) TRAIN_FSDP="$2"; shift 2 ;;
    --train-lr) TRAIN_LR="$2"; shift 2 ;;

    --run-checkpoint-io) RUN_CHECKPOINT_IO=1; shift ;;
    --checkpoint-test-dir) CHECKPOINT_TEST_DIR="$2"; shift 2 ;;
    --checkpoint-bytes) CHECKPOINT_BYTES="$2"; shift 2 ;;
    --checkpoint-block-size) CHECKPOINT_BLOCK_SIZE="$2"; shift 2 ;;
    --checkpoint-files) CHECKPOINT_FILES="$2"; shift 2 ;;
    --checkpoint-fsync) CHECKPOINT_FSYNC="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Defensive defaults: keep bootstrap torch args defined even if unset in caller env.
BOOTSTRAP_HOST_PARITY_IMAGE="${BOOTSTRAP_HOST_PARITY_IMAGE:-cluster_perf_orig_parity:latest}"
BOOTSTRAP_TORCH_INDEX_URL="${BOOTSTRAP_TORCH_INDEX_URL:-https://pypi.ngc.nvidia.com}"
BOOTSTRAP_TORCH_VERSION="${BOOTSTRAP_TORCH_VERSION:-2.10.0a0+a36e1d39eb.nv26.01.42222806}"

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

RUN_DIR="${ROOT_DIR}/runs/${RUN_ID}"
STRUCTURED_DIR="${RUN_DIR}/structured"
RAW_DIR="${RUN_DIR}/raw"
FIGURES_DIR="${RUN_DIR}/figures"
REPORTS_DIR="${RUN_DIR}/reports"
MANIFEST_PATH="${RUN_DIR}/manifest.json"

mkdir -p "${STRUCTURED_DIR}" "${RAW_DIR}" "${FIGURES_DIR}" "${REPORTS_DIR}"
export CLUSTER_RUN_DIR="${RUN_DIR}"
export CLUSTER_RESULTS_STRUCTURED_DIR="${STRUCTURED_DIR}"
export CLUSTER_RESULTS_RAW_DIR="${RAW_DIR}"
export CLUSTER_FIGURES_DIR="${FIGURES_DIR}"
export CLUSTER_REPORTS_DIR="${REPORTS_DIR}"

if [[ -n "$NCCL_NVLS_ENABLE" && "$NCCL_NVLS_ENABLE" != "0" && "$NCCL_NVLS_ENABLE" != "1" && "$NCCL_NVLS_ENABLE" != "2" ]]; then
  echo "ERROR: --nccl-nvls-enable must be 0, 1, or 2 (got: ${NCCL_NVLS_ENABLE})" >&2
  exit 2
fi

if [[ "$HEALTH_GDR_USE_DMABUF" -eq 1 && "$HEALTH_GDR" -ne 1 ]]; then
  echo "ERROR: --health-gdr-use-dmabuf requires --health-gdr" >&2
  exit 2
fi

if [[ -n "$HEALTH_GDR_GPU" && ! "$HEALTH_GDR_GPU" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --health-gdr-gpu must be a non-negative integer (got: ${HEALTH_GDR_GPU})" >&2
  exit 2
fi

if [[ -n "$TRAIN_PRECISION" && "$TRAIN_PRECISION" != "bf16" && "$TRAIN_PRECISION" != "fp16" ]]; then
  echo "ERROR: --train-precision must be bf16 or fp16 (got: ${TRAIN_PRECISION})" >&2
  exit 2
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" != "auto" && "$RUN_VLLM_MULTINODE_MODE" != "on" && "$RUN_VLLM_MULTINODE_MODE" != "off" ]]; then
  echo "ERROR: invalid vLLM multinode mode: ${RUN_VLLM_MULTINODE_MODE}" >&2
  exit 2
fi
if [[ "$RUN_NVBANDWIDTH_MODE" != "auto" && "$RUN_NVBANDWIDTH_MODE" != "on" && "$RUN_NVBANDWIDTH_MODE" != "off" ]]; then
  echo "ERROR: invalid nvbandwidth mode: ${RUN_NVBANDWIDTH_MODE}" >&2
  exit 2
fi
if [[ "$RUN_GPU_STREAM_MODE" != "on" && "$RUN_GPU_STREAM_MODE" != "off" ]]; then
  echo "ERROR: invalid GPU STREAM mode: ${RUN_GPU_STREAM_MODE}" >&2
  exit 2
fi
if [[ "$NVBANDWIDTH_RUNTIME" != "host" && "$NVBANDWIDTH_RUNTIME" != "container" ]]; then
  echo "ERROR: --nvbandwidth-runtime must be host or container (got: ${NVBANDWIDTH_RUNTIME})" >&2
  exit 2
fi
if [[ "$NVBANDWIDTH_RUNTIME" == "container" && -z "$NVBANDWIDTH_IMAGE" ]]; then
  echo "ERROR: --nvbandwidth-image is required when --nvbandwidth-runtime=container" >&2
  exit 2
fi
if ! [[ "$GPU_STREAM_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --gpu-stream-device must be a non-negative integer (got: ${GPU_STREAM_DEVICE})" >&2
  exit 2
fi
if ! [[ "$GPU_STREAM_SIZE_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gpu-stream-size-mb must be a positive integer (got: ${GPU_STREAM_SIZE_MB})" >&2
  exit 2
fi
if ! [[ "$GPU_STREAM_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gpu-stream-iters must be a positive integer (got: ${GPU_STREAM_ITERS})" >&2
  exit 2
fi
if ! [[ "$GPU_STREAM_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --gpu-stream-warmup must be >= 0 (got: ${GPU_STREAM_WARMUP})" >&2
  exit 2
fi
if [[ "$GPU_STREAM_DTYPE" != "fp32" && "$GPU_STREAM_DTYPE" != "fp16" && "$GPU_STREAM_DTYPE" != "bf16" ]]; then
  echo "ERROR: --gpu-stream-dtype must be fp32, fp16, or bf16 (got: ${GPU_STREAM_DTYPE})" >&2
  exit 2
fi
if ! python3 - "$VLLM_SLO_P99_TTFT_MS" "$VLLM_SLO_P99_TPOT_MS" <<'PY'
import sys

ttft = float(sys.argv[1])
tpot = float(sys.argv[2])
if ttft <= 0 or tpot <= 0:
    raise SystemExit(1)
PY
then
  echo "ERROR: --vllm-slo-p99-ttft-ms and --vllm-slo-p99-tpot-ms must be positive numbers" >&2
  exit 2
fi
if ! [[ "$VLLM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-repeats must be a positive integer (got: ${VLLM_REPEATS})" >&2
  exit 2
fi
if ! [[ "$VLLM_MAX_POINTS_PER_RUN" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --vllm-max-points-per-run must be a non-negative integer (got: ${VLLM_MAX_POINTS_PER_RUN})" >&2
  exit 2
fi
if ! [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" =~ ^[01]$ ]]; then
  echo "ERROR: invalid request-rate sweep toggle: ${RUN_VLLM_REQUEST_RATE_SWEEP}" >&2
  exit 2
fi
if ! [[ "$RUN_NCCL_ENV_SENSITIVITY" =~ ^[01]$ ]]; then
  echo "ERROR: invalid NCCL env sensitivity toggle: ${RUN_NCCL_ENV_SENSITIVITY}" >&2
  exit 2
fi
if ! [[ "$VLLM_REQUEST_RATE_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-request-rate-repeats must be a positive integer (got: ${VLLM_REQUEST_RATE_REPEATS})" >&2
  exit 2
fi
if ! [[ "$VLLM_REQUEST_RATE_MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-request-rate-max-concurrency must be a positive integer (got: ${VLLM_REQUEST_RATE_MAX_CONCURRENCY})" >&2
  exit 2
fi
if [[ -n "$VLLM_REQUEST_RATE_NUM_PROMPTS" && ! "$VLLM_REQUEST_RATE_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-request-rate-num-prompts must be a positive integer (got: ${VLLM_REQUEST_RATE_NUM_PROMPTS})" >&2
  exit 2
fi
VLLM_REQUEST_RATE_RANGE="${VLLM_REQUEST_RATE_RANGE//,/ }"
if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
  if [[ -z "${VLLM_REQUEST_RATE_RANGE// }" ]]; then
    echo "ERROR: --vllm-request-rate-range resolved to an empty list" >&2
    exit 2
  fi
  for rate in $VLLM_REQUEST_RATE_RANGE; do
    if ! python3 - "$rate" <<'PY'
import sys
val = float(sys.argv[1])
if val <= 0:
    raise SystemExit(1)
PY
    then
      echo "ERROR: --vllm-request-rate-range contains non-positive value '${rate}'" >&2
      exit 2
    fi
  done
fi

if [[ -n "$VLLM_MULTINODE_CONCURRENCY_RANGE" ]]; then
  VLLM_MULTINODE_CONCURRENCY_RANGE="${VLLM_MULTINODE_CONCURRENCY_RANGE//,/ }"
  for c in $VLLM_MULTINODE_CONCURRENCY_RANGE; do
    if ! [[ "$c" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: --vllm-multinode-concurrency-range contains non-positive integer '${c}'" >&2
      exit 2
    fi
    VLLM_MULTINODE_CONCURRENCY_VALUES+=("$c")
  done
  if [[ "${#VLLM_MULTINODE_CONCURRENCY_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: --vllm-multinode-concurrency-range resolved to an empty list" >&2
    exit 2
  fi
else
  if ! [[ "$VLLM_MULTINODE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --vllm-multinode-concurrency must be a positive integer (got: ${VLLM_MULTINODE_CONCURRENCY})" >&2
    exit 2
  fi
  VLLM_MULTINODE_CONCURRENCY_VALUES=("$VLLM_MULTINODE_CONCURRENCY")
fi
if [[ -n "$VLLM_MULTINODE_NUM_PROMPTS" && ! "$VLLM_MULTINODE_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-num-prompts must be a positive integer (got: ${VLLM_MULTINODE_NUM_PROMPTS})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_RAY_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-ray-port must be a positive integer (got: ${VLLM_MULTINODE_RAY_PORT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-ray-timeout must be a positive integer (got: ${VLLM_MULTINODE_RAY_TIMEOUT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_SERVER_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-server-timeout must be a positive integer (got: ${VLLM_MULTINODE_SERVER_TIMEOUT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_WORKER_STARTUP_WAIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-worker-startup-wait must be a positive integer (got: ${VLLM_MULTINODE_WORKER_STARTUP_WAIT})" >&2
  exit 2
fi

if ! [[ "$CONNECTIVITY_PROBE_MASTER_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-master-port must be a positive integer (got: ${CONNECTIVITY_PROBE_MASTER_PORT})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_BARRIER_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-barrier-iters must be a positive integer (got: ${CONNECTIVITY_PROBE_BARRIER_ITERS})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_PAYLOAD_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-payload-bytes must be a positive integer (got: ${CONNECTIVITY_PROBE_PAYLOAD_BYTES})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-timeout-sec must be a positive integer (got: ${CONNECTIVITY_PROBE_TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$NCCL_ENV_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --nccl-env-warmup must be >= 0 (got: ${NCCL_ENV_WARMUP})" >&2
  exit 2
fi
if ! [[ "$NCCL_ENV_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --nccl-env-iters must be a positive integer (got: ${NCCL_ENV_ITERS})" >&2
  exit 2
fi
if ! [[ "$QUICK_FRICTION_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --quick-friction-timeout-sec must be a positive integer (got: ${QUICK_FRICTION_TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$MONITORING_SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-sample-count must be a positive integer (got: ${MONITORING_SAMPLE_COUNT})" >&2
  exit 2
fi
if ! [[ "$MONITORING_DMESG_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-dmesg-lines must be a positive integer (got: ${MONITORING_DMESG_LINES})" >&2
  exit 2
fi
if ! [[ "$MONITORING_K8S_MODE" =~ ^(auto|expect|skip)$ ]]; then
  echo "ERROR: --monitoring-k8s-mode must be one of: auto, expect, skip (got: ${MONITORING_K8S_MODE})" >&2
  exit 2
fi
if ! [[ "$MONITORING_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-timeout-sec must be a positive integer (got: ${MONITORING_TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$FIO_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --fio-repeats must be a positive integer (got: ${FIO_REPEATS})" >&2
  exit 2
fi
if [[ -n "$VLLM_MAX_CONC_CV_P95_PCT" ]]; then
  if ! python3 - "$VLLM_MAX_CONC_CV_P95_PCT" <<'PY'
import sys
v = float(sys.argv[1])
if v <= 0:
    raise SystemExit(1)
PY
  then
    echo "ERROR: --vllm-max-conc-cv-p95-pct must be a positive number (got: ${VLLM_MAX_CONC_CV_P95_PCT})" >&2
    exit 2
  fi
fi
if [[ -n "$VLLM_MAX_RATE_CV_P95_PCT" ]]; then
  if ! python3 - "$VLLM_MAX_RATE_CV_P95_PCT" <<'PY'
import sys
v = float(sys.argv[1])
if v <= 0:
    raise SystemExit(1)
PY
  then
    echo "ERROR: --vllm-max-rate-cv-p95-pct must be a positive number (got: ${VLLM_MAX_RATE_CV_P95_PCT})" >&2
    exit 2
  fi
fi
if [[ -n "$FIO_MAX_SEQ_BW_CV_PCT" ]]; then
  if ! python3 - "$FIO_MAX_SEQ_BW_CV_PCT" <<'PY'
import sys
v = float(sys.argv[1])
if v <= 0:
    raise SystemExit(1)
PY
  then
    echo "ERROR: --fio-max-seq-bw-cv-pct must be a positive number (got: ${FIO_MAX_SEQ_BW_CV_PCT})" >&2
    exit 2
  fi
fi
if [[ -n "$GPU_HOURLY_COST_USD" ]]; then
  if ! python3 - "$GPU_HOURLY_COST_USD" <<'PY'
import sys
v = float(sys.argv[1])
if v <= 0:
    raise SystemExit(1)
PY
  then
    echo "ERROR: --gpu-hourly-cost-usd must be a positive number (got: ${GPU_HOURLY_COST_USD})" >&2
    exit 2
  fi
fi
if ! [[ "$CANONICAL_MIN_COVERAGE_PCT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --canonical-min-coverage-pct must be an integer in [0,100] (got: ${CANONICAL_MIN_COVERAGE_PCT})" >&2
  exit 2
fi
if ! [[ "$CANONICAL_MIN_ADVANCED_COVERAGE_PCT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --canonical-min-advanced-coverage-pct must be an integer in [0,100] (got: ${CANONICAL_MIN_ADVANCED_COVERAGE_PCT})" >&2
  exit 2
fi
if (( CANONICAL_MIN_COVERAGE_PCT < 0 || CANONICAL_MIN_COVERAGE_PCT > 100 )); then
  echo "ERROR: --canonical-min-coverage-pct must be in [0,100] (got: ${CANONICAL_MIN_COVERAGE_PCT})" >&2
  exit 2
fi
if (( CANONICAL_MIN_ADVANCED_COVERAGE_PCT < 0 || CANONICAL_MIN_ADVANCED_COVERAGE_PCT > 100 )); then
  echo "ERROR: --canonical-min-advanced-coverage-pct must be in [0,100] (got: ${CANONICAL_MIN_ADVANCED_COVERAGE_PCT})" >&2
  exit 2
fi
if ! [[ "$PARTIAL_RESUME_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --partial-resume-max-attempts must be a positive integer (got: ${PARTIAL_RESUME_MAX_ATTEMPTS})" >&2
  exit 2
fi

if [[ "$MODERN_LLM_PROFILE" -eq 1 ]]; then
  RUN_VLLM_REQUEST_RATE_SWEEP=1
  RUN_NVBANDWIDTH_MODE="on"
  ENABLE_ALLREDUCE_STABILITY=1
  ENABLE_ALLREDUCE_LATENCY_COMP=1
  ENABLE_ALLGATHER_CONTROL_PLANE=1
  ENABLE_NCCL_ALLTOALL=1
  ENABLE_NCCL_ALGO_COMPARISON=1
  if (( RUN_TRAIN_STEP_EXPLICIT == 0 )); then
    RUN_TRAIN_STEP=1
  fi
  STRICT_CANONICAL_COMPLETENESS=1
  if [[ -z "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
    export VLLM_GPU_MEMORY_UTILIZATION="0.8"
  fi
  if [[ -z "${VLLM_SERVER_READY_TIMEOUT:-}" ]]; then
    export VLLM_SERVER_READY_TIMEOUT="1800"
  fi
  if (( VLLM_REPEATS_EXPLICIT == 0 )) && (( VLLM_REPEATS < 3 )); then
    VLLM_REPEATS="3"
  fi
  if (( VLLM_REQUEST_RATE_REPEATS_EXPLICIT == 0 )) && (( VLLM_REQUEST_RATE_REPEATS < 3 )); then
    VLLM_REQUEST_RATE_REPEATS="3"
  fi
  if (( FIO_REPEATS_EXPLICIT == 0 )) && (( FIO_REPEATS < 3 )); then
    FIO_REPEATS="3"
  fi
  if [[ -z "$VLLM_MAX_CONC_CV_P95_PCT" ]]; then
    VLLM_MAX_CONC_CV_P95_PCT="15"
  fi
  if [[ -z "$VLLM_MAX_RATE_CV_P95_PCT" ]]; then
    VLLM_MAX_RATE_CV_P95_PCT="10"
  fi
  if [[ -z "$FIO_MAX_SEQ_BW_CV_PCT" ]]; then
    FIO_MAX_SEQ_BW_CV_PCT="12"
  fi
fi

if (( FORCE_DISABLE_ALLREDUCE_STABILITY == 1 )); then
  ENABLE_ALLREDUCE_STABILITY=0
fi
if (( FORCE_DISABLE_NCCL_ALLTOALL == 1 )); then
  ENABLE_NCCL_ALLTOALL=0
fi
if (( FORCE_DISABLE_NCCL_ALGO_COMPARISON == 1 )); then
  ENABLE_NCCL_ALGO_COMPARISON=0
fi
if (( FORCE_DISABLE_VLLM_REQUEST_RATE_SWEEP == 1 )); then
  RUN_VLLM_REQUEST_RATE_SWEEP=0
fi
if (( FORCE_DISABLE_STRICT_CANONICAL_COMPLETENESS == 1 )); then
  STRICT_CANONICAL_COMPLETENESS=0
fi

if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
  if [[ -z "${VLLM_REQUEST_RATE_RANGE// }" ]]; then
    echo "ERROR: --vllm-request-rate-range resolved to an empty list while request-rate sweep is enabled" >&2
    exit 2
  fi
  for rate in $VLLM_REQUEST_RATE_RANGE; do
    if ! python3 - "$rate" <<'PY'
import sys
val = float(sys.argv[1])
if val <= 0:
    raise SystemExit(1)
PY
    then
      echo "ERROR: --vllm-request-rate-range contains non-positive value '${rate}'" >&2
      exit 2
    fi
  done
fi

if [[ "$FP4_RUNTIME" != "host" && "$FP4_RUNTIME" != "container" ]]; then
  echo "ERROR: --fp4-runtime must be host or container (got: ${FP4_RUNTIME})" >&2
  exit 2
fi
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  if [[ -z "$FP4_STACK_PROFILE" ]]; then
    FP4_STACK_PROFILE="$(cluster_perf_default_profile_for_runtime "$ROOT_DIR" "$FP4_RUNTIME")"
  fi
  if ! cluster_perf_profile_exists "$ROOT_DIR" "$FP4_STACK_PROFILE"; then
    echo "ERROR: unknown --fp4-stack-profile: ${FP4_STACK_PROFILE}" >&2
    exit 2
  fi
  if ! cluster_perf_profile_runtime_allowed "$ROOT_DIR" "$FP4_STACK_PROFILE" "$FP4_RUNTIME"; then
    echo "ERROR: --fp4-stack-profile ${FP4_STACK_PROFILE} does not allow --fp4-runtime ${FP4_RUNTIME}" >&2
    exit 2
  fi
  if [[ "$FP4_RUNTIME" == "container" && -z "$FP4_IMAGE" ]]; then
    FP4_IMAGE="$(cluster_perf_profile_image_ref "$ROOT_DIR" "$FP4_STACK_PROFILE")"
  fi
  if [[ "$FP4_RUNTIME" == "container" && -z "$FP4_IMAGE" ]]; then
    echo "ERROR: no FP4 container image resolved for profile=${FP4_STACK_PROFILE}" >&2
    exit 2
  fi
fi

if [[ "$MAMF_MODE" != "quick" && "$MAMF_MODE" != "medium" && "$MAMF_MODE" != "thorough" ]]; then
  echo "ERROR: --mamf-mode must be quick, medium, or thorough (got: ${MAMF_MODE})" >&2
  exit 2
fi

if ! [[ "$ALLGATHER_CONTROL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --allgather-control-iters must be a positive integer (got: ${ALLGATHER_CONTROL_ITERS})" >&2
  exit 2
fi
if ! [[ "$ALLGATHER_CONTROL_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --allgather-control-warmup must be >= 0 (got: ${ALLGATHER_CONTROL_WARMUP})" >&2
  exit 2
fi
if ! [[ "$NCCL_ALLTOALL_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --nccl-alltoall-warmup must be >= 0 (got: ${NCCL_ALLTOALL_WARMUP})" >&2
  exit 2
fi
if ! [[ "$NCCL_ALLTOALL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --nccl-alltoall-iters must be a positive integer (got: ${NCCL_ALLTOALL_ITERS})" >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi
for i in "${!HOST_ARR[@]}"; do
  HOST_ARR[$i]="$(echo "${HOST_ARR[$i]}" | xargs)"
  if [[ -z "${HOST_ARR[$i]}" ]]; then
    echo "ERROR: --hosts contains an empty host entry" >&2
    exit 2
  fi
done
if [[ -n "$LABELS" ]]; then
  for i in "${!LABEL_ARR[@]}"; do
    LABEL_ARR[$i]="$(echo "${LABEL_ARR[$i]}" | xargs)"
    if [[ -z "${LABEL_ARR[$i]}" ]]; then
      echo "ERROR: --labels contains an empty label entry" >&2
      exit 2
    fi
    if (( i > 0 )); then
      for ((j=0; j<i; j++)); do
        if [[ "${LABEL_ARR[$j]}" == "${LABEL_ARR[$i]}" ]]; then
          echo "ERROR: --labels entries must be unique; duplicate '${LABEL_ARR[$i]}'" >&2
          exit 2
        fi
      done
    fi
  done
fi

is_local_host_name() {
  local host="$1"
  local h_full
  h_full="$(hostname -f 2>/dev/null || hostname)"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$(hostname)" || "$host" == "$(hostname -s)" || "$host" == "$h_full" ]]
}

IS_LOCALHOST_PACKAGE=0
if [[ "${#HOST_ARR[@]}" -eq 1 ]]; then
  host0="$(echo "${HOST_ARR[0]}" | xargs)"
  if is_local_host_name "$host0"; then
    IS_LOCALHOST_PACKAGE=1
  fi
fi

if [[ "$IS_LOCALHOST_PACKAGE" -eq 1 && -z "$QUICK_FRICTION_ALLOW_FAILED_CHECKS" ]]; then
  # Localhost canary profiles intentionally classify missing internet/ops tools as expected.
  QUICK_FRICTION_ALLOW_FAILED_CHECKS="uv_torch_install,ip_owner,speedtest"
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" == "on" ]]; then
  RUN_VLLM_MULTINODE=1
elif [[ "$RUN_VLLM_MULTINODE_MODE" == "off" ]]; then
  RUN_VLLM_MULTINODE=0
elif [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  RUN_VLLM_MULTINODE=1
else
  RUN_VLLM_MULTINODE=0
fi

if [[ "$RUN_NVBANDWIDTH_MODE" == "on" ]]; then
  RUN_NVBANDWIDTH=1
elif [[ "$RUN_NVBANDWIDTH_MODE" == "off" ]]; then
  RUN_NVBANDWIDTH=0
elif [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  RUN_NVBANDWIDTH=1
else
  RUN_NVBANDWIDTH=0
fi

if [[ "$RUN_GPU_STREAM_MODE" == "on" ]]; then
  RUN_GPU_STREAM=1
else
  RUN_GPU_STREAM=0
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" == "on" && "${#HOST_ARR[@]}" -lt 2 ]]; then
  echo "ERROR: --run-vllm-multinode requires at least 2 hosts" >&2
  exit 2
fi

if [[ -z "$PRIMARY_LABEL" ]]; then
  if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -gt 0 ]]; then
    PRIMARY_LABEL="$(echo "${LABEL_ARR[0]}" | xargs)"
  else
    PRIMARY_LABEL="$(hostname -s)"
  fi
fi

if [[ -z "$SOCKET_IFNAME" ]]; then
  SOCKET_IFNAME="$OOB_IF"
fi

has_explicit_vllm_profile_override() {
  if (( MODEL_EXPLICIT == 1 || TP_EXPLICIT == 1 || ISL_EXPLICIT == 1 || OSL_EXPLICIT == 1 || CONCURRENCY_RANGE_EXPLICIT == 1 )); then
    return 0
  fi
  if (( VLLM_REPEATS_EXPLICIT == 1 || RUN_VLLM_REQUEST_RATE_SWEEP_EXPLICIT == 1 || VLLM_REQUEST_RATE_RANGE_EXPLICIT == 1 )); then
    return 0
  fi
  if (( VLLM_REQUEST_RATE_REPEATS_EXPLICIT == 1 || VLLM_REQUEST_RATE_MAX_CONCURRENCY_EXPLICIT == 1 || VLLM_REQUEST_RATE_NUM_PROMPTS_EXPLICIT == 1 )); then
    return 0
  fi
  return 1
}

resolve_vllm_profile() {
  local localhost_canary_model="openai-community/gpt2"
  local localhost_canary_tp="1"
  local localhost_canary_isl="64"
  local localhost_canary_osl="32"
  local localhost_canary_concurrency="1"
  local localhost_canary_repeats="1"

  VLLM_PROFILE_CLASS="suite_default"
  if [[ "$MODERN_LLM_PROFILE" -eq 1 ]]; then
    VLLM_PROFILE_SELECTION_REASON="modern LLM profile defaults are active"
  else
    VLLM_PROFILE_SELECTION_REASON="base suite defaults are active"
  fi

  if has_explicit_vllm_profile_override; then
    VLLM_PROFILE_CLASS="explicit_override"
    VLLM_PROFILE_SELECTION_REASON="explicit vLLM model/profile overrides were supplied"
    return 0
  fi

  if [[ "$RUN_FABRIC_EVAL" -eq 1 && "$IS_LOCALHOST_PACKAGE" -eq 1 && "${#HOST_ARR[@]}" -eq 1 ]]; then
    MODEL="$localhost_canary_model"
    TP="$localhost_canary_tp"
    ISL="$localhost_canary_isl"
    OSL="$localhost_canary_osl"
    CONCURRENCY_RANGE="$localhost_canary_concurrency"
    VLLM_REPEATS="$localhost_canary_repeats"
    RUN_VLLM_REQUEST_RATE_SWEEP=0
    VLLM_PROFILE_CLASS="localhost_canary"
    VLLM_PROFILE_SELECTION_REASON="single-host localhost fabric evaluation defaults to the canary vLLM profile when no explicit vLLM overrides are present"
    return 0
  fi

  if [[ "$MODEL" == "openai/gpt-oss-20b" && "$ISL" == "512" && "$OSL" == "128" && "$CONCURRENCY_RANGE" == "8 16 32" && "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 && "$VLLM_REQUEST_RATE_RANGE" == "1 2 4" && "$VLLM_REQUEST_RATE_MAX_CONCURRENCY" == "32" && "${VLLM_REQUEST_RATE_NUM_PROMPTS:-}" == "128" ]]; then
    VLLM_PROFILE_CLASS="mid_tier_default"
    VLLM_PROFILE_SELECTION_REASON="resolved to the compact mid-tier vLLM profile"
  fi
}

resolve_vllm_profile

VLLM_CONCURRENCY_VALUES="$(echo "${CONCURRENCY_RANGE}" | xargs)"
VLLM_REQUEST_RATE_VALUES="$(echo "${VLLM_REQUEST_RATE_RANGE}" | xargs)"

emit_multinode_readiness_artifact() {
  local status="$1"
  shift || true
  local out_path="${STRUCTURED_DIR}/${RUN_ID}_multinode_readiness.json"
  mkdir -p "${STRUCTURED_DIR}"
  python3 - "$out_path" "$RUN_ID" "$HOSTS" "$LABELS" "$STRICT_MULTINODE_READINESS" "$MODERN_LLM_PROFILE" "$RUN_VLLM_REQUEST_RATE_SWEEP" "$RUN_VLLM_MULTINODE" "$RUN_TRAIN_STEP" "$TRAIN_STEP_MULTI_NODE" "$STRICT_CANONICAL_COMPLETENESS" "$FIO_FILE_SIZE" "${OOB_IF:-}" "${SOCKET_IFNAME:-}" "${NCCL_IB_HCA:-}" "$VLLM_PROFILE_CLASS" "$VLLM_PROFILE_SELECTION_REASON" "$MODEL" "${TP:-}" "$ISL" "$OSL" "$VLLM_CONCURRENCY_VALUES" "$VLLM_REPEATS" "$VLLM_REQUEST_RATE_VALUES" "$VLLM_REQUEST_RATE_REPEATS" "$VLLM_REQUEST_RATE_MAX_CONCURRENCY" "${VLLM_REQUEST_RATE_NUM_PROMPTS:-}" "$status" "$@" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
hosts_raw = sys.argv[3]
labels_raw = sys.argv[4]
strict = bool(int(sys.argv[5]))
modern = bool(int(sys.argv[6]))
run_vllm_request_rate_sweep = bool(int(sys.argv[7]))
run_vllm_multinode = bool(int(sys.argv[8]))
run_train_step = bool(int(sys.argv[9]))
train_step_multi_node = bool(int(sys.argv[10]))
strict_canonical_completeness = bool(int(sys.argv[11]))
fio_file_size = sys.argv[12]
oob_if = sys.argv[13]
socket_ifname = sys.argv[14]
nccl_ib_hca = sys.argv[15]
vllm_profile_class = sys.argv[16]
vllm_profile_reason = sys.argv[17]
vllm_model = sys.argv[18]
vllm_tp = sys.argv[19]
vllm_isl = sys.argv[20]
vllm_osl = sys.argv[21]
vllm_concurrency_range = sys.argv[22]
vllm_repeats = sys.argv[23]
vllm_request_rate_range = sys.argv[24]
vllm_request_rate_repeats = sys.argv[25]
vllm_request_rate_max_concurrency = sys.argv[26]
vllm_request_rate_num_prompts = sys.argv[27]
status = sys.argv[28]
messages = [m for m in sys.argv[29:] if m]

hosts = [h.strip() for h in hosts_raw.split(",") if h.strip()]
labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

def _split(raw: str) -> list[str]:
    return [part for part in raw.replace(",", " ").split() if part]

payload = {
    "run_id": run_id,
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "ready": status == "ok",
    "strict_multinode_readiness": strict,
    "inputs": {
        "hosts": hosts,
        "labels": labels,
        "host_count": len(hosts),
        "label_count": len(labels),
        "modern_llm_profile": modern,
        "run_vllm_request_rate_sweep": run_vllm_request_rate_sweep,
        "run_vllm_multinode": run_vllm_multinode,
        "run_train_step": run_train_step,
        "train_step_multi_node": train_step_multi_node,
        "strict_canonical_completeness": strict_canonical_completeness,
        "fio_file_size": fio_file_size,
        "oob_if": oob_if or None,
        "socket_ifname": socket_ifname or None,
        "nccl_ib_hca": nccl_ib_hca or None,
        "vllm_profile": {
            "class": vllm_profile_class,
            "selection_reason": vllm_profile_reason,
            "model": vllm_model,
            "tp": int(vllm_tp) if vllm_tp else None,
            "isl": int(vllm_isl),
            "osl": int(vllm_osl),
            "concurrency_range": _split(vllm_concurrency_range),
            "repeats": int(vllm_repeats),
            "request_rate_enabled": run_vllm_request_rate_sweep,
            "request_rate_range": _split(vllm_request_rate_range),
            "request_rate_repeats": int(vllm_request_rate_repeats),
            "request_rate_max_concurrency": int(vllm_request_rate_max_concurrency),
            "request_rate_num_prompts": int(vllm_request_rate_num_prompts) if vllm_request_rate_num_prompts else None,
        },
    },
    "messages": messages,
    "checks": {
        "labels_explicit": bool(labels_raw.strip()),
        "interface_binding_explicit": bool((socket_ifname or "").strip()),
        "modern_profile_has_multinode_vllm": (not modern) or run_vllm_multinode,
        "train_step_has_multinode_enabled": (not run_train_step) or train_step_multi_node,
        "vllm_multinode_host_shape_valid": (not run_vllm_multinode) or len(hosts) == 2,
    },
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

MULTINODE_STRICT_VIOLATIONS=()
MULTINODE_ADVISORIES=()
MULTINODE_HARD_ERRORS=()

if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ -z "${LABELS// }" ]]; then
    MULTINODE_STRICT_VIOLATIONS+=("explicit --labels is required for multi-node canonical runs")
  fi
  if [[ -z "${SOCKET_IFNAME// }" ]]; then
    MULTINODE_STRICT_VIOLATIONS+=("explicit interface binding is required; pass --oob-if or --socket-ifname")
  fi
  if [[ "$MODERN_LLM_PROFILE" -eq 1 && "$RUN_VLLM_MULTINODE" -ne 1 ]]; then
    MULTINODE_STRICT_VIOLATIONS+=("modern profile requires multinode vLLM enabled in multi-node scope")
  fi
  if [[ "$RUN_TRAIN_STEP" -eq 1 && "$TRAIN_STEP_MULTI_NODE" -ne 1 ]]; then
    MULTINODE_STRICT_VIOLATIONS+=("train-step is enabled but --train-step-multi-node is disabled")
  fi
  if [[ "$RUN_VLLM_MULTINODE" -eq 1 && "${#HOST_ARR[@]}" -ne 2 ]]; then
    MULTINODE_HARD_ERRORS+=("multinode vLLM currently supports exactly 2 hosts (leader,worker); got ${#HOST_ARR[@]}")
  fi
  if [[ -z "${SSH_KEY// }" ]]; then
    MULTINODE_ADVISORIES+=("ssh key path is unset; pass --ssh-key for deterministic remote launch identity")
  fi
fi

if [[ "${#MULTINODE_HARD_ERRORS[@]}" -gt 0 ]]; then
  emit_multinode_readiness_artifact "error" "${MULTINODE_HARD_ERRORS[@]}"
  for msg in "${MULTINODE_HARD_ERRORS[@]}"; do
    echo "ERROR: ${msg}" >&2
  done
  exit 2
fi

if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ "$STRICT_MULTINODE_READINESS" -eq 1 && "${#MULTINODE_STRICT_VIOLATIONS[@]}" -gt 0 ]]; then
    emit_multinode_readiness_artifact "error" "${MULTINODE_STRICT_VIOLATIONS[@]}"
    for msg in "${MULTINODE_STRICT_VIOLATIONS[@]}"; do
      echo "ERROR: strict multi-node readiness violation: ${msg}" >&2
    done
    exit 2
  fi

  if [[ "$STRICT_MULTINODE_READINESS" -eq 1 ]]; then
    if [[ "${#MULTINODE_ADVISORIES[@]}" -gt 0 ]]; then
      emit_multinode_readiness_artifact "ok" "${MULTINODE_ADVISORIES[@]}"
    else
      emit_multinode_readiness_artifact "ok"
    fi
  else
    # Preserve non-strict behavior, but keep the readiness contract visible in structured output.
    MULTINODE_ADVISORIES+=("${MULTINODE_STRICT_VIOLATIONS[@]}")
    if [[ "${#MULTINODE_ADVISORIES[@]}" -gt 0 ]]; then
      emit_multinode_readiness_artifact "advisory" "${MULTINODE_ADVISORIES[@]}"
    else
      emit_multinode_readiness_artifact "ok"
    fi
  fi
elif [[ "$MULTINODE_READINESS_CHECK_ONLY" -eq 1 ]]; then
  emit_multinode_readiness_artifact "not_applicable" "single-host scope: multi-node readiness checks are not applicable"
fi

if [[ "$MULTINODE_READINESS_CHECK_ONLY" -eq 1 ]]; then
  manifest_args=(--root "$ROOT_DIR" --run-id "$RUN_ID" --run-dir "$RUN_DIR" --hosts "$HOSTS" --include-figures)
  if [[ -n "$LABELS" ]]; then
    manifest_args+=(--labels "$LABELS")
  fi
  if ! python3 "${ROOT_DIR}/scripts/write_manifest.py" "${manifest_args[@]}"; then
    echo "ERROR: readiness-only run failed to write manifest: ${MANIFEST_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "ERROR: readiness-only run completed without manifest: ${MANIFEST_PATH}" >&2
    exit 2
  fi
  if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
    echo "Multi-node readiness check completed (no workloads executed)." >&2
  else
    echo "Readiness check-only requested in single-host scope; emitted not_applicable readiness artifact." >&2
  fi
  exit 0
fi

fail=0

SUITE_LOG_DIR="${RAW_DIR}/${RUN_ID}_suite"
SUITE_STEPS_JSON="${STRUCTURED_DIR}/${RUN_ID}_suite_steps.json"
acquire_queue_runner_lock
mkdir -p "$SUITE_LOG_DIR"
mkdir -p "${STRUCTURED_DIR}"
if [[ "$RESUME_RUN" -eq 1 && -f "${SUITE_STEPS_JSON}" ]]; then
  echo "Resuming suite with existing step ledger: ${SUITE_STEPS_JSON}"
else
  printf "[]\n" >"${SUITE_STEPS_JSON}"
fi

CURRENT_STEP_NAME=""
CURRENT_STEP_START=""
CURRENT_STEP_LOG_PATH=""
CURRENT_STEP_CMD=""
CURRENT_STEP_RECORDED=1
SUITE_FORCED_EXIT_CODE=""

record_suite_step() {
  local json_path="$1"
  local name="$2"
  local start="$3"
  local end="$4"
  local rc="$5"
  local log_path="$6"
  local cmd_str="$7"
  python3 - "$json_path" "$name" "$start" "$end" "$rc" "$log_path" "$cmd_str" <<'PY'
import json
import sys

path, name, start, end, rc, log_path, cmd = sys.argv[1:]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    payload = []

if not isinstance(payload, list):
    payload = []

payload.append(
    {
        "name": name,
        "start_time": start,
        "end_time": end,
        "exit_code": int(rc),
        "log_path": log_path,
        "command": cmd,
    }
)

with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

suite_step_completed_ok() {
  local json_path="$1"
  local name="$2"
  python3 - "$json_path" "$name" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
name = sys.argv[2]
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if not isinstance(payload, list):
    raise SystemExit(1)
for row in payload:
    if not isinstance(row, dict):
        continue
    if row.get("name") != name:
        continue
    try:
        rc = int(row.get("exit_code"))
    except Exception:
        continue
    if rc == 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

record_inflight_step() {
  local rc="$1"
  local end=""
  if [[ -z "$CURRENT_STEP_NAME" || "$CURRENT_STEP_RECORDED" -ne 0 ]]; then
    return 0
  fi
  end="$(date -Iseconds)"
  if ! record_suite_step "${SUITE_STEPS_JSON}" "${CURRENT_STEP_NAME}" "${CURRENT_STEP_START}" "${end}" "${rc}" "${CURRENT_STEP_LOG_PATH}" "${CURRENT_STEP_CMD}"; then
    echo "WARNING: failed to append interrupted step metadata for ${CURRENT_STEP_NAME}" >&2
  fi
  CURRENT_STEP_RECORDED=1
  echo "<== [${end}] END ${CURRENT_STEP_NAME} rc=${rc} (interrupted)"
}

on_suite_signal() {
  local signame="$1"
  local rc="$2"
  SUITE_FORCED_EXIT_CODE="$rc"
  echo "ERROR: received signal ${signame}; exiting suite with rc=${rc}" >&2
  exit "$rc"
}

on_suite_exit() {
  local rc=$?
  if [[ -n "$SUITE_FORCED_EXIT_CODE" ]]; then
    rc="$SUITE_FORCED_EXIT_CODE"
  fi
  if [[ -n "$CURRENT_STEP_NAME" && "$CURRENT_STEP_RECORDED" -eq 0 ]]; then
    echo "WARNING: suite exiting with in-flight step '${CURRENT_STEP_NAME}'; recording interrupted status rc=${rc}" >&2
    record_inflight_step "$rc"
  fi
  release_queue_runner_lock
}

trap on_suite_exit EXIT
trap 'on_suite_signal INT 130' INT
trap 'on_suite_signal TERM 143' TERM
trap 'on_suite_signal HUP 129' HUP

run_step() {
  local name="$1"
  shift
  local start end rc log_path cmd_str
  local resume_attempt=0
  if [[ "$RESUME_RUN" -eq 1 ]] && suite_step_completed_ok "${SUITE_STEPS_JSON}" "${name}"; then
    echo "==> [$(date -Iseconds)] SKIP ${name} (already completed)"
    return 0
  fi
  log_path="${SUITE_LOG_DIR}/${name}.log"
  cmd_str="$(printf '%q ' "$@")"
  cmd_str="${cmd_str% }"
  while true; do
    start="$(date -Iseconds)"
    CURRENT_STEP_NAME="$name"
    CURRENT_STEP_START="$start"
    CURRENT_STEP_LOG_PATH="$log_path"
    CURRENT_STEP_CMD="$cmd_str"
    CURRENT_STEP_RECORDED=0
    echo "==> [${start}] START ${name} (attempt=$((resume_attempt + 1)))"
    echo "    cmd: ${cmd_str}"
    echo "    log: ${log_path}"

    set +e
    if (( resume_attempt == 0 )); then
      "$@" >"$log_path" 2>&1
    else
      {
        echo ""
        echo "=== AUTO_RESUME_ATTEMPT $((resume_attempt + 1)) @ $(date -Iseconds) ==="
        "$@"
      } >>"$log_path" 2>&1
    fi
    rc=$?
    set -e

    end="$(date -Iseconds)"
    if ! record_suite_step "${SUITE_STEPS_JSON}" "${name}" "${start}" "${end}" "${rc}" "${log_path}" "${cmd_str}"; then
      echo "WARNING: failed to append step metadata for ${name}" >&2
    fi
    CURRENT_STEP_RECORDED=1
    CURRENT_STEP_NAME=""
    CURRENT_STEP_START=""
    CURRENT_STEP_LOG_PATH=""
    CURRENT_STEP_CMD=""
    echo "<== [${end}] END ${name} rc=${rc}"

    if [[ "$rc" -eq 75 ]]; then
      if [[ "$AUTO_RESUME_PARTIAL_STEPS" -eq 1 ]]; then
        resume_attempt=$((resume_attempt + 1))
        if (( resume_attempt >= PARTIAL_RESUME_MAX_ATTEMPTS )); then
          fail=1
          echo "ERROR: ${name} hit partial-resume limit (${PARTIAL_RESUME_MAX_ATTEMPTS}); stopping retries." >&2
          echo "---- ${name} (last 60 lines) ----" >&2
          tail -n 60 "$log_path" >&2 || true
          return 1
        fi
        echo "INFO: ${name} reported resumable partial progress (rc=75); auto-resuming (attempt ${resume_attempt}/${PARTIAL_RESUME_MAX_ATTEMPTS})." >&2
        echo "---- ${name} (last 60 lines) ----" >&2
        tail -n 60 "$log_path" >&2 || true
        sleep 2
        continue
      fi
      fail=1
      echo "INFO: ${name} reported resumable partial progress (rc=75); exit and resume to continue this step." >&2
      echo "---- ${name} (last 60 lines) ----" >&2
      tail -n 60 "$log_path" >&2 || true
      exit 75
    fi

    if [[ "$rc" -ne 0 ]]; then
      fail=1
      echo "---- ${name} (last 60 lines) ----" >&2
      tail -n 60 "$log_path" >&2 || true
    fi
    return 0
  done
}

run_required_step() {
  local name="$1"
  shift
  run_step "$name" "$@"
  if ! suite_step_completed_ok "${SUITE_STEPS_JSON}" "${name}"; then
    echo "ERROR: required step ${name} failed; aborting suite." >&2
    exit 1
  fi
}

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

label_for_index() {
  local idx="$1"
  local host="${HOST_ARR[$idx]}"
  if [[ -n "$LABELS" ]]; then
    local lbl
    lbl="$(echo "${LABEL_ARR[$idx]}" | xargs)"
    if [[ -n "$lbl" ]]; then
      echo "$lbl"
      return 0
    fi
  fi
  echo "$(sanitize_label "$host")"
}

vllm_step_failure_message() {
  local step_name="$1"
  local startup_path="$2"
  python3 - "$SUITE_STEPS_JSON" "$step_name" "$startup_path" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import summarize_upstream_failure

suite_steps_path = Path(sys.argv[1])
step_name = sys.argv[2]
startup_raw = sys.argv[3]
startup_path = Path(startup_raw) if startup_raw else None
summary = summarize_upstream_failure(suite_steps_path, step_name, startup_path)
if summary.get("has_failure"):
    print(summary.get("message") or step_name)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

validate_required_artifacts() {
  local missing=0
  local path=""
  local label=""
  local serve_startup_path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep_startup.json"
  local request_rate_startup_path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep_startup.json"

  for idx in "${!HOST_ARR[@]}"; do
    label="$(label_for_index "$idx")"
    for suffix in "_fio.json" "_fio_stability.json"; do
      path="${STRUCTURED_DIR}/${RUN_ID}_${label}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required fio artifact: ${path}" >&2
        missing=1
      fi
    done
    path="${STRUCTURED_DIR}/${RUN_ID}_${label}_fio_stability.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" "$FIO_MAX_SEQ_BW_CV_PCT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = payload.get("summary") or {}
repeats = int(payload.get("repeats") or 0)
if repeats <= 0:
    raise SystemExit("fio stability repeats is not positive")
for key in ("seq_read_bw_cv_pct", "seq_write_bw_cv_pct"):
    value = summary.get(key)
    if value is None:
        raise SystemExit(f"fio stability missing {key}")
    if float(value) < 0:
        raise SystemExit(f"fio stability {key} is negative")
max_cv_arg = (sys.argv[2] or "").strip()
if max_cv_arg:
    max_cv = float(max_cv_arg)
    for key in ("seq_read_bw_cv_pct", "seq_write_bw_cv_pct"):
        if float(summary.get(key)) > max_cv:
            raise SystemExit(f"fio stability {key}={summary.get(key)} exceeds threshold {max_cv}")
PY
      then
        echo "ERROR: invalid fio stability artifact: ${path}" >&2
        missing=1
      fi
    fi
  done

  if [[ "$RUN_QUICK_FRICTION" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      path="${STRUCTURED_DIR}/${RUN_ID}_${label}_quick_friction.json"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required quick friction artifact: ${path}" >&2
        missing=1
        continue
      fi
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("test") != "quick_friction":
    raise SystemExit("unexpected test type")
status = payload.get("status")
if status not in {"ok", "degraded", "error"}:
    raise SystemExit(f"invalid quick friction status: {status}")
checks = payload.get("checks") or []
if not checks:
    raise SystemExit("quick friction checks are empty")
PY
      then
        echo "ERROR: invalid quick friction artifact: ${path}" >&2
        missing=1
      fi
    done
  fi

  if [[ "$RUN_MONITORING_EXPECTATIONS" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      path="${STRUCTURED_DIR}/${RUN_ID}_${label}_monitoring_expectations.json"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required monitoring expectations artifact: ${path}" >&2
        missing=1
        continue
      fi
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("test") != "monitoring_expectations":
    raise SystemExit("unexpected test type")
status = payload.get("status")
if status not in {"ok", "degraded", "error"}:
    raise SystemExit(f"invalid monitoring status: {status}")
checks = payload.get("checks") or []
if not checks:
    raise SystemExit("monitoring checks are empty")
categories = payload.get("categories") or {}
if not categories:
    raise SystemExit("monitoring categories are empty")
PY
      then
        echo "ERROR: invalid monitoring expectations artifact: ${path}" >&2
        missing=1
      fi
    done
  fi

  for idx in "${!HOST_ARR[@]}"; do
    label="$(label_for_index "$idx")"
    path="${STRUCTURED_DIR}/${RUN_ID}_${label}_hang_triage_readiness.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required hang triage artifact: ${path}" >&2
      missing=1
      continue
    fi
    if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"hang triage readiness status is not ok: {status}")
tools = payload.get("tools") or {}
if not bool((tools.get("py_spy") or {}).get("available")):
    raise SystemExit("py-spy is not available")
if not bool((tools.get("strace") or {}).get("available")):
    raise SystemExit("strace is not available")
PY
    then
      echo "ERROR: invalid hang triage artifact: ${path}" >&2
      missing=1
    fi
  done

  path="${STRUCTURED_DIR}/${RUN_ID}_torchrun_connectivity_probe.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required connectivity probe artifact: ${path}" >&2
    missing=1
  else
    if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"connectivity probe status is not ok: {status}")
ranks = payload.get("ranks") or []
if not ranks:
    raise SystemExit("connectivity probe ranks are empty")
bad = [r for r in ranks if (r or {}).get("status") != "ok"]
if bad:
    raise SystemExit(f"connectivity probe has failing ranks: {len(bad)}")
PY
    then
      echo "ERROR: invalid connectivity probe artifact: ${path}" >&2
      missing=1
    fi
  fi

  if [[ "$RUN_NCCL_ENV_SENSITIVITY" -eq 1 ]]; then
    path="${STRUCTURED_DIR}/${RUN_ID}_nccl_env_sensitivity.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required NCCL env sensitivity artifact: ${path}" >&2
      missing=1
    else
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"nccl env sensitivity status is not ok: {status}")
if int(payload.get("failure_count", 0)) != 0:
    raise SystemExit(f"nccl env sensitivity failure_count is non-zero: {payload.get('failure_count')}")
profiles = payload.get("profiles") or []
if not profiles:
    raise SystemExit("nccl env sensitivity profiles are empty")
if not any((p or {}).get("profile") == "baseline_auto" and (p or {}).get("status") == "ok" for p in profiles):
    raise SystemExit("nccl env sensitivity missing successful baseline_auto profile")
PY
      then
        echo "ERROR: invalid NCCL env sensitivity artifact: ${path}" >&2
        missing=1
      fi
    fi
  fi

  if [[ "$ENABLE_NCCL_ALLTOALL" -eq 1 ]]; then
    path="${STRUCTURED_DIR}/${RUN_ID}_node1_alltoall_nccl_alltoall.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required NCCL all-to-all artifact: ${path}" >&2
      missing=1
    else
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("results") or []
if not rows:
    raise SystemExit("NCCL all-to-all results are empty")
total_ranks = int(payload.get("total_ranks") or 0)
max_busbw = max(float((r or {}).get("busbw_gbps") or 0.0) for r in rows)
max_algbw = max(float((r or {}).get("algbw_gbps") or 0.0) for r in rows)
if total_ranks >= 2:
    if max_busbw <= 0:
        raise SystemExit("NCCL all-to-all busbw is non-positive for multi-rank run")
else:
    if max_algbw <= 0:
        raise SystemExit("NCCL all-to-all algbw is non-positive for single-rank run")
PY
      then
        echo "ERROR: invalid NCCL all-to-all artifact: ${path}" >&2
        missing=1
      fi
    fi

    if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
      path="${STRUCTURED_DIR}/${RUN_ID}_2nodes_alltoall_nccl_alltoall.json"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multi-node NCCL all-to-all artifact: ${path}" >&2
        missing=1
      else
        if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("results") or []
if not rows:
    raise SystemExit("multi-node NCCL all-to-all results are empty")
if max(float((r or {}).get("busbw_gbps") or 0.0) for r in rows) <= 0:
    raise SystemExit("multi-node NCCL all-to-all busbw is non-positive")
PY
        then
          echo "ERROR: invalid multi-node NCCL all-to-all artifact: ${path}" >&2
          missing=1
        fi
      fi
    fi
  fi

  if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      for suffix in "_nvbandwidth.json" "_nvbandwidth_sums.csv" "_nvbandwidth_clock_lock.json"; do
        path="${STRUCTURED_DIR}/${RUN_ID}_${label}${suffix}"
        if [[ ! -f "$path" ]]; then
          echo "ERROR: missing required nvbandwidth artifact: ${path}" >&2
          missing=1
        fi
      done
      path="${STRUCTURED_DIR}/${RUN_ID}_${label}_nvbandwidth.json"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"nvbandwidth status is not ok: {status}")
clock = payload.get("clock_lock") or {}
if not bool(clock.get("all_devices_locked")):
    raise SystemExit("nvbandwidth clock_lock.all_devices_locked is false")
PY
        then
          echo "ERROR: invalid nvbandwidth summary: ${path}" >&2
          missing=1
        fi
      fi
    done
  fi

  if [[ "$RUN_GPU_STREAM" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      for suffix in "_gpu_stream.json" "_gpu_stream.csv" "_gpu_stream_clock_lock.json"; do
        path="${STRUCTURED_DIR}/${RUN_ID}_${label}${suffix}"
        if [[ ! -f "$path" ]]; then
          echo "ERROR: missing required gpu_stream artifact: ${path}" >&2
          missing=1
        fi
      done
      path="${STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream.json"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"gpu_stream status is not ok: {payload.get('status')}")
ops = payload.get("operations") or []
if not ops:
    raise SystemExit("gpu_stream operations are empty")
if max(float(op.get("bandwidth_gbps", 0.0)) for op in ops) <= 0:
    raise SystemExit("gpu_stream bandwidth values are non-positive")
PY
        then
          echo "ERROR: invalid gpu_stream summary: ${path}" >&2
          missing=1
        fi
      fi
    done
  fi

  if message="$(vllm_step_failure_message "vllm_serve_sweep" "$serve_startup_path")"; then
    echo "ERROR: ${message}" >&2
    missing=1
  else
    for suffix in "_vllm_serve_sweep.csv" "_vllm_serve_sweep.jsonl" "_vllm_serve_sweep_clock_lock.json" "_vllm_serve_sweep_stability.json" "_vllm_serve_sweep_startup.json"; do
      path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required vLLM serve sweep artifact: ${path}" >&2
        missing=1
      fi
    done
    path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_serve_csv

validate_vllm_serve_csv(Path(sys.argv[1]))
PY
      then
        echo "ERROR: invalid vLLM serve sweep csv: ${path}" >&2
        missing=1
      fi
    fi
    path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep_stability.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" "$VLLM_MAX_CONC_CV_P95_PCT" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_stability_summary

threshold_raw = (sys.argv[2] or "").strip()
threshold = float(threshold_raw) if threshold_raw else None
validate_vllm_stability_summary(Path(sys.argv[1]), threshold=threshold, label="vLLM sweep stability")
PY
      then
        echo "ERROR: invalid vLLM serve sweep stability: ${path}" >&2
        missing=1
      fi
    fi
    for suffix in "_vllm_serve_slo_goodput.json" "_vllm_serve_slo_goodput.csv"; do
      path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required vLLM SLO goodput artifact: ${path}" >&2
        missing=1
      fi
    done
    path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_slo_goodput.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_slo_goodput_summary

validate_vllm_slo_goodput_summary(Path(sys.argv[1]))
PY
      then
        echo "ERROR: invalid vLLM SLO goodput summary: ${path}" >&2
        missing=1
      fi
    fi
  fi
  if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
    if message="$(vllm_step_failure_message "vllm_request_rate_sweep" "$request_rate_startup_path")"; then
      echo "ERROR: ${message}" >&2
      missing=1
    else
      for suffix in "_vllm_serve_request_rate_sweep.csv" "_vllm_serve_request_rate_sweep.jsonl" "_vllm_serve_request_rate_sweep_clock_lock.json" "_vllm_serve_request_rate_sweep_stability.json" "_vllm_serve_request_rate_sweep_startup.json" "_vllm_request_rate_slo_goodput.json" "_vllm_request_rate_slo_goodput.csv"; do
        path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}${suffix}"
        if [[ ! -f "$path" ]]; then
          echo "ERROR: missing required vLLM request-rate artifact: ${path}" >&2
          missing=1
        fi
      done
      path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep.csv"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_request_rate_csv

validate_vllm_request_rate_csv(Path(sys.argv[1]))
PY
        then
          echo "ERROR: invalid vLLM request-rate sweep csv: ${path}" >&2
          missing=1
        fi
      fi
      path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep_stability.json"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" "$VLLM_MAX_RATE_CV_P95_PCT" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_stability_summary

threshold_raw = (sys.argv[2] or "").strip()
threshold = float(threshold_raw) if threshold_raw else None
validate_vllm_stability_summary(Path(sys.argv[1]), threshold=threshold, label="vLLM request-rate stability")
PY
        then
          echo "ERROR: invalid vLLM request-rate sweep stability: ${path}" >&2
          missing=1
        fi
      fi
      path="${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_request_rate_slo_goodput.json"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" <<'PY'
import sys
from pathlib import Path

from analysis.vllm_step_contract import validate_vllm_slo_goodput_summary

validate_vllm_slo_goodput_summary(Path(sys.argv[1]), request_rate=True)
PY
        then
          echo "ERROR: invalid vLLM request-rate SLO goodput summary: ${path}" >&2
          missing=1
        fi
      fi
    fi
  fi

  if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
    local leader_label worker_label
    leader_label="$(label_for_index 0)"
    worker_label="$(label_for_index 1)"

    for suffix in "_vllm_multinode_serve.json" "_vllm_multinode_serve.csv" "_vllm_multinode_serve.jsonl"; do
      path="${STRUCTURED_DIR}/${RUN_ID}_${leader_label}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multinode vLLM artifact: ${path}" >&2
        missing=1
      fi
    done
    for suffix in "_vllm_multinode_slo_goodput.json" "_vllm_multinode_slo_goodput.csv"; do
      path="${STRUCTURED_DIR}/${RUN_ID}_${leader_label}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multinode vLLM SLO goodput artifact: ${path}" >&2
        missing=1
      fi
    done
    for suffix in "_vllm_multinode_leader_clock_lock.json" "_vllm_multinode_worker_clock_lock.json"; do
      if [[ "$suffix" == *_leader_clock_lock.json ]]; then
        path="${STRUCTURED_DIR}/${RUN_ID}_${leader_label}${suffix}"
      else
        path="${STRUCTURED_DIR}/${RUN_ID}_${worker_label}${suffix}"
      fi
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multinode vLLM lock artifact: ${path}" >&2
        missing=1
      fi
    done

    path="${STRUCTURED_DIR}/${RUN_ID}_${leader_label}_vllm_multinode_serve.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"multinode vLLM status is not ok: {status}")
rcs = payload.get("return_codes") or {}
if int(rcs.get("leader", 1)) != 0 or int(rcs.get("worker", 1)) != 0:
    raise SystemExit(f"multinode vLLM return codes not clean: {rcs}")
clock = payload.get("clock_lock") or {}
leader = clock.get("leader") or {}
worker = clock.get("worker") or {}
if not bool(leader.get("all_devices_locked")):
    raise SystemExit("multinode vLLM leader clocks are not locked")
if not bool(worker.get("all_devices_locked")):
    raise SystemExit("multinode vLLM worker clocks are not locked")
metrics = payload.get("metrics") or {}
if (metrics.get("total_token_throughput") or 0) <= 0:
    raise SystemExit("multinode vLLM total_token_throughput is not positive")
PY
      then
        echo "ERROR: invalid multinode vLLM summary: ${path}" >&2
        missing=1
      fi
    fi

    path="${STRUCTURED_DIR}/${RUN_ID}_${leader_label}_vllm_multinode_slo_goodput.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"multinode vLLM SLO goodput status is not ok: {payload.get('status')}")
summary = payload.get("summary") or {}
if int(summary.get("concurrency_points", 0)) <= 0:
    raise SystemExit("multinode vLLM SLO goodput concurrency_points is not positive")
PY
      then
        echo "ERROR: invalid multinode vLLM SLO goodput summary: ${path}" >&2
        missing=1
      fi
    fi
  fi

  if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" && "$HEALTH_GDR" -eq 1 ]]; then
    local -a hs_summaries=()
    shopt -s nullglob
    hs_summaries=( "${STRUCTURED_DIR}/${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}_"*"_cluster_health_suite_summary.json" )
    shopt -u nullglob
    if [[ "${#hs_summaries[@]}" -eq 0 ]]; then
      echo "ERROR: health suite summary missing; cannot verify effective GDR." >&2
      missing=1
    else
      if ! python3 - "${hs_summaries[0]}" "${HEALTH_GDR_MEM_TYPES}" "${HEALTH_GDR_USE_DMABUF}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_mem_types = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
expect_dmabuf = bool(int(sys.argv[3]))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
gdr = summary.get("gdr") or {}
if not bool(gdr.get("requested")):
    raise SystemExit(f"GDR requested flag is false in {summary_path}")
if not bool(gdr.get("effective_enabled")):
    reason = gdr.get("disabled_reason")
    raise SystemExit(f"GDR effective_enabled is false in {summary_path}; reason={reason}")
actual_mem_types = [str(x).strip() for x in (gdr.get("mem_types") or []) if str(x).strip()]
if sorted(actual_mem_types) != sorted(expected_mem_types):
    raise SystemExit(
        f"GDR mem_types mismatch in {summary_path}: expected={expected_mem_types} actual={actual_mem_types}"
    )
if expect_dmabuf and not bool(gdr.get("use_dmabuf")):
    raise SystemExit(f"GDR use_dmabuf expected true but false in {summary_path}")
PY
      then
        echo "ERROR: health suite GDR verification failed for ${hs_summaries[0]}" >&2
        missing=1
      fi
    fi
  fi

  if [[ "$RUN_FABRIC_EVAL" -eq 1 ]]; then
    for suffix in "_fabric_command_catalog.json" "_fabric_capability_matrix.json" "_fabric_verification.json" "_fabric_ai_correlation.json" "_fabric_scorecard.json" "_fabric_scorecard.md"; do
      path="${STRUCTURED_DIR}/${RUN_ID}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required fabric artifact: ${path}" >&2
        missing=1
      fi
    done
  fi

  path="${STRUCTURED_DIR}/${RUN_ID}_cluster_scorecard.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required scorecard artifact: ${path}" >&2
    missing=1
  fi
  path="${STRUCTURED_DIR}/${RUN_ID}_cluster_scorecard.md"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required scorecard artifact: ${path}" >&2
    missing=1
  fi
  path="${STRUCTURED_DIR}/${RUN_ID}_benchmark_coverage_analysis.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required benchmark coverage artifact: ${path}" >&2
    missing=1
  fi
  path="${STRUCTURED_DIR}/${RUN_ID}_benchmark_coverage_analysis.md"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required benchmark coverage artifact: ${path}" >&2
    missing=1
  fi
  path="${STRUCTURED_DIR}/${RUN_ID}_mlperf_alignment.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required MLPerf alignment artifact: ${path}" >&2
    missing=1
  fi
  path="${STRUCTURED_DIR}/${RUN_ID}_mlperf_alignment.md"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required MLPerf alignment artifact: ${path}" >&2
    missing=1
  fi

  if [[ "$STRICT_CANONICAL_COMPLETENESS" -eq 1 ]]; then
    path="${STRUCTURED_DIR}/${RUN_ID}_benchmark_coverage_analysis.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" "$CANONICAL_MIN_COVERAGE_PCT" "$CANONICAL_MIN_ADVANCED_COVERAGE_PCT" <<'PY'
import json
import sys
from pathlib import Path

coverage_path = Path(sys.argv[1])
min_cov = int(sys.argv[2])
min_adv = int(sys.argv[3])
payload = json.loads(coverage_path.read_text(encoding="utf-8"))
cov = int(payload.get("coverage_score_pct") or 0)
adv = int(payload.get("advanced_coverage_score_pct") or 0)
missing = payload.get("missing_subsystems") or []
if cov < min_cov:
    raise SystemExit(f"coverage_score_pct={cov} below required {min_cov}")
if adv < min_adv:
    raise SystemExit(f"advanced_coverage_score_pct={adv} below required {min_adv}")
if missing:
    raise SystemExit(f"missing_subsystems is not empty: {missing}")
advanced = payload.get("advanced_coverage") or {}
required = (
    "vllm_request_rate_sweep",
    "vllm_concurrency_repeat_stability",
    "vllm_request_rate_repeat_stability",
    "fio_repeat_stability",
    "allreduce_stability",
    "allreduce_latency_comp",
    "allgather_control_plane",
    "nccl_alltoall",
    "nccl_algo_comparison",
    "train_step_workload",
    "mlperf_alignment",
)
for key in required:
    if not bool(advanced.get(key)):
        raise SystemExit(f"advanced_coverage[{key}] is false")
PY
      then
        echo "ERROR: canonical completeness coverage check failed (${path})" >&2
        missing=1
      fi
    fi

    path="${STRUCTURED_DIR}/${RUN_ID}_mlperf_alignment.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if str(payload.get("overall_status") or "").lower() != "aligned":
    raise SystemExit(f"mlperf alignment overall_status is not aligned: {payload.get('overall_status')}")
if not bool(payload.get("inference_track_ready")):
    raise SystemExit("mlperf alignment inference_track_ready is false")
if not bool(payload.get("training_track_ready")):
    raise SystemExit("mlperf alignment training_track_ready is false")
PY
      then
        echo "ERROR: canonical completeness MLPerf alignment check failed (${path})" >&2
        missing=1
      fi
    fi
  fi

  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
  return 0
}

echo "========================================"
echo "Cluster Eval Suite"
echo "========================================"
echo "Date: $(date -Iseconds)"
echo "RUN_ID(base): ${RUN_ID}"
echo "HOSTS: ${HOSTS}"
echo "SUITE_STEPS_JSON: ${SUITE_STEPS_JSON}"
echo "QUEUE_RUNNER_LOCK: ${QUEUE_RUNNER_LOCK_PATH} (timeout_s=${QUEUE_RUNNER_LOCK_TIMEOUT_SEC})"
if [[ -n "$LABELS" ]]; then
  echo "LABELS: ${LABELS}"
fi
echo "SSH_USER: ${SSH_USER}"
echo "PRIMARY_LABEL: ${PRIMARY_LABEL}"
echo "connectivity_probe: master_port=${CONNECTIVITY_PROBE_MASTER_PORT} barrier_iters=${CONNECTIVITY_PROBE_BARRIER_ITERS} payload_bytes=${CONNECTIVITY_PROBE_PAYLOAD_BYTES} timeout_sec=${CONNECTIVITY_PROBE_TIMEOUT_SEC}"
echo "nccl_env_sensitivity: enabled=${RUN_NCCL_ENV_SENSITIVITY} min=${NCCL_ENV_MIN_BYTES} max=${NCCL_ENV_MAX_BYTES} warmup=${NCCL_ENV_WARMUP} iters=${NCCL_ENV_ITERS}"
echo "quick_friction: enabled=${RUN_QUICK_FRICTION} strict=${QUICK_FRICTION_STRICT} checks='${QUICK_FRICTION_CHECKS}' timeout_sec=${QUICK_FRICTION_TIMEOUT_SEC} torch=${QUICK_FRICTION_TORCH_VERSION} hf_model=${QUICK_FRICTION_HF_MODEL} allow_failed='${QUICK_FRICTION_ALLOW_FAILED_CHECKS:-<none>}'"
echo "monitoring_expectations: enabled=${RUN_MONITORING_EXPECTATIONS} strict=${MONITORING_EXPECTATIONS_STRICT} k8s_mode=${MONITORING_K8S_MODE} checks='${MONITORING_CHECKS}' sample_count=${MONITORING_SAMPLE_COUNT} dmesg_lines=${MONITORING_DMESG_LINES} timeout_sec=${MONITORING_TIMEOUT_SEC}"
echo "fabric_eval: enabled=${RUN_FABRIC_EVAL} require_management_plane=${REQUIRE_MANAGEMENT_PLANE}"
echo "render_localhost_report: mode=${RENDER_LOCALHOST_REPORT_MODE} detected_localhost=${IS_LOCALHOST_PACKAGE}"
if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  echo "OOB_IF: ${OOB_IF:-<unset>}"
  echo "NCCL_SOCKET_IFNAME: ${SOCKET_IFNAME:-<unset>}"
  echo "NCCL_IB_HCA: ${NCCL_IB_HCA:-<auto>}"
  echo "NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE:-<unset>}"
fi
echo "vLLM(profile): class=${VLLM_PROFILE_CLASS} reason=${VLLM_PROFILE_SELECTION_REASON}"
echo "vLLM: model=${MODEL} tp=${TP:-<auto>} isl=${ISL} osl=${OSL} conc='${CONCURRENCY_RANGE}' repeats=${VLLM_REPEATS} port=${PORT}"
echo "vLLM(env): gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION:-<unset>} server_ready_timeout_s=${VLLM_SERVER_READY_TIMEOUT:-<unset>}"
echo "vLLM(resume-segmentation): max_points_per_run=${VLLM_MAX_POINTS_PER_RUN}"
echo "vLLM(SLO): p99_ttft_ms<=${VLLM_SLO_P99_TTFT_MS} p99_tpot_ms<=${VLLM_SLO_P99_TPOT_MS}"
echo "vLLM(stability thresholds): conc_tok_cv_p95<=${VLLM_MAX_CONC_CV_P95_PCT:-<disabled>} rate_tok_cv_p95<=${VLLM_MAX_RATE_CV_P95_PCT:-<disabled>}"
if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
  echo "vLLM(request-rate): enabled rates='${VLLM_REQUEST_RATE_RANGE}' repeats=${VLLM_REQUEST_RATE_REPEATS} max_concurrency=${VLLM_REQUEST_RATE_MAX_CONCURRENCY} num_prompts=${VLLM_REQUEST_RATE_NUM_PROMPTS:-<auto>}"
else
  echo "vLLM(request-rate): disabled"
fi
if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  echo "vLLM(multinode): enabled mode=${RUN_VLLM_MULTINODE_MODE} conc='${VLLM_MULTINODE_CONCURRENCY_VALUES[*]}' prompts=${VLLM_MULTINODE_NUM_PROMPTS:-<auto>} ray_port=${VLLM_MULTINODE_RAY_PORT} ray_timeout_s=${VLLM_MULTINODE_RAY_TIMEOUT} server_timeout_s=${VLLM_MULTINODE_SERVER_TIMEOUT} worker_startup_wait_s=${VLLM_MULTINODE_WORKER_STARTUP_WAIT} image=${VLLM_MULTINODE_IMAGE:-<auto>}"
else
  echo "vLLM(multinode): disabled mode=${RUN_VLLM_MULTINODE_MODE}"
fi
echo "fio: test_dir=${FIO_TEST_DIR} file_size=${FIO_FILE_SIZE} runtime_s=${FIO_RUNTIME} repeats=${FIO_REPEATS}"
echo "fio(stability threshold): seq_bw_cv<=${FIO_MAX_SEQ_BW_CV_PCT:-<disabled>}"
if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  echo "nvbandwidth: enabled mode=${RUN_NVBANDWIDTH_MODE} runtime=${NVBANDWIDTH_RUNTIME} quick=${NVBANDWIDTH_QUICK}"
else
  echo "nvbandwidth: disabled mode=${RUN_NVBANDWIDTH_MODE}"
fi
if [[ "$RUN_GPU_STREAM" -eq 1 ]]; then
  echo "gpu_stream: enabled mode=${RUN_GPU_STREAM_MODE} device=${GPU_STREAM_DEVICE} size_mb=${GPU_STREAM_SIZE_MB} iters=${GPU_STREAM_ITERS} warmup=${GPU_STREAM_WARMUP} dtype=${GPU_STREAM_DTYPE}"
else
  echo "gpu_stream: disabled mode=${RUN_GPU_STREAM_MODE}"
fi
echo "health_suite: ${HEALTH_SUITE_MODE}"
if [[ "$HEALTH_GDR" -eq 1 ]]; then
  echo "health_suite_gdr: enabled gpu=${HEALTH_GDR_GPU} mem_types=${HEALTH_GDR_MEM_TYPES} dmabuf=${HEALTH_GDR_USE_DMABUF}"
fi
if [[ "$CHECK_IB_SHARP" -eq 1 ]]; then
  echo "ib_sharp_check: enabled (attempt_start_sharp_am=${IB_SHARP_ATTEMPT_START_SHARP_AM}, sharp_am_host=${IB_SHARP_AM_HOST:-<auto>})"
fi
echo "c2c_memcpy: ${RUN_C2C} (device=${C2C_DEVICE})"
echo "numa_mem_bw: ${RUN_NUMA_MEM_BW}"
echo "train_step: ${RUN_TRAIN_STEP} (single_node=${TRAIN_STEP_SINGLE_NODE} multi_node=${TRAIN_STEP_MULTI_NODE})"
echo "checkpoint_io: ${RUN_CHECKPOINT_IO}"
echo "fp4_checks: ${ENABLE_FP4} (runtime=${FP4_RUNTIME} stack_profile=${FP4_STACK_PROFILE:-<auto>} preset=${FP4_PRESET} warmup=${FP4_WARMUP} iters=${FP4_ITERS})"
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  if [[ "$FP4_RUNTIME" == "container" ]]; then
    echo "fp4_smoke: shape=${FP4_SMOKE_M}x${FP4_SMOKE_N}x${FP4_SMOKE_K} warmup=${FP4_SMOKE_WARMUP} iters=${FP4_SMOKE_ITERS} rounds=${FP4_SMOKE_ROUNDS} skew_threshold_pct=${FP4_SMOKE_SKEW_THRESHOLD_PCT} stack_profile=${FP4_STACK_PROFILE} image=${FP4_IMAGE}"
  else
    echo "fp4_smoke: shape=${FP4_SMOKE_M}x${FP4_SMOKE_N}x${FP4_SMOKE_K} warmup=${FP4_SMOKE_WARMUP} iters=${FP4_SMOKE_ITERS} rounds=${FP4_SMOKE_ROUNDS} skew_threshold_pct=${FP4_SMOKE_SKEW_THRESHOLD_PCT} stack_profile=${FP4_STACK_PROFILE}"
  fi
fi
echo "bootstrap_nodes: ${BOOTSTRAP_NODES} (sync_code=${BOOTSTRAP_SYNC_CODE} system_packages=${BOOTSTRAP_INSTALL_SYSTEM_PACKAGES} python_deps=${BOOTSTRAP_INSTALL_PYTHON_DEPS})"
echo "bootstrap_host_parity_image: ${BOOTSTRAP_HOST_PARITY_IMAGE}"
echo "mamf: ${ENABLE_MAMF} (mode=${MAMF_MODE} concurrent=${MAMF_CONCURRENT})"
echo "allreduce_stability: ${ENABLE_ALLREDUCE_STABILITY} (payload_gib=${ALLREDUCE_PAYLOAD_GIB} iters=${ALLREDUCE_ITERS} warmup=${ALLREDUCE_WARMUP})"
echo "allreduce_latency_comp: ${ENABLE_ALLREDUCE_LATENCY_COMP} (payload_gib=${ALLREDUCE_LATENCY_PAYLOAD_GIB} chunks=${ALLREDUCE_LATENCY_CHUNKS} iters=${ALLREDUCE_LATENCY_ITERS} warmup=${ALLREDUCE_LATENCY_WARMUP})"
echo "allgather_control_plane: ${ENABLE_ALLGATHER_CONTROL_PLANE} (iters=${ALLGATHER_CONTROL_ITERS} warmup=${ALLGATHER_CONTROL_WARMUP})"
echo "nccl_alltoall: ${ENABLE_NCCL_ALLTOALL} (min=${NCCL_ALLTOALL_MIN_BYTES} max=${NCCL_ALLTOALL_MAX_BYTES} iters=${NCCL_ALLTOALL_ITERS} warmup=${NCCL_ALLTOALL_WARMUP})"
echo "nccl_algo_comparison: ${ENABLE_NCCL_ALGO_COMPARISON} (algos=${NCCL_ALGOS})"
echo "modern_llm_profile: ${MODERN_LLM_PROFILE}"
echo "scorecard_cost_model: gpu_hourly_cost_usd=${GPU_HOURLY_COST_USD:-<unset>}"
echo "coverage_delta: baseline_run_id=${COVERAGE_BASELINE_RUN_ID:-<unset>}"
echo "multinode_readiness: strict=${STRICT_MULTINODE_READINESS} check_only=${MULTINODE_READINESS_CHECK_ONLY} artifact=runs/${RUN_ID}/structured/${RUN_ID}_multinode_readiness.json"
echo "canonical_completeness: strict=${STRICT_CANONICAL_COMPLETENESS} min_coverage=${CANONICAL_MIN_COVERAGE_PCT} min_advanced=${CANONICAL_MIN_ADVANCED_COVERAGE_PCT}"
echo "partial_resume: auto=${AUTO_RESUME_PARTIAL_STEPS} max_attempts=${PARTIAL_RESUME_MAX_ATTEMPTS}"
echo ""

common_args=(--run-id "$RUN_ID" --hosts "$HOSTS")
common_args+=(--ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  common_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  common_args+=(--ssh-key "$SSH_KEY")
fi

if [[ "$BOOTSTRAP_NODES" -eq 1 ]]; then
  bootstrap_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --host-parity-image "$BOOTSTRAP_HOST_PARITY_IMAGE"
    --torch-index-url "$BOOTSTRAP_TORCH_INDEX_URL"
    --torch-version "$BOOTSTRAP_TORCH_VERSION"
  )
  if [[ -n "$LABELS" ]]; then
    bootstrap_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    bootstrap_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$BOOTSTRAP_INSTALL_SYSTEM_PACKAGES" -eq 0 ]]; then
    bootstrap_args+=(--skip-system-packages)
  fi
  if [[ "$BOOTSTRAP_SYNC_CODE" -eq 0 ]]; then
    bootstrap_args+=(--skip-sync-code)
  fi
  if [[ "$BOOTSTRAP_INSTALL_PYTHON_DEPS" -eq 0 ]]; then
    bootstrap_args+=(--skip-python-deps)
  fi
  run_required_step "bootstrap_nodes" "${ROOT_DIR}/scripts/bootstrap_cluster_nodes.sh" "${bootstrap_args[@]}"
fi

preflight_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$SSH_KEY" ]]; then
  preflight_args+=(--ssh-key "$SSH_KEY")
fi
run_required_step "preflight_services" "${ROOT_DIR}/scripts/preflight_cluster_services.sh" "${preflight_args[@]}"

run_required_step "discovery" "${ROOT_DIR}/scripts/collect_discovery_and_tcp_sysctl.sh" "${common_args[@]}"

if [[ "$RUN_QUICK_FRICTION" -eq 1 ]]; then
  quick_friction_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --checks "$QUICK_FRICTION_CHECKS"
    --timeout-sec "$QUICK_FRICTION_TIMEOUT_SEC"
    --torch-version "$QUICK_FRICTION_TORCH_VERSION"
    --torch-index-url "$QUICK_FRICTION_TORCH_INDEX_URL"
    --ngc-image "$QUICK_FRICTION_NGC_IMAGE"
    --hf-model "$QUICK_FRICTION_HF_MODEL"
    --hf-local-dir-base "$QUICK_FRICTION_HF_LOCAL_DIR_BASE"
  )
  if [[ -n "$QUICK_FRICTION_ALLOW_FAILED_CHECKS" ]]; then
    quick_friction_args+=(--allow-failed-checks "$QUICK_FRICTION_ALLOW_FAILED_CHECKS")
  fi
  if [[ -n "$LABELS" ]]; then
    quick_friction_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    quick_friction_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$QUICK_FRICTION_STRICT" -eq 1 ]]; then
    quick_friction_args+=(--strict)
  fi
  run_step "quick_friction_all_nodes" "${ROOT_DIR}/scripts/run_quick_friction_all_nodes.sh" "${quick_friction_args[@]}"
fi

	if [[ "$RUN_MONITORING_EXPECTATIONS" -eq 1 ]]; then
	  monitoring_args=(
	    --run-id "$RUN_ID"
	    --hosts "$HOSTS"
	    --ssh-user "$SSH_USER"
	    --checks "$MONITORING_CHECKS"
	    --k8s-mode "$MONITORING_K8S_MODE"
	    --sample-count "$MONITORING_SAMPLE_COUNT"
	    --dmesg-lines "$MONITORING_DMESG_LINES"
	    --timeout-sec "$MONITORING_TIMEOUT_SEC"
	  )
  if [[ -n "$LABELS" ]]; then
    monitoring_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    monitoring_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$MONITORING_EXPECTATIONS_STRICT" -eq 1 ]]; then
    monitoring_args+=(--strict)
  fi
  run_step "monitoring_expectations_all_nodes" "${ROOT_DIR}/scripts/collect_monitoring_expectations_all_nodes.sh" "${monitoring_args[@]}"
fi

# Required triage readiness bundle (py-spy + strace) for operator-debug path.
triage_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  triage_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  triage_args+=(--ssh-key "$SSH_KEY")
fi
run_required_step "hang_triage_bundle" "${ROOT_DIR}/scripts/collect_hang_triage_bundle.sh" "${triage_args[@]}"

# Required fast distributed connectivity gate before network benchmarks.
connectivity_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --ssh-user "$SSH_USER"
  --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
  --master-port "$CONNECTIVITY_PROBE_MASTER_PORT"
  --barrier-iters "$CONNECTIVITY_PROBE_BARRIER_ITERS"
  --payload-bytes "$CONNECTIVITY_PROBE_PAYLOAD_BYTES"
  --timeout-sec "$CONNECTIVITY_PROBE_TIMEOUT_SEC"
)
if [[ -n "$SSH_KEY" ]]; then
  connectivity_args+=(--ssh-key "$SSH_KEY")
fi
if [[ -n "$OOB_IF" ]]; then
  connectivity_args+=(--oob-if "$OOB_IF")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  connectivity_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
fi
run_required_step "connectivity_probe" "${ROOT_DIR}/scripts/run_torchrun_connectivity_probe.sh" "${connectivity_args[@]}"

# Benchmark A: NCCL all_reduce_perf
run_step "nccl_single_node" "${ROOT_DIR}/scripts/run_nccl_all_reduce.sh" \
  --run-id "${RUN_ID}_node1" \
  --hosts localhost \
  --label "${PRIMARY_LABEL}"

single_nccl_json="${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json"
single_nccl_raw="${RAW_DIR}/${RUN_ID}_node1_${PRIMARY_LABEL}_nccl_all_reduce.log"
if [[ ! -f "$single_nccl_json" && -f "$single_nccl_raw" ]]; then
  single_nccl_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$single_nccl_gpus" =~ ^[1-9][0-9]*$ ]]; then
    run_step "recover_nccl_single_node_json" python3 "${ROOT_DIR}/scripts/parse_nccl_log.py" \
      --input "$single_nccl_raw" \
      --output "$single_nccl_json" \
      --run-id "${RUN_ID}_node1" \
      --hosts localhost \
      --gpus-per-node "$single_nccl_gpus" \
      --command "recovered_from_raw_log:${single_nccl_raw}"
  else
    echo "WARNING: unable to recover single-node NCCL JSON; could not detect GPU count." >&2
  fi
fi

if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ -z "$OOB_IF" ]]; then
    echo "WARNING: multi-node NCCL requested but --oob-if is not set; skipping multi-node NCCL step." >&2
    fail=1
  else
    nccl_multi_args=(
      --run-id "${RUN_ID}_2nodes"
      --hosts "$HOSTS"
      --label "${PRIMARY_LABEL}node2"
      --oob-if "$OOB_IF"
      --socket-ifname "$SOCKET_IFNAME"
    )
    if [[ -n "$SSH_KEY" ]]; then
      nccl_multi_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      nccl_multi_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "nccl_multi_node" "${ROOT_DIR}/scripts/run_nccl_all_reduce.sh" "${nccl_multi_args[@]}"
  fi
fi

# Required NCCL env sensitivity sweep (CROSS_NIC / QPS / CTAs).
nccl_env_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
  --min-bytes "$NCCL_ENV_MIN_BYTES"
  --max-bytes "$NCCL_ENV_MAX_BYTES"
  --warmup "$NCCL_ENV_WARMUP"
  --iters "$NCCL_ENV_ITERS"
)
if [[ -n "$SSH_KEY" ]]; then
  nccl_env_args+=(--ssh-key "$SSH_KEY")
fi
if [[ -n "$OOB_IF" ]]; then
  nccl_env_args+=(--oob-if "$OOB_IF")
fi
if [[ -n "$SOCKET_IFNAME" ]]; then
  nccl_env_args+=(--socket-ifname "$SOCKET_IFNAME")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  nccl_env_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
fi
if [[ "$RUN_NCCL_ENV_SENSITIVITY" -eq 1 ]]; then
  run_step "nccl_env_sensitivity" "${ROOT_DIR}/scripts/run_nccl_env_sensitivity.sh" "${nccl_env_args[@]}"
fi

# Optional multi-node diagnostics
if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" ]]; then
  suite_args=("${ROOT_DIR}/scripts/run_cluster_health_suite.sh" --run-id "${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}" --hosts "$HOSTS")
  suite_args+=(--ssh-user "$SSH_USER")
  if [[ -n "$SSH_KEY" ]]; then
    suite_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$OOB_IF" ]]; then
    suite_args+=(--oob-if "$OOB_IF")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    suite_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "$NCCL_NVLS_ENABLE" ]]; then
    suite_args+=(--nccl-nvls-enable "$NCCL_NVLS_ENABLE")
  fi
  if [[ "$HEALTH_GDR" -eq 1 ]]; then
    suite_args+=(--gdr --gdr-gpu "$HEALTH_GDR_GPU" --gdr-mem-types "$HEALTH_GDR_MEM_TYPES")
    if [[ "$HEALTH_GDR_USE_DMABUF" -eq 1 ]]; then
      suite_args+=(--gdr-use-dmabuf)
    fi
  fi

  case "$HEALTH_SUITE_MODE" in
    collectives)
      suite_args+=(--skip-iperf3 --skip-ib --skip-torchdist)
      ;;
    base)
      ;;
    extended)
      suite_args+=(--extended)
      ;;
    *)
      echo "ERROR: --health-suite must be one of off|collectives|base|extended (got: ${HEALTH_SUITE_MODE})" >&2
      exit 2
      ;;
  esac

  run_step "health_suite_${HEALTH_SUITE_MODE}" "${suite_args[@]}"
fi

# Optional multi-node SHARP readiness check (separate from the health suite so it can be toggled explicitly).
if [[ "${#HOST_ARR[@]}" -gt 1 && "$CHECK_IB_SHARP" -eq 1 ]]; then
  if [[ -z "$OOB_IF" ]]; then
    echo "WARNING: --check-ib-sharp requested but --oob-if is not set; skipping SHARP check." >&2
    fail=1
  else
    sharp_args=(--run-id "${RUN_ID}_ib_sharp_check" --hosts "$HOSTS" --oob-if "$OOB_IF")
    sharp_args+=(--ssh-user "$SSH_USER")
    if [[ -n "$SSH_KEY" ]]; then
      sharp_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      sharp_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    if [[ -n "$IB_SHARP_AM_HOST" ]]; then
      sharp_args+=(--sharp-am-host "$IB_SHARP_AM_HOST")
    fi
    if [[ "$IB_SHARP_ATTEMPT_START_SHARP_AM" -eq 1 ]]; then
      sharp_args+=(--attempt-start-sharp-am)
    fi
    run_step "ib_sharp_check" "${ROOT_DIR}/scripts/check_ib_sharp.sh" "${sharp_args[@]}"
  fi
fi

# Benchmark B: vLLM online serving sweep (local)
vllm_args=(
  --run-id "$RUN_ID"
  --label "$PRIMARY_LABEL"
  --model "$MODEL"
  --isl "$ISL"
  --osl "$OSL"
  --concurrency-range "$CONCURRENCY_RANGE"
  --repeats "$VLLM_REPEATS"
  --port "$PORT"
)
if [[ -n "$TP" ]]; then
  vllm_args+=(--tp "$TP")
fi
run_step "vllm_serve_sweep" env "VLLM_SWEEP_MAX_POINTS_PER_RUN=${VLLM_MAX_POINTS_PER_RUN}" "VLLM_PROFILE_CLASS=${VLLM_PROFILE_CLASS}" "VLLM_PROFILE_SELECTION_REASON=${VLLM_PROFILE_SELECTION_REASON}" "${ROOT_DIR}/scripts/repro/run_vllm_serve_sweep_container.sh" "${vllm_args[@]}"

if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
  vllm_rate_args=(
    --run-id "$RUN_ID"
    --label "$PRIMARY_LABEL"
    --model "$MODEL"
    --isl "$ISL"
    --osl "$OSL"
    --request-rate-range "$VLLM_REQUEST_RATE_RANGE"
    --repeats "$VLLM_REQUEST_RATE_REPEATS"
    --max-concurrency "$VLLM_REQUEST_RATE_MAX_CONCURRENCY"
    --port "$PORT"
  )
  if [[ -n "$TP" ]]; then
    vllm_rate_args+=(--tp "$TP")
  fi
  if [[ -n "$VLLM_REQUEST_RATE_NUM_PROMPTS" ]]; then
    vllm_rate_args+=(--num-prompts "$VLLM_REQUEST_RATE_NUM_PROMPTS")
  fi
  run_step "vllm_request_rate_sweep" env "VLLM_SWEEP_MAX_POINTS_PER_RUN=${VLLM_MAX_POINTS_PER_RUN}" "VLLM_PROFILE_CLASS=${VLLM_PROFILE_CLASS}" "VLLM_PROFILE_SELECTION_REASON=${VLLM_PROFILE_SELECTION_REASON}" "${ROOT_DIR}/scripts/repro/run_vllm_serve_request_rate_sweep_container.sh" "${vllm_rate_args[@]}"
fi

if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
    for vllm_multi_conc in "${VLLM_MULTINODE_CONCURRENCY_VALUES[@]}"; do
      vllm_multi_args=(
        --run-id "$RUN_ID"
        --hosts "$HOSTS"
        --ssh-user "$SSH_USER"
        --model "$MODEL"
        --isl "$ISL"
        --osl "$OSL"
        --concurrency "$vllm_multi_conc"
        --port "$PORT"
        --ray-port "$VLLM_MULTINODE_RAY_PORT"
        --ray-ready-timeout "$VLLM_MULTINODE_RAY_TIMEOUT"
        --server-ready-timeout "$VLLM_MULTINODE_SERVER_TIMEOUT"
        --worker-startup-wait "$VLLM_MULTINODE_WORKER_STARTUP_WAIT"
      )
      if [[ -n "$LABELS" ]]; then
        vllm_multi_args+=(--labels "$LABELS")
      fi
      if [[ -n "$SSH_KEY" ]]; then
        vllm_multi_args+=(--ssh-key "$SSH_KEY")
      fi
      if [[ -n "$TP" ]]; then
        vllm_multi_args+=(--tp "$TP")
      fi
      if [[ -n "$VLLM_MULTINODE_NUM_PROMPTS" ]]; then
        vllm_multi_args+=(--num-prompts "$VLLM_MULTINODE_NUM_PROMPTS")
      fi
      if [[ -n "$VLLM_MULTINODE_IMAGE" ]]; then
        vllm_multi_args+=(--image "$VLLM_MULTINODE_IMAGE")
      fi
      if [[ -n "$SOCKET_IFNAME" ]]; then
        vllm_multi_args+=(--socket-ifname "$SOCKET_IFNAME")
      fi
      if [[ -n "$NCCL_IB_HCA" ]]; then
        vllm_multi_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
      fi
      run_step "vllm_serve_multinode_c${vllm_multi_conc}" "${ROOT_DIR}/scripts/repro/run_vllm_serve_multinode_container.sh" "${vllm_multi_args[@]}"
    done
  else
    echo "WARNING: --run-vllm-multinode requested, but only one host was provided; skipping." >&2
    fail=1
  fi
fi

# Benchmark C: GEMM sanity (all nodes)
gemm_args=(--run-id "$RUN_ID" --hosts "$HOSTS")
gemm_args+=(--ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  gemm_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  gemm_args+=(--ssh-key "$SSH_KEY")
fi
run_step "gemm_sanity" "${ROOT_DIR}/scripts/run_gemm_sanity_all_nodes.sh" "${gemm_args[@]}"

# Benchmark D: GPU STREAM-style memory behavior (all nodes)
if [[ "$RUN_GPU_STREAM" -eq 1 ]]; then
  gpu_stream_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --device "$GPU_STREAM_DEVICE"
    --size-mb "$GPU_STREAM_SIZE_MB"
    --iters "$GPU_STREAM_ITERS"
    --warmup "$GPU_STREAM_WARMUP"
    --dtype "$GPU_STREAM_DTYPE"
  )
  if [[ -n "$LABELS" ]]; then
    gpu_stream_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    gpu_stream_args+=(--ssh-key "$SSH_KEY")
  fi
  run_step "gpu_stream_all_nodes" "${ROOT_DIR}/scripts/run_gpu_stream_bench_all_nodes.sh" "${gpu_stream_args[@]}"
fi

# Optional FP4 checks: DeepGEMM FP8xFP4 smoke + grouped GEMM benchmark.
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  fp4_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --runtime "$FP4_RUNTIME"
    --stack-profile "$FP4_STACK_PROFILE"
    --preset "$FP4_PRESET"
    --warmup "$FP4_WARMUP"
    --iters "$FP4_ITERS"
    --smoke-m "$FP4_SMOKE_M"
    --smoke-n "$FP4_SMOKE_N"
    --smoke-k "$FP4_SMOKE_K"
    --smoke-warmup "$FP4_SMOKE_WARMUP"
    --smoke-iters "$FP4_SMOKE_ITERS"
    --smoke-rounds "$FP4_SMOKE_ROUNDS"
    --smoke-skew-threshold-pct "$FP4_SMOKE_SKEW_THRESHOLD_PCT"
  )
  if [[ -n "$LABELS" ]]; then
    fp4_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    fp4_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$FP4_RUNTIME" == "container" ]]; then
    fp4_args+=(--image "$FP4_IMAGE")
  fi
  # Keep FP4 checks aligned with suite-level bootstrap policy/config.
  if [[ "$BOOTSTRAP_NODES" -eq 1 ]]; then
    fp4_args+=(--bootstrap-nodes)
    if [[ "$BOOTSTRAP_INSTALL_SYSTEM_PACKAGES" -eq 0 ]]; then
      fp4_args+=(--bootstrap-skip-system-packages)
    fi
    if [[ "$BOOTSTRAP_SYNC_CODE" -eq 0 ]]; then
      fp4_args+=(--bootstrap-skip-sync-code)
    fi
    if [[ "$BOOTSTRAP_INSTALL_PYTHON_DEPS" -eq 0 ]]; then
      fp4_args+=(--bootstrap-skip-python-deps)
    fi
  else
    fp4_args+=(--skip-bootstrap-nodes)
  fi
  fp4_args+=(
    --bootstrap-host-parity-image "$BOOTSTRAP_HOST_PARITY_IMAGE"
    --bootstrap-torch-index-url "$BOOTSTRAP_TORCH_INDEX_URL"
    --bootstrap-torch-version "$BOOTSTRAP_TORCH_VERSION"
  )
  run_step "fp4_checks" "${ROOT_DIR}/scripts/run_fp4_checks_all_nodes.sh" "${fp4_args[@]}"
fi

# Optional: MAMF compute-depth scan (all nodes).
if [[ "$ENABLE_MAMF" -eq 1 ]]; then
  mamf_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER" --mode "$MAMF_MODE")
  if [[ -n "$LABELS" ]]; then
    mamf_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    mamf_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$MAMF_CONCURRENT" -eq 1 ]]; then
    mamf_args+=(--concurrent)
  fi
  run_step "mamf_finder" "${ROOT_DIR}/scripts/run_mamf_finder_all_nodes.sh" "${mamf_args[@]}"
fi

# Optional: per-iteration all-reduce jitter profile.
if [[ "$ENABLE_ALLREDUCE_STABILITY" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allreduce-stability requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    stability_args=(
      --run-id "$RUN_ID"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --label "allreduce_stability"
      --payload-gib "$ALLREDUCE_PAYLOAD_GIB"
      --iters "$ALLREDUCE_ITERS"
      --warmup "$ALLREDUCE_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      stability_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      stability_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      stability_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      stability_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allreduce_stability" "${ROOT_DIR}/scripts/run_allreduce_stability.sh" "${stability_args[@]}"
  fi
fi

# Optional: all-reduce latency comparison (1x large vs many small).
if [[ "$ENABLE_ALLREDUCE_LATENCY_COMP" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allreduce-latency-comp requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    latency_args=(
      --run-id "$RUN_ID"
      --label "allreduce_latency_comp"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --payload-gib "$ALLREDUCE_LATENCY_PAYLOAD_GIB"
      --chunks "$ALLREDUCE_LATENCY_CHUNKS"
      --iters "$ALLREDUCE_LATENCY_ITERS"
      --warmup "$ALLREDUCE_LATENCY_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      latency_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      latency_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      latency_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      latency_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allreduce_latency_comp" "${ROOT_DIR}/scripts/run_allreduce_latency_comp.sh" "${latency_args[@]}"
  fi
fi

# Optional: control-plane collectives (all_gather_object vs tensor collectives).
if [[ "$ENABLE_ALLGATHER_CONTROL_PLANE" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allgather-control-plane requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    allgather_args=(
      --run-id "$RUN_ID"
      --label "allgather_control_plane"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --iters "$ALLGATHER_CONTROL_ITERS"
      --warmup "$ALLGATHER_CONTROL_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      allgather_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      allgather_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      allgather_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      allgather_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allgather_control_plane" "${ROOT_DIR}/scripts/run_allgather_control_plane.sh" "${allgather_args[@]}"
  fi
fi

# Optional: NCCL all-to-all benchmark (single-node + multi-node).
if [[ "$ENABLE_NCCL_ALLTOALL" -eq 1 ]]; then
  run_step "nccl_alltoall_single_node" "${ROOT_DIR}/scripts/run_nccl_all_to_all.sh" \
    --run-id "${RUN_ID}_node1_alltoall" \
    --hosts localhost \
    --label "${PRIMARY_LABEL}" \
    --min-bytes "${NCCL_ALLTOALL_MIN_BYTES}" \
    --max-bytes "${NCCL_ALLTOALL_MAX_BYTES}" \
    --iters "${NCCL_ALLTOALL_ITERS}" \
    --warmup "${NCCL_ALLTOALL_WARMUP}"

  if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
    if [[ -z "$OOB_IF" ]]; then
      echo "WARNING: --enable-nccl-alltoall requested for multi-node but --oob-if is not set; skipping multi-node all-to-all." >&2
      fail=1
    else
      alltoall_multi_args=(
        --run-id "${RUN_ID}_2nodes_alltoall"
        --hosts "$HOSTS"
        --label "${PRIMARY_LABEL}node2"
        --min-bytes "${NCCL_ALLTOALL_MIN_BYTES}"
        --max-bytes "${NCCL_ALLTOALL_MAX_BYTES}"
        --iters "${NCCL_ALLTOALL_ITERS}"
        --warmup "${NCCL_ALLTOALL_WARMUP}"
        --oob-if "$OOB_IF"
        --socket-ifname "$SOCKET_IFNAME"
      )
      if [[ -n "$SSH_KEY" ]]; then
        alltoall_multi_args+=(--ssh-key "$SSH_KEY")
      fi
      if [[ -n "$NCCL_IB_HCA" ]]; then
        alltoall_multi_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
      fi
      run_step "nccl_alltoall_multi_node" "${ROOT_DIR}/scripts/run_nccl_all_to_all.sh" "${alltoall_multi_args[@]}"
    fi
  fi
fi

# Optional: NCCL algorithm comparison (Ring/Tree/NVLS/auto).
if [[ "$ENABLE_NCCL_ALGO_COMPARISON" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" ]]; then
    echo "WARNING: --enable-nccl-algo-comparison requested for multi-node but --oob-if is not set; skipping." >&2
    fail=1
  else
    algo_args=(
      --run-id "$RUN_ID"
      --hosts "$HOSTS"
      --algos "$NCCL_ALGOS"
    )
    if [[ -n "$SSH_KEY" ]]; then
      algo_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      algo_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      algo_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      algo_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "nccl_algo_comparison" "${ROOT_DIR}/scripts/run_nccl_algo_comparison.sh" "${algo_args[@]}"
  fi
fi

# Optional: CPU<->GPU C2C memcpy benchmark (local/primary node)
if [[ "$RUN_C2C" -eq 1 ]]; then
  run_step "c2c_memcpy" "${ROOT_DIR}/scripts/run_c2c_memcpy_bench.sh" \
    --run-id "$RUN_ID" \
    --label "$PRIMARY_LABEL" \
    --device "$C2C_DEVICE" \
    --bw-sizes "$C2C_BW_SIZES" \
    --lat-sizes "$C2C_LAT_SIZES" \
    --bw-iters "$C2C_BW_ITERS" \
    --lat-iters "$C2C_LAT_ITERS" \
    --warmup "$C2C_WARMUP"
fi

# Optional: NUMA memory-bandwidth probe (all nodes)
if [[ "$RUN_NUMA_MEM_BW" -eq 1 ]]; then
  numa_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --bytes "$NUMA_BYTES"
    --iters "$NUMA_ITERS"
    --threads "$NUMA_THREADS"
    --warmup "$NUMA_WARMUP"
  )
  if [[ -n "$LABELS" ]]; then
    numa_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    numa_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$NUMA_NODES" ]]; then
    numa_args+=(--nodes "$NUMA_NODES")
  fi
  if [[ -n "$NUMA_CPU_NODE" ]]; then
    numa_args+=(--cpu-node "$NUMA_CPU_NODE")
  fi
  run_step "numa_mem_bw" "${ROOT_DIR}/scripts/run_numa_mem_bw_all_nodes.sh" "${numa_args[@]}"
fi

# Optional: end-to-end transformer train-step benchmark.
if [[ "$RUN_TRAIN_STEP" -eq 1 ]]; then
  train_common=(
    --run-id "$RUN_ID"
    --ssh-user "$SSH_USER"
    --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
    --master-port "$TRAIN_MASTER_PORT"
    --steps "$TRAIN_STEPS"
    --warmup-steps "$TRAIN_WARMUP_STEPS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --seq-len "$TRAIN_SEQ_LEN"
    --hidden "$TRAIN_HIDDEN"
    --layers "$TRAIN_LAYERS"
    --heads "$TRAIN_HEADS"
    --mlp-ratio "$TRAIN_MLP_RATIO"
    --precision "$TRAIN_PRECISION"
    --fsdp "$TRAIN_FSDP"
    --lr "$TRAIN_LR"
  )
  if [[ -n "$SSH_KEY" ]]; then
    train_common+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$OOB_IF" ]]; then
    train_common+=(--oob-if "$OOB_IF")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    train_common+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "$NCCL_NVLS_ENABLE" ]]; then
    train_common+=(--nccl-nvls-enable "$NCCL_NVLS_ENABLE")
  fi

  if [[ "$TRAIN_STEP_SINGLE_NODE" -eq 1 ]]; then
    run_step "train_step_single_node" "${ROOT_DIR}/scripts/run_torchrun_transformer_train_step.sh" \
      "${train_common[@]}" \
      --hosts "${HOST_ARR[0]}" \
      --label "${PRIMARY_LABEL}_single_node"
  fi

  if [[ "$TRAIN_STEP_MULTI_NODE" -eq 1 ]]; then
    if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
      run_step "train_step_multi_node" "${ROOT_DIR}/scripts/run_torchrun_transformer_train_step.sh" \
        "${train_common[@]}" \
        --hosts "$HOSTS" \
        --label "${PRIMARY_LABEL}_multinode"
    else
      echo "WARNING: --run-train-step requested multi-node run, but only one host provided; skipping multi-node train-step." >&2
    fi
  fi
fi

# Optional: checkpoint-like write/read benchmark (all nodes).
if [[ "$RUN_CHECKPOINT_IO" -eq 1 ]]; then
  ckpt_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --test-dir "$CHECKPOINT_TEST_DIR"
    --bytes "$CHECKPOINT_BYTES"
    --block-size "$CHECKPOINT_BLOCK_SIZE"
    --files "$CHECKPOINT_FILES"
    --fsync "$CHECKPOINT_FSYNC"
    --write 1
    --read 1
  )
  if [[ -n "$LABELS" ]]; then
    ckpt_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    ckpt_args+=(--ssh-key "$SSH_KEY")
  fi
  run_step "checkpoint_io" "${ROOT_DIR}/scripts/run_checkpoint_io_all_nodes.sh" "${ckpt_args[@]}"
fi

# Storage: fio (all nodes)
fio_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --ssh-user "$SSH_USER"
  --test-dir "$FIO_TEST_DIR"
  --file-size "$FIO_FILE_SIZE"
  --runtime "$FIO_RUNTIME"
  --repeats "$FIO_REPEATS"
)
if [[ -n "$LABELS" ]]; then
  fio_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  fio_args+=(--ssh-key "$SSH_KEY")
fi
run_step "fio_all_nodes" "${ROOT_DIR}/scripts/run_fio_all_nodes.sh" "${fio_args[@]}"

# Optional: nvbandwidth bundle (all nodes)
if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  nvbw_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --runtime "$NVBANDWIDTH_RUNTIME"
  )
  if [[ -n "$LABELS" ]]; then
    nvbw_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    nvbw_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$NVBANDWIDTH_RUNTIME" == "container" ]]; then
    nvbw_args+=(--image "$NVBANDWIDTH_IMAGE")
  else
    nvbw_args+=(--nvbw-bin "$NVBANDWIDTH_BIN")
  fi
  if [[ "$NVBANDWIDTH_QUICK" -eq 1 ]]; then
    nvbw_args+=(--quick)
  fi
  run_step "nvbandwidth_all_nodes" "${ROOT_DIR}/scripts/run_nvbandwidth_bundle_all_nodes.sh" "${nvbw_args[@]}"
fi

# Plotting (best-effort)
if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json" ]]; then
  run_step "plot_nccl_single_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json" \
    --out-dir "${FIGURES_DIR}" \
    --run-id "${RUN_ID}_node1"
else
  echo "WARNING: skipping plot_nccl_single_node; missing ${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json" >&2
fi

if [[ "${#HOST_ARR[@]}" -gt 1 && -f "${STRUCTURED_DIR}/${RUN_ID}_2nodes_nccl.json" ]]; then
  nccl_plot_multi_args=(
    --input "${STRUCTURED_DIR}/${RUN_ID}_2nodes_nccl.json"
    --out-dir "${FIGURES_DIR}"
    --run-id "${RUN_ID}_2nodes"
  )
  if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json" ]]; then
    nccl_plot_multi_args+=(--baseline-input "${STRUCTURED_DIR}/${RUN_ID}_node1_nccl.json")
  fi
  run_step "plot_nccl_multi_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" "${nccl_plot_multi_args[@]}"
fi

if [[ "$ENABLE_NCCL_ALLTOALL" -eq 1 && -f "${STRUCTURED_DIR}/${RUN_ID}_node1_alltoall_nccl_alltoall.json" ]]; then
  run_step "plot_nccl_alltoall_single_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_node1_alltoall_nccl_alltoall.json" \
    --out-dir "${FIGURES_DIR}" \
    --run-id "${RUN_ID}_node1_alltoall"
fi

if [[ "$ENABLE_NCCL_ALLTOALL" -eq 1 && "${#HOST_ARR[@]}" -gt 1 && -f "${STRUCTURED_DIR}/${RUN_ID}_2nodes_alltoall_nccl_alltoall.json" ]]; then
  nccl_alltoall_multi_args=(
    --input "${STRUCTURED_DIR}/${RUN_ID}_2nodes_alltoall_nccl_alltoall.json"
    --out-dir "${FIGURES_DIR}"
    --run-id "${RUN_ID}_2nodes_alltoall"
  )
  if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_node1_alltoall_nccl_alltoall.json" ]]; then
    nccl_alltoall_multi_args+=(--baseline-input "${STRUCTURED_DIR}/${RUN_ID}_node1_alltoall_nccl_alltoall.json")
  fi
  run_step "plot_nccl_alltoall_multi_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" "${nccl_alltoall_multi_args[@]}"
fi

if [[ "$RUN_NCCL_ENV_SENSITIVITY" -eq 1 && -f "${STRUCTURED_DIR}/${RUN_ID}_nccl_env_sensitivity.json" ]]; then
  run_step "plot_nccl_env_sensitivity" python3 "${ROOT_DIR}/analysis/plot_nccl_env_sensitivity.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_nccl_env_sensitivity.json" \
    --output "${FIGURES_DIR}/${RUN_ID}_nccl_env_sensitivity.png" \
    --title "NCCL Env Sensitivity ${RUN_ID}"
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv" ]]; then
  run_step "plot_vllm_serve" python3 "${ROOT_DIR}/analysis/plot_vllm_serve_sweep.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv" \
    --out-dir "${FIGURES_DIR}" \
    --run-id "${RUN_ID}_${PRIMARY_LABEL}"

  run_step "analyze_vllm_slo_goodput" python3 "${ROOT_DIR}/analysis/analyze_vllm_slo_goodput.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv" \
    --run-id "${RUN_ID}" \
    --label "${PRIMARY_LABEL}" \
    --slo-p99-ttft-ms "${VLLM_SLO_P99_TTFT_MS}" \
    --slo-p99-tpot-ms "${VLLM_SLO_P99_TPOT_MS}" \
    --output-json "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_slo_goodput.json" \
    --output-csv "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_slo_goodput.csv"

  run_step "plot_vllm_slo_goodput" python3 "${ROOT_DIR}/analysis/plot_vllm_goodput_slo.py" \
    --input-json "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_slo_goodput.json" \
    --out-dir "${FIGURES_DIR}" \
    --run-id "${RUN_ID}_${PRIMARY_LABEL}"
fi

if [[ "$RUN_VLLM_REQUEST_RATE_SWEEP" -eq 1 ]]; then
  if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep.csv" ]]; then
    run_step "analyze_vllm_request_rate_slo_goodput" python3 "${ROOT_DIR}/analysis/analyze_vllm_request_rate_slo_goodput.py" \
      --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep.csv" \
      --run-id "${RUN_ID}" \
      --label "${PRIMARY_LABEL}" \
      --slo-p99-ttft-ms "${VLLM_SLO_P99_TTFT_MS}" \
      --slo-p99-tpot-ms "${VLLM_SLO_P99_TPOT_MS}" \
      --output-json "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_request_rate_slo_goodput.json" \
      --output-csv "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_request_rate_slo_goodput.csv"

    run_step "plot_vllm_request_rate_sweep" python3 "${ROOT_DIR}/analysis/plot_vllm_request_rate_sweep.py" \
      --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_request_rate_sweep.csv" \
      --out-dir "${FIGURES_DIR}" \
      --run-id "${RUN_ID}_${PRIMARY_LABEL}"

    run_step "plot_vllm_request_rate_slo_goodput" python3 "${ROOT_DIR}/analysis/plot_vllm_request_rate_slo_goodput.py" \
      --input-json "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_vllm_request_rate_slo_goodput.json" \
      --out-dir "${FIGURES_DIR}" \
      --run-id "${RUN_ID}_${PRIMARY_LABEL}"
  fi
fi

if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  vllm_multi_label="$(label_for_index 0)"
  if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_serve.csv" ]]; then
    run_step "plot_vllm_serve_multinode" python3 "${ROOT_DIR}/analysis/plot_vllm_serve_sweep.py" \
      --input "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_serve.csv" \
      --out-dir "${FIGURES_DIR}" \
      --run-id "${RUN_ID}_${vllm_multi_label}_multinode"

    run_step "analyze_vllm_multinode_slo_goodput" python3 "${ROOT_DIR}/analysis/analyze_vllm_slo_goodput.py" \
      --input "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_serve.csv" \
      --run-id "${RUN_ID}" \
      --label "${vllm_multi_label}_multinode" \
      --slo-p99-ttft-ms "${VLLM_SLO_P99_TTFT_MS}" \
      --slo-p99-tpot-ms "${VLLM_SLO_P99_TPOT_MS}" \
      --output-json "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_slo_goodput.json" \
      --output-csv "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_slo_goodput.csv"

    run_step "plot_vllm_multinode_slo_goodput" python3 "${ROOT_DIR}/analysis/plot_vllm_goodput_slo.py" \
      --input-json "${STRUCTURED_DIR}/${RUN_ID}_${vllm_multi_label}_vllm_multinode_slo_goodput.json" \
      --out-dir "${FIGURES_DIR}" \
      --run-id "${RUN_ID}_${vllm_multi_label}_multinode"
  fi
fi

if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" ]]; then
  shopt -s nullglob
  suite_summaries=( "${STRUCTURED_DIR}/${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}_"*"_cluster_health_suite_summary.json" )
  shopt -u nullglob
  if [[ "${#suite_summaries[@]}" -gt 0 ]]; then
    run_step "plot_iperf3_oob" python3 "${ROOT_DIR}/analysis/plot_iperf3.py" \
      --summary "${suite_summaries[0]}" \
      --out "${FIGURES_DIR}/${RUN_ID}_iperf3_oob_tcp.png" \
      --out-json "${STRUCTURED_DIR}/${RUN_ID}_iperf3_oob_tcp.json" \
      --skip-if-missing
  fi
fi

shopt -s nullglob
gemm_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_gemm_gpu_sanity.csv )
shopt -u nullglob
if [[ "${#gemm_inputs[@]}" -gt 0 ]]; then
  run_step "plot_gemm_sanity" python3 "${ROOT_DIR}/analysis/plot_gemm_bar.py" \
    --inputs "${gemm_inputs[@]}" \
    --output "${FIGURES_DIR}/${RUN_ID}_gemm_gpu_sanity.png" \
    --filter-m 16384
fi

if [[ "$ENABLE_MAMF" -eq 1 ]]; then
  shopt -s nullglob
  mamf_summaries=( "${STRUCTURED_DIR}/${RUN_ID}_"*_mamf_summary.json )
  shopt -u nullglob
  if [[ "${#mamf_summaries[@]}" -gt 0 ]]; then
    run_step "plot_mamf" python3 "${ROOT_DIR}/analysis/plot_mamf.py" \
      --summary-inputs "${mamf_summaries[@]}" \
      --output "${FIGURES_DIR}/${RUN_ID}_mamf_straggler.png" \
      --mode straggler \
      --title "MAMF Straggler View ${RUN_ID}"
  fi
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_allreduce_stability.json" ]]; then
  run_step "plot_allreduce_stability" python3 "${ROOT_DIR}/analysis/plot_allreduce_stability.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_allreduce_stability.json" \
    --output "${FIGURES_DIR}/${RUN_ID}_allreduce_stability.png"
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_allreduce_latency_comp.json" ]]; then
  run_step "plot_allreduce_latency_comp" python3 "${ROOT_DIR}/analysis/plot_allreduce_latency_comp.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_allreduce_latency_comp.json" \
    --output "${FIGURES_DIR}/${RUN_ID}_allreduce_latency_comp.png"
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_allgather_control_plane.json" ]]; then
  run_step "plot_allgather_control_plane" python3 "${ROOT_DIR}/analysis/plot_allgather_control_plane.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_allgather_control_plane.json" \
    --output "${FIGURES_DIR}/${RUN_ID}_allgather_control_plane.png"
fi

if [[ "$ENABLE_NCCL_ALGO_COMPARISON" -eq 1 ]]; then
  shopt -s nullglob
  nccl_algo_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_nccl_algo_"*.json )
  shopt -u nullglob
  if [[ "${#nccl_algo_inputs[@]}" -gt 0 ]]; then
    run_step "plot_nccl_algo_comparison" python3 "${ROOT_DIR}/analysis/plot_nccl_algo_comparison.py" \
      --inputs "${nccl_algo_inputs[@]}" \
      --output "${FIGURES_DIR}/${RUN_ID}_nccl_algo_comparison.png" \
      --title "NCCL Algorithm Comparison ${RUN_ID}"
  fi
fi

shopt -s nullglob
fio_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_fio.json )
shopt -u nullglob
for fio_in in "${fio_inputs[@]}"; do
  fio_base="$(basename "$fio_in" .json)"
  run_step "plot_fio_${fio_base}" python3 "${ROOT_DIR}/analysis/plot_fio.py" \
    --input "$fio_in" \
    --out "${FIGURES_DIR}/${fio_base}.png"
done

if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  shopt -s nullglob
  nvbw_sums_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_nvbandwidth_sums.csv )
  shopt -u nullglob
  for nvbw_csv in "${nvbw_sums_inputs[@]}"; do
    nvbw_base="$(basename "$nvbw_csv" _sums.csv)"
    run_step "plot_nvbandwidth_${nvbw_base}" python3 "${ROOT_DIR}/analysis/plot_nvbandwidth_sums.py" \
      --input "$nvbw_csv" \
      --out "${FIGURES_DIR}/${nvbw_base}_sums.png" \
      --title "nvbandwidth SUM metrics ${nvbw_base}"
  done
fi

if [[ "$RUN_GPU_STREAM" -eq 1 ]]; then
  shopt -s nullglob
  gpu_stream_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_gpu_stream.json )
  shopt -u nullglob
  for gpu_stream_json in "${gpu_stream_inputs[@]}"; do
    gpu_stream_base="$(basename "$gpu_stream_json" .json)"
    run_step "plot_gpu_stream_${gpu_stream_base}" python3 "${ROOT_DIR}/analysis/plot_gpu_stream.py" \
      --input "$gpu_stream_json" \
      --out "${FIGURES_DIR}/${gpu_stream_base}.png"
  done
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_c2c_memcpy.json" ]]; then
  run_step "plot_c2c_memcpy" python3 "${ROOT_DIR}/analysis/plot_c2c_memcpy.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_c2c_memcpy.json" \
    --out-dir "${FIGURES_DIR}" \
    --run-id "${RUN_ID}_${PRIMARY_LABEL}"
fi

if [[ "$RUN_NUMA_MEM_BW" -eq 1 ]]; then
  shopt -s nullglob
  numa_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_numa_mem_bw.json )
  shopt -u nullglob
  for inp in "${numa_inputs[@]}"; do
    base="$(basename "$inp" .json)"
    run_step "plot_numa_mem_bw_${base}" python3 "${ROOT_DIR}/analysis/plot_numa_mem_bw.py" \
      --input "$inp" \
      --out "${FIGURES_DIR}/${base}.png" \
      --title "NUMA memcpy bandwidth: ${base}"
  done
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.json" ]]; then
  run_step "plot_train_step_single" python3 "${ROOT_DIR}/analysis/plot_torchrun_train_step.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.json" \
    --out "${FIGURES_DIR}/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.png"
fi

if [[ -f "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.json" ]]; then
  run_step "plot_train_step_multi" python3 "${ROOT_DIR}/analysis/plot_torchrun_train_step.py" \
    --input "${STRUCTURED_DIR}/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.json" \
    --out "${FIGURES_DIR}/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.png"
fi

shopt -s nullglob
meta_inputs=( "${STRUCTURED_DIR}/${RUN_ID}_"*_meta.json )
shopt -u nullglob
for meta_in in "${meta_inputs[@]}"; do
  if [[ "$meta_in" == *_cluster_meta.json ]]; then
    continue
  fi
  meta_base="$(basename "$meta_in" .json)"
  run_step "plot_nvlink_topology_${meta_base}" python3 "${ROOT_DIR}/analysis/plot_nvlink_topology.py" \
    --meta "$meta_in" \
    --fig-out "${FIGURES_DIR}/${meta_base}_nvlink_topology.png" \
    --summary-out "${STRUCTURED_DIR}/${meta_base}_nvlink_topology.json"
done

dashboard_labels=("${PRIMARY_LABEL}")
if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -gt 1 ]]; then
    dashboard_labels+=("$(echo "${LABEL_ARR[1]}" | xargs)")
  else
    dashboard_labels+=("$(echo "${HOST_ARR[1]}" | xargs)")
  fi
fi
dashboard_node_labels_csv="$(IFS=','; echo "${dashboard_labels[*]}")"
run_step "plot_cluster_story_dashboard" python3 "${ROOT_DIR}/analysis/plot_cluster_story_dashboard.py" \
  --run-id "${RUN_ID}" \
  --structured-dir "${STRUCTURED_DIR}" \
  --node-labels "${dashboard_node_labels_csv}" \
  --fig-out "${FIGURES_DIR}/${RUN_ID}_cluster_story_dashboard.png" \
  --summary-out "${STRUCTURED_DIR}/${RUN_ID}_node_parity_summary.json"

if [[ "$RUN_QUICK_FRICTION" -eq 1 || "$RUN_MONITORING_EXPECTATIONS" -eq 1 ]]; then
  operator_labels=()
  for idx in "${!HOST_ARR[@]}"; do
    operator_labels+=("$(label_for_index "$idx")")
  done
  operator_node_labels_csv="$(IFS=','; echo "${operator_labels[*]}")"
  run_step "plot_operator_checks_dashboard" python3 "${ROOT_DIR}/analysis/plot_operator_checks_dashboard.py" \
    --run-id "${RUN_ID}" \
    --structured-dir "${STRUCTURED_DIR}" \
    --node-labels "${operator_node_labels_csv}" \
    --fig-out "${FIGURES_DIR}/${RUN_ID}_operator_checks_dashboard.png" \
    --summary-out "${STRUCTURED_DIR}/${RUN_ID}_operator_checks_dashboard.json"
fi

if [[ "$RUN_FABRIC_EVAL" -eq 1 ]]; then
  fabric_args=(
    --run-id "${RUN_ID}"
    --run-dir "${RUN_DIR}"
    --primary-label "${PRIMARY_LABEL}"
  )
  if [[ -n "$LABELS" ]]; then
    fabric_args+=(--labels "${LABELS}")
  fi
  if [[ -n "$SSH_USER" ]]; then
    fabric_args+=(--ssh-user "${SSH_USER}")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    fabric_args+=(--ssh-key "${SSH_KEY}")
  fi
  if [[ -n "$FABRIC_NMX_URL" ]]; then
    fabric_args+=(--nmx-url "${FABRIC_NMX_URL}")
  fi
  if [[ -n "$FABRIC_NMX_TOKEN" ]]; then
    fabric_args+=(--nmx-token "${FABRIC_NMX_TOKEN}")
  fi
  if [[ -n "$FABRIC_IB_MGMT_HOST" ]]; then
    fabric_args+=(--ib-mgmt-host "${FABRIC_IB_MGMT_HOST}")
  fi
  if [[ -n "$FABRIC_IB_MGMT_USER" ]]; then
    fabric_args+=(--ib-mgmt-user "${FABRIC_IB_MGMT_USER}")
  fi
  if [[ -n "$FABRIC_IB_MGMT_SSH_KEY" ]]; then
    fabric_args+=(--ib-mgmt-ssh-key "${FABRIC_IB_MGMT_SSH_KEY}")
  fi
  if [[ -n "$FABRIC_CUMULUS_HOSTS" ]]; then
    fabric_args+=(--cumulus-hosts "${FABRIC_CUMULUS_HOSTS}")
  fi
  if [[ -n "$FABRIC_CUMULUS_USER" ]]; then
    fabric_args+=(--cumulus-user "${FABRIC_CUMULUS_USER}")
  fi
  if [[ -n "$FABRIC_CUMULUS_SSH_KEY" ]]; then
    fabric_args+=(--cumulus-ssh-key "${FABRIC_CUMULUS_SSH_KEY}")
  fi
  if [[ "$REQUIRE_MANAGEMENT_PLANE" -eq 1 ]]; then
    fabric_args+=(--require-management-plane)
  fi
  run_step "build_fabric_eval" bash -lc 'cd "$1" && shift && python3 -m cluster.scripts.build_fabric_eval "$@"' _ "${ROOT_DIR}/.." "${fabric_args[@]}"
fi

scorecard_args=(
  --run-id "${RUN_ID}"
  --primary-label "${PRIMARY_LABEL}"
  --structured-dir "${STRUCTURED_DIR}"
)
if [[ -n "$GPU_HOURLY_COST_USD" ]]; then
  scorecard_args+=(--gpu-hourly-cost-usd "$GPU_HOURLY_COST_USD")
fi
run_step "build_cluster_scorecard" python3 "${ROOT_DIR}/analysis/build_cluster_scorecard.py" "${scorecard_args[@]}"

run_step "build_mlperf_alignment" python3 "${ROOT_DIR}/analysis/build_mlperf_alignment.py" \
  --run-id "${RUN_ID}" \
  --structured-dir "${STRUCTURED_DIR}"

run_step "analyze_benchmark_coverage" python3 "${ROOT_DIR}/analysis/analyze_benchmark_coverage.py" \
  --run-id "${RUN_ID}" \
  --structured-dir "${STRUCTURED_DIR}"

if [[ -n "$COVERAGE_BASELINE_RUN_ID" ]]; then
  run_step "build_coverage_delta" python3 "${ROOT_DIR}/analysis/build_coverage_delta.py" \
    --run-id "${RUN_ID}" \
    --baseline-run-id "${COVERAGE_BASELINE_RUN_ID}" \
    --structured-dir "${STRUCTURED_DIR}"
fi

run_step "validate_required_artifacts" validate_required_artifacts

manifest_args=(--root "$ROOT_DIR" --run-id "$RUN_ID" --run-dir "$RUN_DIR" --hosts "$HOSTS" --include-figures)
if [[ -n "$LABELS" ]]; then
  manifest_args+=(--labels "$LABELS")
fi
run_step "manifest_refresh" python3 "${ROOT_DIR}/scripts/write_manifest.py" "${manifest_args[@]}"

RENDER_LOCALHOST_REPORT=0
if [[ "$RENDER_LOCALHOST_REPORT_MODE" == "on" ]]; then
  RENDER_LOCALHOST_REPORT=1
elif [[ "$RENDER_LOCALHOST_REPORT_MODE" == "auto" && "$IS_LOCALHOST_PACKAGE" -eq 1 ]]; then
  RENDER_LOCALHOST_REPORT=1
fi

if [[ "$RENDER_LOCALHOST_REPORT" -eq 1 ]]; then
  localhost_label="$(label_for_index 0)"
  run_step "render_localhost_field_report_package" python3 "${ROOT_DIR}/scripts/render_localhost_field_report_package.py" \
    --run-id "${RUN_ID}" \
    --label "${localhost_label}" \
    --root "${ROOT_DIR}" \
    --run-dir "${RUN_DIR}" \
    --report "${REPORTS_DIR}/field-report-localhost.md" \
    --notes "${REPORTS_DIR}/field-report-localhost-notes.md" \
    --publish-report "${ROOT_DIR}/field-report-localhost.md" \
    --publish-notes "${ROOT_DIR}/field-report-localhost-notes.md"
fi

# Rewrite the canonical manifest after every recorded suite step and report
# artifact is stable. This direct call is intentionally not redirected into
# the run tree, so every listed file keeps the digest captured here.
final_manifest_args=("${manifest_args[@]}" --finalize)
if ! python3 "${ROOT_DIR}/scripts/write_manifest.py" "${final_manifest_args[@]}"; then
  echo "ERROR: final manifest write failed: ${MANIFEST_PATH}" >&2
  fail=1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: suite completed without a final manifest: ${MANIFEST_PATH}" >&2
  fail=1
fi

echo ""
echo "========================================"
echo "Suite Complete"
echo "========================================"
echo "Run directory: runs/${RUN_ID}"
echo "Manifest: runs/${RUN_ID}/manifest.json"
echo "Field report template: docs/field-report-template.md"

if [[ "$fail" -ne 0 ]]; then
  echo "STATUS: FAILED (one or more steps failed)" >&2
  exit 1
fi
echo "STATUS: OK"
