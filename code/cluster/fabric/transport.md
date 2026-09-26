# Transport Diagnostics Guide

`cluster.fabric.transport` executes bounded client probes for a transport ladder:

1. Local pairwise GPU communication
2. Host-buffer RDMA perftest
3. CUDA GPUDirect RDMA perftest
4. An explicit NCCL command
5. Optional TCP throughput with iperf3

Every successful case must exit cleanly and produce a recognized metric. A clean process with unparseable output is `invalid`. Missing tools and unsupported CUDA memory modes are `unsupported`. The tool never substitutes synthetic data.

The tool does not prepare remote servers or change host, NIC, or switch configuration. Results are diagnostic and noncanonical.

Client commands run in a new local process group. On timeout, the runner sends TERM and then KILL only to that owned group. It does not use a global process-name kill. Raw evidence records each timeout and termination action.

## Prepare Servers

Start one perftest server for each requested client case. Use separate control ports so both can wait while the ladder runs. Match the client HCA, HCA port, payload, iterations, queue-pair count, GPU, and CUDA memory mode.

```bash
# Host-buffer server on the prepared peer
ib_write_bw -d <hca> -i 1 -q 1 -s 8388608 -n 100 \
  --report_gbits --out_json -p 18515

# GDR server on the prepared peer
ib_write_bw -d <hca> -i 1 -q 1 -s 8388608 -n 100 \
  --report_gbits --out_json --use_cuda 0 -p 18516

# Optional TCP server on the prepared peer
iperf3 -s
```

For `read` or `send`, use the matching `ib_read_bw` or `ib_send_bw` executable on both peers. If the selected perftest release requires more GDR flags on its server, supply the same supported mode there.

## Run the Ladder

The local rung accepts an explicit bounded pairwise command. This repository's Chapter 4 suite provides a pairwise NCCL path:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m ch04.bandwidth_benchmark_suite_multigpu --quick
```

Pass that command to the ladder from the `code` directory. The tool captures `nvidia-smi topo -m` and `nvidia-smi topo -p2p r` before it runs the command. It parses lines such as `GPU 0 → GPU 1: 812.50 GB/s`. A custom command can emit `AISP_LOCAL_P2P_JSON={"pairs":[{"src_gpu":0,"dst_gpu":1,"bandwidth_gbps":812.5}]}` on one line. The topology links and peer read capability remain separate from the measured pair rate. Capability does not prove the runtime route.

```bash
python -m cluster.fabric.transport \
  --run-id 2026-09-26_transport \
  --run-dir cluster/runs/2026-09-26_transport \
  --cases local_p2p,host_rdma,gdr,nccl,iperf \
  --local-p2p-command 'torchrun --standalone --nproc_per_node=2 -m ch04.bandwidth_benchmark_suite_multigpu --quick' \
  --workload-id <serving-profile-id> \
  --server-address <data-plane-address> \
  --hca <hca> \
  --rail rail0 \
  --ib-port 1 \
  --host-rdma-control-port 18515 \
  --gdr-control-port 18516 \
  --payload-bytes 8388608 \
  --iterations 100 \
  --queue-pairs 1 \
  --direction write \
  --gpu-id 0 \
  --nccl-command '<bounded nccl-tests launch command>' \
  --timeout-seconds 180
```

The NCCL command is intentionally explicit because launchers, rank placement, interfaces, and containers vary across clusters. Use the same host pair, HCA, rail, and payload regime as the lower transport layers. The exact command and output are retained. Command text is a declaration. It does not prove the transport or path that NCCL used.

Set `--workload-id` to the same value used by the serving profile or application signal export when these measurements belong to that workload. The supplied value takes precedence over the fallback command hash. It links declared scope across artifacts but does not prove an effective network path.

Run only the cases that have prepared endpoints. For example:

```bash
python -m cluster.fabric.transport \
  --run-id 2026-09-26_host_rdma \
  --run-dir cluster/runs/2026-09-26_host_rdma \
  --cases host_rdma \
  --server-address <data-plane-address> \
  --hca <hca> \
  --rail rail0
```

## Interpretation

| Shape | Next check |
| --- | --- |
| Local GPU P2P is weak | GPU topology, peer access, NVLink state, PCIe fallback, and pair placement |
| Host RDMA is weak | HCA binding, link state, path, MTU, congestion, and host memory path |
| Host RDMA is healthy and GDR is weak | GPUDirect support, CUDA memory type, registration, PCIe topology, and GPU-to-NIC affinity |
| GDR is healthy and NCCL is weak | Rank placement, selected NICs, collective algorithm, protocol, channels, and message regime |
| NCCL is healthy and the application is weak | Rank arrival skew, resource contention, dependencies, and lost compute overlap |

iperf3 is a TCP control measurement. It does not prove that the RDMA data path is healthy. Compare only measurements with matching identities and retained provenance.

The structured case rows keep `declared_identity`, `observed_identity`, and `dimension_validation` separate. A generated perftest command records the configured peer, HCA, port, and direction. Parsed output observes the message size and rate. A successful GDR case is also evidence that the selected CUDA memory mode executed. The selected HCA and effective path remain unverified unless runtime evidence reports them.

For NCCL, the requested payload must appear in parsed benchmark rows. A mismatch is `invalid`. `NCCL INFO NET/IB` lines can corroborate runtime HCA and port use. Caller labels, environment tokens, and command substrings remain declarations. Without corroborating peer and rank evidence, `matched_path_eligible` stays false even when the rate is valid.

Perftest and iperf rates use `Gb/s`. NCCL tests and the local pairwise parser use `GB/s`. The structured metric and normalized signal retain the unit so these values cannot be compared without conversion.

When host RDMA and GDR both produce positive rates with the same generated settings and parsed message size, `comparisons.host_vs_gdr` reports the GDR to host rate ratio. The ratio remains descriptive. Missing effective-path and data-correctness evidence blocks causal conclusions.

## Output Contract

| Path | Contents |
| --- | --- |
| `raw/<run_id>_transport_<case>.json` | Exact command, stdout, stderr, return code, duration, identity, and parsed metric |
| `structured/<run_id>_transport_diagnostics.json` | Ladder status, matching dimensions, case metrics, limitations, and artifact references |
| `structured/<run_id>_transport_diagnostics_manifest.json` | File sizes and SHA-256 values for this diagnostic package |
| `reports/<run_id>_transport_diagnostics.md` | Human-readable comparison table |

The tool also merges a noncanonical reference under `tools.transport_diagnostics` in the run directory's `manifest.json`. Existing manifest keys are preserved through the shared diagnostic manifest lock and atomic replacement.

Each successful case exports one normalized `role=measurement` throughput signal. It contains the command start and end Unix timestamps, the hashed local collector host, the collector clock domain, a stable workload ID, the selected or observed path label, the raw evidence reference, and the metric unit. These are contextual measurements. They are not causal fabric counters or application symptoms.

The CLI exits 0 for `ok` and for `partial` results whose remaining cases are unsupported. It exits 2 when no requested measurement is supported and 1 if any case has invalid output or a command error. `--require-all` also makes a partial result exit 2. An omitted server, HCA, GPU ID, local P2P command, or NCCL command produces an explicit unsupported case.
