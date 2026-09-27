# Diagnostic tools on Verda B200

Validation date: September 26, 2026.

The [SGLang counter follow-up](sglang_failure_counter_validation.md) fixes the missing native metrics and completes the four-arm serving comparison on the same B200s. This report retains the original validation stage and its receipts.

These are functional diagnostic checks on two NVIDIA B200 GPUs. They do not establish a canonical benchmark speedup. The initial implementation was merged at `127f8ad5e439cd0193a3a6a0dfce4fb8c8fe072a`. Source hashes in the receipts identify fixes tested after that commit.

## Environment

The target was a Verda KVM instance with two B200 GPUs, 183,359 MiB each, and an NV18 link. The host used Python 3.12.3, PyTorch 2.13.0 with CUDA 13.0, and NVIDIA driver 580.173.02. GPU workloads ran sequentially in an isolated source checkout. The process table was empty before each GPU handoff.

The guest exposes a virtio network adapter and no RDMA NIC. PCI enumeration, `rdma link`, and `/sys/class/infiniband` agreed. NCCL reported no InfiniBand device and used P2P/CUMEM for local GPU data channels. No network RDMA, GPUDirect RDMA, or switch-fabric performance was measured. At this validation stage, the four-arm serving comparison was unqualified because the unmodified SGLang build did not expose required native failure-counter evidence.

Verda documents ordinary B200 instances with NVLink and Instant Clusters with InfiniBand. See [B200 configurations](https://verda.com/b200) and [cluster networking](https://docs.verda.com/clusters/instant-clusters/). These checks qualify the observed instance capabilities, not every Verda product.

## Coverage

| Area | Real execution | Result |
| --- | --- | --- |
| Collective diagnosis | Four collectives, three payload sizes, four scenarios, two ranks | 48 groups, 36 usable comparisons, 960 measured rank records, all correctness checks passed |
| Profiling | Separate PyTorch and Nsight Systems sweeps | Both PyTorch traces contain NCCL collective events. Nsight records 64 scenario ranges and 144 NCCL kernels |
| Local transport | TCP iperf, Chapter 4 pairwise GPU workload, and nccl-tests all-reduce | Recognized measurements, NV18 topology, peer access, and actual P2P/CUMEM channels |
| Network diagnosis | Linux state, IPv4 and IPv6 loopback, MTU probes, forward and reverse TCP with one and four flows | Loopback probes passed. The management interface rejected the 9,000-byte MTU probe |
| Packet analysis | Native TShark 4.2.2 export and live collector PCAP path | Fixed boolean parsing. A focused capture verified eight SYN, four SYN/ACK, four resets, and IPv4/IPv6 echo |
| Serving | vLLM monolithic, vLLM P/D, and SGLang monolithic in two opposite orders | Six accepted iterations with exact output vectors, prompt echoes, cancellations, and GPU release |
| SGLang P/D | Two native warmups with real KV handoff | Requests and transfers executed. Both runs rejected missing required failure-counter families before measurement |
| Fabric and RDMA | Capability probes | Explicit unsupported results. No RDMA NIC or switch endpoint was available |
| Cross-layer analysis | Actual collective, network, and transport artifacts plus an unsupported fabric-capability artifact | 91 signals and 144 contextual overlaps. No fabric counter correlation was claimed |

The focused packet capture observed no gateway ARP exchange because the neighbor entry remained cached. No neighbor cache was flushed. A separate native replay parsed ambient ARP requests. ARP replies and IPv6 neighbor-discovery events retain fixture coverage.

The TCP rates are loopback measurements used to exercise collection and parsing. They do not measure external network bandwidth. The management-network result is partial because the selected jumbo-MTU probe failed.

## Serving results and limit

The final runs used vLLM 0.26.0, SGLang 0.5.20, and Qwen3-8B on the same two B200s. Each run visited the three supported arms once with one warmup. The second run reversed their order. SGLang P/D ran last in both attempts, so its expected rejection could not prevent the other arms from being tested.

Across the six accepted iterations, both completed output token vectors matched the vLLM monolithic reference. Every prompt echo matched the authoritative trace, and all six measured cancellations passed. Both vLLM P/D iterations showed positive native transfer requests, transfer time, and prefill/decode activity, with no increase in failed transfers. All eight arm lifecycles retained GPU release evidence. The final process table was empty, both GPUs used zero memory, and persistence mode was restored to disabled.

SGLang P/D completed its warmup requests, cancellation, and real NIXL handoff. Its multiprocess metrics response omitted both required failure-counter families and their type metadata. Each attempt therefore rejected before the measured P/D interval. The raw responses are retained. A declared zero cannot substitute for these missing native counters.

These are two repeated validations of the three supported arms, plus two rejected SGLang P/D attempts. They do not qualify a four-arm comparison. The [serving receipt](validation/diagnostic-tools-b200-20260926/serving.json) records exact source, profile, trace, request, telemetry, and cleanup evidence. The [serving guide](../labs/serving_comparison/README.md) explains the required runtime capabilities.

## Fixes found by validation

- TShark emitted `True` and `False`. The parser previously recognized only `1` and `0`, which missed TCP flags. Both formats are now supported and malformed values fail explicitly.
- The Linux socket collector passed bare IPv6 destinations to `ss`, which rejected `::1`. The collector now supplies an explicit address family and host prefix.
- NVIDIA SMI underlined matrix headers with terminal escape sequences. The transport parser now strips that formatting before reading NVLink and peer-access matrices.
- Serving readiness could accept a healthy router before its prefill and decode workers had loaded. The lifecycle now waits for every configured readiness and identity endpoint and the complete owned GPU allocation. Requests have bounded control-plane timeouts.
- Serving profiles declared clocks and P/D launch provenance without proving them. The lifecycle now reads application clocks and binds the launch-file digest, child roles, backend routes, model name, and GPU pools to the actual launch and observed GPU processes.
- Text prompt transport requires exact returned prompt token ids, including canceled requests. Missing or mismatched ids reject the request.
- The vLLM P/D proxy accepted only token-id arrays. It now forwards explicit text or token ids unchanged, so both stages use the selected prompt transport.
- Native KV telemetry accepted intervals with failed transfers. Any increase in the failure counter now rejects the interval, and telemetry reads have a timeout. SGLang can declare an uninitialized failure counter as zero only when the exact native counter family is present. A missing family, wrong type, or label filter still rejects the evidence.
- Startup failures retain the process termination and GPU release check in a cleanup artifact. Request rows are saved before validation can reject a warmup or measured interval.
- The initial README edits were missing from the generator. The generator now preserves all diagnostic links, and all 61 generated READMEs match their source.

## Evidence

The [collective receipt](validation/diagnostic-tools-b200-20260926/collective.json), [native packet receipt](validation/diagnostic-tools-b200-20260926/network-packets.json), and [host receipt](validation/diagnostic-tools-b200-20260926/host.json) include source hashes, checked counts, and raw artifact hashes. Receipts retain the source identity of each validation stage. The [final network receipt](validation/diagnostic-tools-b200-20260926/network-final.json) records the corrected IPv4/IPv6 collector and native packet replay. The [public entrypoint receipt](validation/diagnostic-tools-b200-20260926/public-entrypoints.json) records retained-artifact analysis and profiler inspection. It also preserves the IPv6 failure that led to the fix. Raw host logs and profiler traces remain outside the public repository because they contain live infrastructure identifiers.

## Regression checks

The [control-test receipt](validation/diagnostic-tools-b200-20260926/control-tests.json) records 102 passing tests on the Verda host with CUDA hidden and no skips. These cover parsers, process control, collection, analysis, command dispatch, and README generation. A separate 41-test serving suite also passed on Verda. These regression checks complement the GPU runs.

Repository checks found no correctness lint or silent-fallback audit findings. The benchmark contract check covered 963 files with zero errors or warnings.

## Reproduce the GPU diagnostics

From `code/`, use a new artifact directory for each run:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m ch04.collective_diagnosis_tool run \
  --output <artifacts>/collective-default.json \
  --collectives all_reduce,all_gather,reduce_scatter,all_to_all \
  --message-sizes 256KiB,4MiB,32MiB \
  --scenarios healthy,delayed_rank,competing_gpu_workload,forced_dependency \
  --warmups 3 --rounds 10 --timeout-seconds 300

python -m cli.aisp tools transport-diagnose -- \
  --run-id b200-local-transports --run-dir <artifacts>/transport \
  --cases local_p2p,nccl --payload-bytes 8388608 --iterations 20 \
  --local-p2p-command 'torchrun --standalone --nproc_per_node=2 -m ch04.bandwidth_benchmark_suite_multigpu --quick' \
  --nccl-command '<nccl-tests>/all_reduce_perf -b 8388608 -e 8388608 -f 2 -g 2 -w 5 -n 20 -c 1' \
  --timeout-seconds 180
```

The [collective guide](../ch04/collective_diagnosis.md) gives the separate profiling commands. The [network guide](../ch03/network_diagnosis.md) covers the live collector and retained packet replay. Run the [serving comparison](../labs/serving_comparison/README.md) with the engine versions, model, GPU placement, and launch digests in its profile.
