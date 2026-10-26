# Zymtrace capture preparation

Status: preparation only. Zymtrace was not installed, configured, started, or
used for a capture while preparing this directory.

This runbook adds an opt-in Zymtrace visibility lane for the existing
`nested_break_depth8` benchmark. It reuses the repository's maintained
`code/core/scripts/profiling/profile.sh --tool zymtrace` path. It does not alter
the benchmark, existing profiler captures, or the Zymtrace backend.

## Support boundary

The official documentation reviewed on 2026-10-26 supports Linux x86_64 and
arm64, kernel 5.10 or newer with BTF, and NVIDIA GPUs with CUDA 12.x or newer.
This preparation is fixed to backend and profiler release `26.10.1` so it can
reuse the retained B200 installation instead of requiring an upgrade. The
changelog says CUDA 13 workloads use CUPTI 13.1 as of release `26.2.6`.

Release `26.10.2` now exists. It reduces host overhead, lowers the launch
sampling target from 200 to 50, and adds separate CUDA Graph sampling with a
target of 20. Those graph controls are not part of this `26.10.1` pin and are
not set here. Moving to `26.10.2` requires a separately owner-approved matching
backend and profiler plus an updated capture plan. This preparation does not
install or update either component.

The intended lane is NVIDIA B200 (`sm_100`), CUDA 13.x, and PyTorch 2.14. CUDA
13 falls within the published CUDA range. The official docs do not publish a
single compatibility row that names B200 and PyTorch 2.14 together. The first
capture must therefore be treated as a compatibility smoke. It is not a
qualified measurement and it must not be used for a latency acceptance claim.
Do not enable the experimental Blackwell hardware event system for this first
capture.

## Required inputs and privileges

Before any capture, an authorized runtime owner must supply and verify all of
these inputs:

- A running Zymtrace `26.10.1` backend and profiler owned by the current runtime
  operator. The backend is not deployed by this runbook.
- The actual gateway address in `host:port` form. Do not substitute a guessed
  endpoint. TLS is the default. Use the disabled TLS mode only when the owner
  confirms that the gateway has no TLS. This value is a host-profiler
  precondition. It is recorded in the plan but is not passed to the workload.
- The Zymtrace project used by the running profiler. The project is an agent
  setting, not a workload setting. It is recorded in the plan but is not passed
  to the workload.
- `ZYMTRACE_AUTH_TOKEN` in the separately owned profiler process when backend
  authentication is enabled. Never place the token in a command, plan, log, or
  repository file.
- A readable `libzymtracecudaprofiler.so` from the same `26.10.1` profiler
  installation.
- A full B200 `GPU-...` UUID and fresh, arm-specific output and cache paths.
- A current runtime-owner checkpoint. Recheck ownership immediately before
  launching each arm. Idle hardware does not establish ownership.

The host profiler uses eBPF and requires root or `CAP_SYS_ADMIN`. The selected
basic CUDA injection path does not require privileged access in the workload.
PC sampling is explicitly disabled. Zymtrace documents that PC sampling needs
`CAP_SYS_ADMIN` or the NVIDIA driver setting `RmProfilingAdminOnly=0`.

## Existing repository path

The repository wrapper already:

- resolves `CUDA_INJECTION64_PATH` or `ZYMTRACE_CUDA_INJECTION64_PATH`
- defaults to `/var/lib/zymtrace/profiler/libzymtracecudaprofiler.so`
- launches the target with both injection variables set
- writes `zymtrace_launch_manifest.json` beside the wrapper output

That wrapper proves only that the injection library was readable and the
workload was launched. It does not start the host profiler, verify the gateway,
prove backend ingestion, or prove that a Zymtrace view exists. A successful
wrapper exit is not a successful capture until backend readback is retained.

## Build the dry-run plan

For execution, `run_capture.py` wraps the plan with one task-owned profiler,
two sequential workers, backend export, and cleanup. It requires a current
finite resource handoff and explicit permission for one expiring capture rule.
The rule is scoped to the actual host and a fresh capture project. Each export
must then match the worker PID, stream, and assigned GPU. PID-based live-config
expressions are not documented, so the controller does not invent one.

The controller reserves time for flushing and cleanup, stops only its own
process groups, and removes only its unchanged rule using the backend's current
configuration version. It writes a COMPLETE or PARTIAL handback in either case.
An archive deployment binds every executed source file through a hashed source
manifest. Run `python run_capture.py --help` for the required runtime pins and
window arguments. A prepared controller is not a completed capture.

Run this only on the authorized B200 host after checking runtime ownership.
Every path must be absolute. The run root must not exist yet.

```bash
python research/2026-10-22_nested-graph-breaks/zymtrace/prepare_capture.py \
  --repo-root /ABSOLUTE/PATH/ai-performance-engineering \
  --run-root /ABSOLUTE/NEW/PATH/nested-graph-zymtrace-26.10.1 \
  --python /ABSOLUTE/PATH/TO/python \
  --injection-library /opt/zymtrace/profiler/libzymtracecudaprofiler.so \
  --runtime-owner-checkpoint /ABSOLUTE/PATH/TO/CURRENT-OWNER-CHECKPOINT \
  --collection-agent zymtrace-gateway.example.internal:443 \
  --zymtrace-project nested-graph-breaks \
  --expected-gpu-uuid GPU-REPLACE-WITH-FULL-UUID \
  --tls-mode tls \
  --timeout-seconds 240 \
  > /ABSOLUTE/PATH/capture-plan.json
```

The builder does not create directories or run commands. It rejects blank or
malformed endpoint strings and three known placeholder forms. This validation
does not prove that a backend exists or is reachable. The builder also rejects
existing run roots, missing owner checkpoints, missing source files, unreadable
injection libraries, and noncanonical GPU UUIDs. The JSON binds the benchmark,
repository wrapper, injection library, plan builder, and owner checkpoint to
SHA-256 hashes. It creates separate commands and cold cache paths for the
disabled and enabled arms.

Each command uses GNU `timeout`, sets `CUDA_VISIBLE_DEVICES` to one full UUID,
uses seed 0, and invokes the repository wrapper. It disables PC sampling and
keeps the `26.10.1` kernel-launch sampling target at 200 launches per OS thread
per one-second window. This `26.10.1` setting limits capture volume without
requesting the extra PC-sampling privilege. No `GRAPH_SAMPLING` variable is set
because that separate control arrived in `26.10.2`. The command omits the
benchmark's independent Chrome trace option. The 240-second per-arm timeout plus
a 30-second kill grace limits the two-arm plan to 540 seconds of controller wall
time.

The benchmark has no reduced-iteration CLI mode. Each selected worker therefore
keeps its fixed 10 cold calls, 25 warmup calls, 100 steady calls, and ten blocks
of 50 throughput calls. This smoke selects one of three cases and one of four
seeds for both arms, which is 2 of the full experiment's 24 cells. Do not feed
these two results into the qualification analyzer, merge them into the retained
qualification results, or call them a repeatability run.

Review the plan before an authorized owner executes either `argv` or
`shell_command`. Execute the arms sequentially. The controller must retain
stdout and stderr separately. A timeout or nonzero exit is a failed capture,
not partial evidence.

## Capture acceptance and provenance

Retain these items for each arm:

1. The reviewed `capture-plan.json`.
2. Controller stdout, stderr, start time, finish time, exit status, and timeout
   status.
3. The benchmark `result.json` and wrapper `zymtrace_launch_manifest.json`.
4. A backend readback showing the expected project, host, process, GPU, arm,
   and capture time range.
5. Zymtrace JSON exports for the relevant flamegraph and events view, when the
   deployed UI exposes the documented export controls.
6. SHA-256 hashes and sizes for every retained local result and export.
7. The exact backend and profiler version read from the deployed system.

The two benchmark arms are visibility captures. Do not compare their observed
durations as a new statistical result. The fixed first post-warmup call remains
descriptive and is not a latency acceptance metric.

## Export, sharing, and MCP

The official changelog documents JSON export for flamegraphs and events on
profile views. It also documents support-dialog export of CPU and GPU events
and profiler diagnostics. The official overview says UI data is available via
REST API and the self-hosted ClickHouse database.

Zymtrace MCP is optional read-only access to the same backend data. The endpoint
is the backend gateway URL plus `/mcp`. Backend version `25.11.0` or newer is
required. If authentication is enabled, use a separately managed MCP token.
MCP tokens are documented as read-only. This runbook does not create a token or
configure a client.

The docs do not promise a public, unauthenticated share link. Treat any UI URL
as backend-local and access-controlled unless the owner proves otherwise.

## Existing capture import decision

The reviewed official Quick Start, GPU profiler, CLI configuration, export,
and MCP documentation does not document importing Nsight Systems reports or
SQLite exports, Nsight Compute reports or CSV exports, or PyTorch Chrome trace
JSON into Zymtrace. Those retained files remain valid evidence in their native
tools, but they are not Zymtrace captures. A fresh Zymtrace-injected workload
is required for Zymtrace visibility unless Zymtrace publishes a supported
import path later.

## Cleanup

The injected library exits with each benchmark process. Do not stop, restart,
or reconfigure a shared Zymtrace profiler or backend as part of workload
cleanup. After provenance is complete, the runtime owner may remove only the
fresh cache directories named in the reviewed plan. Keep the plan, logs,
benchmark results, wrapper manifests, exports, hashes, and readback receipt.

## Official sources

- [Quick Start](https://docs.zymtrace.com/getting-started/)
- [System Requirements](https://docs.zymtrace.com/system-requirements/)
- [GPU Profiler Quick Start](https://docs.zymtrace.com/install/profiler/gpu-profiler-quick-start/)
- [Profiling GPU Workloads](https://docs.zymtrace.com/install/profiler/profiling-gpu-workloads/)
- [Profiler environment and CLI arguments](https://docs.zymtrace.com/install/profiler/profiler-configuration/)
- [GPU profiler environment variables](https://docs.zymtrace.com/install/profiler/profiler-gpu-variables/)
- [Configure MCP](https://docs.zymtrace.com/mcp/configure-mcp/)
- [MCP token](https://docs.zymtrace.com/mcp/mcp-token/)
- [Changelog](https://docs.zymtrace.com/changelog/)
