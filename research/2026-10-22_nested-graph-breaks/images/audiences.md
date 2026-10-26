# Audience variants

The six variants use the same historical cohort but change the explanation,
composition, and information density for the reader. None invents serving,
cost, capacity, or revenue results.

## Curious non-specialists

**Stem:** `aud-curious`

The picture uses a trip metaphor. Sixteen small trips and two larger trips map
to replay launches. The exact steady latencies keep the metaphor tied to the
measurement.

**Alt text:** Nested graph-break support is off on the left and on on the right.
The left side shows 16 small blue trips and 407.567 microseconds. The right side
shows 2 larger orange trips and 89.935 microseconds. A note says the support-on
path is 4.531795 times faster after warmup and 33.2 percent longer over the
first ten compiled-call intervals.

## ML/AI engineers

**Stem:** `aud-ml-engineer`

The picture maps the setting to compiled shape. It shows the region and replay
counts for each arm, then states that the workload code and 64 captured ATen
operations stayed fixed.

**Alt text:** Two compiler-shape cards compare nested graph-break support off
and on. Support off has 16 emitted regions and 16 replay launches. Support on
has 2 emitted regions and 2 replay launches. Both cards show 64 captured ATen
operations. The footer identifies the same explicit graph-break workload,
depth 8, and PyTorch 2.14.1+cu130.

## Performance and capacity engineers

**Stem:** `aud-performance-engineer`

The picture uses a four-point seed plot rather than a two-bar summary. It shows
the observed seed range, the descriptive geometric mean, and the split verdict
between steady and startup behavior.

**Alt text:** Four orange points show paired-seed steady speedups of 4.522609,
4.579454, 4.530169, and 4.495352 times. The descriptive geometric mean is
4.531795 times. The steady route passed and the startup route failed. The ratio
is support-off latency divided by support-on latency after 25 warmups and over
100 synchronized calls per worker.

## CUDA and hardware engineers

**Stem:** `aud-kernel-engineer`

The picture focuses on launch granularity. It uses one invocation in each panel
and pairs replay count with operations per emitted region. This keeps launch
evidence separate from latency.

**Alt text:** Support off shows 16 blue graph-launch blocks and 4 ATen
operations per emitted region. Support on shows 2 orange graph-launch blocks
and 32 ATen operations per emitted region. A note says both arms captured the
same 64 ATen operations. The counts come from five untimed historical PyTorch
profile invocations and are not latency measurements.

## Platform operations

**Stem:** `aud-platform-operations`

The picture is an operational sequence. It separates 25 warmups, 100
synchronized steady calls, and the passing steady result from the failing
first-ten startup result.

**Alt text:** Three boxes show 25 warmups, 100 synchronized calls, and a
4.531795-times steady pass with nested graph-break support on. A separate orange
warning says the support-on first ten intervals were 33.2 percent longer and
the startup route failed. The footer identifies one B200 and the historical
standalone protocol.

## Business and capacity readers

**Stem:** `aud-business-capacity`

The picture is a measured-versus-not-established decision card. It carries the
useful steady result and startup cost while blocking extrapolation to serving,
cost, or capacity.

**Alt text:** The measured column says nested graph-break support on was
4.531795 times faster in steady execution, 33.2 percent longer over the first
ten intervals, and used the same synthetic depth-8 work. The not-established
column lists model throughput, serving latency, and cost or capacity. A note
says to use the result for compiler diagnosis, not a capacity forecast.

