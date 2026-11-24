# AI Systems Performance Engineering

This project teaches and measures performance changes in AI workloads. The terms below describe how the repository compares implementations and retains evidence.

## Language

### Benchmarking

A **benchmark pair** contains a baseline and an optimized variant that measure the same workload while changing the mechanism under test.

A **target** selects a chapter, a lab, or a specific benchmark pair for the harness.

A **workload** specifies the inputs, work performed, and output requirements for a benchmark comparison.

A **validity profile** specifies the measurement conditions a run must satisfy and the limitations that accompany its results.

**Verification** checks workload equivalence and agreement of full outputs under the applicable tolerance. Post-timing verification uses outputs captured during timing. Fresh-run verification can also check dependence on live inputs.

An **expectation** is a stored performance reference for a target and hardware profile, used to detect regressions.

**Throughput** is the number of completed workload units per unit time. The benchmark defines the unit, such as tokens, samples, or bytes.

### Run evidence

A **run ID** associates a run's results, logs, and progress records.

A **run manifest** records a run's identity and execution context. A cluster run manifest also inventories its retained files.

**Provenance** is the recorded origin and conditions of a measurement, including its source revision, workload, hardware, timing configuration, and validation outcomes.

**Retained evidence** consists of saved results, captures, logs, and receipts that can be traced to the runs that produced them.

### Retained operator replay

A **capture event** records one operator call, its parameters, input origin, and disposition. A completed event contains full retained input and output values.

A **capture bundle** contains an ordered roster of capture events, their retained tensors, and metadata for validating and identifying the capture.

**Retained operator replay** runs completed capture events independently, in recorded order, using each operation's baseline and optimized implementations. Its timing applies to isolated operator execution.

### Cluster evaluation

The **multi-node readiness preset** checks the declared host, label, interface, and profile configuration before a cluster run. Its verdict describes the planned configuration.

A **canonical run** is the run selected as the reference for a specified cluster field report or comparison.

A **published cluster package** is the retained artifact set selected for publication. Its manifest identifies the run that produced it.
