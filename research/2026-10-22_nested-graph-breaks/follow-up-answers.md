# Follow-up answers

## What exactly improved?

The measured improvement is steady synchronized invocation latency for the
depth-8 compiler microbenchmark. The original four-seed cohort measured a
4.531795x geometric-mean ratio. Independent fresh run 022 measured 4.541758x.
The two cohorts remain separate.

## Was the work the same?

Yes within the declared microbenchmark. Both arms ran the same worker and the
same 64 FP32 operations. The flag changed how TorchDynamo and Inductor grouped
work across nested graph breaks. The eager path was a correctness reference,
not a performance competitor.

## What happened at startup?

Startup did not qualify. In the original cohort, the candidate's first ten
compiled-call intervals were 33.2 percent slower, and one no-break startup
control missed its fixed range. Calls 2 through 10 were faster with nested
graph-break support on, but the slower first compiled call dominated the fixed total. Fresh run 022 also
failed the startup measurement rule, although its startup control passed.

## Did the result repeat?

The original depth-8 per-seed steady ratios ranged from 4.495352x to 4.579454x.
Fresh run 022 ranged from 4.438828x to 4.653092x and passed its steady control,
correctness, and capture gates. Four seeds in each cohort remain a limited
descriptive sample, not a population-coverage claim.

## What do the profiles show?

The fixed Nsys view shows 16 graph executions with nested graph-break support
off and 2 with it on for the selected first post-warmup invocation. Across each whole Nsys capture, the
counts were 10,160 and 1,270 graph-launch calls. NCU retained one matching
fused Triton kernel per arm. The saved durations were 6.400 microseconds with
support off and 20.992 microseconds with it on. The longer kernel from the
support-on arm does more work per launch.

Normal unprofiled workers establish the latency result. Nsys tracing and NCU
counter replay add overhead, so timeline widths and single-kernel durations do
not measure the 4.53x or 4.54x whole-call gains.

## Is this specific to B200?

The evidence is B200-specific. H100, A100, and other architectures were not
tested. The result also does not establish portability across PyTorch versions
or compiler backends.

## What did the separate memory diagnostic find?

One later seed-0 B200 pair measured 8 MiB of PyTorch allocated memory in both
arms after 4,000 calls. Reserved memory was 296 MiB with nested graph-break
support off and 168 MiB with it on. First-call peak allocated memory was 170.5 MiB and 146.5 MiB. Final host
RSS was 1410.609375 MiB and 1416.65234375 MiB. These are one pair's allocator
counters and process RSS, not whole-device memory or a general equivalence
claim.

## Does this establish a training or serving win?

No. The workload is synthetic. It does not include a representative model,
backward pass, optimizer, batching system, scheduler, or serving stack. The
result supports a compiler-mechanism claim only.

## Was Zymtrace used?

No Zymtrace capture or import was retained for this experiment. The package
contains preparation code and methods notes only. The retained profiler files
are Nsys, NCU, SQLite, CSV, and PyTorch Chrome artifacts.
