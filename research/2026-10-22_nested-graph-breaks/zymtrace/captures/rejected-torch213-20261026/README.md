# Rejected runtime capture

This is a real B200 Zymtrace capture from one worker without nested graph-break
support. The worker ran with PyTorch 2.13.0+cu130 because the launcher resolved
the virtualenv interpreter symlink to the system interpreter. The controller
rejected it before starting the other worker. It is not evidence for the
intended PyTorch 2.14.1 pair or a new performance result.

The retained export contains 348 CUDA records matching worker PID 545816,
the assigned GPU UUID, and stream IDs. The raw events and flamegraph are
preserved with hashes. These API records do not establish complete per-launch
coverage or an exact CPU/GPU causal split.

The profiler exited cleanly. The owned worker process group was empty, the
single temporary capture rule was removed at config version 10, and immediate
readback showed no GPU compute process. The original PARTIAL handback is kept.

The launcher now preserves the virtualenv invocation path and checks the exact
Torch and CUDA versions before any GPU operation. The corrected 2.14.1 pair is
waiting for the PIQ comparison to finish and a fresh B200 window.
