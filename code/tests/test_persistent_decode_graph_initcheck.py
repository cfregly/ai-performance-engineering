"""Real CUDA Graph replay, including its prefill output, under initcheck."""

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
SANITIZER = shutil.which("compute-sanitizer")


@pytest.mark.skipif(
    not torch.cuda.is_available() or SANITIZER is None,
    reason="requires CUDA and Compute Sanitizer",
)
@pytest.mark.parametrize(
    "producer,mode",
    [
        ("graphs", "full"),
        ("graphs", "piecewise"),
        ("tma", "full"),
        ("tma", "piecewise"),
        ("native", "piecewise"),
    ],
)
def test_prefill_graph_replay_has_no_uninitialized_copy(producer, mode, tmp_path):
    if producer != "graphs" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TMA examples require Blackwell")
    # Each case needs its own process because initcheck must observe graph
    # creation as well as the first replay. Poison outputs and change inputs
    # after capture so stale setup values cannot satisfy correctness checks.
    script = r"""
import sys
import torch
torch.manual_seed(731)
producer, mode = sys.argv[1:]
if producer == "graphs":
    from labs.persistent_decode.optimized_persistent_decode_graphs import (
        GraphMode, OptimizedPersistentDecodeGraphsBenchmark,
    )
    bench = OptimizedPersistentDecodeGraphsBenchmark(graph_mode=GraphMode(mode))
elif producer == "tma":
    from labs.persistent_decode.optimized_tma_prefill_decode import (
        GraphMode, OptimizedTmaPrefillDecodeBenchmark,
    )
    bench = OptimizedTmaPrefillDecodeBenchmark(graph_mode=GraphMode(mode))
else:
    from labs.persistent_decode.optimized_native_tma_prefill_decode import (
        OptimizedNativeTmaPrefillDecodeBenchmark,
    )
    bench = OptimizedNativeTmaPrefillDecodeBenchmark()
try:
    bench.setup()
    for scale in (0.75, -0.5):
        bench.inputs.q.mul_(scale)
        if producer == "graphs":
            bench.prefill_out.fill_(float("nan"))
        else:
            bench.prefill_src.mul_(scale)
            bench.prefill_dst.fill_(float("nan"))
        bench.inputs.out.fill_(float("nan"))
        bench.benchmark_fn()
        torch.cuda.synchronize()
        q, k, v = (t.detach().cpu() for t in
                   (bench.inputs.q, bench.inputs.k, bench.inputs.v))
        prefill = (q * k).sum(-1)
        decode = prefill.unsqueeze(-1) * v
        if producer == "graphs":
            torch.testing.assert_close(bench.prefill_out.cpu(), prefill, rtol=1e-5, atol=5e-5)
        else:
            torch.testing.assert_close(bench.prefill_dst, bench.prefill_src, rtol=0, atol=0)
        torch.testing.assert_close(bench.inputs.out.cpu(), decode, rtol=1e-5, atol=5e-5)
    bench.capture_verification_payload()
    print("PREFILL_GRAPH_REPLAY_PASS", flush=True)
finally:
    bench.teardown()
"""
    env = dict(os.environ, PYTHONPATH=str(CODE_ROOT))
    env.pop("PD_GRAPH_MODE", None)
    env.pop("PD_MAX_CAPTURE_SEQ", None)
    process = subprocess.Popen(
        [
            SANITIZER,
            "--tool",
            "initcheck",
            "--error-exitcode",
            "86",
            sys.executable,
            "-c",
            script,
            producer,
            mode,
        ],
        cwd=CODE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=180)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=10)
        (tmp_path / "initcheck.log").write_text(output)
        raise
    (tmp_path / "initcheck.log").write_text(output)
    assert process.returncode == 0, output[-12000:]
    assert output.count("PREFILL_GRAPH_REPLAY_PASS") == 1
    summaries = re.findall(r"^========= ERROR SUMMARY: (\d+) errors", output, re.MULTILINE)
    assert summaries and all(int(count) == 0 for count in summaries), output[-12000:]
