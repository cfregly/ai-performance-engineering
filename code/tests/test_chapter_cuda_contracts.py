"""Host/source gates only; actual CUDA acceptance is in tests/cuda.

These checks must never be reported as a successful CUDA build or execution.
"""
from pathlib import Path
import json
import importlib.util
import os
import shutil
import subprocess
import sys
import time

import pytest

CODE = Path(__file__).resolve().parents[1]


def source(path):
    return (CODE / path).read_text()


def test_production_tile_schedule_covers_every_tile_once(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    probe = tmp_path / "coverage.cpp"
    probe.write_text(r'''
#include "ch10/grouped_tile_schedule.cuh"
#include <cstdio>
#include <vector>
int main() {
  for (int m : {1, 2, 3, 9}) for (int n : {1, 2, 7, 8, 9, 15, 16, 17}) {
    std::vector<int> visits(m * n);
    for (int i = 0; i < m * n; ++i) {
      const auto tile = ch10::grouped_tile_coord(i, m, n);
      if (tile.m < 0 || tile.m >= m || tile.n < 0 || tile.n >= n) return 1;
      ++visits[tile.m * n + tile.n];
    }
    for (int count : visits) if (count != 1) {
      std::fprintf(stderr, "non-bijective schedule for %d x %d\n", m, n);
      return 2;
    }
  }
}
''')
    binary = tmp_path / "coverage"
    subprocess.run([compiler, "-std=c++17", "-I", str(CODE), str(probe), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    assert "ch10::grouped_tile_coord(" in source("ch10/tcgen05_warp_specialized.cu")


@pytest.mark.parametrize("path", ["ch08/baseline_thresholdtma.cu", "ch12/baseline_cuda_graphs.cu", "ch12/optimized_cuda_graphs.cu"])
def test_single_stream_timer_records_both_events_on_work_stream(path):
    text = source(path)
    assert "cudaEventRecord(start, stream)" in text
    assert "cudaEventRecord(stop, stream)" in text
    assert "cudaEventRecord(start)" not in text
    assert "cudaEventRecord(stop)" not in text


def test_final_async_group_is_drained_before_consumption():
    text = source("ch08/threshold_async_kernel.cuh")
    assert "(tiles_enqueued - tiles_processed) > 1" in text


def test_unaligned_threshold_tail_does_not_claim_aligned_size():
    text = source("ch08/threshold_tma_kernel.cuh")
    assert "copy_bytes % 16 == 0" in text
    assert "cuda::aligned_size_t<16>(copy_bytes)" in text
    assert "stage_ptr(stage), inputs + offset, copy_bytes, pipe" in text


def test_occupancy_query_accounts_for_actual_dynamic_smem():
    text = source("ch06/occupancy_api.cu")
    assert "cudaOccupancyMaxPotentialBlockSizeVariableSMem(" in text
    assert "sample_shared_bytes(block_size)" in text


def test_shared_kernel_has_uniform_barrier_and_four_element_launch():
    text = source("ch11/streams_overlap_demo.cu")
    assert "scale_kernel_async<<<grid_vec_float4" in text
    assert "stream2 = stream1" not in text
    assert "On Blackwell, compiler optimizes this with TMA automatically" not in text
    assert "2 * PIPELINE_BATCHES" in text


def test_stream_timing_joins_all_workers():
    for path in ("ch11/streams_ordered_demo.cu", "ch11/streams_warp_specialized_demo.cu"):
        text = source(path)
        assert "cudaStreamWaitEvent(timing_stream," in text
        assert "cudaEventRecord(stop, timing_stream)" in text
        assert "cudaEventRecord(start, timing_stream)" in text


def test_ilp_reference_checks_all_eight_lanes_and_every_output():
    text = source("ch06/optimized_ilp_low_occupancy_vec4_impl.cuh")
    assert "i < N && i < 1000" not in text
    assert "switch (i % 8)" in text
    assert "case 7: expected_unrolled" in text
    assert "!std::isfinite(h_output_unrolled[i])" in text


def test_real_graph_reference_rejects_last_element_corruption_and_nan(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    probe = tmp_path / "graph_reference.cpp"
    probe.write_text(r'''
#include "ch12/cuda_graphs_workload.cuh"
#include <vector>
int main() {
  constexpr int n = 257;
  std::vector<float> output(n);
  for (int i = 0; i < n; ++i) {
    float x = std::sin(0.001f * i);
    for (int iter = 0; iter < 6; ++iter) for (const auto& stage : kStageSpecs) {
      for (int pass = 0; pass < kInnerPasses; ++pass) {
        x = std::tanh(x * stage.scale + stage.bias);
        x = 0.65f * std::sin(x * stage.frequency + 0.05f * pass)
          + 0.35f * std::cos(x * 0.35f + 0.02f * pass);
      }
    }
    output[i] = x;
  }
  if (!verify_graph_output(output.data(), n, 6)) return 1;
  output.back() += 1.0f;
  if (verify_graph_output(output.data(), n, 6)) return 2;
  output.back() = NAN;
  if (verify_graph_output(output.data(), n, 6)) return 3;
}
''')
    binary = tmp_path / "graph_reference"
    subprocess.run([compiler, "-std=c++17", "-I", str(CODE), str(probe), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("runner", ["run_chapter_cuda_validation.py", "run_ch10_warp_specialized_validation.py"])
def test_gpu_acceptance_fails_closed_without_compiler(tmp_path, runner):
    output = tmp_path / "receipt"
    env = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(CODE / "tests/cuda" / runner),
                             "--arch", "sm_100a", "--output-dir", str(output)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 3
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "HOLD"
    assert report["checks"] == []
    assert "no CUDA" in report["reason"]


def test_validation_timeout_terminates_real_descendant_process(tmp_path):
    path = CODE / "tests/cuda/validation_process.py"
    spec = importlib.util.spec_from_file_location("chapter_validation_process", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ready, survived = tmp_path / "ready", tmp_path / "survived"
    child = ("from pathlib import Path; import time; "
             f"Path({str(ready)!r}).touch(); time.sleep(2); Path({str(survived)!r}).touch()")
    parent = ("import subprocess, sys, time; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(10)")
    result = module.run_command([sys.executable, "-c", parent], timeout=1)
    assert ready.exists(), "the negative control must actually start a descendant"
    assert result.returncode == 124
    assert "process group terminated" in result.stderr
    time.sleep(1.3)
    assert not survived.exists(), "timeout left a descendant alive"
