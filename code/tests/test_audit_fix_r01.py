"""Regression checks for audit work unit r01 (W1-106 and W2-002 residuals).

The CLI checks run the real `cli.aisp` entrypoint. The CUDA check reads source
text because nvcc is not available on the host test runner. The B200 build and
run command for that binary is recorded with the audit fix.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli.aisp import app, profile_app

CODE_ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL_SOURCE = CODE_ROOT / "ch12" / "optimized_graph_conditional_runtime.cu"


def _profile_command(name: str):
    return next(cmd for cmd in profile_app.registered_commands if cmd.name == name)


def _option_declarations(name: str) -> set[str]:
    signature = inspect.signature(_profile_command(name).callback)
    return {
        declaration
        for parameter in signature.parameters.values()
        for declaration in (getattr(parameter.default, "param_decls", None) or ())
    }


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cli.aisp", *map(str, args)],
        cwd=CODE_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "NO_COLOR": "1", "COLUMNS": "200"},
    )


def test_profile_memory_does_not_offer_an_output_it_never_writes() -> None:
    assert "--output" not in _option_declarations("memory")
    assert "-o" not in _option_declarations("memory")


def test_profile_memory_rejects_output_flag_without_creating_a_file(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}', encoding="utf-8")
    output = tmp_path / "timeline.json"

    result = _run_cli("profile", "memory", trace, "--output", output)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "No such option" in result.stdout + result.stderr
    assert not output.exists()


def test_profile_memory_reports_not_implemented_and_exits_one(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}', encoding="utf-8")

    result = _run_cli("profile", "memory", trace)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "not implemented" in result.stdout
    assert "No output was written" in result.stdout


def test_unimplemented_profile_commands_say_so_in_help() -> None:
    for name in ("memory", "kernels"):
        help_text = _profile_command(name).help or ""
        assert help_text.startswith("Not implemented."), (name, help_text)
        assert "Exits 1" in help_text, (name, help_text)

    result = CliRunner().invoke(app, ["profile", "memory", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "Not implemented" in result.output
    assert "JSON timeline" not in result.output


def _enclosing_cuda_version_gate(text: str, position: int) -> int:
    """Return the CUDA_VERSION threshold of the innermost open #if at position."""
    stack: list[int | None] = []
    for match in re.finditer(r"^\s*#\s*(if|ifdef|ifndef|else|elif|endif)\b(.*)$", text, re.MULTILINE):
        if match.start() > position:
            break
        directive, rest = match.group(1), match.group(2)
        if directive in ("if", "ifdef", "ifndef"):
            gate = re.search(r"CUDA_VERSION\s*>=\s*(\d+)", rest) if directive == "if" else None
            stack.append(int(gate.group(1)) if gate else None)
        elif directive in ("else", "elif"):
            stack[-1] = None
        elif directive == "endif":
            stack.pop()
    gates = [gate for gate in stack if gate is not None]
    assert gates, "conditional graph code is not guarded by a CUDA_VERSION gate"
    return gates[-1]


def test_if_else_conditional_node_is_gated_on_cuda_12_8() -> None:
    text = CONDITIONAL_SOURCE.read_text(encoding="utf-8")

    else_body = text.index("cond_node_params.conditional.size = 2;")
    assert _enclosing_cuda_version_gate(text, else_body) >= 12080

    # Device code that names the conditional-handle API must sit inside the
    # same gate, so an older toolkit can still build the SKIPPED: binary.
    handle_kernel = text.index("__global__ void set_condition_kernel(")
    assert _enclosing_cuda_version_gate(text, handle_kernel) >= 12080

    assert "12040" not in text
    assert "CUDA 12.4" not in text


def test_conditional_runtime_reports_skipped_instead_of_zero_timing() -> None:
    text = CONDITIONAL_SOURCE.read_text(encoding="utf-8")

    assert "TIME_MS: 0.0" not in text
    assert "-DCUDA_VERSION" not in text
    assert "cudaDriverGetVersion(&driver_version)" in text
    assert re.search(r"if \(driver_version < 12080\) \{\s*printf\(\"SKIPPED:", text)
    assert text.count("return SKIPPED_EXIT_CODE;") == 3
    assert "constexpr int SKIPPED_EXIT_CODE = 3;" in text


# ---------------------------------------------------------------------------
# Follow-ups: the baseline binary's skip contract, the Python wrapper's CUDA
# 12.8 claims, and the cuda_graphs setup capture on the timed buffer.
# ---------------------------------------------------------------------------

BASELINE_CONDITIONAL_SOURCE = CODE_ROOT / "ch12" / "baseline_graph_conditional_runtime.cu"


def test_baseline_conditional_runtime_skip_matches_the_harness_contract() -> None:
    from core.benchmark.cuda_binary_benchmark import _extract_binary_skip_message

    text = BASELINE_CONDITIONAL_SOURCE.read_text(encoding="utf-8")
    assert "TIME_MS: 0.0" not in text
    block = text.split("if (!supports_graphs) {", 1)[1].split("}", 1)[0]
    printed = re.findall(r'std::printf\("((?:[^"\\]|\\.)*)"\);', block)
    exit_code = re.search(r"return (\d+);", block)
    assert printed and exit_code is not None, block

    # Feed what the binary prints and returns to the harness's real skip parser.
    stdout = "".join(line.replace("\\n", "\n") for line in printed)
    completed = subprocess.CompletedProcess(
        args=["baseline_graph_conditional_runtime"],
        returncode=int(exit_code.group(1)),
        stdout=stdout,
        stderr="",
    )
    assert _extract_binary_skip_message(completed) == (
        "SKIPPED: CUDA Graphs require compute capability 7.5+"
    )


def test_conditional_runtime_wrapper_names_the_cuda_12_8_requirement() -> None:
    import ch12.optimized_graph_conditional_runtime as wrapper

    doc = " ".join((wrapper.__doc__ or "").split())
    assert (
        "optimized_graph_conditional_runtime.cu uses an IF conditional node with an ELSE body, "
        "which needs CUDA 12.8+" in doc
    )
    assert "12.4" not in doc
    assert "12.4" not in (wrapper.supports_conditional_graphs.__doc__ or "")


def test_conditional_support_metric_uses_the_cuda_12_8_threshold(monkeypatch) -> None:
    import torch

    import ch12.optimized_graph_conditional_runtime as wrapper

    def metric() -> float:
        return wrapper.OptimizedGraphBenchmark().get_custom_metrics()[
            "cuda_runtime.conditional_support"
        ]

    # Narrow control-plane check of version parsing. No CUDA work runs here.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for version, supported in (
        ("12.4", False),
        ("12.7", False),
        ("12.8", True),
        ("12.9", True),
        ("13.0", True),
        (None, False),
    ):
        monkeypatch.setattr(torch.version, "cuda", version)
        assert wrapper.supports_conditional_graphs() is supported, version
        assert metric() == float(supported), version

    monkeypatch.setattr(torch.version, "cuda", "12.x")
    with pytest.raises(ValueError):
        wrapper.supports_conditional_graphs()


@pytest.mark.parametrize(
    "relative", ("ch12/optimized_cuda_graphs.py", "ch12/optimized_cuda_graphs_router.py")
)
def test_cuda_graphs_setup_captures_on_the_buffer_the_timed_loop_replays(relative: str) -> None:
    source = (CODE_ROOT / relative).read_text(encoding="utf-8")
    setup = source.split("def setup", 1)[1].split("def benchmark_fn", 1)[0]
    compact = "".join(setup.split())

    assert compact.count("self.data=") == 1
    order = [
        "self.data=torch.linspace(",
        "self._verify_input=self.data.detach().clone()",
        "torch.cuda.synchronize(self.device)",
        "self._extension.graph_replay(self.data,self.iterations)",
        "self.data.copy_(self._verify_input)",
        "torch.cuda.synchronize(self.device)",
    ]
    position = -1
    for step in order:
        position = compact.index(step, position + 1)
    assert compact.count("graph_replay(") == 1
    assert "Warmup capture" not in setup
    assert "so capture/instantiation happens before measurement" not in setup


class _RecordingGraphExtension:
    """Stand-in for the compiled extension. It records the buffer each replay uses."""

    def __init__(self, events: list) -> None:
        self.events = events

    def graph_replay(self, data, iterations: int) -> None:
        self.events.append(("graph_replay", data.data_ptr(), iterations))
        data.add_(1.0)


def test_optimized_cuda_graphs_setup_replay_uses_the_timed_buffer(monkeypatch) -> None:
    import torch

    import ch12.optimized_cuda_graphs as module

    events: list = []
    monkeypatch.setattr(module, "load_cuda_graphs_extension", lambda: _RecordingGraphExtension(events))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: events.append(("synchronize",)))
    bench = module.OptimizedCudaGraphsBenchmark()
    bench.device = torch.device("cpu")
    bench.setup()
    try:
        assert [event[0] for event in events] == ["synchronize", "graph_replay", "synchronize"]
        setup_pointer = events[1][1]
        assert setup_pointer == bench.data.data_ptr()
        # The setup replay changed the buffer. setup() must restore the input.
        torch.testing.assert_close(bench.data, bench._verify_input, rtol=0.0, atol=0.0)

        bench.benchmark_fn()
        assert events[-1] == ("graph_replay", setup_pointer, bench.iterations)
    finally:
        bench.teardown()
