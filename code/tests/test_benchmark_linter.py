"""Test that the benchmark linter works correctly."""

import subprocess
import sys
from pathlib import Path
import pytest

from core.benchmark.contract import check_benchmark_file_ast
from core.discovery import discover_benchmark_entrypoints


@pytest.mark.parametrize("relative_path", [
    "ch10/warpgroup_specialization_demo.py",
    "ch13/fp8_perchannel_demo.py",
    "labs/decode_optimization/decode_multigpu_demo.py",
    "ch13/dtensor_mesh_tool.py",
    "ch19/fp8_calibration_free_tool.py",
    "ch13/fp8_perchannel_bench.py",
    "ch20/kernel_verification_tool.py",
    "ch15/kv_cache_management_math.py",
    "ch18/nvfp4_trtllm_tool.py",
    "ch20/proofwright_verify_tool.py",
])
def test_registered_standalone_workloads_allow_real_main_entrypoint(relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    valid, errors, warnings = check_benchmark_file_ast(path)
    assert valid, errors
    assert not warnings


def _lint_tmp_file(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.write_text(source)
    return check_benchmark_file_ast(path)


def test_linter_discovers_benchmarks():
    """Test that linter can discover benchmarks without crashing."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "scripts" / "linting" / "check_benchmarks.py"
    
    # Run linter (should not crash even if files have errors)
    result = subprocess.run(
        [sys.executable, str(linter_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), **dict(subprocess.os.environ)},
    )
    
    # Should exit with non-zero if there are errors, but not crash
    # Check that it doesn't raise TypeError about missing arguments
    assert "TypeError" not in result.stderr, f"Linter crashed with TypeError: {result.stderr}"
    assert "discover_benchmarks() missing" not in result.stderr, f"Linter has discovery bug: {result.stderr}"
    
    # Should discover some benchmarks
    assert "Checking" in result.stdout or "benchmark files" in result.stdout.lower(), \
        f"Linter should discover benchmarks: {result.stdout}"


def test_linter_with_specific_path():
    """Test that linter works with specific file paths."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "core" / "scripts" / "linting" / "check_benchmarks.py"
    
    # Find a valid benchmark file
    benchmark_files = list(repo_root.glob("ch*/baseline_*.py"))
    if not benchmark_files:
        pytest.skip("No benchmark files found")
    
    test_file = benchmark_files[0]
    
    result = subprocess.run(
        [sys.executable, str(linter_path), str(test_file)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), **dict(subprocess.os.environ)},
    )
    
    # Should not crash
    assert "TypeError" not in result.stderr, f"Linter crashed: {result.stderr}"


def test_linter_run_setup_flag():
    """Test that --run-setup flag exists and works."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "scripts" / "linting" / "check_benchmarks.py"
    
    # Check help output
    result = subprocess.run(
        [sys.executable, str(linter_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    
    assert "--run-setup" in result.stdout, "Linter should have --run-setup flag"


def test_linter_works_without_cuda():
    """Test that linter works without CUDA (uses AST parsing)."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "core" / "scripts" / "linting" / "check_benchmarks.py"
    template_path = repo_root / "templates" / "benchmark_template.py"
    
    if not template_path.exists():
        pytest.skip("Template file not found")
    
    # Run linter without --run-setup (should use AST, not require CUDA)
    result = subprocess.run(
        [sys.executable, str(linter_path), str(template_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), "CUDA_VISIBLE_DEVICES": "", **dict(subprocess.os.environ)},
    )
    
    # Should not fail with CUDA error
    assert "CUDA required" not in result.stderr, f"Linter should not require CUDA: {result.stderr}"
    assert "RuntimeError" not in result.stderr or "CUDA" not in result.stderr, \
        f"Linter should not raise CUDA RuntimeError: {result.stderr}"
    
    # Should complete successfully (may have warnings but no CUDA errors)
    assert result.returncode == 0 or "CUDA" not in result.stderr, \
        f"Linter should not fail due to CUDA: {result.stderr}"


def test_linter_sync_only_fails_on_hot_path_sync(tmp_path: Path):
    """Test that --sync-only isolates benchmark_fn synchronization warnings."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "core" / "scripts" / "linting" / "check_benchmarks.py"
    bench_path = tmp_path / "baseline_tmp_sync.py"
    bench_path.write_text(
        "class TmpBench:\n"
        "    def setup(self):\n"
        "        pass\n\n"
        "    def benchmark_fn(self):\n"
        "        stream.synchronize()\n\n"
        "    def teardown(self):\n"
        "        pass\n\n"
        "def get_benchmark():\n"
        "    return TmpBench()\n"
    )

    result = subprocess.run(
        [sys.executable, str(linter_path), "--sync-only", "--fail-on-warnings", str(bench_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), **dict(subprocess.os.environ)},
    )

    assert result.returncode != 0
    assert "stream/event synchronize()" in result.stdout


def test_contract_ast_rejects_benchmark_module_main_guard(tmp_path: Path) -> None:
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_tmp_main_guard.py",
        "class TmpBench:\n"
        "    def setup(self):\n"
        "        pass\n\n"
        "    def benchmark_fn(self):\n"
        "        pass\n\n"
        "    def teardown(self):\n"
        "        pass\n\n"
        "def get_benchmark():\n"
        "    return TmpBench()\n\n"
        "if __name__ == \"__main__\":\n"
        "    from core.harness.benchmark_harness import benchmark_main\n"
        "    benchmark_main(get_benchmark)\n",
    )

    assert not ok
    assert any("must not define __main__ blocks" in error for error in errors)
    assert warnings == []


def test_contract_ast_rejects_indirect_benchmark_module_main_guard(tmp_path: Path) -> None:
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_tmp_main_guard_shim.py",
        "class TmpBench:\n"
        "    def setup(self):\n"
        "        pass\n\n"
        "    def benchmark_fn(self):\n"
        "        pass\n\n"
        "    def teardown(self):\n"
        "        pass\n\n"
        "def get_benchmark():\n"
        "    return TmpBench()\n\n"
        "def _maybe_run_cli():\n"
        "    if globals().get(\"__name__\") != \"__main__\":\n"
        "        return\n"
        "    print('bad entrypoint')\n\n"
        "_maybe_run_cli()\n",
    )

    assert not ok
    assert any("must not define __main__ blocks" in error for error in errors)
    assert warnings == []


def test_contract_ast_accepts_payload_mixin_inherited_verification_methods():
    """AST validation should recognize VerificationPayloadMixin via inheritance."""
    repo_root = Path(__file__).parent.parent
    bench_path = repo_root / "ch01" / "baseline_performance.py"

    ok, errors, warnings = check_benchmark_file_ast(bench_path)

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_accepts_imported_shared_benchmark_base():
    """AST validation should follow imported benchmark base classes."""
    repo_root = Path(__file__).parent.parent
    bench_path = repo_root / "labs" / "blackwell_matmul" / "optimized_blackwell_matmul_cluster.py"

    ok, errors, warnings = check_benchmark_file_ast(bench_path)

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_accepts_lab_local_block_scaling_shared_base():
    """AST validation should follow lab-local shared benchmark bases."""
    repo_root = Path(__file__).parent.parent
    bench_path = repo_root / "labs" / "block_scaling" / "optimized_block_scaling.py"

    ok, errors, warnings = check_benchmark_file_ast(bench_path)

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_accepts_lab_local_flashattention_shared_base():
    """AST validation should accept benchmark files backed by lab-local shared bases."""
    repo_root = Path(__file__).parent.parent
    bench_path = repo_root / "labs" / "flashattention4" / "optimized_best_available_attention.py"

    ok, errors, warnings = check_benchmark_file_ast(bench_path)

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_accepts_ch15_moe_shared_base():
    """AST validation should follow the chapter 15 MoE inference shared benchmark base."""
    repo_root = Path(__file__).parent.parent
    bench_path = repo_root / "ch15" / "optimized_moe_inference.py"

    ok, errors, warnings = check_benchmark_file_ast(bench_path)

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_discovery_can_include_unpaired_benchmark_entrypoints():
    """Audit discovery should surface benchmark files outside baseline_/optimized_ naming."""
    repo_root = Path(__file__).parent.parent
    discovered = discover_benchmark_entrypoints(repo_root, include_unpaired=True)
    relative_paths = {path.relative_to(repo_root).as_posix() for path in discovered}

    assert "ch18/scheduling_vllm_sglang.py" in relative_paths


def test_contract_ast_follows_imported_wrapper_benchmark_body():
    """Wrapper modules should inherit warnings/errors from imported shared benchmark classes."""
    repo_root = Path(__file__).parent.parent
    wrapper_path = repo_root / "tests" / "fixtures_contract" / "imported_wrapper_entry.py"

    ok, errors, warnings = check_benchmark_file_ast(wrapper_path)

    assert ok, errors
    assert errors == []
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_contract_ast_follows_imported_helper_function_body() -> None:
    """Imported helper functions called from benchmark_fn() should surface warnings."""
    repo_root = Path(__file__).parent.parent
    wrapper_path = repo_root / "tests" / "fixtures_contract" / "imported_helper_function_entry.py"

    ok, errors, warnings = check_benchmark_file_ast(wrapper_path)

    assert ok, errors
    assert errors == []
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_contract_ast_follows_imported_helper_object_method_body() -> None:
    """Imported helper-object methods called from benchmark_fn() should surface warnings."""
    repo_root = Path(__file__).parent.parent
    wrapper_path = repo_root / "tests" / "fixtures_contract" / "imported_helper_object_entry.py"

    ok, errors, warnings = check_benchmark_file_ast(wrapper_path)

    assert ok, errors
    assert errors == []
    assert any("transfers tensors to CPU via .cpu()" in warning for warning in warnings)


def test_contract_ast_ignores_style_only_noise(tmp_path: Path):
    """Missing docstrings/recommended helpers should not fail the contract gate."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_style_only.py",
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class StyleOnlyBench(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        pass\n"
        "    def capture_verification_payload(self):\n"
        "        self._set_verification_payload(\n"
        "            inputs={'x': self._x}, output=self._x, batch_size=1, parameter_count=0\n"
        "        )\n"
        "    def setup(self):\n"
        "        import torch\n"
        "        self._x = torch.zeros(1)\n"
        "def get_benchmark():\n"
        "    return StyleOnlyBench()\n",
    )

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_flags_missing_benchmark_fn_as_contract_regression(tmp_path: Path):
    """Concrete benchmarks still must implement benchmark_fn()."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_missing_benchmark_fn.py",
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class MissingFnBench(BaseBenchmark):\n"
        "    pass\n\n"
        "def get_benchmark():\n"
        "    return MissingFnBench()\n",
    )

    assert not ok
    assert "Missing required method: benchmark_fn()" in errors
    assert warnings == []


def test_contract_ast_flags_hot_path_random_regeneration_as_warning(tmp_path: Path):
    """Hot-path random input generation should surface as a linter warning."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_random_regeneration.py",
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class RandomBench(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def setup(self):\n"
        "        pass\n"
        "    def benchmark_fn(self):\n"
        "        import torch\n"
        "        self._x = torch.randn(8, 8, device=self.device)\n"
        "    def capture_verification_payload(self):\n"
        "        self._set_verification_payload(\n"
        "            inputs={'x': self._x}, output=self._x, batch_size=1, parameter_count=0\n"
        "        )\n\n"
        "def get_benchmark():\n"
        "    return RandomBench()\n",
    )

    assert ok, errors
    assert errors == []
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_contract_ast_respects_benchmark_antipattern_allowlist(tmp_path: Path):
    """Benchmarks can explicitly allow intentional host-transfer baselines."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_allowed_host_transfer.py",
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class AllowedBench(VerificationPayloadMixin, BaseBenchmark):\n"
        "    allowed_benchmark_fn_antipatterns = ('host_transfer',)\n"
        "    def setup(self):\n"
        "        pass\n"
        "    def benchmark_fn(self):\n"
        "        return value.cpu()\n"
        "    def capture_verification_payload(self):\n"
        "        self._set_verification_payload(\n"
        "            inputs={'x': value}, output=value, batch_size=1, parameter_count=0\n"
        "        )\n\n"
        "def get_benchmark():\n"
        "    return AllowedBench()\n",
    )

    assert ok, errors
    assert errors == []
    assert warnings == []


def test_contract_ast_flags_missing_verification_contract_as_regression(tmp_path: Path):
    """BaseBenchmark-only classes without verification support should fail."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_missing_verification.py",
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class MissingVerifyBench(BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        pass\n\n"
        "def get_benchmark():\n"
        "    return MissingVerifyBench()\n",
    )

    assert not ok
    assert "Missing verification method: get_input_signature()" in errors
    assert "Missing verification method: get_verify_output()" in errors
    assert "Missing verification method: get_output_tolerance()" in errors
    assert warnings == []


def test_contract_ast_flags_bad_factory_signature_as_regression(tmp_path: Path):
    """Parameterized get_benchmark() factories are contract regressions."""
    ok, errors, warnings = _lint_tmp_file(
        tmp_path,
        "baseline_bad_factory.py",
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n\n"
        "class FactoryBench(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def setup(self):\n"
        "        import torch\n"
        "        self._x = torch.zeros(1)\n"
        "    def benchmark_fn(self):\n"
        "        pass\n"
        "    def capture_verification_payload(self):\n"
        "        self._set_verification_payload(\n"
        "            inputs={'x': self._x}, output=self._x, batch_size=1, parameter_count=0\n"
        "        )\n\n"
        "def get_benchmark(size=1):\n"
        "    return FactoryBench()\n",
    )

    assert not ok
    assert "get_benchmark() should take no arguments" in errors
    assert warnings == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
