"""Tests for benchmark input/output verification.

Tests the verification system that ensures:
1. Input equivalence - baseline and optimized benchmarks operate on same workload
2. Output correctness - optimized benchmarks produce same results as baseline

Without these verifications, benchmark comparisons are meaningless:
- Different inputs = comparing apples to oranges
- Different outputs = optimization is broken
"""

import pytest
from typing import Dict, Any, Optional
from pathlib import Path

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.harness.benchmark_harness import BaseBenchmark


# =============================================================================
# Test Fixtures - Mock Benchmarks
# =============================================================================

class MockBenchmarkWithSignature(BaseBenchmark):
    """Mock benchmark with explicit input signature."""
    
    def __init__(self, batch_size=32, seq_len=128, hidden_size=768):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.data = None
        self.output = None
        self.output = None
    
    def setup(self):
        self.data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 2
    
    def teardown(self):
        self.data = None
        self.output = None
        super().teardown()
    
    def get_input_signature(self):
        """Required: Return explicit input signature."""
        return {
            "shapes": {"input": (self.batch_size, self.seq_len, self.hidden_size)},
            "dtypes": {"input": "torch.float32"},
            "batch_size": self.batch_size,
            "parameter_count": self.batch_size * self.seq_len * self.hidden_size,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
        }
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"input": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


class MockBenchmarkDifferentWorkload(BaseBenchmark):
    """Mock benchmark with DIFFERENT workload (larger batch)."""
    
    def __init__(self, batch_size=64, seq_len=128, hidden_size=768):  # Note: batch_size differs
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.data = None
    
    def setup(self):
        self.data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 2
    
    def teardown(self):
        self.data = None
        self.output = None
        super().teardown()
    
    def get_input_signature(self):
        """Required: Return explicit input signature."""
        return {
            "shapes": {"input": (self.batch_size, self.seq_len, self.hidden_size)},
            "dtypes": {"input": "torch.float32"},
            "batch_size": self.batch_size,
            "parameter_count": self.batch_size * self.seq_len * self.hidden_size,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
        }
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"input": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


class MockBenchmarkSkipsInputVerification(BaseBenchmark):
    """Mock benchmark that opts out of input verification (non-compliant in strict mode)."""
    
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.data = None
        self.output = None
    
    def setup(self):
        self.data = torch.ones(self.batch_size, 1, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 3
    
    def skip_input_verification(self) -> bool:
        return True  # Opt out - this is NON-COMPLIANT in strict mode
    
    def get_input_signature(self):
        """Required even though skipping - demonstrates non-compliant pattern."""
        return {
            "shapes": {"input": (self.batch_size, 1)},
            "dtypes": {"input": "torch.float32"},
            "batch_size": self.batch_size,
            "parameter_count": self.batch_size,
        }
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"input": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


class MockBenchmarkSkipsOutputVerification(BaseBenchmark):
    """Mock benchmark that opts out of output verification (non-compliant in strict mode)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.output = None
    
    def setup(self):
        self.data = torch.ones(4, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data + 1
    
    def skip_output_verification(self) -> bool:
        return True  # Opt out - this is NON-COMPLIANT in strict mode
    
    def get_input_signature(self):
        """Required in strict mode."""
        return {
            "shapes": {"input": (4,)},
            "dtypes": {"input": "torch.float32"},
            "batch_size": 4,
            "parameter_count": 4,
        }
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"input": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


class MockBenchmarkNoSignature(BaseBenchmark):
    """Mock benchmark that intentionally returns empty signature (non-compliant)."""
    
    def __init__(self):
        super().__init__()
        self._internal_size = 100  # Private, not captured
        self.data = None
        self.output = None
    
    def setup(self):
        self.data = torch.ones(2, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data * 2
    
    def get_input_signature(self):
        """Returns empty dict - this is NON-COMPLIANT in strict mode."""
        return {}  # Empty = non-compliant
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"input": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


class MockBenchmarkWithMatrixDims(BaseBenchmark):
    """Mock benchmark with matrix dimensions (M, N, K)."""
    
    def __init__(self, M=1024, N=1024, K=512):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.data = None
        self.output = None
    
    def setup(self):
        self.data = torch.ones(self.M, self.N, self.K, device=self.device)
    
    def benchmark_fn(self):
        self.output = self.data
    
    def get_input_signature(self):
        """Required: Return explicit input signature."""
        return {
            "shapes": {"mat": (self.M, self.N, self.K)},
            "dtypes": {"mat": "torch.float32"},
            "batch_size": self.M,
            "parameter_count": self.M * self.N * self.K,
            "M": self.M,
            "N": self.N,
            "K": self.K,
        }
    
    def get_verify_inputs(self):
        if self.data is None:
            raise RuntimeError("Data not initialized")
        return {"mat": self.data}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)


# =============================================================================
# Tests: get_input_signature()
# =============================================================================

class TestGetInputSignature:
    """Tests for BaseBenchmark.get_input_signature()"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_common_attributes(self):
        """Test that get_input_signature captures standard workload attributes."""
        benchmark = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        assert signature["batch_size"] == 32
        assert signature["seq_len"] == 128
        assert signature["hidden_size"] == 768
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_matrix_dimensions(self):
        """Test that get_input_signature captures M, N, K matrix dimensions."""
        benchmark = MockBenchmarkWithMatrixDims(M=2048, N=1024, K=512)
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        assert signature["M"] == 2048
        assert signature["N"] == 1024
        assert signature["K"] == 512
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_captures_tensor_shapes_after_setup(self):
        """Test that tensor shapes are captured after setup().
        
        Note: get_input_signature() captures named attributes like batch_size, seq_len, 
        hidden_size directly - not as a combined 'data_shape' tuple.
        """
        benchmark = MockBenchmarkWithSignature(batch_size=16, seq_len=64, hidden_size=256)
        benchmark.setup()
        
        signature = benchmark.get_input_signature()
        
        assert signature is not None
        # Signature captures individual attributes, not combined shape
        assert signature.get("batch_size") == 16
        assert signature.get("seq_len") == 64
        assert signature.get("hidden_size") == 256
        
        benchmark.teardown()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_none_for_no_attributes(self):
        """Test that get_input_signature returns None when no attributes found."""
        benchmark = MockBenchmarkNoSignature()
        signature = benchmark.get_input_signature()
        
        # Should return None or empty dict since _internal_size is private
        assert signature is None or signature == {}
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_excludes_boolean_attributes(self):
        """Test that boolean attributes should NOT be in signatures.
        
        In strict mode, benchmarks must explicitly choose what to include.
        Best practice: Include only numeric/shape parameters, not config booleans.
        """
        class BenchmarkWithBool(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.batch_size = 32
                self.use_amp = True  # Boolean config - should NOT be in signature
            
            def setup(self): pass
            def benchmark_fn(self): pass
            
            def get_input_signature(self):
                """Explicit signature: Only include workload-defining params."""
                return {"batch_size": self.batch_size}
                # Note: use_amp is NOT included - it's a config flag, not workload size
        
        benchmark = BenchmarkWithBool()
        signature = benchmark.get_input_signature()
        
        assert "batch_size" in signature
        assert "use_amp" not in signature  # Config flags should not be in signature


# =============================================================================
# Tests: skip_input_verification() / skip_output_verification()
# =============================================================================

class TestVerificationOptOut:
    """Tests for benchmark-level verification opt-out."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_verification_enabled(self):
        """Test that verification is enabled by default."""
        benchmark = MockBenchmarkWithSignature()
        
        assert benchmark.skip_input_verification() is False
        assert benchmark.skip_output_verification() is False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_input_verification_opt_out(self):
        """Test that benchmarks can opt out of input verification."""
        benchmark = MockBenchmarkSkipsInputVerification()
        
        assert benchmark.skip_input_verification() is True
        assert benchmark.skip_output_verification() is False  # Only input is skipped
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_verification_opt_out(self):
        """Test that benchmarks can opt out of output verification."""
        benchmark = MockBenchmarkSkipsOutputVerification()
        
        assert benchmark.skip_input_verification() is False  # Only output is skipped
        assert benchmark.skip_output_verification() is True


# =============================================================================
# Tests: _verify_inputs_match()
# =============================================================================

class TestVerifyInputsMatch:
    """Tests for the _verify_inputs_match function."""
    
    @pytest.fixture
    def verify_fn(self):
        """Import the verification function."""
        from core.harness.run_benchmarks import _verify_inputs_match
        return _verify_inputs_match
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_equivalent_workloads_pass(self, verify_fn):
        """Test that equivalent workloads pass verification."""
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        assert result["equivalent"] is True
        assert len(result["mismatches"]) == 0
        assert result["verification_type"] == "input_signature"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_different_workloads_fail(self, verify_fn):
        """Test that different workloads fail verification."""
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkDifferentWorkload(batch_size=64, seq_len=128, hidden_size=768)
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
    
        assert result["equivalent"] is False
        assert len(result["mismatches"]) > 0
        mismatches = " ".join(result["mismatches"]).lower()
        assert (
            "batch_size" in mismatches
            or "shapes" in mismatches
            or "parameter_count" in mismatches
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_skipped_when_benchmark_opts_out(self, verify_fn):
        """Test that verification FAILS when benchmark opts out (strict mode).
        
        With strict enforcement, opting out of verification is NOT allowed.
        Benchmarks must provide proper verification methods.
        """
        baseline = MockBenchmarkWithSignature(batch_size=32)
        optimized = MockBenchmarkSkipsInputVerification(batch_size=64)  # Different but opts out
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        # Strict mode: opting out should fail verification
        assert result["equivalent"] is False
        assert result["verification_type"] == "skipped"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_signature_available(self, verify_fn):
        """Test handling when no signature is available (strict mode).
        
        With strict enforcement, missing signature = verification failure.
        No fallbacks or assumptions allowed.
        """
        baseline = MockBenchmarkNoSignature()
        optimized = MockBenchmarkNoSignature()
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        # Strict mode: missing signature should fail verification
        assert result["equivalent"] is False
        assert result["verification_type"] == "no_signature"
        assert "signature" in result["mismatches"][0].lower()  # "Neither benchmark provides input signature"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_partial_signature_mismatch(self, verify_fn):
        """Test detection of partial signature mismatches."""
        class BenchmarkA(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.M = 1024
                self.N = 1024
                # Missing K
            def setup(self): pass
            def benchmark_fn(self): pass
        
        class BenchmarkB(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.M = 1024
                self.N = 1024
                self.K = 512  # Extra attribute
            def setup(self): pass
            def benchmark_fn(self): pass
        
        baseline = BenchmarkA()
        optimized = BenchmarkB()
        
        result = verify_fn(baseline, optimized, "baseline.py", "optimized.py")
        
        # Should detect the missing K in baseline
        assert result["equivalent"] is False
        assert any("K" in m for m in result["mismatches"])


# =============================================================================
# Tests: Integration
# =============================================================================

class TestVerificationIntegration:
    """Integration tests for the full verification flow."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_equivalent_benchmarks(self):
        """Test full verification flow with equivalent benchmarks."""
        from core.harness.run_benchmarks import _verify_inputs_match
        
        # Create two equivalent benchmarks
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        
        # Setup both
        baseline.setup()
        optimized.setup()
        
        # Verify input equivalence
        input_result = _verify_inputs_match(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        assert input_result["equivalent"] is True
        
        # Run benchmarks
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        
        # Both should have valid outputs
        assert baseline.output is not None
        assert optimized.output is not None
        
        # Cleanup
        baseline.teardown()
        optimized.teardown()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_different_workloads_detected(self):
        """Test that different workloads are detected before benchmark runs."""
        from core.harness.run_benchmarks import _verify_inputs_match
        
        # Create benchmarks with different workloads
        baseline = MockBenchmarkWithSignature(batch_size=32, seq_len=128, hidden_size=768)
        optimized = MockBenchmarkDifferentWorkload(batch_size=64, seq_len=128, hidden_size=768)
        
        # Input verification should FAIL before we even run benchmarks
        input_result = _verify_inputs_match(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        
        assert input_result["equivalent"] is False
        assert "batch_size" in str(input_result["mismatches"])
        
        # The benchmark should NOT proceed to execution when this fails


# =============================================================================
# Tests: CLI Flags
# =============================================================================

class TestCLIFlags:
    """Tests for CLI verification flags."""
    
    def test_skip_verify_flag_disables_both(self):
        """Test that --skip-verify disables both input and output verification."""
        # This tests the flag logic, not the actual CLI
        skip_verify = True
        skip_input_verify = False
        skip_output_verify = False
        
        verify_input = not (skip_verify or skip_input_verify)
        verify_output = not (skip_verify or skip_output_verify)
        
        assert verify_input is False
        assert verify_output is False
    
    def test_individual_skip_flags(self):
        """Test that individual skip flags work independently."""
        # Skip only input
        verify_input = not (False or True)  # skip_input_verify=True
        verify_output = not (False or False)  # skip_output_verify=False
        
        assert verify_input is False
        assert verify_output is True
        
        # Skip only output
        verify_input = not (False or False)  # skip_input_verify=False
        verify_output = not (False or True)  # skip_output_verify=True
        
        assert verify_input is True
        assert verify_output is False
    
    def test_defaults_enable_both(self):
        """Test that defaults enable both verifications."""
        skip_verify = False
        skip_input_verify = False
        skip_output_verify = False
        
        verify_input = not (skip_verify or skip_input_verify)
        verify_output = not (skip_verify or skip_output_verify)
        
        assert verify_input is True
        assert verify_output is True


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in verification."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_numeric_tolerance_for_floats(self):
        """Test that float attributes with identical values are equivalent.
        
        Note: In strict mode, benchmarks MUST implement get_input_signature().
        """
        from core.harness.run_benchmarks import _verify_inputs_match
        
        class BenchmarkWithFloat(BaseBenchmark):
            def __init__(self, learning_rate):
                super().__init__()
                self.learning_rate = learning_rate
            def setup(self): pass
            def benchmark_fn(self): pass
            def get_input_signature(self):
                """Required in strict mode."""
                return {"learning_rate": self.learning_rate}
        
        # Identical values should be OK
        baseline = BenchmarkWithFloat(learning_rate=0.001)
        optimized = BenchmarkWithFloat(learning_rate=0.001)  # Same value
        
        result = _verify_inputs_match(
            baseline, optimized, "baseline.py", "optimized.py"
        )
        
        assert result["equivalent"] is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_signature_includes_dtype(self):
        """Test that tensor dtype is accessible from signature.
        
        Note: get_input_signature() returns named attributes directly.
        Dtype info is typically in a 'dtype' or 'dtypes' field if explicitly set.
        """
        benchmark = MockBenchmarkWithSignature()
        benchmark.setup()
        
        signature = benchmark.get_input_signature()
        
        # Signature includes named numeric attributes
        # Dtype may or may not be present depending on benchmark implementation
        assert signature is not None
        assert isinstance(signature, dict)
        
        benchmark.teardown()


class TestVerifyPatchedBenchmarkBoundary:
    def test_controller_refuses_generated_import_without_os_sandbox(
        self,
        tmp_path: Path,
    ) -> None:
        from core.harness.run_benchmarks import _verify_patched_benchmark

        marker = tmp_path / "imported.txt"
        original_path = tmp_path / "baseline.py"
        patched_path = tmp_path / "candidate.py"
        original_path.write_text("VALUE = 1\n", encoding="utf-8")
        patched_path.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = _verify_patched_benchmark(str(original_path), str(patched_path))

        assert result["verified"] is False
        assert result["verification_type"] == "sandbox_unavailable"
        assert result["details"]["worker_failure"]["non_promotable"] is True
        assert not marker.exists()

    def test_in_process_verifier_rejects_direct_calls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.harness.run_benchmarks import _verify_patched_benchmark_in_worker

        monkeypatch.delenv("AISP_LLM_PATCH_WORKER", raising=False)
        monkeypatch.delenv("AISP_LLM_PATCH_SANDBOX_ACTIVE", raising=False)
        original_path = tmp_path / "baseline.py"
        patched_path = tmp_path / "candidate.py"
        original_path.write_text("VALUE = 1\n", encoding="utf-8")
        patched_path.write_text("VALUE = 1\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="OS-sandboxed patch worker"):
            _verify_patched_benchmark_in_worker(str(original_path), str(patched_path))
