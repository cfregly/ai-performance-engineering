"""Tests for benchmark verification enforcement system.

This module tests the verification data models, quarantine management,
and verification runner functionality.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
import torch
import numpy as np

from core.benchmark.verification import (
    ComparisonDetails,
    EnforcementPhase,
    InputSignature,
    PrecisionFlags,
    QuarantineReason,
    QuarantineRecord,
    ToleranceSpec,
    VerifyResult,
    DEFAULT_TOLERANCES,
    compare_workload_metrics,
    detect_seed_mutation,
    get_enforcement_phase,
    is_verification_enabled,
    get_tolerance_for_dtype,
    is_tolerance_looser,
    select_jitter_dimension,
    set_deterministic_seeds,
)
from core.benchmark.quarantine import (
    QuarantineManager,
    check_benchmark_compliance,
    detect_skip_flags,
    SKIP_FLAGS,
)
from core.benchmark.verify_runner import (
    GoldenOutput,
    GoldenOutputCache,
    TimingConfig,
    VerifyConfig,
    VerifyRunner,
)


@contextmanager
def _temp_environ(updates: Optional[Dict[str, str]] = None, *, clear: bool = False):
    """Temporarily mutate os.environ without mocks/monkeypatch."""
    old = dict(os.environ)
    try:
        if clear:
            os.environ.clear()
        if updates:
            os.environ.update({k: str(v) for k, v in updates.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


# =============================================================================
# PrecisionFlags Tests
# =============================================================================


class TestPrecisionFlags:
    """Tests for PrecisionFlags dataclass."""
    
    def test_default_values(self):
        """Test default precision flag values."""
        flags = PrecisionFlags()
        assert flags.fp16 is False
        assert flags.bf16 is False
        assert flags.fp8 is False
        assert flags.tf32 is True  # Default on for CUDA
    
    def test_to_dict(self):
        """Test serialization to dict."""
        flags = PrecisionFlags(fp16=True, bf16=True)
        d = flags.to_dict()
        assert d == {"fp16": True, "bf16": True, "fp8": False, "tf32": True}
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"fp16": True, "bf16": False, "fp8": True, "tf32": False}
        flags = PrecisionFlags.from_dict(d)
        assert flags.fp16 is True
        assert flags.bf16 is False
        assert flags.fp8 is True
        assert flags.tf32 is False
    
    def test_from_dict_with_missing_keys(self):
        """Test deserialization handles missing keys."""
        d = {"fp16": True}
        flags = PrecisionFlags.from_dict(d)
        assert flags.fp16 is True
        assert flags.bf16 is False
        assert flags.fp8 is False
        assert flags.tf32 is True  # Default


# =============================================================================
# InputSignature Tests
# =============================================================================


class TestInputSignature:
    """Tests for InputSignature dataclass."""
    
    def test_create_basic_signature(self):
        """Test creating a basic input signature."""
        sig = InputSignature(
            shapes={"input": (32, 256, 256)},
            dtypes={"input": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(fp16=True),
        )
        assert sig.shapes["input"] == (32, 256, 256)
        assert sig.dtypes["input"] == "float16"
        assert sig.batch_size == 32
    
    def test_hash_is_deterministic(self):
        """Test that hash is deterministic."""
        sig1 = InputSignature(
            shapes={"input": (32, 256, 256)},
            dtypes={"input": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        sig2 = InputSignature(
            shapes={"input": (32, 256, 256)},
            dtypes={"input": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        assert sig1.hash() == sig2.hash()
    
    def test_hash_changes_with_different_values(self):
        """Test that hash changes when values differ."""
        sig1 = InputSignature(
            shapes={"input": (32, 256, 256)},
            dtypes={"input": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        sig2 = InputSignature(
            shapes={"input": (64, 256, 256)},  # Different batch
            dtypes={"input": "float16"},
            batch_size=64,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        assert sig1.hash() != sig2.hash()
    
    def test_matches(self):
        """Test signature matching."""
        sig1 = InputSignature(
            shapes={"input": (32, 256)},
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        sig2 = InputSignature(
            shapes={"input": (32, 256)},
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        assert sig1.matches(sig2)
    
    def test_validate_missing_shapes(self):
        """Test validation catches missing shapes."""
        sig = InputSignature(
            shapes={},  # Empty!
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        # In strict mode, empty shapes should fail
        errors = sig.validate(strict=True)
        assert any("shapes" in e.lower() for e in errors)
        # In non-strict mode (default), empty shapes is OK
        errors_non_strict = sig.validate(strict=False)
        assert not any("shapes" in e.lower() for e in errors_non_strict)
    
    def test_validate_invalid_batch_size(self):
        """Test validation catches invalid batch size."""
        sig = InputSignature(
            shapes={"input": (32, 256)},
            dtypes={"input": "float32"},
            batch_size=-1,  # Invalid (negative)!
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        errors = sig.validate()
        assert any("batch_size" in e for e in errors)

    def test_pipeline_stage_boundaries_required(self):
        sig = InputSignature(
            shapes={"input": (32, 256)},
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
            pipeline_stages=2,
        )
        errors = sig.validate(strict=True)
        assert any("pipeline_stage_boundaries" in e for e in errors)

    def test_pipeline_stage_boundaries_contiguous(self):
        sig = InputSignature(
            shapes={"input": (32, 256)},
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
            pipeline_stages=2,
            pipeline_stage_boundaries=[(0, 0), (2, 2)],
        )
        errors = sig.validate(strict=True)
        assert any("contiguous" in e for e in errors)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        sig = InputSignature(
            shapes={"input": (32, 256), "mask": (32,)},
            dtypes={"input": "float16", "mask": "bool"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(fp16=True),
            world_size=4,
            num_streams=2,
            pipeline_stages=2,
            pipeline_stage_boundaries=[(0, 1), (2, 3)],
        )
        d = sig.to_dict()
        sig2 = InputSignature.from_dict(d)
        assert sig.matches(sig2)


# =============================================================================
# ToleranceSpec Tests
# =============================================================================


class TestToleranceSpec:
    """Tests for ToleranceSpec dataclass."""
    
    def test_default_tolerances_exist(self):
        """Test that default tolerances exist for common dtypes."""
        assert torch.float32 in DEFAULT_TOLERANCES
        assert torch.float16 in DEFAULT_TOLERANCES
        assert torch.bfloat16 in DEFAULT_TOLERANCES
        assert torch.int32 in DEFAULT_TOLERANCES
    
    def test_get_tolerance_for_dtype(self):
        """Test getting tolerance for specific dtype."""
        tol = get_tolerance_for_dtype(torch.float16)
        assert tol.rtol == 1e-3
        assert tol.atol == 1e-5
    
    def test_get_tolerance_for_unknown_dtype(self):
        """Test fallback tolerance for unknown dtype."""
        # Create a mock dtype that doesn't exist in defaults
        tol = get_tolerance_for_dtype(torch.complex64)  # Known but testing fallback logic
        assert tol is not None
        assert tol.rtol >= 0
    
    def test_is_tolerance_looser(self):
        """Test looser tolerance detection."""
        default = ToleranceSpec(rtol=1e-5, atol=1e-8)
        looser = ToleranceSpec(rtol=1e-3, atol=1e-5)  # Looser
        tighter = ToleranceSpec(rtol=1e-7, atol=1e-10)  # Tighter
        
        assert is_tolerance_looser(looser, default) is True
        assert is_tolerance_looser(tighter, default) is False


# =============================================================================
# QuarantineReason Tests
# =============================================================================


class TestQuarantineReason:
    """Tests for QuarantineReason enum."""
    
    def test_all_reasons_have_values(self):
        """Test that all enum members have string values."""
        for reason in QuarantineReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0
    
    def test_reason_values_are_unique(self):
        """Test that all reason values are unique."""
        values = [r.value for r in QuarantineReason]
        assert len(values) == len(set(values))


# =============================================================================
# EnforcementPhase Tests  
# =============================================================================


class TestEnforcementPhase:
    """Tests for EnforcementPhase enum and get_enforcement_phase."""
    
    def test_default_phase_is_detect(self):
        """Test default enforcement phase is DETECT."""
        with _temp_environ(clear=True):
            phase = get_enforcement_phase()
            assert phase == EnforcementPhase.DETECT
    
    def test_get_phase_from_env(self):
        """Test reading phase from environment."""
        with _temp_environ({"VERIFY_ENFORCEMENT_PHASE": "gate"}):
            phase = get_enforcement_phase()
            assert phase == EnforcementPhase.GATE
    
    def test_quarantine_phase(self):
        """Test QUARANTINE phase from environment."""
        with _temp_environ({"VERIFY_ENFORCEMENT_PHASE": "quarantine"}):
            phase = get_enforcement_phase()
            assert phase == EnforcementPhase.QUARANTINE
    
    def test_invalid_phase_defaults_to_detect(self):
        """Test invalid phase value defaults to DETECT."""
        with _temp_environ({"VERIFY_ENFORCEMENT_PHASE": "invalid_value"}):
            phase = get_enforcement_phase()
            assert phase == EnforcementPhase.DETECT


class TestVerificationEnabled:
    """Tests for is_verification_enabled function."""
    
    def test_verification_enabled_by_default(self):
        """Test verification is enabled by default."""
        with _temp_environ(clear=True):
            assert is_verification_enabled() is True
    
    def test_verification_can_be_disabled(self):
        """Test verification can be disabled via env var."""
        with _temp_environ({"VERIFY_DISABLED": "1"}):
            assert is_verification_enabled() is False
    
    def test_verification_enabled_with_zero(self):
        """Test VERIFY_DISABLED=0 keeps verification enabled."""
        with _temp_environ({"VERIFY_DISABLED": "0"}):
            assert is_verification_enabled() is True


# =============================================================================
# Seed Management Tests
# =============================================================================


class TestSeedManagement:
    """Tests for seed management functions."""
    
    def test_set_deterministic_seeds(self):
        """Test setting deterministic seeds."""
        seed_info = set_deterministic_seeds(42)
        
        assert seed_info["random_seed"] == 42
        assert seed_info["numpy_seed"] == 42
        assert seed_info["torch_seed"] == 42
        assert seed_info["cudnn_deterministic"] is True
        assert seed_info["cudnn_benchmark"] is False
    
    def test_seeds_produce_reproducible_results(self):
        """Test that seeds produce reproducible random values."""
        set_deterministic_seeds(42)
        val1 = torch.rand(10)
        
        set_deterministic_seeds(42)
        val2 = torch.rand(10)
        
        assert torch.allclose(val1, val2)


# =============================================================================
# Jitter Check Tests
# =============================================================================


class TestJitterCheck:
    """Tests for jitter dimension selection."""
    
    def test_select_jitter_dimension_with_multidim(self):
        """Test selecting jitter dimension with multi-dimensional tensor."""
        sig = InputSignature(
            shapes={"input": (32, 256, 256)},
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        dim = select_jitter_dimension(sig)
        assert dim is not None
        assert dim == ("input", 1)  # First non-batch dimension
    
    def test_select_jitter_dimension_batch_only(self):
        """Test jitter selection with batch-only tensor returns None."""
        sig = InputSignature(
            shapes={"input": (32,)},  # 1D tensor
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        dim = select_jitter_dimension(sig)
        assert dim is None  # No suitable dimension


# =============================================================================
# Workload Comparison Tests
# =============================================================================


class TestWorkloadComparison:
    """Tests for workload metrics comparison."""
    
    def test_compare_matching_metrics(self):
        """Test comparing identical workload metrics."""
        baseline = {"bytes_per_iter": 1000.0, "flops_per_iter": 2000.0}
        optimized = {"bytes_per_iter": 1000.0, "flops_per_iter": 2000.0}
        
        passed, deltas = compare_workload_metrics(baseline, optimized)
        assert passed is True
        assert all(d == 0 for d in deltas.values())
    
    def test_compare_within_tolerance(self):
        """Test comparing metrics within tolerance."""
        baseline = {"bytes_per_iter": 1000.0}
        optimized = {"bytes_per_iter": 1005.0}  # 0.5% difference
        
        passed, deltas = compare_workload_metrics(baseline, optimized, tolerance=0.01)
        assert passed is True
    
    def test_compare_outside_tolerance(self):
        """Test comparing metrics outside tolerance."""
        baseline = {"bytes_per_iter": 1000.0}
        optimized = {"bytes_per_iter": 1100.0}  # 10% difference
        
        passed, deltas = compare_workload_metrics(baseline, optimized, tolerance=0.01)
        assert passed is False


# =============================================================================
# QuarantineManager Tests
# =============================================================================


class TestQuarantineManager:
    """Tests for QuarantineManager."""
    
    @pytest.fixture
    def temp_quarantine_path(self, tmp_path):
        """Fixture providing a temporary quarantine file path."""
        return tmp_path / "quarantine.json"
    
    def test_create_manager(self, temp_quarantine_path):
        """Test creating a quarantine manager."""
        manager = QuarantineManager(temp_quarantine_path)
        assert len(manager.get_all_quarantined()) == 0
    
    def test_quarantine_benchmark(self, temp_quarantine_path):
        """Test quarantining a benchmark."""
        manager = QuarantineManager(temp_quarantine_path)
        
        record = manager.quarantine(
            "ch01/baseline_gemm.py",
            QuarantineReason.MISSING_INPUT_SIGNATURE,
        )
        
        assert manager.is_quarantined("ch01/baseline_gemm.py")
        assert record.quarantine_reason == QuarantineReason.MISSING_INPUT_SIGNATURE
    
    def test_clear_quarantine(self, temp_quarantine_path):
        """Test clearing quarantine."""
        manager = QuarantineManager(temp_quarantine_path)
        manager.quarantine("ch01/baseline_gemm.py", QuarantineReason.SKIP_FLAG_PRESENT)
        
        assert manager.is_quarantined("ch01/baseline_gemm.py")
        
        cleared = manager.clear_quarantine("ch01/baseline_gemm.py")
        assert cleared is True
        assert not manager.is_quarantined("ch01/baseline_gemm.py")
    
    def test_persistence(self, temp_quarantine_path):
        """Test quarantine persistence across manager instances."""
        manager1 = QuarantineManager(temp_quarantine_path)
        manager1.quarantine("ch01/baseline_gemm.py", QuarantineReason.OUTPUT_MISMATCH)
        
        # Create new manager instance
        manager2 = QuarantineManager(temp_quarantine_path)
        assert manager2.is_quarantined("ch01/baseline_gemm.py")
    
    def test_get_quarantine_summary(self, temp_quarantine_path):
        """Test getting quarantine summary."""
        manager = QuarantineManager(temp_quarantine_path)
        manager.quarantine("a.py", QuarantineReason.MISSING_INPUT_SIGNATURE)
        manager.quarantine("b.py", QuarantineReason.MISSING_INPUT_SIGNATURE)
        manager.quarantine("c.py", QuarantineReason.OUTPUT_MISMATCH)
        
        summary = manager.get_quarantine_summary()
        assert summary["missing_input_signature"] == 2
        assert summary["output_mismatch"] == 1


# =============================================================================
# Skip Flag Detection Tests
# =============================================================================


class TestSkipFlagDetection:
    """Tests for skip flag detection."""
    
    def test_detect_skip_output_check_attribute(self):
        """Test detecting skip_output_check attribute."""
        class Benchmark:
            skip_output_check = True
            skip_input_check = False
            skip_verification = False

            def skip_input_verification(self) -> bool:
                return False

            def skip_output_verification(self) -> bool:
                return False
        
        benchmark = Benchmark()
        result = detect_skip_flags(benchmark)
        assert result == QuarantineReason.SKIP_FLAG_PRESENT
    
    def test_detect_skip_input_verification_method(self):
        """Test detecting skip_input_verification method."""
        class Benchmark:
            skip_output_check = False
            skip_input_check = False
            skip_verification = False

            def skip_input_verification(self) -> bool:
                return True

            def skip_output_verification(self) -> bool:
                return False
        
        benchmark = Benchmark()
        result = detect_skip_flags(benchmark)
        assert result == QuarantineReason.SKIP_FLAG_PRESENT
    
    def test_no_skip_flags(self):
        """Test benchmark without skip flags."""
        class Benchmark:
            skip_output_check = False
            skip_input_check = False
            skip_verification = False

            def skip_input_verification(self) -> bool:
                return False

            def skip_output_verification(self) -> bool:
                return False
        
        benchmark = Benchmark()
        result = detect_skip_flags(benchmark)
        assert result is None


# =============================================================================
# GoldenOutputCache Tests
# =============================================================================


class TestGoldenOutputCache:
    """Tests for GoldenOutputCache."""

    def _signature(self, batch_size: int = 1) -> Dict[str, Any]:
        return {
            "shapes": {"input": [batch_size, 1]},
            "dtypes": {"input": "float32"},
            "batch_size": batch_size,
            "parameter_count": 0,
            "precision_flags": {"fp16": False, "bf16": False, "fp8": False, "tf32": True},
        }
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Fixture providing a temporary cache directory."""
        return tmp_path / "golden_cache"
    
    def test_cache_put_and_get(self, temp_cache_dir):
        """Test storing and retrieving golden output."""
        cache = GoldenOutputCache(temp_cache_dir)
        
        golden = GoldenOutput(
            signature_hash="abc123",
            signature=self._signature(batch_size=1),
            outputs={"output": torch.tensor([1.0, 2.0, 3.0])},
            workload_metrics={"bytes_per_iter": 1000.0},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden.checksum = golden.compute_checksum()
        
        cache.put(golden)
        
        retrieved = cache.get("abc123")
        assert retrieved is not None
        assert torch.allclose(retrieved.outputs["output"], golden.outputs["output"])

    def test_cache_put_and_get_bfloat16(self, temp_cache_dir):
        """Test bf16-safe golden output caching."""
        cache = GoldenOutputCache(temp_cache_dir)

        output = torch.randn(8, dtype=torch.bfloat16)
        golden = GoldenOutput(
            signature_hash="bf16",
            signature=self._signature(batch_size=1),
            outputs={"output": output},
            workload_metrics={"tokens_per_iteration": 8.0},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden.checksum = golden.compute_checksum()

        cache.put(golden)
        retrieved = cache.get("bf16")
        assert retrieved is not None
        assert retrieved.outputs["output"].dtype == torch.bfloat16
        assert torch.equal(retrieved.outputs["output"], output)
    
    def test_cache_has(self, temp_cache_dir):
        """Test checking cache existence."""
        cache = GoldenOutputCache(temp_cache_dir)
        
        assert not cache.has("nonexistent")
        
        golden = GoldenOutput(
            signature_hash="exists",
            signature=self._signature(batch_size=1),
            outputs={"out": torch.tensor([1.0])},
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        cache.put(golden)
        
        assert cache.has("exists")
    
    def test_cache_invalidate(self, temp_cache_dir):
        """Test invalidating cache entry."""
        cache = GoldenOutputCache(temp_cache_dir)
        
        golden = GoldenOutput(
            signature_hash="to_delete",
            signature=self._signature(batch_size=1),
            outputs={"out": torch.tensor([1.0])},
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        cache.put(golden)
        
        assert cache.has("to_delete")
        
        deleted = cache.invalidate("to_delete")
        assert deleted is True
        assert not cache.has("to_delete")


# =============================================================================
# VerifyRunner Tests
# =============================================================================


class MockWorkloadMetadata:
    """Mock workload metadata for testing."""
    
    def __init__(self):
        self.requests_per_iteration = 1.0
        self.tokens_per_iteration = None
        self.samples_per_iteration = None
        self.bytes_per_iteration = 1024
        self.custom_units_per_iteration = None
        self.custom_unit_name = None
        self.goodput = None


class MockBenchmark:
    """Mock benchmark for testing verification."""
    
    def __init__(
        self,
        name: str = "mock",
        output: Optional[torch.Tensor] = None,
        signature: Optional[Dict[str, Any]] = None,
        include_workload: bool = True,
    ):
        self.name = name
        self.output = output if output is not None else torch.randn(10)
        self._signature = signature or {
            "shapes": {"input": (32, 256)},
            "dtypes": {"input": "float32"},
            "batch_size": 32,
            "parameter_count": 100,
        }
        self._workload_metadata = MockWorkloadMetadata() if include_workload else None
    
    def setup(self):
        """Setup method."""
        pass
    
    def benchmark_fn(self):
        """Benchmark function."""
        # Generate deterministic output based on current seed
        self.output = torch.randn(10)
    
    def teardown(self):
        """Teardown method."""
        pass
    
    def get_input_signature(self) -> Dict[str, Any]:
        """Return input signature."""
        return self._signature
    
    def validate_result(self) -> Optional[str]:
        """Validate result."""
        return None
    
    def get_workload_metadata(self) -> Optional[Any]:
        """Return workload metadata."""
        return self._workload_metadata
    
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification.
        
        MANDATORY: Explicitly implemented as required by strict mode.
        """
        return self.output

    def get_output_tolerance(self) -> tuple:
        """Return numeric tolerance for comparisons."""
        return (1e-3, 1e-3)

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        """Return inputs for aliasing and signature checks."""
        return {"input": torch.zeros(self._signature["shapes"]["input"])}


class TestVerifyRunner:
    """Tests for VerifyRunner."""
    
    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Fixture providing temporary directories."""
        return {
            "cache": tmp_path / "cache",
            "quarantine": tmp_path / "quarantine.json",
        }
    
    def test_verify_baseline_success(self, temp_dirs):
        """Test successful baseline verification."""
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        
        baseline = MockBenchmark("baseline")
        result = runner.verify_baseline(baseline)
        
        assert result.passed is True
        assert result.signature_hash is not None

    def test_default_world_size_does_not_reject_extra_visible_capacity(
        self,
        temp_dirs,
        monkeypatch,
    ):
        """Available GPUs do not imply that a one-process benchmark uses every GPU."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )

        result = runner.verify_baseline(MockBenchmark("baseline"))

        assert result.passed is True

    def test_explicit_world_size_still_requires_exact_visibility(
        self,
        temp_dirs,
        monkeypatch,
    ):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        baseline = MockBenchmark("baseline")
        baseline.required_world_size = 1

        result = runner.verify_baseline(baseline)

        assert result.passed is False
        assert result.reason == (
            "World size mismatch: requires exactly 1 visible GPU(s), found 2."
        )

    def test_multi_gpu_requirement_still_rejects_one_visible_device(
        self,
        temp_dirs,
        monkeypatch,
    ):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        baseline = MockBenchmark("baseline")
        baseline.multi_gpu_required = True

        result = runner.verify_baseline(baseline)

        assert result.passed is False
        assert result.reason == "SKIPPED: requires >=2 GPUs"

    def test_signature_validation_ignores_payload_output(self, temp_dirs):
        """Benchmarks using VerificationPayloadMixin include 'output' in signature; input validation must ignore it."""
        from core.benchmark.verification_mixin import VerificationPayloadMixin
        from core.harness.benchmark_harness import WorkloadMetadata

        class PayloadBenchmark(VerificationPayloadMixin):
            def __init__(self):
                self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=8.0)
                self._x: Optional[torch.Tensor] = None
                self._y: Optional[torch.Tensor] = None

            def setup(self) -> None:
                torch.manual_seed(42)
                self._x = torch.randn(8)

            def benchmark_fn(self) -> None:
                if self._x is None:
                    raise RuntimeError("setup() must run first")
                self._y = self._x * 2

            def capture_verification_payload(self) -> None:
                if self._x is None or self._y is None:
                    raise RuntimeError("benchmark_fn() must run first")
                self._set_verification_payload(
                    inputs={"x": self._x},
                    output=self._y,
                    batch_size=self._x.shape[0],
                    parameter_count=0,
                )

            def get_workload_metadata(self) -> Optional[Any]:
                return self._workload

            def validate_result(self) -> Optional[str]:
                return None

            def teardown(self) -> None:
                self._x = None
                self._y = None

        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        result = runner.verify_baseline(PayloadBenchmark())
        assert result.passed is True
    
    def test_verify_pair_matching_outputs(self, temp_dirs):
        """Test verifying pair with matching outputs."""
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        
        # Both benchmarks produce same output with same seed
        baseline = MockBenchmark("baseline")
        optimized = MockBenchmark("optimized")
        
        config = VerifyConfig(seed=42)
        result = runner.verify_pair(baseline, optimized, config)
        
        # Both get same random output with same seed
        assert result.passed is True
    
    def test_verify_missing_signature(self, temp_dirs):
        """Test verification fails for missing signature."""
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        
        baseline = MockBenchmark("baseline")
        baseline._signature = {}  # Empty signature
        
        result = runner.verify_baseline(baseline)
        assert result.passed is False
        assert "signature" in result.reason.lower()

    def test_verify_baseline_fails_when_mark_execution_complete_raises(self, temp_dirs):
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )

        class BrokenExecutionMarkerBenchmark(MockBenchmark):
            def mark_execution_complete(self) -> None:
                raise RuntimeError("marker boom")

        result = runner.verify_baseline(BrokenExecutionMarkerBenchmark("baseline"))

        assert result.passed is False
        assert "mark_execution_complete() failed during verification" in (result.reason or "")

    def test_verify_baseline_warns_when_teardown_fails(self, temp_dirs):
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )

        class BrokenTeardownBenchmark(MockBenchmark):
            def teardown(self) -> None:
                raise RuntimeError("teardown boom")

        with pytest.warns(RuntimeWarning, match="teardown\\(\\) failed during verification cleanup"):
            result = runner.verify_baseline(BrokenTeardownBenchmark("baseline"))

        assert result.passed is True

    def test_verify_baseline_fails_when_skip_hook_raises(self, temp_dirs):
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )

        class BrokenSkipHookBenchmark(MockBenchmark):
            def skip_input_verification(self) -> bool:
                raise RuntimeError("skip hook boom")

        result = runner.verify_baseline(BrokenSkipHookBenchmark("baseline"))

        assert result.passed is False
        assert result.reason == "skip hook boom"

    def test_verify_pair_surfaces_advisory_check_warnings(self, temp_dirs):
        runner = VerifyRunner(
            cache_dir=temp_dirs["cache"],
            quarantine_manager=QuarantineManager(temp_dirs["quarantine"]),
        )
        baseline = MockBenchmark("baseline")
        optimized = MockBenchmark("optimized")

        with patch.object(
            runner,
            "_run_fresh_input_check",
            return_value=(True, "Fresh input check warning: deterministic-by-design"),
        ), patch.object(
            runner,
            "_run_jitter_check",
            return_value=(True, "Jitter check skipped due to error: no mutable input tensor"),
        ):
            result = runner.verify_pair(baseline, optimized)

        assert result.passed is True
        assert result.details == {
            "warnings": [
                "Fresh input check warning: deterministic-by-design",
                "Jitter check skipped due to error: no mutable input tensor",
            ]
        }

    def test_timing_config_warns_when_get_config_raises(self):
        class BrokenGetConfigBenchmark:
            warmup_iterations = 7
            iterations = 11

            def get_config(self):
                raise RuntimeError("config boom")

        with pytest.warns(RuntimeWarning, match="Failed to inspect .*get_config\\(\\) for verification timing"):
            config = TimingConfig.from_benchmark(BrokenGetConfigBenchmark())

        assert config.warmup_iterations == 7
        assert config.measurement_iterations == 11


# =============================================================================
# VerifyResult Tests
# =============================================================================


class TestVerifyResult:
    """Tests for VerifyResult dataclass."""
    
    def test_success_factory(self):
        """Test success factory method."""
        result = VerifyResult.success("abc123")
        assert result.passed is True
        assert result.signature_hash == "abc123"
        assert result.timestamp is not None
    
    def test_fail_factory(self):
        """Test fail factory method."""
        result = VerifyResult.fail("Something went wrong")
        assert result.passed is False
        assert result.reason == "Something went wrong"
        assert result.timestamp is not None
    
    def test_to_dict_serialization(self):
        """Test serialization to dict."""
        result = VerifyResult.success(
            "abc123",
            baseline_checksum="check1",
            optimized_checksum="check2",
        )
        d = result.to_dict()
        
        assert d["passed"] is True
        assert d["signature_hash"] == "abc123"
        assert d["baseline_checksum"] == "check1"
        assert d["optimized_checksum"] == "check2"


# =============================================================================
# ComparisonDetails Tests
# =============================================================================


class TestComparisonDetails:
    """Tests for ComparisonDetails dataclass."""
    
    def test_passed_comparison(self):
        """Test creating passed comparison details."""
        details = ComparisonDetails(
            passed=True,
            max_diff=1e-7,
        )
        assert details.passed is True
    
    def test_failed_comparison_with_location(self):
        """Test creating failed comparison with location."""
        details = ComparisonDetails(
            passed=False,
            max_diff=0.5,
            location=(10, 20),
            expected_sample=1.0,
            actual_sample=1.5,
        )
        assert details.passed is False
        assert details.location == (10, 20)
    
    def test_to_dict(self):
        """Test serialization."""
        tol = ToleranceSpec(rtol=1e-3, atol=1e-5)
        details = ComparisonDetails(
            passed=False,
            max_diff=0.1,
            tolerance_used=tol,
        )
        d = details.to_dict()
        
        assert d["passed"] is False
        assert d["max_diff"] == 0.1
        assert "tolerance_used" in d


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerificationIntegration:
    """Integration tests for the full verification workflow."""
    
    @pytest.fixture
    def temp_environment(self, tmp_path):
        """Fixture setting up a complete temp environment."""
        cache_dir = tmp_path / "cache"
        quarantine_file = tmp_path / "quarantine.json"
        return {
            "cache_dir": cache_dir,
            "quarantine_file": quarantine_file,
        }
    
    def test_full_verification_workflow(self, temp_environment):
        """Test complete verification workflow."""
        # Create manager and runner
        manager = QuarantineManager(temp_environment["quarantine_file"])
        runner = VerifyRunner(
            cache_dir=temp_environment["cache_dir"],
            quarantine_manager=manager,
        )
        
        # Create mock benchmarks
        baseline = MockBenchmark("baseline")
        optimized = MockBenchmark("optimized")
        
        # Run verification
        result = runner.verify_pair(baseline, optimized)
        
        assert result.passed is True
        
        # Verify golden output was cached
        sig = runner._extract_signature(baseline)
        assert sig is not None
        assert runner.cache.has(sig.hash())
    
    def test_quarantine_on_compliance_failure(self, temp_environment):
        """Test quarantine is applied on compliance failure."""
        manager = QuarantineManager(temp_environment["quarantine_file"])
        
        class Benchmark:
            skip_output_check = True
            skip_input_check = False
            skip_verification = False

            def skip_input_verification(self) -> bool:
                return False

            def skip_output_verification(self) -> bool:
                return False

            def get_input_signature(self) -> Dict[str, Any]:
                return {"shapes": {"x": (10,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0}

            def get_verify_output(self) -> torch.Tensor:
                return torch.ones(10)

            def get_output_tolerance(self) -> tuple:
                return (1e-3, 1e-3)

            def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
                return {"x": torch.ones(10)}

            def validate_result(self) -> Optional[str]:
                return None

            def get_workload_metadata(self) -> Optional[Any]:
                return None
        
        benchmark = Benchmark()
        issues = check_benchmark_compliance(benchmark)
        assert len(issues) > 0
        
        # Apply quarantine for first issue
        if issues:
            manager.quarantine("test_benchmark.py", issues[0])
        
        assert manager.is_quarantined("test_benchmark.py")


# =============================================================================
# Property-Based Tests (using Hypothesis)
# =============================================================================

try:
    from hypothesis import given, strategies as st, settings
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20)
    def test_signature_hash_deterministic(self, batch_size):
        """Test that signature hash is always deterministic."""
        sig1 = InputSignature(
            shapes={"input": (batch_size, 256)},
            dtypes={"input": "float32"},
            batch_size=batch_size,
            parameter_count=100,
            precision_flags=PrecisionFlags(),
        )
        sig2 = InputSignature(
            shapes={"input": (batch_size, 256)},
            dtypes={"input": "float32"},
            batch_size=batch_size,
            parameter_count=100,
            precision_flags=PrecisionFlags(),
        )
        assert sig1.hash() == sig2.hash()
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20)
    def test_tolerance_looser_reflexive(self, rtol, atol):
        """Test that is_tolerance_looser is consistent."""
        tol = ToleranceSpec(rtol=rtol, atol=atol)
        # A tolerance should never be looser than itself
        assert not is_tolerance_looser(tol, tol)
    
    @given(
        st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20)
    def test_tolerance_looser_consistent(self, rtol1, atol1, rtol2, atol2):
        """Test that tolerance comparison is consistent with OR semantics.
        
        is_tolerance_looser uses OR logic: custom is looser if rtol > default.rtol OR atol > default.atol.
        This means both tolerances can be "looser" than each other if they differ in different dimensions.
        """
        tol1 = ToleranceSpec(rtol=rtol1, atol=atol1)
        tol2 = ToleranceSpec(rtol=rtol2, atol=atol2)
        
        # Test that the function is consistent: looser iff either rtol or atol is larger
        expected_1_looser = rtol1 > rtol2 or atol1 > atol2
        expected_2_looser = rtol2 > rtol1 or atol2 > atol1
        
        assert is_tolerance_looser(tol1, tol2) == expected_1_looser
        assert is_tolerance_looser(tol2, tol1) == expected_2_looser
    
    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_signature_different_shapes_different_hash(self, dim1, dim2, dim3):
        """Test that different shapes produce different hashes."""
        sig1 = InputSignature(
            shapes={"x": (dim1, dim2)},
            dtypes={"x": "float32"},
            batch_size=dim1,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        sig2 = InputSignature(
            shapes={"x": (dim1, dim3)},
            dtypes={"x": "float32"},
            batch_size=dim1,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )
        if dim2 != dim3:
            assert sig1.hash() != sig2.hash()
    
    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20)
    def test_signature_roundtrip_serialization(self, batch_size):
        """Test that signature can be serialized and deserialized."""
        original = InputSignature(
            shapes={"x": (batch_size, 64), "y": (64, batch_size)},
            dtypes={"x": "float32", "y": "float16"},
            batch_size=batch_size,
            parameter_count=batch_size * 64,
            precision_flags=PrecisionFlags(fp16=True),
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = InputSignature.from_dict(data)
        
        # Hashes should match
        assert original.hash() == restored.hash()
    
    @given(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20)
    def test_workload_metrics_equality_within_tolerance(self, baseline_val, optimized_val):
        """Test workload comparison respects tolerance."""
        baseline_metrics = {"bytes_per_iteration": baseline_val}
        optimized_metrics = {"bytes_per_iteration": optimized_val}
        
        # Default tolerance is 1% (0.01)
        passed, diff = compare_workload_metrics(baseline_metrics, optimized_metrics)
        
        if baseline_val > 0:
            ratio = optimized_val / baseline_val
            if abs(ratio - 1.0) <= 0.01:
                assert passed is True
            else:
                assert passed is False
    
    @given(st.integers(min_value=42, max_value=100))
    @settings(max_examples=10)
    def test_deterministic_seeds_are_reproducible(self, seed):
        """Test that set_deterministic_seeds produces reproducible state."""
        # Set seeds first time
        set_deterministic_seeds(seed)
        random1 = torch.rand(10)
        
        # Set seeds again with same value
        set_deterministic_seeds(seed)
        random2 = torch.rand(10)
        
        # Results should be identical
        assert torch.allclose(random1, random2)
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65)))
    @settings(max_examples=10)
    def test_quarantine_path_normalization(self, benchmark_path):
        """Test quarantine manager handles various path formats."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = QuarantineManager(Path(tmp_dir) / "quarantine.json")
            
            # Quarantine and check
            manager.quarantine(benchmark_path, QuarantineReason.OUTPUT_MISMATCH)
            
            # Should be quarantined regardless of path normalization
            assert manager.is_quarantined(benchmark_path)
            
            # Clear and verify
            manager.clear_quarantine(benchmark_path)
            assert not manager.is_quarantined(benchmark_path)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedVerifyRunner:
    """Property-based tests specifically for VerifyRunner."""
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10)
    def test_verify_pair_symmetry_of_output_comparison(self, tensor_size):
        """Test that output comparison is symmetric."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = VerifyRunner(cache_dir=Path(tmp_dir) / "cache")
            
            # Create identical outputs
            output1 = torch.randn(tensor_size, tensor_size)
            output2 = output1.clone()
            
            # ToleranceSpec requires rtol and atol arguments
            tol = ToleranceSpec(rtol=1e-5, atol=1e-8)
            
            # Comparison should be symmetric
            result1 = runner._compare_outputs(
                {"output": output1}, {"output": output2}, tol
            )
            result2 = runner._compare_outputs(
                {"output": output2}, {"output": output1}, tol
            )
            
            assert result1.passed == result2.passed
    
    @given(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=10)
    def test_output_comparison_respects_rtol(self, rtol):
        """Test that output comparison respects relative tolerance."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = VerifyRunner(cache_dir=Path(tmp_dir) / "cache")
            
            # Create outputs with relative difference
            output1 = torch.tensor([1.0, 2.0, 3.0])
            # Add relative error smaller than rtol
            output2 = output1 * (1.0 + rtol * 0.5)
            
            tol = ToleranceSpec(rtol=rtol, atol=1e-8)
            result = runner._compare_outputs({"output": output1}, {"output": output2}, tol)
            
            # Should pass with error less than rtol
            assert result.passed is True
    
    @given(st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False))
    @settings(max_examples=10)
    def test_output_comparison_fails_beyond_tolerance(self, rtol):
        """Test that output comparison fails beyond tolerance.
        
        Note: rtol capped at 0.5 because rtol=1.0 (100%) allows very large differences
        mathematically, which is technically correct but unintuitive.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = VerifyRunner(cache_dir=Path(tmp_dir) / "cache")
            
            # Create outputs with relative difference clearly exceeding rtol
            output1 = torch.tensor([1.0, 2.0, 3.0])
            # Use 3x the tolerance to ensure we're clearly beyond
            output2 = output1 * (1.0 + rtol * 3.0)
            
            tol = ToleranceSpec(rtol=rtol, atol=1e-8)
            result = runner._compare_outputs({"output": output1}, {"output": output2}, tol)
            
            # Should fail with error exceeding rtol
            assert result.passed is False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedGoldenOutput:
    """Property-based tests for GoldenOutput checksum consistency."""
    
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=10)
    def test_golden_output_checksum_deterministic(self, size):
        """Test that GoldenOutput checksum is deterministic."""
        outputs = {"output": torch.randn(size, size)}
        
        golden1 = GoldenOutput(
            signature_hash="test_hash",
            signature={"batch_size": size},
            outputs=outputs,
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden1.checksum = golden1.compute_checksum()
        
        golden2 = GoldenOutput(
            signature_hash="test_hash",
            signature={"batch_size": size},
            outputs={k: v.clone() for k, v in outputs.items()},
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden2.checksum = golden2.compute_checksum()
        
        assert golden1.checksum == golden2.checksum
    
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=10)
    def test_golden_output_checksum_sensitive_to_changes(self, size):
        """Test that GoldenOutput checksum changes with different outputs."""
        outputs1 = {"output": torch.randn(size, size)}
        outputs2 = {"output": torch.randn(size, size)}
        
        golden1 = GoldenOutput(
            signature_hash="hash1",
            signature={"batch_size": size},
            outputs=outputs1,
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden1.checksum = golden1.compute_checksum()
        
        golden2 = GoldenOutput(
            signature_hash="hash2",
            signature={"batch_size": size},
            outputs=outputs2,
            workload_metrics={},
            checksum="",
            created_at=datetime.now(),
            seed=42,
        )
        golden2.checksum = golden2.compute_checksum()
        
        # Different random outputs should produce different checksums
        # (extremely unlikely to be the same)
        assert golden1.checksum != golden2.checksum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
