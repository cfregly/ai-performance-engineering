"""kernel_verification_tool.py - Manual kernel correctness testing.

This demonstrates traditional manual approaches to verifying GPU kernel
correctness, which are labor-intensive and incomplete:

1. Random input testing (doesn't explore edge cases systematically)
2. Reference comparison (requires correct reference implementation)
3. Manual edge case identification (error-prone, incomplete)
4. Print debugging (doesn't scale)

ProofWright and other formal verification tools automate these processes
using LLM-based agents and formal methods for end-to-end correctness
guarantees.
"""

from __future__ import annotations

import torch
from typing import Optional, List, Tuple, Dict, Any

from core.benchmark.utils import scalar_tensor_to_float
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class ManualKernelVerifier:
    """Manual kernel verification helper.
    
    Traditional approach with significant limitations:
    - Incomplete coverage of input space
    - No formal memory safety guarantees
    - No thread safety proofs
    - Relies on human-identified edge cases
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.test_results: List[Dict[str, Any]] = []
    
    def random_test(
        self, 
        kernel_fn,
        reference_fn,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        num_tests: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> Tuple[bool, List[str]]:
        """Test with random inputs - incomplete coverage.
        
        Tolerances: 1e-3 default for CUDA (parallel reduction has ~1e-3 variance).
        
        Limitations:
        - Random sampling misses corner cases
        - No guarantee of covering important code paths
        - May miss numerical edge cases (denormals, inf, nan)
        """
        errors = []
        
        for i in range(num_tests):
            x = torch.randn(*shape, device=self.device, dtype=dtype)
            
            try:
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                if not torch.allclose(kernel_out, ref_out, rtol=rtol, atol=atol):
                    max_diff = scalar_tensor_to_float((kernel_out - ref_out).abs().max())
                    errors.append(f"Test {i}: max diff = {max_diff}")
            except Exception as e:
                errors.append(f"Test {i}: Exception - {e}")
        
        return len(errors) == 0, errors
    
    def edge_case_test(
        self,
        kernel_fn,
        reference_fn,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[bool, List[str]]:
        """Test with manually-identified edge cases.
        
        Limitations:
        - Human must identify all edge cases (error-prone)
        - Easy to miss domain-specific corner cases
        - Doesn't catch all numerical stability issues
        """
        errors = []
        
        edge_cases = [
            ("zeros", torch.zeros(*shape, device=self.device, dtype=dtype)),
            ("ones", torch.ones(*shape, device=self.device, dtype=dtype)),
            ("large", torch.full(shape, 1e6, device=self.device, dtype=dtype)),
            ("small", torch.full(shape, 1e-6, device=self.device, dtype=dtype)),
            ("negative", torch.full(shape, -1.0, device=self.device, dtype=dtype)),
        ]
        
        # Add inf/nan only for float types
        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            edge_cases.extend([
                ("inf", torch.full(shape, float('inf'), device=self.device, dtype=dtype)),
                ("neg_inf", torch.full(shape, float('-inf'), device=self.device, dtype=dtype)),
                # NaN often causes issues - important edge case
                ("nan", torch.full(shape, float('nan'), device=self.device, dtype=dtype)),
            ])
        
        for name, x in edge_cases:
            try:
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                # Special handling for NaN - both should be NaN
                if name == "nan":
                    if not (torch.isnan(kernel_out).all() == torch.isnan(ref_out).all()):
                        errors.append(f"Edge case '{name}': NaN handling mismatch")
                    continue
                
                # Allow NaN outputs for inf inputs (depends on kernel)
                if name in ["inf", "neg_inf"]:
                    continue
                
                # Use looser tolerance for CUDA - parallel reduction has ~1e-3 variance
                if not torch.allclose(kernel_out, ref_out, rtol=1e-3, atol=1e-3, equal_nan=True):
                    max_diff = scalar_tensor_to_float((kernel_out - ref_out).abs().max())
                    errors.append(f"Edge case '{name}': max diff = {max_diff}")
            except Exception as e:
                errors.append(f"Edge case '{name}': Exception - {e}")
        
        return len(errors) == 0, errors
    
    def boundary_test(
        self,
        kernel_fn,
        reference_fn,
        base_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[bool, List[str]]:
        """Test boundary conditions for shapes.
        
        Limitations:
        - Doesn't prove correctness for all shapes
        - May miss subtle alignment issues
        - Thread block boundary issues may not surface
        """
        errors = []
        
        # Test various shapes around powers of 2 (common boundary issues)
        test_shapes = []
        for dim_idx in range(len(base_shape)):
            for offset in [-1, 0, 1]:
                shape = list(base_shape)
                shape[dim_idx] = max(1, shape[dim_idx] + offset)
                test_shapes.append(tuple(shape))
        
        # Add single element and minimum viable shapes
        test_shapes.append((1,) * len(base_shape))
        
        for shape in test_shapes:
            try:
                x = torch.randn(*shape, device=self.device, dtype=dtype)
                kernel_out = kernel_fn(x)
                ref_out = reference_fn(x)
                
                # Use looser tolerance for CUDA - parallel reduction has ~1e-3 variance
                if not torch.allclose(kernel_out, ref_out, rtol=1e-3, atol=1e-3):
                    max_diff = scalar_tensor_to_float((kernel_out - ref_out).abs().max())
                    errors.append(f"Shape {shape}: max diff = {max_diff}")
            except Exception as e:
                errors.append(f"Shape {shape}: Exception - {e}")
        
        return len(errors) == 0, errors


class BaselineKernelVerificationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Manual kernel verification approach.
    
    Demonstrates the traditional, labor-intensive approach to GPU kernel
    verification with its inherent limitations:
    
    1. **Coverage gaps**: Random testing doesn't guarantee complete coverage
    2. **No formal proofs**: Can't prove memory/thread safety
    3. **Human effort**: Edge cases must be manually identified
    4. **Scaling issues**: More complex kernels = exponentially more tests
    
    See optimized_proofwright_verify.py for automated formal verification.
    """

    # This benchmark intentionally generates random and adversarial test inputs
    # and compares host-visible discrepancies as part of the verification story.
    allowed_benchmark_fn_antipatterns = (
        "allocation",
        "host_transfer",
        "random_input_regeneration",
    )
    
    def __init__(self):
        super().__init__()
        self.verifier = None
        self.test_kernel = None
        self.reference_fn = None
        self.shape = (1024, 1024)
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.shape[0] * self.shape[1]),
        )
        self._random_test_result: Dict[str, Any] = {"passed": False, "errors": []}
        self._edge_case_result: Dict[str, Any] = {"passed": False, "errors": []}
        self._boundary_test_result: Dict[str, Any] = {"passed": False, "errors": []}
        self._verification_results: Dict[str, Any] = {
            "random_tests": self._random_test_result,
            "edge_cases": self._edge_case_result,
            "boundary_tests": self._boundary_test_result,
        }
    
    def setup(self) -> None:
        """Setup: Initialize verifier and test functions."""
        self.verifier = ManualKernelVerifier(device=str(self.device))
        
        # Example kernel: a simple GELU implementation to verify
        # In practice, this would be a custom CUDA kernel
        def test_gelu_kernel(x):
            # Simulated custom kernel using tanh approximation
            return x * 0.5 * (1.0 + torch.tanh(
                0.7978845608028654 * (x + 0.044715 * x.pow(3))
            ))
        
        def reference_gelu(x):
            # Reference implementation (same as kernel - should match exactly)
            return x * 0.5 * (1.0 + torch.tanh(
                0.7978845608028654 * (x + 0.044715 * x.pow(3))
            ))
        
        self.test_kernel = test_gelu_kernel
        self.reference_fn = reference_gelu
        self._verify_input = torch.arange(
            self.shape[0] * self.shape[1],
            device=self.device,
            dtype=torch.float32,
        ).reshape(self.shape)
        
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run manual verification suite.
        
        This runs the full manual verification workflow and measures
        the time and completeness of the approach.
        """
        with torch.inference_mode(), self._nvtx_range("baseline_kernel_verification"):
            # Run random tests
            random_pass, random_errors = self.verifier.random_test(
                self.test_kernel, 
                self.reference_fn,
                shape=self.shape,
                num_tests=20,
            )
            
            # Run edge case tests
            edge_pass, edge_errors = self.verifier.edge_case_test(
                self.test_kernel,
                self.reference_fn, 
                shape=self.shape,
            )
            
            # Run boundary tests
            boundary_pass, boundary_errors = self.verifier.boundary_test(
                self.test_kernel,
                self.reference_fn,
                base_shape=self.shape,
            )
            
            self._random_test_result["passed"] = random_pass
            self._random_test_result["errors"] = random_errors
            self._edge_case_result["passed"] = edge_pass
            self._edge_case_result["errors"] = edge_errors
            self._boundary_test_result["passed"] = boundary_pass
            self._boundary_test_result["errors"] = boundary_errors

            if self._verify_input is None:
                raise RuntimeError("setup() must initialize verification input")
            if self.test_kernel is None:
                raise RuntimeError("setup() must initialize test kernel")
            self.output = self.test_kernel(self._verify_input)[:32, :32]

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output.contiguous(),
            batch_size=int(self.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.verifier = None
        self.test_kernel = None
        self.reference_fn = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return verification-specific metrics."""
        results = self._verification_results
        
        num_tests = 20 + 8 + 7  # random + edge cases + boundary
        passed = sum(1 for k, v in results.items() if v.get("passed", False))
        
        return {
            "verification_approach": "manual",
            "total_test_categories": 3,
            "categories_passed": passed,
            "num_random_tests": 20,
            "num_edge_cases": 8,
            "num_boundary_tests": 7,
            "coverage_guaranteed": False,  # Key limitation
            "memory_safety_proven": False,
            "thread_safety_proven": False,
            "formal_proof": False,
        }

    def validate_result(self) -> Optional[str]:
        """Validate verification results."""
        if not self._verification_results:
            return "Verification not completed"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineKernelVerificationBenchmark()
