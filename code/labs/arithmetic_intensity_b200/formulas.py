"""Hardware-independent arithmetic-intensity calculations for the B200 lab."""

from __future__ import annotations

import math
from typing import Any

from core.benchmark.metrics import BLACKWELL_B200

BF16_BYTES = 2
EXPECTED_B200_BF16_TFLOPS = 2_250.0
EXPECTED_B200_HBM_GBS = 8_000.0
KV_BYTES_PER_CONTEXT_TOKEN_PER_SEQUENCE = 144 * 1024
SEQUENCE_COUNT = 64
WEIGHT_BYTES = 15_140_000_000
EXAMPLE_CONTEXT_TOKENS = 872


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def verify_reviewed_b200_profile() -> dict[str, Any]:
    """Verify the reviewed dense BF16 and HBM values in the shared profile."""
    expected = {
        "tensor_tflops": EXPECTED_B200_BF16_TFLOPS,
        "hbm_bandwidth_gbps": EXPECTED_B200_HBM_GBS,
    }
    observed = {
        "tensor_tflops": float(BLACKWELL_B200.tensor_tflops),
        "hbm_bandwidth_gbps": float(BLACKWELL_B200.hbm_bandwidth_gbps),
    }
    mismatches = {
        field: {"expected": expected[field], "observed": observed[field]}
        for field in expected
        if not math.isclose(observed[field], expected[field], rel_tol=0.0, abs_tol=0.0)
    }
    if mismatches:
        raise RuntimeError(f"Reviewed B200 profile changed: {mismatches}")
    ridge = observed["tensor_tflops"] * 1e12 / (observed["hbm_bandwidth_gbps"] * 1e9)
    return {
        "source": "core.benchmark.metrics.BLACKWELL_B200",
        "profile_name": BLACKWELL_B200.name,
        "profile_source": BLACKWELL_B200.profile_source,
        "dense_bf16_tflops": observed["tensor_tflops"],
        "hbm_bandwidth_decimal_gb_per_s": observed["hbm_bandwidth_gbps"],
        "ridge_flop_per_byte": ridge,
        "verified_against_reviewed_values": True,
    }


def gemm_logical_cost(
    *,
    m: int,
    n: int,
    k: int,
    element_bytes: int = BF16_BYTES,
    beta_zero: bool = True,
) -> dict[str, Any]:
    """Return FLOPs and requested bytes for C = A @ B with explicit beta semantics."""
    _positive_int("m", m)
    _positive_int("n", n)
    _positive_int("k", k)
    _positive_int("element_bytes", element_bytes)
    if not isinstance(beta_zero, bool):
        raise TypeError("beta_zero must be a bool")

    a_read_bytes = m * k * element_bytes
    b_read_bytes = k * n * element_bytes
    c_write_bytes = m * n * element_bytes
    c_read_bytes = 0 if beta_zero else c_write_bytes
    logical_requested_bytes = a_read_bytes + b_read_bytes + c_read_bytes + c_write_bytes
    flops = 2 * m * n * k
    return {
        "shape": {"m": m, "n": n, "k": k},
        "dtype": "bf16" if element_bytes == BF16_BYTES else "unspecified",
        "element_bytes": element_bytes,
        "beta_zero": beta_zero,
        "flops": flops,
        "logical_requested_bytes": logical_requested_bytes,
        "logical_byte_breakdown": {
            "a_read_bytes": a_read_bytes,
            "b_read_bytes": b_read_bytes,
            "c_read_bytes": c_read_bytes,
            "c_write_bytes": c_write_bytes,
        },
        "logical_arithmetic_intensity_flop_per_byte": flops / logical_requested_bytes,
        "physical_hbm_counter_bytes": None,
        "physical_hbm_counter_source": "requires_ncu_dram_read_write_counters",
    }


def memory_bound_ms(byte_count: int, bandwidth_decimal_gb_per_s: float) -> float:
    """Return the ideal transfer-only lower bound in milliseconds."""
    _positive_int("byte_count", byte_count)
    if (
        isinstance(bandwidth_decimal_gb_per_s, bool)
        or not isinstance(bandwidth_decimal_gb_per_s, int | float)
        or not math.isfinite(bandwidth_decimal_gb_per_s)
        or bandwidth_decimal_gb_per_s <= 0
    ):
        raise ValueError("bandwidth_decimal_gb_per_s must be finite and positive")
    return byte_count / (float(bandwidth_decimal_gb_per_s) * 1e9) * 1e3


def formula_report() -> dict[str, Any]:
    """Build the reproducible analytical report used by the live diagnostic."""
    profile = verify_reviewed_b200_profile()
    m1 = gemm_logical_cost(m=1, n=4096, k=4096, beta_zero=True)
    m64 = gemm_logical_cost(m=64, n=4096, k=4096, beta_zero=True)
    kv_bytes_per_context_position = KV_BYTES_PER_CONTEXT_TOKEN_PER_SEQUENCE * SEQUENCE_COUNT
    equality_context_tokens = WEIGHT_BYTES / kv_bytes_per_context_position
    example_kv_bytes = kv_bytes_per_context_position * EXAMPLE_CONTEXT_TOKENS
    combined_bytes = WEIGHT_BYTES + example_kv_bytes
    bandwidth = float(profile["hbm_bandwidth_decimal_gb_per_s"])

    return {
        "schema_version": 1,
        "kind": "b200_arithmetic_intensity_formula_verification",
        "status": "verified",
        "hardware_profile": profile,
        "gemm_beta_zero": {"m1_k4096_n4096": m1, "m64_k4096_n4096": m64},
        "kv_cache_model": {
            "kv_bytes_per_context_token_per_sequence": (KV_BYTES_PER_CONTEXT_TOKEN_PER_SEQUENCE),
            "kv_kib_per_context_token_per_sequence": 144,
            "sequence_count": SEQUENCE_COUNT,
            "kv_bytes_per_context_position_all_sequences": (kv_bytes_per_context_position),
            "weight_bytes": WEIGHT_BYTES,
            "weight_decimal_gb": WEIGHT_BYTES / 1e9,
            "weight_kv_equality_context_tokens": equality_context_tokens,
            "example_context_tokens": EXAMPLE_CONTEXT_TOKENS,
            "example_kv_bytes": example_kv_bytes,
            "example_combined_weight_and_kv_bytes": combined_bytes,
            "example_combined_weight_and_kv_decimal_gb": combined_bytes / 1e9,
        },
        "ideal_hbm_transfer_bounds": {
            "assumed_hbm_bandwidth_decimal_gb_per_s": bandwidth,
            "weights_only_ms": memory_bound_ms(WEIGHT_BYTES, bandwidth),
            "example_weight_and_kv_ms": memory_bound_ms(combined_bytes, bandwidth),
            "scope": "transfer_only_ideal_lower_bound",
        },
        "interpretation_limits": {
            "logical_requested_bytes_are_physical_hbm_bytes": False,
            "nominal_profile_is_measured_peak": False,
            "ncu_required_for_physical_hbm_bytes": True,
        },
    }
