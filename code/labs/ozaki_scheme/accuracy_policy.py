"""Source-defined Ozaki arithmetic bounds; configuration is not run evidence."""

import json
import math
import os
from pathlib import Path


POLICY_ID = "ozaki-fp64-emulation-arithmetic-ceilings-v1"
REFERENCE_ID = "native-fp64-full-plus-cpu-long-double-edge-v1"
DEFAULT_POLICY_PATH = Path(__file__).with_name("accuracy_policy.json")
WORKLOAD = {
    "m": 4096,
    "n": 4096,
    "k": 4096,
    "input_scale": 0.001,
    "emulation_strategy": "eager",
    "dynamic_max_bits": 16,
    "dynamic_offset": -56,
    "fixed_bits": 12,
}
QUALIFICATION_VARIANTS = ["dynamic", "fixed"]
QUALIFICATION_RECEIPTS = [
    {"cohort": "nominal", "seeds": [2026], "m": 4096, "n": 4096, "k": 4096,
     "input_scale": 0.001, "input_pattern": "uniform", "reference_mode": "native_fp64"},
    {"cohort": "holdout", "seeds": [2027, 2029], "m": 4096, "n": 4096, "k": 4096,
     "input_scale": 0.001, "input_pattern": "uniform", "reference_mode": "native_fp64"},
    {"cohort": "alternating_edge", "seeds": [2039], "m": 17, "n": 19, "k": 23,
     "input_scale": 1.0, "input_pattern": "alternating", "reference_mode": "cpu_long_double"},
    {"cohort": "dynamic_range_edge", "seeds": [2053], "m": 31, "n": 29, "k": 37,
     "input_scale": 1.0, "input_pattern": "dynamic_range", "reference_mode": "cpu_long_double"},
]

# The dynamic policy is an explicit five-bit output-quality floor for the
# deliberately accuracy-reducing -56 offset. It is a conservative engineering
# requirement, not a cuBLAS guarantee. Fixed-12 is held to one 2^-12 grid step
# in aggregate and two steps at the worst output. Checksum tolerances are only
# secondary harness bounds, derived with Cauchy-Schwarz from the full-array L2
# ceiling and |a|,|b| <= input_scale: limit * m*n*k*input_scale^2.
ENGINEERING_CEILINGS = {
    "dynamic": {
        "relative_l2": 2.0**-5,
        "normalized_max_abs": 2.0**-4,
        "checksum_rtol": 0.0,
        "checksum_atol": (2.0**-5) * 4096**3 * 0.001**2,
    },
    "fixed": {
        "relative_l2": 2.0**-12,
        "normalized_max_abs": 2.0**-11,
        "checksum_rtol": 0.0,
        "checksum_atol": (2.0**-12) * 4096**3 * 0.001**2,
    },
}


def _limits_from_item(item: dict) -> dict[str, float]:
    result = {}
    for name in ("relative_l2", "normalized_max_abs", "checksum_rtol", "checksum_atol"):
        try:
            value = float(item[name])
        except KeyError as exc:
            raise ValueError(f"Ozaki accuracy policy missing {name}") from exc
        if not math.isfinite(value) or value < 0 or (name != "checksum_atol" and value >= 1):
            interval = "nonnegative" if name == "checksum_atol" else "in [0,1)"
            raise ValueError(f"{name} must be finite and {interval}")
        result[name] = value
    return result


def load_accuracy_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    schema_version = policy.get("schema_version")
    if schema_version == 1:
        # Exact-zero fixture policies remain useful for CPU comparator tests,
        # but legacy documents cannot carry a nonzero benchmark acceptance bar.
        for variant in ("dynamic", "fixed"):
            if variant not in policy:
                continue
            if any(_limits_from_item(policy[variant]).values()):
                raise ValueError("schema_version=1 is permitted only for exact-zero test policies")
        return policy
    if schema_version != 2:
        raise ValueError("Ozaki accuracy policy requires schema_version=2")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError(f"Ozaki accuracy policy requires policy_id={POLICY_ID}")
    if policy.get("reference", {}).get("id") != REFERENCE_ID:
        raise ValueError(f"Ozaki accuracy policy requires reference.id={REFERENCE_ID}")
    if policy.get("workload") != WORKLOAD:
        raise ValueError("Ozaki accuracy policy workload does not match the benchmark contract")
    qualification = policy.get("qualification", {})
    if (qualification.get("variants") != QUALIFICATION_VARIANTS or
            qualification.get("required_receipts") != QUALIFICATION_RECEIPTS):
        raise ValueError("Ozaki accuracy policy qualification matrix does not match the source contract")
    variants = policy.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Ozaki accuracy policy requires a variants object")
    for variant, ceilings in ENGINEERING_CEILINGS.items():
        if variant not in variants:
            raise ValueError(f"Ozaki accuracy policy missing variant {variant}")
        limits = _limits_from_item(variants[variant])
        for name, ceiling in ceilings.items():
            if limits[name] > ceiling:
                raise ValueError(
                    f"{variant}.{name} exceeds the source-defined engineering ceiling {ceiling:.8g}"
                )
    return policy


def configured_accuracy(variant: str) -> tuple[list[str], tuple[float, float]]:
    path = os.environ.get("AISP_OZAKI_ACCURACY_POLICY")
    if not path:
        # The binary rejects emulation without bounds before allocating/running.
        return [], (0.0, 0.0)
    policy = load_accuracy_policy(Path(path))
    try:
        item = policy[variant] if policy["schema_version"] == 1 else policy["variants"][variant]
    except KeyError as exc:
        raise ValueError(f"Ozaki accuracy policy missing variant {variant}") from exc
    limits = _limits_from_item(item)
    relative_arg = item["relative_l2"] if policy["schema_version"] == 1 else limits["relative_l2"]
    normalized_arg = item["normalized_max_abs"] if policy["schema_version"] == 1 else limits["normalized_max_abs"]
    return (["--relative-l2-limit", str(relative_arg),
             "--normalized-max-abs-limit", str(normalized_arg)],
            (limits["checksum_rtol"], limits["checksum_atol"]))
