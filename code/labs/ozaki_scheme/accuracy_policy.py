"""Explicit, externally reviewed Ozaki bounds; configuration is not calibration evidence."""

import json
import math
import os
from pathlib import Path


def configured_accuracy(variant: str) -> tuple[list[str], tuple[float, float]]:
    path = os.environ.get("AISP_OZAKI_ACCURACY_POLICY")
    if not path:
        # The binary rejects emulation without bounds before allocating/running.
        return [], (0.0, 0.0)
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Ozaki accuracy policy requires schema_version=1")
    item = policy[variant]
    for name in ("relative_l2", "normalized_max_abs", "checksum_rtol"):
        value = float(item[name])
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0,1)")
    atol = float(item["checksum_atol"])
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("checksum_atol must be finite and nonnegative")
    return (["--relative-l2-limit", str(item["relative_l2"]),
             "--normalized-max-abs-limit", str(item["normalized_max_abs"])],
            (float(item["checksum_rtol"]), atol))
