"""Shared 128-row by four-scale-column NVFP4 layout from the GEMM reference.

Only storage layout is shared. Quantization and grouped-GEMM numerical oracles
remain independent of the optimized kernels.
"""

import torch


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Block aligned scale matrices; preserve optional leading batch dimensions.

    Input is [..., rows, scale_columns], with rows divisible by 128 and columns
    divisible by four. Output is [..., rows * scale_columns]. Callers own padding.
    """
    if input_matrix.ndim < 2:
        raise ValueError("scale layout requires at least two dimensions")
    rows, cols = input_matrix.shape[-2:]
    if rows <= 0 or cols <= 0 or rows % 128 or cols % 4:
        raise ValueError("scale matrices must be nonempty and aligned to 128 by 4")
    leading = input_matrix.shape[:-2]
    blocks = input_matrix.reshape(-1, rows // 128, 128, cols // 4, 4).permute(0, 1, 3, 2, 4)
    rearranged = blocks.reshape(-1, (rows // 128) * (cols // 4), 4, 32, 4).transpose(2, 3)
    return rearranged.reshape(*leading, rows * cols)
