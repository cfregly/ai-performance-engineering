import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    if use_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("No compatible GPU found. Falling back to CPU.")
    return torch.device("cpu")


# Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/utils.py
@torch.inference_mode()
def verbose_allclose(
    received: torch.Tensor, expected: torch.Tensor, rtol=1e-05, atol=1e-08, max_print=5
) -> list[str]:
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    rtol (float): Relative tolerance; relative to expected
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if received.shape != expected.shape:
        return ["SIZE MISMATCH"]

    # Calculate the difference between the tensors
    diff = torch.abs(received - expected)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(expected)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(
        torch.isposinf(received), torch.isposinf(expected)
    )
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(
        torch.isneginf(received), torch.isneginf(expected)
    )

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched, as_tuple=False)

    # Count the number of mismatched elements
    num_mismatched = int(mismatched_indices.shape[0])

    # Generate detailed information if there are mismatches
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]

        mismatch_index_rows = mismatched_indices[:max_print].detach().cpu().tolist()
        for index in mismatch_index_rows:
            i = tuple(int(dim) for dim in index)
            mismatch_details.append(f"ERROR AT {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(
                f"... and {num_mismatched - max_print} more mismatched elements."
            )
        return mismatch_details

    return []


@torch.inference_mode()
def verbose_allequal(
    received: torch.Tensor, expected: torch.Tensor, max_print: int = 5
):
    """
    Assert that two tensors are element-wise perfectly equal, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    max_print (int): Maximum number of mismatched elements to print.

    Returns:
         Empty string if tensors are equal, otherwise detailed error information
    """
    mismatched = torch.not_equal(received, expected)
    mismatched_indices = torch.nonzero(mismatched, as_tuple=False)

    # Count the number of mismatched elements
    num_mismatched = int(mismatched_indices.shape[0])

    # Generate detailed information if there are mismatches
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]

        mismatch_index_rows = mismatched_indices[:max_print].detach().cpu().tolist()
        for index in mismatch_index_rows:
            i = tuple(int(dim) for dim in index)
            mismatch_details.append(f"ERROR AT {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(
                f"... and {num_mismatched - max_print} more mismatched elements."
            )
        return mismatch_details

    return []


def match_reference(
    data, output, reference: callable, rtol=1e-05, atol=1e-08
) -> tuple[bool, str]:
    """
    Convenient "default" implementation for tasks' `check_implementation` function.
    """
    expected = reference(data)
    reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)

    if len(reasons) > 0:
        return (
            False,
            "mismatch found! custom implementation doesn't match reference: "
            + " ".join(reasons),
        )

    return True, ""


def make_match_reference(reference: callable, **kwargs):
    def wrapped(data, output):
        return match_reference(data, output, reference=reference, **kwargs)

    return wrapped


class DeterministicContext:
    def __init__(self):
        self.allow_tf32 = None
        self.deterministic = None
        self.cublas = None

    def __enter__(self):
        self.cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.cublas


_L2_FLUSH_BUFFERS: dict[tuple[int, int], torch.Tensor] = {}


def _clear_l2_cache_numel(numel: int) -> None:
    device_index = torch.cuda.current_device()
    key = (device_index, int(numel))
    buffer = _L2_FLUSH_BUFFERS.get(key)
    if buffer is None or buffer.device.index != device_index:
        buffer = torch.empty(numel, device=f"cuda:{device_index}", dtype=torch.float32)
        _L2_FLUSH_BUFFERS[key] = buffer
    buffer.zero_()


def clear_l2_cache():
    _clear_l2_cache_numel(1024 * 1024 * 1024)


def clear_l2_cache_large():
    _clear_l2_cache_numel(16000 * 1024 * 1024)
