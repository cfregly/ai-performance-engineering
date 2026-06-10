#!POPCORN leaderboard nvfp4_gemv
#!POPCORN gpu NVIDIA

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
try:
    from task import input_t, output_t
    from utils import make_match_reference
except ModuleNotFoundError:
    from labs.nvfp4_gemv.task import input_t, output_t
    from labs.nvfp4_gemv.utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16

# Optional case0 fast path from nvfp4_gemm challenge.
_GEMM_V3B = None
_CASE0_OUT_CACHE: dict[tuple[int, int, int, int], torch.Tensor] = {}
_STREAM_POOL: dict[tuple[int, int], list[torch.cuda.Stream]] = {}
_MISSING = object()

# GB300 fast path: a torch.compile-fused dequant GEMV. The fp4 (e2m1) matrix is unpacked
# and scaled (per-16 e4m3 block) and reduced against the dequantized vector; inductor fuses
# the dequant into the GEMV reduction (no [m,k] fp16 materialization), which beats the
# scaled_mm path (it pads N=1 to 128) and gemm_v3b (a GEMM kernel on a GEMV). Bit-exact to
# the reference (max abs diff 0); ~2.5x on GB300.
_NVFP4_GEMV_LUT = None
_NVFP4_GEMV_DQ_COMPILED = None


def _nvfp4_e2m1_lut() -> torch.Tensor:
    global _NVFP4_GEMV_LUT
    if _NVFP4_GEMV_LUT is None:
        vals = []
        for n in range(16):
            sign = (n >> 3) & 1
            exp = (n >> 1) & 3
            man = n & 1
            mag = (man * 0.5) if exp == 0 else (2.0 ** (exp - 1)) * (1 + man * 0.5)
            vals.append(-mag if sign else mag)
        _NVFP4_GEMV_LUT = torch.tensor(vals, dtype=torch.float16, device="cuda")
    return _NVFP4_GEMV_LUT


def _nvfp4_dequant_gemv_one(a8, b8, sfa, sfb, lut):
    lo = (a8 & 0xF).long()
    hi = ((a8 >> 4) & 0xF).long()
    a_idx = torch.stack([lo, hi], dim=-1).reshape(a8.shape[0], -1)
    a_f = lut[a_idx] * sfa.repeat_interleave(16, dim=1)
    blo = (b8 & 0xF).long()
    bhi = ((b8 >> 4) & 0xF).long()
    b_idx = torch.stack([blo, bhi], dim=-1).reshape(-1)
    b_f = lut[b_idx] * sfb.repeat_interleave(16)
    return (a_f.float() @ b_f.float()).half()


def _nvfp4_dequant_gemv_compiled():
    global _NVFP4_GEMV_DQ_COMPILED
    if _NVFP4_GEMV_DQ_COMPILED is None:
        _NVFP4_GEMV_DQ_COMPILED = torch.compile(
            _nvfp4_dequant_gemv_one, mode="default", fullgraph=True
        )
    return _NVFP4_GEMV_DQ_COMPILED


def _run_dequant_gemv(data: input_t) -> torch.Tensor:
    a_ref, b_ref, sfa_ref, sfb_ref, _sfa_p, _sfb_p, c_ref = data
    _, _, l = c_ref.shape
    lut = _nvfp4_e2m1_lut()
    dq = _nvfp4_dequant_gemv_compiled()
    for l_idx in range(int(l)):
        a8 = a_ref[:, :, l_idx].contiguous().view(torch.uint8)
        b8 = b_ref[0, :, l_idx].contiguous().view(torch.uint8)
        sfa = sfa_ref[:, :, l_idx].to(torch.float16)
        sfb = sfb_ref[0, :, l_idx].to(torch.float16)
        c_ref[:, 0, l_idx] = dq(a8, b8, sfa, sfb, lut)
    return c_ref


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def _install_module_aliases(aliases):
    previous = {name: sys.modules.get(name, _MISSING) for name in aliases}
    try:
        sys.modules.update(aliases)
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Convert [rows, cols] scale tensor to blocked layout for torch._scaled_mm."""
    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Official reference assumes pre-aligned inputs from generate_input.
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _load_gemm_v3b():
    global _GEMM_V3B
    if _GEMM_V3B is not None:
        return _GEMM_V3B

    if os.getenv("AISP_NVFP4_GEMV_USE_GEMM_V3B_CASE0", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return None

    module_path = Path(__file__).resolve().parents[1] / "nvfp4_gemm" / "optimized_submission.py"
    task_mod = _load_module_from_path(module_path.parent / "task.py", "nvfp4_gemm_task_for_gemv")
    utils_mod = _load_module_from_path(module_path.parent / "utils.py", "nvfp4_gemm_utils_for_gemv")
    with _install_module_aliases({"task": task_mod, "utils": utils_mod}):
        ref_mod = _load_module_from_path(
            module_path.parent / "reference_submission.py",
            "nvfp4_gemm_reference_for_gemv",
        )
    with _install_module_aliases(
        {"task": task_mod, "utils": utils_mod, "reference_submission": ref_mod}
    ):
        mod = _load_module_from_path(module_path, "nvfp4_gemm_opt_for_gemv")
    _GEMM_V3B = getattr(mod, "gemm_v3b", None)
    return _GEMM_V3B


def _get_case0_out_slot(m: int, n: int, c: torch.Tensor, slot: int) -> torch.Tensor:
    key = (int(c.device.index), int(m), int(n), int(slot))
    cached = _CASE0_OUT_CACHE.get(key)
    if cached is not None and cached.device == c.device and cached.dtype == torch.float16:
        return cached
    out = torch.empty((m, n), dtype=torch.float16, device=c.device)
    _CASE0_OUT_CACHE[key] = out
    return out


def _get_case0_out(m: int, n: int, c: torch.Tensor) -> torch.Tensor:
    return _get_case0_out_slot(m, n, c, 0)


def _run_case0_gemm_v3b(data: input_t, gemm_v3b) -> torch.Tensor:
    a_ref, b_ref, _sfa_ref, _sfb_ref, sfa_permuted, sfb_permuted, c_ref = data
    out = _get_case0_out(int(a_ref.shape[0]), int(b_ref.shape[0]), c_ref)
    ret = gemm_v3b(
        a_ref[:, :, 0],
        b_ref[:, :, 0],
        sfa_permuted[..., 0],
        sfb_permuted[..., 0],
        out,
    )
    if ret.dim() == 3:
        c_ref[:, 0, 0].copy_(ret[:, 0, 0])
    else:
        c_ref[:, 0, 0].copy_(ret[:, 0])
    return c_ref


def _run_gemm_v3b_all(data: input_t, gemm_v3b) -> torch.Tensor:
    a_ref, b_ref, _sfa_ref, _sfb_ref, sfa_permuted, sfb_permuted, c_ref = data
    _, _, l = c_ref.shape
    m = int(a_ref.shape[0])
    n = int(b_ref.shape[0])
    k = int(a_ref.shape[1]) * 2
    case1_n_eff = int(os.getenv("AISP_NVFP4_GEMV_CASE1_N_EFF", "96"))
    if k == 7168:
        case1_n_eff = max(64, min(n, case1_n_eff))
        case1_n_eff = min(n, max(64, (case1_n_eff // 16) * 16))
    else:
        case1_n_eff = n

    stream_count = int(os.getenv("AISP_NVFP4_GEMV_GEMM_V3B_STREAMS", "1"))
    stream_count = max(1, min(int(l), stream_count))
    if stream_count <= 1:
        out = _get_case0_out(m, case1_n_eff, c_ref)
        for l_idx in range(int(l)):
            b_in = b_ref[:case1_n_eff, :, l_idx] if case1_n_eff != n else b_ref[:, :, l_idx]
            ret = gemm_v3b(
                a_ref[:, :, l_idx],
                b_in,
                sfa_permuted[..., l_idx],
                sfb_permuted[..., l_idx],
                out,
            )
            if ret.dim() == 3:
                c_ref[:, 0, l_idx].copy_(ret[:, 0, 0])
            else:
                c_ref[:, 0, l_idx].copy_(ret[:, 0])
        return c_ref

    default_stream = torch.cuda.current_stream()
    streams = _get_streams(c_ref, stream_count)
    used_streams: set[torch.cuda.Stream] = set()
    for l_idx in range(int(l)):
        stream_slot = int(l_idx % stream_count)
        stream = streams[stream_slot]
        used_streams.add(stream)
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            out = _get_case0_out_slot(m, case1_n_eff, c_ref, stream_slot + 1)
            b_in = b_ref[:case1_n_eff, :, l_idx] if case1_n_eff != n else b_ref[:, :, l_idx]
            ret = gemm_v3b(
                a_ref[:, :, l_idx],
                b_in,
                sfa_permuted[..., l_idx],
                sfb_permuted[..., l_idx],
                out,
            )
            if ret.dim() == 3:
                c_ref[:, 0, l_idx].copy_(ret[:, 0, 0])
            else:
                c_ref[:, 0, l_idx].copy_(ret[:, 0])
    for stream in used_streams:
        default_stream.wait_stream(stream)
    return c_ref


def _run_scaled_mm(data: input_t) -> torch.Tensor:
    a_ref, b_ref, sfa_ref, sfb_ref, _sfa_permuted, _sfb_permuted, c_ref = data
    _, _, l = c_ref.shape

    packed_scale_a = [to_blocked(sfa_ref[:, :, l_idx]) for l_idx in range(int(l))]
    packed_scale_b = [to_blocked(sfb_ref[:, :, l_idx]) for l_idx in range(int(l))]

    stream_count = _resolve_stream_count(int(l))
    if stream_count <= 1 or int(l) <= 1:
        for l_idx in range(int(l)):
            res = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_ref[:, :, l_idx].transpose(0, 1),
                packed_scale_a[l_idx],
                packed_scale_b[l_idx],
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = res[:, 0]
        return c_ref

    default_stream = torch.cuda.current_stream()
    streams = _get_streams(c_ref, stream_count)
    used_streams: set[torch.cuda.Stream] = set()
    for l_idx in range(int(l)):
        stream = streams[l_idx % stream_count]
        used_streams.add(stream)
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            res = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_ref[:, :, l_idx].transpose(0, 1),
                packed_scale_a[l_idx],
                packed_scale_b[l_idx],
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = res[:, 0]
    for stream in used_streams:
        default_stream.wait_stream(stream)
    return c_ref


def _resolve_stream_count(l: int) -> int:
    raw = os.getenv("AISP_NVFP4_GEMV_STREAMS")
    if raw is not None and raw.strip() != "":
        return max(1, min(l, int(raw)))
    if l >= 8:
        return 8
    if l >= 4:
        return 4
    return 1


def _get_streams(c_ref: torch.Tensor, stream_count: int) -> list[torch.cuda.Stream]:
    key = (int(c_ref.device.index), int(stream_count))
    cached = _STREAM_POOL.get(key)
    if cached is not None:
        return cached
    streams = [torch.cuda.Stream() for _ in range(int(stream_count))]
    _STREAM_POOL[key] = streams
    return streams


def custom_kernel(data: input_t) -> output_t:
    a_ref, _b_ref, _sfa_ref, _sfb_ref, _sfa_perm, _sfb_perm, c_ref = data
    _, _, l = c_ref.shape
    k = int(a_ref.shape[1]) * 2

    # GB300 case0 (l=1, k=16384): the tensor-core scaled_mm path is ~2.8x faster than the
    # dequant GEMV at this large-k shape (measured 339->121 us same-process, bit-exact
    # maxdiff=0): the big-K NVFP4 tensor-core GEMM amortizes the N=1->128 pad while the
    # dequant is SM-ALU-bound. case1/case2 stay on the dequant GEMV (best there).
    if (
        int(l) == 1
        and int(k) == 16384
        and os.getenv("AISP_NVFP4_GEMV_CASE0_SCALED_MM", "1").strip().lower()
        in {"1", "true", "on", "yes"}
    ):
        try:
            return _run_scaled_mm(data)
        except Exception:
            pass

    # GB300: a torch.compile-fused dequant GEMV beats the scaled_mm / gemm_v3b paths here
    # (those pad N=1 to 128 or run a GEMM kernel on a GEMV). Bit-exact; ~2.5x. Falls through
    # to the legacy paths on any error or when explicitly disabled.
    if os.getenv("AISP_NVFP4_GEMV_USE_DEQUANT_GEMV", "1").strip().lower() in {"1", "true", "on", "yes"}:
        try:
            return _run_dequant_gemv(data)
        except Exception:
            pass

    if os.getenv("AISP_NVFP4_GEMV_USE_GEMM_V3B_ALL", "1").strip().lower() in {"1", "true", "on", "yes"}:
        gemm_v3b = _load_gemm_v3b()
        if gemm_v3b is not None:
            try:
                return _run_gemm_v3b_all(data, gemm_v3b)
            except Exception:
                pass

    if int(l) == 1 and int(k) == 16384:
        gemm_v3b = _load_gemm_v3b()
        if gemm_v3b is not None:
            try:
                return _run_case0_gemm_v3b(data, gemm_v3b)
            except Exception:
                # Keep a verify-safe fallback when optional fast path fails.
                pass

    return _run_scaled_mm(data)


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    return c_ref


def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.

    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [1, k, l] - Input vector in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [1, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, 1, l] - Output vector in torch.float16 data type
    """
    torch.manual_seed(seed)

    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    # Pad b tensor's N dimension to 128 to call torch._scaled_mm for nvfp4 dot product computation
    b_ref = torch.randint(
        0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )

    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(0, 3, ref_shape, dtype=torch.int8, device="cuda")
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(0, 3, mma_shape, dtype=torch.int8, device="cuda")
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device="cuda")
        j_idx = torch.arange(sf_k, device="cuda")
        b_idx = torch.arange(l, device="cuda")

        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[
            i_grid, j_grid, b_grid
        ]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref = sfa_ref_cpu.to("cuda")
    sfb_ref = sfb_ref_cpu.to("cuda")

    return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
