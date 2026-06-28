from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Use new SDPA API when available (PyTorch 2.2+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _EFFICIENT_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    _NEW_SDPA_API = True
except ImportError:
    sdpa_kernel = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    _EFFICIENT_BACKENDS = []
    _NEW_SDPA_API = False

_FLASH3_ACCEPTS_CLUSTERS: bool | None = None
_CU_SEQLENS_CACHE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _efficient_sdpa_context():
    """Return context manager for memory-efficient + math attention backends."""
    if _NEW_SDPA_API and sdpa_kernel is not None:
        return sdpa_kernel(_EFFICIENT_BACKENDS)
    return nullcontext()


def _expand_gqa_heads(x: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat <= 1:
        return x
    batch, heads, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(batch, heads, repeat, seq_len, head_dim).reshape(
        batch,
        heads * repeat,
        seq_len,
        head_dim,
    )


def _flash3_accepts_clusters(flash3_fn) -> bool:
    global _FLASH3_ACCEPTS_CLUSTERS
    if _FLASH3_ACCEPTS_CLUSTERS is None:
        try:
            import inspect

            _FLASH3_ACCEPTS_CLUSTERS = "num_sm_clusters" in inspect.signature(flash3_fn).parameters
        except Exception:
            _FLASH3_ACCEPTS_CLUSTERS = False
    return _FLASH3_ACCEPTS_CLUSTERS


def _cu_seqlens_for(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    key = (batch, seq_len, device)
    cached = _CU_SEQLENS_CACHE.get(key)
    if cached is None:
        cached = torch.arange(0, (batch + 1) * seq_len, step=seq_len, device=device, dtype=torch.int32)
        _CU_SEQLENS_CACHE[key] = cached
    return cached


def _flash3_clustered(q, k, v, causal: bool, num_sm_clusters: int | None, enable_gqa: bool):
    # q,k,v: (B, H, T, D)
    if enable_gqa and q.size(1) != k.size(1):
        repeat_k = q.size(1) // k.size(1)
        k = _expand_gqa_heads(k, repeat_k)
        v = _expand_gqa_heads(v, repeat_k)
    B, Hq, Tq, D = q.shape
    _, Hk, Tk, _ = k.shape
    q_flat = q.transpose(1, 2).reshape(B * Tq, Hq, D)
    k_flat = k.transpose(1, 2).reshape(B * Tk, Hk, D)
    v_flat = v.transpose(1, 2).reshape(B * Tk, Hk, D)
    cu_q = _cu_seqlens_for(B, Tq, q.device)
    cu_k = _cu_seqlens_for(B, Tk, q.device)

    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func  # type: ignore

        kwargs = dict(
            dropout_p=0.0,
            causal=causal,
        )
        if num_sm_clusters is not None and _flash3_accepts_clusters(flash_attn_varlen_func):
            kwargs["num_sm_clusters"] = num_sm_clusters
        out = flash_attn_varlen_func(  # type: ignore[misc]
            q_flat,
            k_flat,
            v_flat,
            cu_q,
            cu_k,
            Tq,
            Tk,
            **kwargs,
        )
        return out.view(B, Tq, Hq, D).transpose(1, 2).contiguous()
    except Exception:
        return None


def clustered_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    causal: bool,
    num_sm_clusters: int | None = None,
    enable_gqa: bool = False,
):
    """
    Clustered attention entry point.
    - Tries FlashAttention-3 varlen with num_sm_clusters if available.
    - Falls back to SDPA.
    """
    # Masks are not supported in FA3 varlen path; fall back to SDPA when provided.
    use_mask = attn_mask is not None
    if (
        not use_mask
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        fa3_out = _flash3_clustered(q, k, v, causal=causal, num_sm_clusters=num_sm_clusters, enable_gqa=enable_gqa)
        if fa3_out is not None:
            return fa3_out

    # SDPA fallback
    # attn_mask semantics: True=keep, False=mask
    with _efficient_sdpa_context():
        if use_mask:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=enable_gqa)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)
