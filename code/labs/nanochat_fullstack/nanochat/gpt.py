"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when running standalone
    prefer_sdpa_backends = None  # type: ignore

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.kernels.clustered_attention import clustered_attention
from nanochat.kernels.stubs import resolve_clustered_attention_kernel


def _maybe_make_weight_only_linear(in_features, out_features, config, name="linear"):
    """Create a linear layer; optionally use Transformer Engine when flagged."""
    use_te = getattr(config, "use_te_weight_only", False)
    if not use_te:
        return nn.Linear(in_features, out_features, bias=False), None
    try:  # pragma: no cover - optional dependency
        import transformer_engine.pytorch as te  # type: ignore
    except Exception as exc:
        raise ImportError(f"use_te_weight_only=True but Transformer Engine is unavailable for {name}") from exc
    params_dtype = torch.float16 if str(getattr(config, "te_weight_dtype", "fp8")).lower() in ("fp4", "int4", "fp16") else torch.float32
    layer = te.Linear(in_features, out_features, bias=False, params_dtype=params_dtype)
    return layer, "te"

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    use_fp32_logits: bool = True  # if False, keep logits in bf16 during loss to hit fused CE
    use_flash_sdp: bool = True  # if True, prefer Flash/TE SDP kernels (training path)
    use_padded_attention: bool = False  # if True, enable attention masks for padded batches
    use_flash3: bool = True  # prefer FlashAttention-3 varlen kernels (B200/TMA) when available
    flash3_block_size: int = 128  # sequence tile for FA3 varlen kernels (aligns to TMEM staging)
    kv_block_size: Optional[int] = None  # optional KV cache block size for TMA/paged layout
    kv_page_size: Optional[int] = None  # optional KV cache page size for growth hints
    enable_persistent_decode: bool = False  # gate persistent decode kernels (Engine)
    use_cuda_graphs: bool = False  # gate CUDA Graph capture in Engine generate paths
    use_te_weight_only: bool = False  # if True, prefer Transformer Engine weight-only linears (q/k/v/proj + lm_head)
    te_weight_dtype: str = "fp8"  # fp8|fp4|int4 hint for TE weight-only path
    use_cta_clustering: bool = False  # if True, enable CTA clustering (prefill) when kernels are available
    cta_cluster_size: int = 2  # default CTAs per cluster (auto-tuned per sequence length)
    cta_cluster_seq_threshold: int = 1024  # minimum sequence length before attempting clustering
    use_clustered_attention_kernel: bool = False  # experimental: attempt custom clustered attention kernel (requires build)
    use_persistent_decode_kernel: bool = False  # experimental: attempt custom resident decode kernel (requires build)
    clustered_attention_impl: Optional[str] = None  # optional module:function override for clustered attention
    persistent_decode_impl: Optional[str] = None  # optional module:function override for persistent decode
    allow_kernel_stub_fallback: bool = False  # allow falling back to reference path instead of raising when kernel flags are on


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    if not torch.is_grad_enabled() or not x.requires_grad:
        out = torch.empty_like(x)
        torch.mul(x1, cos, out=out[..., :d])
        out[..., :d].addcmul_(x2, sin)
        torch.mul(x2, cos, out=out[..., d:])
        out[..., d:].addcmul_(x1, sin, value=-1)
        return out
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.use_flash_sdp = config.use_flash_sdp
        self.use_flash3 = getattr(config, "use_flash3", False)
        self.flash3_block_size = getattr(config, "flash3_block_size", None)
        self.use_te_weight_only = getattr(config, "use_te_weight_only", False)
        self.use_cta_clustering = getattr(config, "use_cta_clustering", False)
        self.cta_cluster_seq_threshold = getattr(config, "cta_cluster_seq_threshold", 1024)
        self.cta_cluster_size = getattr(config, "cta_cluster_size", 2)
        self.use_clustered_attention_kernel = getattr(config, "use_clustered_attention_kernel", False)
        self.use_padded_attention = config.use_padded_attention
        self.clustered_attention_impl = getattr(config, "clustered_attention_impl", None)
        self.allow_kernel_stub_fallback = getattr(config, "allow_kernel_stub_fallback", False)
        # Allow swapping in custom clustered-attention kernels when flagged
        self.clustered_attention_kernel = resolve_clustered_attention_kernel(
            fallback=clustered_attention,
            impl=self.clustered_attention_impl,
            allow_fallback=self.allow_kernel_stub_fallback,
        )
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.weight_only_backend = None
        self.c_q, backend = _maybe_make_weight_only_linear(self.n_embd, self.n_head * self.head_dim, config, name="c_q")
        self.weight_only_backend = backend or self.weight_only_backend
        self.c_k, backend = _maybe_make_weight_only_linear(self.n_embd, self.n_kv_head * self.head_dim, config, name="c_k")
        self.weight_only_backend = backend or self.weight_only_backend
        self.c_v, backend = _maybe_make_weight_only_linear(self.n_embd, self.n_kv_head * self.head_dim, config, name="c_v")
        self.weight_only_backend = backend or self.weight_only_backend
        self.c_proj, backend = _maybe_make_weight_only_linear(self.n_embd, self.n_embd, config, name="c_proj")
        self.weight_only_backend = backend or self.weight_only_backend
        self.sdpa_ctx_factory = prefer_sdpa_backends if prefer_sdpa_backends is not None else (lambda order=None: nullcontext())
        if self.use_flash3 and torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major < 10:
                self.use_flash3 = False
        self.flash3_fn = None
        self.flash3_error = None
        self._cu_q_cache = None
        self._cu_k_cache = None
        self._causal_mask_cache = None
        self._prefix_causal_mask_cache = None
        if self.use_flash3:
            self._init_flash3()

    def _init_flash3(self):
        """Best-effort lazy import of FlashAttention-3 varlen kernel."""
        try:  # pragma: no cover - import guard
            from flash_attn.flash_attn_interface import flash_attn_varlen_func  # type: ignore

            self.flash3_fn = flash_attn_varlen_func
        except Exception as exc:
            self.flash3_error = str(exc)
            self.flash3_fn = None
            self.use_flash3 = False

    def _flash3_supported(self, q, attn_mask):
        if not self.use_flash3 or self.flash3_fn is None:
            return False
        if attn_mask is not None or self.use_padded_attention:
            return False
        if not q.is_cuda:
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        return True

    def _auto_cluster_size(self, seq_len):
        # Simple heuristic: larger sequences benefit from larger clusters
        if seq_len >= 4096:
            return max(4, self.cta_cluster_size)
        if seq_len >= 2048:
            return max(3, self.cta_cluster_size)
        return self.cta_cluster_size

    def _cu_seqlens_buffer(self, batch, seq_len, device, cache_name):
        spec_name = f"{cache_name}_spec"
        spec = (batch, seq_len, device)
        cached = getattr(self, cache_name)
        if cached is None or getattr(self, spec_name, None) != spec:
            cached = torch.arange(0, (batch + 1) * seq_len, step=seq_len, device=device, dtype=torch.int32)
            setattr(self, cache_name, cached)
            setattr(self, spec_name, spec)
        return cached

    def _causal_mask_for(self, t_q, t_k, device):
        spec = (t_q, t_k, device)
        if self._causal_mask_cache is None or getattr(self, "_causal_mask_spec", None) != spec:
            self._causal_mask_cache = torch.tril(
                torch.ones((t_q, t_k), dtype=torch.bool, device=device)
            )
            self._causal_mask_spec = spec
        return self._causal_mask_cache

    def _prefix_causal_mask_for(self, t_q, t_k, device):
        spec = (t_q, t_k, device)
        if self._prefix_causal_mask_cache is None or getattr(self, "_prefix_causal_mask_spec", None) != spec:
            prefix_len = t_k - t_q
            mask = torch.zeros((t_q, t_k), dtype=torch.bool, device=device)
            if prefix_len > 0:
                mask[:, :prefix_len] = True
            mask[:, prefix_len:] = torch.tril(
                torch.ones((t_q, t_q), dtype=torch.bool, device=device)
            )
            self._prefix_causal_mask_cache = mask
            self._prefix_causal_mask_spec = spec
        return self._prefix_causal_mask_cache

    def _flash3_attention(self, q, k, v, kv_cache, enable_gqa, use_clustering=False):
        """Varlen FlashAttention-3 path (no masks). Returns None on fallback."""
        Tq, Tk = q.size(2), k.size(2)
        if kv_cache is None or Tq == Tk:
            causal = True
        elif Tq == 1:
            causal = False  # steady-state decode: allow full prefix
        else:
            return None  # unsupported shape, fall back to SDPA
        # Expand GQA heads if FA3 build doesn't expose num_heads_k
        if enable_gqa and self.n_head != self.n_kv_head:
            repeat_k = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_k, dim=1)
            v = v.repeat_interleave(repeat_k, dim=1)
        B, Hq, _, D = q.size()
        _, Hk, _, _ = k.size()
        q_flat = q.transpose(1, 2).reshape(B * Tq, Hq, D)
        k_flat = k.transpose(1, 2).reshape(B * Tk, Hk, D)
        v_flat = v.transpose(1, 2).reshape(B * Tk, Hk, D)
        cu_q = self._cu_seqlens_buffer(B, Tq, q.device, "_cu_q_cache")
        cu_k = self._cu_seqlens_buffer(B, Tk, q.device, "_cu_k_cache")
        
        # CTA clustering hint: Some FlashAttention-3 builds support num_sm_clusters
        # to enable cooperative thread array clustering on Hopper/Blackwell
        fa3_kwargs = dict(
            dropout_p=0.0,
            causal=causal,
        )
        if use_clustering:
            # Try to pass cluster size hint if FlashAttention-3 supports it
            # This enables __cluster_dims__ in CUDA kernels for better L1 sharing
            try:
                import inspect
                if 'num_sm_clusters' in inspect.signature(self.flash3_fn).parameters:
                    fa3_kwargs['num_sm_clusters'] = self._auto_cluster_size(Tk)
            except Exception:
                pass  # Clustering not supported in this FA3 build, continue without it
        
        out = self.flash3_fn(  # type: ignore[misc]
            q_flat,
            k_flat,
            v_flat,
            cu_q,
            cu_k,
            Tq,
            Tk,
            **fa3_kwargs,
        )
        return out.view(B, Tq, Hq, D).transpose(1, 2).contiguous()

    def forward(self, x, cos_sin, kv_cache, attention_mask=None, token_mask=None):
        B, T, C = x.size()
        
        # CTA clustering hint for attention kernels (Blackwell/Hopper optimization)
        # Note: Full CTA clustering requires custom CUDA kernels with __cluster_dims__ annotations
        # or FlashAttention-3 cluster support. This flag enables best-effort optimizations:
        # 1. Use larger tile sizes when T >= threshold (better SM occupancy)
        # 2. Provide hint to flash_attn_varlen_func if it supports clustering
        # 3. Enable when custom cluster kernels become available
        use_cta_hint = self.use_cta_clustering and T >= self.cta_cluster_seq_threshold
        
        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            cache_token_mask = token_mask if self.use_padded_attention else None
            if cache_token_mask is not None:
                assert cache_token_mask.shape[0] == B and cache_token_mask.shape[1] == T, f"token_mask shape mismatch: {cache_token_mask.shape} vs ({B}, {T})"
            k, v = kv_cache.insert_kv(self.layer_idx, k, v, token_mask=cache_token_mask)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        use_mask = self.use_padded_attention and attention_mask is not None
        if attention_mask is not None and not self.use_padded_attention:
            raise ValueError("attention_mask provided but use_padded_attention=False")
        sdpa_order = None
        if not self.use_flash_sdp:
            # Allow callers to force efficient/math paths when flash/TE is undesired.
            sdpa_order = [
                backend
                for backend in (
                    getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
                    getattr(SDPBackend, "MATH", None),
                )
                if backend is not None
            ]
        attn_mask = None
        if use_mask:
            if attention_mask.dim() == 2:
                key_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                key_mask = attention_mask[:, None, :, :]
            else:
                key_mask = attention_mask
            key_mask = key_mask.to(dtype=torch.bool, device=q.device)
            assert key_mask.size(-1) == Tk, f"attention_mask length mismatch: {key_mask.size(-1)} != {Tk}"
            if kv_cache is not None and Tq != Tk:
                causal = self._prefix_causal_mask_for(Tq, Tk, q.device)
            else:
                causal = self._causal_mask_for(Tq, Tk, q.device)
            attn_mask = key_mask & causal
        fa3_out = None
        if attn_mask is None and self._flash3_supported(q, attn_mask):
            fa3_out = self._flash3_attention(q, k, v, kv_cache, enable_gqa=enable_gqa, use_clustering=use_cta_hint)
        if fa3_out is not None:
            y = fa3_out
        else:
            if self.use_clustered_attention_kernel:
                y = self.clustered_attention_kernel(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    causal=(kv_cache is None or Tq == Tk) if kv_cache is None else (Tq == Tk or Tq == 1),
                    num_sm_clusters=self._auto_cluster_size(Tk) if use_cta_hint else None,
                    enable_gqa=enable_gqa and self.n_head != self.n_kv_head,
                )
            else:
                with self.sdpa_ctx_factory(sdpa_order):
                    if attn_mask is not None:
                        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
                    elif kv_cache is None or Tq == Tk:
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
                    elif Tq == 1:
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
                    else:
                        attn_mask = self._prefix_causal_mask_for(Tq, Tk, q.device)
                        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, attention_mask=None, token_mask=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, attention_mask=attention_mask, token_mask=token_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head, _ = _maybe_make_weight_only_linear(config.n_embd, config.vocab_size, config, name="lm_head")
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("_position_offsets", torch.empty(0, dtype=torch.long), persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def _position_offsets_for(self, length: int, device: torch.device) -> torch.Tensor:
        offsets = self._position_offsets
        if offsets.device != device or offsets.numel() < length:
            self._position_offsets = torch.arange(length, device=device, dtype=torch.long)
            offsets = self._position_offsets
        return offsets[:length]

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        use_dist = getattr(self.config, "use_dist_adamw", True)
        AdamWFactory = DistAdamW if (ddp and use_dist) else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, attention_mask=None, token_mask=None, loss_reduction='mean'):
        B, T = idx.size()

        if attention_mask is not None:
            if not self.config.use_padded_attention:
                raise ValueError("attention_mask provided but config.use_padded_attention=False")
            attention_mask = attention_mask.to(device=idx.device, dtype=torch.bool)
            assert attention_mask.size(0) == B, f"attention_mask batch mismatch: {attention_mask.size(0)} != {B}"
        if token_mask is not None:
            token_mask = token_mask.to(device=idx.device, dtype=torch.bool)
        elif attention_mask is not None and attention_mask.shape[-1] == T:
            # Default to using the attention mask for KV cache insertion when shapes match
            token_mask = attention_mask

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        if kv_cache is not None and kv_cache.get_row_pos() is not None and self.config.use_padded_attention:
            row_pos = kv_cache.get_row_pos()
            assert row_pos.numel() == B, f"kv_cache row_pos mismatch: {row_pos.numel()} != {B}"
            positions = row_pos[:, None] + self._position_offsets_for(T, idx.device)
            max_pos = int(positions.max().item()) + 1
            assert max_pos <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {max_pos} > {self.cos.size(1)}"
            cos_sin = self.cos[:, positions, :, :].squeeze(0), self.sin[:, positions, :, :].squeeze(0)
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            assert T0 + T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T0 + T} > {self.cos.size(1)}"
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, attention_mask=attention_mask, token_mask=token_mask)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            if self.config.use_fp32_logits:
                logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
