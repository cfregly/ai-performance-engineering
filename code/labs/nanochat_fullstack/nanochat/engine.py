"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
import os
from contextlib import contextmanager
from collections import deque
from nanochat.kernels.persistent_decode import PersistentDecodeRunner
from nanochat.kernels.stubs import resolve_persistent_decode_kernel
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext 

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, block_size=None, page_size=None):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.block_size = block_size
        self.page_size = page_size
        seq_capacity = self._round_seq_len(seq_len)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_capacity, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache
        self.row_pos = None # optional per-row positions when using padded variable-length inputs
        self.cache_gen = 0  # incremented when storage grows
        self._batch_idx = None
        self._row_position_idx = None

    def _round_seq_len(self, seq_len):
        """Round sequence length to block/page boundaries to align with TMA paging."""
        rounded = seq_len
        if self.block_size is not None and self.block_size > 0:
            rounded = ((rounded + self.block_size - 1) // self.block_size) * self.block_size
        if self.page_size is not None and self.page_size > 0:
            rounded = ((rounded + self.page_size - 1) // self.page_size) * self.page_size
        return rounded

    def reset(self):
        self.pos = 0
        self.row_pos = None

    def get_pos(self):
        return self.pos

    def get_row_pos(self):
        return self.row_pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        if self.block_size != other.block_size or self.page_size != other.page_size:
            raise AssertionError("KV cache block/page configuration mismatch")
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos
        if other.row_pos is not None:
            self.row_pos = other.row_pos.clone()

    def _maybe_grow_cache(self, t_needed, dtype, device):
        """Grow kv_cache time dimension to at least t_needed."""
        if t_needed <= self.kv_cache.size(4):
            return
        # Mark that cache has grown (important for CUDA graphs - they capture memory pointers!)
        self.cache_gen += 1
        # round up to block/page/1024 boundary to keep allocations coarse
        round_step = 1024
        if self.block_size is not None:
            round_step = max(round_step, self.block_size)
        if self.page_size is not None:
            round_step = max(round_step, self.page_size)
        t_needed = ((t_needed + round_step - 1) // round_step) * round_step
        t_needed = self._round_seq_len(t_needed)
        grown_shape = list(self.kv_cache.shape)
        old_seq_len = self.kv_cache.size(4)
        grown_shape[4] = t_needed
        grown_cache = torch.empty(grown_shape, dtype=dtype, device=device)
        grown_cache[:, :, :, :, :old_seq_len, :].copy_(self.kv_cache)
        self.kv_cache = grown_cache
        self.kv_shape = self.kv_cache.shape

    def get_block_info(self):
        if self.block_size is None:
            return None
        current = self.get_pos()
        blocks = (current + self.block_size - 1) // self.block_size if current > 0 else 0
        capacity_blocks = self.kv_shape[4] // self.block_size if self.block_size else 0
        return dict(
            block_size=self.block_size,
            page_size=self.page_size,
            num_blocks=blocks,
            capacity_blocks=capacity_blocks,
        )

    def get_blocked_kv(self, layer_idx):
        """Return block-major view for TMA/TMEM kernels."""
        if self.block_size is None or self.kv_cache is None:
            return None
        total_tokens = self.get_pos()
        if total_tokens == 0:
            return None
        num_blocks = (total_tokens + self.block_size - 1) // self.block_size
        block_tokens = num_blocks * self.block_size
        k = self.kv_cache[layer_idx, 0, :, :, :block_tokens, :].view(
            self.kv_cache.size(2),
            self.kv_cache.size(3),
            num_blocks,
            self.block_size,
            self.kv_cache.size(5),
        )
        v = self.kv_cache[layer_idx, 1, :, :, :block_tokens, :].view(
            self.kv_cache.size(2),
            self.kv_cache.size(3),
            num_blocks,
            self.block_size,
            self.kv_cache.size(5),
        )
        return k, v

    def _batch_index_buffer(self, batch_size, device):
        batch_idx = self._batch_idx
        if batch_idx is None or batch_idx.device != device or batch_idx.numel() < batch_size:
            batch_idx = torch.arange(batch_size, device=device)
            self._batch_idx = batch_idx
        return batch_idx[:batch_size]

    def _row_position_buffer(self, batch_size, device):
        positions = self._row_position_idx
        if positions is None or positions.device != device or positions.numel() < batch_size:
            positions = torch.empty(batch_size, device=device, dtype=torch.long)
            self._row_position_idx = positions
        return positions[:batch_size]

    def insert_kv(self, layer_idx, k, v, token_mask=None, max_cache_len=None):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        use_row_pos = token_mask is not None or self.row_pos is not None
        if token_mask is not None:
            if token_mask.device != k.device or token_mask.dtype != torch.bool:
                token_mask = token_mask.to(device=k.device, dtype=torch.bool)
        if use_row_pos:
            if self.row_pos is None:
                self.row_pos = torch.zeros(B, device=k.device, dtype=torch.long)
            else:
                assert self.row_pos.numel() == B, f"row_pos shape mismatch: {self.row_pos.numel()} != {B}"
            # ensure we have enough capacity for the maximum position that will be written
            base_row_pos = self.row_pos
            if token_mask is None:
                dense_positions = self._row_position_buffer(B, k.device)
                dense_positions.copy_(base_row_pos)
                max_needed = int(max_cache_len) if max_cache_len is not None else int(dense_positions.max().item()) + T_add
            else:
                token_increments = token_mask.sum(dim=1)
                next_row_pos = base_row_pos + token_increments
                max_needed = int(max_cache_len) if max_cache_len is not None else int(next_row_pos.max().item())
            self._maybe_grow_cache(max_needed, k.dtype, k.device)
            batch_idx = self._batch_index_buffer(B, k.device)
            if token_mask is None:
                for t in range(T_add):
                    self.kv_cache[layer_idx, 0, batch_idx, :, dense_positions] = k[:, :, t, :]
                    self.kv_cache[layer_idx, 1, batch_idx, :, dense_positions] = v[:, :, t, :]
                    dense_positions.add_(1)
            else:
                for t in range(T_add):
                    active = token_mask[:, t]
                    rows = batch_idx[active]
                    if rows.numel() == 0:
                        continue
                    positions = base_row_pos[rows] + t
                    self.kv_cache[layer_idx, 0, rows, :, positions] = k[rows, :, t, :]
                    self.kv_cache[layer_idx, 1, rows, :, positions] = v[rows, :, t, :]
            if layer_idx == self.kv_cache.size(0) - 1:
                if token_mask is None:
                    self.row_pos.copy_(dense_positions)
                else:
                    self.row_pos = next_row_pos
            t1 = max_needed
        else:
            t0, t1 = self.pos, self.pos + T_add
            self._maybe_grow_cache(t1, k.dtype, k.device)
            self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
            self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
            if layer_idx == self.kv_cache.size(0) - 1:
                self.pos = t1
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes (for per-row, track max)
        if use_row_pos and layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(
    logits,
    rng,
    temperature=1.0,
    top_k=None,
    *,
    out=None,
    choice_out=None,
    max_values_out=None,
    probs_out=None,
    topk_values_out=None,
    topk_indices_out=None,
    topk_probs_out=None,
):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        if out is not None and max_values_out is not None:
            torch.max(logits, dim=-1, keepdim=True, out=(max_values_out, out))
            return out
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        if (
            out is not None
            and choice_out is not None
            and topk_values_out is not None
            and topk_indices_out is not None
            and topk_probs_out is not None
        ):
            torch.topk(logits, k, dim=-1, out=(topk_values_out, topk_indices_out))
            topk_values_out.div_(temperature)
            torch.softmax(topk_values_out, dim=-1, out=topk_probs_out)
            torch.multinomial(topk_probs_out, num_samples=1, generator=rng, out=choice_out)
            torch.gather(topk_indices_out, 1, choice_out, out=out)
            return out
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        if out is not None and probs_out is not None:
            probs_out.copy_(logits)
            probs_out.div_(temperature)
            torch.softmax(probs_out, dim=-1, out=probs_out)
            torch.multinomial(probs_out, num_samples=1, generator=rng, out=out)
            return out
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer, reuse_ids_buffer=True, enable_batch_decode=False):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.reuse_ids_buffer = reuse_ids_buffer
        self.enable_batch_decode = enable_batch_decode
        self.enable_persistent_decode = bool(getattr(self.model.config, "enable_persistent_decode", False))
        self.use_cuda_graphs = bool(getattr(self.model.config, "use_cuda_graphs", False))
        self._decode_graph_disabled = False
        self.use_persistent_decode_kernel = bool(getattr(self.model.config, "use_persistent_decode_kernel", False))
        self._kernel_stub_fallback = bool(getattr(self.model.config, "allow_kernel_stub_fallback", False))
        self._persistent_decode_impl = getattr(self.model.config, "persistent_decode_impl", None)
        self._compile_error = None
        self._graph_cache_gen = None  # cache generation tied to current capture
        self._attention_positions = None
        self._attention_mask = None
        self._batch_row_indices = None
        self._token_column_host = None
        self._active_mask_host = None
        self._active_mask_device = None
        self._active_indices_host = None
        self._active_indices_device = None
        self._sample_token_device_buffer = None
        self._sample_token_host_buffer = None
        self._sample_next_id_buffer = None
        self._sample_choice_buffer = None
        self._sample_max_values_buffer = None
        self._sample_probs_buffer = None
        self._sample_active_logits_buffer = None
        self._sample_topk_values_buffer = None
        self._sample_topk_indices_buffer = None
        self._sample_topk_probs_buffer = None
        self._prompt_lengths_host = None
        self._prompt_ids_host = None
        self._prompt_ids_device = None
        self._lengths_by_batch = None
        self._ids_step_buffer = None
        self._pd_runner = PersistentDecodeRunner() if self.use_persistent_decode_kernel else None
        self._persistent_decode_kernel = (
            resolve_persistent_decode_kernel(
                fallback=self._pd_runner.forward,
                impl=self._persistent_decode_impl,
                allow_fallback=self._kernel_stub_fallback,
            )
            if self.use_persistent_decode_kernel
            else None
        )
        
        # Persistent decode state: preallocated buffers for steady-state decode
        self._persistent_logits_buffer = None
        self._persistent_probs_buffer = None
        self._persistent_stream = None
        if self.enable_persistent_decode and not self.use_cuda_graphs:
            # Persistent decode path: 
            # 1. Reuse buffers to minimize allocations
            # 2. Use dedicated CUDA stream for decode pipeline
            # 3. Preallocate common intermediate buffers
            # This reduces kernel launch overhead in steady-state decode
            self.reuse_ids_buffer = True
            if torch.cuda.is_available():
                # Dedicated high-priority stream for decode operations
                try:
                    self._persistent_stream = torch.cuda.Stream(priority=-1)
                except Exception:
                    self._persistent_stream = None
        self._decode_graph = None
        self._graph_stream = None
        self._graph_static_ids = None
        self._graph_static_attention = None
        self._graph_static_token_mask = None
        self._graph_output = None
        self._graph_device = None
        self._graph_cache_gen = None
        if self.enable_batch_decode:
            self.model.config.use_padded_attention = True
            # propagate to attention modules (they cache the flag at init time)
            for block in getattr(self.model.transformer, "h", []):
                block.attn.use_padded_attention = True
        self._reset_decode_graph()
        self._maybe_compile_model()

    def _maybe_compile_model(self):
        """Use torch.compile by default on Blackwell/GB200 for steadier decode overhead."""
        if os.getenv("NANOCHAT_DISABLE_COMPILE", "0") == "1":
            return
        if not torch.cuda.is_available() or not hasattr(torch, "compile"):
            return
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major < 10:
            return
        compile_kwargs = dict(mode="max-autotune-tiny", fullgraph=True, dynamic=True)
        if self.use_cuda_graphs:
            compile_kwargs["options"] = {"triton.cudagraphs": True}
        try:
            self.model = torch.compile(self.model, **compile_kwargs)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            self._compile_error = str(exc)

    def _expand_param(self, value, batch_size, default=None):
        if isinstance(value, (list, tuple)):
            assert len(value) == batch_size, f"Expected {batch_size} values, got {len(value)}"
            return list(value)
        if value is None:
            return [default] * batch_size
        return [value] * batch_size

    def _kv_cache_params(self, batch_size, seq_len):
        m = self.model.config
        return dict(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=m.n_kv_head,
            head_dim=m.n_embd // m.n_head,
            num_layers=m.n_layer,
            block_size=getattr(m, "kv_block_size", None),
            page_size=getattr(m, "kv_page_size", None),
        )

    def _decode_forward_step(self, ids, kv_cache, attention_mask=None, token_mask=None):
        """Decode forward path with optional persistent/graph gating."""
        if self.use_persistent_decode_kernel and self._persistent_decode_kernel is not None:
            try:
                return self._persistent_decode_kernel(self.model, ids, kv_cache, attention_mask=attention_mask, token_mask=token_mask)
            except NotImplementedError:
                # Surface stub errors so the caller knows a real kernel is required
                raise
            except Exception:
                if self._pd_runner is not None:
                    self._pd_runner.reset()
        # Persistent decode optimization: use dedicated stream and preallocated buffers
        if self.enable_persistent_decode and self._persistent_stream is not None:
            # Run decode in dedicated stream to allow better kernel pipelining
            # and overlap with CPU operations (e.g., sampling, token processing)
            with torch.cuda.stream(self._persistent_stream):
                logits = self.model.forward(ids, kv_cache=kv_cache, attention_mask=attention_mask, token_mask=token_mask)
            # Synchronize only if needed (caller will handle this in most cases)
            return logits
        else:
            # Standard decode path
            return self.model.forward(ids, kv_cache=kv_cache, attention_mask=attention_mask, token_mask=token_mask)

    def _reset_decode_graph(self):
        self._decode_graph = None
        self._graph_stream = None
        self._graph_static_ids = None
        self._graph_static_attention = None
        self._graph_static_token_mask = None
        self._graph_output = None
        self._graph_device = None
        self._graph_cache_gen = None
    
    def _get_or_create_persistent_buffer(self, name, shape, dtype, device):
        """Get or create a persistent buffer for decode operations (reduces allocations)."""
        if not self.enable_persistent_decode:
            return None
        
        buffer_attr = f"_persistent_{name}"
        buffer = getattr(self, buffer_attr, None)
        
        # Check if buffer exists and matches required shape
        if buffer is not None:
            # Convert devices to torch.device for comparison (cuda == cuda:0)
            device_obj = torch.device(device) if isinstance(device, str) else device
            buffer_device_obj = torch.device(buffer.device.type if hasattr(buffer.device, 'type') else buffer.device)
            if buffer.device.index is not None:
                buffer_device_obj = torch.device(f"{buffer.device.type}:{buffer.device.index}")
            
            shape_match = buffer.shape == tuple(shape)
            dtype_match = buffer.dtype == dtype
            # Compare device types (cuda == cuda:0 should match)
            device_match = (buffer_device_obj.type == device_obj.type and 
                          (buffer_device_obj.index == device_obj.index or device_obj.index is None))
            
            if shape_match and dtype_match and device_match:
                return buffer
            # Shape/dtype/device mismatch, need to reallocate
        
        # Allocate new buffer
        new_buffer = torch.empty(shape, dtype=dtype, device=device)
        setattr(self, buffer_attr, new_buffer)
        return new_buffer

    def _graph_shapes_match(self, ids, attention_mask, token_mask):
        if self._graph_static_ids is None:
            return False
        if self._graph_static_ids.shape != ids.shape:
            return False
        if self._graph_device is None or self._graph_device != ids.device:
            return False
        attn_present = attention_mask is not None
        token_present = token_mask is not None
        if attn_present != (self._graph_static_attention is not None):
            return False
        if token_present != (self._graph_static_token_mask is not None):
            return False
        return True

    def _capture_decode_graph(self, ids, kv_cache, attention_mask=None, token_mask=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA runtimes")
        device = ids.device
        torch.cuda.synchronize(device)
        self._decode_graph = torch.cuda.CUDAGraph()
        self._graph_stream = torch.cuda.Stream(device=device)
        self._graph_static_ids = ids.clone()
        self._graph_static_attention = attention_mask.clone() if attention_mask is not None else None
        self._graph_static_token_mask = token_mask.clone() if token_mask is not None else None
        with torch.cuda.graph(self._decode_graph, stream=self._graph_stream):
            self._graph_output = self._decode_forward_step(
                self._graph_static_ids,
                kv_cache,
                attention_mask=self._graph_static_attention,
                token_mask=self._graph_static_token_mask,
            )
        self._graph_device = device
        self._graph_cache_gen = getattr(kv_cache, "cache_gen", None)

    def _graph_decode(self, ids, kv_cache, attention_mask=None, token_mask=None):
        cache_gen = getattr(kv_cache, "cache_gen", None)
        if (not self._graph_shapes_match(ids, attention_mask, token_mask)) or (cache_gen != self._graph_cache_gen):
            self._capture_decode_graph(ids, kv_cache, attention_mask, token_mask)
        self._graph_static_ids.copy_(ids)
        if self._graph_static_attention is not None and attention_mask is not None:
            self._graph_static_attention.copy_(attention_mask)
        if self._graph_static_token_mask is not None and token_mask is not None:
            self._graph_static_token_mask.copy_(token_mask)
        self._decode_graph.replay()
        return self._graph_output

    def _execute_decode(self, ids, kv_cache, attention_mask=None, token_mask=None):
        if (
            self.use_cuda_graphs
            and not self._decode_graph_disabled
            and torch.cuda.is_available()
            and ids.is_cuda
            and ids.size(1) == 1
        ):
            try:
                return self._graph_decode(ids, kv_cache, attention_mask, token_mask)
            except Exception:
                self._reset_decode_graph()
        return self._decode_forward_step(ids, kv_cache, attention_mask, token_mask)

    def _attention_position_buffer(self, max_len, device):
        positions = self._attention_positions
        if positions is None or positions.device != device or positions.numel() < max_len:
            positions = torch.arange(max_len, device=device)
            self._attention_positions = positions
        return positions[:max_len]

    def _attention_mask_buffer(self, batch_size, max_len, device):
        mask = self._attention_mask
        if (
            mask is None
            or mask.device != device
            or mask.size(0) < batch_size
            or mask.size(1) < max_len
        ):
            mask = torch.empty((batch_size, max_len), dtype=torch.bool, device=device)
            self._attention_mask = mask
        return mask[:batch_size, :max_len]

    def _build_attention_mask(self, lengths, max_len=None):
        max_len = int(max_len if max_len is not None else lengths.max().item())
        mask = self._attention_mask_buffer(lengths.size(0), max_len, lengths.device)
        if max_len <= 0:
            return mask
        positions = self._attention_position_buffer(max_len, lengths.device)
        torch.lt(positions.unsqueeze(0), lengths.unsqueeze(1), out=mask)
        return mask

    def _batch_row_index_buffer(self, batch_size, device):
        indices = self._batch_row_indices
        if indices is None or indices.device != device or indices.numel() < batch_size:
            indices = torch.arange(batch_size, device=device)
            self._batch_row_indices = indices
        return indices[:batch_size]

    def _lengths_by_batch_buffer(self, lengths):
        count = lengths.numel()
        buffer = self._lengths_by_batch
        if (
            buffer is None
            or buffer.device != lengths.device
            or buffer.dtype != lengths.dtype
            or buffer.numel() < count
        ):
            buffer = torch.empty(count, dtype=lengths.dtype, device=lengths.device)
            self._lengths_by_batch = buffer
        view = buffer[:count].view_as(lengths)
        view.copy_(lengths)
        return view

    def _host_long_buffer(self, count):
        if self._token_column_host is None or self._token_column_host.numel() < count:
            self._token_column_host = torch.empty(count, dtype=torch.long, device="cpu")
        return self._token_column_host[:count]

    def _host_bool_buffer(self, count):
        if self._active_mask_host is None or self._active_mask_host.numel() < count:
            self._active_mask_host = torch.empty(count, dtype=torch.bool, device="cpu")
        return self._active_mask_host[:count]

    def _prompt_pack_buffers(self, batch_size, max_prompt_len, pad_id, pin_memory):
        lengths_host = self._prompt_lengths_host
        if (
            lengths_host is None
            or lengths_host.numel() < batch_size
            or lengths_host.is_pinned() != pin_memory
        ):
            lengths_host = torch.empty(batch_size, dtype=torch.long, pin_memory=pin_memory)
            self._prompt_lengths_host = lengths_host

        ids_host = self._prompt_ids_host
        if (
            ids_host is None
            or ids_host.size(0) < batch_size
            or ids_host.size(1) < max_prompt_len
            or ids_host.is_pinned() != pin_memory
        ):
            ids_host = torch.empty(
                (batch_size, max_prompt_len),
                dtype=torch.long,
                pin_memory=pin_memory,
            )
            self._prompt_ids_host = ids_host

        ids_view = ids_host[:batch_size, :max_prompt_len]
        ids_view.fill_(pad_id)
        return lengths_host[:batch_size], ids_view

    def _single_prompt_ids(self, tokens, device):
        token_count = len(tokens)
        pin_memory = device.type == "cuda"
        lengths_host, ids_host = self._prompt_pack_buffers(1, token_count, 0, pin_memory)
        lengths_host[0] = token_count
        row = ids_host[0]
        for idx, token in enumerate(tokens):
            row[idx] = int(token)
        if (
            self._prompt_ids_device is None
            or self._prompt_ids_device.device != device
            or self._prompt_ids_device.size(1) < token_count
        ):
            self._prompt_ids_device = torch.empty((1, token_count), dtype=torch.long, device=device)
        prompt_ids = self._prompt_ids_device[:, :token_count]
        prompt_ids.copy_(ids_host, non_blocking=pin_memory)
        return prompt_ids

    def _active_mask_buffer(self, count, device):
        if (
            self._active_mask_device is None
            or self._active_mask_device.device != device
            or self._active_mask_device.numel() < count
        ):
            self._active_mask_device = torch.empty(count, dtype=torch.bool, device=device)
        return self._active_mask_device[:count]

    def _active_index_buffers(self, count, device):
        if (
            self._active_indices_device is None
            or self._active_indices_device.device != device
            or self._active_indices_device.numel() < count
        ):
            self._active_indices_device = torch.empty(count, dtype=torch.long, device=device)
        active_indices = self._active_indices_device[:count]
        if device.type == "cpu":
            return active_indices, active_indices

        pin_memory = device.type == "cuda"
        if (
            self._active_indices_host is None
            or self._active_indices_host.numel() < count
            or self._active_indices_host.is_pinned() != pin_memory
        ):
            self._active_indices_host = torch.empty(
                count,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
        return active_indices, self._active_indices_host[:count]

    def _full_active_mask(self, count, device):
        active_mask = self._active_mask_buffer(count, device)
        active_mask.fill_(True)
        return active_mask

    def _fill_ids_buffer_from_tokens(self, ids_buf, token_column):
        count = len(token_column)
        target = ids_buf[:count, 0]
        if target.device.type == "cpu":
            for idx, token in enumerate(token_column):
                target[idx] = int(token)
        else:
            host = self._host_long_buffer(count)
            for idx, token in enumerate(token_column):
                host[idx] = int(token)
            target.copy_(host, non_blocking=True)
        return ids_buf[:count]

    def _ids_step_buffer_for(self, batch_size, device):
        if (
            self._ids_step_buffer is None
            or self._ids_step_buffer.device != device
            or self._ids_step_buffer.size(0) < batch_size
        ):
            self._ids_step_buffer = torch.empty((batch_size, 1), dtype=torch.long, device=device)
        return self._ids_step_buffer[:batch_size]

    def _active_mask_for_rows(
        self,
        row_states,
        generated_counts,
        row_max_tokens,
        device,
        return_active_rows=False,
        return_active_indices=False,
    ):
        count = len(row_states)
        active_mask = self._active_mask_buffer(count, device)
        active_rows = [] if return_active_rows else None
        active_indices = None
        active_index_target = None
        active_count = 0
        if return_active_indices:
            active_indices, active_index_target = self._active_index_buffers(count, device)
        if active_mask.device.type == "cpu":
            for idx, state in enumerate(row_states):
                is_active = (not state.completed) and generated_counts[idx] < row_max_tokens[idx]
                active_mask[idx] = is_active
                if is_active and active_rows is not None:
                    active_rows.append(idx)
                if is_active and active_index_target is not None:
                    active_index_target[active_count] = idx
                    active_count += 1
        else:
            host = self._host_bool_buffer(count)
            for idx, state in enumerate(row_states):
                is_active = (not state.completed) and generated_counts[idx] < row_max_tokens[idx]
                host[idx] = is_active
                if is_active and active_rows is not None:
                    active_rows.append(idx)
                if is_active and active_index_target is not None:
                    active_index_target[active_count] = idx
                    active_count += 1
            active_mask.copy_(host, non_blocking=True)
            if active_indices is not None:
                active_indices[:active_count].copy_(
                    active_index_target[:active_count],
                    non_blocking=True,
                )
        if active_indices is not None:
            active_indices = active_indices[:active_count]
        if return_active_rows and return_active_indices:
            return active_mask, active_rows, active_indices
        if return_active_rows:
            return active_mask, active_rows
        if return_active_indices:
            return active_mask, active_indices
        return active_mask

    def _sample_host_token_buffer(self, count, source_device):
        pin_memory = source_device.type == "cuda"
        if (
            self._sample_token_host_buffer is None
            or self._sample_token_host_buffer.numel() < count
            or self._sample_token_host_buffer.is_pinned() != pin_memory
        ):
            self._sample_token_host_buffer = torch.empty(
                count,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
        return self._sample_token_host_buffer[:count]

    def _sample_token_buffers(self, count, device):
        if (
            self._sample_token_device_buffer is None
            or self._sample_token_device_buffer.device != device
            or self._sample_token_device_buffer.numel() < count
        ):
            self._sample_token_device_buffer = torch.empty(count, dtype=torch.long, device=device)
        return self._sample_token_device_buffer[:count], self._sample_host_token_buffer(count, device)

    def _sample_long_buffer(self, name, shape, device):
        shape = tuple(int(dim) for dim in shape)
        buffer = getattr(self, name)
        if (
            buffer is None
            or buffer.device != device
            or buffer.dim() != len(shape)
            or any(buffer.size(dim) < size for dim, size in enumerate(shape))
        ):
            buffer = torch.empty(shape, dtype=torch.long, device=device)
            setattr(self, name, buffer)
        return buffer[tuple(slice(0, size) for size in shape)]

    def _sample_like_buffer(self, name, tensor):
        shape = tuple(int(dim) for dim in tensor.shape)
        numel = int(tensor.numel())
        buffer = getattr(self, name)
        if (
            buffer is None
            or buffer.device != tensor.device
            or buffer.dtype != tensor.dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, dtype=tensor.dtype, device=tensor.device)
            setattr(self, name, buffer)
        return buffer[:numel].view(shape)

    def _sample_active_logits_buffer_for(self, logits, row_count):
        vocab_size = logits.size(-1)
        buffer = self._sample_active_logits_buffer
        if (
            buffer is None
            or buffer.device != logits.device
            or buffer.dtype != logits.dtype
            or buffer.size(0) < row_count
            or buffer.size(1) != vocab_size
        ):
            buffer = torch.empty((row_count, vocab_size), dtype=logits.dtype, device=logits.device)
            self._sample_active_logits_buffer = buffer
        return buffer[:row_count, :vocab_size]

    def _sample_workspace(self, logits, top_k, temperature):
        batch_size = logits.size(0)
        workspace = {
            "out": self._sample_long_buffer("_sample_next_id_buffer", (batch_size, 1), logits.device),
        }
        if temperature == 0.0:
            workspace["max_values_out"] = self._sample_like_buffer("_sample_max_values_buffer", logits[:, :1])
        elif top_k is None:
            workspace["probs_out"] = self._sample_like_buffer("_sample_probs_buffer", logits)
        else:
            k = min(top_k, logits.size(-1))
            workspace["choice_out"] = self._sample_long_buffer(
                "_sample_choice_buffer",
                (batch_size, 1),
                logits.device,
            )
            workspace["topk_values_out"] = self._sample_like_buffer(
                "_sample_topk_values_buffer",
                logits[:, :k],
            )
            workspace["topk_indices_out"] = self._sample_long_buffer(
                "_sample_topk_indices_buffer",
                (batch_size, k),
                logits.device,
            )
            workspace["topk_probs_out"] = self._sample_like_buffer(
                "_sample_topk_probs_buffer",
                workspace["topk_values_out"],
            )
        return workspace

    def _token_tensor_to_list(self, token_tensor):
        flat_tokens = token_tensor.reshape(-1)
        host_tokens = self._sample_host_token_buffer(flat_tokens.numel(), flat_tokens.device)
        host_tokens.copy_(flat_tokens, non_blocking=flat_tokens.device.type == "cuda")
        return host_tokens.tolist()

    def _sample_batch_tokens(
        self,
        logits,
        rng,
        temperatures,
        top_ks,
        active_mask,
        pad_id,
        active_rows=None,
        active_indices=None,
        uniform_sampling=None,
    ):
        batch_size = logits.size(0)
        if active_rows is None:
            if active_indices is None:
                active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                return [pad_id] * batch_size
            if active_indices.device.type == "cpu":
                active_rows = active_indices.tolist()
            else:
                active_rows = self._token_tensor_to_list(active_indices)
        elif len(active_rows) == 0:
            return [pad_id] * batch_size
        full_active_rows = (
            len(active_rows) == batch_size
            and active_rows[0] == 0
            and active_rows[-1] == batch_size - 1
        )
        first_idx = active_rows[0]
        first_temp = temperatures[first_idx]
        first_top_k = top_ks[first_idx]
        if uniform_sampling is None:
            uniform_sampling = all(temperatures[idx] == first_temp and top_ks[idx] == first_top_k for idx in active_rows)
        if uniform_sampling:
            if full_active_rows:
                active_logits = logits
            else:
                if active_indices is None:
                    active_indices = torch.as_tensor(active_rows, device=logits.device, dtype=torch.long)
                active_logits = self._sample_active_logits_buffer_for(logits, len(active_rows))
                torch.index_select(logits, 0, active_indices, out=active_logits)
            next_ids = sample_next_token(
                active_logits,
                rng,
                temperature=first_temp,
                top_k=first_top_k,
                **self._sample_workspace(active_logits, first_top_k, first_temp),
            )
            next_tokens = self._token_tensor_to_list(next_ids[:, 0])
            if full_active_rows:
                return next_tokens
            sampled_tokens = [pad_id] * batch_size
            for idx, token in zip(active_rows, next_tokens):
                sampled_tokens[idx] = token
            return sampled_tokens

        sampled_device, sampled_host = self._sample_token_buffers(len(active_rows), logits.device)
        for sample_idx, idx in enumerate(active_rows):
            temp = temperatures[idx]
            top_k = top_ks[idx]
            row_logits = logits[idx:idx+1]
            next_id = sample_next_token(
                row_logits,
                rng,
                temperature=temp,
                top_k=top_k,
                **self._sample_workspace(row_logits, top_k, temp),
            )
            sampled_device[sample_idx].copy_(next_id[0, 0])
        sampled_host.copy_(sampled_device, non_blocking=sampled_device.device.type == "cuda")
        next_tokens = sampled_host.tolist()
        if full_active_rows:
            return next_tokens
        sampled_tokens = [pad_id] * batch_size
        for idx, token in zip(active_rows, next_tokens):
            sampled_tokens[idx] = token
        return sampled_tokens

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        kv_cache_prefill = KVCache(**self._kv_cache_params(batch_size=1, seq_len=len(tokens)))
        ids = self._single_prompt_ids(tokens, device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(
            logits,
            rng,
            temperature,
            top_k,
            **self._sample_workspace(logits, top_k, temperature),
        )  # (B, 1)
        sampled_tokens = self._token_tensor_to_list(next_ids[:, 0])

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(**self._kv_cache_params(batch_size=num_samples, seq_len=kv_length_hint))
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around
        self._reset_decode_graph()

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        ids_buf = self._ids_step_buffer_for(num_samples, device)

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        active_count = num_samples
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if active_count == 0:
                break

            # Get sampled tokens - either from prefill or from forward pass
            sampled_ids = None
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                if num_samples == 1:
                    sampled_ids = next_ids
                else:
                    ids_buf[:, 0].fill_(sampled_tokens[0])
                    sampled_ids = ids_buf
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self._execute_decode(ids, kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                next_ids = sample_next_token(
                    logits,
                    rng,
                    temperature,
                    top_k,
                    **self._sample_workspace(logits, top_k, temperature),
                )  # (B, 1)
                sampled_tokens = self._token_tensor_to_list(next_ids[:, 0])
                sampled_ids = next_ids

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            has_forced_tokens = False
            for i, state in enumerate(row_states):
                was_completed = state.completed
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                has_forced_tokens |= is_forced
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
                if not was_completed and state.completed:
                    active_count -= 1

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            if sampled_ids is not None and not has_forced_tokens:
                ids = sampled_ids
            else:
                ids = self._fill_ids_buffer_from_tokens(ids_buf, token_column)

    @torch.inference_mode()
    def generate_batched(self, prompt_tokens_batch, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Stream tokens for a batch of variable-length prompts using padded attention."""
        if not self.enable_batch_decode:
            raise ValueError("enable_batch_decode is False; cannot use batched generation")
        assert isinstance(prompt_tokens_batch, list) and len(prompt_tokens_batch) > 0 and all(isinstance(p, list) for p in prompt_tokens_batch), "expecting list of token lists"

        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        batch_size = len(prompt_tokens_batch)
        pad_id = self.tokenizer.get_bos_token_id()
        temps = self._expand_param(temperature, batch_size, temperature if temperature is not None else 1.0)
        top_ks = self._expand_param(top_k, batch_size, top_k)
        first_temp = temps[0]
        first_top_k = top_ks[0]
        uniform_sampling = all(temp == first_temp and top_k == first_top_k for temp, top_k in zip(temps, top_ks, strict=True))
        uniform_sampling_hint = True if uniform_sampling else None
        fallback_max = self.model.config.sequence_len if max_tokens is None else (max_tokens if isinstance(max_tokens, int) else self.model.config.sequence_len)
        max_tokens_list = self._expand_param(max_tokens, batch_size, fallback_max)
        row_max_tokens = [mt if mt is not None else fallback_max for mt in max_tokens_list]
        generated_counts = [0] * batch_size

        use_pinned_transfer = device.type == "cuda"
        prompt_lengths = [len(seq) for seq in prompt_tokens_batch]
        max_prompt_len = max(prompt_lengths)
        if max_prompt_len == 0:
            raise ValueError("prompt batch must contain at least one token")
        lengths_host, ids_host = self._prompt_pack_buffers(
            batch_size,
            max_prompt_len,
            pad_id,
            use_pinned_transfer,
        )
        for i, seq in enumerate(prompt_tokens_batch):
            seq_len = prompt_lengths[i]
            lengths_host[i] = seq_len
            if seq_len == 0:
                continue
            ids_host[i, :seq_len] = torch.as_tensor(seq, dtype=torch.long)
        lengths = lengths_host.to(device=device, non_blocking=use_pinned_transfer)
        ids = ids_host.to(device=device, non_blocking=use_pinned_transfer)

        attention_mask = self._build_attention_mask(lengths, max_prompt_len)

        kv_length_hint = max_prompt_len + int(max(row_max_tokens)) if row_max_tokens else max_prompt_len
        kv_cache_prefill = KVCache(**self._kv_cache_params(batch_size=batch_size, seq_len=max_prompt_len))
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill, attention_mask=attention_mask, token_mask=attention_mask)
        last_indices = (lengths - 1).clamp(min=0)
        batch_rows = self._batch_row_index_buffer(batch_size, device)
        logits = logits[batch_rows, last_indices, :]

        active_mask = self._full_active_mask(batch_size, device)
        sampled_tokens = self._sample_batch_tokens(
            logits,
            rng,
            temps,
            top_ks,
            active_mask,
            pad_id,
            active_rows=range(batch_size),
            uniform_sampling=uniform_sampling_hint,
        )

        kv_cache_decode = KVCache(**self._kv_cache_params(batch_size=batch_size, seq_len=kv_length_hint))
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill
        self._reset_decode_graph()

        row_states = [RowState(tokens.copy()) for tokens in prompt_tokens_batch]
        lengths_by_batch = self._lengths_by_batch_buffer(lengths)
        lengths_by_row = prompt_lengths.copy()
        current_decode_len = max_prompt_len
        ids_buf = self._ids_step_buffer_for(batch_size, device)

        # Special tokens for control flow
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        num_generated = 0
        max_total_steps = max(row_max_tokens) if row_max_tokens else 0
        first_iteration = True
        token_column = None
        active_count = sum(1 for limit in row_max_tokens if limit > 0)

        while True:
            if active_count == 0:
                break
            if max_total_steps and num_generated >= max_total_steps:
                break

            if first_iteration:
                current_tokens = sampled_tokens
                first_iteration = False
            else:
                step_ids = self._fill_ids_buffer_from_tokens(ids_buf, token_column)
                active_mask, active_rows, active_indices = self._active_mask_for_rows(
                    row_states,
                    generated_counts,
                    row_max_tokens,
                    device,
                    return_active_rows=True,
                    return_active_indices=True,
                )
                step_token_mask = active_mask.unsqueeze(1)
                lengths_by_batch.add_(step_token_mask[:, 0])
                if len(active_rows) == batch_size:
                    for row_idx in active_rows:
                        lengths_by_row[row_idx] += 1
                    current_decode_len += 1
                else:
                    for row_idx in active_rows:
                        row_len = lengths_by_row[row_idx] + 1
                        lengths_by_row[row_idx] = row_len
                        if row_len > current_decode_len:
                            current_decode_len = row_len
                attn_mask = self._build_attention_mask(lengths_by_batch, current_decode_len)
                logits = self._execute_decode(step_ids, kv_cache_decode, attention_mask=attn_mask, token_mask=step_token_mask)
                logits = logits[:, -1, :]
                sampled_tokens = self._sample_batch_tokens(
                    logits,
                    rng,
                    temps,
                    top_ks,
                    active_mask,
                    pad_id,
                    active_rows=active_rows,
                    active_indices=active_indices,
                    uniform_sampling=uniform_sampling_hint,
                )
                current_tokens = sampled_tokens

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                if state.completed or generated_counts[i] >= row_max_tokens[i]:
                    token_masks.append(-1)
                    token_column.append(pad_id)
                    continue

                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else current_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                generated_counts[i] += 1

                if next_token == assistant_end or next_token == bos or generated_counts[i] >= row_max_tokens[i]:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
                if state.completed:
                    active_count -= 1

            yield token_column, token_masks
            num_generated += 1
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        remaining = num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                        remaining -= 1
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if remaining == 0:
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with autocast_ctx:
        for token_column, _token_masks in stream:
            token = token_column[0] # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
