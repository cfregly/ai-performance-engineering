"""Disaggregated Inference with Speculative Decoding Demo.

Chapter 15: Disaggregated Inference

This file demonstrates disaggregated prefill/decode architecture, which
naturally combines with speculative decoding for improved throughput.

NOTE: Speculative decoding is covered in depth in Chapter 18. This file
provides a working integration as a forward reference. For the core
speculative decoding algorithms and optimizations, see:
- ch18/optimized_speculative_decode.py (EAGLE-style draft + verify)
- ch18/optimized_vllm_decode_graphs.py (CUDA graph optimization)
- ch18/run_vllm_decoder.py (production vLLM integration)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from core.utils.compile_utils import compile_callable, maybe_nested_compile_region
from core.utils.architecture_runtime import (
    get_arch_config,
)
from core.benchmark.gpu_requirements import require_min_gpus

_ARCH_CFG = get_arch_config()

"""disaggregated_inference_multigpu.py
Chapter 15: Disaggregated Inference Architectures

Simulated disaggregated prefill-decode benchmarking on Blackwell clusters."""

import time
import numpy as np
import json


@maybe_nested_compile_region
def _run_attn(attn: nn.Module, x: torch.Tensor, kv_state=None):
    if kv_state is None:
        return attn(x)
    return attn(x, kv_state=kv_state)


@maybe_nested_compile_region
def _run_ffn(ffn: nn.Module, x: torch.Tensor):
    return ffn(x)

def is_cuda_usable() -> bool:
    """Return True when a CUDA device is present and supported (treat sm121 as valid)."""
    if not torch.cuda.is_available():
        return False
    try:
        device_idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_idx)
        if major < 8:
            return False
        if major > 12:
            return False
        if major == 12 and minor > 1:
            return False
        return True
    except Exception:
        return False


HAS_USABLE_CUDA = is_cuda_usable()

if not HAS_USABLE_CUDA:
    raise RuntimeError(
        "Chapter 15 requires an NVIDIA GPU with compute capability up to sm121. "
        "No supported GPU was detected."
    )

PREFERRED_SDP_BACKENDS: Tuple[SDPBackend, ...] = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
)

# Blackwell (SM12x) may not ship compatible Flash/CUTLASS kernels in all builds.
if HAS_USABLE_CUDA:
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        major = None
    if major is not None and major >= 12:
        PREFERRED_SDP_BACKENDS = (SDPBackend.MATH,)
        torch.backends.cuda.enable_flash_sdp(False)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_math_sdp(True)  # type: ignore[attr-defined]


def get_sdpa_context():
    """Return an SDPA backend context for the configured backend set."""
    return sdpa_kernel(list(PREFERRED_SDP_BACKENDS))


def get_compute_dtype() -> torch.dtype:
    """Select the preferred compute dtype for inference."""
    if HAS_USABLE_CUDA:
        return torch.bfloat16
    return torch.float32


class ParallelismStrategy(Enum):
    TENSOR = "tensor"
    PIPELINE = "pipeline" 
    EXPERT = "expert"
    DATA = "data"
    CONTEXT = "context"

@dataclass
class InferenceConfig:
    """Configuration for multi-node inference."""
    model_size: int = 7_000_000_000  # 7B parameters
    num_gpus: int = None  # Will be set to actual GPU count
    batch_size: int = 32
    sequence_length: int = 2048
    num_experts: int = 8
    top_k_experts: int = 2
    capacity_factor: float = 1.2
    use_speculative: bool = True
    use_disaggregated: bool = True
    speculative_method: str = "eagle-2"
    num_speculative_tokens: int = 4
    moe_activation_precision: Optional[str] = "fp8"
    guided_decoding_backend: str = "tensorrt-llm-guided_json"
    compile_prefill: bool = False
    compile_decode: bool = False
    nic_bandwidth_gbps: float = 800.0
    
    def __post_init__(self):
        """Initialize num_gpus to actual available GPUs if not set."""
        if self.num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("Chapter 15 requires at least one supported GPU.")
        self.num_gpus = int(self.num_gpus)
        if self.num_gpus < 1:
            raise RuntimeError("num_gpus must be >= 1")
        if self.num_gpus > available_gpus:
            raise RuntimeError(
                f"num_gpus={self.num_gpus} exceeds available GPUs ({available_gpus})."
            )
        if self.use_disaggregated and self.num_gpus < 2:
            raise RuntimeError("Disaggregated inference requires >= 2 GPUs.")
        self.num_speculative_tokens = int(self.num_speculative_tokens)
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if self.moe_activation_precision:
            self.moe_activation_precision = self.moe_activation_precision.lower()


class PrecisionManager:
    """Manage precision compression (FP8/NVFP4 simulation)."""

    def __init__(self, mode: Optional[str], device: torch.device):
        self.mode = (mode or "fp8").lower() if mode else None
        self.device = device
        self.fp8_dtype = getattr(torch, "float8_e4m3fn", torch.float16)
        # NVFP4 currently maps to the closest available dtype in public PyTorch builds.
        self.fp4_dtype = getattr(torch, "float8_e4m3fn", torch.float16)
        self.restore_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode is None or not isinstance(tensor, torch.Tensor):
            return tensor
        tensor = tensor.to(self.device)
        target_dtype = None
        if self.mode == "fp8":
            target_dtype = self.fp8_dtype
        elif self.mode == "nvfp4":
            target_dtype = self.fp4_dtype

        if target_dtype is None:
            return tensor

        try:
            return tensor.to(target_dtype)
        except (TypeError, RuntimeError):
            return tensor

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        try:
            return tensor.to(self.restore_dtype)
        except (TypeError, RuntimeError):
            return tensor


class ScaledDotProductAttentionLayer(nn.Module):
    """Attention layer that prefers PyTorch SDPA fused backends."""

    def __init__(self, embed_dim: int, num_heads: int, device: torch.device, compute_dtype: torch.dtype):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.compute_dtype = compute_dtype

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to(device=device, dtype=compute_dtype)
        self._k_cat_buffer: Optional[torch.Tensor] = None
        self._v_cat_buffer: Optional[torch.Tensor] = None

    def _concat_kv_cache(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_len = past_k.size(2)
        total_len = past_len + k.size(2)
        capacity = total_len
        current_capacity = self._k_cat_buffer.size(2) if self._k_cat_buffer is not None else 0
        if current_capacity > 0:
            capacity = max(capacity, current_capacity)
        needs_buffer = (
            self._k_cat_buffer is None
            or self._v_cat_buffer is None
            or self._k_cat_buffer.size(0) != k.size(0)
            or self._k_cat_buffer.size(1) != k.size(1)
            or self._k_cat_buffer.size(2) < total_len
            or self._k_cat_buffer.size(3) != k.size(3)
            or self._k_cat_buffer.device != k.device
            or self._k_cat_buffer.dtype != k.dtype
        )
        if needs_buffer:
            capacity = max(total_len, max(1, current_capacity) * 2)
            self._k_cat_buffer = torch.empty(
                k.size(0),
                k.size(1),
                capacity,
                k.size(3),
                device=k.device,
                dtype=k.dtype,
            )
            self._v_cat_buffer = torch.empty_like(self._k_cat_buffer)

        k_out = self._k_cat_buffer[:, :, :total_len, :]
        v_out = self._v_cat_buffer[:, :, :total_len, :]
        k_out[:, :, :past_len, :].copy_(past_k)
        v_out[:, :, :past_len, :].copy_(past_v)
        k_out[:, :, past_len:total_len, :].copy_(k)
        v_out[:, :, past_len:total_len, :].copy_(v)
        return k_out, v_out

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if x.dtype != self.compute_dtype:
            raise RuntimeError("ScaledDotProductAttentionLayer expects inputs in compute_dtype")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_state is not None and all(isinstance(t, torch.Tensor) for t in kv_state):
            past_k, past_v = kv_state
            if past_k is not None and past_v is not None:
                k, v = self._concat_kv_cache(past_k, past_v, k, v)

        sdpa_ctx = get_sdpa_context()
        with sdpa_ctx:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_out)
        return output, k, v


class PrefillKernel(nn.Module):
    """Prefill compute graph that returns updated states and KV cache."""

    def __init__(self, attention_layers: nn.ModuleList, ffn_layers: nn.ModuleList):
        super().__init__()
        self.attention_layers = attention_layers
        self.ffn_layers = ffn_layers

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        layer_count = min(len(self.attention_layers), len(self.ffn_layers))
        key_states: List[torch.Tensor] = [x] * layer_count
        value_states: List[torch.Tensor] = [x] * layer_count
        for idx, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            attn_out, key_state, value_state = _run_attn(attn, x)
            key_states[idx] = key_state
            value_states[idx] = value_state
            x = _run_ffn(ffn, attn_out)
        return x, tuple(key_states), tuple(value_states)


class DecodeKernel(nn.Module):
    """Decode compute graph that consumes cached states and emits logits."""

    def __init__(self, attention_layers: nn.ModuleList, ffn_layers: nn.ModuleList, lm_head: nn.Module):
        super().__init__()
        self.attention_layers = attention_layers
        self.ffn_layers = ffn_layers
        self.lm_head = lm_head

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        kv_state: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        layer_count = min(len(self.attention_layers), len(self.ffn_layers))
        key_states: List[torch.Tensor] = [x] * layer_count
        value_states: List[torch.Tensor] = [x] * layer_count

        for idx, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            past_state = kv_state[idx] if kv_state and idx < len(kv_state) else None
            attn_out, key_state, value_state = _run_attn(attn, x, kv_state=past_state)
            key_states[idx] = key_state
            value_states[idx] = value_state
            x = _run_ffn(ffn, attn_out)

        logits = self.lm_head(x[:, -1, :])
        return logits, tuple(key_states), tuple(value_states)


class GuidedDecoder:
    """Simulate structured/guided decoding backends."""

    def __init__(self, backend: str = "tensorrt-llm-guided_json"):
        self.backend = backend
        self.schema: Optional[Dict] = None
        self._schema_keys: Optional[Tuple[str, ...]] = None
        self._cached_backend = ""
        self._simulated_latency_ms = 0.0
        self._refresh_backend_cache()
        self._compiled = False

    def _refresh_backend_cache(self):
        self._cached_backend = self.backend
        backend_lower = self.backend.lower()
        # Simulated latencies: TensorRT-LLM guided_json tends to be faster than generic fallbacks.
        self._simulated_latency_ms = (
            3.5 if "tensorrt" in backend_lower or "guided_json" in backend_lower else 7.0
        )

    def load_schema(self, schema: Dict):
        self.schema = schema
        self._schema_keys = tuple(schema.keys())
        self._compiled = True

    def compile_backend(self):
        """Simulate backend compilation."""
        if self.backend != self._cached_backend:
            self._refresh_backend_cache()
        self._compiled = True

    def generate(self, prompt: str) -> Dict:
        if not self._compiled:
            raise RuntimeError("GuidedDecoder requires compile_backend() or load_schema() before generation.")
        if self.backend != self._cached_backend:
            self._refresh_backend_cache()

        payload = {
            "backend": self.backend,
            "prompt_prefix": prompt[:32],
            "latency_ms": self._simulated_latency_ms,
        }
        if self._schema_keys:
            payload["schema_keys"] = self._schema_keys
        return payload

class DisaggregatedInferenceSystem:
    """Demonstrates disaggregated prefill-decode architecture."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.prefill_workers = []
        self.decode_workers = []
        self.kv_cache = {}
        
    def setup_prefill_workers(self):
        """Initialize dedicated prefill GPU workers."""
        print("Setting up prefill workers...")
        num_prefill_workers = max(1, self.config.num_gpus // 2)
        for i in range(num_prefill_workers):
            gpu_id = i % self.config.num_gpus  # Use modulo to avoid exceeding GPU count
            worker = PrefillWorker(
                worker_id=i,
                gpu_id=gpu_id,
                config=self.config
            )
            self.prefill_workers.append(worker)
            
    def setup_decode_workers(self):
        """Initialize dedicated decode GPU workers."""
        print("Setting up decode workers...")
        num_decode_workers = max(1, self.config.num_gpus // 2)
        for i in range(num_decode_workers):
            gpu_id = (i + self.config.num_gpus // 2) % self.config.num_gpus  # Use modulo to avoid exceeding GPU count
            worker = DecodeWorker(
                worker_id=i,
                gpu_id=gpu_id,
                config=self.config
            )
            self.decode_workers.append(worker)
            
    def process_request(self, prompt: str) -> str:
        """Process a request using disaggregated architecture."""
        # Phase 1: Prefill on dedicated workers
        kv_cache = self.prefill_phase(prompt)
        
        # Phase 2: Transfer KV cache to decode workers
        self.transfer_kv_cache(kv_cache)
        
        # Phase 3: Decode on dedicated workers
        response = self.decode_phase()
        
        return response
        
    def prefill_phase(self, prompt: str) -> Dict:
        """Process prompt and generate KV cache."""
        print(f"Prefill phase: Processing prompt of length {len(prompt)}")
        
        # Simulate prefill computation
        start_time = time.perf_counter()
        
        # Distribute across prefill workers
        kv_cache = {}
        for worker in self.prefill_workers:
            worker_kv = worker.process_prompt(prompt)
            kv_cache.update(worker_kv)
            
        prefill_time = time.perf_counter() - start_time
        print(f"Prefill completed in {prefill_time:.3f}s")
        
        return kv_cache
        
    def transfer_kv_cache(self, kv_cache: Dict):
        """Transfer KV cache from prefill to decode workers."""
        print("Transferring KV cache to decode workers...")

        total_bytes = 0
        for value in kv_cache.values():
            if isinstance(value, torch.Tensor):
                total_bytes += value.element_size() * value.numel()
        if total_bytes == 0:
            total_bytes = len(str(kv_cache))

        bytes_per_second = (self.config.nic_bandwidth_gbps * 1_000_000_000) / 8.0
        transfer_time = min(total_bytes / max(bytes_per_second, 1), 0.5)

        time.sleep(transfer_time)
        
        # Distribute KV cache to decode workers
        for worker in self.decode_workers:
            worker.load_kv_cache(kv_cache)
            
    def decode_phase(self) -> str:
        """Generate response using decode workers."""
        print("Decode phase: Generating response tokens...")
        
        start_time = time.perf_counter()
        response_tokens = []
        
        # Generate tokens autoregressively
        for _i in range(100):  # Generate up to 100 tokens
            token = self.decode_workers[0].generate_next_token()
            response_tokens.append(token)
            
            if token == "<EOS>":
                break
                
        decode_time = time.perf_counter() - start_time
        print(f"Decode completed in {decode_time:.3f}s")
        
        return " ".join(response_tokens)

class PrefillWorker:
    """Dedicated worker for prompt processing."""
    
    def __init__(self, worker_id: int, gpu_id: int, config: InferenceConfig):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config

        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU id {gpu_id} is out of range for available devices.")
        self.device = torch.device(f"cuda:{gpu_id}")

        self.compute_dtype = get_compute_dtype()
        
        # Initialize model components
        self.attention_layers = self._create_attention_layers()
        self.ffn_layers = self._create_ffn_layers()
        self.prefill_kernel = PrefillKernel(self.attention_layers, self.ffn_layers)
        self.prefill_kernel_compiled: Optional[torch.nn.Module] = None
        if self.config.compile_prefill:
            self.prefill_kernel_compiled = compile_callable(
                self.prefill_kernel,
                mode="reduce-overhead",
                dynamic=True,
                nested_compile_region=True,
                error_on_graph_break=True,
            )
        
    def _create_attention_layers(self) -> nn.ModuleList:
        """Create attention layers for prefill."""
        layers = nn.ModuleList()
        for _ in range(6):  # 6 attention layers
            layer = ScaledDotProductAttentionLayer(
                embed_dim=4096,
                num_heads=32,
                device=self.device,
                compute_dtype=self.compute_dtype,
            )
            layers.append(layer)
        return layers
        
    def _create_ffn_layers(self) -> nn.ModuleList:
        """Create feed-forward layers for prefill."""
        layers = nn.ModuleList()
        for _ in range(6):  # 6 FFN layers
            layer = nn.Sequential(
                nn.Linear(4096, 16384, bias=False),
                nn.GELU(),
                nn.Linear(16384, 4096, bias=False),
            )
            layer = layer.to(self.device, dtype=self.compute_dtype)
            layers.append(layer)
        return layers
        
    @torch.inference_mode()
    def process_prompt(self, prompt: str) -> Dict:
        """Process prompt and generate KV cache."""
        tokens = self._tokenize(prompt)
        
        embed_dim = 4096
        x = torch.randn(
            1,
            len(tokens),
            embed_dim,
            device=self.device,
            dtype=self.compute_dtype,
        )
        
        kernel = self.prefill_kernel_compiled or self.prefill_kernel
        _, key_states, value_states = kernel(x)

        kv_cache: Dict[str, torch.Tensor] = {}
        for idx, (key_state, value_state) in enumerate(zip(key_states, value_states)):
            kv_cache[f"layer_{idx}_k"] = key_state
            kv_cache[f"layer_{idx}_v"] = value_state
            
        return kv_cache
        
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization for demonstration."""
        return [ord(c) % 1000 for c in text[:self.config.sequence_length]]


class DecodeWorker:
    """Dedicated worker for token generation."""
    
    def __init__(self, worker_id: int, gpu_id: int, config: InferenceConfig):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config
        
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU id {gpu_id} is out of range for available devices.")
        self.device = torch.device(f"cuda:{gpu_id}")

        self.compute_dtype = get_compute_dtype()
            
        self.kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = tuple()
        
        # Initialize model components
        self.attention_layers = self._create_attention_layers()
        self.ffn_layers = self._create_ffn_layers()
        self.vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "<EOS>"]
        self.vocab_size = len(self.vocab)
        self.token_embedding = nn.Embedding(self.vocab_size, 4096).to(self.device, dtype=self.compute_dtype)
        self.lm_head = nn.Linear(4096, self.vocab_size, bias=False).to(self.device, dtype=self.compute_dtype)

        self.decode_kernel = DecodeKernel(self.attention_layers, self.ffn_layers, self.lm_head)
        self.decode_kernel_compiled: Optional[torch.nn.Module] = None
        if self.config.compile_decode:
            self.decode_kernel_compiled = compile_callable(
                self.decode_kernel,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
                nested_compile_region=True,
                error_on_graph_break=True,
            )

        # Seed with a random token for the first decode step.
        seed_token = torch.randint(0, self.vocab_size, (1,), device=self.device)
        self._last_token_id = seed_token
        self._sample_logits = torch.empty(self.vocab_size, dtype=torch.float32, device=self.device)
        self._sample_probs = torch.empty_like(self._sample_logits)
        self._sample_token = torch.empty(1, dtype=torch.long, device=self.device)
        self._sample_token_host = torch.empty(1, dtype=torch.long, device="cpu", pin_memory=True)
        
    def _create_attention_layers(self) -> nn.ModuleList:
        """Create attention layers for decode."""
        layers = nn.ModuleList()
        for _ in range(6):  # 6 attention layers
            layer = ScaledDotProductAttentionLayer(
                embed_dim=4096,
                num_heads=32,
                device=self.device,
                compute_dtype=self.compute_dtype,
            )
            layers.append(layer)
        return layers
        
    def _create_ffn_layers(self) -> nn.ModuleList:
        """Create feed-forward layers for decode."""
        layers = nn.ModuleList()
        for _ in range(6):  # 6 FFN layers
            layer = nn.Sequential(
                nn.Linear(4096, 16384, bias=False),
                nn.GELU(),
                nn.Linear(16384, 4096, bias=False),
            )
            layer = layer.to(self.device, dtype=self.compute_dtype)
            layers.append(layer)
        return layers
        
    def load_kv_cache(self, kv_cache: Dict):
        """Load KV cache from prefill phase."""
        kv_state: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx in range(len(self.attention_layers)):
            key_tensor = kv_cache.get(f"layer_{idx}_k")
            value_tensor = kv_cache.get(f"layer_{idx}_v")
            if key_tensor is None or value_tensor is None:
                continue
            kv_state.append(
                (
                    key_tensor.to(self.device, dtype=self.compute_dtype),
                    value_tensor.to(self.device, dtype=self.compute_dtype),
                )
            )
        self.kv_cache = tuple(kv_state)
        
    @torch.inference_mode()
    def generate_next_token(self) -> str:
        """Generate next token using cached KV values."""
        if not self.kv_cache:
            return "<EOS>"

        token_embed = self.token_embedding(self._last_token_id).unsqueeze(1)

        kernel = self.decode_kernel_compiled or self.decode_kernel
        logits, key_states, value_states = kernel(token_embed, self.kv_cache)
        self.kv_cache = tuple((k, v) for k, v in zip(key_states, value_states))

        self._sample_logits.copy_(logits[0])
        torch.softmax(self._sample_logits, dim=-1, out=self._sample_probs)
        torch.multinomial(self._sample_probs, num_samples=1, out=self._sample_token)
        self._last_token_id.copy_(self._sample_token)

        self._sample_token_host.copy_(self._sample_token)
        token_index = int(self._sample_token_host[0]) % len(self.vocab)
        return self.vocab[token_index]

class MoERouter:
    """Demonstrates MoE routing, compression, and load balancing."""
    
    def __init__(self, num_experts: int, top_k: int, capacity_factor: float, precision_mode: Optional[str] = None):
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.expert_loads = {i: 0 for i in range(num_experts)}
        self.expert_capacities = {i: int(32 * capacity_factor) for i in range(num_experts)}
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.precision_manager = PrecisionManager(precision_mode, device) if precision_mode else None
        self.compression_stats: Optional[Dict[str, float]] = None
        
    def route_tokens(self, tokens: List[int]) -> Dict[int, List[int]]:
        """Route tokens to experts using load balancing."""
        expert_assignments = {i: [] for i in range(self.num_experts)}

        if self.precision_manager is not None and tokens:
            activation_shape = (len(tokens), 256)
            activations = torch.randn(
                *activation_shape,
                device=self.precision_manager.device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            compressed = self.precision_manager.compress(activations)
            restored = self.precision_manager.restore(compressed)
            try:
                compression_ratio = restored.element_size() / max(compressed.element_size(), 1)
            except AttributeError:
                compression_ratio = 1.0
            self.compression_stats = {
                "precision_mode": self.precision_manager.mode,
                "activation_bytes_ratio": compression_ratio,
            }
            del activations, compressed, restored
        
        for token in tokens:
            expert_scores = self._get_expert_scores(token)
            top_experts = self._rank_top_experts(expert_scores)
            
            routed = False
            for expert_id in top_experts:
                if self.expert_loads[expert_id] < self.expert_capacities[expert_id]:
                    expert_assignments[expert_id].append(token)
                    self.expert_loads[expert_id] += 1
                    routed = True
                    break
                    
            if not routed:
                for expert_id in range(self.num_experts):
                    if self.expert_loads[expert_id] < self.expert_capacities[expert_id]:
                        expert_assignments[expert_id].append(token)
                        self.expert_loads[expert_id] += 1
                        break
                        
        return expert_assignments
        
    def _get_expert_scores(self, token: int) -> np.ndarray:
        """Get expert preference scores for a token."""
        return np.random.random(self.num_experts)

    def _rank_top_experts(self, expert_scores: np.ndarray) -> List[int]:
        """Return expert ids sorted by descending score without a full Python sort."""
        expert_count = int(expert_scores.shape[0])
        ranked_count = min(max(int(self.top_k), 0), expert_count)
        if ranked_count == 0:
            return []
        if ranked_count == expert_count:
            ranked = np.argsort(expert_scores)[::-1]
        else:
            split_index = expert_count - ranked_count
            candidate_indices = np.argpartition(expert_scores, split_index)[split_index:]
            ranked = candidate_indices[np.argsort(expert_scores[candidate_indices])[::-1]]
        return [int(expert_id) for expert_id in ranked]
        
    def get_load_balance_metrics(self) -> Dict:
        """Get load balancing metrics."""
        load_count = len(self.expert_loads)
        if load_count == 0:
            mean_load = std_load = max_load = min_load = 0.0
        else:
            load_total = 0.0
            max_load = float("-inf")
            min_load = float("inf")
            for load in self.expert_loads.values():
                load_value = float(load)
                load_total += load_value
                max_load = max(max_load, load_value)
                min_load = min(min_load, load_value)
            mean_load = load_total / load_count
            variance = 0.0
            for load in self.expert_loads.values():
                delta = float(load) - mean_load
                variance += delta * delta
            std_load = float(np.sqrt(variance / load_count))
        metrics = {
            "mean_load": mean_load,
            "std_load": std_load,
            "max_load": max_load,
            "min_load": min_load,
            "load_imbalance": max_load - min_load,
        }
        if self.compression_stats:
            metrics.update(self.compression_stats)
        return metrics

class SpeculativeDecoder:
    """Demonstrates speculative decoding techniques."""

    PROFILES = {
        "medusa-1": {"acceptance": 0.72, "draft": 3, "speedup": (2.1, 2.3)},
        "medusa-2": {"acceptance": 0.75, "draft": 4, "speedup": (2.3, 3.6)},
        "eagle-2": {"acceptance": 0.78, "draft": 4, "speedup": (1.8, 2.8)},
        "eagle-3": {"acceptance": 0.80, "draft": 4, "speedup": (2.5, 6.5)},
    }
    
    def __init__(self, method: str = "eagle-2", num_draft: int = 4, draft_model_size: int = 1_000_000_000):
        self.method = method.lower()
        self.draft_model_size = draft_model_size
        profile = self.PROFILES.get(self.method, {"acceptance": 0.7, "draft": num_draft, "speedup": (1.2, 1.6)})
        self.acceptance_rate = profile["acceptance"]
        self.num_draft = max(1, num_draft or profile["draft"])
        self.speedup_range = profile["speedup"]
        
    def expected_speedup(self) -> float:
        return float(sum(self.speedup_range) / len(self.speedup_range))

    def speculative_decode(self, prompt: str, target_tokens: int = 10) -> List[str]:
        """Generate tokens using speculative decoding."""
        print(f"Speculative decoding ({self.method}): target {target_tokens} tokens | acceptance≈{self.acceptance_rate:.2f}")
        
        generated_tokens = []
        current_position = 0
        
        while len(generated_tokens) < target_tokens:
            draft_tokens = self._draft_phase(prompt, generated_tokens)
            accepted_tokens = self._verify_phase(prompt, generated_tokens, draft_tokens)
            
            generated_tokens.extend(accepted_tokens)
            current_position += len(accepted_tokens)
            
            if len(accepted_tokens) == 0:
                single_token = self._generate_single_token(prompt, generated_tokens)
                generated_tokens.append(single_token)
                current_position += 1
                
        return generated_tokens[:target_tokens]
        
    def _draft_phase(self, prompt: str, context: List[str]) -> List[str]:
        """Generate draft tokens using smaller model."""
        print(f"Draft phase: Generating {self.num_draft} candidate tokens")
        
        vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "with"]
        draft_tokens = np.random.choice(vocab, self.num_draft).tolist()
        
        return draft_tokens
        
    def _verify_phase(self, prompt: str, context: List[str], draft_tokens: List[str]) -> List[str]:
        """Verify draft tokens against target model."""
        print(f"Verify phase: Checking {len(draft_tokens)} draft tokens")
        
        accepted_tokens = []
        
        for draft_token in draft_tokens:
            if np.random.random() < self.acceptance_rate:
                accepted_tokens.append(draft_token)
            else:
                break
                
        print(f"Accepted {len(accepted_tokens)} out of {len(draft_tokens)} draft tokens")
        return accepted_tokens
        
    def _generate_single_token(self, prompt: str, context: List[str]) -> str:
        """Generate single token using target model."""
        vocab = ["the", "a", "is", "was", "in", "on", "at", "to", "for", "with"]
        return np.random.choice(vocab)

class ParallelismManager:
    """Manages different parallelism strategies."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.strategy = None
        
    def setup_tensor_parallelism(self):
        """Setup tensor parallelism across GPUs."""
        print("Setting up tensor parallelism...")
        
        # Split model layers across GPUs
        num_layers_per_gpu = 6 // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_layer = gpu_id * num_layers_per_gpu
            end_layer = start_layer + num_layers_per_gpu
            
            print(f"GPU {gpu_id}: Layers {start_layer}-{end_layer}")
            
    def setup_pipeline_parallelism(self):
        """Setup pipeline parallelism across GPUs."""
        print("Setting up pipeline parallelism...")
        
        # Each GPU handles different stages
        stages_per_gpu = 1
        
        for gpu_id in range(self.config.num_gpus):
            stage_start = gpu_id * stages_per_gpu
            stage_end = stage_start + stages_per_gpu
            
            print(f"GPU {gpu_id}: Pipeline stages {stage_start}-{stage_end}")
            
    def setup_expert_parallelism(self):
        """Setup expert parallelism for MoE."""
        print("Setting up expert parallelism...")
        
        experts_per_gpu = self.config.num_experts // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_expert = gpu_id * experts_per_gpu
            end_expert = start_expert + experts_per_gpu
            
            print(f"GPU {gpu_id}: Experts {start_expert}-{end_expert}")
            
    def setup_data_parallelism(self):
        """Setup data parallelism."""
        print("Setting up data parallelism...")
        
        # Each GPU gets a full model replica
        for gpu_id in range(self.config.num_gpus):
            print(f"GPU {gpu_id}: Full model replica")
            
    def setup_context_parallelism(self):
        """Setup context parallelism for long sequences."""
        print("Setting up context parallelism...")
        
        # Split sequence across GPUs
        tokens_per_gpu = self.config.sequence_length // self.config.num_gpus
        
        for gpu_id in range(self.config.num_gpus):
            start_token = gpu_id * tokens_per_gpu
            end_token = start_token + tokens_per_gpu
            
            print(f"GPU {gpu_id}: Tokens {start_token}-{end_token}")

def benchmark_inference_system():
    """Benchmark the disaggregated inference system."""
    print("=== Multi-Node Inference Benchmark ===\n")
    require_min_gpus(2)
    
    # Configuration - will automatically detect available GPUs
    config = InferenceConfig(
        model_size=7_000_000_000,
        batch_size=32,
        sequence_length=2048,
        num_experts=8,
        top_k_experts=2,
        capacity_factor=1.2,
        use_speculative=True,
        use_disaggregated=True,
        speculative_method="eagle-3",
        num_speculative_tokens=4,
        moe_activation_precision="fp8",
        guided_decoding_backend="tensorrt-llm-guided_json",
    )
    
    print(f"Using {config.num_gpus} GPU(s) for inference")
    
    # Test disaggregated inference
    print("1. Testing Disaggregated Prefill-Decode Architecture")
    system = DisaggregatedInferenceSystem(config)
    system.setup_prefill_workers()
    system.setup_decode_workers()
    
    prompt = "The quick brown fox jumps over the lazy dog. " * 50
    response = system.process_request(prompt)
    print(f"Response: {response[:100]}...\n")
    
    # Test MoE routing
    print("2. Testing MoE Routing and Load Balancing")
    router = MoERouter(
        num_experts=config.num_experts,
        top_k=config.top_k_experts,
        capacity_factor=config.capacity_factor,
        precision_mode=config.moe_activation_precision,
    )
    
    tokens = list(range(100))  # Simulate 100 tokens
    assignments = router.route_tokens(tokens)
    
    print("Expert assignments:")
    for expert_id, assigned_tokens in assignments.items():
        print(f"Expert {expert_id}: {len(assigned_tokens)} tokens")
        
    metrics = router.get_load_balance_metrics()
    print(f"Load balance metrics: {metrics}\n")
    
    # Test speculative decoding
    print("3. Testing Speculative Decoding")
    decoder = SpeculativeDecoder(
        method=config.speculative_method,
        num_draft=config.num_speculative_tokens,
    )
    print(f"Expected speedup ≈ {decoder.expected_speedup():.2f}× for method {config.speculative_method}")
    speculative_tokens = decoder.speculative_decode(prompt, target_tokens=20)
    print(f"Speculative generation: {' '.join(speculative_tokens)}\n")

    # Structured / guided decoding simulation
    print("4. Structured / Guided Decoding")
    guided_decoder = GuidedDecoder(backend=config.guided_decoding_backend)
    guided_decoder.load_schema({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "latency_budget_ms": {"type": "number"},
        },
        "required": ["summary"],
    })
    structured = guided_decoder.generate(prompt)
    print(f"Guided decode payload: {json.dumps(structured)}\n")
    
    # Test parallelism strategies
    print("5. Testing Parallelism Strategies")
    manager = ParallelismManager(config)
    
    print("Tensor Parallelism:")
    manager.setup_tensor_parallelism()
    
    print("\nPipeline Parallelism:")
    manager.setup_pipeline_parallelism()
    
    print("\nExpert Parallelism:")
    manager.setup_expert_parallelism()
    
    print("\nData Parallelism:")
    manager.setup_data_parallelism()
    
    print("\nContext Parallelism:")
    manager.setup_context_parallelism()
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    benchmark_inference_system()

# Architecture-specific optimizations
if torch.cuda.is_available():
    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
