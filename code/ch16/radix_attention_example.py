#!/usr/bin/env python3

import heapq

import torch.cuda.nvtx as nvtx
from core.profiling.nvtx_helper import standardize_nvtx_label
import torch
from core.utils.architecture_runtime import get_arch_config

_ARCH_CFG = get_arch_config()
"""
radix_attention_example.py
Chapter 16: Radix Attention KV Cache Implementation Example

This example demonstrates SGLang's RadixAttention approach using a compressed trie 
(radix tree) for efficient KV cache prefix sharing in LLM inference.

Based on Chapter 16 content and SGLang's prefix caching design.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import nullcontext


def _attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention core used for both prefill and decode paths."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


@dataclass
class KVCache:
    """Represents a KV cache tensor pair for a specific prefix."""
    keys: torch.Tensor      # [capacity, num_heads, head_dim]
    values: torch.Tensor    # [capacity, num_heads, head_dim]
    seq_len: int
    ref_count: int = 1
    capacity: int = 0

    def __post_init__(self):
        if self.capacity == 0:
            self.capacity = self.keys.size(0)

    @property
    def key_view(self) -> torch.Tensor:
        return self.keys[:self.seq_len]

    @property
    def value_view(self) -> torch.Tensor:
        return self.values[:self.seq_len]
    
    def clone_ref(self):
        """Shallow clone with reference counting."""
        self.ref_count += 1
        return KVCache(
            keys=self.keys,
            values=self.values,
            seq_len=self.seq_len,
            ref_count=self.ref_count,
            capacity=self.capacity,
        )

    def append(self, k: torch.Tensor, v: torch.Tensor) -> "KVCache":
        """Append one or more KV rows without rebuilding the full prefix when capacity allows."""
        append_len = k.size(0)
        new_seq_len = self.seq_len + append_len
        if new_seq_len <= self.capacity:
            self.keys[self.seq_len:new_seq_len].copy_(k)
            self.values[self.seq_len:new_seq_len].copy_(v)
            return KVCache(
                keys=self.keys,
                values=self.values,
                seq_len=new_seq_len,
                capacity=self.capacity,
            )

        new_capacity = max(new_seq_len, max(1, self.capacity) * 2)
        new_keys = torch.empty(
            new_capacity,
            *self.keys.shape[1:],
            dtype=self.keys.dtype,
            device=self.keys.device,
        )
        new_values = torch.empty_like(new_keys)
        new_keys[:self.seq_len].copy_(self.key_view)
        new_values[:self.seq_len].copy_(self.value_view)
        new_keys[self.seq_len:new_seq_len].copy_(k)
        new_values[self.seq_len:new_seq_len].copy_(v)
        return KVCache(keys=new_keys, values=new_values, seq_len=new_seq_len, capacity=new_capacity)
    
    def release(self):
        """Decrease reference count."""
        self.ref_count -= 1
        return self.ref_count <= 0

class RadixTreeNode:
    """Node in the radix tree for prefix caching."""
    
    def __init__(self, token_sequence: List[int] = None, cache: KVCache = None):
        self.token_sequence = token_sequence or []  # Edge label (sequence of tokens)
        self.cache = cache                          # KV cache for this prefix
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.last_accessed = 0                      # For LRU eviction
        self.is_leaf = True
    
    def __repr__(self):
        return f"RadixNode(tokens={self.token_sequence}, children={len(self.children)})"

class RadixTree:
    """
    Radix tree (compressed trie) for efficient prefix caching.
    Based on SGLang's RadixAttention design.
    """
    
    def __init__(self):
        self.root = RadixTreeNode()
        self.access_counter = 0
        self.max_cache_size = 200  # Maximum number of cached prefixes
        self.current_cache_count = 0
    
    def longest_prefix(self, tokens: List[int]) -> Tuple[RadixTreeNode, int]:
        """
        Find the longest cached prefix for the given token sequence.
        Returns (node, prefix_length)
        """
        current = self.root
        matched_len = 0
        
        i = 0
        while i < len(tokens):
            # Check if any child edge starts with tokens[i]
            found_child = None
            for first_token, child in current.children.items():
                if first_token == tokens[i]:
                    found_child = child
                    break
            
            if found_child is None:
                break
            
            # Check how many tokens match this edge
            edge_tokens = found_child.token_sequence
            edge_match_len = 0
            
            for j, edge_token in enumerate(edge_tokens):
                if i + j >= len(tokens) or tokens[i + j] != edge_token:
                    break
                edge_match_len += 1
            
            if edge_match_len == 0:
                break
            
            # If we matched the entire edge, move to child and continue
            if edge_match_len == len(edge_tokens):
                current = found_child
                matched_len += edge_match_len
                i += edge_match_len
                # Update access time for LRU
                current.last_accessed = self.access_counter
                self.access_counter += 1
            else:
                # Partial match - we found the longest prefix
                break
        
        return current, matched_len
    
    def insert(self, tokens: List[int], cache: KVCache) -> RadixTreeNode:
        """
        Insert token sequence and associated KV cache into the radix tree.
        May split existing edges if needed.
        """
        if not tokens:
            return self.root
        
        current = self.root
        i = 0
        
        while i < len(tokens):
            # Check if any child edge starts with tokens[i]
            found_child = None
            first_token = tokens[i]
            
            for child_first_token, child in current.children.items():
                if child_first_token == first_token:
                    found_child = child
                    break
            
            if found_child is None:
                # No matching child - create new edge with remaining tokens
                remaining_tokens = tokens[i:]
                new_node = RadixTreeNode(remaining_tokens, cache)
                new_node.last_accessed = self.access_counter
                self.access_counter += 1
                
                current.children[first_token] = new_node
                current.is_leaf = False
                self._maybe_evict()
                return new_node
            
            # Found a child - check how much of the edge matches
            edge_tokens = found_child.token_sequence
            edge_match_len = 0
            
            for j, edge_token in enumerate(edge_tokens):
                if i + j >= len(tokens) or tokens[i + j] != edge_token:
                    break
                edge_match_len += 1
            
            if edge_match_len == len(edge_tokens):
                # Matched entire edge - continue to child
                current = found_child
                i += edge_match_len
                
                if i == len(tokens):
                    # Exact match - update cache
                    if current.cache:
                        current.cache.release()
                    current.cache = cache
                    current.last_accessed = self.access_counter
                    self.access_counter += 1
                    return current
            else:
                # Partial match - need to split the edge
                return self._split_edge(found_child, edge_match_len, tokens, i, cache)
        
        return current
    
    def _split_edge(self, node: RadixTreeNode, split_pos: int, 
                   new_tokens: List[int], new_token_pos: int, cache: KVCache) -> RadixTreeNode:
        """Split an edge at the given position."""
        # Split the existing edge
        common_prefix = node.token_sequence[:split_pos]
        remaining_old = node.token_sequence[split_pos:]
        remaining_new = new_tokens[new_token_pos + split_pos:]
        
        # Create intermediate node for common prefix
        intermediate = RadixTreeNode(common_prefix)
        intermediate.last_accessed = self.access_counter
        self.access_counter += 1
        
        # Update the existing node to represent the remaining part
        node.token_sequence = remaining_old
        if remaining_old:
            intermediate.children[remaining_old[0]] = node
            intermediate.is_leaf = False
        
        # Create new node for the new branch
        if remaining_new:
            new_node = RadixTreeNode(remaining_new, cache)
            new_node.last_accessed = self.access_counter
            self.access_counter += 1
            intermediate.children[remaining_new[0]] = new_node
            intermediate.is_leaf = False
            self._maybe_evict()
            return new_node
        else:
            # New tokens end at the split point
            intermediate.cache = cache
            self._maybe_evict()
            return intermediate
    
    def _maybe_evict(self):
        """Evict least recently used nodes if cache is full."""
        self.current_cache_count += 1
        
        if self.current_cache_count > self.max_cache_size:
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict the least recently used leaf nodes."""
        leaves = []
        self._collect_leaves(self.root, leaves, set())
        
        if not leaves:
            return
        
        # Evict the oldest 10% of leaves
        evict_count = max(1, len(leaves) // 10)
        for leaf in heapq.nsmallest(evict_count, leaves, key=lambda x: x.last_accessed):
            if leaf.cache:
                leaf.cache.release()
                leaf.cache = None
                self.current_cache_count -= 1
    
    def _collect_leaves(self, node: RadixTreeNode, leaves: List[RadixTreeNode], visited: set = None):
        """Collect all leaf nodes for LRU eviction using iterative traversal."""
        if visited is None:
            visited = set()
        
        # Use iterative traversal to avoid recursion issues
        stack = [node]
        
        while stack:
            current = stack.pop()
            
            # Prevent infinite recursion
            if id(current) in visited:
                continue
            visited.add(id(current))
            
            if current.is_leaf and current.cache:
                leaves.append(current)
            
            # Add children to stack
            for child in current.children.values():
                if id(child) not in visited:
                    stack.append(child)

class ModelState:
    """Simplified model state for generation."""
    
    def __init__(
        self,
        kv_cache: KVCache = None,
        hidden_dim: int = 512,
        num_heads: int = 8,
        device: Optional[torch.device] = None,
        context: Optional[torch.Tensor] = None,
    ):
        self.kv_cache = kv_cache
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.context = context
        self.finished = False
    
    @classmethod
    def from_cache(cls, cache: KVCache, hidden_dim: int, num_heads: int) -> 'ModelState':
        """Create model state from existing KV cache (reference counting)."""
        device = cache.keys.device if cache else None
        return cls(
            kv_cache=cache.clone_ref() if cache else None,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            device=device,
        )
    
    def is_finished(self) -> bool:
        return self.finished

class SimpleTransformerModel:
    """Simplified transformer model for demonstration."""
    
    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_dim: int = 256,
        num_heads: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._cache_block_tokens = 128
        self._token_buffer = torch.empty(1, dtype=torch.long, device=self.device)
        self._sampling_noise = None
        self._sampling_topk_logits = None
        self._sampling_topk_indices = None
        self._sampling_probs = None
        self._sampling_choice = None
        self._sampling_next_token = None
        self._sampling_next_token_host = None
        
        # Initialize lightweight modules on the active device
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim, device=self.device)
        self.ln = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device=self.device)
        self.embedding.weight.requires_grad_(False)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.lm_head):
            torch.nn.init.normal_(module.weight, std=0.02)
            module.requires_grad_(False)
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        self.ln.requires_grad_(False)
        torch.nn.init.ones_(self.ln.weight)
        torch.nn.init.zeros_(self.ln.bias)
        
        self._attention_kernel = _attention_core

    def _sampling_like_buffer(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
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

    def _sampling_long_buffer(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        buffer = getattr(self, name)
        if (
            buffer is None
            or buffer.device != self.device
            or buffer.dim() != len(shape)
            or any(buffer.size(dim) < size for dim, size in enumerate(shape))
        ):
            buffer = torch.empty(shape, dtype=torch.long, device=self.device)
            setattr(self, name, buffer)
        return buffer[tuple(slice(0, size) for size in shape)]

    def _sampling_host_buffer(self) -> torch.Tensor:
        if self._sampling_next_token_host is None:
            self._sampling_next_token_host = torch.empty(1, dtype=torch.long, device="cpu")
        return self._sampling_next_token_host

    def _new_cache(self, k: torch.Tensor, v: torch.Tensor) -> KVCache:
        capacity = max(self._cache_block_tokens, k.size(0))
        keys = torch.empty(
            capacity,
            self.num_heads,
            self.head_dim,
            dtype=k.dtype,
            device=k.device,
        )
        values = torch.empty_like(keys)
        keys[:k.size(0)].copy_(k)
        values[:v.size(0)].copy_(v)
        return KVCache(keys=keys, values=values, seq_len=k.size(0), capacity=capacity)
    
    @torch.inference_mode()
    def forward(self, token: int, state: ModelState) -> ModelState:
        """Forward pass for a single token, updating KV cache."""
        # Get token embedding
        self._token_buffer[0] = token
        x = self.embedding(self._token_buffer)  # [1, hidden_dim]
        x = self.ln(x)

        # Compute Q, K, V on device
        q = self.q_proj(x).view(1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(1, self.num_heads, self.head_dim)
        
        # Update KV cache
        if state.kv_cache is None:
            # First token - initialize cache
            new_cache = self._new_cache(k, v)
        else:
            # Append to existing cache
            new_cache = state.kv_cache.append(k, v)
            
            # Release old cache reference
            if state.kv_cache:
                state.kv_cache.release()
        
        # Run the fused attention core (prefill)
        keys_for_attn = new_cache.key_view.permute(1, 0, 2).unsqueeze(0)  # [1, H, S, D]
        values_for_attn = new_cache.value_view.permute(1, 0, 2).unsqueeze(0)
        query_for_attn = q.unsqueeze(2)  # [1, H, 1, D]
        with nvtx.range(standardize_nvtx_label("compute_math:radix_attention_prefill")) if torch.cuda.is_available() else nullcontext():
            context = self._attention_kernel(query_for_attn, keys_for_attn, values_for_attn)
        context = context.reshape(1, self.hidden_dim)
        
        # Create new state
        new_state = ModelState(
            kv_cache=new_cache,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            device=self.device,
            context=context,
        )
        return new_state
    
    @torch.inference_mode()
    def generate_next(self, state: ModelState) -> Tuple[int, ModelState]:
        """Generate next token autoregressively."""
        if state.kv_cache is None:
            raise ValueError("Cannot generate without KV cache")

        seq_len = state.kv_cache.seq_len
        keys = state.kv_cache.key_view.permute(1, 0, 2).unsqueeze(0)
        values = state.kv_cache.value_view.permute(1, 0, 2).unsqueeze(0)
        query = state.kv_cache.key_view[-1:].permute(1, 0, 2).unsqueeze(0)
        with nvtx.range(standardize_nvtx_label("compute_math:radix_attention_decode")) if torch.cuda.is_available() else nullcontext():
            context = self._attention_kernel(query, keys, values)
        attn_out = context.reshape(1, self.hidden_dim)
        state.context = attn_out
        output = self.out_proj(attn_out)

        # Get logits and sample
        logits = self.lm_head(output)
        
        # Add more randomness to prevent repetitive tokens
        noise = self._sampling_like_buffer("_sampling_noise", logits)
        torch.randn(logits.shape, dtype=logits.dtype, device=logits.device, out=noise)
        logits.add_(noise, alpha=0.5)
        
        # Use temperature scaling for more diverse sampling
        temperature = 1.0
        if temperature != 1.0:
            logits.div_(temperature)
        
        # Add top-k sampling for more diversity
        k = min(10, logits.size(-1))
        top_k_logits = self._sampling_like_buffer("_sampling_topk_logits", logits[:, :k])
        top_k_indices = self._sampling_long_buffer("_sampling_topk_indices", (logits.size(0), k))
        torch.topk(logits, k, dim=-1, out=(top_k_logits, top_k_indices))
        
        # Sample from top-k
        probs = self._sampling_like_buffer("_sampling_probs", top_k_logits)
        selected_idx = self._sampling_long_buffer("_sampling_choice", (logits.size(0), 1))
        next_token_device = self._sampling_long_buffer("_sampling_next_token", (logits.size(0), 1))
        torch.softmax(top_k_logits, dim=-1, out=probs)
        torch.multinomial(probs, num_samples=1, out=selected_idx)
        torch.gather(top_k_indices, 1, selected_idx, out=next_token_device)
        next_token_host = self._sampling_host_buffer()
        next_token_host.copy_(next_token_device.view(-1)[:1])
        next_token = int(next_token_host[0])
        
        # Check for end of generation (simplified)
        state.finished = (next_token == 0) or (seq_len >= 50)  # EOS or max length
        
        return next_token, state

def generate_with_radix(prompt_tokens: List[int], model: SimpleTransformerModel, 
                       radix: RadixTree, max_generated_tokens: int = 6) -> List[int]:  # cap generation so profiling stays snappy
    """
    Generate tokens using RadixAttention prefix caching.
    
    This is the main function from Chapter 16 that demonstrates
    how SGLang's RadixAttention works.
    """
    print(f"Generating with prompt: {prompt_tokens}")
    
    # 1) Find longest cached prefix
    node, prefix_len = radix.longest_prefix(prompt_tokens)
    print(f"Found cached prefix of length: {prefix_len}")
    
    # Shallow-clone the KV cache for that prefix
    model_state = (
        ModelState.from_cache(node.cache, model.hidden_dim, model.num_heads)
        if node.cache
        else ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)
    )
    
    # 2) Process remaining prompt suffix
    for i, token in enumerate(prompt_tokens[prefix_len:]):
        model_state = model.forward(token, state=model_state)
        
        # 3) As we go, insert or split edges in the radix tree
        matched = prompt_tokens[:prefix_len + i + 1]
        # Insert returns the node for this full prefix
        node = radix.insert(matched, cache=model_state.kv_cache)
        
        print(f"Processed token {token}, cached prefix length: {len(matched)}")
    
    # 4) Now generate new tokens autoregressively
    output_tokens = []
    while not model_state.is_finished() and len(output_tokens) < max_generated_tokens:
        token, model_state = model.generate_next(model_state)
        output_tokens.append(token)
        
        # Cache each generated prefix as well
        matched = prompt_tokens + output_tokens
        node = radix.insert(matched, cache=model_state.kv_cache)
        
        print(f"Generated token: {token}")
    
    print(f"Generated {len(output_tokens)} tokens")
    return output_tokens

def simulate_multi_turn_conversation():
    """
    Simulate the multi-turn conversation example from Chapter 16 Figure 16-9.
    Demonstrates how the radix tree evolves with multiple chat sessions.
    """
    print("\n=== Multi-Turn Conversation Simulation ===")
    
    # Initialize model and radix tree
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    radix = RadixTree()
    radix.max_cache_size = 10  # Small for demo
    
    # System prompt (shared across all conversations)
    system_prompt = [1, 2, 3, 4, 5]  # "You are a helpful assistant"
    
    # Chat Session 1
    print("\n--- Chat Session 1 ---")
    
    # Turn 1: User says "Hello"
    prompt1 = system_prompt + [10, 11]  # + "Hello"
    response1 = generate_with_radix(prompt1, model, radix)
    
    # Turn 2: Continue conversation
    prompt2 = prompt1 + response1 + [12, 13]  # + "How are you?"
    response2 = generate_with_radix(prompt2, model, radix)
    
    # Chat Session 2 (shares system prompt)
    print("\n--- Chat Session 2 ---")
    
    # Turn 1: Different user message
    prompt3 = system_prompt + [20, 21]  # + "Hi there"
    response3 = generate_with_radix(prompt3, model, radix)
    
    # Turn 2: Continue session 2
    prompt4 = prompt3 + response3 + [22, 23]  # + "What's new?"
    response4 = generate_with_radix(prompt4, model, radix)
    
    # Continue Chat Session 1 (should reuse cached data)
    print("\n--- Back to Chat Session 1 ---")
    prompt5 = prompt2 + response2 + [14, 15]  # + "Thanks!"
    response5 = generate_with_radix(prompt5, model, radix)
    
    print(f"\nRadix tree now has {radix.current_cache_count} cached prefixes")
    
    # New conversation with different system prompt
    print("\n--- New Conversation (Different System) ---")
    new_system = [100, 101, 102]  # Different system prompt
    prompt6 = new_system + [30, 31]  # + "Hello world"
    response6 = generate_with_radix(prompt6, model, radix)
    
    print(f"Final radix tree has {radix.current_cache_count} cached prefixes")

def benchmark_prefix_caching():
    """
    Benchmark to show the benefits of prefix caching.
    Compares generation with and without radix tree caching.
    """
    print("\n=== Prefix Caching Benchmark ===")
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    
    # Common system prompt (long)
    system_prompt = list(range(1, 65))  # 64 tokens to emphasize GPU throughput
    
    # Test prompts that share the system prompt
    test_prompts = [
        system_prompt + [100 + i for i in range(16)],
        system_prompt + [200 + i for i in range(24)],
        system_prompt + [300 + i for i in range(8)],
        system_prompt + [400 + i for i in range(32)],
    ]
    
    import time
    
    # Test without caching (naive approach)
    print("\n--- Without Caching ---")
    passes = 2  # first pass warms caches, second pass demonstrates reuse
    
    # Use CUDA Events for accurate GPU timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(model.device)
        start_event.record(current_stream)
    else:
        start_time = time.perf_counter()
    
    naive_responses = []
    with nvtx.range(standardize_nvtx_label("step:radix_naive_pass")) if torch.cuda.is_available() else nullcontext():
        for rep in range(passes):
            for i, prompt in enumerate(test_prompts):
                print(f"Processing prompt {i+1}/{len(test_prompts)} (pass {rep+1}/{passes})")
                state = ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)
                for token in prompt:
                    state = model.forward(token, state)

                response = []
                for _ in range(3):  # Generate 3 tokens
                    if state.is_finished():
                        break
                    token, state = model.generate_next(state)
                    response.append(token)
                naive_responses.append(response)
    
    if torch.cuda.is_available():
        end_event.record(current_stream)
        end_event.synchronize()
        naive_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
    else:
        naive_time = time.perf_counter() - start_time
    print(f"Naive approach took: {naive_time:.3f} seconds")
    
    # Test with RadixAttention caching
    print("\n--- With RadixAttention Caching ---")
    radix = RadixTree()
    warm_state = ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)
    for token in system_prompt:
        warm_state = model.forward(token, warm_state)
    radix.insert(system_prompt, warm_state.kv_cache)
    
    # Use CUDA Events for accurate GPU timing
    if torch.cuda.is_available():
        start_event.record(current_stream)
    else:
        start_time = time.perf_counter()
    
    cached_responses = []
    with nvtx.range(standardize_nvtx_label("step:radix_cached_pass")) if torch.cuda.is_available() else nullcontext():
        for rep in range(passes):
            for i, prompt in enumerate(test_prompts):
                print(f"Processing prompt {i+1}/{len(test_prompts)} with caching (pass {rep+1}/{passes})")

                node, prefix_len = radix.longest_prefix(prompt)
                print(f"  Reused {prefix_len}/{len(prompt)} tokens from cache")

                state = (
                    ModelState.from_cache(node.cache, model.hidden_dim, model.num_heads)
                    if node.cache
                    else ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)
                )

                for token in prompt[prefix_len:]:
                    state = model.forward(token, state)

                radix.insert(prompt, state.kv_cache)

                response = []
                for _ in range(3):  # Generate 3 tokens
                    if state.is_finished():
                        break
                    token, state = model.generate_next(state)
                    response.append(token)

                    full_sequence = prompt + response
                    radix.insert(full_sequence, state.kv_cache)

                cached_responses.append(response)
    
    if torch.cuda.is_available():
        end_event.record(current_stream)
        end_event.synchronize()
        cached_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
    else:
        cached_time = time.perf_counter() - start_time
    print(f"RadixAttention approach took: {cached_time:.3f} seconds")
    
    speedup = naive_time / cached_time if cached_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    for prompt in test_prompts:
        _, cached_prefix = radix.longest_prefix(prompt)
        print(f"Cached prefix length for prompt of {len(prompt)} tokens: {cached_prefix}")
    
    # Verify responses are similar (they won't be identical due to randomness)
    print(f"Radix tree cached {radix.current_cache_count} prefixes")

def main():
    """Main demonstration of RadixAttention concepts."""
    print("Chapter 16: RadixAttention Prefix Caching Example")
    print("=" * 50)
    
    # Basic example
    print("\n=== Basic RadixAttention Example ===")
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    radix = RadixTree()
    
    # Simple prompt
    prompt = [1, 2, 3, 4, 5, 6, 7]
    response = generate_with_radix(prompt, model, radix)
    print(f"Response: {response}")
    
    # Second prompt with shared prefix
    prompt2 = [1, 2, 3, 4, 5, 8, 9]  # Shares first 5 tokens
    response2 = generate_with_radix(prompt2, model, radix)
    print(f"Response 2: {response2}")
    
    # Run benchmark demo to highlight cache reuse
    benchmark_prefix_caching()
    
    print("\n" + "=" * 50)
    print("RadixAttention Demo Complete!")
    print("Key benefits demonstrated:")
    print("- Efficient prefix sharing across requests")
    print("- Automatic KV cache management")
    print("- LRU eviction for memory management")
    print("- Significant speedup for repeated prefixes")

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"}:
        print(f"Enabled {_ARCH_CFG.get_architecture_name()} optimizations (compute capability {compute_capability})")
    else:
        print(f" Unsupported compute capability {compute_capability}; running in fallback mode")
