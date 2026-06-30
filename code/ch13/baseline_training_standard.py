"""baseline_training_standard.py - Standard transformer training without checkpointing.

Standard training stores all activations for backward pass, including:
- Attention weights: O(batch * heads * seq_len²) per layer
- Intermediate FFN activations: O(batch * seq_len * 4*hidden) per layer

This is faster but uses significantly more memory than gradient checkpointing.
Compare with optimized_training_standard.py which uses checkpointing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class TransformerModel(nn.Module):
    """Transformer model - stores all attention weights and activations during forward."""
    
    def __init__(
        self, 
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        seq_len: int = 512,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.register_buffer(
            "_position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        
        # Transformer layers - standard PyTorch implementation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        pos_ids = self._position_ids[:, :seq_len].expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        
        # Transformer (stores ALL attention matrices in memory for backward)
        x = self.transformer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class BaselineTrainingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Standard transformer training that stores all activations (memory heavy, but fast).
    
    Memory usage dominated by:
    - Model parameters: ~num_layers * hidden² * 12 (weights + grads + optimizer states)
    - Activations: ~num_layers * batch * seq_len * hidden * 2 (attention + FFN)
    - Attention weights: ~num_layers * batch * heads * seq_len² (quadratic!)
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input_ids = None
        self.targets = None
        self._targets_flat = None
        self.optimizer = None
        self.criterion = None
        
        # Workload config - optimized for demonstrating activation memory
        self.hidden_dim = 1024
        self.num_layers = 24  # Deep enough to show memory difference
        self.num_heads = 16
        self.seq_len = 1024   # Long sequences = more activation memory
        self.batch_size = 8   # Reasonable batch for training
        self.vocab_size = 32000
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._peak_memory_gb = 0.0
        self._memory_bytes_to_gb = 1e-9
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize transformer model and data."""
        # Clear memory before setup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = TransformerModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
        )
        self.model = self.model.to(self.device).train()
        
        # Random input tokens
        self.input_ids = torch.randint(
            0, self.vocab_size, 
            (self.batch_size, self.seq_len), 
            device=self.device
        )
        # Shifted targets for language modeling
        self.targets = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device
        )
        self._targets_flat = self.targets.view(-1)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self._verify_output_buffer = torch.empty(
            (1, 1, min(8, self.vocab_size)),
            device=self.device,
            dtype=torch.float32,
        )
        
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Training step without checkpointing - stores all attention weights."""
        if (
            self.model is None
            or self.input_ids is None
            or self.targets is None
            or self._targets_flat is None
            or self.optimizer is None
            or self.criterion is None
        ):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_training"):
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass - stores ALL activations for backward
            logits = self.model(self.input_ids)
            
            # Compute loss
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                self._targets_flat
            )
            self.output = None
            
            # Backward pass - uses stored activations
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
        if self.input_ids is None:
            raise RuntimeError("benchmark_fn() requires input_ids for verification")

    def capture_verification_payload(self) -> None:
        if self.model is None or self.input_ids is None:
            raise RuntimeError("capture_verification_payload() requires model and inputs")
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        with torch.inference_mode():
            verify_logits = self.model(self.input_ids)
            output_slice = verify_logits[
                : self._verify_output_buffer.shape[0],
                : self._verify_output_buffer.shape[1],
                : self._verify_output_buffer.shape[2],
            ]
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output,
            batch_size=self.input_ids.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.5, 10.0),
        )

    def finalize_iteration_metrics(self) -> Optional[dict]:
        """Poll peak memory after harness timing has already finalized."""
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) * self._memory_bytes_to_gb
        self._peak_memory_gb = max(self._peak_memory_gb, peak_memory_gb)
        return None
    
    def teardown(self) -> None:
        """Cleanup and report memory usage."""
        if torch.cuda.is_available():
            self.finalize_iteration_metrics()
        if self._peak_memory_gb > 0:
            print(f"\n[Baseline] Peak GPU Memory: {self._peak_memory_gb:.2f} GB")
        
        self.model = None
        self.input_ids = None
        self.targets = None
        self._targets_flat = None
        self.optimizer = None
        self.criterion = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=True,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_ids is None:
            return "Input tensor not initialized"
        
        try:
            with torch.inference_mode():
                test_output = self.model(self.input_ids[:1])
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        
        return None


def get_benchmark() -> BaselineTrainingBenchmark:
    """Factory function for harness discovery."""
    return BaselineTrainingBenchmark()
