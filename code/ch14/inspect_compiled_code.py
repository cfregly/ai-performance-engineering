#!/usr/bin/env python3

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
"""
Inspecting torch.compile Generated Code

Shows how to dump and inspect compiled kernels to understand
what optimizations torch.compile applies to transformer blocks.
"""
import os

import shutil
from pathlib import Path

from core.utils.warning_filters import suppress_benchmark_import_warnings

with suppress_benchmark_import_warnings(context="ch14.inspect_compiled_code torch import"):
    import torch
    import torch.nn as nn


class SimpleLLMBlock(nn.Module):
    """Simplified LLM block for demonstrating torch.compile optimizations."""
    def __init__(self, d_model=2048, n_heads=16, d_ff=8192):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # Attention block with residual
        attn_input = self.ln1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_out
        
        # MLP block with residual
        x = x + self.mlp(self.ln2(x))
        return x


def setup_code_dump_directory():
    """Setup directory for dumping compiled code"""
    output_dir = Path("./compiled_code_output")
    
    # Clean up old output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    return output_dir


def inspect_compiled_code():
    """Compile an LLM block and inspect generated code."""
    print("=" * 80)
    print("INSPECTING torch.compile GENERATED CODE")
    print("=" * 80)
    print()
    
    # Setup output directory
    output_dir = setup_code_dump_directory()
    
    # Set environment variables to dump compiled code
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(output_dir.absolute())
    
    print("Configuration:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Debug mode: Enabled")
    print(f"  - Code tracing: Enabled")
    print()
    
    # Create a simple LLM block
    print("Creating LLM block...")
    model = SimpleLLMBlock(d_model=2048, n_heads=16, d_ff=8192).cuda()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.1f}M")
    print()
    
    # Create sample input (typical for LLM inference)
    batch_size = 4
    seq_len = 512
    d_model = 2048
    
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    print(f"Input shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimension: {d_model}")
    print()
    
    # Compile with code dumping enabled
    print("Compiling model...")
    print("  (This will generate Triton kernels and save them to disk)")
    print()
    
    with suppress_benchmark_import_warnings(context="ch14.inspect_compiled_code torch.compile"):
        compiled_model = torch.compile(
            model,
            mode='max-autotune',
            fullgraph=True,
            backend='inductor'
        )
    
    # Run once to trigger compilation
    print("Running compiled model (triggers code generation)...")
    with torch.inference_mode():
        output = compiled_model(x)
    
    print("Compilation complete!")
    print()
    
    # Show what was generated
    print("=" * 80)
    print("GENERATED CODE ARTIFACTS")
    print("=" * 80)
    print()
    
    # Look for generated files
    torch_compile_debug = Path("/tmp/torchinductor_" + os.environ.get('USER', 'user'))
    
    if torch_compile_debug.exists():
        print(f"Compiled code location: {torch_compile_debug}")
        print()
        
        # Count generated files without retaining the full debug tree.
        generated_file_count = 0
        generated_file_preview = []
        sample_kernel_file = None
        sample_kernel_lines = None
        sample_kernel_line_count = 0
        for pyfile in torch_compile_debug.rglob("*.py"):
            generated_file_count += 1
            if len(generated_file_preview) < 5:
                generated_file_preview.append(pyfile)
            if sample_kernel_file is None:
                content = pyfile.read_text()
                content_lower = content.lower()
                if 'triton' in content_lower and 'kernel' in content_lower:
                    sample_kernel_file = pyfile
                    all_lines = content.split('\n')
                    sample_kernel_lines = all_lines[:50]
                    sample_kernel_line_count = len(all_lines)
        triton_file_count = sum(1 for _ in torch_compile_debug.rglob("*triton*"))
        
        print(f"Found {generated_file_count} Python files")
        print(f"Found {triton_file_count} Triton-related files")
        print()
        
        if generated_file_preview:
            print("Generated kernel files (first 5):")
            for f in generated_file_preview:
                print(f"  - {f.name}")
            print()
        
        # Find and display a sample Triton kernel
        if sample_kernel_file is not None and sample_kernel_lines is not None:
            print("=" * 80)
            print(f"SAMPLE TRITON KERNEL: {sample_kernel_file.name}")
            print("=" * 80)
            print()
            
            # Show first 50 lines
            for i, line in enumerate(sample_kernel_lines, 1):
                print(f"{i:3}: {line}")
            
            if sample_kernel_line_count > 50:
                print(f"\n... ({sample_kernel_line_count} total lines)")
    else:
        print("WARNING: Generated code not found in expected location")
        print("    Set TORCH_COMPILE_DEBUG=1 to enable code dumps")
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS FROM COMPILED CODE")
    print("=" * 80)
    print()
    print("What to look for in generated Triton kernels:")
    print()
    print("1. **Kernel Fusion:**")
    print("   - Multiple ops combined into single kernel")
    print("   - Example: LayerNorm + Linear + GELU fused")
    print()
    print("2. **Memory Access Patterns:**")
    print("   - Coalesced loads/stores (16-byte aligned)")
    print("   - Vectorized operations (float4, float2)")
    print("   - Shared memory usage for fast access")
    print()
    print("3. **Block/Grid Configuration:**")
    print("   - BLOCK_M, BLOCK_N, BLOCK_K sizes")
    print("   - Number of warps per block")
    print("   - Number of pipeline stages")
    print()
    print("4. **Hardware-Specific Optimizations:**")
    print("   - TF32 for matmuls (faster on B200)")
    print("   - Tensor Core instructions")
    print("   - Cache hints and prefetching")
    print()
    print("5. **Graph Optimizations:")
    print("   - Dead code elimination")
    print("   - Common subexpression elimination")
    print("   - Constant folding")
    print()
    print("=" * 80)
    print("HOW TO USE THIS FOR DEBUGGING")
    print("=" * 80)
    print()
    print("1. Run this script to generate code")
    print("2. Look in /tmp/torchinductor_<user>/ for generated kernels")
    print("3. Search for 'triton' to find Triton kernel implementations")
    print("4. Compare kernel configs (BLOCK_M, etc.) for performance tuning")
    print("5. Check if expected fusions are happening")
    print()
    print("For more details, see:")
    print("  - PyTorch Compilation Docs: pytorch.org/docs/stable/torch.compiler.html")
    print("  - Triton Language: triton-lang.org")
    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)


def main():
    inspect_compiled_code()


if __name__ == "__main__":
    main()
