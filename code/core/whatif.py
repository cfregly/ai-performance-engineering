"""
Shared what-if scenario helpers (static for now, can be extended to dynamic sims).
"""

from __future__ import annotations

from typing import Dict, Any


def get_scenarios() -> Dict[str, Any]:
    scenarios = []

    scenarios.append({
        "id": "fp8",
        "name": "Use FP8 Precision",
        "description": "Evaluate FP8 using compatible Transformer Engine modules; measure throughput and quality on your workload",
        "requirements": ["FP8-capable CUDA GPU", "Transformer Engine 2.18.x for this template", "Model ported to FP8-compatible TE modules and supported tensor shapes"],
        "estimated_speedup": 1.8,
        "memory_impact": -0.5,
        "accuracy_impact": "Workload-dependent; validate against the original precision before deployment",
        "implementation_effort": "low",
        "estimate_kind": "illustrative_unmeasured",
        "code_example_kind": "integration_template",
        "code_example_executable": False,
        "code_example_api_version": "Transformer Engine 2.18.x",
        "code_example_validation": "API documentation checked; CUDA execution remains unvalidated",
        "code_example_source": "https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/examples/fp8_primer.html",
        "code_example": (
            "# Integration template for Transformer Engine 2.18.x, not a standalone script.\n"
            "# model must already contain TE modules (e.g. te.Linear); inputs must be\n"
            "# CUDA tensors with supported shapes. Wrapping an arbitrary torch.nn model\n"
            "# does not convert its operations to FP8. Validate quality and performance.\n"
            "import transformer_engine.pytorch as te\n"
            "from transformer_engine.common.recipe import DelayedScaling, Format\n"
            "recipe = DelayedScaling(fp8_format=Format.HYBRID)\n"
            "with te.autocast(enabled=True, recipe=recipe):\n"
            "    output = model(inputs)"
        ),
    })

    scenarios.append({
        "id": "flash_attention",
        "name": "Enable Flash Attention",
        "description": "Use memory-efficient attention for O(n) memory instead of O(n²)",
        "requirements": ["PyTorch 2.0+", "Attention-based model"],
        "estimated_speedup": 2.5,
        "memory_impact": -0.8,
        "accuracy_impact": "None (mathematically equivalent)",
        "implementation_effort": "low",
        "code_example": "from torch.nn.functional import scaled_dot_product_attention\noutput = scaled_dot_product_attention(q, k, v, is_causal=True)",
    })

    scenarios.append({
        "id": "torch_compile",
        "name": "Enable torch.compile",
        "description": "JIT compile model for kernel fusion and optimization",
        "requirements": ["PyTorch 2.0+"],
        "estimated_speedup": 1.4,
        "memory_impact": 0.1,
        "accuracy_impact": "None",
        "implementation_effort": "low",
        "code_example": "model = torch.compile(model, mode='max-autotune')",
    })

    scenarios.append({
        "id": "batch_size",
        "name": "Double Batch Size",
        "description": "Increase batch size for better GPU utilization",
        "requirements": ["Sufficient VRAM"],
        "estimated_speedup": 1.6,
        "memory_impact": 1.0,
        "accuracy_impact": "May need learning rate adjustment for training",
        "implementation_effort": "low",
        "code_example": "# Update dataloader\nbatch_size = current_batch_size * 2",
    })

    return {"scenarios": scenarios, "count": len(scenarios)}
