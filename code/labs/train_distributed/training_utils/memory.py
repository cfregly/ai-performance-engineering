"""
Utility functions to visualize memory differences
"""

import torch


def get_size_in_mb(tensor):
    """Get size of tensor in MB"""
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement() / 1024**2


def get_optimizer_memory(optimizer):
    """Calculate total memory used by optimizer states"""
    total_memory = 0
    if hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    elif hasattr(optimizer, "optim"):
        # ZeroRedundancyOptimizer keeps this rank's AdamW state in .optim;
        # the wrapper's .state mapping is empty even after successful updates.
        optimizer = optimizer.optim
    for state in optimizer.state.values():
        for state_tensor in state.values():
            if torch.is_tensor(state_tensor):
                total_memory += get_size_in_mb(state_tensor)
    return total_memory


def get_model_memory(model):
    """Calculate memory used by model parameters"""
    return sum(get_size_in_mb(p) for p in model.parameters())


def get_gradient_memory(model):
    """Calculate memory used by gradients"""
    return sum(get_size_in_mb(p.grad) for p in model.parameters() if p.grad is not None)


def print_memory_stats(prefix: str, model, optimizer, rank, device):
    model_memory = get_model_memory(model)
    grad_memory = get_gradient_memory(model)
    optim_memory = get_optimizer_memory(optimizer)
    total_allocated = torch.cuda.memory_allocated(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"\nGPU {rank} - {prefix}:")
    print(f"  Model parameters: {model_memory:.2f} MB")
    print(f"  Gradients: {grad_memory:.2f} MB")
    print(f"  Optimizer states: {optim_memory:.2f} MB")
    print(f"  Total allocated: {total_allocated:.2f} MB")
    print(f"  Max allocated: {max_allocated:.2f} MB")
    print("-" * 40)
