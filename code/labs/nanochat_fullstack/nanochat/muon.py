"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        groups_by_size = {}
        for p in params:
            groups_by_size.setdefault(p.numel(), []).append(p)
        param_groups = []
        for size in sorted(groups_by_size):
            param_groups.append(dict(params=groups_by_size[size]))
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton–Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0,1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton–Schulz iterations for the orthogonalization
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # Group all parameters by their shape
        groups_by_shape = {}
        for p in params:
            groups_by_shape.setdefault(p.shape, []).append(p)
        param_groups = []
        world_size = dist.get_world_size()
        for shape in sorted(groups_by_shape): # sort to ensure consistent / deterministic ordering
            group_params = groups_by_shape[shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(
                params=group_params,
                zero_buffer=torch.zeros_like(group_params[0]),
                scatter_pad_buffer=torch.empty_like(group_params[0]),
                gather_pad_buffers=[
                    torch.empty_like(group_params[0])
                    for _ in range(world_size - 1)
                ],
            ))
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            params = group["params"]
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                if owner_idx < len(params):
                    p = params[owner_idx]
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            scatter_pad_buffer = group["scatter_pad_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else scatter_pad_buffer
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            gather_pad_buffers = group["gather_pad_buffers"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the Muon update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                missing = world_size - len(ag_output)
                if missing:
                    ag_output.extend(gather_pad_buffers[:missing]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()
