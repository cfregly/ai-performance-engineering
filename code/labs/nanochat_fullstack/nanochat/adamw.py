"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
        self._reduce_scatter_futures: list[torch.Future] = []
        self._all_gather_futures: list[torch.Future] = []
        self._grad_slices: list[Tensor] = []
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for group in self.param_groups:
            for p in group["params"]:
                rank_size = p.shape[0] // world_size
                p_slice = p[:rank_size]
                state = self.state[p]
                state["_grad_slice"] = torch.empty_like(p_slice)
                state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
                state["denom"] = torch.empty_like(p_slice)
                state["lr_mul"] = getattr(p, "lr_mul", 1.0)
                state["wd_mul"] = getattr(p, "wd_mul", 1.0)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures = self._reduce_scatter_futures
        all_gather_futures = self._all_gather_futures
        grad_slices = self._grad_slices
        reduce_scatter_futures.clear()
        all_gather_futures.clear()
        grad_slices.clear()
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                p = params[base_i]
                grad = p.grad
                rank_size = grad.shape[0] // world_size
                state = self.state[p]
                grad_slice = state["_grad_slice"]
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                state = self.state[p]
                lr = group['lr'] * state["lr_mul"]
                g_slice = grad_slices[idx]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                denom = state['denom']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * state["wd_mul"]
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                torch.sqrt(exp_avg_sq, out=denom)
                denom.add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                torch.div(exp_avg, denom, out=g_slice)
                g_slice.mul_(step_size)
                p_slice.add_(other=g_slice, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()
        reduce_scatter_futures.clear()
        all_gather_futures.clear()
        grad_slices.clear()
