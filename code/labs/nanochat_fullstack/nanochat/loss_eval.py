"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.inference_mode()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record [total_nats, total_bytes] in one tensor for one reduction/readback
    totals = torch.empty(2, dtype=torch.float64, device=model.get_device())
    totals.zero_()
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        # Branchless path avoids a per-batch CUDA sync from testing whether ignore_index appears.
        # mps does not currently have kernel support for int64 comparisons here, only int32.
        valid = y.int() >= 0
        y_safe = y.clamp_min(0)
        # map valid targets to their byte length; ignored targets contribute 0 bytes
        num_bytes2d = token_bytes[y_safe] * valid.to(dtype=token_bytes.dtype)
        totals[0].add_((loss2d * (num_bytes2d > 0)).sum())
        totals[1].add_(num_bytes2d.sum())
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    totals_host = totals.detach().cpu()
    total_nats = float(totals_host[0])
    total_bytes = float(totals_host[1])
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
