from __future__ import annotations

import pytest
import torch

from labs.moe_cuda_ptx.moe_cuda_ptx_common import (
    MoECudaPtxWorkload,
    backward_verification_numel,
    build_backward_verification,
    build_state,
    run_layer_baseline,
    run_layer_cuda,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for moe_cuda_ptx validation",
)


def test_run_layer_cuda_fwd_bwd_matches_reference_verification_slice() -> None:
    workload = MoECudaPtxWorkload(
        num_tokens=128,
        hidden_dim=128,
        expert_ffn_dim=64,
        mode="fwd_bwd",
    )
    state = build_state(workload, torch.device("cuda"))

    x_ref = state.x.detach().clone().requires_grad_(True)
    gate_ref = state.gate_proj.detach().clone().requires_grad_(True)
    up_ref = state.up_proj.detach().clone().requires_grad_(True)
    down_ref = state.down_proj.detach().clone().requires_grad_(True)
    ref_state = type(state)(
        x=x_ref,
        expert_indices=state.expert_indices,
        expert_weights=state.expert_weights,
        gate_proj=gate_ref,
        up_proj=up_ref,
        down_proj=down_ref,
        loss_grad=state.loss_grad,
    )
    ref_output = run_layer_baseline(ref_state, workload)
    (ref_output * state.loss_grad).sum().backward()
    ref_verify = build_backward_verification(
        ref_output,
        x_ref.grad,
        gate_ref.grad,
        down_ref.grad,
        torch.empty(
            backward_verification_numel(ref_output, x_ref.grad, gate_ref.grad, down_ref.grad),
            device=ref_output.device,
            dtype=torch.float32,
        ),
    )

    x_opt = state.x.detach().clone().requires_grad_(True)
    gate_opt = state.gate_proj.detach().clone().requires_grad_(True)
    up_opt = state.up_proj.detach().clone().requires_grad_(True)
    down_opt = state.down_proj.detach().clone().requires_grad_(True)
    opt_state = type(state)(
        x=x_opt,
        expert_indices=state.expert_indices,
        expert_weights=state.expert_weights,
        gate_proj=gate_opt,
        up_proj=up_opt,
        down_proj=down_opt,
        loss_grad=state.loss_grad,
    )
    opt_output = run_layer_cuda(opt_state, workload)
    (opt_output * state.loss_grad).sum().backward()
    opt_verify = build_backward_verification(
        opt_output,
        x_opt.grad,
        gate_opt.grad,
        down_opt.grad,
        torch.empty(
            backward_verification_numel(opt_output, x_opt.grad, gate_opt.grad, down_opt.grad),
            device=opt_output.device,
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(opt_verify, ref_verify, atol=2e-2, rtol=2e-2)
