"""MoE metadata lifetime and independent complete expert-output checks."""

import pytest
import torch
import torch.nn.functional as F

from labs.moe_optimization_journey.level4_triton import GroupedMoEExperts


@pytest.mark.parametrize("module,level", [
    ("optimized_moe_fp8", 2), ("optimized_moe_streams", 2),
    ("optimized_moe_sorted", 2), ("optimized_moe_permuted", 2),
    ("optimized_moe_parallel", 4), ("optimized_moe_expert_parallel", 5),
])
def test_legacy_named_factories_select_actual_shared_bf16_paths(module, level):
    import importlib
    from labs.moe_optimization_journey.moe_model import create_model

    benchmark = importlib.import_module("labs.moe_optimization_journey." + module).get_benchmark()
    assert benchmark.LEVEL == level
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        reference, _ = create_model(0, vocab_size=32, hidden_size=16, intermediate_size=32,
                                    num_layers=1, num_heads=4, num_experts=3, num_experts_per_tok=2)
        actual, opts = create_model(benchmark.LEVEL, vocab_size=32, hidden_size=16, intermediate_size=32,
                                    num_layers=1, num_heads=4, num_experts=3, num_experts_per_tok=2)
        actual.load_state_dict(reference.state_dict())
        actual.eval(), reference.eval()
        tokens = torch.tensor([[1, 3, 7, 2], [4, 2, 9, 3]])
        torch.testing.assert_close(actual(tokens), reference(tokens), rtol=1e-5, atol=1e-5)
        assert opts.use_compile is False and opts.use_cuda_graphs is False
        assert opts.use_grouped is (level >= 4)
        # Shared benchmark precision contract does not depend on its legacy name.
        assert benchmark.get_input_signature()["precision_flags"]["bf16"] is True


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Actual CUDA D2H metadata lifetime check requires a GPU"))])
def test_grouped_experts_preserve_dynamic_routing_and_full_output(device):
    rng = torch.Generator().manual_seed(91)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        layer = GroupedMoEExperts(5, 16, 32).double().to(device).eval()
        weights = [weight.detach().cpu().clone() for weight in (layer.w1, layer.w2, layer.w3)]
        for iteration in range(16):
            tokens = torch.randn(7, 16, generator=rng, dtype=torch.float64)
            # Change metadata each call, including empty experts and maximal skew.
            ids = torch.full((7, 2), iteration % 5, dtype=torch.int64) if iteration % 2 else torch.randint(0, 5, (7, 2), generator=rng)
            routing = torch.softmax(torch.randn(7, 2, generator=rng, dtype=torch.float64), -1)
            expected = torch.zeros_like(tokens)
            for row in range(7):
                for route in range(2):
                    expert = int(ids[row, route])
                    hidden = F.silu(tokens[row] @ weights[0][expert]) * (tokens[row] @ weights[2][expert])
                    expected[row].add_((hidden @ weights[1][expert]) * routing[row, route])
            inputs, assignment, probabilities = tokens.to(device), ids.to(device), routing.to(device)
            if device == "cuda":
                torch.cuda._sleep(100000)  # real current-stream work before metadata D2H
            actual = layer(inputs, assignment, probabilities).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
