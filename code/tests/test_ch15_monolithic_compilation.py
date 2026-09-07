"""Real CPU graph capture controls; CUDA graph replay is checked on B200."""

import torch

from ch15.inference_monolithic_common import SimpleLLM
from ch15.optimized_inference_monolithic import OptimizedInferenceMonolithicBenchmark


def test_compiled_request_preserves_prompt_and_every_autoregressive_output():
    class CpuBenchmark(OptimizedInferenceMonolithicBenchmark):
        allow_cpu = True

    torch.manual_seed(15)
    benchmark = CpuBenchmark()
    benchmark.model = SimpleLLM(vocab_size=32, hidden_dim=16, num_layers=2).eval()
    benchmark.num_tokens = 7
    compiled = torch.compile(benchmark._full_inference, backend="eager", fullgraph=True)
    with torch.inference_mode():
        for offset in (0, 3):
            prompt = ((torch.arange(18).view(2, 9) + offset) % 32).long()
            original_prompt = prompt.clone()
            current = benchmark.model.prefill(prompt)
            expected_tokens = []
            for _ in range(benchmark.num_tokens):
                for layer in benchmark.model.layers:
                    current = torch.relu(torch.nn.functional.linear(current, layer.weight, layer.bias))
                expected_tokens.append(current)
            expected = torch.cat(expected_tokens, dim=1)
            actual = compiled(prompt)
            assert actual.shape == (2, 7, 16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(prompt, original_prompt, rtol=0, atol=0)
