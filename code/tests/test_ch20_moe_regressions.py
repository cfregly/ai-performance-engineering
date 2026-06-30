from pathlib import Path

import torch

from ch20.baseline_moe import ToyMoe as BaselineToyMoe
from ch20.optimized_moe import ToyMoe as OptimizedToyMoe


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ch20_moe_routes_without_softmax_or_float_masks() -> None:
    cases = (
        ("baseline_moe.py", "class BaselineMoeBenchmark", BaselineToyMoe),
        ("optimized_moe.py", "class OptimizedMoeBenchmark", OptimizedToyMoe),
    )

    for filename, end_marker, model_cls in cases:
        source = (REPO_ROOT / "ch20" / filename).read_text(encoding="utf-8")
        forward_section = source.split("def forward(self, x: torch.Tensor)", maxsplit=1)[1].split(
            end_marker,
            maxsplit=1,
        )[0]

        assert "top_expert = self.gate(x).argmax(dim=-1, keepdim=True)" in forward_section
        assert "return torch.where(route_expert0, out0, out1)" in forward_section
        assert "torch.softmax(self.gate(x), dim=-1)" not in forward_section
        assert ".float().unsqueeze(-1)" not in forward_section
        assert "out0 * mask0 + out1 * mask1" not in forward_section

        model = model_cls(hidden_dim=8).eval()
        x = torch.randn(4, 8, dtype=torch.float32)
        with torch.inference_mode():
            logits = model.gate(x)
            top_expert = torch.softmax(logits, dim=-1).argmax(dim=-1, keepdim=True)
            expected = torch.where(top_expert == 0, model.expert0(x), model.expert1(x))
            actual = model(x)

        torch.testing.assert_close(actual, expected)
