"""The independent FP64 reference must also support its declared graph path."""
from __future__ import annotations

import pytest
import torch

from labs.nvfp4_group_gemm.reference_math import dequantize_fp4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
def test_fp4_reference_graph_replay_reads_changing_packed_values() -> None:
    packed = torch.arange(256, dtype=torch.uint8, device="cuda").reshape(1, 128, 1)
    scales = torch.ones((1, 16, 1), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dequantize_fp4(packed, scales, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dequantize_fp4(packed, scales, 256)

    # All 16 signed E2M1 codes, then changed scale and input, use a CPU oracle.
    magnitudes = (0, .5, 1, 1.5, 2, 3, 4, 6)
    for factor in (1, 2):
        packed.bitwise_xor_(0xFF)
        scales.fill_(factor)
        raw = packed.cpu().flatten().tolist()
        expected = torch.tensor([
            factor * (-1 if code & 8 else 1) * magnitudes[code & 7]
            for byte in raw for code in (byte & 15, byte >> 4)
        ], dtype=torch.float64).reshape(1, 256, 1)
        graph.replay()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
