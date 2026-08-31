"""Real matched model/data/update checks for both ZeRO comparison variants."""

import copy
import importlib
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from labs.train_distributed.zero2_common import (
    build_model,
    build_training_components,
    parse_args,
    training_step,
)


def test_all_four_entrypoints_select_identical_model_and_wrapper_workload():
    outputs = []
    for variant in ("single", "multigpu"):
        arguments = []
        for mode in ("baseline", "optimized"):
            suffix = "_multigpu" if variant == "multigpu" else ""
            module = importlib.import_module(f"labs.train_distributed.{mode}_zero2{suffix}")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(25)
                model = module._build_model(8, torch.device("cpu"))
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert sum(isinstance(layer, torch.nn.GELU) for layer in model) == 6
            outputs.append(model(torch.arange(24.).reshape(3, 8)))
            args = list(module.get_benchmark()._base_args)
            mode_index = args.index("--mode")
            del args[mode_index:mode_index + 2]
            arguments.append(args)
        assert arguments[0] == arguments[1]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


@pytest.mark.parametrize("option,value", [("--steps", "0"), ("--grad-accum", "0"),
                                          ("--batch-size", "-1"), ("--learning-rate", "nan")])
def test_invalid_workload_arguments_fail(option, value):
    with pytest.raises(SystemExit):
        parse_args([option, value])


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Actual Gloo required")
def test_both_actual_training_steps_match_independent_dense_reference(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=(tmp_path / "rdzv").as_uri(),
                            rank=0, world_size=1, timeout=timedelta(seconds=20))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            initial = build_model(8, torch.device("cpu"))
        # Exercise actual clipping rather than merely selecting the config.
        with torch.no_grad():
            initial[-1].bias.fill_(100)
        snapshots = []
        for optimized in (False, True):
            model = copy.deepcopy(initial)
            reference = copy.deepcopy(initial)
            ddp, optimizer = build_training_components(model, .01, optimized=optimized)
            dense = torch.optim.AdamW(reference.parameters(), lr=.01, betas=(.9, .95),
                                      weight_decay=.05, fused=True)
            x, y = torch.empty(5, 8), torch.empty(5, 8)
            pointers = (x.data_ptr(), y.data_ptr())
            actual_rng = torch.Generator().manual_seed(711)
            reference_rng = torch.Generator().manual_seed(711)
            before = [p.detach().clone() for p in model.parameters()]
            for _ in range(4):  # one warmup-equivalent update and three measured updates
                actual_loss = training_step(ddp, optimizer, x, y, actual_rng, 2)
                dense.zero_grad(set_to_none=True)
                for _ in range(2):
                    rx = torch.randn(5, 8, generator=reference_rng)
                    ry = torch.randn(5, 8, generator=reference_rng)
                    expected_loss = torch.nn.functional.mse_loss(reference(rx), ry) / 2
                    expected_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                assert float(norm) > 1
                dense.step()
                assert pointers == (x.data_ptr(), y.data_ptr())
                torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
                for got, expected in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
            assert sum(float((p.detach() - old).abs().sum()) for p, old in zip(model.parameters(), before)) > 0
            snapshots.append([p.detach().clone() for p in model.parameters()])
        for baseline, optimized in zip(*snapshots, strict=True):
            torch.testing.assert_close(baseline, optimized, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
