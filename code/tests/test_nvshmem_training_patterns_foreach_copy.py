from __future__ import annotations

import torch

from ch04.nvshmem_training_patterns import NVSHMEMGradientSync


def _parameter(shape: tuple[int, ...], values: torch.Tensor) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
    parameter.grad = values.to(dtype=torch.float32).reshape(shape).clone()
    return parameter


def test_foreach_pack_and_unpack_round_trip_through_disjoint_bucket_views() -> None:
    parameters = [
        _parameter((2, 3), torch.arange(6, dtype=torch.float32)),
        _parameter((4,), torch.arange(4, dtype=torch.float32) + 10),
        _parameter((1, 2), torch.tensor([21.0, 22.0])),
    ]
    sync = NVSHMEMGradientSync(parameters, world_size=2)
    expected_packed = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])

    gradients, bucket_views = sync._pack_gradients()

    assert sync.bucket.tensor is not None
    torch.testing.assert_close(sync.bucket.tensor, expected_packed, rtol=0, atol=0)
    assert [view.storage_offset() for view in bucket_views] == [0, 6, 10]
    assert all(
        view.untyped_storage().data_ptr() == sync.bucket.tensor.untyped_storage().data_ptr()
        for view in bucket_views
    )
    intervals = [
        (view.storage_offset(), view.storage_offset() + view.numel()) for view in bucket_views
    ]
    assert all(
        left_end <= right_start
        for (_, left_end), (right_start, _) in zip(intervals, intervals[1:], strict=False)
    )

    sync.bucket.tensor.mul_(0.25)
    sync._unpack_gradients(gradients, bucket_views)

    for parameter, expected in zip(parameters, expected_packed.split([6, 4, 2]), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            expected.view_as(parameter).mul(0.25),
            rtol=0,
            atol=0,
        )


def test_foreach_pack_preserves_active_gradient_order_and_stale_tail() -> None:
    first = _parameter((2,), torch.tensor([1.0, 2.0]))
    inactive = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    last = _parameter((2, 2), torch.tensor([3.0, 4.0, 5.0, 6.0]))
    sync = NVSHMEMGradientSync([first, inactive, last], world_size=2)
    assert sync.bucket.tensor is not None
    sync.bucket.tensor.fill_(-7.0)

    gradients, bucket_views = sync._pack_gradients()

    assert len(gradients) == 2
    assert gradients[0] is first.grad
    assert gradients[1] is last.grad
    assert [view.storage_offset() for view in bucket_views] == [0, 2]
    torch.testing.assert_close(
        sync.bucket.tensor[:6],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sync.bucket.tensor[6:],
        torch.full((3,), -7.0),
        rtol=0,
        atol=0,
    )


def test_foreach_full_production_bucket_preserves_replaced_gradients() -> None:
    parameters = []
    for layer in range(256):
        parameters.append(_parameter((64, 64), torch.arange(4096) + layer))
        parameters.append(_parameter((64,), torch.arange(64) - layer))
    # Match the production FP32 parameters and full 1,064,960-element bucket.
    sync = NVSHMEMGradientSync(parameters, world_size=2)
    assert sync.bucket.numel == 1_064_960
    for step in range(3):
        for parameter in parameters:
            parameter.grad = parameter.grad.add(float(step))
        expected = torch.cat([p.grad.reshape(-1) for p in parameters])
        gradients, views = sync._pack_gradients()
        assert len(gradients) == len(views) == 512
        torch.testing.assert_close(sync.bucket.tensor, expected, rtol=0, atol=0)
        sync.bucket.tensor.mul_(0.5)
        sync._unpack_gradients(gradients, views)
        torch.testing.assert_close(
            torch.cat([p.grad.reshape(-1) for p in parameters]),
            expected * 0.5,
            rtol=0,
            atol=0,
        )
