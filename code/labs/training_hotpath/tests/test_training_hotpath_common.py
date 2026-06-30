import torch

from labs.training_hotpath.training_hotpath_common import PackedLinear


def test_packed_linear_workspace_reuses_larger_first_dim_capacity():
    layer = PackedLinear(4, 8, generator=torch.Generator().manual_seed(0))
    device = torch.device("cpu")

    first = layer._workspace(
        "_packed_input",
        (8, 4),
        device=device,
        dtype=torch.float32,
    )
    first_ptr = first.data_ptr()

    smaller = layer._workspace(
        "_packed_input",
        (3, 4),
        device=device,
        dtype=torch.float32,
    )

    assert smaller.shape == (3, 4)
    assert smaller.data_ptr() == first_ptr
    assert layer._packed_input.shape == (8, 4)

    wider = layer._workspace(
        "_packed_input",
        (3, 5),
        device=device,
        dtype=torch.float32,
    )

    assert wider.shape == (3, 5)
    assert wider.data_ptr() != first_ptr
