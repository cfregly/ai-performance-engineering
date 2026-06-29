import torch

from ch15.speculative_decoding_common import TokenMLP as ChapterTokenMLP
from labs.speculative_decode.speculative_decode_common import TokenMLP as LabTokenMLP


def _assert_forward_into_matches_forward(model_cls) -> None:
    torch.manual_seed(1234)
    model = model_cls(
        vocab_size=17,
        hidden_size=8,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    logits_out = torch.empty(
        (token_ids.size(0), token_ids.size(1), model.vocab_size),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = model(token_ids)
        actual = model.forward_into(token_ids, logits_out)

    assert actual.data_ptr() == logits_out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_lab_token_mlp_forward_into_matches_forward() -> None:
    _assert_forward_into_matches_forward(LabTokenMLP)


def test_ch15_token_mlp_forward_into_matches_forward() -> None:
    _assert_forward_into_matches_forward(ChapterTokenMLP)
