import torch

from ch13.training_speed_common import TrainingSpeedConfig, TrainingSpeedModel, make_training_batch


def test_training_speed_model_output_shape_cpu():
    cfg = TrainingSpeedConfig(hidden_dim=64, num_layers=2, num_heads=4, seq_len=32, batch_size=2, vocab_size=256)
    model = TrainingSpeedModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    logits = model(input_ids)
    assert logits.shape == (cfg.batch_size, cfg.seq_len, cfg.vocab_size)


def test_training_speed_model_reuses_position_id_buffer_cpu():
    cfg = TrainingSpeedConfig(hidden_dim=64, num_layers=1, num_heads=4, seq_len=32, batch_size=2, vocab_size=256)
    model = TrainingSpeedModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, 16))

    _ = model(input_ids)

    assert model._position_ids.shape == (1, cfg.seq_len)
    assert model._position_ids.data_ptr() == model._position_ids[:, :16].data_ptr()


def test_make_training_batch_shapes():
    cfg = TrainingSpeedConfig(seq_len=16, batch_size=4, vocab_size=128)
    input_ids, targets = make_training_batch(cfg, torch.device("cpu"))
    assert input_ids.shape == (4, 16)
    assert targets.shape == (4, 16)
    assert input_ids.dtype == torch.int64
    assert targets.dtype == torch.int64
