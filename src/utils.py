import pathlib
import torch
from pytorch_metric_learning import losses, miners
from src import trainers


def pretrain_config_validator(cfg):
    assert cfg.dataset.samples_per_class > 1
    assert cfg.trainer.batch_size >= 2
    assert cfg.trainer.batch_size % cfg.dataset.samples_per_class == 0
    assert cfg.trainer.batch_size // cfg.dataset.samples_per_class >= 2
    assert cfg.controller.source_length == cfg.dataset.max_seq_len
    assert cfg.controller.encoder_length == cfg.dataset.max_seq_len
    assert cfg.controller.decoder_length == cfg.dataset.max_seq_len


def train_config_validator(cfg):
    if cfg.pretrained_model_path:
        assert pathlib.Path(cfg.pretrained_model_path).exists()
    assert cfg.controller.source_length == cfg.dataset.max_seq_len
    assert cfg.controller.encoder_length == cfg.dataset.max_seq_len
    assert cfg.controller.decoder_length == cfg.dataset.max_seq_len


def load_pretrained_weights(model, path):
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model


def get_optimizer(name, parameters, **kwargs):
    return getattr(torch.optim, name)(parameters, **kwargs)


def get_scheduler(name, optimizer, **kwargs):
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)


def get_miner(name, **kwargs):
    return getattr(miners, name)(**kwargs)


def get_loss(name, **kwargs):
    return getattr(losses, name)(**kwargs)


def get_trainer(name, **kwargs):
    return getattr(trainers, name)(**kwargs)
