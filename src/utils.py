import pathlib
import torch
from pytorch_metric_learning import losses, miners
from src import trainers


def pretrain_config_validator(cfg):
    assert cfg.dataset.samples_per_class > 1
    assert cfg.trainer.batch_size >= 2
    assert cfg.trainer.batch_size % cfg.dataset.samples_per_class == 0
    assert cfg.trainer.batch_size // cfg.dataset.samples_per_class >= 2


def train_config_validator(cfg):
    if cfg.pretrained_model_path:
        assert pathlib.Path(cfg.pretrained_model_path).exists()


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
