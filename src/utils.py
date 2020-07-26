import torch
from pytorch_metric_learning import losses, miners


def config_validator(cfg):
    assert cfg.dataset.samples_per_class > 1
    assert cfg.trainer.batch_size >= 2
    assert cfg.trainer.batch_size % cfg.dataset.samples_per_class == 0
    assert cfg.trainer.batch_size // cfg.dataset.samples_per_class >= 2


def get_optimizer(name, parameters, **kwargs):
    return {
        'Adam': torch.optim.Adam(parameters, **kwargs),
    }[name]


def get_miner(name, **kwargs):
    return getattr(miners, name)(**kwargs)


def get_loss(name, **kwargs):
    return getattr(losses, name)(**kwargs)
