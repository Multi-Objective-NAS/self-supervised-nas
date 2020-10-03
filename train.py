import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import hydra

from libs.SemiNAS.nas_bench.controller import NAO
from src.datasets import get_dataset
from src.utils import train_config_validator, get_loss, get_optimizer, get_miner, get_scheduler, get_trainer, load_pretrained_weights


def _get_target_parameters(model, freeze_encoder_decoder):
    if freeze_encoder_decoder:
        return list(model.encoder.mlp.parameters()) + list(model.encoder.regressor.parameters())
    else:
        return model.parameters()


@hydra.main(config_path='configs/train.yaml')
def train(cfg):
    print(cfg.pretty())
    train_config_validator(cfg)

    # Fix seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    cudnn.enabled = True
    cudnn.benchmark = True

    writer = SummaryWriter(log_dir='logs')
    controller = load_pretrained_weights(
        NAO(**cfg.controller).to(0), cfg.pretrained_model_path)
    dataset = get_dataset(writer=writer, **cfg.dataset)
    optimizer = get_optimizer(
        parameters=_get_target_parameters(controller, cfg.freeze_encoder_decoder), **cfg.optimizer)
    lr_scheduler = get_scheduler(optimizer=optimizer, **cfg.scheduler)

    get_trainer(
        controller=controller,
        dataset=dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        writer=writer,
        **cfg.trainer,
    ).train()


if __name__ == '__main__':
    train()
