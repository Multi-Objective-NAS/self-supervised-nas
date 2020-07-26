import hydra
import torch
import torch.nn
from pytorch_metric_learning import losses, miners
from libs.SemiNAS.nas_bench.controller import NAO
from src.datasets import get_dataset
from src.utils import config_validator, get_loss, get_optimizer, get_miner
from src.models import GraphEmbeddingTrainer


@hydra.main(config_path='configs/experiment.yaml')
def train(cfg):
    print(cfg.pretty())
    config_validator(cfg)

    dataset = get_dataset(**cfg.dataset)
    models = {'trunk': NAO(**cfg.controller)}
    optimizers = {'trunk_optimizer': get_optimizer(parameters=models['trunk'].parameters(), **cfg.optimizer)}
    loss_funcs = {'reconstruction_loss': torch.nn.NLLLoss(), 'metric_loss': get_loss(**cfg.loss)}
    mining_funcs = {"tuple_miner": get_miner(**cfg.miner)}

    trainer = GraphEmbeddingTrainer(
        models=models,
        optimizers=optimizers,
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,
        dataset=dataset,
        **cfg.trainer,
    ).train()

if __name__ == '__main__':
    train()
