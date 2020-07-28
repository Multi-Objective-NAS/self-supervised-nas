import hydra
import torch
import torch.nn
from pytorch_metric_learning import losses, miners
from libs.SemiNAS.nas_bench.controller import NAO
from src.datasets import get_dataset
from src.utils import config_validator, get_loss, get_optimizer, get_miner, get_scheduler
from src.models import GraphEmbeddingTrainer
from src.hooks import TensorboardHook, ModelSaverHook


@hydra.main(config_path='configs/experiment.yaml')
def train(cfg):
    print(cfg.pretty())
    config_validator(cfg)

    models = {'trunk': NAO(**cfg.controller).to(0)}
    dataset = get_dataset(**cfg.dataset)
    optimizers = {'trunk_optimizer': get_optimizer(parameters=models['trunk'].parameters(), **cfg.optimizer)}
    lr_schedulers = {'trunk_scheduler_by_iteration': get_scheduler(optimizer=optimizers['trunk_optimizer'], **cfg.scheduler)}
    loss_funcs = {'reconstruction_loss': torch.nn.NLLLoss(), 'metric_loss': get_loss(**cfg.loss)}
    mining_funcs = {"tuple_miner": get_miner(**cfg.miner)}
    end_of_iteration_hook = TensorboardHook().end_of_iteration_hook
    end_of_epoch_hook = ModelSaverHook().end_of_epoch_hook

    GraphEmbeddingTrainer(
        models=models,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,
        dataset=dataset,
        end_of_iteration_hook=end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
        **cfg.trainer,
    ).train()

if __name__ == '__main__':
    train()
