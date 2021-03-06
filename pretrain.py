import logging
import hydra
import torch.nn
import umap
from pytorch_metric_learning import losses, miners
from libs.SemiNAS.nas_bench.controller import NAO
from src.datasets import get_dataset
from src.utils import pretrain_config_validator, get_loss, get_optimizer, get_miner, get_scheduler, get_trainer, load_pretrained_weights, fix_seed
from src.hooks import TensorboardHook, ModelSaverHook


@hydra.main(config_path='configs/pretrain.yaml')
def pretrain(cfg):
    print(cfg.pretty())
    pretrain_config_validator(cfg)
    fix_seed(cfg.seed)

    controller = load_pretrained_weights(
        NAO(**cfg.controller).to(0), cfg.pretrained_model_path)
    models = {'trunk': controller}
    dataset = get_dataset(seed=cfg.seed, **cfg.dataset)
    optimizers = {'trunk_optimizer': get_optimizer(parameters=models['trunk'].parameters(), **cfg.optimizer)}
    lr_schedulers = {'trunk_scheduler_by_iteration': get_scheduler(optimizer=optimizers['trunk_optimizer'], **cfg.scheduler)}
    loss_funcs = {'reconstruction_loss': torch.nn.NLLLoss(), 'metric_loss': get_loss(**cfg.loss)}
    mining_funcs = {"tuple_miner": get_miner(**cfg.miner)}
    visualizers = [umap.UMAP(**params) for params in cfg.visualizers]
    end_of_iteration_hook = TensorboardHook(visualizers).end_of_iteration_hook
    end_of_epoch_hook = ModelSaverHook().end_of_epoch_hook
    get_trainer(
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
    pretrain()
