import pathlib
import umap
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class TensorboardHook:
    def __init__(self, visualizers):
        self.total_iterations = 0
        self.writer = SummaryWriter(log_dir='logs')
        self.visualizers = visualizers

    def end_of_iteration_hook(self, trainer):
        for loss_name, loss in trainer.losses.items():
            self.writer.add_scalar(
                f'Loss/{loss_name}', loss, self.total_iterations)
        for sched_name, sched in trainer.lr_schedulers.items():
            for lr_index, lr in enumerate(sched.get_last_lr()):
                self.writer.add_scalar(
                    f'LR/{sched_name}/{lr_index}', lr, self.total_iterations)
        for key, value in trainer.scalar_scratchpad.items():
            self.writer.add_scalar(
                f'Metric/{key}', value, self.total_iterations)

        if trainer.visualize_scratchpad:
            for viz in self.visualizers:
                name = f'{viz.metric}/{viz}'
                img = self._get_embedding_visualization(
                    viz.fit_transform, **trainer.visualize_scratchpad)
                self.writer.add_image(
                    name, img, self.total_iterations, dataformats='HWC')
            trainer.visualize_scratchpad.clear()

        self.total_iterations += 1

    def _get_embedding_visualization(self, fit_transform, embeddings, labels):
        reduced_embeddings = fit_transform(embeddings)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 16:
            unique_labels = unique_labels[:16]
        num_classes = len(unique_labels)

        fig, ax = plt.subplots(figsize=(4, 4))
        label_color = [plt.cm.nipy_spectral(
            i) for i in np.linspace(0, 0.9, num_classes)]
        for label_index, label in enumerate(unique_labels):
            indices = labels == label
            ax.plot(reduced_embeddings[indices, 0],
                    reduced_embeddings[indices, 1],
                    ".", markersize=4, color=label_color[label_index])
        fig.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height() + (3, ))
        plt.close(fig)
        return image_from_plot


class ModelSaverHook:
    def __init__(self):
        self.total_epochs = 0
        pathlib.Path('weights').mkdir(exist_ok=True)

    def end_of_epoch_hook(self, trainer):
        for model_name, model in trainer.models.items():
            torch.save(model.state_dict(),
                       f'weights/{model_name}-{self.total_epochs}.h5')
        for optim_name, optim in trainer.optimizers.items():
            torch.save(optim.state_dict(),
                       f'weights/{optim_name}-{self.total_epochs}.h5')
        self.total_epochs += 1
