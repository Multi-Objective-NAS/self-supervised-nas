import pathlib
import umap
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class TensorboardHook:
    def __init__(self):
        self.writer = SummaryWriter(log_dir='logs')
        self.visualizer = umap.UMAP()
        self.total_iterations = 0

    def end_of_iteration_hook(self, trainer):
        for loss_name, loss in trainer.losses.items():
            self.writer.add_scalar(
                f'Loss/{loss_name}', loss, self.total_iterations)
        for sched_name, sched in trainer.lr_schedulers.items():
            for lr_index, lr in enumerate(sched.get_last_lr()):
                self.writer.add_scalar(
                    f'LR/{sched_name}/{lr_index}', lr, self.total_iterations)
        if trainer.visualize_scratchpad:
            img = self._get_embedding_visualization(
                **trainer.visualize_scratchpad)
            self.writer.add_image('UMAP Embeddings', img,
                                  self.total_iterations)
            trainer.visualize_scratchpad.clear()

        self.total_iterations += 1

    def _get_embedding_visualization(self, embeddings, labels):
        reduced_embeddings = self.visualizer.fit_transform(embeddings)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        fig, ax = plt.subplots(figsize=(4, 4))
        label_color = [plt.cm.nipy_spectral(
            i) for i in np.linspace(0, 0.9, num_classes)]
        for label_index, label in enumerate(unique_labels):
            indices = labels == label
            ax.plot(reduced_embeddings[indices, 0],
                    reduced_embeddings[indices, 1],
                    ".", markersize=1, color=label_color[label_index])
        fig.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            (3,) + fig.canvas.get_width_height())
        return image_from_plot


class ModelSaverHook:
    def __init__(self):
        self.total_epochs = 0
        pathlib.Path('weights').mkdir(exist_ok=True)

    def end_of_epoch_hook(self, trainer):
        for model_name, model in trainer.models.items():
            torch.save(model.state_dict(),
                       f'weights/{model_name}-{self.total_epochs}.h5')
        self.total_epochs += 1
