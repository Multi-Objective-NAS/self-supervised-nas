import logging

import torch
import tqdm
from pytorch_metric_learning import trainers


class GraphEmbeddingTrainer(trainers.MetricLossOnly):
    def __init__(self, num_epochs, visualize_step, **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.visualize_step = visualize_step
        self.visualize_scratchpad = {}

    def calculate_loss(self, curr_batch):
        (encoder_input, decoder_input), labels = curr_batch
        encoder_input = torch.stack(encoder_input, dim=1).to(0)
        decoder_input = torch.stack(decoder_input, dim=1).to(0)
        labels = labels.to(0)

        enc_outputs, _, embeddings, _ = self.models['trunk'].encoder(
            encoder_input)
        dec_hidden = (embeddings.unsqueeze(0), embeddings.unsqueeze(0))
        dec_outputs, _ = self.models['trunk'].decoder(
            decoder_input, dec_hidden, enc_outputs)

        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses['metric_loss'] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple)
        self.losses['reconstruction_loss'] = self.get_reconstruction_loss(
            dec_outputs, encoder_input)

        if self.iteration % self.visualize_step == 0:
            self.visualize_scratchpad['embeddings'] = embeddings.detach().cpu()
            self.visualize_scratchpad['labels'] = labels.detach().cpu()

    def initialize_dataloader(self):
        logging.info("Initializing dataloader")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.batch_size),
            sampler=self.sampler,
            drop_last=True,
            num_workers=self.dataloader_num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=False
        )
        if not self.iterations_per_epoch:
            self.iterations_per_epoch = len(self.dataloader) // self.batch_size
        logging.info("Initializing dataloader iterator")
        self.dataloader_iter = iter(self.dataloader)
        logging.info("Done creating dataloader iterator")

    def compute_embeddings(self, data):
        raise NotImplementedError

    def get_reconstruction_loss(self, output, target):
        return self.loss_funcs['reconstruction_loss'](
            output.contiguous().view(-1, output.size(-1)),
            target.view(-1)
        )

    def allowed_model_keys(self):
        return super().allowed_model_keys()

    def allowed_loss_funcs_keys(self):
        return super().allowed_loss_funcs_keys() + ['reconstruction_loss']

    def train(self):
        self.initialize_dataloader()
        for self.epoch in range(1, self.num_epochs + 1):
            self.set_to_train()
            logging.info("TRAINING EPOCH %d" % self.epoch)
            pbar = tqdm.tqdm(range(self.iterations_per_epoch))
            for self.iteration in pbar:
                self.forward_and_backward()
                self.end_of_iteration_hook(self)
                pbar.set_description(' '.join(
                    f'{k}={v:.5f}' for k, v in self.losses.items()))
                self.step_lr_schedulers(end_of_epoch=False)
            self.step_lr_schedulers(end_of_epoch=True)
            if self.end_of_epoch_hook(self) is False:
                break
