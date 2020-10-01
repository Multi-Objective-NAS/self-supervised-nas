import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pytorch_metric_learning import trainers

from libs.SemiNAS.nas_bench.controller import NAO


class GraphEmbeddingTrainer(trainers.MetricLossOnly):
    def __init__(self, num_epochs, visualize_step, **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.visualize_step = visualize_step
        self.visualize_scratchpad = {}
        self.scalar_scratchpad = {}

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
        self.scalar_scratchpad['accuracy'] = self.get_accuracy(
            dec_outputs, encoder_input
        )

        if self.iteration % self.visualize_step == 0:
            self.visualize_scratchpad['embeddings'] = embeddings.detach().cpu()
            self.visualize_scratchpad['labels'] = labels.detach().cpu()

    def initialize_dataloader(self):
        logging.info("Initializing dataloader")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.batch_size),
            sampler=None,
            drop_last=True,
            num_workers=1,
            collate_fn=None,
            shuffle=False,
            pin_memory=False,
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

    def get_accuracy(self, output, target):
        output_argmax = output.contiguous().view(-1, output.size(-1)).argmax(-1)
        target_argmax = target.view(-1)
        return (output_argmax == target_argmax).float().mean().item()

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


class NAOTrainer:
    def __init__(
        self,
        controller,
        dataset,
        optimizer,
        lr_scheduler,
        writer,
        outer_epochs,
        inner_epochs,
        number_of_initial_archs,
        number_of_seed_archs,
        number_of_candidate_archs,
        max_step_size,
        loss_tradeoff,
        gradient_bound,
    ):
        self.controller = controller
        self.dataset = dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.outer_epochs = outer_epochs
        self.inner_epochs = inner_epochs
        self.number_of_initial_archs = number_of_initial_archs
        self.number_of_seed_archs = number_of_seed_archs
        self.number_of_candidate_archs = number_of_candidate_archs
        self.max_step_size = max_step_size
        self.loss_tradeoff = loss_tradeoff
        self.gradient_bound = gradient_bound
        self.total_iterations = 0

    def train(self):
        self.dataset.prepare(self.number_of_initial_archs)
        for self.outer_epoch in range(1, self.outer_epochs+1):
            for self.inner_epoch in tqdm.tqdm(range(1, self.inner_epochs)):
                for self.iteration, sample in enumerate(self.dataset.shuffled(), start=1):
                    self.single_iteration(sample)
                self.lr_scheduler.step()
            candidates = self.generate_architectures()
            self.dataset.add(candidates)
            self.writer.add_scalar(
                f'Generated Architectures', len(candidates), self.outer_epoch
            )

    def generate_architectures(self):
        generated_archs = []
        step_size = 0
        while step_size < self.max_step_size and len(generated_archs) < self.number_of_candidate_archs:
            step_size += 1
            for sample in self.dataset.sorted(self.number_of_seed_archs):
                encoder_input = sample.to(0)
                self.controller.zero_grad()
                new_archs, _ = self.controller.generate_new_arch(encoder_input, step_size, direction='+')
                new_archs = new_archs.data.squeeze().tolist()
                for arch in new_archs:
                    if self.dataset.is_valid(arch) and arch not in generated_archs:
                        generated_archs.append(arch)
                        if len(generated_archs) >= self.number_of_candidate_archs:
                            break
        return generated_archs

    def single_iteration(self, sample):
        for k in sample.keys():
            sample[k] = sample[k].to(0)

        self.optimizer.zero_grad()
        predict_value, log_prob, _ = self.controller(
            sample['encoder_input'],
            sample['decoder_input'],
        )
        loss_1 = F.mse_loss(
            predict_value.squeeze(),
            sample['encoder_target'].squeeze(),
        )
        flat_log_prob = log_prob.contiguous().view(-1, log_prob.size(-1))
        loss_2 = F.nll_loss(
            flat_log_prob,
            sample['decoder_target'].view(-1),
        )
        loss = self.loss_tradeoff * loss_1 + (1 - self.loss_tradeoff) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.gradient_bound)
        self.optimizer.step()

        self.total_iterations += 1
        self.writer.add_scalar(
            'Metric/Accuracy',
            (flat_log_prob.argmax(-1) == sample['decoder_target'].view(-1)).float().mean().item(),
            self.total_iterations
        )
        self.writer.add_scalar(
            'Loss/MSE', loss_1, self.total_iterations
        )
        self.writer.add_scalar(
            'Loss/NLL', loss_2, self.total_iterations
        )
        self.writer.add_scalar(
            'Loss/Total', loss, self.total_iterations
        )
        for lr_index, lr in enumerate(self.lr_scheduler.get_last_lr()):
            self.writer.add_scalar(
                f'LR/{lr_index}', lr, self.total_iterations)
