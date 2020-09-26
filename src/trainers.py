import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pytorch_metric_learning import trainers

from libs.SemiNAS.nasbench.controller import NAO


class GraphEmbeddingTrainer(trainers.MetricLossOnly):
    def __init__(self, num_epochs, visualize_step, **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.visualize_step = visualize_step
        self.visualize_scratchpad = {}

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        data = torch.stack(data, dim=1).to(0)
        labels = labels.to(0)
 
        enc_outputs, _, embeddings, _ = self.models['trunk'].encoder(
            data)
        dec_hidden = (embeddings.unsqueeze(0), embeddings.unsqueeze(0))
        dec_outputs, _ = self.models['trunk'].decoder(
            data, dec_hidden, enc_outputs)
 
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses['metric_loss'] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple)
        self.losses['reconstruction_loss'] = self.get_reconstruction_loss(
            dec_outputs, data)

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
        for self.epoch in range(1, self.num_epochs+1):
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
        # train_controller() in SemiNAS : prepare dataLoader
        self.dataset.prepare(self.number_of_initial_archs)
        for self.outer_epoch in range(1, self.outer_epochs+1):
            for self.inner_epoch in tqdm.tqdm(range(1, self.inner_epochs)):
                # controller_train() in SemiNAS
                for self.iteration, sample in enumerate(self.dataset.shuffled(), start=1):
                    self.single_iteration(sample)
            self.dataset.add(self.generate_architectures())

    def generate_architectures(self):
        generated_archs = []
        step_size = 0
        while step_size < self.max_step_size and len(generated_archs) < self.number_of_candidate_archs:
            step_size += 1
            for sample in self.dataset.sorted(self.number_of_seed_archs):
                encoder_input_unsorted = sample['encoder_input'].long() # shape maybe (batch size, max seq length, word length)
                input_len_unsorted = sample['input_len']
                # sort input batch
                input_len, sort_index = torch.sort(input_len_unsorted, 0, descending=True)
                input_len = input_len.numpy().tolist()
                encoder_input = torch.index_select(encoder_input_unsorted, 0, sort_index)
                # move to gpu
                encoder_input = utils.move_to_cuda(encoder_input)

                self.controller.zero_grad()
                new_archs, _ = self.controller.generate_new_arch(encoder_input, input_len, step, direction='+')
                new_archs = new_archs.data.squeeze().tolist()
                for arch in new_archs:
                    if self.dataset.is_valid(arch):
                        generated_archs.append(arch)
                        if len(generated_archs) < self.number_of_candidate_archs:
                            break
        return generated_archs

    def _move_to_cuda(tensor):
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor

    def single_iteration(self, sample):
        encoder_input_unsorted = sample['encoder_input'].long() # shape maybe (batch size, max seq length, word length)
        encoder_target_unsorted = sample['encoder_target'].float()
        decoder_input_unsorted = sample['decoder_input'].long()
        decoder_target_unsorted = sample['decoder_target'].long()
        input_len_unsorted = sample['input_len']
        
        # sort input batch
        input_len, sort_index = torch.sort(input_len_unsorted, 0, descending=True)
        input_len = input_len.numpy().tolist()
        encoder_input = torch.index_select(encoder_input_unsorted, 0, sort_index)
        encoder_target = torch.index_select(encoder_target_unsorted, 0, sort_index)
        decoder_input = torch.index_select(decoder_input_unsorted, 0, sort_index)
        decoder_target = torch.index_select(decoder_target_unsorted, 0, sort_index)

        # move to cuda
        encoder_input = self._move_to_cuda(encoder_input) # shape maybe (batch size, max seq length, word length)
        encoder_target = self._move_to_cuda(encoder_target)
        decoder_input = self._move_to_cuda(decoder_input)
        decoder_target = self._move_to_cuda(decoder_target)

        self.optimizer.zero_grad()
        predict_value, log_prob, _ = self.controller(encoder_input, input_len, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = self.loss_tradeoff * loss_1 + (1 - self.loss_tradeoff) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.gradient_bound)
        self.optimizer.step()

        self.total_iterations += 1
        self.writer.add_scalar(
            f'Loss/MSE', loss_1, self.total_iterations
        )
        self.writer.add_scalar(
            f'Loss/NLL', loss_2, self.total_iterations
        )
        self.writer.add_scalar(
            f'Loss/Total', loss, self.total_iterations
        )
