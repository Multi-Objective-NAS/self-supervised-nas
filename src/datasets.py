import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

import nasbench
from libs.SemiNAS.nas_bench import utils as seminas_utils


def get_dataset(name, **kwargs):
    assert name in ('train_nasbench101', 'pretrain_nasbench101')
    if name == 'pretrain_nasbench101':
        return PretrainNASBench101(**kwargs)
    if name == 'train_nasbench101':
        return TrainNASBench101(**kwargs)


class PretrainNASBench101(torch.utils.data.IterableDataset):
    def __init__(self, path, samples_per_class):
        if not pathlib.Path(path).is_absolute():
            path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

        self.engine = nasbench.api.NASBench(path)
        self.samples_per_class = samples_per_class

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            for pmatrix, pops in self._generate_isomorphic_graphs(matrix, ops):
                yield self._encode(pmatrix, pops), index

    def __len__(self):
        # 359082: Number of graphs with 7 vertices
        return 359082 * self.samples_per_class

    def _random_graph_generator(self):
        for index, key in enumerate(self.engine.hash_iterator()):
            fixed_stat, _ = self.engine.get_metrics_from_hash(key)
            matrix, ops = np.array(
                fixed_stat['module_adjacency']), fixed_stat['module_operations']
            if matrix.shape[0] == 7:
                yield (index, matrix, ops)

    def _generate_isomorphic_graphs(self, matrix, ops):
        vertices = matrix.shape[0]
        count = 0

        while count < self.samples_per_class:
            # Permute except first (input) and last (output)
            perm = np.random.permutation(range(1, vertices-1))
            perm = np.insert(perm, 0, 0)
            perm = np.insert(perm, vertices-1, vertices-1)

            pmatrix, pops = nasbench.lib.graph_util.permute_graph(matrix, ops, perm)
            if self.engine.is_valid(
                self.engine.api.ModelSpec(matrix=matrix, ops=ops)
            ):
                count += 1
                yield (pmatrix, pops)

        raise StopIteration

    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops)


class TrainNASBench101(torch.utils.data.Dataset):
    def __init__(self, path, batch_size, writer):
        if not pathlib.Path(path).is_absolute():
            path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

        self.engine = nasbench.api.NASBench(path)
        self.batch_size = batch_size
        self.writer = writer

        self.dataset = []
        self.seqs = []

    def _query(self, matrix, ops):
        arch = nasbench.api.ModelSpec(matrix=matrix, ops=ops)
        return self.engine.query(arch)['validation_accuracy']

    def _append(self, seq, perf):
        self.seqs.append(seq)
        self.dataset.append({
            'encoder_input': torch.LongTensor(seq),
            'decoder_input': torch.LongTensor([0] + seq[:-1]),
            'encoder_target': torch.FloatTensor([max(perf-0.8, 0.0) * 5.0]),
            'decoder_target': torch.LongTensor(seq),
        })
        self.writer.add_scalar(
            f'Metric/performance', perf, len(self.dataset))

    def prepare(self, count):
        for key in self.engine.hash_iterator():
            fixed_stat, _ = self.engine.get_metrics_from_hash(key)
            matrix, ops = np.array(fixed_stat['module_adjacency']), fixed_stat['module_operations']
            if matrix.shape[0] == 7:
                self._append(
                    seq=seminas_utils.convert_arch_to_seq(matrix, ops),
                    perf=self._query(matrix, ops)
                )
                if len(self.dataset) >= count:
                    break

    def add(self, seqs):
        for seq in seqs:
            matrix, ops = seminas_utils.convert_seq_to_arch(seq)
            self._append(
                seq=seminas_utils.convert_arch_to_seq(matrix, ops),
                perf=self._query(matrix, ops),
            )

    def is_valid(self, seq):
        matrix, ops = seminas_utils.convert_seq_to_arch(seq)
        arch = nasbench.api.ModelSpec(matrix=matrix, ops=ops)
        return self.engine.is_valid(arch) and len(arch.ops) == 7 and seq not in self.seqs

    def shuffled(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def sorted(self, count):
        indices = sorted(
            range(len(self.dataset)),
            key=lambda i: self.dataset[i]['encoder_target'],
            reverse=True
        )
        return torch.utils.data.DataLoader(
            dataset=[self.dataset[i]['encoder_input'] for i in indices[:count]],
            shuffle=True,
            batch_size=self.batch_size,
        )
