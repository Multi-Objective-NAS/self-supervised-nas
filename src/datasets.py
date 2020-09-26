import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

from nasbench import api as api101
from nasbench.lib import graph_util
from nas_201_api import api_201 as api201
from libs.SemiNAS.nasbench import utils as seminas_utils

def get_dataset(name, path, **kwargs):
    assert name in ('train_nasbench101', 'pretrain_nasbench101', 'train_nasbench201', 'pretrain_nasbench201')
    
    if not pathlib.Path(path).is_absolute():
        path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

    mode, dataset_name = name.split('_')
    if mode == 'pretrain':
        if dataset_name == 'nasbench101':
            return PretrainNASBench(engine=api101.NASBench(path), model_spec=api101.ModelSpec, **kwargs)
        if dataset_name == 'nasbench201':
            return PretrainNASBench(engine=api201.NASBench201API(path), model_spec=api201.ModelSpec, **kwargs)
    if mode == 'train':
        if dataset_name == 'nasbench101':
            return TrainNASBench(engine=api101.NASBench(path), model_spec=api101.ModelSpec, **kwargs)
        if dataset_name == 'nasbench201':
            return TrainNASBench(engine=api201.NASBench201API(path), model_spec=api201.ModelSpec, **kwargs)

class PretrainNASBench(torch.utils.data.IterableDataset):
    def __init__(self, engine, model_spec, samples_per_class, graph_modify_ratio):
        self.engine = engine
        self.model_spec = model_spec
        self.samples_per_class = samples_per_class

        # list of possible operations ex.[ CONV 3x3, MAXPOOL 3x3, ... ]
        self.search_space = engine.search_space

        # Find dataset length
        length = 0
        for index, key in enumerate(engine.hash_iterator()):
            arch = engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            # remove if matrix.shape[0] == 7:
            length += 1
        self._dataset_length = length

        self.graph_modifier = GraphModifier(
            validate=self.is_valid,
            operations=set(self.search_space) - set(["input", "ouput"]),
            samples_per_class=samples_per_class,
            **graph_modify_ratio)

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            for pmatrix, pops in self.graph_modifier.generate_modified_models(matrix, ops):
                seq = self._encode(pmatrix, pops)
                yield (seq, [0] + seq[:-1]), index

    def __len__(self):
        return self._dataset_length * self.samples_per_class

    def _random_graph_generator(self):
        for index, key in enumerate(self.engine.hash_iterator()):
            arch = self.engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            # remove if matrix.shape[0] == 7:
            yield (index, matrix, ops)

    def _generate_isomorphic_graphs(self, matrix, ops):
        """Substituted by graph_modifier.generate_modified_models"""
        vertices = matrix.shape[0]
        count = 0

        while count < self.samples_per_class:
            # Permute except first (input) and last (output)
            perm = np.random.permutation(range(1, vertices - 1))
            perm = np.insert(perm, 0, 0)
            perm = np.insert(perm, vertices - 1, vertices - 1)

            pmatrix, pops = graph_util.permute_graph(matrix, ops, perm)
            modelspec = self.engine.get_modelspec(matrix=pmatrix, ops=pops)
            if self.engine.is_valid(modelspec):
                count += 1
                yield (pmatrix, pops)

        raise StopIteration

    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops, self.search_space)

    def is_valid(self, matrix, ops):
        model = self.model_spec(matrix=matrix, ops=ops)
        if not model.valid_spec:
            return False
        if model.ops != ops:
            return False
        if (model.matrix != matrix).any():
            return False
        return self.engine.is_valid(model)

class TrainNASBench(torch.utils.data.Dataset):
    def __init__(self, engine, model_spec, batch_size, writer):
        
        self.engine = engine
        self.model_spec = model_spec
        self.batch_size = batch_size
        self.writer = writer

        # pool
        self.dataset = []
        self.seqs = []
        self.seq_len = []
        self.targets = []

    def prepare(self, count):
        # At first, arch_pool is set randomly.
        for key in self.engine.hash_iterator():
            arch = self.engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            seq = seminas_utils.convert_arch_to_seq(matrix, ops, self.engine.search_space)
            
            # remove "if matrix.shape[0] == 7"
            self.seqs.append(seq)
            self.seq_len.append(len(seq))
            self.targets.append(self.engine.query(arch, option='valid'))
            if len(self.dataset) >= count:
                break

        # prepare for self.dataset
        self.max_len = max(len(seq) for seq in self.seqs)
        for index in range(len(self.seqs)):
            encoder_input = self.seqs[index] + [0 for _ in range(self.max_len - len(self.seqs[index]))] # fix length as max_len
            len_input = self.seq_len[index]
            encoder_target = [self.targets[index]]
            decoder_input = [0] + encoder_input[:-1]
            sample = {
                'encoder_input': np.array(encoder_input, dtype=np.int64),
                'input_len': len_input,
                'encoder_target': np.array(encoder_target, dtype=np.float64),
                'decoder_input': np.array(decoder_input, dtype=np.int64),
                'decoder_target': np.array(encoder_input, dtype=np.int64),
            }
            self.dataset.append(sample)
            self.writer.add_scalar(
            f'Metric/performance', len(self.dataset))

    def add(self, seqs):
        # add seqs
        past_idx = len(self.seqs)
        for seq in seqs:
            matrix, ops = seminas_utils.convert_seq_to_arch(seq, self.engine.search_space)
            arch = self.engine.get_modelspec(matrix=matrix, ops=ops)
            perf=self.engine.query(arch, option='valid')

            self.seqs.append(seq)
            self.seq_len.append(len(seq))
            self.targets.append(perf)

        changed_max_len = max(len(seq) for seq in self.seqs)
        if self.max_len < changed_max_len :
            self.max_len = changed_max_len
            self.dataset = [] # Because max_len is changed, it needs to rewrite self.dataset.
            past_idx = 0

        for index in range(past_idx, len(self.seqs)):
            encoder_input = self.seqs[index] + [0 for _ in range(self.max_len - len(self.seqs[index]))] # fix length as max_len
            len_input = self.seq_len[index]
            encoder_target = [self.targets[index]]
            decoder_input = [0] + encoder_input[:-1]
            sample = {
                'encoder_input': np.array(encoder_input, dtype=np.int64),
                'input_len': len_input,
                'encoder_target': np.array(encoder_target, dtype=np.float64),
                'decoder_input': np.array(decoder_input, dtype=np.int64),
                'decoder_target': np.array(encoder_input, dtype=np.int64),
            }
            self.dataset.append(sample)
            self.writer.add_scalar(
            f'Metric/performance', len(self.dataset))

    def is_valid(self, seq):
        matrix, ops = seminas_utils.convert_seq_to_arch(seq, self.engine.search_space)
        arch = self.engine.get_modelspec(matrix=matrix, ops=ops)
        return self.engine.is_valid(arch) and seq not in self.seqs

    def shuffled(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def sorted(self, count):
        # use for picking current topk architectures
        indices = sorted(
            range(len(self.dataset)),
            key=lambda i: self.dataset[i]['encoder_target'],
            reverse=True
        )
        return torch.utils.data.DataLoader(
            dataset=[self.dataset[i] for i in indices[:count]],
            shuffle=True,
            batch_size=self.batch_size,
        )
