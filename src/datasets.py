import pathlib

from hydra import utils as hydra_utils
import numpy as np
import torch
import math

from nasbench import api as api101
from nasbench.lib import graph_util
from nas_201_api import api_201 as api201
from libs.SemiNAS.nas_bench import utils as seminas_utils
from .graph_modifier import GraphModifier


def get_engine_modelspec(name, path):
    if name == 'nasbench101':
        return api101.NASBench(path), api101.ModelSpec
    elif name == 'nasbench201':
        return api201.NASBench201API(path), api201.ModelSpec
    else:
        raise ValueError('Invalid name')


def get_dataset(name, path, mode, **kwargs):
    assert name == 'nasbench101' or name == 'nasbench201'
    assert mode in ('pretrain', 'train')

    if not pathlib.Path(path).is_absolute():
        path = hydra_utils.to_absolute_path(path)
    assert pathlib.Path(path).exists()

    engine, model_spec = get_engine_modelspec(name, path)
    if mode == 'pretrain':
        dataset_cls = PretrainNASBench
    elif mode == 'train':
        dataset_cls = TrainNASBench
    return dataset_cls(
        engine=engine,
        model_spec=model_spec,
        **kwargs
    )


class PretrainNASBench(torch.utils.data.IterableDataset):
    def __init__(self, engine, model_spec, max_seq_len, samples_per_class, graph_modify_ratio, seed):
        self.engine = engine
        self.model_spec = model_spec
        self.max_seq_len = max_seq_len
        self.samples_per_class = samples_per_class

        # list of possible operations ex.[ CONV 3x3, MAXPOOL 3x3, ... ]
        self.search_space = engine.search_space

        # Find dataset length
        self._dataset_length = len(engine.hash_iterator())

        self.graph_modifier = GraphModifier(
            validate=self.is_valid,
            operations=set(self.search_space),
            samples_per_class=samples_per_class - 1,
            **graph_modify_ratio)

        self.seed = seed

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            seq = self._encode(matrix, ops)
            yield (seq, [0] + seq[:-1]), index
            for pmatrix, pops in self.graph_modifier.generate_modified_models(matrix, ops):
                seq = self._encode(pmatrix, pops)
                yield (seq, [0] + seq[:-1]), index

    def __len__(self):
        return self._dataset_length * self.samples_per_class

    def _random_graph_generator(self):
        keys = list(self.engine.hash_iterator())
        np.random.RandomState(seed=self.seed).shuffle(keys)
        for index, key in enumerate(keys):
            arch = self.engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            yield (index, matrix, ops)
        self.seed += 1

    def _encode(self, matrix, ops):
        seq = seminas_utils.convert_arch_to_seq(matrix, ops, self.search_space)
        seq += [0] * (self.max_seq_len - len(seq))
        assert len(seq) == self.max_seq_len
        return seq

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
    def __init__(self, engine, model_spec, max_seq_len, batch_size, writer, seed):
        self.engine = engine
        self.model_spec = model_spec
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.writer = writer
        self.seed = seed

        self.dataset = []
        self.seqs = []

    def _append(self, seq, sampled_perf, true_perf):
        self.seqs.append(seq)
        self.dataset.append({
            'encoder_input': torch.LongTensor(seq),
            'decoder_input': torch.LongTensor([0] + seq[:-1]),
            'encoder_target': torch.FloatTensor([max(sampled_perf-0.8, 0.0) * 5.0]),
            'decoder_target': torch.LongTensor(seq),
        })
        self.writer.add_scalar(
            f'Metric/performance', sampled_perf, len(self.dataset))
        self.writer.add_scalar(
            f'Metric/true_performance', true_perf, len(self.dataset))

    def _encode(self, matrix, ops):
        seq = seminas_utils.convert_arch_to_seq(matrix, ops, self.engine.search_space)
        seq += [0] * (self.max_seq_len - len(seq))
        assert len(seq) == self.max_seq_len
        return seq

    def _decode(self, seq):
        assert 6 in seq
        seq = seq[:seq.index(6) + 1]

        n = int(math.floor(math.sqrt((len(seq) + 1) * 2)))
        assert n >= 2
        assert len(seq) == (n + 2) * (n - 1) / 2

        for i in range(n - 1):
            offset = (i + 3) * i // 2
            for j in range(i + 1):
                assert 1 <= seq[offset + j] <= 2
            idx = seq[offset + i + 1] - 3
            assert 0 <= idx <= len(self.engine.search_space)

        matrix, ops = seminas_utils.convert_seq_to_arch(seq, self.engine.search_space)
        arch = self.model_spec(matrix=matrix, ops=ops)
        return arch

    def _query(self, arch):
        return self.engine.query(arch, 'valid'), self.engine.query(arch, 'test')

    def prepare(self, count):
        keys = list(self.engine.hash_iterator())
        np.random.RandomState(seed=self.seed).shuffle(keys)
        for key in keys:
            arch = self.engine.get_modelspec_by_hash(key)
            sampled_perf, true_perf = self._query(arch)
            self._append(
                seq=self._encode(arch.matrix, arch.ops),
                sampled_perf=sampled_perf,
                true_perf=true_perf,
            )
            if len(self.dataset) >= count:
                break
        self.seed += 1

    def add(self, seqs):
        for seq in seqs:
            arch = self._decode(seq)
            sampled_perf, true_perf = self._query(arch)
            self._append(
                seq=self._encode(arch.matrix, arch.ops),
                sampled_perf=sampled_perf,
                true_perf=true_perf,
            )

    def is_valid(self, seq):
        try:
            arch = self._decode(seq)
        except AssertionError:
            return False
        if self.engine.is_valid(arch):
            # arch.matrix is None if arch is not valid.
            return self._encode(arch.matrix, arch.ops) not in self.seqs
        else:
            return False

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
