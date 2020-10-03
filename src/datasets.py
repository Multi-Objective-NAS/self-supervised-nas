import pathlib

from hydra import utils as hydra_utils
import numpy as np
import torch

from nasbench import api as api101
from nasbench.lib import graph_util
from nas_201_api import api_201 as api201
from libs.SemiNAS.nas_bench import utils as seminas_utils
from .graph_modifier import GraphModifier


def get_engine_modelspec_maxseqlen(name, path):
    if name == 'nasbench101':
        return api101.NASBench(path), api101.ModelSpec, 27
    elif name == 'nasbench201':
        return api201.NASBench201API(path), api201.ModelSpec, 35
    else:
        raise ValueError('Invalid name')


def get_dataset(name, path, mode, **kwargs):
    assert name == 'nasbench101' or name == 'nasbench201'
    assert mode in ('pretrain', 'train')

    if not pathlib.Path(path).is_absolute():
        path = hydra_utils.to_absolute_path(path)
    assert pathlib.Path(path).exists()

    engine, model_spec, max_seq_len = get_engine_modelspec_maxseqlen(name, path)
    if mode == 'pretrain':
        dataset_cls = PretrainNASBench
    elif mode == 'train':
        dataset_cls = TrainNASBench
    return dataset_cls(
        engine=engine,
        model_spec=model_spec,
        max_seq_len=max_seq_len,
        **kwargs
    )


class PretrainNASBench(torch.utils.data.IterableDataset):
    def __init__(self, engine, model_spec, max_seq_len, samples_per_class, graph_modify_ratio):
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
        for index, key in enumerate(self.engine.hash_iterator()):
            arch = self.engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            yield (index, matrix, ops)

    def _encode(self, matrix, ops):
        seq = seminas_utils.convert_arch_to_seq(matrix, ops, self.search_space)
        return seq + [0] * (self.max_seq_len - len(seq))

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
    def __init__(self, engine, model_spec, max_seq_len, batch_size, writer):
        self.engine = engine
        self.model_spec = model_spec
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.writer = writer

        self.dataset = []
        self.seqs = []

    def _query(self, matrix, ops):
        arch = self.model_spec(matrix=matrix, ops=ops)
        return self.engine.query(arch, 'valid'), self.engine.query(arch, 'test')

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
        seq = seminas_utils.convert_arch_to_seq(matrix, ops, self.search_space)
        return seq + [0] * (self.max_seq_len - len(seq))

    def prepare(self, count):
        for key in self.engine.hash_iterator():
            fixed_stat, _ = self.engine.get_metrics_from_hash(key)
            matrix, ops = np.array(fixed_stat['module_adjacency']), fixed_stat['module_operations']
            sampled_perf, true_perf = self._query(matrix, ops)
            self._append(
                seq=self._encode(matrix, ops),
                sampled_perf=sampled_perf,
                true_perf=true_perf,
            )
            if len(self.dataset) >= count:
                break

    def add(self, seqs):
        for seq in seqs:
            matrix, ops = seminas_utils.convert_seq_to_arch(seq, self.engine.search_space)
            sampled_perf, true_perf = self._query(matrix, ops)
            self._append(
                seq=self._encode(matrix, ops),
                sampled_perf=sampled_perf,
                true_perf=true_perf,
            )

    def is_valid(self, seq):
        matrix, ops = seminas_utils.convert_seq_to_arch(seq, self.engine.search_space)
        arch = self.model_spec(matrix=matrix, ops=ops)
        return self.engine.is_valid(arch) and seq not in self.seqs

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
