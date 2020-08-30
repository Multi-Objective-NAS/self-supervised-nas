import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

from libs.nasbench201 import nas_201_api as api201
from libs.nasbench.nasbench import api as api101
from libs.nasbench.nasbench.lib import graph_util
from libs.SemiNAS.nas_bench import utils as seminas_utils


def get_dataset(name, path, **kwargs):
    assert name == 'nasbench101' or name == 'nasbench201'

    if not pathlib.Path(path).is_absolute():
        path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

    if name == 'nasbench101':
        return NASBench(engine=api101.NASBench(path), **kwargs)
    elif name == 'nasbench201':
        return NASBench(engine=api201.NASBench201API(path), **kwargs)

class NASBench(torch.utils.data.IterableDataset):
    def __init__(self, engine, samples_per_class):
        
        self.engine = engine
            
        self.samples_per_class = samples_per_class
        self.search_space = engine.search_space # list of possible operations ex.[ CONV 3x3, MAXPOOL 3x3, ... ]

        length = 0
        for index, key in enumerate(engine.hash_iterator()):
                arch = engine.get_model_spec_by_hash(key)
                matrix, ops = arch.matrix, arch.ops
                if matrix.shape[0] == 7:
                    length += 1
        self._dataset_length = length

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            for pmatrix, pops in self._generate_isomorphic_graphs(matrix, ops):
                yield self._encode(pmatrix, pops), index

    def __len__(self):
        return self._dataset_length * self.samples_per_class

    def _random_graph_generator(self):
        for index, key in enumerate(self.engine.hash_iterator()):
            arch = self.engine.get_model_spec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
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

            pmatrix, pops = graph_util.permute_graph(matrix, ops, perm)
            modelspec = self.engine.get_modelspec(matrix=matrix, ops=ops)
            if self.engine.is_valid(modelspec):
                count += 1
                yield (pmatrix, pops)

        raise StopIteration

    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops, self.search_space)
