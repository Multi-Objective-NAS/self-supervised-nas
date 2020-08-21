import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

#import nasbench
#import NAS-Bench-201
from NAS-Bench-201.nas_201_api import NASBench201API as api201
from nasbench import api as api101

from libs.SemiNAS.nas_bench import utils as seminas_utils


def get_dataset(name, **kwargs):
    # nasbench.api.NASBench
    
    if name == 'nasbench101' :
        return api101(**kwargs)
    elif name == 'nasbench201' :
        return api201(**kwargs)
    else:
        return None


class NASBench(torch.utils.data.IterableDataset):
    def __init__(self, name, path, samples_per_class):
        if not pathlib.Path(path).is_absolute():
            path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

        self.engine = get_dataset(name, path)
        self.samples_per_class = samples_per_class
        self.name = name
        self.len = -1

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            for pmatrix, pops in self._generate_isomorphic_graphs(matrix, ops):
                yield self._encode(pmatrix, pops), index

    def __len__(self):
        if self.len == 0:
            for index, key in enumerate(self.engine.hash_iterator()):
                arch = self.engine.get_model_spec_by_hash(hash_val)
                matrix, ops = arch.matrix, arch.ops
                if matrix.shape[0] == 7:
                    self.len += 1
        return self.len * self.samples_per_class

    def _random_graph_generator(self):
        for index, key in enumerate(self.engine.hash_iterator()):
            arch = self.engine.get_model_spec_by_hash(hash_val)
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

            pmatrix, pops = nasbench.lib.graph_util.permute_graph(matrix, ops, perm)
            if self.engine.get_modelspec(matrix=matrix, ops=ops):
                count += 1
                yield (pmatrix, pops)

        raise StopIteration

    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops)
