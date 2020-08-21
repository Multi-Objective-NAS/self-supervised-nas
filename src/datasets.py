import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

import nasbench
from libs.SemiNAS.nas_bench import utils as seminas_utils


def get_dataset(name, **kwargs):
    assert name == 'nasbench101'
    return NASBench101(**kwargs)


class NASBench101(torch.utils.data.IterableDataset):
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
                nasbench.api.ModelSpec(matrix=matrix, ops=ops)
            ):
                count += 1
                yield (pmatrix, pops)

        raise StopIteration

    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops)
