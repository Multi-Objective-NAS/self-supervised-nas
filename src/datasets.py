import itertools
import pathlib

import torch
import numpy as np
from hydra import utils as hydra_utils

import nasbench
from libs.SemiNAS.nas_bench import utils as seminas_utils
from .graph_modifier import GraphModifier

def get_dataset(name, **kwargs):
    assert name == 'nasbench101'
    return NASBench101(**kwargs)


class NASBench101(torch.utils.data.IterableDataset):
    def __init__(self, path, samples_per_class, graph_modifier):
        if not pathlib.Path(path).is_absolute():
            path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

        self.engine = nasbench.api.NASBench(path)
        self.samples_per_class = samples_per_class
        self.graph_modifier = GraphModifier(engine=self.engine, samples_per_class=samples_per_class, **graph_modifier)

    def __iter__(self):
        for index, matrix, ops in self._random_graph_generator():
            for pmatrix, pops in self._generate_positive_samples(matrix, ops):
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

    def _generate_positive_samples(self, matrix, ops):
        yield from self.graph_modifier.generate_edited_models(matrix, ops)
        
    def _encode(self, matrix, ops):
        return seminas_utils.convert_arch_to_seq(matrix, ops)
        
