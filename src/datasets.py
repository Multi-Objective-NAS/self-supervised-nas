import pathlib

from hydra import utils as hydra_utils
import numpy as np
import torch

from nasbench import api as api101
from nasbench.lib import graph_util
from nas_201_api import api_201 as api201
from libs.SemiNAS.nas_bench import utils as seminas_utils
from .graph_modifier import GraphModifier


def get_dataset(name, path, **kwargs):
    assert name == 'nasbench101' or name == 'nasbench201'

    if not pathlib.Path(path).is_absolute():
        path = hydra_utils.to_absolute_path(path)
        assert pathlib.Path(path).exists()

    if name == 'nasbench101':
        return NASBench(engine=api101.NASBench(path), model_spec=api101.ModelSpec, **kwargs)
    elif name == 'nasbench201':
        return NASBench(engine=api201.NASBench201API(path), model_spec=api101.ModelSpec, **kwargs)


class NASBench(torch.utils.data.IterableDataset):
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
            if matrix.shape[0] == 7:
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
            if matrix.shape[0] == 7:
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
            modelspec = self.engine.get_modelspec(matrix=matrix, ops=ops)
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
