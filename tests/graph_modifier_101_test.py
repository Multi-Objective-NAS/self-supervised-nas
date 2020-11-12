import random
import unittest

from hydra.experimental import compose, initialize
from nasbench import api as api101
import networkx as nx
import numpy as np


from src.datasets import PretrainNASBench
from src.graph_modifier import GraphModifier, NoValidModelExcpetion


class GraphModifierTest(unittest.TestCase):

    # graph modify not possible
    model_size2 = (
        np.array([[0, 1],
                  [0, 0]]),
        ["input", "output"]
    )

    model_size3 = (
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]]),
        ["input", "conv3x3-bn-relu", "output"]
    )

    model_size4 = (
        np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "output"]
    )

    @classmethod
    def setUpClass(cls):
        with initialize(config_path="../configs"):
            cfg = compose(config_name="test")
            cls.TESTCASE_COUNT = cfg.TESTCASE_COUNT
            cls.EDIT_DISTANCE = cfg.EDIT_DISTANCE

            dataset = PretrainNASBench(
                engine=api101.NASBench(cfg.NASBENCH_101_DATASET),
                model_spec=api101.ModelSpec,
                samples_per_class=cls.TESTCASE_COUNT,
                max_seq_len=cfg.MAX_SEQ_LEN,
                graph_modify_ratio=cfg.GRAPH_MODIFY_RATIO
            )

        cls.graph_modifier = dataset.graph_modifier

        cls.testcases = []
        for _, key in enumerate(random.sample(dataset.engine.hash_iterator(), cls.TESTCASE_COUNT)):
            arch = dataset.engine.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            cls.testcases.append((matrix, ops))

    def get_edit_distance(self, matrix1, ops1, matrix2, ops2):
        def to_nx_graph(matrix, ops):
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            for idx, op in enumerate(ops):
                G.add_node(idx, operation=op)
            return G

        def node_match(node1, node2):
            return node1["operation"] == node2["operation"]

        def edge_match(edge1, edge2):
            return edge1 == edge2

        G1 = to_nx_graph(matrix1, ops1)
        G2 = to_nx_graph(matrix2, ops2)

        return nx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match)

    def test_edit_node_with_random_graphs(self):
        for edit_distance in range(1, 4):
            for matrix, ops in self.testcases:
                try:
                    new_matrix, new_ops = self.graph_modifier._generate_edit_node_model(
                        matrix, ops, edit_distance)
                    self.assertTrue(self.get_edit_distance(
                        matrix, ops, new_matrix, new_ops) <= edit_distance)
                except NoValidModelExcpetion:
                    pass

    def test_edit_edge_with_random_graphs(self):
        for edit_distance in range(1, 4):
            for matrix, ops in self.testcases:
                try:
                    new_matrix, new_ops = self.graph_modifier._generate_edit_edge_model(
                        matrix, ops, edit_distance)
                    self.assertTrue(self.get_edit_distance(
                        matrix, ops, new_matrix, new_ops) <= edit_distance)
                except NoValidModelExcpetion:
                    pass

    def test_edit_edge_raise_exception(self):
        for edit_distance in range(1, 4):
            matrix, ops = self.model_size2
            with self.assertRaises(NoValidModelExcpetion):
                self.graph_modifier._generate_edit_edge_model(
                    matrix, ops, edit_distance)

        for edit_distance in range(2, 4):
            matrix, ops = self.model_size3
            with self.assertRaises(NoValidModelExcpetion):
                self.graph_modifier._generate_edit_edge_model(
                    matrix, ops, edit_distance)

    def test_edit_node_raise_exception(self):
        edit_distance = 1
        matrix, ops = self.model_size2
        with self.assertRaises(NoValidModelExcpetion):
            self.graph_modifier._generate_edit_node_model(
                matrix, ops, edit_distance)

        edit_distance = 2
        matrix, ops = self.model_size3
        with self.assertRaises(NoValidModelExcpetion):
            self.graph_modifier._generate_edit_node_model(
                matrix, ops, edit_distance)

        edit_distance = 3
        matrix, ops = self.model_size4
        with self.assertRaises(NoValidModelExcpetion):
            self.graph_modifier._generate_edit_node_model(
                matrix, ops, edit_distance)

    def test_genereate_modified_models_with_random_graphs(self):
        for idx, (matrix, ops) in enumerate(self.testcases):
            print(idx)
            try:
                for iidx, (new_matrix, new_ops) in enumerate(self.graph_modifier.generate_modified_models(matrix, ops)):
                    self.assertTrue(self.get_edit_distance(
                        matrix, ops, new_matrix, new_ops) <= self.EDIT_DISTANCE)
            except NoValidModelExcpetion:
                pass
