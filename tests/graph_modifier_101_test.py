import random
import unittest

import numpy as np
from nasbench import api as api101

from src.datasets import NASBench
from src.graph_modifier import GraphModifier, NoValidModelExcpetion

NASBENCH_101_DATASET = "../datasets/nasbench/nasbench_only108.tfrecord"
TESTCASE_COUNT = 20


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

    @classmethod
    def setUpClass(cls):
        cls.graph_modifier = GraphModifier(
            validate=NASBench.is_valid,
            operations=set(["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]),
            samples_per_class=8,
            edit_distance_one=1,
            edit_distance_two=1,
            edit_distance_three=1
        )

        dataset = api101.NASBench(NASBENCH_101_DATASET)
        cls.testcases = []
        for _, key in enumerate(random.sample(dataset.hash_iterator(), TESTCASE_COUNT)):
            arch = dataset.get_modelspec_by_hash(key)
            matrix, ops = arch.matrix, arch.ops
            cls.testcases.append((matrix, ops))

    def is_eqaul(self, original, variant):
        return (np.array(original) == np.array(variant)).all()

    def get_edit_distance(self, original, variant):
        return np.sum(np.array(original) != np.array(variant))

    def test_edit_node_with_random_graphs(self):
        for edit_distance in range(1, 4):
            for matrix, ops in enumerate(self.testcases):
                try:
                    new_matrix, new_ops = self.graph_modifier._generate_edit_node_model(matrix, ops, edit_distance)
                    self.assertTrue(self.is_eqaul(matrix, new_matrix))
                    self.assertFalse(self.is_eqaul(ops, new_ops))
                    self.assertTrue(self.get_edit_distance(ops, new_ops) == edit_distance)
                except NoValidModelExcpetion:
                    pass

    def test_edit_edge_with_random_graphs(self):
        for edit_distance in range(1, 4):
            for matrix, ops in enumerate(self.testcases):
                try:
                    new_matrix, new_ops = self.graph_modifier._generate_edit_edge_model(matrix, ops, edit_distance)
                    self.assertFalse(self.is_eqaul(matrix, new_matrix))
                    self.assertTrue(self.is_eqaul(ops, new_ops))
                    self.assertTrue(self.get_edit_distance(matrix, new_matrix) == edit_distance)
                except NoValidModelExcpetion:
                    pass

    def test_edit_node_raise_exception(self):
        for edit_distance in range(1, 4):
            matrix, ops = self.model_size2
            with self.assertRaises(NoValidModelExcpetion):
                self.graph_modifier._generate_edit_node_model(matrix, ops, edit_distance)

        for edit_distance in range(2, 4):
            matrix, ops = self.model_size3
            with self.assertRaises(NoValidModelExcpetion):
                self.graph_modifier._generate_edit_node_model(matrix, ops, edit_distance)

    def test_edit_edge_raise_exception(self):
        edit_distance = 1
        matrix, ops = self.model_size2
        with self.assertRaises(NoValidModelExcpetion):
            self.graph_modifier._generate_edit_edge_model(matrix, ops, edit_distance)

        for edit_distance in range(2, 4):
            matrix, ops = self.model_size3
            with self.assertRaises(NoValidModelExcpetion):
                self.graph_modifier._generate_edit_edge_model(matrix, ops, edit_distance)

        edit_distance = 3
        matrix, ops = self.model_size4
        with self.assertRaises(NoValidModelExcpetion):
            self.graph_modifier._generate_edit_edge_model(matrix, ops, edit_distance)
