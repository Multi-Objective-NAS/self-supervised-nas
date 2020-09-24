from collections import defaultdict, OrderedDict
import logging
import random
import unittest

from nasbench import api as api101
import networkx as nx
import pandas as pd

from src.datasets import NASBench
from src.graph_modifier import GraphModifier, NoValidModelExcpetion

NASBENCH_101_DATASET = "/home/dzzp/workspace/dataset/test3000.tfrecord"
TESTCASE_COUNT = 200
SAMPLES_PER_CLASS = 200
EDIT_DISTANCE = 3

MIN_CORRECT_RATIO = 0.9
MIN_DIFF = 0.05

GRAPH_MODIFY_RATIO = OrderedDict({
    "edit_distance_one": 1,
    "edit_distance_two": 1,
    "edit_distance_three": 1
})

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GraphModifierStatsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        dataset = NASBench(
            engine=api101.NASBench(NASBENCH_101_DATASET),
            model_spec=api101.ModelSpec,
            samples_per_class=SAMPLES_PER_CLASS,
            graph_modify_ratio=GRAPH_MODIFY_RATIO
        )

        cls.graph_modifier = dataset.graph_modifier

        cls.testcases = []
        for _, key in enumerate(random.sample(dataset.engine.hash_iterator(), TESTCASE_COUNT)):
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

    def stats(self, function_to_test, edit_distance_list):
        stats_df = pd.DataFrame([[0] * 5] * len(edit_distance_list), columns=[0, 1, 2, 3, -1])
        stats_df["target_edit_distance"] = edit_distance_list
        stats_df = stats_df.set_index("target_edit_distance")

        correct_ratios = []
        for edit_distance in edit_distance_list:
            counts = defaultdict(int)
            for matrix, ops in self.testcases:
                try:
                    new_matrix, new_ops = function_to_test(matrix, ops, edit_distance)
                    output_edit_distnace = self.get_edit_distance(matrix, ops, new_matrix, new_ops)
                    counts[output_edit_distnace] += 1
                except NoValidModelExcpetion:
                    counts[-1] += 1

            for output_edit_distance, count in counts.items():
                stats_df.loc[edit_distance, output_edit_distance] = count

        for edit_distance in edit_distance_list:
            correct_ratios.append(stats_df.loc[edit_distance, edit_distance] / TESTCASE_COUNT)

        stats_df["correct_ratio"] = correct_ratios

        logger.info(f"[STATS] {function_to_test.__name__}")
        logger.info(f"\n{stats_df.to_string()}")
        logger.info("\n\n\n")

        for ed in edit_distance_list:
            self.assertGreaterEqual(stats_df.loc[ed, "correct_ratio"], MIN_CORRECT_RATIO)

    def stats_moddifier(self):
        stats_df = pd.DataFrame([[0] * 5] * 3, columns=[0, 1, 2, 3, -1], index=["count", "target_ratio", "output_ratio"])

        counts = defaultdict(int)
        matrix, ops = self.testcases[0]
        for new_matrix, new_ops in self.graph_modifier.generate_modified_models(matrix, ops):
            try:
                output_edit_distnace = self.get_edit_distance(matrix, ops, new_matrix, new_ops)
                counts[output_edit_distnace] += 1
            except NoValidModelExcpetion:
                counts[-1] += 1

        ratio_sum = sum(GRAPH_MODIFY_RATIO.values())
        TARGET_RATIO = defaultdict(int)
        for k, v in enumerate(GRAPH_MODIFY_RATIO.values(), start=1):
            TARGET_RATIO[k] = v / ratio_sum

        for k, v in counts.items():
            stats_df.loc["count", k] = v
            stats_df.loc["target_ratio"] = TARGET_RATIO[k]
            stats_df.loc["output_ratio"] = v / self.graph_modifier.samples_per_class

        logger.info("[STATS] graph_modifier.generate_modified_models")
        logger.info(f"\n{stats_df.to_string()}")
        logger.info("\n\n\n")

        for ed in TARGET_RATIO.keys():
            self.assertLessEqual(abs(stats_df.loc["output_ratio", ed] - stats_df.loc["target_ratio", ed]), MIN_DIFF)

    def test_stats_edit_node(self):
        self.stats(self.graph_modifier._generate_edit_node_model, list(range(1, EDIT_DISTANCE + 1)))

    def test_stats_edit_edge(self):
        self.stats(self.graph_modifier._generate_edit_edge_model, list(range(1, EDIT_DISTANCE + 1)))

    def wrapped_generate_edit_distance_one(self, matrix, ops, _):
        return self.graph_modifier.generate_edit_distance_one_model(matrix, ops)

    def wrapped_generate_edit_distance_two(self, matrix, ops, _):
        return self.graph_modifier.generate_edit_distance_two_model(matrix, ops)

    def wrapped_generate_edit_distance_three(self, matrix, ops, _):
        return self.graph_modifier.generate_edit_distance_three_model(matrix, ops)

    def test_stats_edit_distance_one_model(self):
        self.stats(self.wrapped_generate_edit_distance_one, [1])

    def test_stats_edit_distance_two_model(self):
        self.stats(self.wrapped_generate_edit_distance_two, [2])

    def test_stats_edit_distance_three_model(self):
        self.stats(self.wrapped_generate_edit_distance_three, [3])

    def test_stats_modified_models(self):
        self.stats_moddifier()
