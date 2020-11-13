from collections import defaultdict, OrderedDict
import logging
import random
import unittest

from hydra.experimental import compose, initialize
from nasbench import api as api101
import networkx as nx
import pandas as pd

from src.datasets import PretrainNASBench
from src.graph_modifier import GraphModifier, NoValidModelExcpetion

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GraphModifierStatsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with initialize(config_path="../configs"):
            cfg = compose(config_name="test")
            cls.edit_distance = cfg.edit_distance
            cls.graph_modify_ratio = cfg.graph_modify_ratio
            cls.TESTCASE_COUNT = cfg.TESTCASE_COUNT

            stats_cfg = cfg.stats
            cls.SAMPLES_PER_CLASS = cls.TESTCASE_COUNT

            cls.min_accuracy = stats_cfg.min_accuracy
            cls.min_diff = stats_cfg.min_diff

            dataset = PretrainNASBench(
                engine=api101.NASBench(cfg.dataset_path),
                model_spec=api101.ModelSpec,
                samples_per_class=cls.SAMPLES_PER_CLASS,
                max_seq_len=cfg.max_seq_len,
                graph_modify_ratio=cls.graph_modify_ratio
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

    def check_accuracy(self, function_to_test, edit_distance_list, title=None):
        stats_df = pd.DataFrame(
            [[0] * 5] * len(edit_distance_list), columns=[0, 1, 2, 3, -1])
        stats_df["target_edit_distance"] = edit_distance_list
        stats_df = stats_df.set_index("target_edit_distance")

        accuracies = []
        for edit_distance in edit_distance_list:
            counts = defaultdict(int)
            for matrix, ops in self.testcases:
                try:
                    new_matrix, new_ops = function_to_test(
                        matrix, ops, edit_distance)
                    output_edit_distance = self.get_edit_distance(
                        matrix, ops, new_matrix, new_ops)
                    counts[output_edit_distance] += 1
                except NoValidModelExcpetion:
                    counts[-1] += 1

            for output_edit_distance, count in counts.items():
                stats_df.loc[edit_distance, output_edit_distance] = count

        for edit_distance in edit_distance_list:
            accuracies.append(
                stats_df.loc[edit_distance, edit_distance] / len(self.testcases))

        stats_df["accuracy"] = accuracies

        if not title:
            title = function_to_test.__name__
        logger.info(f"[STATS] {title}")
        logger.info(f"\n{stats_df.to_string()}")
        logger.info("\n\n\n")

        for ed in edit_distance_list:
            self.assertGreaterEqual(
                stats_df.loc[ed, "accuracy"], self.min_accuracy)

    def check_accuracy_modifier(self):
        stats_df = pd.DataFrame([[0] * 5] * 3, columns=[0, 1, 2, 3, -1],
                                index=["count", "target_ratio", "output_ratio"])

        counts = defaultdict(int)
        matrix, ops = self.testcases[0]
        for new_matrix, new_ops in self.graph_modifier.generate_modified_models(matrix, ops):
            try:
                output_edit_distance = self.get_edit_distance(
                    matrix, ops, new_matrix, new_ops)
                counts[output_edit_distance] += 1
            except NoValidModelExcpetion:
                counts[-1] += 1

        TARGET_RATIO = {
            i: v / sum(self.graph_modify_ratio.values()) for i, v in enumerate(self.graph_modify_ratio.values(), start=1)
        }
        for k, v in counts.items():
            k = int(k)
            stats_df.loc["count", k] = int(v)
            stats_df.loc["target_ratio", k] = TARGET_RATIO.get(k, 0)
            stats_df.loc["output_ratio", k] = v / \
                self.graph_modifier.samples_per_class

        logger.info("[STATS] graph_modifier.generate_modified_models")
        logger.info(f"\n{stats_df.to_string()}")
        logger.info("\n\n\n")

        for ed in TARGET_RATIO.keys():
            self.assertLessEqual(abs(
                stats_df.loc["output_ratio", ed] - stats_df.loc["target_ratio", ed]), self.min_diff)

    def test_accuracy_edit_node(self):
        self.check_accuracy(self.graph_modifier._generate_edit_node_model, list(
            range(1, self.edit_distance + 1)))

    # test edit-distance=1,2
    def test_accuracy_edit_edge(self):
        self.check_accuracy(self.graph_modifier._generate_edit_edge_model, list(
            range(1, self.edit_distance)))

    def test_accuracy_edit_distance_one_model(self):
        self.check_accuracy(lambda matrix, ops, _: self.graph_modifier.generate_edit_distance_one_model(
            matrix, ops), [1], title="edit_distance_one")

    def test_accuracy_edit_distance_two_model(self):
        self.check_accuracy(lambda matrix, ops, _: self.graph_modifier.generate_edit_distance_two_model(
            matrix, ops), [2], title="edit_distance_two")

    def test_accuracy_edit_distance_three_model(self):
        self.check_accuracy(lambda matrix, ops, _: self.graph_modifier.generate_edit_distance_three_model(
            matrix, ops), [3], title="edit_distance_three")

    def test_accuracy_modified_models(self):
        self.check_accuracy_modifier()
