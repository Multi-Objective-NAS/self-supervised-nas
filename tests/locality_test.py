import argparse
import csv
import datetime
from multiprocessing import Pool
import logging
import os
import random

import networkx as nx

from nasbench import api as api101


NASBENCH_101_DATASET = "/home/dzzp/workspace/dataset/nasbench_only108.tfrecord"

SEED_ARCH_COUNT = 300
MAX_EDIT_DISTANCE = 6
OUTPUT_PATH = "outputs_test_acc"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(processName)s %(message)s")


def get_edit_distance(matrix1, ops1, matrix2, ops2, upper_bound):
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

    return nx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match, upper_bound=upper_bound) or 0


def single_lookup(seed_hash, dataset, output_path):
    logging.info(f"START: seed_hash: {seed_hash}")
    seed_arch = dataset.get_modelspec_by_hash(seed_hash)
    seed_matrix, seed_ops = seed_arch.matrix, seed_arch.ops
    seed_test_acc = dataset.query(
        api101.ModelSpec(matrix=seed_matrix, ops=seed_ops), 'test', epochs=108)

    row_dict = dict()
    hash_iterator = list(dataset.hash_iterator())
    random.shuffle(hash_iterator)
    for _, arch_hash in enumerate(hash_iterator):
        arch = dataset.get_modelspec_by_hash(arch_hash)
        matrix, ops = arch.matrix, arch.ops
        arch_test_acc = dataset.query(
            api101.ModelSpec(matrix=matrix, ops=ops), 'test', epochs=108)

        true_distance = get_edit_distance(seed_matrix, seed_ops, matrix, ops, max_edit_distance)
        row_dict[true_distance] = ((true_distance, seed_hash, arch_hash, seed_test_acc - arch_test_acc, seed_test_acc, arch_test_acc))

        # row_dict.keys() : 0, 1, ..., max_edit_distance
        if len(row_dict) == max_edit_distance + 1:
            break

    with open(f'{output_path}/{seed_hash}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list(row_dict.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--seed_arch_count', type=int, required=False, default=SEED_ARCH_COUNT,
                        help=f'number_of_seed_architectures. default: {SEED_ARCH_COUNT}')
    parser.add_argument('-ed', '--max_edit_distance', type=int, default=MAX_EDIT_DISTANCE, required=False,
                        help=f'max edit distance to look up from dataset. default: {MAX_EDIT_DISTANCE}')
    parser.add_argument('-i', '--dataset_path', required=False, default=NASBENCH_101_DATASET, help='path to dataset')
    parser.add_argument('-o', '--output_path', required=False, default=OUTPUT_PATH,
                        help=f'path to output directory, default "{OUTPUT_PATH}"')
    args = parser.parse_args()

    seed_arch_count = args.seed_arch_count
    max_edit_distance = args.max_edit_distance
    output_path = args.output_path

    dataset = api101.NASBench(args.dataset_path)

    output_path = f"{output_path}/csv-{datetime.datetime.now().strftime('%m-%d-%H-%M')}"
    os.makedirs(output_path, exist_ok=True)

    seed_hashes = random.sample(dataset.hash_iterator(), seed_arch_count)
    with Pool(8) as p:
        p.starmap(single_lookup, [(seed_hash, dataset, output_path) for seed_hash in seed_hashes])
