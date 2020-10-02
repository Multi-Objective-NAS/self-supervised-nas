import argparse
import random
import os

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from nasbench import api as api101


NASBENCH_101_DATASET = "/home/dzzp/workspace/dataset/nasbench_only108.tfrecord"

RANDOM_SEED = 100

SEED_ARCH_COUNT = 1000
MAX_EDIT_DISTANCE = 4
OUTPUT_PATH = "outputs_test_acc"
do_not_save_plot = False


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

    return nx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match, upper_bound=upper_bound) or -1


# TODO: change plot type to cdf?
def save_plot(df, output_path):
    max_edit_distance = int(df["edit_distance"].max())
    fig, axes = plt.subplots(1, max_edit_distance, sharey=True, figsize=(4, 6))
    fig.suptitle('Distribution of test_accuracy difference by edit-distance')
    axes[0].set_ylabel("test accuracy difference")
    plt.gca().invert_yaxis()

    for ed, ax in zip(range(1, max_edit_distance + 1), axes):
        diffs = df.loc[df["edit_distance"] == ed, "test_difference"]
        ax.hist(diffs, orientation="horizontal")
        ax.set_xlabel(f'edit-distance={ed}')

    plt.savefig(f"{output_path}/figure_distribution.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--seed_arch_count', type=int, required=False, default=SEED_ARCH_COUNT,
                        help=f'number_of_seed_architectures. default: {SEED_ARCH_COUNT}')
    parser.add_argument('-ed', '--max_edit_distance', type=int, default=MAX_EDIT_DISTANCE,
                        help=f'max edit distance to look up from dataset. default: {MAX_EDIT_DISTANCE}')
    parser.add_argument('-i', '--dataset_path', required=True, help='path to dataset')
    parser.add_argument('-o', '--output_path', required=False, default=OUTPUT_PATH,
                        help=f'path to output directory, default "{OUTPUT_PATH}"')
    parser.add_argument('-ns', '--no-save-plot', dest='do_not_save_plot', action='store_true',required=False, 
                        help='do not save plot as images. default: save plot on given "output_path"')
    args = parser.parse_args()

    seed_arch_count = args.seed_arch_count
    max_edit_distance = args.max_edit_distance
    output_path = args.output_path
    do_not_save_plot = args.do_not_save_plot

    random.seed(RANDOM_SEED)
    dataset = api101.NASBench(args.dataset_path)

    rows = []
    for idx, seed_hash in enumerate(tqdm.tqdm(random.sample(dataset.hash_iterator(), seed_arch_count))):
        seed_arch = dataset.get_modelspec_by_hash(seed_hash)
        seed_matrix, seed_ops = seed_arch.matrix, seed_arch.ops
        seed_test_acc = dataset.query(
            api101.ModelSpec(matrix=seed_matrix, ops=seed_ops), 'test', epochs=108)

        row_dict = dict()
        for _, arch_hash in enumerate(dataset.hash_iterator()):
            arch = dataset.get_modelspec_by_hash(arch_hash)
            matrix, ops = arch.matrix, arch.ops
            arch_test_acc = dataset.query(
                api101.ModelSpec(matrix=matrix, ops=ops), 'test', epochs=108)

            with Pool(12) as p:
                rows = p.map(, zip(graphs_1, graphs_2))

            true_distance = get_edit_distance(seed_matrix, seed_ops, matrix, ops, max_edit_distance)
            row_dict[true_distance] = ((true_distance, seed_hash, arch_hash, abs(seed_test_acc - arch_test_acc), seed_test_acc, arch_test_acc))
            # row_dict.keys() : 0, 1, ..., max_edit_distance
            if len(row_dict) == max_edit_distance + 1:
                rows.extend(list(row_dict.values()))
                break

    columns = ['edit_distance', 'seed_hash', 'arch_hash', 'test_difference', 'seed_test_accuracy', 'arch_test_accuracy']
    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f'{output_path}/distribution.csv', index=False)
    if not do_not_save_plot:
        save_plot(df, output_path)