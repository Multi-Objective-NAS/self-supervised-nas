import copy
import random
import itertools

import numpy as np

import nasbench

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = set([CONV1X1, CONV3X3, MAXPOOL3X3])


class GraphModifier():
    def __init__(self, engine, samples_per_class, edit_distance_one, edit_distance_two, edit_distance_three):
        self.engine = engine
        self.samples_per_class = samples_per_class
        self.edit_distance_one = edit_distance_one
        self.edit_distance_two = edit_distance_two
        self.edit_distance_three = edit_distance_three

    def is_valid(self, matrix, ops):
        # is it okay to hard code importing nasbench?
        return self.engine.is_valid(nasbench.api.ModelSpec(
            matrix=matrix,
            ops=ops,
        ))

    def _random_matrix_idx_generator(self, len_matrix, repeat):
        def one_random_matrix_idx_generator(len_matrix):
            indexes = []
            for i in range(len_matrix):
                for j in range(i+1, len_matrix):
                    indexes.append((i, j))

            random.shuffle(indexes)
            yield from indexes

        checked = set()
        for idxs in itertools.product(one_random_matrix_idx_generator(len_matrix), repeat=repeat):
            idxs = tuple(sorted(idxs))
            # sample without replacement
            if len(set(idxs)) != len(idxs):
                continue
            if tuple(sorted(idxs)) in checked:
                continue
            checked.add(idxs)
            yield idxs

    def _random_op_generator(self, ops, repeat):
        def one_random_op_generator(ops):
            op_pairs = []

            # exclude INPUT, OUTPUT node
            for idx, op in enumerate(ops[1:-1]):
                new_ops = list(OPS - set(op))
                op_pairs.append((idx+1, new_ops[0]))
                op_pairs.append((idx+1, new_ops[1]))

            random.shuffle(op_pairs)
            yield from op_pairs

        checked = set()
        for op_pairs in itertools.product(one_random_op_generator(ops), repeat=repeat):
            op_pairs = tuple(sorted(op_pairs))
            idxs = [op_pair[0] for op_pair in op_pairs]
            # sample without replacement
            if len(set(idxs)) != len(idxs):
                continue
            if tuple(sorted(op_pairs)) in checked:
                continue
            checked.add(op_pairs)
            yield op_pairs

    def _generate_edit_edge_models(self, original_matrix, ops, edit_distance, count):
        generated_count = 0

        len_matrix = len(original_matrix)
        for matrix_idxs in self._random_matrix_idx_generator(len_matrix, repeat=edit_distance):
            if generated_count >= count:
                raise StopIteration

            matrix = copy.deepcopy(original_matrix)
            for idx in matrix_idxs:
                matrix[idx] = 1 - matrix[idx]

            fake_ops = [INPUT] + [CONV3X3] * (len_matrix-2) + [OUTPUT]
            if self.is_valid(matrix, fake_ops):
                generated_count += 1
                yield (matrix, ops)

    def _generate_edit_node_models(self, matrix, original_ops, edit_distance, count):
        choices = self._random_op_generator(original_ops, repeat=edit_distance)

        for generated_count, op_pairs in enumerate(choices):
            if generated_count >= count:
                raise StopIteration
            ops = copy.deepcopy(original_ops)
            for op_pair in op_pairs:
                ops[op_pair[0]] = op_pair[1]
            yield (matrix, ops)

    def get_delete_one_node_models(self, original_matrix, original_ops, count):

        def to_hashable(obj):
            obj = np.array(obj)
            return tuple(obj.reshape(1, -1)[0])

        def get_delete_node_matrix(original_matrix, idx):
            matrix = copy.deepcopy(original_matrix)
            remaining_idxs = [i for i in range(len(matrix)) if i != idx]
            matrix = matrix[remaining_idxs][:, remaining_idxs]
            return matrix

        def delete_one_node_model_generator(original_matrix, original_ops):
            generated = list()
            for idx in range(1, len(original_ops)):
                # find node that has two edges
                if (np.sum(original_matrix[idx]) + np.sum(original_matrix[:, idx])) == 2:
                    matrix = get_delete_node_matrix(original_matrix, idx)
                    ops = [op for op_idx, op in enumerate(
                        original_ops) if op_idx != idx]
                    if not self.is_valid(matrix, ops):
                        continue
                    if (to_hashable(matrix), to_hashable(ops)) in generated:
                        continue
                    generated.append((matrix, ops))

            random.shuffle(generated)
            yield from generated

        for generated_count, (matrix, ops) in enumerate(delete_one_node_model_generator(original_matrix, original_ops)):
            if generated_count >= count:
                raise StopIteration
            yield (matrix, ops)

    def get_sample_model_count(self, len_nodes, node_replace_count, max_sample_count):
        if node_replace_count == 0:
            return 0

        max_model_count = 1
        for i in range(1, node_replace_count):
            max_model_count *= (len_nodes - 1 - i) / i

        model_count = random.randint(0, min(max_model_count, max_sample_count))
        return model_count

    def generate_edit_distance_one_models(self, matrix, ops, count):
        node_edit_model_count = random.randint(0, min(len(ops)-2, count))
        edge_edit_model_count = count - node_edit_model_count

        yield from self._generate_edit_node_models(matrix, ops, edit_distance=1, count=node_edit_model_count)
        yield from self._generate_edit_edge_models(matrix, ops, edit_distance=1, count=edge_edit_model_count)

    def generate_edit_distance_two_models(self, matrix, ops, count):
        len_nodes = len(ops)
        two_node_max = int((len_nodes - 2) * (len_nodes - 3) / 2)
        one_node_max = len_nodes - 2
        node_node_edit_model_count = random.randint(
            0, min(two_node_max, count))
        node_edge_edit_model_count = random.randint(
            0, min(one_node_max, count - node_node_edit_model_count))
        edge_edge_edit_model_count = count - \
            node_edge_edit_model_count - node_node_edit_model_count

        # replace 2 node
        yield from self._generate_edit_node_models(matrix, ops, edit_distance=2, count=node_node_edit_model_count)

        # replace 1 node + edit 1 edge
        new_ops_models = self._generate_edit_node_models(
            matrix, ops, edit_distance=1, count=node_edge_edit_model_count)
        new_matrixes_models = self._generate_edit_edge_models(
            matrix, ops, edit_distance=1, count=node_edge_edit_model_count)
        for (new_matrix, _), (_, new_ops) in zip(new_matrixes_models, new_ops_models):
            yield (new_matrix, new_ops)

        # edit 2 edge
        yield from self._generate_edit_edge_models(matrix, ops, edit_distance=2, count=edge_edge_edit_model_count)

    def generate_edit_distance_three_models(self, matrix, ops, count):
        max_node_replace_count = 3

        # 1 node delete + 2 edges delete
        node_delete_sample_count = random.randint(0, min(len(ops)-2, count))
        node_delete_model_count = 0
        for new_model in self.get_delete_one_node_models(matrix, ops, count=node_delete_sample_count):
            node_delete_model_count += 1
            yield new_model

        # node replace + edge edit
        len_nodes = len(ops)
        max_sample_count = count - node_delete_model_count
        for node_replace_count in range(max_node_replace_count):
            node_replace_sample_count = self.get_sample_model_count(
                len_nodes, node_replace_count, max_sample_count)
            max_sample_count -= node_replace_sample_count

            new_ops_models = self._generate_edit_node_models(
                matrix, ops, edit_distance=node_replace_count, count=node_replace_sample_count)
            new_matrixes_models = self._generate_edit_edge_models(matrix, ops, edit_distance=(
                max_node_replace_count-node_replace_count), count=node_replace_sample_count)
            for (new_matrix, _), (_, new_ops) in zip(new_matrixes_models, new_ops_models):
                yield (new_matrix, new_ops)

        # sample_model_count = rest of sample count
        node_replace_count = 3
        node_replace_sample_count = max_sample_count
        new_nodes_models = self._generate_edit_node_models(
            matrix, ops, edit_distance=node_replace_count, count=node_replace_sample_count)
        for _, new_node in new_nodes_models:
            yield (matrix, new_node)

    def generate_edited_models(self, matrix, ops):
        assert (self.edit_distance_one + self.edit_distance_two +
                self.edit_distance_three == 1)

        edit_distance_one_count = int(
            self.samples_per_class * self.edit_distance_one)
        edit_distance_two_count = int(
            self.samples_per_class * self.edit_distance_two)
        edit_distance_three_count = self.samples_per_class - \
            edit_distance_one_count - edit_distance_two_count

        yield from(self.generate_edit_distance_one_models(matrix, ops, edit_distance_one_count))
        yield from(self.generate_edit_distance_two_models(matrix, ops, edit_distance_two_count))
        yield from(self.generate_edit_distance_three_models(matrix, ops, edit_distance_three_count))
