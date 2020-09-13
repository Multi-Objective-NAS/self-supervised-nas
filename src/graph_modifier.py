import random

import numpy as np


class NoValidModelExcpetion(Exception):
    def __init__(self, *args):
        self.message = "No valid model"
        if len(args) > 0:
            self.message = f"{self.message} with {args}"

    def __str__(self):
        return self.message


class GraphModifier():
    def __init__(self, validate, operations, samples_per_class, edit_distance_one, edit_distance_two, edit_distance_three):
        self.validate = validate
        self.operations = operations
        self.samples_per_class = samples_per_class

        total = edit_distance_one + edit_distance_two + edit_distance_three
        edit_distance_one = edit_distance_one / total
        edit_distance_two = edit_distance_two / total
        edit_distance_three = edit_distance_three / total
        self.modify_ratio = [edit_distance_one, edit_distance_two, edit_distance_three]
        self.modify_functions = [self.generate_edit_distance_one_model, self.generate_edit_distance_two_model, self.generate_edit_distance_three_model]

    def _random_matrix_idx_generator(self, len_matrix, repeat):
        indices = []
        for i in range(len_matrix):
            for j in range(i + 1, len_matrix):
                indices.append((i, j))

        random.shuffle(indices)
        return indices[:repeat]

    def _random_op_generator(self, ops, repeat):
        assert repeat <= len(ops) - 2

        indices = list(range(1, len(ops) - 1))
        random.shuffle(indices)

        op_pairs = []
        # exclude INPUT, OUTPUT node
        for idx in indices[:repeat]:
            new_ops = list(self.operations - set(ops[idx]))
            new_op = random.choice(new_ops)
            op_pairs.append((idx, new_op))

        return op_pairs

    def _generate_edit_edge_model(self, original_matrix, ops, edit_distance):
        len_matrix = len(original_matrix)

        max_tries = 21  # number of matrix indices in upper triangle

        fake_ops = ["input"] + [list(self.operations)[0]] * (len_matrix - 2) + ["output"]
        for _ in range(int(max_tries)):
            matrix = original_matrix.copy()
            matrix_idxs = self._random_matrix_idx_generator(len_matrix, repeat=edit_distance)
            row = [idx[0] for idx in matrix_idxs]
            col = [idx[1] for idx in matrix_idxs]
            matrix[row, col] = 1 - matrix[row, col]

            if self.validate(matrix, fake_ops):
                return (matrix, ops)
        raise NoValidModelExcpetion(f"edit_distance={edit_distance}", original_matrix)

    def _generate_edit_node_model(self, matrix, original_ops, edit_distance):
        op_pairs = self._random_op_generator(original_ops, repeat=edit_distance)
        ops = original_ops.copy()
        for idx, op in op_pairs:
            ops[idx] = op
        return (matrix, ops)

    def get_delete_one_node_model(self, original_matrix, original_ops):
        inwards = np.sum(original_matrix, axis=0)
        outwards = np.sum(original_matrix, axis=1)
        indices = np.where((inwards + outwards) == 2)[0]
        random.shuffle(indices)

        for idx in indices:
            matrix = original_matrix.copy()
            matrix = np.delete(matrix, idx, axis=0)
            matrix = np.delete(matrix, idx, axis=1)
            ops = [op for op_idx, op in enumerate(original_ops) if op_idx != idx]
            if self.validate(matrix, ops):
                return (matrix, ops)

        raise NoValidModelExcpetion("one-node deleted", original_matrix, original_ops)

    def generate_edit_distance_one_model(self, matrix, ops):
        modify_options = [self._generate_edit_node_model, self._generate_edit_edge_model]
        return random.choice(modify_options)(matrix, ops, edit_distance=1)

    def generate_edit_distance_two_model(self, matrix, ops):
        choice = random.choice(range(3))
        if choice == 0:
            # replace 2 node
            return self._generate_edit_node_model(matrix, ops, edit_distance=2)
        elif choice == 1:
            # replace 1 node + edit 1 edge
            new_matrix, _ = self._generate_edit_edge_model(matrix, ops, edit_distance=1)
            _, new_ops = self._generate_edit_node_model(matrix, ops, edit_distance=1)
            return (new_matrix, new_ops)
        else:
            # edit 2 edge
            return self._generate_edit_edge_model(matrix, ops, edit_distance=2)

    def generate_edit_distance_three_model(self, matrix, ops):
        choice = random.choice(range(4))
        if choice == 0:
            return self.get_delete_one_node_model(matrix, ops)
        elif choice == 1:
            # replace 3 node
            return self._generate_edit_node_model(matrix, ops, edit_distance=3)
        elif choice == 2:
            # replace 2 node + edit 1 edge
            new_matrix, _ = self._generate_edit_edge_model(matrix, ops, edit_distance=1)
            _, new_ops = self._generate_edit_node_model(matrix, ops, edit_distance=2)
            return (new_matrix, new_ops)
        elif choice == 3:
            # replace 1 node + edit 2 edge
            new_matrix, _ = self._generate_edit_edge_model(matrix, ops, edit_distance=2)
            _, new_ops = self._generate_edit_node_model(matrix, ops, edit_distance=1)
            return (new_matrix, new_ops)
        else:
            # edit 3 edge
            return self._generate_edit_edge_model(matrix, ops, edit_distance=3)

    def generate_modified_models(self, matrix, ops):
        generated_count = 0
        choices = range(len(self.modify_functions))
        while True:
            if generated_count >= self.samples_per_class:
                raise StopIteration
            try:
                generated_count += 1
                yield self.modify_functions[np.random.choice(choices, p=self.modify_ratio)](matrix, ops)
            except NoValidModelExcpetion:
                pass
