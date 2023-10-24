from typing import List
from qplex import QModel


def build_knapsack_model(values: List, weights: List, max_weight: int) -> QModel:
    n_items = len(values)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n_items, name="x")
    knapsack_model.add_constraint(sum(weights[i] * x[i] for i in range(n_items)) <= max_weight)
    obj_fn = sum(values[i] * x[i] for i in range(n_items))
    knapsack_model.set_objective('max', obj_fn)

    return knapsack_model


def main():
    # Problem set-up
    # --------------
    t = 7  # Time steps
    l1 = [5, 3, 3, 6, 9, 7, 1]  # Daily return for battery 1
    l2 = [8, 4, 5, 12, 10, 11, 2]  # Daily return for battery 2
    c1 = [1, 1, 2, 1, 1, 1, 2]  # Daily degradation for battery 1
    c2 = [3, 2, 3, 2, 4, 3, 3]  # Daily degradation for battery 1
    c_max = 16  # Maximum degradation

    # Knapsack definition
    # ------------------
    values = list(map(lambda x, y: x - y, l2, l1))
    weights = list(map(lambda x, y: x - y, c2, c1))
    max_weight = c_max - sum(c1)
    knapsack_model = build_knapsack_model(values, weights, max_weight)

    # Run experiments
    # ---------------

    experiments = [
        {"provider": "d-wave"},
        {"provider": "ibmq", "algorithm": "qaoa", "p": 2, "shots": 2048},
        {"provider": "ibmq", "algorithm": "vqe", "layers": 2, "shots": 2048},
        {"provider": "braket", "algorithm": "qaoa", "p": 3, "shots": 1024},
        {"provider": "braket", "algorithm": "vqe", "layers": 3, "shots": 1024},
    ]

    for execution_params in experiments:
        knapsack_model.solve("quantum", **execution_params, backend="simulator")
        solution = knapsack_model.solution
        print(knapsack_model.print_solution())


if __name__ == "__main__":
    main()
