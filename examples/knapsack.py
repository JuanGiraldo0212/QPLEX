from typing import List
from qplex import QModel


def model_knapsack_problem(values: List, weights: List, const: int) -> QModel:
    n_items = len(values)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n_items, name="x")
    knapsack_model.add_constraint(sum(weights[i] * x[i] for i in range(n_items)) <= const)
    obj_fn = sum(values[i] * x[i] for i in range(n_items))
    knapsack_model.set_objective('max', obj_fn)

    return knapsack_model


def main():
    values = [10, 5, 18, 12, 15, 1, 2, 8]
    weights = [4, 2, 5, 4, 5, 1, 3, 5]
    const = 15

    knapsack_model = model_knapsack_problem(values, weights, const)

    execution_params = {
        "provider": "braket",
        "backend": "simulator",  # Change to the desired backend (i.e., ibmq_qasm_simulator)
        "algorithm": "qaoa",
        "p": 4,
        "max_iter": 500,
        "shots": 10000
    }

    knapsack_model.solve("quantum", **execution_params)
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
