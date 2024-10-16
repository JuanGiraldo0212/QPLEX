from typing import List
from qplex import QModel
from qplex.model.options import Options


def model_knapsack_problem(values: List, weights: List, const: int) -> QModel:
    n_items = len(values)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n_items, name="x")
    knapsack_model.add_constraint(
        sum(weights[i] * x[i] for i in range(n_items)) <= const)
    obj_fn = sum(values[i] * x[i] for i in range(n_items))
    knapsack_model.set_objective('max', obj_fn)

    return knapsack_model


def main():
    values = [10, 5, 18, 12, 15, 1, 2, 8]
    weights = [4, 2, 5, 4, 5, 1, 3, 5]
    const = 15

    knapsack_model = model_knapsack_problem(values, weights, const)

    execution_params = {
        "provider": "ibmq",
        # Change to the desired backend (i.e., ibmq_sherbrooke)
        "backend": "simulator",
        "verbose": True,
        "penalty": 10,
        "algorithm": "qaoa",
        "p": 2,
        "max_iter": 500,
        "shots": 1024
    }

    knapsack_model.solve("quantum", Options(**execution_params))
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
