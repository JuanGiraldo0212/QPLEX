from typing import List
from qplex import QModel
from qplex.model.options import Options


def model_complex_knapsack_problem(values: List, weights: List, const: int,
                                   groups: List[int],
                                   max_group_items: int) -> QModel:
    n_items = len(values)
    knapsack_model = QModel('complex_knapsack')
    x = knapsack_model.binary_var_list(n_items, name="x")

    # Constraint: Total weight should not exceed capacity
    knapsack_model.add_constraint(
        sum(weights[i] * x[i] for i in range(n_items)) <= const,
        "weight_constraint")

    # Constraint: At most max_group_items can be selected from each group
    group_ids = set(groups)
    for group_id in group_ids:
        knapsack_model.add_constraint(
            sum(x[i] for i in range(n_items) if
                groups[i] == group_id) <= max_group_items,
            f"group_{group_id}_constraint")

    # Objective: Maximize the value
    obj_fn = sum(values[i] * x[i] for i in range(n_items))
    knapsack_model.set_objective('max', obj_fn)

    return knapsack_model


def main():
    # Values, weights, and groups for items
    values = [20, 15, 40, 25, 35, 10, 50, 45, 30, 60, 25, 55]
    weights = [5, 4, 8, 6, 7, 3, 9, 10, 6, 12, 5, 11]
    groups = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]  # Group IDs for each item
    const = 30  # Total weight capacity
    max_group_items = 2  # Max items allowed from any single group

    # Model the complex knapsack problem
    knapsack_model = model_complex_knapsack_problem(values, weights, const,
                                                    groups, max_group_items)

    execution_params = {
        "provider": "d-wave",
        # Adjust backend and other quantum options as necessary
        "backend": "simulator",
        "verbose": True,
        "penalty": 10,
        "algorithm": "vqe",
        "max_iter": 1000,
        "shots": 2048,
        "provider_options": {
            "time_limit": 6
        }
    }

    knapsack_model.solve("quantum", Options(**execution_params))
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
