from qplex import QModel
from qplex.model.options import Options


def model_discrete_knapsack(values, weights, const):
    n_items = len(values)

    model = QModel('discrete-knapsack')

    x = model.integer_var_list(n_items, lb=0, ub=2, name='x')
    model.add_constraint(
        sum((weights[i] * x[i] for i in range(n_items))) <= const)
    obj_fun = sum(values[i] * x[i] for i in range(n_items))

    model.set_objective('max', obj_fun)

    return model


def main():
    values = [10, 5, 18, 12, 15, 1, 2, 8]
    weights = [4, 2, 5, 4, 5, 1, 3, 5]
    const = 15

    knapsack_model = model_discrete_knapsack(values, weights, const)

    execution_params = {
        "provider": "ibmq",
        # Change to the desired backend (i.e., ibmq_sherbrooke)
        "backend": "simulator",
        "verbose": True,
        # "penalty": 10,
        "algorithm": "qaoa",
        "p": 6,
        "shots": 5120,
        "max_iter": 10000,
    }

    knapsack_model.solve('quantum', Options(**execution_params))
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
