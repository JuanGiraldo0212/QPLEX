from qplex import QModel
from qplex.model.options import Options


def model_discrete_knapsack(items, values, weights, capacity, max_quantities):
    n_items = len(values)

    model = QModel('discrete-knapsack')

    x = model.integer_var_list(n_items, lb=0, ub=max_quantities, name='item')

    (model.add_constraint(
        sum((weights[i] * x[i] for i in range(n_items))) <= capacity),
     'weight_constraint')
    obj_fun = sum(values[i] * x[i] for i in range(n_items))

    model.set_objective('max', obj_fun)

    return model


def main():
    items = ['laptop', 'phone', 'camera', 'watch', 'headphones']
    weights = [25, 5, 10, 2, 3]
    values = [2000, 1000, 800, 500, 300]
    capacity = 30
    max_quantities = [2, 1, 2, 1, 1]

    knapsack_model = model_discrete_knapsack(items, values, weights, capacity,
                                             max_quantities)

    execution_params = {
        "provider": "ibmq",
        # Change to the desired backend (i.e., ibmq_sherbrooke)
        "backend": "simulator",
        "verbose": True,
        # "penalty": 10,
        "algorithm": "qaoa",
        "p": 8,
        "shots": 10240,
        "max_iter": 10000,
    }

    knapsack_model.solve('quantum', Options(**execution_params))
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
