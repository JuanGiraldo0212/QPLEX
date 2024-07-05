from qplex import KnapsackModel
from qplex.model import Options

import numpy as np


def main():
    values = [10, 5, 18, 12, 15, 1, 2, 8]
    weights = [4, 2, 5, 4, 5, 1, 3, 5]
    const = 15
    penalty = int(2 * np.max(values))

    knapsack_model = KnapsackModel('knapsack', values, weights, const, penalty)

    execution_params = {
        "provider": "ibmq",
        "verbose": True,
        "backend": "simulator",
        # Change to the desired backend (i.e., ibmq_qasm_simulator)
        "algorithm": "qaoa",
        "p": 2,
        "max_iter": 500,
        "shots": 10000
    }

    knapsack_model.solve("quantum", Options(**execution_params))
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
