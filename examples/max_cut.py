import networkx as nx
import numpy as np

from qplex import QModel
from qplex.model.options import Options


def model_max_cut_problem() -> QModel:
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, 6, 1))
    edges = [(0, 1, 2.0), (0, 2, 3.0), (0, 3, 2.0), (0, 4, 4.0), (0, 5, 1.0),
             (1, 2, 4.0), (1, 3, 1.0), (1, 4, 1.0),
             (1, 5, 3.0), (2, 4, 2.0), (2, 5, 3.0), (3, 4, 5.0), (3, 5, 1.0)]
    graph.add_weighted_edges_from(edges)
    weight_matrix = nx.adjacency_matrix(graph)
    shape = weight_matrix.shape
    size = shape[0]
    qubo_matrix = np.zeros((size, size))
    qubo_vector = np.zeros(size)
    for i in range(size):
        for j in range(size):
            qubo_matrix[i, j] -= weight_matrix[i, j]
    for i in range(size):
        for j in range(size):
            qubo_vector[i] += weight_matrix[i, j]
    model = QModel('max_cut')
    x = model.binary_var_list(6, name="x")
    linear_terms = sum(qubo_vector[i] * x[i] for i in range(size))
    quadratic_terms = sum(
        qubo_matrix[i][j] * x[i] * x[j] for i in range(size) for j in
        range(size))
    obj_fn = linear_terms + quadratic_terms
    model.set_objective('max', obj_fn)

    return model


def main():
    max_cut_model = model_max_cut_problem()

    execution_params = {
        "provider": "ibmq",
        "backend": "simulator",
        # Change to the desired backend (i.e., ibmq_qasm_simulator)
        "algorithm": "qaoa",
        "p": 2,
        "max_iter": 500,
        "shots": 5000
    }

    max_cut_model.solve("quantum", Options(**execution_params))
    print(max_cut_model.print_solution())


if __name__ == '__main__':
    main()
