from qplex import QModel
import networkx as nx
import numpy as np


def main():
    max_cut_example()


def max_cut_example():
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, 6, 1))
    edges = [(0, 1, 2.0), (0, 2, 3.0), (0, 3, 2.0), (0, 4, 4.0), (0, 5, 1.0), (1, 2, 4.0), (1, 3, 1.0), (1, 4, 1.0),
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
    quadratic_terms = sum(qubo_matrix[i][j] * x[i] * x[j] for i in range(size) for j in range(size))
    obj_fn = linear_terms + quadratic_terms
    model.set_objective('max', obj_fn)
    execution_params = {
        "provider": "ibmq",
        "backend": "simulator",
        # "backend": "ibmq_qasm_simulator"
        "algorithm": "qaoa",
        # "layers": 3,
        "p": 2,
        "max_iter": 500,
        "shots": 10000
    }
    model.solve("quantum", **execution_params)
    # print(model.objective_value)
    print(model.print_solution())


def knapsack_example():
    w = [4, 2, 5, 4, 5, 1, 3, 5]
    v = [10, 5, 18, 12, 15, 1, 2, 8]
    bonus = [[0],
             [3, 0],
             [9, 5, 0],
             [2, 3, 9, 0],
             [6, 5, 9, 8, 0],
             [5, 6, 9, 1, 2, 0],
             [9, 1, 6, 2, 1, 5, 0],
             [6, 9, 2, 3, 5, 4, 8, 0]]
    c = 15
    n = len(w)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n, name="x")
    knapsack_model.add_constraint(sum(w[i] * x[i] for i in range(n)) <= c)
    obj_fn = sum(v[i] * x[i] for i in range(n))
    # obj_fn = sum((v[i] * x[i] + bonus[i][j] * x[i] * x[j]) for i in range(n) for j in range(n) if i > j)
    knapsack_model.set_objective('max', obj_fn)
    # knapsack_model.solve()
    knapsack_model.solve('quantum', provider='d-wave')
    print(knapsack_model.objective_value)
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
