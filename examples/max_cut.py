import networkx as nx
import numpy as np

from qplex import MaxCutModel
from qplex.model import Options


def main():
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, 6, 1))
    edges = [(0, 1, 2.0), (0, 2, 3.0), (0, 3, 2.0), (0, 4, 4.0),
             (0, 5, 1.0),
             (1, 2, 4.0), (1, 3, 1.0), (1, 4, 1.0),
             (1, 5, 3.0), (2, 4, 2.0), (2, 5, 3.0), (3, 4, 5.0),
             (3, 5, 1.0)]
    graph.add_weighted_edges_from(edges)
    max_cut_model = MaxCutModel('max-cut', graph)

    execution_params = {
        "provider": "ibmq",
        "backend": "simulator",
        "verbose": True,
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
