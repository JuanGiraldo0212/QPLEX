from qplex.model.qmodel import QModel

import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo


class MaxCutModel(QModel):
    """
    Creates an instance of a MaxCutModel.

    Parameters
    ----------
    name: str
        The name of the model.
    g: nx.Graph
        The input graph as a NetworkX graph instance.

    Returns
    ----------
        An instance of a MaxCutModel.
    """
    def __init__(self, name: str, g: nx.Graph):
        super(MaxCutModel, self).__init__(name)
        weight_matrix = nx.adjacency_matrix(g)
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
        x = self.binary_var_list(6, name="x")
        linear_terms = sum(qubo_vector[i] * x[i] for i in range(size))
        quadratic_terms = sum(
            qubo_matrix[i][j] * x[i] * x[j] for i in range(size) for j in
            range(size))
        obj_fn = linear_terms + quadratic_terms
        self.set_objective('max', obj_fn)

    @property
    def qubo(self) -> QuadraticProgram:
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        mod = from_docplex_mp(self)
        converter = QuadraticProgramToQubo()
        return converter.convert(mod)
