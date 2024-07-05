from typing import List

from qplex.model.qmodel import QModel
from qplex.model.qubo import QUBO, parse_hamiltonian

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from openqaoa.problems import Knapsack


class KnapsackModel(QModel):
    """
    Creates an instance of a KnapsackModel.

    Parameters
    ----------
    name: str
        The name of the model.
    values: List
        The list of values for each of the variables.
    weights: List
        The list of weights for each variable. Must be the same length as
        values.
    const: int
        The constraint for the Knapsack problem.

    Returns
    ----------
        An instance of a KnapsackModel.
    """

    def __init__(self, name: str, values: List, weights: List, const: int,
                 penalty: int | float):
        n_items = len(values)
        if n_items != len(weights):
            raise ValueError(
                "Number of values does not match the number of weights.")
        super(KnapsackModel, self).__init__(name)
        x = self.binary_var_list(n_items, name="x")
        self.add_constraint(
            sum(weights[i] * x[i] for i in range(n_items)) <= const)
        obj_fn = sum(values[i] * x[i] for i in range(n_items))
        self.set_objective('max', obj_fn)
        self.problem_definition = Knapsack(values, weights, const, penalty)

    @property
    def qubo(self) -> QuadraticProgram | QUBO:
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        # model = from_docplex_mp(self)
        # converter = QuadraticProgramToQubo()
        # return converter.convert(model)
        return parse_hamiltonian(str(self.problem_definition.qubo.hamiltonian))
