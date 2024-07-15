from abc import ABC, abstractmethod
from qiskit_optimization import QuadraticProgram
import numpy as np

from qplex.solvers.base_solver import Solver


class Algorithm(ABC):
    """Abstract class for a quantum algorithm"""

    def __init__(self, model, solver: Solver, verbose: bool,
                 shots: int):
        self.model = model
        self.solver = solver
        self.verbose = verbose
        self.shots: int = shots
        self.qubo: QuadraticProgram | None = None
        self.iteration = 0

    @abstractmethod
    def create_circuit(self) -> str:
        """Creates a quantum circuit in the form of a OpenQASM2 string from
        an optimization model.

        Returns:
            An OpenQASM2 string.
        """
        ...

    @abstractmethod
    def update_params(self, params: np.ndarray) -> str:
        """Updates the parameters from an openQASM2 string.

        Args:
            params: The new set of parameters for the circuit.

        Returns:
            The updated OpenQASM2 string.
        """
        ...

    def cost_function(self, params: np.ndarray) -> float:
        """Defines the cost function to be used for the classical
        optimization routine.

        Args:
            params: The new set of parameters for the circuit.

        Returns:
            The cost for the current parameters.
        """
        qc = self.update_params(params)
        counts = self.solver.solve(qc)
        energy = 0
        for sample, count in counts.items():
            sample = [int(n) for n in sample]
            energy += count * self.qubo.objective.evaluate(sample)
        self.iteration += 1
        cost = energy / self.shots
        if self.verbose:
            print(f'Iteration {self.iteration}:\nCost = {cost}')
        return cost
        ##########################
        # qc = self.update_params(params)
        # expect_value = compute_expectation_value(qc, self.qubo.to_ising()[
        # 0], self.solver)
        # return expect_value

    @abstractmethod
    def get_starting_point(self) -> np.ndarray:
        """Defines what is going to be the starting point of the optimization.

        Returns:
            An array representing the starting point.
        """
        ...
