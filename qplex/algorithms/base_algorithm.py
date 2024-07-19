from abc import ABC, abstractmethod
from qiskit_optimization import QuadraticProgram
import numpy as np

from qplex.solvers.base_solver import Solver


class Algorithm(ABC):
    """
    Abstract base class for a quantum algorithm.

    This class provides a template for quantum algorithms, encapsulating
    the common elements and enforcing the implementation of specific methods
    in subclasses.

    Attributes
    ----------
    model : Model
        The optimization model to be solved.
    solver : Solver
        The solver to be used for solving the model.
    verbose : bool
        If True, enables verbose output.
    shots : int
        The number of shots for quantum execution.
    qubo : QuadraticProgram or None
        The QUBO (Quadratic Unconstrained Binary Optimization) encoding of the problem.
    iteration : int
        The current iteration number of the optimization process.
    """

    def __init__(self, model, solver: Solver, verbose: bool,
                 shots: int):
        """
        Initializes the Algorithm with the given model, solver,
        verbosity, and shots.

        Parameters
        ----------
        model : Model
            The optimization model to be solved.
        solver : Solver
            The solver to be used for solving the model.
        verbose : bool
            If True, enables verbose output.
        shots : int
            The number of shots for quantum execution.
        """
        self.model = model
        self.solver = solver
        self.verbose = verbose
        self.shots: int = shots
        self.qubo: QuadraticProgram | None = None
        self.iteration = 0

    @abstractmethod
    def create_circuit(self) -> str:
        """
        Creates a quantum circuit in the form of an OpenQASM2 string from
        an optimization model.

        Returns
        -------
        str
            An OpenQASM2 string representing the quantum circuit.
        """
        ...

    @abstractmethod
    def update_params(self, params: np.ndarray) -> str:
        """
        Updates the parameters of the quantum circuit.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the circuit.

        Returns
        -------
        str
            The updated OpenQASM2 string.
        """
        ...

    def cost_function(self, params: np.ndarray) -> float:
        """
        Defines the cost function to be used for the classical optimization routine.

        This method calculates the cost based on the given parameters by
        updating the quantum circuit, solving it, and evaluating the energy.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the circuit.

        Returns
        -------
        float
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
        """
        Defines the starting point for the optimization.

        Returns
        -------
        np.ndarray
            An array representing the starting point.
        """
        ...
