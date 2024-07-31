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

    def __init__(self, model):
        """
        Initializes the Algorithm with the given model, solver,
        verbosity, and shots.

        Parameters
        ----------
        model : Model
            The optimization model to be solved.
        """
        self.model = model
        self.qubo: QuadraticProgram | None = None
        self.iteration = 0

    @property
    @abstractmethod
    def num_params(self):
        ...

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

    @abstractmethod
    def parse_to_vqc(self):
        """
        Returns the variational circuit version of the algorithm using
        OpenQASM3 input types.

        Returns
        -------
        str
            The string representing the OpenQASM3 variational quantum circuit.
        """
        ...
