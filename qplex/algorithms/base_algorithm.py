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
    qubo : QuadraticProgram or None
        The QUBO (Quadratic Unconstrained Binary Optimization) encoding of
        the problem.
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
        self.circuit = None

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

    def remove_parameters(self):
        """
        Removes the theta parameters from the OpenQASM3 parameterized
        string. It does not change the placeholders for the parameters in
        the circuit.
        """
        if not hasattr(self, 'circuit') or self.circuit is None:
            raise AttributeError(
                "The 'circuit' attribute is not defined. Ensure the circuit "
                "is instantiated in the specific algorithm.")

        lines = self.circuit.splitlines()

        filtered_lines = [line for line in lines if
                          not line.startswith("input float[64]")]

        self.circuit = "\n".join(filtered_lines)
