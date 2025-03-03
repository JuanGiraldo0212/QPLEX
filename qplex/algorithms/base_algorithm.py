from abc import ABC, abstractmethod
from qiskit_optimization import QuadraticProgram
import numpy as np

from qplex.solvers.base_solver import Solver


class Algorithm(ABC):
    """
    Abstract base class for quantum algorithms.

    This class provides a template for quantum algorithms, encapsulating
    the common elements and enforcing the implementation of specific methods
    in subclasses. It is designed for variational quantum algorithms like
    QAOA and VQE, which require creating quantum circuits and updating
    parameters for optimization.

    Attributes
    ----------
    model : Model
        The optimization model to be solved, typically defined using the
        QuadraticProgram class from qiskit_optimization.
    qubo : QuadraticProgram or None
        The QUBO (Quadratic Unconstrained Binary Optimization) encoding of
        the problem, initialized as None and generated as needed.
    iteration : int
        The current iteration number of the optimization process, used to
        track the progress of the algorithm.
    circuit : str or None
        A string representation of the quantum circuit, in OpenQASM3
        format. Initially set to None and constructed via the `create_circuit`
        method in the subclass.
    """

    def __init__(self, model):
        """
        Initializes the Algorithm with the provided optimization model.

        Parameters
        ----------
        model : Model
            The optimization model to be solved, which should be an instance
            of QModel or similar.
        """
        self.model = model
        self.qubo: QuadraticProgram | None = None  # Holds the QUBO encoding
        self.iteration = 0  # Tracks the current iteration of the optimization
        self.circuit = None  # Quantum circuit string, initialized as None

    @abstractmethod
    def create_circuit(self) -> str: # pragma: no cover
        """
        Creates a quantum circuit in the form of an OpenQASM3 string from
        an optimization model.

        This method constructs the circuit using the quantum algorithm's
        ansatz and problem-specific gates. It needs to be implemented by
        subclasses like QAOA or VQE.

        Returns
        -------
        str
            An OpenQASM3 string representing the quantum circuit.
        """
        ...

    @abstractmethod
    def update_params(self, params: np.ndarray) -> str: # pragma: no cover
        """
        Updates the parameters of the quantum circuit.

        This method updates the circuit's gates according to new parameters,
        often used in variational quantum algorithms where parameters are
        iteratively optimized.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the circuit, typically representing
            angles for rotation gates or other tunable parameters.

        Returns
        -------
        str
            The updated OpenQASM3 string with the new parameter values.
        """
        ...

    @abstractmethod
    def get_starting_point(self) -> np.ndarray: # pragma: no cover
        """
        Defines the starting point for the optimization process.

        This method provides the initial parameter values for the optimization
        routine, often used in variational quantum algorithms to begin
        parameter tuning.

        Returns
        -------
        np.ndarray
            An array representing the starting point parameters for the
            optimization.
        """
        ...

    def remove_parameters(self):
        """
        Removes the parameter input lines from the quantum circuit string.

        This method removes lines in the circuit string that declare
        parameter inputs (i.e., `input float[64] thetaX;`) but keeps the
        placeholders (like 'thetaX') in the circuit. This is useful for
        working with unparameterized circuits or circuits where parameters
        are defined externally, such as when using the GGAE workflow from
        QPLEX.

        Raises
        ------
        AttributeError
            If the 'circuit' attribute is not defined or is None.
        """
        if not hasattr(self, 'circuit') or self.circuit is None:
            raise AttributeError(
                "The 'circuit' attribute is not defined. Ensure the circuit "
                "is instantiated in the specific algorithm.")

        lines = self.circuit.splitlines()

        filtered_lines = [line for line in lines if
                          not line.startswith("input float[64]")]

        self.circuit = "\n".join(filtered_lines)
