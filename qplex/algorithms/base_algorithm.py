from abc import ABC, abstractmethod
from typing import List

from qplex.solvers.base_solver import Solver
import numpy as np


class Algorithm(ABC):

    """Abstract class for a quantum algorithm"""

    def __int__(self, model):
        ...

    @abstractmethod
    def create_circuit(self, model) -> str:
        """Creates a quantum circuit in the form of a OpenQASM2 string from an optimization model.

        Args:
            model: The Qmodel to be converted.

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

    @abstractmethod
    def cost_function(self, params: np.ndarray) -> float:
        """Defines the cost function to be used for the classical optimization routine.

        Args:
            params: The new set of parameters for the circuit.

        Returns:
            The cost for the current parameters.
        """
        ...

    @abstractmethod
    def get_starting_point(self) -> np.ndarray:
        """Defines what is going to be the starting point of the optimization.

        Returns:
            An array representing the starting point.
        """
        ...
