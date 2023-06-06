from abc import ABC, abstractmethod
from typing import Dict, Any


class Solver(ABC):

    """Abstract class for a quantum solver"""

    @abstractmethod
    def solve(self, formulation) -> Dict:
        """Determines how the solver will execute the problem formulation.

        Args:
            formulation: The formulation to be solved. Can be a QModel or a 2 string.

        Returns:
            A dictionary with the solution for the execution.
        """
        ...

    @abstractmethod
    def parse_input(self, input_form) -> Any:
        """Determines how the solver needs to parse the input model.

        Args:
            input_form: The input formulation to be parsed. Can be a QModel or a 2 string.

        Returns:
            The object needed to solve the optimization problem.
        """
        ...

    @abstractmethod
    def parse_response(self, response: Any) -> Dict:
        """Determines how the solver needs to parse the response of the backend.

        Args:
            response: The response from the backend to be parsed.

        Returns:
            A dictionary with the parsed response.
        """
        ...

    @abstractmethod
    def select_backend(self, qubits: int) -> Any:
        """Selects the most appropriate backend with at least a certain number of qubits.

        Args:
            qubits: The minimum number of qubits.

        Returns:
            The selected backend for a given provider.
        """
        ...

