from abc import ABC, abstractmethod
from typing import Dict, Any


class Solver(ABC):
    """
    Abstract base class for a quantum solver.

    This class defines the necessary methods for interacting with a quantum
    backend provider, including solving problems, parsing inputs and responses,
    and selecting the appropriate backend.
    """

    @abstractmethod
    def solve(self, formulation) -> Dict:
        """
        Solves the given problem formulation and returns the solution.

        Args:
            formulation: The problem formulation to be solved. This could be a
                         model instance or a string representation of the
                         problem.

        Returns:
            A dictionary containing the solution to the problem. The
            structure of this dictionary depends on the specific
            implementation and problem.
        """
        ...

    @abstractmethod
    def parse_input(self, input_form) -> Any:
        """
        Parses the input model or problem formulation into a format suitable
        for solving.

        Args:
            input_form: The input model or problem formulation, which could
            be a QModel instance or a string representation.

        Returns:
            An object or data structure that the solver can use to perform
            the optimization.
        """
        ...

    @abstractmethod
    def parse_response(self, response: Any) -> Dict:
        """
        Parses the response received from the backend solver.

        Args:
            response: The raw response from the backend. The format of this
            response depends on the specific backend and solver implementation.

        Returns:
            A dictionary containing the parsed response, which typically
            includes the solution and any relevant metadata or metrics.
        """
        ...

    @abstractmethod
    def select_backend(self, qubits: int) -> Any:
        """
        Selects and configures the appropriate backend based on the number
        of qubits required.

        Args:
            qubits: The minimum number of qubits needed for the backend.

        Returns:
            The selected backend that meets the qubit requirement. This may
            involve initialization or configuration specific to the quantum
            provider.
        """
        ...
