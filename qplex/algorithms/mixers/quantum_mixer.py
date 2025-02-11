from abc import ABC, abstractmethod
from typing import List


class QuantumMixer(ABC):
    """Abstract base class defining the interface for quantum mixers.

    Provides a template for implementing various mixing operators that preserve
    problem constraints while exploring the solution space.
    """

    @abstractmethod
    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate OpenQASM3 circuit instructions for the mixing operation.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit
        theta : str
            Mixing angle parameter name/value

        Returns
        -------
        List[str]
            OpenQASM3 instructions implementing the mixer
        """
        pass
