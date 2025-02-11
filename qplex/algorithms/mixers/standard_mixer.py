from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class StandardMixer(QuantumMixer):
    """Standard X-mixer implementation for QAOA.
    """

    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate RX rotation gates for each qubit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits
        theta : str
            Mixing angle parameter

        Returns
        -------
        List[str]
            OpenQASM3 RX rotation instructions
        """
        return [f"rx(2 * {theta}) q[{i}];" for i in range(n_qubits)]
