from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class InequalityMixer(QuantumMixer):
    """Mixer implementation for inequality-constrained problems.

    Implements controlled rotations based on cumulative weights to preserve
    inequality constraints while mixing quantum states.
    """

    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate circuit for inequality-preserving mixing.

        Creates a sequence of controlled operations that respect inequality
        constraints during state mixing.

        Parameters
        ----------
        n_qubits : int
            Number of qubits
        theta : str
            Mixing angle parameter

        Returns
        -------
        List[str]
            OpenQASM3 instructions for inequality mixer
        """
        lines = []
        for i in range(n_qubits - 1):
            lines.extend([
                f"cx q[{i}], q[{i + 1}];",
                f"rz({theta}) q[{i + 1}];",
                f"cx q[{i}], q[{i + 1}];"
            ])
        return lines
