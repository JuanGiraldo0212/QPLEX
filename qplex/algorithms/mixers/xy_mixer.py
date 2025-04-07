from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class XYMixer(QuantumMixer):
    """XY-mixer implementation.

    Implements an XY-mixing Hamiltonian that preserves the Hamming weight of
    the quantum state, suitable for problems with fixed-sum or cardinality
    constraints.
    """

    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate circuit for XY-mixing operation.

        Creates a series of controlled operations between qubit pairs to
        implement the XY-mixer while preserving total number of 1s in the
        state.

        Parameters
        ----------
        n_qubits : int
            Number of qubits
        theta : str
            Mixing angle parameter

        Returns
        -------
        List[str]
            OpenQASM3 instructions for XY-mixer
        """
        lines = []
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            lines.extend([
                f"h q[{i}]; h q[{j}];",
                f"cx q[{i}], q[{j}];",
                f"rz(2*{theta}) q[{j}];",
                f"cx q[{i}], q[{j}];",
                f"h q[{i}]; h q[{j}];"
            ])
        return lines
