from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class PartitionMixer(QuantumMixer):
    """Mixer implementation for partition/grouping problems.

    Implements SWAP-based mixing operations suitable for problems requiring
    partition of items into groups while preserving group sizes.
    """

    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate circuit for partition-preserving mixing.

        Creates SWAP operations between adjacent qubits followed by rotations
        to mix states while maintaining partition structure.

        Parameters
        ----------
        n_qubits : int
            Number of qubits
        theta : str
            Mixing angle parameter

        Returns
        -------
        List[str]
            OpenQASM3 instructions for partition mixer
        """
        lines = []
        for i in range(0, n_qubits - 1, 2):
            lines.extend([
                f"swap q[{i}], q[{i + 1}];",
                f"rz({theta}) q[{i}];",
                f"rz({theta}) q[{i + 1}];"
            ])
        return lines

    def get_valid_constraints(self) -> List[str]:
        """Return supported constraint types.

        Returns
        -------
        List[str]
            Valid constraint types: partition and grouping
        """
        return ["partition", "grouping"]
