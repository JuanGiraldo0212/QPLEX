from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class CompositeMixer(QuantumMixer):
    """Mixer combining multiple constraint-preserving mixers.

    Allows composition of multiple mixer types to handle problems with
    multiple types of constraints simultaneously.
    """

    def __init__(self, mixers: List[QuantumMixer]):
        """Initialize with list of component mixers.

        Parameters
        ----------
        mixers : List[QuantumMixer]
            List of mixer instances to combine
        """
        self.mixers = mixers

    def generate_circuit(self, n_qubits: int, theta: str) -> List[str]:
        """Generate composite mixing circuit.

        Concatenates the circuits from all component mixers in sequence.

        Args
        ----
        n_qubits : int
            Number of qubits
        theta : str
            Mixing angle parameter

        Returns
        -------
        List[str]
            Combined OpenQASM3 instructions from all mixers
        """
        lines = []
        for mixer in self.mixers:
            lines.extend(mixer.generate_circuit(n_qubits, theta))
        return lines

    def get_valid_constraints(self) -> List[str]:
        """Return union of supported constraint types from all mixers.

        Returns
        -------
        List[str]
            Combined list of unique constraint types
        """
        constraints = []
        for mixer in self.mixers:
            constraints.extend(mixer.get_valid_constraints())
        return list(set(constraints))
