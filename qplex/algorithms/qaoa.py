import numpy as np

from qplex.algorithms.base_algorithm import Algorithm
from qplex.algorithms.mixers import QuantumMixer
from qplex.utils.circuit_utils import replace_params


class QAOA(Algorithm):
    """
    Quantum Approximate Optimization Algorithm (QAOA).

    This class implements the QAOA algorithm for solving combinatorial
    optimization problems using quantum resources. It creates a quantum
    circuit, updates parameters, and provides a starting point for the
    optimization process.

    Attributes
    ----------
    p : int
        The number of repetitions of the two parameterized unitary operations.
    n : int
        The number of qubits in the quantum circuit, which corresponds to
        the number of binary variables in the optimization problem.
    circuit : str
        The OpenQASM3 representation of the quantum circuit.
    num_params : int
        The number of parameters for the QAOA variational circuit, which is
        equal to 2 times the number of repetitions (p).
    """

    def __init__(self, model, p: int, seed: int, penalty: float,
                 mixer: QuantumMixer = None):
        """
        Initializes the QAOA algorithm with the given parameters.

        Parameters
        ----------
        model : QModel
            The optimization model to be solved, represented as a
            Quadratic Unconstrained Binary Optimization (QUBO) problem.
        p : int
            The number of repetitions of the two parameterized unitary
            operations (often referred to as the "depth" of the QAOA circuit).
        seed : int
            The seed for the random number generator, ensuring reproducibility.
        penalty : float
            The penalty factor for the QUBO conversion, used to penalize
            constraint violations in the QUBO formulation.
        mixer : QuantumMixer
            The mixer or driver of the QAOA variational circuit
        """
        super().__init__(model)
        self.p: int = p
        self.n: int = 0
        self.num_params = 2 * self.p
        if mixer is None:
            raise ValueError(
                f"Expected mixer to be provided, got {mixer}")
        self.mixer = mixer
        self.circuit: str = self.create_circuit(penalty=penalty)
        np.random.seed(seed)

    def create_circuit(self, *args, **kwargs) -> str:
        """
        Creates a quantum circuit in the form of an OpenQASM3 string
        for QAOA.

        The circuit is constructed based on the QUBO encoding of the
        model, including all the necessary gates for the QAOA ansatz.
        The ansatz includes mixing and problem unitary operations with
        parameterized rotation angles.

        Parameters
        ----------
        kwargs : dict
            Optional parameters, including 'penalty' for the QUBO
            conversion.

        Returns
        -------
        str
            An OpenQASM3 string representing the quantum circuit for
            QAOA.
        """
        self.qubo = self.model.get_qubo(penalty=kwargs['penalty'])
        self.n = self.qubo.get_num_binary_vars()

        circuit_lines = [f"input float[64] theta{i};" for i in
                         range(self.num_params)]

        circuit_lines.extend([f"qreg q[{self.n}];", f"creg c[{self.n}];"])

        circuit_lines.extend([f"h q[{i}];" for i in range(self.n)])

        linear_terms = self.qubo.objective.linear.to_array()
        quadratic_terms = self.qubo.objective.quadratic.to_array()

        for idx in range(self.p):
            theta_2idx = f"theta{2 * idx}"
            theta_2idx_plus_1 = f"theta{2 * idx + 1}"

            for i, w in enumerate(linear_terms):
                h_sum = sum(quadratic_terms[i])
                circuit_lines.append(
                    f"rz({theta_2idx} * {(w + h_sum)}) q[{i}];")

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    w = quadratic_terms[i, j]
                    if w != 0:
                        circuit_lines.append(f"cx q[{i}], q[{j}];")
                        circuit_lines.append(
                            f"rz({theta_2idx} * {w / 2}) q[{j}];")
                        circuit_lines.append(f"cx q[{i}], q[{j}];")

            circuit_lines.extend(
                self.mixer.generate_circuit(self.n, theta_2idx_plus_1))

        circuit_lines.extend(
            [f"measure q[{i}] -> c[{i}];" for i in range(self.n)])

        return "\n".join(circuit_lines)

    def update_params(self, params: np.ndarray) -> str:
        """
        Updates the parameters of the QAOA circuit.

        The circuit is updated by replacing the placeholder angles ('thetaX')
        with the values provided in the parameter array. This allows the
        quantum circuit to be re-parameterized as part of a variational
        optimization process.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the QAOA circuit, typically representing
            rotation angles in the circuit's gates.

        Returns
        -------
        str
            The updated OpenQASM3 string for the QAOA circuit with the new
            parameter values.
        """
        if len(params) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, got {len(params)}")
        return replace_params(self.circuit, params)

    def get_starting_point(self) -> np.ndarray:
        """
        Defines the starting point for the QAOA optimization.

        The starting point consists of a set of randomly initialized
        parameters (rotation angles) for the variational QAOA circuit.

        Returns
        -------
        np.ndarray
            An array representing the starting point for QAOA, initialized
            with random values between 0 and 1.
        """
        return np.random.rand(2 * self.p)
