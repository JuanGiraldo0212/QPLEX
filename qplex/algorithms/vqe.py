import numpy as np

from qplex.algorithms.base_algorithm import Algorithm
import qplex


class VQE(Algorithm):
    """
    Variational Quantum Eigensolver (VQE).

    This class implements the VQE algorithm for solving optimization problems
    using a variational approach. It creates a quantum circuit, updates
    parameters, and provides a starting point for the optimization.

    Attributes
    ----------
    layers : int
        The number of layers in the VQE ansatz.
    n : int
        The number of qubits in the quantum circuit.
    circuit : str
        The OpenQASM3 representation of the VQE circuit.
    num_params : int
        The number of parameters for the VQE variational circuit.
    """

    def __init__(self, model, layers: int, seed: int, penalty: float,
                 ansatz: str):
        """
        Initializes the VQE algorithm with the given parameters.

        Parameters
        ----------
        model : QModel
            The optimization model to be solved.
        layers : int
            The number of layers in the VQE ansatz.
        seed : int
            The seed for the random number generator.
        penalty : float
            The penalty factor for the QUBO conversion.
        ansatz : str
            The ansatz type used in the VQE algorithm.
        """
        super().__init__(model)
        self.layers: int = layers
        self.n: int = 0
        self.num_params = None
        self.circuit: str = self.create_circuit(penalty=penalty)
        np.random.seed(seed)

    def create_circuit(self, *args, **kwargs) -> str:
        """
        Creates a quantum circuit in the form of an OpenQASM3 string
        for VQE.

        The circuit is constructed based on the QUBO encoding of the
        model, including all necessary gates for the VQE ansatz.

        Parameters
        ----------
        kwargs
            Optional parameters, including 'penalty' for QUBO
            conversion.

        Returns
        -------
        str
            An OpenQASM3 string representing the quantum circuit for
            VQE.
        """
        self.qubo = self.model.get_qubo(penalty=kwargs['penalty'])
        self.n = self.qubo.get_num_binary_vars()
        self.num_params = self.n + (4 * (self.n - 1) * self.layers)

        circuit_lines = [f"input float[64] theta{i};" for i in
                         range(self.num_params)]

        circuit_lines.extend([f"qreg q[{self.n}];", f"creg c[{self.n}];"])

        pc = 0

        circuit_lines.extend(
            [f"ry(param{pc + i}) q[{i}];" for i in range(self.n)])
        pc += self.n

        for d in range(self.layers):
            for i in range(self.n - 1):
                circuit_lines.append(f"cx q[{i}], q[{i + 1}];")

                circuit_lines.append(f"ry(param{pc}) q[{i}];")
                pc += 1
                circuit_lines.append(f"ry(param{pc}) q[{i + 1}];")
                pc += 1

                circuit_lines.append(f"cx q[{i}], q[{i + 1}];")

                circuit_lines.append(f"ry(param{pc}) q[{i}];")
                pc += 1
                circuit_lines.append(f"ry(param{pc}) q[{i + 1}];")
                pc += 1

        circuit_lines.extend(
            [f"measure q[{i}] -> c[{i}];" for i in range(self.n)])

        return "\n".join(circuit_lines)

    def update_params(self, params: np.ndarray) -> str:
        """
        Updates the parameters of the VQE circuit.

        The circuit is updated by replacing the placeholder angles with
        the values provided in the parameter array.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the VQE circuit.

        Returns
        -------
        str
            The updated OpenQASM3 string for the VQE circuit.
        """

        return qplex.utils.circuit_utils.replace_params(self.circuit, params)

    def get_starting_point(self) -> np.ndarray:
        """
        Defines the starting point for the VQE optimization.

        Returns
        -------
        np.ndarray
            An array representing the starting point for VQE, initialized
            with random values.
        """
        return np.random.rand(self.n + (4 * (self.n - 1) * self.layers))
