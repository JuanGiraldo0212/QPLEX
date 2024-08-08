import numpy as np

from qplex.algorithms.base_algorithm import Algorithm
from qplex.solvers.base_solver import Solver


class QAOA(Algorithm):
    """
    Quantum Approximate Optimization Algorithm (QAOA).

    This class implements the QAOA algorithm for solving optimization
    problems.
    It creates a quantum circuit, updates parameters, and provides a
    starting point for the optimization.

    Attributes
    ----------
    p : int
        The number of repetitions of the two parameterized unitary operations.
    n : int
        The number of qubits in the quantum circuit.
    circuit : str
        The OpenQASM2 representation of the quantum circuit.
    """

    def __init__(self, model, solver: Solver, verbose: bool,
                 shots: int, p: int, seed: int, penalty: float):
        """
        Initializes the QAOA algorithm with the given parameters.

        Parameters
        ----------
        model : QModel
            The optimization model to be solved.
        solver : Solver
            The solver to be used for solving the model.
        verbose : bool
            If True, enables verbose output.
        shots : int
            The number of shots for quantum execution.
        p : int
            The number of repetitions of the two parameterized unitary
            operations.
        seed : int
            The seed for the random number generator.
        penalty : float
            The penalty factor for the QUBO conversion.
        """
        super().__init__(model, solver, verbose, shots)
        self.p: int = p
        self.n: int = 0
        self.circuit: str = self.create_circuit(penalty=penalty)
        np.random.seed(seed)

    def create_circuit(self, *args, **kwargs) -> str:
        """
        Creates a quantum circuit in the form of an OpenQASM3 string
        for QAOA.

        The circuit is constructed based on the QUBO encoding of the
        model,
        including all the necessary gates for the QAOA ansatz.

        Parameters
        ----------
        kwargs
            Optional parameters, including 'penalty' for QUBO
            conversion.

        Returns
        -------
        str
            An OpenQASM2 string representing the quantum circuit for
            QAOA.
        """
        self.qubo = self.model.get_qubo(penalty=kwargs['penalty'])
        self.n = self.qubo.get_num_binary_vars()
        circuit = f"""
        qreg q[{self.n}];
        creg c[{self.n}];
        """
        for i in range(self.n):
            circuit += f"h q[{i}];\n"

        for idx in range(self.p):
            linear_terms = self.qubo.objective.linear.to_array()
            for i, w in enumerate(linear_terms):
                circuit += f"rz(rz_angle_{i}_{idx}) q[{i}];\n"

            quadratic_terms = self.qubo.objective.quadratic.to_array()
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    w = quadratic_terms[i, j]
                    if w != 0:
                        circuit += f"cx q[{int(i)}], q[{int(j)}];\n"
                        circuit += f"rz(rzz_angle_{i}_{j}_{idx}) q[" \
                                   f"{int(j)}];\n"
                        circuit += f"cx q[{int(i)}], q[{int(j)}];\n"

            for i in range(self.n):
                circuit += f"rx(rx_angle_{i}_{idx}) q[{i}];\n"

        for i in range(self.n):
            circuit += f"measure q[{i}] -> c[{i}];\n"

        return circuit

    def update_params(self, params: np.ndarray) -> str:
        """
        Updates the parameters of the QAOA circuit.

        The circuit is updated by replacing the placeholder angles with
        the values provided in the parameter array.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the QAOA circuit.

        Returns
        -------
        str
            The updated OpenQASM3 string for the QAOA circuit.
        """
        updated_circuit = self.circuit
        quadratic_terms = self.qubo.objective.quadratic.to_array(
            symmetric=True)
        linear_terms = self.qubo.objective.linear.to_array()
        for idx in range(self.p):
            for i, w in enumerate(linear_terms):
                h_sum = 0
                for j in range(len(linear_terms)):
                    h_sum += quadratic_terms[i][j]
                updated_circuit = updated_circuit.replace(
                    f"rz_angle_{i}_{idx}",
                    f"{params[2 * idx] * (w + h_sum)}")

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    w = quadratic_terms[i, j]
                    if w != 0:
                        updated_circuit = updated_circuit.replace(
                            f"rzz_angle_{i}_{j}_{idx}",
                            f"{params[2 * idx] * w / 2}")

            for i in range(self.n):
                updated_circuit = updated_circuit.replace(
                    f"rx_angle_{i}_{idx}", f"{2 * params[2 * idx + 1]}")

        return updated_circuit

    def get_starting_point(self) -> np.ndarray:
        """
        Defines the starting point for the QAOA optimization.

        Returns
        -------
        np.ndarray
            An array representing the starting point for QAOA,
            initialized with random values.
        """
        return np.random.rand(2 * self.p)
