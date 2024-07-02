import numpy as np
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from qplex.algorithms.base_algorithm import Algorithm
from qplex.solvers.base_solver import Solver


class QAOA(Algorithm):

    def __init__(self, model, solver: Solver, verbose: bool, shots: int,
                 p: int, seed: int, penalty: float):
        super().__init__(model, solver, verbose, shots)
        self.p: int = p
        self.n: int = 0
        self.circuit: str = self.create_circuit()
        self.penalty = penalty
        np.random.seed(seed)

    def create_circuit(self) -> str:
        mod = from_docplex_mp(self.model)
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(mod)
        self.qubo = qubo
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
        return np.random.rand(2 * self.p)
