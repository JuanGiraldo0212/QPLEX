import numpy as np
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qplex.algorithms.base_algorithm import Algorithm
from qplex.solvers.base_solver import Solver


class VQE(Algorithm):

    def __init__(self, model, solver: Solver, verbose: bool, shots: int,
                 layers: int, seed: int, penalty: float, ansatz: str):
        super().__init__(model, solver, verbose, shots)
        self.layers: int = layers
        self.n: int = 0
        self.ansatz: str = self.create_circuit()
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
        pc = 0
        for i in range(self.n):
            circuit += f"ry(ry_angle_{pc}) q[{i}];\n"
            pc += 1

        for d in range(self.layers):
            for i in range(self.n - 1):
                circuit += f"cx q[{i}], q[{i + 1}];\n"
                circuit += f"ry(ry_angle_{pc}) q[{i}];\n"
                pc += 1
                circuit += f"ry(ry_angle_{pc}) q[{i + 1}];\n"
                pc += 1
                circuit += f"cx q[{i}], q[{i + 1}];\n"
                circuit += f"ry(ry_angle_{pc}) q[{i}];\n"
                pc += 1
                circuit += f"ry(ry_angle_{pc}) q[{i + 1}];\n"
                pc += 1

        for i in range(self.n):
            circuit += f"measure q[{i}] -> c[{i}];\n"

        return circuit

    def update_params(self, params: np.ndarray) -> str:
        updated_circuit = self.ansatz
        for pc, param in enumerate(params):
            updated_circuit = updated_circuit.replace(f"ry_angle_{pc}",
                                                      str(param))
        return updated_circuit

    def get_starting_point(self) -> np.ndarray:
        return np.random.rand(self.n + (4 * (self.n - 1) * self.layers))
