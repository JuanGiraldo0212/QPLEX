from typing import Any
from braket.aws import AwsDevice
from qplex.solvers.base_solver import Solver
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.devices import LocalSimulator


class BraketSolver(Solver):

    def __init__(self, shots: int, backend: str):
        self.shots = shots
        self.backend = backend

    def solve(self, model: str):
        qc = self.parse_input(model)
        backend = self.select_backend(0)
        response = backend.run(qc, shots=self.shots).result()
        counts = self.parse_response(response)
        return counts

    def parse_input(self, circuit: str):
        circuit = ("""
        OPENQASM 3.0;
        """ + circuit).replace("cx", "cnot")
        return OpenQASMProgram(source=circuit)

    def parse_response(self, response):
        return response.measurement_counts

    def select_backend(self, qubits: int) -> Any:
        if self.backend != "simulator":
            return AwsDevice(f"arn:aws:braket:::{self.backend}")
        return LocalSimulator(backend="braket_sv")
