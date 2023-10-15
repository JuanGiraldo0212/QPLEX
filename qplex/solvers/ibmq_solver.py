import qiskit.qasm3
from qiskit.providers import Backend
from qplex.solvers.base_solver import Solver
from qiskit import QuantumCircuit, Aer, execute


class IBMQSolver(Solver):

    def __init__(self, shots: int, backend: str):
        self.shots = shots
        self.backend = backend

    def solve(self, model: str):
        qc = self.parse_input(model)
        backend = self.select_backend(qc.num_qubits)
        response = execute(qc, backend, shots=self.shots).result()
        counts = self.parse_response(response)
        return counts

    def parse_input(self, circuit: str):
        circuit = """
        OPENQASM 3.0;
        include "stdgates.inc";
        """ + circuit
        qc = qiskit.qasm3.loads(circuit)
        return qc

    def parse_response(self, response):
        response = response.get_counts()
        parsed_response = {}
        for sample, count in response.items():
            x = [int(bit) for bit in reversed(sample)]
            parsed_response["".join(str(n) for n in x)] = count
        return parsed_response

    def select_backend(self, qubits: int) -> Backend:
        # TODO
        return Aer.get_backend("qasm_simulator")
