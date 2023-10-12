from qiskit.providers import Backend
from qplex.solvers.base_solver import Solver
from qiskit import QuantumCircuit, Aer, execute


class IBMQSolver(Solver):

    def __init__(self, shots: int, backend: str):
        self.shots = shots
        self.backend = backend

    def solve(self, model: str):
        qc = self.parse_input(model)
        qc.measure_all()
        backend = self.select_backend(qc.num_qubits)
        result = execute(qc, backend, shots=self.shots).result()
        counts = result.get_counts(qc)
        counts = self.parse_response(counts)
        return counts

    def parse_input(self, circuit: str):
        qc = QuantumCircuit().from_qasm_str(circuit)
        return qc

    def parse_response(self, response):
        parsed_response = {}
        for sample, count in response.items():
            x = [int(bit) for bit in reversed(sample)]
            parsed_response["".join(str(n) for n in x)] = count

        return parsed_response

    def select_backend(self, qubits: int) -> Backend:
        # TODO
        return Aer.get_backend("qasm_simulator")
