import qiskit.qasm3
from qiskit import transpile
from qplex.solvers.base_solver import Solver
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


class IBMQSolver(Solver):

    def __init__(self, token: str, shots: int, backend: str):
        self.shots = shots
        self.backend = backend
        QiskitRuntimeService.save_account(channel="ibm_quantum",
                                          token=token, overwrite=True)
        self.service = QiskitRuntimeService()

    def solve(self, model: str):
        qc = self.parse_input(model)
        backend = self.select_backend(qc.num_qubits)
        transpiled_qc = transpile(qc, backend)
        if self.backend == 'simulator':
            response = backend.run(transpiled_qc).result().get_counts()
        else:
            sampler = Sampler(backend)
            pub = (transpiled_qc,)
            result = sampler.run([pub], shots=self.shots).result()
            data = result[0].data
            bits = data.c
            response = bits.get_counts()
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
        parsed_response = {}
        for sample, count in response.items():
            x = [int(bit) for bit in reversed(sample)]
            parsed_response["".join(str(n) for n in x)] = count
        return parsed_response

    def select_backend(self, qubits: int):
        if self.backend != "simulator":
            return self.service.backend(self.backend)
        return AerSimulator()
