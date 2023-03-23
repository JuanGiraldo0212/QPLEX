from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp, to_ising

class GateBasedSolver:

    def solve(self, model):
        simulator = Aer.get_backend('aer_simulator')
        test = self.parse_model(model)
        rdy = test.assign_parameters(parameters=[1.0] * len(test.parameters))
        circ = transpile(rdy, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(circ)
        print(counts)


    def parse_model(self, model):
        q_mod = from_docplex_mp(model)
        qubo = QuadraticProgramToQubo().convert(q_mod)
        ising, offset = to_ising(qubo)
        qubits = ising.num_qubits
        circ = QuantumCircuit(qubits)
        operators = ising.to_pauli_op()
        gammas = ParameterVector('gamma', 1)
        betas = ParameterVector('beta', 1)
        circ.h([i for i in range(qubits)])
        for i in range(qubits):
            op = operators[i]
            coeff = op.coeff
            paulis = op.primitive
            indices = [i for i, ltr in enumerate(paulis) if ltr == "Z"]
            if len(indices) == 1:
                circ.rz(-coeff * gammas[0], indices[0])
            elif len(indices) == 2:
                circ.rzz(coeff * gammas[0], indices[0], indices[1])
                circ.rzz(coeff * gammas[0], indices[1], indices[0])
        circ.rx(2 * betas[0], [i for i in range(qubits)])
        circ.measure_all()
        return circ



    def parse_response(self, response):
        pass