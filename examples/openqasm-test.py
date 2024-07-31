from qiskit import QuantumCircuit
import qiskit.qasm3
from qiskit.circuit import Parameter

params = [Parameter('θ' + str(i)) for i in range(5)]

qc = QuantumCircuit(5)

qc.h([0, 1, 2, 3, 4])
qc.rx(params[0], 0)
qc.rx(params[1], 1)
qc.rx(params[2], 2)
qc.rx(params[3], 3)
qc.rx(params[4], 4)

qc.measure_all()

print(qiskit.qasm3.dumps(qc))

