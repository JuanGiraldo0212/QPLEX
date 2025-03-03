import qiskit
from qiskit import QuantumCircuit

import qiskit.qasm3 as qasm3
import qiskit.transpiler.preset_passmanagers as qiskit_pm
import qiskit_aer as aer
import qiskit_ibm_runtime as qir
from qiskit.providers import BackendV2

from qplex.solvers.base_solver import Solver


class IBMQSolver(Solver):
    """
    Solver for IBMQ. Can execute circuits on IBM backend or local simulators.

    Attributes
    ----------
    shots : int
        The number of shots for the quantum experiment.
    _backend : str
        The name of the backend to be used, which can be an IBMQ
        device or a local simulator.
    service : QiskitRuntimeService
        The Qiskit runtime service instance for interacting with IBMQ's
        backend.
    optimization_level : int
        The desired optimization level for the Qiskit circuit.
    """

    def __init__(self, token: str, shots: int, backend: str,
                 optimization_level: int):
        """
        Initializes the IBMQSolver with the specified token, number of
        shots, and backend.

        Parameters
        ----------
        token : str
            The IBMQ API token for authentication.
        shots : int
            The number of shots for the quantum experiment.
        backend : str
            The backend to use for solving the problem,
            which can be an IBMQ device or a local simulator.
        optimization_level : int
            The desired optimization level for the Qiskit circuit.
        """
        self.shots = shots
        if backend is None:
            print('No backend specified. Using least busy...')
            self._backend = ''
        else:
            self._backend = backend
        qir.QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=token, overwrite=True)
        self.service = qir.QiskitRuntimeService()
        self.optimization_level = optimization_level

    @property
    def backend(self):
        """
        Returns the currently selected backend name.

        Returns
        -------
        str
            The name of the backend.
        """
        return self._backend

    def solve(self, model: str) -> dict:
        """
        Solves the given problem formulation using the specified backend.

        Parameters
        ----------
        model : str
            The quantum circuit as an OpenQASM string to be
            executed.

        Returns
        -------
        dict
            A dictionary containing the measurement counts from the
            backend.
        """
        qc = self.parse_input(model)
        backend = self.select_backend(qc.num_qubits)
        pass_manager = qiskit_pm.generate_preset_pass_manager(backend=backend,
                                                              optimization_level=
                                                              self.optimization_level)
        isa_circuit = pass_manager.run(qc)

        if self._backend == 'simulator':
            raw_counts = backend.run(isa_circuit).result().get_counts()
        else:
            sampler = qir.SamplerV2(backend)
            raw_counts = self.run(isa_circuit, sampler)
        counts = self.parse_response(raw_counts)
        return counts

    def run(self, qc, sampler):
        """
        Executes the given quantum circuit using the provided sampler.

        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit to run.
        sampler : Sampler
            The sampler instance to use for executing the circuit.

        Returns
        -------
        dict
            A dictionary with the raw measurement counts.
        """
        pub = (qc,)
        result = sampler.run([pub], shots=self.shots).result()
        data = result[0].data
        bits = data.c
        raw_counts = bits.get_counts()
        return raw_counts

    def parse_input(self, circuit: str) -> QuantumCircuit:
        """
        Converts a circuit string to a Qiskit QuantumCircuit object.

        Parameters
        ----------
        circuit : str
            The quantum circuit as an OpenQASM string.

        Returns
        -------
        qiskit.QuantumCircuit
            The quantum circuit object.
        """
        circuit = """
        OPENQASM 3.0;
        include "stdgates.inc";
        """ + circuit
        qc = qasm3.loads(circuit)
        return qc

    def parse_response(self, response: dict) -> dict:
        """
        Parses the response from the backend to extract measurement counts.

        Parameters
        ----------
        response : dict
            The raw response from the backend.

        Returns
        -------
        dict
            A dictionary with the measurement counts.
        """
        parsed_response = {}
        for sample, count in response.items():
            x = [int(bit) for bit in reversed(sample)]
            parsed_response["".join(str(n) for n in x)] = count
        return parsed_response

    def select_backend(self, qubits: int) -> BackendV2:
        """
        Selects the appropriate backend based on the number of qubits and
        the specified backend name.

        Parameters
        ----------
        qubits : int
            The number of qubits in the quantum circuit.

        Returns
        -------
        Any
            The selected backend, which could be an IBMQ device or a  local
            simulator.
        """
        if self._backend != "simulator":
            if self._backend == "":
                return self.service.least_busy(min_num_qubits=qubits)
            return self.service.backend(self._backend)
        return aer.AerSimulator()
