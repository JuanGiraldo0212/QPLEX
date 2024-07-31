from qiskit.qasm3 import loads
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers import BackendV2
from qplex.solvers.base_solver import Solver
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


class IBMQSolver(Solver):
    """
    A quantum solver for IBMQ that can execute quantum circuits on IBM's
    backend or a local simulator.

    Attributes
    ----------
    shots : int
        The number of shots for the quantum experiment.
    backend : str
        The name of the backend to be used, which can be an IBMQ
        device or a local simulator.
    service : QiskitRuntimeService
        The Qiskit runtime service instance for interacting with IBMQ's
        backend.
    """

    def __init__(self, token: str, shots: int, backend: str):
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
        """
        self.shots = shots
        self.backend = backend
        QiskitRuntimeService.save_account(channel="ibm_quantum",
                                          token=token, overwrite=True)
        self.service = QiskitRuntimeService()

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
        pass_manager = generate_preset_pass_manager(backend=backend,
                                                    optimization_level=1)
        isa_circuit = pass_manager.run(qc)
        if self.backend == 'simulator':
            raw_counts = backend.run(isa_circuit).result().get_counts()
        else:
            sampler = Sampler(backend)
            pub = (isa_circuit,)
            result = sampler.run([pub], shots=self.shots).result()
            data = result[0].data
            bits = data.c
            raw_counts = bits.get_counts()
        counts = self.parse_response(raw_counts)
        return counts

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
        qc = loads(circuit)
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

    def select_backend(self, qubits: int) -> AerSimulator | BackendV2:
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
        if self.backend != "simulator":
            if self.backend is None or self.backend == "":
                print('No backend specified. Using least busy...')
                return self.service.least_busy(min_num_qubits=qubits)
            return self.service.backend(self.backend)
        return AerSimulator()
