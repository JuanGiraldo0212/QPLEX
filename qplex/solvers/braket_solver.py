from typing import Any
from qplex.solvers.base_solver import Solver
import braket.ir.openqasm
import braket.devices
import braket.aws


class BraketSolver(Solver):
    """
    A quantum solver for Braket that can execute quantum circuits on AWS
    Braket's devices or local simulators.

    Attributes
    ----------
    shots : int
        The number of shots for the quantum experiment.
    _backend : str
        The name of the backend to be used, which can be a Braket
        device or a local simulator.
    """

    def __init__(self, shots: int, backend: str, device_parameters):
        """
        Initializes the BraketSolver with the specified number of shots and
        backend.

        Parameters
        ----------
        shots : int
            The number of shots for the quantum experiment.
        backend : str
            The backend to use for solving the problem, which can
            be a Braket device or a local simulator.
        """
        self.shots = shots
        self._backend = backend
        self.device_parameters = device_parameters

    @property
    def backend(self):
        return self._backend

    def solve(self, model: str) -> dict:
        """
        Solves the given problem formulation using the specified backend.

        Parameters
        ----------
        model : str
            The quantum circuit as an OpenQASM string to be executed.

        Returns
        -------
        dict
            A dictionary containing the measurement counts from the backend.
        """
        qc = self.parse_input(model)
        backend = self.select_backend(0)
        response = (backend.run(qc, shots=self.shots,
                                device_parameters=self.device_parameters)
                    .result())
        counts = self.parse_response(response)
        return counts

    def parse_input(self, circuit: str) -> braket.ir.openqasm.Program:
        """
        Converts a circuit string to an OpenQASMProgram, replacing 'cx' with
        'cnot'.

        Parameters
        ----------
        circuit : str
            The quantum circuit as an OpenQASM string.

        Returns
        -------
        OpenQASMProgram
            An OpenQASMProgram instance with the modified circuit.
        """
        circuit = ("""
        OPENQASM 3.0;
        """ + circuit).replace("cx", "cnot")
        return braket.ir.openqasm.Program(source=circuit)

    def parse_response(self, response: Any) -> dict:
        """
        Parses the response from the backend to extract measurement counts.

        Parameters
        ----------
        response : Any
            The raw response from the backend.

        Returns
        -------
        dict
            A dictionary with the measurement counts.
        """
        return response.measurement_counts

    def select_backend(self, qubits: int) -> Any:
        """
        Selects the appropriate backend based on the number of qubits and
        the specified backend name.

        Parameters
        ----------
        qubits : int
            The minimum number of qubits required.

        Returns
        -------
        Any
            The selected backend, which could be an AWS device or a local
            simulator.
        """
        if self._backend != "simulator":
            return braket.aws.AwsDevice(f"arn:aws:braket:::{self._backend}")
        return braket.devices.LocalSimulator(backend="braket_sv")
