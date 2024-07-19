from typing import Any
from braket.aws import AwsDevice
from qplex.solvers.base_solver import Solver
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.devices import LocalSimulator


class BraketSolver(Solver):
    """
    A quantum solver for Braket that can execute quantum circuits on AWS
    Braket's devices or local simulators.

    Attributes
    ----------
    shots : int
        The number of shots for the quantum experiment.
    backend : str
        The name of the backend to be used, which can be a Braket
        device or a local simulator.
    """

    def __init__(self, shots: int, backend: str):
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
        self.backend = backend

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
        response = backend.run(qc, shots=self.shots).result()
        counts = self.parse_response(response)
        return counts

    def parse_input(self, circuit: str) -> OpenQASMProgram:
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
        return OpenQASMProgram(source=circuit)

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
        if self.backend != "simulator":
            return AwsDevice(f"arn:aws:braket:::{self.backend}")
        return LocalSimulator(backend="braket_sv")
