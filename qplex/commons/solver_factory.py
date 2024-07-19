from qplex.solvers import IBMQSolver
from qplex.solvers import DWaveSolver
from qplex.solvers.braket_solver import BraketSolver


class SolverFactory:
    """
    A factory class for creating quantum solvers based on the specified
    provider.
    """

    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict, shots: int,
                   backend: str):
        """
        Returns the appropriate solver based on the specified provider and
        available tokens.

        Parameters
        ----------
        provider: str
            The quantum provider (e.g., 'd-wave', 'ibmq', 'braket').
        quantum_api_tokens: dict
            A dictionary containing API tokens for various quantum providers.
        shots: int
            The number of shots for quantum execution.
        backend: str
            The backend to use for the quantum provider.

        Returns
        -------
        solver
            An instance of the appropriate solver based on the specified
            provider.

        Raises
        ------
        RuntimeError
            If the necessary credentials for the specified provider are
            missing.
        ValueError
            If the specified provider is not recognized.
        """
        dwave_token = quantum_api_tokens.get("d-wave_token")
        ibmq_token = quantum_api_tokens.get("ibmq_token")
        if provider is None:
            if dwave_token:
                return DWaveSolver()
            if ibmq_token:
                return IBMQSolver(token=ibmq_token, shots=shots,
                                  backend=backend)
            raise RuntimeError("Missing credentials for quantum provider")
        if provider == 'd-wave':
            if dwave_token is None:
                raise RuntimeError(
                    "Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if provider == 'ibmq':
            if ibmq_token is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return IBMQSolver(token=ibmq_token, shots=shots, backend=backend)
        if provider == "braket":
            return BraketSolver(shots=shots, backend=backend)
        raise ValueError(provider)


solver_factory = SolverFactory()
