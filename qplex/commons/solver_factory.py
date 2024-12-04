from qplex.solvers import IBMQSolver, DWaveSolver, BraketSolver

from typing import Any


class SolverFactory:
    """
    A factory class for creating quantum solvers based on the specified
    provider.
    """

    PROVIDERS = {
        'd-wave': 'd-wave_token',
        'ibmq': 'ibmq_token',
        'braket': None
    }

    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict, shots: int,
                   backend: str, provider_options: dict[str, Any]):
        """
        Return a solver based on the specified provider and available tokens.

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
        provider_options: dict[str, Any]
            A dictionary containing the configuration for the provider.

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
        if provider not in SolverFactory.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        token_name = SolverFactory.PROVIDERS.get(provider)

        if token_name:
            token = quantum_api_tokens.get(token_name)
            if token is None:
                raise RuntimeError(
                    f"Missing credentials for the {provider} provider")
        else:
            token = None

        if provider == 'd-wave':
            return DWaveSolver(token=token, backend=backend,
                               time_limit=provider_options.get('time_limit',
                                                               None),
                               num_reads=provider_options.get('num_reads',
                                                              100))

        elif provider == 'ibmq':
            return IBMQSolver(token=token, shots=shots, backend=backend,
                              optimization_level=provider_options.get(
                                  'optimization_level', 1))

        elif provider == 'braket':
            return BraketSolver(shots=shots, backend=backend,
                                device_parameters=provider_options.get(
                                    'device_parameters', {}))

        raise ValueError(f"Unsupported provider: {provider}")


solver_factory = SolverFactory()
