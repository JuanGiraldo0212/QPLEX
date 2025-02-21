import typing
import dataclasses
import enum

import qplex.solvers
import qplex.solvers.base_solver


class ProviderType(enum.Enum):
    DWAVE = "d-wave"
    IBMQ = "ibmq"
    BRAKET = "braket"


@dataclasses.dataclass
class ProviderConfig:
    backend: str
    shots: typing.Optional[int]
    provider_options: typing.Optional[typing.Dict[str, typing.Any]] = None


@dataclasses.dataclass
class DWaveConfig:
    backend: str
    time_limit: typing.Optional[int] = None
    num_reads: int = 100
    topology: str = "pegasus"
    embedding: typing.Optional[typing.Any] = None


@dataclasses.dataclass
class IMBQConfig:
    backend: str
    shots: typing.Optional[int]
    optimization_level: int = 1


@dataclasses.dataclass
class BraketConfig:
    backend: str
    shots: typing.Optional[int]
    device_parameters: typing.Dict[str, typing.Any] = None


class SolverFactory:
    """
    A factory class for creating quantum solvers based on the specified
    provider.
    """

    _TOKEN_MAP = {
        ProviderType.DWAVE: "d-wave_token",
        ProviderType.IBMQ: "ibmq_token",
        ProviderType.BRAKET: None
    }

    @classmethod
    def get_solver(cls, provider: ProviderType, quantum_api_tokens: dict,
                   config: ProviderConfig) -> qplex.solvers.base_solver.Solver:
        """
        Return a solver based on the specified provider and available tokens.

        Parameters
        ----------
        provider: ProviderType
            The quantum provider (e.g., 'd-wave', 'ibmq', 'braket').
        quantum_api_tokens: dict
            A dictionary containing API tokens for various quantum providers.
        config: ProviderConfig
            A ProviderConfig instance containing the configuration for the
            provider.

        Returns
        -------
        solver: Solver
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
        if not isinstance(provider, ProviderType):
            raise ValueError(f"Unknown provider: {provider}")

        token_key = cls._TOKEN_MAP.get(provider)
        token = quantum_api_tokens.get(token_key) if token_key else None

        if token_key and token is None:
            raise RuntimeError(f"Missing credentials for {provider.value}")

        if provider == ProviderType.DWAVE:
            d_wave_config = DWaveConfig(
                backend=config.backend,
                **config.provider_options
            )
            return qplex.solvers.DWaveSolver(token=token,
                                             time_limit=d_wave_config.time_limit,
                                             num_reads=d_wave_config.num_reads,
                                             topology=d_wave_config.topology,
                                             embedding=d_wave_config.embedding,
                                             backend=d_wave_config.backend
                                             )

        elif provider == ProviderType.IBMQ:
            ibmq_config = IMBQConfig(
                backend=config.backend,
                shots=config.shots,
                **config.provider_options
            )
            return qplex.solvers.IBMQSolver(
                token=token,
                shots=ibmq_config.shots,
                backend=ibmq_config.backend,
                optimization_level=ibmq_config.optimization_level
            )

        elif provider == ProviderType.BRAKET:
            braket_config = BraketConfig(
                backend=config.backend,
                shots=config.shots,
                **config.provider_options
            )
            return qplex.solvers.BraketSolver(
                shots=braket_config.shots,
                backend=braket_config.backend,
                device_parameters=braket_config.device_parameters or {}
            )

        raise ValueError(f"Unsupported provider: {provider}")
