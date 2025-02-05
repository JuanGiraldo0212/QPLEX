from typing import Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from qplex.solvers import IBMQSolver, DWaveSolver, BraketSolver
from qplex.solvers.base_solver import Solver


class ProviderType(Enum):
    DWAVE = "d-wave"
    IBMQ = "ibmq"
    BRAKET = "braket"


@dataclass
class ProviderConfig:
    backend: str
    shots: Optional[int]
    provider_options: Optional[Dict[str, Any]] = None


@dataclass
class DWaveConfig(ProviderConfig):
    time_limit: Optional[int] = None
    num_reads: int = 100
    topology: str = "pegasus"
    embedding: Optional[Any] = None


@dataclass
class IMBQConfig(ProviderConfig):
    optimization_level: int = 1


@dataclass
class BraketConfig(ProviderConfig):
    device_parameters: Dict[str, Any] = None


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
                   config: ProviderConfig) -> Solver:
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
            return DWaveSolver(token=token,
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
            return IBMQSolver(
                token=token,
                shots=ibmq_config.shots,
                backend=ibmq_config.backend,
                optimization_level=ibmq_config.optimization_level
            )

        elif provider == 'braket':
            braket_config = BraketConfig(
                backend=config.backend,
                shots=config.shots,
                **config.provider_options
            )
            return BraketSolver(
                shots=braket_config.shots,
                backend=braket_config.backend,
                device_parameters=braket_config.device_parameters or {}
            )

        raise ValueError(f"Unsupported provider: {provider}")
