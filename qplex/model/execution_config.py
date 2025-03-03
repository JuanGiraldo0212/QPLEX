from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
import numpy as np

from qplex.commons.optimization_callback import OptimizationCallback
from qplex.model.constants import ALLOWED_OPTIMIZERS


@dataclass
class ExecutionConfig:
    """Configuration for quantum optimization execution."""

    # Execution method
    method: str = "classical"
    verbose: bool = False

    # Provider configuration
    provider: Optional[str] = None
    workflow: str = "default"
    backend: Optional[str] = None
    provider_options: Dict[str, Any] = field(default_factory=dict)

    # Algorithm configuration
    algorithm: str = "qaoa"
    ansatz: Optional[str] = None
    p: int = 2
    mixer: Optional[Any] = None
    layers: int = 2

    # Optimization configuration
    optimizer: str = "COBYLA"
    callback: Optional[Callable[[np.ndarray], None]] = None
    tolerance: float = 1e-10
    max_iter: int = 1000
    penalty: Optional[float] = None

    # Execution parameters
    shots: int = 1024
    seed: int = 1

    def __post_init__(self):
        self._validate_optimizer()

        # Set default callback if None
        if self.callback is None:
            self.callback = OptimizationCallback()

    def _validate_optimizer(self):
        """Validate optimizer configuration."""
        if not (isinstance(self.optimizer, str) and
                self.optimizer in ALLOWED_OPTIMIZERS) and not \
                callable(self.optimizer):
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. Must be one of "
                f"{ALLOWED_OPTIMIZERS} or a callable."
            )

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if
                not k.startswith('_')}
