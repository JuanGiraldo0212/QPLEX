from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
import numpy as np


@dataclass
class OptimizationHistory:
    """Tracks optimization progress"""
    iterations: List[int] = field(default_factory=list)
    parameters: List[np.ndarray] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationCallback:
    """
    A class to track the optimization process and provide an optional
    customizable callback.

    This class allows tracking the current iteration and parameter values
    during the optimization
    process using the callback mechanism of `scipy.optimize.minimize`.
    Additionally, users can
    provide a custom callback function to extend or modify the behavior
    during each iteration.

    Attributes:
    -----------
    iteration : int
        Tracks the current iteration count of the optimization process.

    user_callback : Optional[Callable[[np.ndarray], None]]
        A user-defined callback function that accepts the current parameters
        `xk` and is executed
        at each iteration. This function allows for customized behavior
        during optimization.

    Methods:
    --------
    __call__(xk: np.ndarray):
        The method called at each iteration during the optimization process.
        It prints the current
        iteration number and parameters, and invokes the user-defined
        callback if provided.
    """

    def __init__(self,
                 user_callback: Optional[Callable[[np.ndarray], None]] =
                 None, verbose: bool = True):
        """
        Initialize the callback class.

        Parameters:
        -----------
        user_callback : Optional[Callable[[np.ndarray], None]]
            A user-defined function to be called at each iteration,
            which takes the current
            parameter vector `xk` as its input. If no function is provided,
            the default behavior
            will just print the iteration number and parameters.
        """
        self.iteration = 0
        self.user_callback = user_callback
        self.verbose = verbose
        self.history = OptimizationHistory()

    def __call__(self, xk: np.ndarray) -> None:
        """
        The method called at each iteration of the optimization process.

        Parameters:
        -----------
        xk : array_like
            The current parameter vector at this iteration of the
            optimization process.

        Behavior:
        ---------
        1. Prints the current iteration number and parameter values.
        2. If a user callback is provided, it is invoked with the current
        parameters `xk`.
        """
        self.iteration += 1

        if self.user_callback:
            self.user_callback(xk)
            return

        self.history.iterations.append(self.iteration)
        self.history.parameters.append(xk.copy())

        if self.verbose:
            self._log_progress(xk)

    def _log_progress(self, xk: np.ndarray) -> None:
        """Log optimization progress"""

        print(f"\nIteration {self.iteration}")
        print("-" * 50)
        print(f"Parameters: {xk}")
