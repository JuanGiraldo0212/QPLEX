from collections.abc import MutableMapping
from qplex.model.constants import ALLOWED_OPTIMIZERS
from qplex.commons.optimization_callback import OptimizationCallback

import numpy as np
from typing import Callable, Optional


class Options(MutableMapping):
    """
    A class to represent and manage optimization options for the QModel's
    solve method.

    This class acts as a flexible container for various options related to
    solving an optimization problem using either classical or quantum
    methods. It inherits from MutableMapping to allow dictionary-like
    behavior for setting and retrieving options.

    Parameters
    ----------
    method : str, optional
        The method for solving the model, either 'classical' or 'quantum'.
        Default is 'classical'.
    verbose : bool, optional
        If True, enables verbose output during the solve process. Default is
        False.
    provider : str, optional
        The quantum provider to use for quantum execution (e.g., 'ibmq').
        Default is None.
    workflow : str, optional
        The execution workflow to use (e.g., 'default', 'ibm_session').
        Default is 'default'.
    backend : str, optional
        The backend to use for the quantum provider (e.g.,
        'ibmq_qasm_simulator'). Default is None.
    algorithm : str, optional
        The algorithm to use for solving the model (e.g., 'qaoa', 'vqe').
        Default is "qaoa".
    ansatz : str, optional
        The ansatz (quantum circuit structure) to use in the quantum
        algorithm. Default is None.
    p : int, optional
        The depth of the ansatz for QAOA. Higher values result in more
        layers of gates. Default is 2.
    layers : int, optional
        The number of layers in the ansatz for VQE or similar algorithms.
        Default is 2.
    optimizer : str, optional
        The classical optimizer to use for optimizing the parameters of the
        quantum circuit.
        Default is "COBYLA". It must be a valid optimizer name or a callable.
    callback : Callable, optional
        A callback function to be called at each iteration of the
        optimization. Default is an instance of `OptimizationCallback`.
    tolerance : float, optional
        The tolerance for the optimizer to determine convergence. Default is
        1e-10.
    max_iter : int, optional
        The maximum number of iterations allowed for the optimizer. Default
        is 1000.
    penalty : float, optional
        The penalty factor for the QUBO conversion. This parameter is used
        when formulating the problem as a QUBO. Default is None.
    shots : int, optional
        The number of shots (measurements) to execute on the quantum backend
        for each evaluation. Default is 1024.
    seed : int, optional
        The seed for the random number generator to ensure reproducibility.
        Default is 1.
    provider_options : dict, optional
        Additional options specific to the quantum provider, such as
        credentials or backend-specific configurations. Default is an empty
        dictionary.
    """

    def __init__(self,
                 method: str = 'classical',
                 verbose: bool = False,
                 provider: str = None,
                 workflow: str = 'default',
                 backend: str = None,
                 algorithm: str = "qaoa",
                 ansatz: str = None,
                 p: int = 2,
                 layers: int = 2,
                 optimizer: str = "COBYLA",
                 callback: Optional[
                     Callable[[np.ndarray], None]] = OptimizationCallback(),
                 tolerance: float = 1e-10,
                 max_iter: int = 1000,
                 penalty: float = None,
                 shots: int = 1024,
                 seed: int = 1,
                 provider_options=None):
        if provider_options is None:
            provider_options = {}
        self._options = {
            'method': method,
            'verbose': verbose,
            'provider': provider,
            'workflow': workflow,
            'backend': backend,
            'algorithm': algorithm,
            'ansatz': ansatz,
            'p': p,
            'layers': layers,
            'optimizer': optimizer,
            'callback': callback,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'penalty': penalty,
            'shots': shots,
            'seed': seed,
            'provider_options': provider_options
        }

        self._validate_optimizer()

    def __getitem__(self, key):
        """
        Gets the value associated with the given key.

        Parameters
        ----------
        key : str
            The key for the desired value.

        Returns
        -------
        The value associated with the key.
        """
        return self._options[key]

    def __setitem__(self, key, value):
        """
        Sets the value for the given key.

        Parameters
        ----------
        key : str
            The key for the value to set.
        value
            The value to set for the given key.
        """
        self._options[key] = value

    def __delitem__(self, key):
        """
        Deletes the item associated with the given key.

        Parameters
        ----------
        key : str
            The key for the item to delete.
        """
        del self._options[key]

    def __iter__(self):
        """
        Returns an iterator over the keys of the options.

        Returns
        -------
        iterator
            An iterator over the keys of the options.
        """
        return iter(self._options)

    def __len__(self):
        """
        Returns the number of options.

        Returns
        -------
        int
            The number of options stored in the object.
        """
        return len(self._options)

    def to_dict(self):
        """
        Converts the options to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the options.
        """
        return self._options

    def _validate_optimizer(self):
        """
        Validates the optimizer option.

        Ensures that the optimizer is either a valid string from the list of
        allowed optimizers or a callable function. Raises an error if the
        provided optimizer is not valid.

        Raises
        ------
        ValueError
            If the optimizer is not a valid string or callable.
        """
        if not (isinstance(self._options['optimizer'], str) and
                self._options['optimizer'] in ALLOWED_OPTIMIZERS) and not \
                callable(self._options['optimizer']):
            raise ValueError(
                f"Invalid optimizer: {self._options['optimizer']}. Must be "
                f"one of {ALLOWED_OPTIMIZERS} or a callable.")

    def __repr__(self):
        """
        Returns a string representation of the options.

        Returns
        -------
        str
            A string representation of the options, showing all key-value
            pairs.
        """
        items = [f"{k}={v!r}" for k, v in self.to_dict().items()]
        return f"{type(self).__name__}({', '.join(items)})"
