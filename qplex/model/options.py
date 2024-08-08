from collections.abc import MutableMapping
from qplex.model.constants import ALLOWED_OPTIMIZERS


class Options(MutableMapping):
    """
        A class to represent and manage optimization options for the
        QModel's solve method.

        Parameters
        ----------
        method: str, optional
            The method for solving the model, either 'classical' or
            'quantum'. Default is 'classical'.
        verbose: bool, optional
            If True, enables verbose output. Default is False.
        provider: str, optional
            The quantum provider to use. Default is None.
        backend: str, optional
            The backend to use for the quantum provider. Default is None.
        algorithm: str, optional
            The algorithm to use for solving the model. Default is "qaoa".
        ansatz: str, optional
            The ansatz to use in the quantum algorithm. Default is None.
        p: int, optional
            The depth of the ansatz for QAOA. Default is 2.
        layers: int, optional
            The number of layers in the ansatz. Default is 2.
        optimizer: str, optional
            The optimizer to use. Default is "COBYLA".
        tolerance: float, optional
            The tolerance for the optimizer. Default is 1e-10.
        max_iter: int, optional
            The maximum number of iterations for the optimizer. Default is
            1000.
        penalty: float, optional
            The penalty factor for the QUBO conversion. Default is None.
        shots: int, optional
            The number of shots for quantum execution. Default is 1024.
        seed: int, optional
            The seed for the random number generator. Default is 1.
        """
    def __init__(self,
                 method: str = 'classical',
                 verbose: bool = False,
                 provider: str = None,
                 backend: str = None,
                 algorithm: str = "qaoa",
                 ansatz: str = None,
                 p: int = 2,
                 layers: int = 2,
                 optimizer: str = "COBYLA",
                 tolerance: float = 1e-10,
                 max_iter: int = 1000,
                 penalty: float = None,
                 shots: int = 1024,
                 seed: int = 1):
        self._options = {
            'method': method,
            'verbose': verbose,
            'provider': provider,
            'backend': backend,
            'algorithm': algorithm,
            'ansatz': ansatz,
            'p': p,
            'layers': layers,
            'optimizer': optimizer,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'penalty': penalty,
            'shots': shots,
            'seed': seed,
        }
        self._validate_optimizer()

    def __getitem__(self, key):
        """
        Gets the value associated with the given key.

        Parameters
        ----------
        key: str
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
        key: str
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
        key: str
            The key for the item to delete.
        """
        del self._options[key]

    def __iter__(self):
        """
        Returns an iterator over the keys of the options.

        Returns
        -------
        iterator
            An iterator over the keys.
        """
        return iter(self._options)

    def __len__(self):
        """
        Returns the number of options.

        Returns
        -------
        int
            The number of options.
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
            A string representation of the options.
        """
        items = [f"{k}={v!r}" for k, v in self.to_dict().items()]
        return f"{type(self).__name__}({', '.join(items)})"
