from collections.abc import MutableMapping
from qplex.model.constants import ALLOWED_OPTIMIZERS


class Options(MutableMapping):
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
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value

    def __delitem__(self, key):
        del self._options[key]

    def __iter__(self):
        return iter(self._options)

    def __len__(self):
        return len(self._options)

    def to_dict(self):
        return self._options

    def _validate_optimizer(self):
        if not (isinstance(self._options['optimizer'], str) and
                self._options['optimizer'] in ALLOWED_OPTIMIZERS) and not \
                callable(self._options['optimizer']):
            raise ValueError(
                f"Invalid optimizer: {self._options['optimizer']}. Must be "
                f"one of {ALLOWED_OPTIMIZERS} or a callable.")

    def __repr__(self):
        items = [f"{k}={v!r}" for k, v in self.to_dict().items()]
        return f"{type(self).__name__}({', '.join(items)})"
