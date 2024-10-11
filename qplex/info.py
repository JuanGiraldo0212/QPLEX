import importlib
import yaml
import logging
from pathlib import Path

from qplex import __version__


class Info:
    _config = None

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, "r") as f:
                cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def providers(cls):
        config = cls._load_config()
        return config.get("providers", [])

    @classmethod
    def backends(cls, provider):
        if provider not in cls.providers():
            raise ValueError(f"Unsupported provider: {provider}")

        try:
            provider_module = importlib.import_module(
                f"qplex.solvers.{provider}_solver")
            return provider_module.get_backends()
        except ImportError as e:
            logging.error(
                f"Failed to import module for provider {provider}: {e}")
            return []

    @classmethod
    def algorithms(cls):
        config = cls._load_config()
        return config.get("algorithms", [])

    @classmethod
    def version(cls):
        return __version__
