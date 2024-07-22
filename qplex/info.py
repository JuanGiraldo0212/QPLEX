import importlib
import yaml
from pathlib import Path


class Info:

    @classmethod
    def _load_config(cls):
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def providers(cls):
        config = cls._load_config()
        return config.get("providers", [])

    @classmethod
    def backends(cls, provider):
        if provider not in cls.providers():
            raise ValueError(f"Unsupported provider for : {provider}")

        try:
            provider_module = importlib.import_module(
                f"qplex.solvers.{provider}_solver")
            return provider_module.get_backends()
        except ImportError:
            return []

    @classmethod
    def algorithms(cls):
        config = cls._load_config()
        return config.get("algorithms", [])

    @classmethod
    def version(cls):
        return "1.0.0"
