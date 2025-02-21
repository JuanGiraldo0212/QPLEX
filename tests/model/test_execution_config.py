import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from qplex.model.execution_config import ExecutionConfig
from qplex.commons.optimization_callback import OptimizationCallback
from qplex.model.constants import ALLOWED_OPTIMIZERS


class TestExecutionConfig:
    """Test suite for ExecutionConfig class."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        config = ExecutionConfig()

        assert config.method == "classical"
        assert config.verbose is False
        assert config.provider is None
        assert config.workflow == "default"
        assert config.backend is None
        assert config.provider_options == {}
        assert config.algorithm == "qaoa"
        assert config.ansatz is None
        assert config.p == 2
        assert config.mixer is None
        assert config.layers == 2
        assert config.optimizer == "COBYLA"
        assert isinstance(config.callback, OptimizationCallback)
        assert config.tolerance == 1e-10
        assert config.max_iter == 1000
        assert config.penalty is None
        assert config.shots == 1024
        assert config.seed == 1

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_callback = MagicMock()
        custom_mixer = MagicMock()

        config = ExecutionConfig(
            method="quantum",
            verbose=True,
            provider="qiskit",
            workflow="custom",
            backend="simulator",
            provider_options={"token": "abc123"},
            algorithm="vqe",
            ansatz="hardware_efficient",
            p=3,
            mixer=custom_mixer,
            layers=4,
            optimizer="SLSQP",
            callback=custom_callback,
            tolerance=1e-8,
            max_iter=500,
            penalty=0.5,
            shots=2048,
            seed=42
        )

        assert config.method == "quantum"
        assert config.verbose is True
        assert config.provider == "qiskit"
        assert config.workflow == "custom"
        assert config.backend == "simulator"
        assert config.provider_options == {"token": "abc123"}
        assert config.algorithm == "vqe"
        assert config.ansatz == "hardware_efficient"
        assert config.p == 3
        assert config.mixer is custom_mixer
        assert config.layers == 4
        assert config.optimizer == "SLSQP"
        assert config.callback is custom_callback
        assert config.tolerance == 1e-8
        assert config.max_iter == 500
        assert config.penalty == 0.5
        assert config.shots == 2048
        assert config.seed == 42

    def test_validate_optimizer_with_valid_string(self):
        """Test validating optimizer with valid string values."""
        for optimizer in ALLOWED_OPTIMIZERS:
            config = ExecutionConfig(optimizer=optimizer)
            assert config.optimizer == optimizer

    def test_validate_optimizer_with_callable(self):
        """Test validating optimizer with a callable."""

        def custom_optimizer():
            pass

        config = ExecutionConfig(optimizer=custom_optimizer)
        assert config.optimizer is custom_optimizer

    def test_validate_optimizer_with_invalid_string(self):
        """Test validating optimizer with invalid string values."""
        with pytest.raises(ValueError) as excinfo:
            ExecutionConfig(optimizer="INVALID_OPTIMIZER")

        assert "Invalid optimizer" in str(excinfo.value)
        assert "INVALID_OPTIMIZER" in str(excinfo.value)

    def test_validate_optimizer_with_invalid_type(self):
        """Test validating optimizer with invalid types."""
        invalid_values = [123, [1, 2, 3], {"key": "value"}, None]

        for invalid_value in invalid_values:
            with pytest.raises(ValueError) as excinfo:
                ExecutionConfig(optimizer=invalid_value)

            assert "Invalid optimizer" in str(excinfo.value)

    def test_default_callback_creation(self):
        """Test that default callback is created when None is provided."""
        config = ExecutionConfig(callback=None)
        assert isinstance(config.callback, OptimizationCallback)
