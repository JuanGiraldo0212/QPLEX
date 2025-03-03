import pytest
from unittest.mock import MagicMock
import textwrap
import numpy as np
from qplex.algorithms import Algorithm


class ConcreteAlgorithm(Algorithm):
    """Concrete implementation of Algorithm for testing purposes."""

    def create_circuit(self) -> str:
        return "dummy circuit"

    def update_params(self, params: np.ndarray) -> str:
        return "updated circuit"

    def get_starting_point(self) -> np.ndarray:
        return np.array([0.0, 0.0])


class TestBaseAlgorithm:
    """Test suite for the remove_parameters method in Algorithm class."""

    def setup_method(self):
        """Set up a concrete instance of Algorithm for testing."""
        # Mock the model parameter that Algorithm requires
        mock_model = MagicMock()
        self.algorithm = ConcreteAlgorithm(mock_model)

    def test_remove_parameters_with_input_lines(self):
        """Test that parameter input lines are correctly removed."""
        test_circuit = textwrap.dedent("""
        OPENQASM 3.0;
        input float[64] theta1;
        input float[64] theta2;

        qubit[3] q;

        rx(theta1) q[0];
        ry(theta2) q[1];
        cx q[0], q[2];
        """).strip()

        expected_circuit = textwrap.dedent("""
        OPENQASM 3.0;

        qubit[3] q;

        rx(theta1) q[0];
        ry(theta2) q[1];
        cx q[0], q[2];
        """).strip()

        self.algorithm.circuit = test_circuit
        self.algorithm.remove_parameters()

        assert self.algorithm.circuit == expected_circuit

    def test_remove_parameters_without_input_lines(self):
        """Test that circuit without parameter input lines remains unchanged."""
        test_circuit = textwrap.dedent("""
        OPENQASM 3.0;

        qubit[3] q;

        rx(0.5) q[0];
        ry(0.3) q[1];
        cx q[0], q[2];
        """).strip()

        self.algorithm.circuit = test_circuit
        self.algorithm.remove_parameters()

        assert self.algorithm.circuit == test_circuit

    def test_remove_parameters_with_mixed_input_lines(self):
        """Test with a mix of parameter inputs and other inputs."""
        test_circuit = textwrap.dedent("""
        OPENQASM 3.0;
        input float[64] theta1;
        input int[32] shots;  // This should stay
        input float[64] theta2;

        qubit[3] q;

        rx(theta1) q[0];
        ry(theta2) q[1];
        cx q[0], q[2];
        """).strip()

        expected_circuit = textwrap.dedent("""
        OPENQASM 3.0;
        input int[32] shots;  // This should stay

        qubit[3] q;

        rx(theta1) q[0];
        ry(theta2) q[1];
        cx q[0], q[2];
        """).strip()

        self.algorithm.circuit = test_circuit
        self.algorithm.remove_parameters()

        assert self.algorithm.circuit == expected_circuit

    def test_remove_parameters_with_empty_circuit(self):
        """Test with an empty circuit string."""
        self.algorithm.circuit = ""
        self.algorithm.remove_parameters()

        assert self.algorithm.circuit == ""

    def test_remove_parameters_with_none_circuit(self):
        """Test that AttributeError is raised when circuit is None."""
        self.algorithm.circuit = None

        with pytest.raises(AttributeError) as excinfo:
            self.algorithm.remove_parameters()

        assert "not defined" in str(excinfo.value)
