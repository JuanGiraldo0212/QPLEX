import pytest
import numpy as np

from qplex.utils.circuit_utils import replace_params


class TestCircuitUtils:
    def test_replace_params_simple_circuit(self):
        """Test replacing parameters in a simple circuit with few parameters"""
        circuit = """
        ry(theta0) q[0];
        rz(theta1) q[1];
        rx(theta2) q[2];
        """

        params = np.array([0.5, 1.2, 2.7])

        expected_circuit = """
        ry(0.5) q[0];
        rz(1.2) q[1];
        rx(2.7) q[2];
        """

        result = replace_params(circuit, params)

        assert result == expected_circuit

    def test_replace_params_complex_circuit(self):
        """Test replacing parameters in a more complex circuit with many parameters"""
        circuit = """
        ry(theta0) q[0];
        cx q[0], q[1];
        rz(theta1) q[1];
        ry(theta2 + theta0) q[2];
        cx q[2], q[3];
        rx(theta3 * 2.0) q[3];
        rz(theta1 + theta4) q[1];
        """

        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        expected_circuit = """
        ry(0.1) q[0];
        cx q[0], q[1];
        rz(0.2) q[1];
        ry(0.3 + 0.1) q[2];
        cx q[2], q[3];
        rx(0.4 * 2.0) q[3];
        rz(0.2 + 0.5) q[1];
        """

        result = replace_params(circuit, params)

        assert result == expected_circuit

    def test_replace_params_with_repeated_parameters(self):
        """Test replacing parameters when parameters are used multiple times"""
        circuit = """
        ry(theta0) q[0];
        rz(theta0) q[1];
        rx(theta0) q[2];
        """

        params = np.array([0.5])

        expected_circuit = """
        ry(0.5) q[0];
        rz(0.5) q[1];
        rx(0.5) q[2];
        """

        result = replace_params(circuit, params)

        assert result == expected_circuit

    def test_replace_params_with_missing_parameters(self):
        """Test that an IndexError is raised when parameters are missing"""
        circuit = """
        ry(theta0) q[0];
        rz(theta1) q[1];
        rx(theta2) q[2];
        """

        params = np.array([0.5, 1.2])

        with pytest.raises(IndexError):
            replace_params(circuit, params)

    def test_replace_params_with_no_parameters(self):
        """Test that the circuit is returned unchanged when there are no parameters to replace"""
        circuit = """
        ry(0.5) q[0];
        rz(1.2) q[1];
        rx(2.7) q[2];
        """

        params = np.array([0.1, 0.2, 0.3])

        result = replace_params(circuit, params)

        assert result == circuit

    def test_replace_params_with_empty_circuit(self):
        """Test that an empty circuit is returned unchanged"""
        circuit = ""

        params = np.array([0.1, 0.2, 0.3])

        result = replace_params(circuit, params)

        assert result == circuit
