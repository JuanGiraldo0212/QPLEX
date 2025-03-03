import pytest
from unittest.mock import Mock, patch
import numpy as np

from qplex.algorithms.vqe import VQE


class TestVQE:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        model = Mock()
        return model

    @pytest.fixture
    def mock_qubo(self):
        """Fixture for a mocked QUBO"""
        qubo = Mock()
        qubo.get_num_binary_vars.return_value = 3
        return qubo

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_init(self, _, mock_model, mock_qubo):
        """Test that VQE is initialized correctly"""
        mock_model.get_qubo.return_value = mock_qubo

        # Initialize VQE
        vqe = VQE(mock_model, layers=2, seed=42, penalty=1.0,
                  ansatz="test_ansatz")

        assert vqe.layers == 2
        assert vqe.n == 3
        assert vqe.num_params == 3 + (
                4 * (3 - 1) * 2)  # n + (4 * (n - 1) * layers)

        mock_model.get_qubo.assert_called_once_with(penalty=1.0)

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_create_circuit(self, _, mock_model, mock_qubo):
        """Test that create_circuit generates the correct circuit"""
        mock_model.get_qubo.return_value = mock_qubo

        vqe = VQE(mock_model, layers=1, seed=42, penalty=1.0,
                  ansatz="test_ansatz")

        circuit = vqe.create_circuit(penalty=1.0)

        for i in range(11):
            assert f"input float[64] theta{i};" in circuit

        assert "qreg q[3];" in circuit
        assert "creg c[3];" in circuit

        for i in range(3):
            assert f"ry(param{i}) q[{i}];" in circuit

        for i in range(2):
            assert f"cx q[{i}], q[{i + 1}];" in circuit
            assert f"ry(param{3 + 4 * i}) q[{i}];" in circuit
            assert f"ry(param{3 + 4 * i + 1}) q[{i + 1}];" in circuit
            assert f"cx q[{i}], q[{i + 1}];" in circuit
            assert f"ry(param{3 + 4 * i + 2}) q[{i}];" in circuit
            assert f"ry(param{3 + 4 * i + 3}) q[{i + 1}];" in circuit

        for i in range(3):
            assert f"measure q[{i}] -> c[{i}];" in circuit

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_update_params(self, mock_replace_params, mock_model, mock_qubo):
        """Test that update_params correctly calls replace_params"""
        mock_model.get_qubo.return_value = mock_qubo

        mock_replace_params.return_value = "Updated circuit"

        vqe = VQE(mock_model, layers=1, seed=42, penalty=1.0,
                  ansatz="test_ansatz")
        vqe.circuit = "Original circuit"

        params = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])

        result = vqe.update_params(params)

        mock_replace_params.assert_called_once_with("Original circuit", params)

        assert result == "Updated circuit"

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_get_starting_point(self, _, mock_model,
                                mock_qubo):
        """Test that get_starting_point returns correct array"""
        mock_model.get_qubo.return_value = mock_qubo

        mock_random_values = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])

        with patch('numpy.random.rand', return_value=mock_random_values):
            vqe = VQE(mock_model, layers=1, seed=42, penalty=1.0,
                      ansatz="test_ansatz")

            result = vqe.get_starting_point()

            assert isinstance(result, np.ndarray)
            assert result.size == 11
            assert np.array_equal(result, mock_random_values)
