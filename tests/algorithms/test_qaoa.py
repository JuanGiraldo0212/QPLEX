import pytest
from unittest.mock import Mock, patch
import numpy as np

from qplex.algorithms.qaoa import QAOA
from qplex.algorithms.mixers.quantum_mixer import QuantumMixer


class TestQAOA:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        model = Mock()
        return model

    @pytest.fixture
    def mock_mixer(self):
        """Fixture for a mocked quantum mixer"""
        mixer = Mock(spec=QuantumMixer)

        mixer.generate_circuit.return_value = [
            "rx(theta) q[0];",
            "rx(theta) q[1];"
        ]

        return mixer

    @pytest.fixture
    def mock_qubo(self):
        """Fixture for a mocked QUBO"""
        qubo = Mock()

        objective = Mock()
        objective.linear = Mock()
        objective.linear.to_array.return_value = np.array([0.5, -1.0])

        objective.quadratic = Mock()
        objective.quadratic.to_array.return_value = np.array(
            [[0.0, 2.0], [2.0, 0.0]])

        qubo.objective = objective
        qubo.get_num_binary_vars.return_value = 2

        return qubo

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_init_no_mixer(self, _, mock_model, mock_qubo):
        """
        Test that QAOA initialization raises ValueError when no mixer is
        provided
        """
        mock_model.get_qubo.return_value = mock_qubo

        with pytest.raises(ValueError,
                           match="Expected mixer to be provided, got None"):
            QAOA(mock_model, p=2, seed=42, penalty=1.0, mixer=None)

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_create_circuit(self, _, mock_model, mock_mixer, mock_qubo):
        """
        Test that create_circuit generates the correct circuit
        """
        mock_model.get_qubo.return_value = mock_qubo

        qaoa = QAOA(mock_model, p=1, seed=42, penalty=1.0, mixer=mock_mixer)

        circuit = qaoa.create_circuit(penalty=1.0)

        expected_circuit = """input float[64] theta0;
    input float[64] theta1;
    qreg q[2];
    creg c[2];
    h q[0];
    h q[1];
    rz(theta0 * 2.5) q[0];
    rz(theta0 * 1.0) q[1];
    cx q[0], q[1];
    rz(theta0 * 1.0) q[1];
    cx q[0], q[1];
    rx(theta) q[0];
    rx(theta) q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];"""

        expected_normalized = '\n'.join(
            line.strip() for line in expected_circuit.split('\n') if
            line.strip())
        actual_normalized = '\n'.join(
            line.strip() for line in circuit.split('\n') if line.strip())

        assert actual_normalized == expected_normalized

        mock_mixer.generate_circuit.assert_called_with(2, "theta1")

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_update_params(self, mock_replace_params, mock_model, mock_mixer,
                           mock_qubo):
        """Test that update_params correctly calls replace_params"""
        mock_model.get_qubo.return_value = mock_qubo

        mock_replace_params.return_value = "Updated circuit"

        qaoa = QAOA(mock_model, p=1, seed=42, penalty=1.0, mixer=mock_mixer)
        qaoa.circuit = "Original circuit"

        params = np.array([0.1, 0.2])

        result = qaoa.update_params(params)

        mock_replace_params.assert_called_once_with("Original circuit", params)

        assert result == "Updated circuit"

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_update_params_invalid_count(self, _, mock_model,
                                         mock_mixer, mock_qubo):
        """Test that update_params raises ValueError when parameter count is invalid"""
        mock_model.get_qubo.return_value = mock_qubo

        qaoa = QAOA(mock_model, p=1, seed=42, penalty=1.0, mixer=mock_mixer)

        params = np.array([0.1])

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Expected 2 parameters, got 1"):
            qaoa.update_params(params)

    @patch('qplex.utils.circuit_utils.replace_params')
    def test_get_starting_point(self, _, mock_model,
                                mock_mixer, mock_qubo):
        """Test that get_starting_point returns correct array"""
        mock_model.get_qubo.return_value = mock_qubo

        with patch('numpy.random.rand', return_value=np.array([0.1, 0.2])):
            qaoa = QAOA(mock_model, p=1, seed=42, penalty=1.0,
                        mixer=mock_mixer)

            result = qaoa.get_starting_point()

            assert isinstance(result, np.ndarray)
            assert result.size == 2  # 2*p
            assert np.array_equal(result, np.array([0.1, 0.2]))
