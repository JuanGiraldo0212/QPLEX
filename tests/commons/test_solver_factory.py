import pytest
from unittest.mock import Mock, patch

from qplex.commons.solver_factory import SolverFactory, ProviderType, \
    ProviderConfig


class TestSolverFactory:
    @pytest.fixture
    def mock_quantum_api_tokens(self):
        """Fixture for mocked quantum API tokens"""
        return {
            'd-wave_token': 'test_dwave_token',
            'ibmq_token': 'test_ibmq_token'
        }

    @patch('qplex.solvers.DWaveSolver')
    def test_get_solver_dwave(self, mock_dwave_solver_class,
                              mock_quantum_api_tokens):
        """
        Test that get_solver returns a DWaveSolver for the DWAVE provider type
        """
        mock_dwave_solver = Mock()
        mock_dwave_solver_class.return_value = mock_dwave_solver

        config = ProviderConfig(
            backend='hybrid_solver',
            shots=1000,
            provider_options={
                'time_limit': 100,
                'num_reads': 1000,
                'topology': 'pegasus',
                'embedding': None
            }
        )

        result = SolverFactory.get_solver(ProviderType.DWAVE,
                                          mock_quantum_api_tokens, config)

        mock_dwave_solver_class.assert_called_once_with(
            token='test_dwave_token',
            time_limit=100,
            num_reads=1000,
            topology='pegasus',
            embedding=None,
            backend='hybrid_solver'
        )

        assert result is mock_dwave_solver

    @patch('qplex.solvers.IBMQSolver')
    def test_get_solver_ibmq(self, mock_ibmq_solver_class,
                             mock_quantum_api_tokens):
        """
        Test that get_solver returns an IBMQSolver for the IBMQ provider type
        """
        # Configure mocks
        mock_ibmq_solver = Mock()
        mock_ibmq_solver_class.return_value = mock_ibmq_solver

        config = ProviderConfig(
            backend='ibmq_qasm_simulator',
            shots=1000,
            provider_options={
                'optimization_level': 1
            }
        )

        result = SolverFactory.get_solver(ProviderType.IBMQ,
                                          mock_quantum_api_tokens, config)

        mock_ibmq_solver_class.assert_called_once_with(
            token='test_ibmq_token',
            shots=1000,
            backend='ibmq_qasm_simulator',
            optimization_level=1
        )

        assert result is mock_ibmq_solver

    @patch('qplex.solvers.BraketSolver')
    def test_get_solver_braket(self, mock_braket_solver_class,
                               mock_quantum_api_tokens):
        """
        Test that get_solver returns a BraketSolver for the BRAKET provider
        type
        """
        mock_braket_solver = Mock()
        mock_braket_solver_class.return_value = mock_braket_solver

        config = ProviderConfig(
            backend='braket/sv',
            shots=1000,
            provider_options={
                'device_parameters': {'param1': 'value1'}
            }
        )

        result = SolverFactory.get_solver(ProviderType.BRAKET,
                                          mock_quantum_api_tokens, config)

        mock_braket_solver_class.assert_called_once_with(
            shots=1000,
            backend='braket/sv',
            device_parameters={'param1': 'value1'}
        )

        assert result is mock_braket_solver

    def test_get_solver_missing_credentials(self, mock_quantum_api_tokens):
        """
        Test that get_solver raises RuntimeError when credentials are missing
        """
        config = ProviderConfig(
            backend='test_backend',
            shots=1000
        )

        tokens_without_dwave = mock_quantum_api_tokens.copy()
        tokens_without_dwave['d-wave_token'] = None

        with pytest.raises(RuntimeError,
                           match="Missing credentials for d-wave"):
            SolverFactory.get_solver(ProviderType.DWAVE, tokens_without_dwave,
                                     config)

    def test_get_solver_unknown_provider(self, mock_quantum_api_tokens):
        """
        Test that get_solver raises ValueError for unknown provider types
        """
        config = ProviderConfig(
            backend='test_backend',
            shots=1000
        )

        unknown_provider = "unknown_provider"

        with pytest.raises(ValueError,
                           match="Unknown provider: unknown_provider"):
            SolverFactory.get_solver(unknown_provider, mock_quantum_api_tokens,
                                     config)
