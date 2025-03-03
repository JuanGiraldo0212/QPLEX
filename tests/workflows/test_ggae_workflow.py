import pytest
from unittest.mock import Mock, patch
import numpy as np

from qplex.commons.algorithm_factory import AlgorithmConfig, AlgorithmType
from qplex.model.execution_config import ExecutionConfig
from qplex.workflows.ggae_workflow import ggae_workflow


class TestGGAEWorkflow:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        return Mock()

    @pytest.fixture
    def mock_solver(self):
        """Fixture for a mocked quantum solver"""
        solver = Mock()
        solver.solve.return_value = {"00": 500, "11": 500}
        return solver

    @pytest.fixture
    def mock_options(self):
        """Fixture for mocked execution options"""
        options = Mock(spec=ExecutionConfig)
        options.shots = 1000
        options.verbose = True
        options.optimizer = 'COBYLA'
        options.callback = None
        options.max_iter = 100
        options.tolerance = 1e-3
        options.algorithm = 'qaoa'
        options.penalty = 1.0
        options.seed = 42
        options.p = 1
        options.mixer = None
        options.layers = None
        options.ansatz = None
        return options

    @patch('qplex.commons.algorithm_factory.AlgorithmFactory.get_algorithm')
    @patch('scipy.optimize.minimize')
    def test_ggae_workflow(self, mock_minimize, mock_get_algorithm, mock_model,
                           mock_solver, mock_options):
        """Test that ggae_workflow correctly orchestrates the optimization process"""
        mock_algorithm = Mock()
        mock_algorithm.get_starting_point.return_value = np.array([0.0, 0.0])
        mock_algorithm.update_params.return_value = "mocked_circuit"
        mock_get_algorithm.return_value = mock_algorithm

        mock_result = Mock()
        mock_result.x = np.array([0.5, 0.7])
        mock_minimize.return_value = mock_result

        def calc_energy_side_effect(counts, shots, alg):
            return 0.5

        with patch('qplex.utils.workflow_utils.calculate_energy',
                   side_effect=calc_energy_side_effect):
            result = ggae_workflow(mock_model, mock_solver, mock_options)

            mock_get_algorithm.assert_called_once()
            algorithm_config = mock_get_algorithm.call_args[0][1]
            assert isinstance(algorithm_config, AlgorithmConfig)
            assert algorithm_config.algorithm == AlgorithmType('qaoa')
            assert algorithm_config.penalty == mock_options.penalty
            assert algorithm_config.seed == mock_options.seed
            assert algorithm_config.p == mock_options.p

            mock_algorithm.remove_parameters.assert_called_once()

            mock_algorithm.get_starting_point.assert_called_once()

            mock_minimize.assert_called_once()
            assert mock_minimize.call_args[1][
                       'x0'] is mock_algorithm.get_starting_point.return_value
            assert mock_minimize.call_args[1][
                       'method'] == mock_options.optimizer
            assert mock_minimize.call_args[1]['tol'] == mock_options.tolerance
            assert mock_minimize.call_args[1]['options'] == {
                'maxiter': mock_options.max_iter}

            mock_algorithm.update_params.assert_called_with(mock_result.x)

            mock_solver.solve.assert_called_with("mocked_circuit")

            assert result == {"00": 500, "11": 500}
