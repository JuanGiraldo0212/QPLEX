import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from qplex.workflows.ibm_session_workflow import run_ibm_session_workflow, \
    compute_counts
from qplex.commons.algorithm_factory import AlgorithmConfig, AlgorithmType
from qplex.model.execution_config import ExecutionConfig


class TestIBMSessionWorkflow:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        return Mock()

    @pytest.fixture
    def mock_ibmq_solver(self):
        """Fixture for a mocked IBMQ solver"""
        solver = Mock()
        solver.service = Mock()
        solver.shots = 1000
        solver.optimization_level = 1

        solver.parse_input.return_value = Mock()
        solver.select_backend.return_value = Mock()
        solver.parse_response.return_value = {"00": 500, "11": 500}
        solver.run.return_value = {"raw_counts": {"00": 500, "11": 500}}

        return solver

    @pytest.fixture
    def mock_options(self):
        """Fixture for mocked execution options"""
        options = Mock(spec=ExecutionConfig)
        options.shots = 1000
        options.verbose = False
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
    @patch(
        'qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager')
    @patch('qplex.workflows.ibm_session_workflow.compute_counts')
    @patch('qplex.utils.workflow_utils.calculate_energy')
    @patch('qiskit_ibm_runtime.SamplerV2')
    @patch('qiskit_ibm_runtime.Session')
    def test_ibm_session_workflow(self, mock_session_class, mock_sampler_class,
                                  mock_calculate_energy,
                                  mock_compute_counts, mock_generate_pm,
                                  mock_minimize,
                                  mock_get_algorithm, mock_model,
                                  mock_ibmq_solver, mock_options):
        """
        Test that ibm_session_workflow correctly orchestrates the optimization
        process
        """
        # Setup mocks
        mock_algorithm = Mock()
        mock_algorithm.circuit = "mocked_openqasm_circuit"
        mock_algorithm.get_starting_point.return_value = np.array([0.0, 0.0])
        mock_get_algorithm.return_value = mock_algorithm

        mock_circuit = Mock()
        mock_circuit.num_qubits = 2
        mock_ibmq_solver.parse_input.return_value = mock_circuit

        mock_pm = Mock()
        mock_isa_circuit = Mock()
        mock_pm.run.return_value = mock_isa_circuit
        mock_generate_pm.return_value = mock_pm

        mock_session = Mock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__.return_value = mock_session
        mock_session_class.return_value = mock_session_context

        mock_sampler = Mock()
        mock_sampler_class.return_value = mock_sampler

        mock_result = Mock()
        mock_result.x = np.array([0.5, 0.7])
        mock_minimize.return_value = mock_result

        mock_compute_counts.side_effect = [
            {"00": 200, "11": 800}
        ]

        mock_calculate_energy.return_value = 0.5

        result = run_ibm_session_workflow(mock_model, mock_ibmq_solver,
                                          mock_options)

        mock_get_algorithm.assert_called_once()
        algorithm_config = mock_get_algorithm.call_args[0][1]
        assert isinstance(algorithm_config, AlgorithmConfig)
        assert algorithm_config.algorithm == AlgorithmType('qaoa')

        mock_ibmq_solver.parse_input.assert_called_once_with(
            mock_algorithm.circuit)

        mock_ibmq_solver.select_backend.assert_called_once_with(
            mock_circuit.num_qubits)

        mock_generate_pm.assert_called_once_with(
            backend=mock_ibmq_solver.select_backend.return_value,
            optimization_level=mock_ibmq_solver.optimization_level
        )

        mock_pm.run.assert_called_once_with(mock_circuit)

        mock_session_class.assert_called_once_with(
            service=mock_ibmq_solver.service,
            backend=mock_ibmq_solver.select_backend.return_value
        )

        mock_sampler_class.assert_called_once_with(mode=mock_session)

        mock_minimize.assert_called_once()
        assert mock_minimize.call_args[1][
                   'x0'] is mock_algorithm.get_starting_point.return_value
        assert mock_minimize.call_args[1]['method'] == mock_options.optimizer
        assert mock_minimize.call_args[1]['tol'] == mock_options.tolerance

        mock_compute_counts.assert_called_with(
            mock_result.x, mock_ibmq_solver, mock_isa_circuit, mock_sampler
        )

        assert result == {"00": 200, "11": 800}

    def test_compute_counts(self, mock_ibmq_solver):
        """Test that compute_counts correctly processes parameters and calls the solver"""
        mock_circuit = Mock()
        mock_sampler = Mock()
        params = np.array([0.1, 0.2])

        mock_bound_circuit = Mock()
        mock_circuit.assign_parameters.return_value = mock_bound_circuit

        mock_raw_counts = {"raw_counts": {"00": 400, "11": 600}}
        mock_ibmq_solver.run.return_value = mock_raw_counts

        result = compute_counts(params, mock_ibmq_solver, mock_circuit,
                                mock_sampler)

        mock_circuit.assign_parameters.assert_called_once()
        params_dict = mock_circuit.assign_parameters.call_args[0][0]
        assert params_dict == {"theta0": 0.1, "theta1": 0.2}

        mock_ibmq_solver.run.assert_called_once_with(mock_bound_circuit,
                                                     mock_sampler)
        mock_ibmq_solver.parse_response.assert_called_once_with(
            mock_raw_counts)
        assert result == {"00": 500, "11": 500}
