import pytest
from unittest.mock import Mock, patch
import os

from qiskit_optimization import QuadraticProgram

from qplex.model.qmodel import QModel, ModelSolution
from qplex.model.execution_config import ExecutionConfig


class TestQModel:
    @pytest.fixture
    def mock_qmodel(self):
        """Fixture for a mocked QModel instance"""
        with patch.dict(os.environ, {
            'D-WAVE_API_TOKEN': 'test_dwave_token',
            'IBMQ_API_TOKEN': 'test_ibmq_token'
        }):
            model = QModel("test_model")

            model._create_solution = Mock()
            model._set_solution = Mock()

            return model

    @patch('qiskit_optimization.translators.from_docplex_mp')
    @patch('qiskit_optimization.converters.QuadraticProgramToQubo')
    def test_get_qubo(self, mock_qubo_converter_class, mock_from_docplex,
                      mock_qmodel):
        """Test that get_qubo correctly converts the model to a QUBO"""
        mock_docplex_model = Mock()
        mock_from_docplex.return_value = mock_docplex_model

        mock_qubo_converter = Mock()
        mock_qubo_result = Mock(spec=QuadraticProgram)
        mock_qubo_converter.convert.return_value = mock_qubo_result
        mock_qubo_converter_class.return_value = mock_qubo_converter

        result = mock_qmodel.get_qubo(penalty=2.0)

        mock_from_docplex.assert_called_once_with(mock_qmodel)

        mock_qubo_converter_class.assert_called_once_with(penalty=2.0)

        mock_qubo_converter.convert.assert_called_once_with(mock_docplex_model)

        assert result is mock_qubo_result

    @patch('docplex.mp.model.Model.solve')
    def test_solve_classical(self, mock_parent_solve, mock_qmodel):
        """
        Test that solve with 'classical' method calls the parent class solve
        method
        """
        mock_qmodel.solve(method='classical')

        mock_parent_solve.assert_called_once()

        mock_qmodel._create_solution.assert_called_once()
        call_kwargs = mock_qmodel._create_solution.call_args[1]
        assert call_kwargs['method'] == 'classical'

        mock_qmodel._set_solution.assert_called_once_with(
            mock_qmodel._create_solution.return_value
        )

    @patch('qplex.commons.solver_factory.SolverFactory.get_solver')
    @patch('qplex.commons.solver_factory.ProviderConfig')
    @patch('qplex.commons.solver_factory.ProviderType')
    def test_solve_quantum_dwave(self, mock_provider_type,
                                 mock_provider_config,
                                 mock_get_solver, mock_qmodel):
        """Test that solve with 'quantum' method and d-wave provider works correctly"""
        mock_config = Mock(spec=ExecutionConfig)
        mock_config.provider = 'd-wave'
        mock_config.shots = 1000
        mock_config.backend = 'test_backend'
        mock_config.provider_options = {'option1': 'value1'}

        mock_solver = Mock()
        mock_solver.solve.return_value = {'objective': 42.0,
                                          'solution': {'var1': 1}}
        mock_get_solver.return_value = mock_solver

        mock_qmodel.solve(method='quantum', config=mock_config)

        mock_get_solver.assert_called_once()
        assert mock_get_solver.call_args[1][
                   'provider'] == mock_provider_type.return_value
        assert mock_get_solver.call_args[1][
                   'quantum_api_tokens'] == mock_qmodel.quantum_api_tokens
        assert mock_get_solver.call_args[1][
                   'config'] == mock_provider_config.return_value

        mock_solver.solve.assert_called_once_with(mock_qmodel)

        mock_qmodel._create_solution.assert_called_once()
        call_kwargs = mock_qmodel._create_solution.call_args[1]
        assert call_kwargs['method'] == 'quantum'
        assert call_kwargs['provider'] == 'd-wave'
        assert call_kwargs['backend'] == 'test_backend'
        assert call_kwargs['result'] == {'objective': 42.0,
                                         'solution': {'var1': 1}}

        mock_qmodel._set_solution.assert_called_once_with(
            mock_qmodel._create_solution.return_value
        )

    @patch('qplex.commons.solver_factory.SolverFactory.get_solver')
    @patch('qplex.workflows.run_ibm_session_workflow')
    @patch('qplex.utils.workflow_utils.get_solution_from_counts')
    def test_solve_quantum_ibm_session(self, mock_get_solution,
                                       mock_session_workflow,
                                       mock_get_solver, mock_qmodel):
        """
        Test that solve with 'quantum' method and ibmq provider with
        ibm_session workflow works correctly
        """
        mock_config = Mock(spec=ExecutionConfig)
        mock_config.provider = 'ibmq'
        mock_config.workflow = 'ibm_session'
        mock_config.algorithm = 'qaoa'
        mock_config.shots = 1000
        mock_config.backend = 'test_backend'
        mock_config.provider_options = {}

        mock_solver = Mock()
        mock_get_solver.return_value = mock_solver

        mock_session_workflow.return_value = {"00": 400, "11": 600}
        mock_get_solution.return_value = {'objective': 42.0,
                                          'solution': {'var1': 1}}

        mock_qmodel.solve(method='quantum', config=mock_config)

        mock_session_workflow.assert_called_once_with(
            model=mock_qmodel,
            ibmq_solver=mock_solver,
            options=mock_config
        )

        mock_get_solution.assert_called_once_with(
            model=mock_qmodel,
            optimal_counts={"00": 400, "11": 600}
        )

        mock_qmodel._create_solution.assert_called_once()
        call_kwargs = mock_qmodel._create_solution.call_args[1]
        assert call_kwargs['method'] == 'quantum'
        assert call_kwargs['provider'] == 'ibmq'
        assert call_kwargs['backend'] == 'test_backend'
        assert call_kwargs['result'] == {'objective': 42.0,
                                         'solution': {'var1': 1}}

    @patch('qplex.commons.solver_factory.SolverFactory.get_solver')
    @patch('qplex.workflows.ggae_workflow')
    @patch('qplex.utils.workflow_utils.get_solution_from_counts')
    def test_solve_quantum_generic_workflow(self, mock_get_solution,
                                            mock_ggae_workflow,
                                            mock_get_solver, mock_qmodel):
        """Test that solve with 'quantum' method and a provider that uses the general workflow works correctly"""
        mock_config = Mock(spec=ExecutionConfig)
        mock_config.provider = 'braket'
        mock_config.workflow = 'generic'
        mock_config.shots = 1000
        mock_config.backend = 'test_backend'
        mock_config.provider_options = {}

        mock_solver = Mock()
        mock_get_solver.return_value = mock_solver

        mock_ggae_workflow.return_value = {"00": 300, "11": 700}
        mock_get_solution.return_value = {'objective': 42.0,
                                          'solution': {'var1': 1}}

        mock_qmodel.solve(method='quantum', config=mock_config)

        mock_ggae_workflow.assert_called_once_with(
            model=mock_qmodel,
            solver=mock_solver,
            options=mock_config
        )

        mock_get_solution.assert_called_once_with(
            model=mock_qmodel,
            optimal_counts={"00": 300, "11": 700}
        )

        mock_qmodel._create_solution.assert_called_once()
        call_kwargs = mock_qmodel._create_solution.call_args[1]
        assert call_kwargs['method'] == 'quantum'
        assert call_kwargs['provider'] == 'braket'
        assert call_kwargs['backend'] == 'test_backend'
        assert call_kwargs['result'] == {'objective': 42.0,
                                         'solution': {'var1': 1}}

    def test_solve_invalid_method(self, mock_qmodel):
        """Test that solve with an invalid method raises a ValueError"""
        with pytest.raises(ValueError,
                           match="Invalid value for argument 'method'. Must be 'classical' or 'quantum'"):
            mock_qmodel.solve(method='invalid')

    def test_create_solution_quantum(self, mock_qmodel):
        """Test _create_solution for quantum solution"""
        mock_qmodel._create_solution = QModel._create_solution.__get__(
            mock_qmodel)

        result = mock_qmodel._create_solution(
            execution_time=2.34,
            method='quantum',
            provider="ibmq",
            backend="test_backend",
            algorithm="qaoa",
            result={"objective": 42.0, "solution": {"var1": 1, "var2": 0}}
        )

        assert isinstance(result, ModelSolution)
        assert result.solution == {"var1": 1, "var2": 0}
        assert result.objective == 42.0
        assert result.execution_time == 2.34
        assert result.method == 'quantum'
        assert result.provider == "ibmq"
        assert result.backend == "test_backend"
        assert result.algorithm == "qaoa"

    def test_set_solution(self, mock_qmodel):
        """Test _set_solution correctly sets the solution on the model"""
        mock_qmodel._set_solution = QModel._set_solution.__get__(mock_qmodel)

        solution = ModelSolution(
            solution={"var1": 1, "var2": 0},
            objective=42.0,
            execution_time=1.23,
            method='quantum',
            provider="ibmq",
            backend="test_backend",
            algorithm="qaoa"
        )

        mock_solve_solution = Mock()

        with patch(
                'docplex.mp.model.Model._set_solution') as mock_parent_set_solution:
            with patch('docplex.mp.solution.SolveSolution',
                       return_value=mock_solve_solution):
                mock_qmodel._set_solution(solution)

                assert mock_qmodel._qmodel_solution is solution

                assert mock_parent_set_solution.call_args[1][
                           'new_solution'] is mock_solve_solution
