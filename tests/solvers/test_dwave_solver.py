import pytest
from unittest.mock import patch, MagicMock
from qplex.solvers.dwave_solver import DWaveSolver
from qplex.model.constants import VAR_TYPE


class TestDWaveSolver:
    """Test suite for DWaveSolver class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.token = "fake_token"
        self.time_limit = 30
        self.num_reads = 100
        self.topology = "pegasus"
        self.embedding = None
        self.backend = "d-wave_sampler"

        self.solver = DWaveSolver(
            token=self.token,
            time_limit=self.time_limit,
            num_reads=self.num_reads,
            topology=self.topology,
            embedding=self.embedding,
            backend=self.backend
        )

    def test_initialization(self):
        """Test initialization of DWaveSolver with parameters."""
        assert self.solver.token == self.token
        assert self.solver.time_limit == self.time_limit
        assert self.solver.num_reads == self.num_reads
        assert self.solver.topology == self.topology
        assert self.solver.embedding == self.embedding
        assert self.solver._backend == self.backend
        assert self.solver.presolver is None
        assert self.solver.original_cqm is None

    def test_initialization_default_backend(self):
        """Test initialization with None backend defaults to hybrid_solver."""
        solver = DWaveSolver(
            token=self.token,
            time_limit=self.time_limit,
            num_reads=self.num_reads,
            topology=self.topology,
            embedding=self.embedding,
            backend=None
        )
        assert solver._backend == 'hybrid_solver'

    def test_backend_property(self):
        """Test the backend property returns the _backend value."""
        assert self.solver.backend == self.solver._backend

    @patch('dwave.system.LeapHybridCQMSampler')
    @patch('dwave.system.LeapHybridDQMSampler')
    @patch('dwave.system.LeapHybridBQMSampler')
    def test_select_backend_hybrid_solver(self, mock_bqm, mock_dqm, mock_cqm):
        """Test select_backend with hybrid_solver for different model types."""
        self.solver._backend = 'hybrid_solver'

        # Test with constrained model
        mock_cqm_instance = MagicMock()
        mock_cqm.return_value = mock_cqm_instance
        result = self.solver.select_backend(MagicMock(), VAR_TYPE['C'])
        assert result == mock_cqm_instance
        mock_cqm.assert_called_once_with(token=self.token)

        # Test with integer model
        mock_dqm_instance = MagicMock()
        mock_dqm.return_value = mock_dqm_instance
        result = self.solver.select_backend(MagicMock(), VAR_TYPE['I'])
        assert result == mock_dqm_instance
        mock_dqm.assert_called_once_with(token=self.token)

        # Test with binary model
        mock_bqm_instance = MagicMock()
        mock_bqm.return_value = mock_bqm_instance
        result = self.solver.select_backend(MagicMock(), VAR_TYPE['B'])
        assert result == mock_bqm_instance
        mock_bqm.assert_called_once_with(token=self.token)

    @patch('dwave.system.DWaveSampler')
    @patch('dwave.system.AutoEmbeddingComposite')
    def test_select_backend_dwave_sampler_binary(self, mock_auto_embed,
                                                 mock_dwave_sampler):
        """Test select_backend with d-wave_sampler for binary model."""
        self.solver._backend = 'd-wave_sampler'

        # Setup DWaveSampler mock
        mock_sampler = MagicMock()
        mock_sampler.solver.name = "mock_solver"
        mock_sampler.nodelist = [1, 2, 3, 4]
        mock_dwave_sampler.return_value = mock_sampler

        # Setup AutoEmbeddingComposite mock
        mock_embed_instance = MagicMock()
        mock_auto_embed.return_value = mock_embed_instance

        result = self.solver.select_backend(MagicMock(), VAR_TYPE['B'])

        # Verify the correct configuration
        mock_dwave_sampler.assert_called_once_with(
            solver=dict(topology__type=self.topology),
            token=self.token
        )
        mock_auto_embed.assert_called_once_with(mock_sampler)
        assert result == mock_embed_instance
        assert self.solver._backend == "mock_solver"

    @patch('dwave.system.DWaveSampler')
    @patch('dwave.system.FixedEmbeddingComposite')
    def test_select_backend_with_embedding(self, mock_fixed_embed,
                                           mock_dwave_sampler):
        """Test select_backend with custom embedding."""
        self.solver._backend = 'd-wave_sampler'
        self.solver.embedding = {"q0": [0, 1], "q1": [2, 3]}

        # Setup DWaveSampler mock
        mock_sampler = MagicMock()
        mock_sampler.solver.name = "mock_solver"
        mock_sampler.nodelist = [0, 1, 2, 3]
        mock_dwave_sampler.return_value = mock_sampler

        # Setup FixedEmbeddingComposite mock
        mock_embed_instance = MagicMock()
        mock_fixed_embed.return_value = mock_embed_instance

        result = self.solver.select_backend(MagicMock(), VAR_TYPE['B'])

        # Verify the correct configuration
        mock_dwave_sampler.assert_called_once()
        mock_fixed_embed.assert_called_once_with(
            mock_sampler,
            self.solver.embedding
        )
        assert result == mock_embed_instance

    @patch('dwave.system.LeapHybridCQMSampler')
    def test_select_backend_dwave_sampler_with_constrained_model(self,
                                                                 mock_cqm):
        """Test select_backend with d-wave_sampler for constrained model switches to hybrid."""
        self.solver._backend = 'd-wave_sampler'

        # Setup LeapHybridCQMSampler mock
        mock_cqm_instance = MagicMock()
        mock_cqm.return_value = mock_cqm_instance

        result = self.solver.select_backend(MagicMock(), VAR_TYPE['C'])

        # Should switch to hybrid solver
        mock_cqm.assert_called_once_with(token=self.token)
        assert result == mock_cqm_instance

    def test_select_backend_unsupported(self):
        """Test select_backend with unsupported backend raises ValueError."""
        self.solver._backend = 'unsupported_backend'

        with pytest.raises(ValueError) as excinfo:
            self.solver.select_backend(MagicMock(), VAR_TYPE['B'])

        assert "Unsupported backend" in str(excinfo.value)

    def test_parse_input_with_constrained_model(self):
        """Test parse_input method with a model containing constraints."""
        mock_model = MagicMock()

        mock_constraint = MagicMock()
        mock_constraint.sense.operator_symbol = ">="
        mock_constraint.right_expr.constant = 5
        mock_constraint.lpt_name = "test_constraint"

        mock_var = MagicMock()
        mock_var.name = "var1"
        mock_var.vartype.cplex_typecode = "B"
        mock_var.lb = 0
        mock_var.ub = 1

        mock_model.iter_constraints.return_value = [mock_constraint]
        mock_model.iter_variables.return_value = [mock_var]

        mock_constraint.iter_variables.return_value = [mock_var]
        mock_constraint.left_expr.iter_terms.return_value = [(mock_var, 1.0)]
        mock_constraint.left_expr.iter_quad_triplets.return_value = []

        mock_model.get_objective_expr.return_value.iter_terms.return_value = [
            (mock_var, 2.0)]
        mock_model.get_objective_expr.return_value.iter_quad_triplets \
            .return_value = []
        mock_model.objective_sense.name = "Minimize"

        with (patch('qplex.solvers.dwave_solver.QuadraticModel') as mock_qm, \
              patch(
                  'qplex.solvers.dwave_solver.ConstrainedQuadraticModel') as
              mock_cqm):
            mock_qm_instance = MagicMock()
            mock_qm.return_value = mock_qm_instance

            mock_cqm_instance = MagicMock()
            mock_cqm.return_value = mock_cqm_instance

            result, model_type = self.solver.parse_input(mock_model)

            assert result == mock_cqm_instance
            assert model_type == VAR_TYPE['C']

            mock_cqm.assert_called_once()
            mock_cqm_instance.set_objective.assert_called_once()
            mock_cqm_instance.add_constraint.assert_called_once_with(
                mock_qm_instance,
                sense=">=",
                rhs=5,
                label="test_constraint"
            )

    def test_parse_input_without_constraints_integer(self):
        """Test parse_input method with a model without constraints and integer variables."""
        mock_model = MagicMock()

        mock_var_int = MagicMock()
        mock_var_int.name = "var_int"
        mock_var_int.vartype.cplex_typecode = "I"
        mock_var_int.lb = 0
        mock_var_int.ub = 5

        mock_var_bin = MagicMock()
        mock_var_bin.name = "var_bin"
        mock_var_bin.vartype.cplex_typecode = "B"
        mock_var_bin.lb = 0
        mock_var_bin.ub = 1

        mock_model.iter_constraints.return_value = []
        mock_model.iter_variables.return_value = [mock_var_int, mock_var_bin]

        mock_model.get_objective_expr.return_value.iter_terms.return_value = [
            (mock_var_int, 3.0),
            (mock_var_bin, 2.0)
        ]
        mock_model.get_objective_expr.return_value.iter_quad_triplets\
            .return_value = [
            (mock_var_int, mock_var_bin, 1.5)
        ]
        mock_model.objective_sense.name = "Maximize"  # Test maximize objective

        with (patch(
                'qplex.solvers.dwave_solver.DiscreteQuadraticModel') as
        mock_dqm):
            mock_dqm_instance = MagicMock()
            mock_dqm.return_value = mock_dqm_instance

            result, model_type = self.solver.parse_input(mock_model)

            assert result == mock_dqm_instance
            assert model_type == VAR_TYPE['I']

            mock_dqm.assert_called_once()
            assert mock_dqm_instance.add_variable.call_count == 2
            assert mock_dqm_instance.set_linear.call_count == 2

            mock_dqm_instance.set_linear.assert_any_call("var_int", -3.0)
            mock_dqm_instance.set_linear.assert_any_call("var_bin", -2.0)
            mock_dqm_instance.set_quadratic.assert_called_once_with("var_int",
                                                                    "var_bin",
                                                                    -1.5)

    @patch.object(DWaveSolver, 'parse_input')
    @patch.object(DWaveSolver, 'select_backend')
    @patch.object(DWaveSolver, 'parse_response')
    def test_solve_with_hybrid_solver(self, mock_parse_response,
                                      mock_select_backend, mock_parse_input):
        """Test solve method with hybrid solver."""
        model = MagicMock(name="test_model")
        parsed_model = MagicMock()
        model_type = VAR_TYPE['C']
        mock_parse_input.return_value = (parsed_model, model_type)

        mock_sampler = MagicMock()
        mock_select_backend.return_value = mock_sampler

        mock_sampleset = MagicMock()
        mock_sampler.sample_cqm.return_value.filter.return_value = mock_sampleset

        mock_first = MagicMock()
        mock_sampleset.first = mock_first

        mock_result = {"objective": 1.0, "solution": {"var1": 1, "var2": 0}}
        mock_parse_response.return_value = mock_result

        self.solver._backend = 'hybrid_solver'
        result = self.solver.solve(model)

        mock_parse_input.assert_called_once_with(model)
        mock_select_backend.assert_called_once_with(parsed_model, model_type)
        mock_sampler.sample_cqm.assert_called_once_with(
            parsed_model,
            time_limit=self.time_limit,
            label=model.name
        )
        mock_sampler.sample_cqm.return_value.filter.assert_called_once()
        mock_parse_response.assert_called_once_with(mock_first)
        assert result == mock_result

    @patch.object(DWaveSolver, 'parse_input')
    @patch.object(DWaveSolver, 'select_backend')
    @patch.object(DWaveSolver, 'parse_response')
    def test_solve_with_qpu_binary_model(self, mock_parse_response,
                                         mock_select_backend,
                                         mock_parse_input):
        """Test solve method with QPU and binary model."""
        model = MagicMock(name="test_model")
        parsed_model = MagicMock()
        model_type = VAR_TYPE['B']
        mock_parse_input.return_value = (parsed_model, model_type)

        mock_sampler = MagicMock()
        mock_select_backend.return_value = mock_sampler

        mock_sampleset = MagicMock()
        mock_sampler.sample.return_value = mock_sampleset

        mock_first = MagicMock()
        mock_sampleset.first = mock_first

        mock_result = {"objective": 1.0, "solution": {"var1": 1, "var2": 0}}
        mock_parse_response.return_value = mock_result

        self.solver._backend = 'd-wave_sampler'
        result = self.solver.solve(model)

        mock_parse_input.assert_called_once_with(model)
        mock_select_backend.assert_called_once_with(parsed_model, model_type)
        mock_sampler.sample.assert_called_once_with(
            parsed_model,
            num_reads=self.num_reads,
            label=model.name
        )
        mock_parse_response.assert_called_once_with(mock_first)
        assert result == mock_result

    def test_parse_response(self):
        """Test parse_response method extracts energy and sample correctly."""
        response = MagicMock()
        response.energy = -2.5
        response.sample = {"var1": 1, "var2": 0, "var3": 1}

        result = self.solver.parse_response(response)

        assert result["objective"] == 2.5  # abs(-2.5)
        assert result["solution"] == {"var1": 1, "var2": 0, "var3": 1}
        assert isinstance(result["objective"], float)
