import pytest
from unittest.mock import Mock

from qplex.utils.workflow_utils import get_solution_from_counts, \
    calculate_energy


class TestWorkflowUtils:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        model = Mock()

        var1 = Mock(name="var1")
        var2 = Mock(name="var2")
        var3 = Mock(name="var3")

        var1.name = "var1"
        var2.name = "var2"
        var3.name = "var3"

        model.iter_variables.return_value = [var1, var2, var3]

        return model

    def test_get_solution_from_counts_linear_only(self, mock_model):
        """
        Test get_solution_from_counts with only linear terms in the objective
        """
        mock_objective = Mock()

        linear_terms = [
            (Mock(name="var1"), 2),
            (Mock(name="var2"), 3),
            (Mock(name="var3"), -1)
        ]
        linear_terms[0][0].name = "var1"
        linear_terms[1][0].name = "var2"
        linear_terms[2][0].name = "var3"

        mock_objective.iter_terms.return_value = linear_terms
        mock_objective.iter_quad_triplets.return_value = []

        mock_model.get_objective_expr.return_value = mock_objective

        optimal_counts = {
            "101": 1000,
            "001": 100,
            "010": 50
        }

        result = get_solution_from_counts(mock_model, optimal_counts)

        # Expected solution: var1=1, var2=0, var3=1
        # Expected objective: 1*2 + 0*3 + 1*(-1) = 1

        assert "solution" in result
        assert "objective" in result
        assert result["solution"] == {"var1": 1, "var2": 0, "var3": 1}
        assert result["objective"] == 1

    def test_get_solution_from_counts_with_quadratic_terms(self, mock_model):
        """
        Test get_solution_from_counts with both linear and quadratic terms in
        the objective
        """
        mock_objective = Mock()

        linear_terms = [
            (Mock(name="var1"), 2),
            (Mock(name="var2"), 3)
        ]
        linear_terms[0][0].name = "var1"
        linear_terms[1][0].name = "var2"

        # Quadratic terms: var1 * var3 * 4 + var2 * var3 * (-2)
        var1 = Mock(name="var1")
        var2 = Mock(name="var2")
        var3 = Mock(name="var3")
        var1.name = "var1"
        var2.name = "var2"
        var3.name = "var3"

        quad_terms = [
            (var1, var3, 4),
            (var2, var3, -2)
        ]

        mock_objective.iter_terms.return_value = linear_terms
        mock_objective.iter_quad_triplets.return_value = quad_terms

        mock_model.get_objective_expr.return_value = mock_objective

        optimal_counts = {
            "101": 1000,
            "001": 100,
            "010": 50
        }

        result = get_solution_from_counts(mock_model, optimal_counts)

        # Expected solution: var1=1, var2=0, var3=1
        # Expected objective: 1*2 + 0*3 + 1*1*4 + 0*1*(-2) = 2 + 4 = 6

        assert "solution" in result
        assert "objective" in result
        assert result["solution"] == {"var1": 1, "var2": 0, "var3": 1}
        assert result["objective"] == 6

    def test_calculate_energy(self):
        """
        Test calculate_energy correctly computes the weighted average energy
        """
        mock_algorithm = Mock()
        mock_algorithm.iteration = 0
        mock_qubo = Mock()
        mock_objective = Mock()
        mock_qubo.objective = mock_objective
        mock_algorithm.qubo = mock_qubo

        def evaluate_side_effect(sample):
            if sample == [0, 0]:
                return 0.0
            elif sample == [0, 1]:
                return 1.0
            elif sample == [1, 0]:
                return 2.0
            elif sample == [1, 1]:
                return 3.0
            else:
                return 0.0

        mock_objective.evaluate = evaluate_side_effect

        counts = {
            "00": 200,  # Energy = 0.0
            "01": 300,  # Energy = 1.0
            "10": 400,  # Energy = 2.0
            "11": 100  # Energy = 3.0
        }

        # Total shots = 1000
        # Expected energy = (200*0 + 300*1 + 400*2 + 100*3) / 1000
        # = (0 + 300 + 800 + 300) / 1000 = 1.4

        result = calculate_energy(counts, 1000, mock_algorithm)

        assert result == 1.4

        assert mock_algorithm.iteration == 1
