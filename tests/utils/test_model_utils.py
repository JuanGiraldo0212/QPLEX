import pytest
from unittest.mock import Mock

import docplex.mp.model as docplex_model
import docplex.mp.constr as docplex_constr
import docplex.mp.constants as docplex_constants

from qplex.model.constants import ConstraintType
from qplex.utils.model_utils import get_model_constraint_info, ConstraintInfo


class TestModelUtils:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked DOcplex model"""
        model = Mock(spec=docplex_model.Model)
        return model

    def test_get_model_constraint_info_unconstrained(self, mock_model):
        """
        Test get_model_constraint_info returns UNCONSTRAINED when no
        constraints exist
        """
        mock_model.iter_constraints.return_value = []

        result = get_model_constraint_info(mock_model)

        assert isinstance(result, ConstraintInfo)
        assert result.type == ConstraintType.UNCONSTRAINED
        assert result.parameters is None
        assert result.additional_constraints is None

    def test_get_model_constraint_info_cardinality(self, mock_model):
        """
        Test get_model_constraint_info detects CARDINALITY constraints
        """
        mock_constraint = Mock(spec=docplex_constr.LinearConstraint)
        mock_constraint.sense = docplex_constants.ComparisonType.EQ

        mock_left_expr = Mock()
        mock_left_expr.iter_terms.return_value = [
            (Mock(name="x1"), 1.0),
            (Mock(name="x2"), 1.0),
            (Mock(name="x3"), 1.0)
        ]
        mock_constraint.get_left_expr.return_value = mock_left_expr

        mock_constraint.get_right_expr.return_value = 2

        mock_model.iter_constraints.return_value = [mock_constraint]

        result = get_model_constraint_info(mock_model)

        assert isinstance(result, ConstraintInfo)
        assert result.type == ConstraintType.CARDINALITY
        assert result.parameters is not None
        assert "cardinality_k" in result.parameters
        assert result.parameters["cardinality_k"] == 2

    def test_get_model_constraint_info_partition(self, mock_model):
        """
        Test get_model_constraint_info detects PARTITION constraints
        """
        mock_constraint = Mock(spec=docplex_constr.LinearConstraint)
        mock_constraint.sense = docplex_constants.ComparisonType.EQ

        mock_left_expr = Mock()
        mock_left_expr.iter_terms.return_value = [
            (Mock(name="x1"), 1.0),
            (Mock(name="x2"), 1.0),
            (Mock(name="x3"), -1.0),
            (Mock(name="x4"), -1.0)
        ]
        mock_constraint.get_left_expr.return_value = mock_left_expr

        mock_constraint.get_right_expr.return_value = 0

        mock_model.iter_constraints.return_value = [mock_constraint]

        result = get_model_constraint_info(mock_model)

        assert isinstance(result, ConstraintInfo)
        assert result.type == ConstraintType.PARTITION
        assert result.parameters is not None

    def test_get_model_constraint_info_inequality(self, mock_model):
        """
        Test get_model_constraint_info detects INEQUALITY constraints
        """
        mock_constraint = Mock(spec=docplex_constr.LinearConstraint)
        mock_constraint.sense = docplex_constants.ComparisonType.LE

        mock_left_expr = Mock()
        mock_left_expr.iter_terms.return_value = [
            (Mock(name="x1"), 2.0),
            (Mock(name="x2"), 3.0),
            (Mock(name="x3"), 1.0)
        ]
        mock_constraint.get_left_expr.return_value = mock_left_expr

        mock_constraint.get_right_expr.return_value = 5

        mock_model.iter_constraints.return_value = [mock_constraint]

        result = get_model_constraint_info(mock_model)

        assert isinstance(result, ConstraintInfo)
        assert result.type == ConstraintType.INEQUALITY
        assert result.parameters is not None
        assert "inequality_bounds" in result.parameters
        assert len(result.parameters["inequality_bounds"]) == 1
        assert result.parameters["inequality_bounds"][0][
                   0] == docplex_constants.ComparisonType.LE
        assert result.parameters["inequality_bounds"][0][1] == 5

    def test_get_model_constraint_info_multiple(self, mock_model):
        """
        Test get_model_constraint_info detects MULTIPLE constraints
        """
        mock_cardinality = Mock(spec=docplex_constr.LinearConstraint)
        mock_cardinality.sense = docplex_constants.ComparisonType.EQ

        mock_cardinality_left = Mock()
        mock_cardinality_left.iter_terms.return_value = [
            (Mock(name="x1"), 1.0),
            (Mock(name="x2"), 1.0),
            (Mock(name="x3"), 1.0)
        ]
        mock_cardinality.get_left_expr.return_value = mock_cardinality_left
        mock_cardinality.get_right_expr.return_value = 1

        mock_inequality = Mock(spec=docplex_constr.LinearConstraint)
        mock_inequality.sense = docplex_constants.ComparisonType.LE

        mock_inequality_left = Mock()
        mock_inequality_left.iter_terms.return_value = [
            (Mock(name="x1"), 2.0),
            (Mock(name="x2"), 3.0)
        ]
        mock_inequality.get_left_expr.return_value = mock_inequality_left
        mock_inequality.get_right_expr.return_value = 5

        mock_model.iter_constraints.return_value = [mock_cardinality,
                                                    mock_inequality]

        result = get_model_constraint_info(mock_model)

        assert isinstance(result, ConstraintInfo)
        assert result.type == ConstraintType.MULTIPLE
        assert result.parameters is not None
        assert "cardinality_k" in result.parameters
        assert "inequality_bounds" in result.parameters
        assert result.additional_constraints is not None
        assert len(result.additional_constraints) == 2
        assert ConstraintType.CARDINALITY in result.additional_constraints
        assert ConstraintType.INEQUALITY in result.additional_constraints
