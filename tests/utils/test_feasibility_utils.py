import pytest
from unittest.mock import Mock

import docplex.mp.constants as docplex_constants

from qplex.utils.feasibility_utils import (
    get_constraint_id,
    get_constraint_type_label,
    evaluate_linear_expr,
    compute_violation,
)


class TestGetConstraintId:
    """Tests for get_constraint_id."""

    def test_named_constraint(self):
        """Return the constraint's name when it exists."""
        ct = Mock()
        ct.name = "capacity_limit"
        assert get_constraint_id(ct, 0) == "capacity_limit"

    def test_unnamed_constraint(self):
        """Fall back to index-based name when no name is set."""
        ct = Mock()
        ct.name = None
        assert get_constraint_id(ct, 3) == "constraint_3"

    def test_empty_string_name(self):
        """Treat empty string as unnamed."""
        ct = Mock()
        ct.name = ""
        assert get_constraint_id(ct, 7) == "constraint_7"


class TestGetConstraintTypeLabel:
    """Tests for get_constraint_type_label."""

    def test_equality(self):
        assert get_constraint_type_label(
            docplex_constants.ComparisonType.EQ) == "equality"

    def test_less_equal(self):
        assert get_constraint_type_label(
            docplex_constants.ComparisonType.LE) == "inequality_le"

    def test_greater_equal(self):
        assert get_constraint_type_label(
            docplex_constants.ComparisonType.GE) == "inequality_ge"


class TestEvaluateLinearExpr:
    """Tests for evaluate_linear_expr."""

    def _make_expr(self, terms, constant=0.0):
        """Create a mock linear expression with given terms."""
        expr = Mock()
        expr.constant = constant
        mock_terms = []
        for name, coef in terms:
            var = Mock()
            var.name = name
            mock_terms.append((var, coef))
        expr.iter_terms.return_value = mock_terms
        return expr

    def test_simple_sum(self):
        """Evaluate x + y with x=3, y=4."""
        expr = self._make_expr([("x", 1.0), ("y", 1.0)])
        assert evaluate_linear_expr(expr, {"x": 3, "y": 4}) == 7.0

    def test_weighted_sum(self):
        """Evaluate 2x + 3y with x=1, y=2."""
        expr = self._make_expr([("x", 2.0), ("y", 3.0)])
        assert evaluate_linear_expr(expr, {"x": 1, "y": 2}) == 8.0

    def test_with_constant(self):
        """Evaluate 2x + 5 with x=3."""
        expr = self._make_expr([("x", 2.0)], constant=5.0)
        assert evaluate_linear_expr(expr, {"x": 3}) == 11.0

    def test_missing_variable_raises(self):
        """Raise KeyError when a variable is not in the solution."""
        expr = self._make_expr([("x", 1.0)])
        with pytest.raises(KeyError):
            evaluate_linear_expr(expr, {"y": 5})


class TestComputeViolation:
    """Tests for compute_violation."""

    def test_equality_satisfied(self):
        ok, mag = compute_violation(
            5.0, 5.0, docplex_constants.ComparisonType.EQ, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_equality_satisfied_within_tolerance(self):
        ok, mag = compute_violation(
            5.0000001, 5.0, docplex_constants.ComparisonType.EQ, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_equality_violated(self):
        ok, mag = compute_violation(
            7.0, 5.0, docplex_constants.ComparisonType.EQ, 1e-6)
        assert ok is False
        assert mag == pytest.approx(2.0)

    def test_le_satisfied(self):
        ok, mag = compute_violation(
            3.0, 5.0, docplex_constants.ComparisonType.LE, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_le_satisfied_at_boundary(self):
        ok, mag = compute_violation(
            5.0, 5.0, docplex_constants.ComparisonType.LE, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_le_violated(self):
        ok, mag = compute_violation(
            7.0, 5.0, docplex_constants.ComparisonType.LE, 1e-6)
        assert ok is False
        assert mag == pytest.approx(2.0)

    def test_ge_satisfied(self):
        ok, mag = compute_violation(
            7.0, 5.0, docplex_constants.ComparisonType.GE, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_ge_satisfied_at_boundary(self):
        ok, mag = compute_violation(
            5.0, 5.0, docplex_constants.ComparisonType.GE, 1e-6)
        assert ok is True
        assert mag == 0.0

    def test_ge_violated(self):
        ok, mag = compute_violation(
            3.0, 5.0, docplex_constants.ComparisonType.GE, 1e-6)
        assert ok is False
        assert mag == pytest.approx(2.0)

    def test_custom_tolerance(self):
        """A value that fails with tight tolerance passes with loose."""
        ok_tight, _ = compute_violation(
            5.01, 5.0, docplex_constants.ComparisonType.EQ, 1e-6)
        ok_loose, _ = compute_violation(
            5.01, 5.0, docplex_constants.ComparisonType.EQ, 0.1)
        assert ok_tight is False
        assert ok_loose is True
