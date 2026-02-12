"""
feasibility_utils.py

Utility functions for evaluating DOcplex constraint satisfaction
against solution variable values.
"""
from typing import Dict, Any, Tuple

import docplex.mp.constr as constr
import docplex.mp.constants as constants


def get_constraint_id(constraint: constr.AbstractConstraint,
                      index: int) -> str:
    """
    Return a human-readable identifier for a constraint.

    Uses the constraint's name if set, otherwise falls back to
    an index-based name.

    Parameters
    ----------
    constraint : AbstractConstraint
        The DOcplex constraint.
    index : int
        Positional index of the constraint in the model.

    Returns
    -------
    str
        The constraint identifier.
    """
    name = getattr(constraint, 'name', None)
    return name if name else f"constraint_{index}"


def get_constraint_type_label(
        sense: constants.ComparisonType) -> str:
    """
    Map a DOcplex comparison sense to a readable label.

    Parameters
    ----------
    sense : ComparisonType
        The DOcplex comparison type (EQ, LE, GE).

    Returns
    -------
    str
        One of ``"equality"``, ``"inequality_le"``,
        ``"inequality_ge"``.
    """
    mapping = {
        constants.ComparisonType.EQ: "equality",
        constants.ComparisonType.LE: "inequality_le",
        constants.ComparisonType.GE: "inequality_ge",
    }
    return mapping.get(sense, "equality")


def evaluate_linear_expr(expr, solution: Dict[str, Any]) -> float:
    """
    Evaluate a DOcplex linear expression given variable assignments.

    Parameters
    ----------
    expr : LinearExpr
        A DOcplex linear expression (left-hand side of a constraint).
    solution : dict
        Variable assignments ``{var_name: value}``.

    Returns
    -------
    float
        The numeric value of the expression.

    Raises
    ------
    KeyError
        If a variable in the expression is missing from the solution.
    """
    total = float(expr.constant)
    for var, coef in expr.iter_terms():
        total += coef * solution[var.name]
    return total


def compute_violation(actual: float, bound: float,
                      sense: constants.ComparisonType,
                      tolerance: float) -> Tuple[bool, float]:
    """
    Determine whether a constraint is satisfied and compute its violation.

    Parameters
    ----------
    actual : float
        The evaluated left-hand side value.
    bound : float
        The right-hand side bound.
    sense : ComparisonType
        The comparison operator (EQ, LE, GE).
    tolerance : float
        Absolute tolerance for floating-point comparisons.

    Returns
    -------
    tuple[bool, float]
        ``(is_satisfied, violation_magnitude)`` where
        ``violation_magnitude`` is 0.0 when satisfied.
    """
    if sense == constants.ComparisonType.EQ:
        diff = abs(actual - bound)
        return (diff <= tolerance, diff if diff > tolerance else 0.0)
    elif sense == constants.ComparisonType.LE:
        excess = actual - bound
        return (excess <= tolerance, max(0.0, excess) if excess > tolerance else 0.0)
    else:  # GE
        deficit = bound - actual
        return (deficit <= tolerance, max(0.0, deficit) if deficit > tolerance else 0.0)
