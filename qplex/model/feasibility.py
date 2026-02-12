"""
feasibility.py

Provides constraint feasibility checking for quantum solutions.

After QUBO conversion and quantum solving, the original constraint
semantics are lost.  ``FeasibilityChecker`` evaluates the original
DOcplex constraints against a solution and produces a
``FeasibilityResult`` describing which constraints are satisfied,
which are violated, and by how much.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import docplex.mp.model as dpmodel

from qplex.utils.feasibility_utils import (
    get_constraint_id,
    get_constraint_type_label,
    evaluate_linear_expr,
    compute_violation,
)


@dataclass
class ConstraintViolation:
    """Detailed information about a single constraint violation.

    Attributes
    ----------
    constraint_id : str
        Human-readable constraint identifier.
    constraint_type : str
        One of ``"equality"``, ``"inequality_le"``,
        ``"inequality_ge"``.
    expected_bound : float
        The right-hand side value of the constraint.
    actual_value : float
        The evaluated left-hand side value from the solution.
    violation_magnitude : float
        Non-negative difference between actual and bound.
    expression : str or None
        String representation of the constraint, if available.
    """

    constraint_id: str
    constraint_type: str
    expected_bound: float
    actual_value: float
    violation_magnitude: float
    expression: Optional[str] = None


@dataclass
class FeasibilityResult:
    """Result of a constraint feasibility check.

    Attributes
    ----------
    is_feasible : bool or str
        ``True`` if all constraints satisfied, ``False`` if any
        violated, or ``"unknown"`` if checking failed.
    violated_constraints : list of str
        Identifiers of violated constraints.
    satisfied_constraints : list of str
        Identifiers of satisfied constraints.
    violations : list of ConstraintViolation
        Detailed violation information.
    error : str or None
        Error message when ``is_feasible == "unknown"``.
    """

    is_feasible: bool | str
    violated_constraints: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    violations: List[ConstraintViolation] = field(default_factory=list)
    error: Optional[str] = None


class FeasibilityChecker:
    """Evaluate DOcplex model constraints against a solution.

    Parameters
    ----------
    model : docplex.mp.model.Model
        The DOcplex model whose constraints will be checked.
    tolerance : float
        Absolute tolerance for floating-point comparisons.
    """

    def __init__(self, model: dpmodel.Model,
                 tolerance: float = 1e-6):
        self._model = model
        self._tolerance = tolerance

    def check(self, solution: Dict[str, Any]) -> FeasibilityResult:
        """Check if *solution* satisfies all model constraints.

        This method never raises exceptions.  If constraint checking
        fails for any reason the returned result has
        ``is_feasible="unknown"`` and the error description in
        ``error``.

        Parameters
        ----------
        solution : dict
            Variable assignments ``{var_name: value}``.

        Returns
        -------
        FeasibilityResult
            Feasibility status and violation details.
        """
        try:
            return self._check_constraints(solution)
        except Exception as exc:
            return FeasibilityResult(
                is_feasible="unknown",
                error=f"Constraint checking failed: {exc}",
            )

    def _check_constraints(
            self, solution: Dict[str, Any]) -> FeasibilityResult:
        """Internal constraint-checking loop."""
        violated: List[str] = []
        satisfied: List[str] = []
        violations: List[ConstraintViolation] = []

        for idx, ct in enumerate(self._model.iter_constraints()):
            cid = get_constraint_id(ct, idx)
            left = evaluate_linear_expr(ct.get_left_expr(), solution)
            bound = float(ct.get_right_expr())
            sense = ct.sense
            ok, magnitude = compute_violation(
                left, bound, sense, self._tolerance)

            if ok:
                satisfied.append(cid)
            else:
                violated.append(cid)
                violations.append(ConstraintViolation(
                    constraint_id=cid,
                    constraint_type=get_constraint_type_label(sense),
                    expected_bound=bound,
                    actual_value=left,
                    violation_magnitude=magnitude,
                    expression=str(ct),
                ))

        return FeasibilityResult(
            is_feasible=len(violated) == 0,
            violated_constraints=violated,
            satisfied_constraints=satisfied,
            violations=violations,
        )
