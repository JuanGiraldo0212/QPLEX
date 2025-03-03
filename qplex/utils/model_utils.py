from dataclasses import dataclass
from typing import Dict, Optional, List, Set

import docplex.mp.constr
import docplex.mp.constants
import docplex.mp.model

from qplex.model.constants import ConstraintType


@dataclass
class ConstraintInfo:
    """Stores information about detected constraints"""
    type: ConstraintType
    parameters: Optional[Dict] = None
    additional_constraints: Optional[List[ConstraintType]] = None


def get_model_constraint_info(model: docplex.mp.model.Model) -> ConstraintInfo:
    """
    Analyzes a DOcplex model to determine constraint types.

    Parameters
    ----------
    model : Model
        The DOcplex model to analyze

    Returns
    -------
    ConstraintInfo
        Information about detected constraints
    """
    constraints: List[docplex.mp.constr.LinearConstraint] = list(
        model.iter_constraints())
    detected_constraints: Set[ConstraintType] = set()
    parameters = {}

    # Helper functions for coefficient analysis
    def get_coefficients(const: docplex.mp.constr.LinearConstraint) -> Dict:
        return {var.name: coef for var, coef in
                const.get_left_expr().iter_terms()}

    def get_unique_coefs(const: docplex.mp.constr.LinearConstraint) -> Set:
        return set(coef for _, coef in const.get_left_expr().iter_terms())

    # Check for cardinality constraints (sum x_i = k)
    cardinality_constraints = [
        const for const in constraints
        if const.sense == docplex.mp.constants.ComparisonType.EQ and all(
            coef == 1 for coef
            in get_coefficients(const)
            .values()) and isinstance(
            const.get_right_expr(), (int, float))
    ]

    if cardinality_constraints:
        detected_constraints.add(ConstraintType.CARDINALITY)
        parameters["cardinality_k"] = cardinality_constraints[
            0].get_right_expr()

    # Check for partition constraints (sum x_i - sum y_i = 0)
    partition_constraints = [
        const for const in constraints
        if const.sense == docplex.mp.constants.ComparisonType.EQ
           and const.get_right_expr() == 0
           and get_unique_coefs(const) == {1, -1}
    ]

    if partition_constraints:
        detected_constraints.add(ConstraintType.PARTITION)

    # Check for inequality constraints
    inequality_constraints = [const for const in constraints
                              if (
                                          const.sense == docplex.mp.constants.ComparisonType.LE or
                                          const.sense == docplex.mp.constants.ComparisonType.GE)
                              and not all(coef == 1 for coef in
                                          get_coefficients(const).values())
                              ]

    if inequality_constraints:
        detected_constraints.add(ConstraintType.INEQUALITY)
        parameters["inequality_bounds"] = [
            (const.sense, const.get_right_expr())
            for const in inequality_constraints
        ]

    # Multiple constraints handling
    if len(detected_constraints) > 1:
        return ConstraintInfo(
            type=ConstraintType.MULTIPLE,
            parameters=parameters,
            additional_constraints=list(detected_constraints)
        )
    elif len(detected_constraints) == 1:
        constraint_type = detected_constraints.pop()
        return ConstraintInfo(
            type=constraint_type,
            parameters=parameters
        )

    return ConstraintInfo(type=ConstraintType.UNCONSTRAINED)
