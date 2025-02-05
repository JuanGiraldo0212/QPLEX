"""
constants.py

This module defines various constants used throughout QPLEX.
"""
from enum import Enum

# Variable types used for modeling
VAR_TYPE = {
    'B': 'BINARY',  # Binary variables
    'I': 'INTEGER',  # Integer variables
    'C': 'REAL'  # Real (continuous) variables
}

# Allowed optimizers for optimization routines
ALLOWED_OPTIMIZERS = {
    'Nelder-Mead',  # Nelder-Mead algorithm
    'Powell',  # Powell's method
    'CG',  # Conjugate Gradient
    'BFGS',  # Broyden-Fletcher-Goldfarb-Shanno algorithm
    'Newton-CG',  # Newton Conjugate Gradient
    'L-BFGS-B',
    # Limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints
    'TNC',  # Truncated Newton Conjugate Gradient
    'COBYLA',  # Constrained Optimization BY Linear Approximations
    'SLSQP',  # Sequential Least Squares Quadratic Programming
    'trust-constr',  # Trust Region Constrained Optimization
    'dogleg',  # Dogleg method
    'trust-ncg',  # Trust Region Newton Conjugate Gradient
    'trust-exact',  # Trust Region Exact
    'trust-krylov'  # Trust Region Krylov
}


class ConstraintType(Enum):
    UNCONSTRAINED = "unconstrained"
    CARDINALITY = "cardinality"
    PARTITION = "partition"
    INEQUALITY = "inequality"
    MULTIPLE = "multiple"
