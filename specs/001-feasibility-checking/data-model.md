# Data Model: Solution Feasibility Checking

**Phase**: 1 (Design & Contracts)
**Date**: 2026-02-11
**Purpose**: Define all data structures and relationships

## Entity Definitions

### 1. ModelSolution (Modified)

**Location**: `qplex/model/qmodel.py`
**Type**: Dataclass (existing, will be extended)
**Purpose**: Represents solution from QModel.solve() with feasibility information

**Fields**:
```python
@dataclass
class ModelSolution:
    """Solution returned by QModel.solve() with feasibility information."""

    # Existing fields
    solution: Dict[str, Any]              # Variable assignments {var_name: value}
    objective: float                      # Objective function value
    execution_time: float                 # Total execution time in seconds
    method: str                          # "classical" or "quantum"
    provider: Optional[str] = None       # Quantum provider if method="quantum"
    backend: Optional[str] = None        # Backend name if method="quantum"
    algorithm: str = "N/A"              # Algorithm used

    # NEW: Feasibility fields (defaults for backward compatibility)
    is_feasible: Optional[bool | str] = None  # True/False or "unknown"
    violated_constraints: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    constraint_violations: List['ConstraintViolation'] = field(default_factory=list)
    feasibility_error: Optional[str] = None  # Error details if checking failed
```

**Validation Rules**:
- `is_feasible` can be: `True`, `False`, or `"unknown"` (string for error cases)
- If `is_feasible == True`: `violated_constraints` must be empty
- If `is_feasible == False`: `violated_constraints` must not be empty
- If `is_feasible == "unknown"`: `feasibility_error` must be set
- `constraint_violations` length equals `violated_constraints` length

**State Transitions**:
- Initial state (after solve, before feasibility check): all feasibility fields are None/empty
- After successful check: `is_feasible` = True/False, lists populated
- After failed check: `is_feasible` = "unknown", `feasibility_error` set

---

### 2. ConstraintViolation (New)

**Location**: `qplex/model/feasibility.py`
**Type**: Dataclass (new)
**Purpose**: Detailed information about a single constraint violation

**Fields**:
```python
@dataclass
class ConstraintViolation:
    """Detailed information about a constraint violation."""

    constraint_id: str            # Constraint name or "constraint_{idx}"
    constraint_type: str          # "equality", "inequality_le", "inequality_ge"
    expected_bound: float         # Right-hand side value (constraint bound)
    actual_value: float          # Left-hand side evaluated value
    violation_magnitude: float   # abs(actual_value - expected_bound)
    expression: Optional[str] = None  # String representation of constraint (if available)
```

**Validation Rules**:
- `violation_magnitude >= 0` (always non-negative)
- `constraint_type` must be one of: "equality", "inequality_le", "inequality_ge"
- For equality: `violation_magnitude = abs(actual_value - expected_bound)`
- For inequality_le: `violation_magnitude = max(0, actual_value - expected_bound)`
- For inequality_ge: `violation_magnitude = max(0, expected_bound - actual_value)`

**Example**:
```python
# For constraint: x + y <= 5 with x=3, y=4
ConstraintViolation(
    constraint_id="capacity_limit",
    constraint_type="inequality_le",
    expected_bound=5.0,
    actual_value=7.0,
    violation_magnitude=2.0,
    expression="x + y <= 5"
)
```

---

### 3. FeasibilityResult (New)

**Location**: `qplex/model/feasibility.py`
**Type**: TypedDict or dataclass (internal use)
**Purpose**: Internal return type from FeasibilityChecker

**Fields**:
```python
@dataclass
class FeasibilityResult:
    """Internal result from constraint feasibility checking."""

    is_feasible: bool | str           # True/False or "unknown"
    violated_constraints: List[str]   # IDs of violated constraints
    satisfied_constraints: List[str]  # IDs of satisfied constraints
    violations: List[ConstraintViolation]  # Detailed violation info
    error: Optional[str] = None      # Error message if checking failed
```

**Validation Rules**:
- If `is_feasible == True`: `violated_constraints` and `violations` must be empty
- If `is_feasible == False`: `violated_constraints` must match `violations` length
- If `is_feasible == "unknown"`: `error` must be set

---

### 4. FeasibilityChecker (New)

**Location**: `qplex/model/feasibility.py`
**Type**: Class (new)
**Purpose**: Evaluates constraints against solution values

**Interface**:
```python
class FeasibilityChecker:
    """Evaluates DOcplex constraints against solution values."""

    def __init__(self, model: docplex.mp.model.Model, tolerance: float = 1e-6):
        """
        Initialize feasibility checker.

        Parameters
        ----------
        model : Model
            DOcplex model with constraints
        tolerance : float
            Absolute tolerance for floating-point comparisons (default 1e-6)
        """

    def check(self, solution: Dict[str, Any]) -> FeasibilityResult:
        """
        Check if solution satisfies all model constraints.

        Parameters
        ----------
        solution : dict
            Variable assignments {var_name: value}

        Returns
        -------
        FeasibilityResult
            Feasibility status and violation details

        Notes
        -----
        Never raises exceptions. Returns "unknown" status on errors.
        """
```

**Methods**:
1. `__init__(model, tolerance)`: Store model and tolerance
2. `check(solution)`: Main entry point, returns FeasibilityResult
3. `_evaluate_constraint(constraint, solution)`: Evaluate single constraint (private)
4. `_get_constraint_id(constraint, index)`: Get constraint name/ID (private)

**Error Handling**:
- All methods use try-except to catch evaluation errors
- Errors converted to "unknown" status with error message
- Never propagates exceptions to caller

---

## Entity Relationships

```text
ModelSolution
├── contains → List[ConstraintViolation]  (constraint_violations field)
└── references → QModel (implicitly, created by QModel.solve())

FeasibilityChecker
├── operates on → QModel (passed in __init__)
├── consumes → Dict[str, Any] (solution values)
└── produces → FeasibilityResult

FeasibilityResult
├── converted to → ModelSolution fields (in QModel.solve())
└── contains → List[ConstraintViolation]
```

**Data Flow**:
```text
1. QModel.solve() obtains solution from quantum solver
2. QModel creates FeasibilityChecker(model=self, tolerance=config.tolerance)
3. FeasibilityChecker.check(solution_dict) → FeasibilityResult
4. QModel populates ModelSolution fields from FeasibilityResult
5. Return ModelSolution to user
```

---

## Validation & Invariants

### ModelSolution Invariants

```python
def validate_model_solution(sol: ModelSolution):
    """Invariants that must hold for ModelSolution with feasibility data."""

    # If feasibility checked (is_feasible is not None)
    if sol.is_feasible is not None:
        if sol.is_feasible is True:
            assert len(sol.violated_constraints) == 0
            assert len(sol.constraint_violations) == 0
            assert sol.feasibility_error is None

        elif sol.is_feasible is False:
            assert len(sol.violated_constraints) > 0
            assert len(sol.violated_constraints) == len(sol.constraint_violations)
            assert sol.feasibility_error is None

        elif sol.is_feasible == "unknown":
            assert sol.feasibility_error is not None
```

### ConstraintViolation Invariants

```python
def validate_violation(v: ConstraintViolation):
    """Invariants for ConstraintViolation."""

    assert v.violation_magnitude >= 0.0
    assert v.constraint_type in ["equality", "inequality_le", "inequality_ge"]

    # Type-specific magnitude checks
    if v.constraint_type == "equality":
        expected_mag = abs(v.actual_value - v.expected_bound)
        assert abs(v.violation_magnitude - expected_mag) < 1e-9

    elif v.constraint_type == "inequality_le":
        expected_mag = max(0.0, v.actual_value - v.expected_bound)
        assert abs(v.violation_magnitude - expected_mag) < 1e-9

    elif v.constraint_type == "inequality_ge":
        expected_mag = max(0.0, v.expected_bound - v.actual_value)
        assert abs(v.violation_magnitude - expected_mag) < 1e-9
```

---

## Type Definitions

### ExecutionConfig Extension

**Location**: `qplex/model/execution_config.py`
**Modification**: Add tolerance field

```python
@dataclass
class ExecutionConfig:
    """Configuration for quantum optimization execution."""

    # ... existing fields ...

    # NEW: Feasibility checking tolerance
    feasibility_tolerance: float = 1e-6
```

**Backward Compatibility**: Default value maintains existing behavior

---

## Summary

**New Entities**: 3 (ConstraintViolation, FeasibilityResult, FeasibilityChecker)
**Modified Entities**: 2 (ModelSolution, ExecutionConfig)
**Total Dataclasses**: 3 (ConstraintViolation, FeasibilityResult, ModelSolution extended)
**Total Classes**: 1 (FeasibilityChecker)

All entities follow existing codebase patterns (dataclasses, NumPy-style docstrings, type hints).
