# API Contract: Solution Feasibility Checking

**Phase**: 1 (Design & Contracts)
**Date**: 2026-02-11
**Purpose**: Define public API contracts and behavior guarantees

## Public API Surface

### 1. QModel.solve() - Extended

**Signature**:
```python
def solve(
    self,
    method: str = 'classical',
    config: ExecutionConfig = ExecutionConfig()
) -> ModelSolution
```

**Changes**:
- **Return Type**: ModelSolution (extended with feasibility fields)
- **Behavior**: For `method="quantum"`, automatically performs feasibility checking and populates feasibility fields

**Contract**:
```python
# Input constraints
assert method in ['classical', 'quantum']
assert isinstance(config, ExecutionConfig)

# Output guarantees for quantum method
if method == 'quantum':
    result = model.solve(method='quantum', config=config)

    # Feasibility fields are always populated (never None after quantum solve)
    assert result.is_feasible in [True, False, "unknown"]
    assert isinstance(result.violated_constraints, list)
    assert isinstance(result.satisfied_constraints, list)
    assert isinstance(result.constraint_violations, list)

    # Feasibility invariants
    if result.is_feasible is True:
        assert len(result.violated_constraints) == 0
        assert len(result.constraint_violations) == 0

    elif result.is_feasible is False:
        assert len(result.violated_constraints) > 0
        assert len(result.constraint_violations) == len(result.violated_constraints)

    elif result.is_feasible == "unknown":
        assert result.feasibility_error is not None

# Classical method: feasibility fields remain None/empty (no change)
if method == 'classical':
    result = model.solve(method='classical')
    assert result.is_feasible is None  # Not checked for classical
```

**Example Usage**:
```python
model = QModel("my_problem")
# ... add variables, constraints, objective ...

config = ExecutionConfig(
    provider="ibmq",
    backend="ibm_brisbane",
    feasibility_tolerance=1e-6
)

solution = model.solve(method="quantum", config=config)

# Check feasibility
if solution.is_feasible:
    print(f"Found feasible solution: {solution.solution}")
elif solution.is_feasible == "unknown":
    print(f"Could not check feasibility: {solution.feasibility_error}")
else:
    print(f"Solution violates {len(solution.violated_constraints)} constraints")
    for violation in solution.constraint_violations:
        print(f"  {violation.constraint_id}: off by {violation.violation_magnitude}")
```

---

### 2. ModelSolution - Extended Properties

**New Properties** (all read-only, set by solve()):

#### is_feasible
```python
is_feasible: Optional[bool | str]
```
- **Type**: `bool` (True/False) or `str` ("unknown") or `None`
- **Meaning**:
  - `True`: All constraints satisfied
  - `False`: At least one constraint violated
  - `"unknown"`: Feasibility checking failed (see `feasibility_error`)
  - `None`: Feasibility not checked (classical solve or pre-check state)
- **Guarantees**: Never raises when accessed

#### violated_constraints
```python
violated_constraints: List[str]
```
- **Type**: List of constraint identifiers (strings)
- **Content**: IDs of constraints that failed satisfaction check
- **Guarantees**:
  - Empty list if `is_feasible is True`
  - Non-empty if `is_feasible is False`
  - Order matches `constraint_violations` list
- **Constraint ID Format**: DOcplex constraint name, or `"constraint_{idx}"` if unnamed

#### satisfied_constraints
```python
satisfied_constraints: List[str]
```
- **Type**: List of constraint identifiers (strings)
- **Content**: IDs of constraints that passed satisfaction check
- **Guarantees**:
  - Contains all constraint IDs if `is_feasible is True`
  - May be non-empty even if `is_feasible is False` (partial feasibility)

#### constraint_violations
```python
constraint_violations: List[ConstraintViolation]
```
- **Type**: List of ConstraintViolation objects
- **Content**: Detailed violation information for each violated constraint
- **Guarantees**:
  - Empty list if `is_feasible is True`
  - Length matches `violated_constraints` if `is_feasible is False`
  - Each entry provides: constraint ID, bounds, actual value, violation magnitude

#### feasibility_error
```python
feasibility_error: Optional[str]
```
- **Type**: Optional string
- **Content**: Human-readable error message if feasibility checking failed
- **Guarantees**:
  - `None` if `is_feasible in [True, False]`
  - Set to error message if `is_feasible == "unknown"`

---

### 3. ExecutionConfig - New Field

**New Field**:
```python
@dataclass
class ExecutionConfig:
    # ... existing fields ...

    feasibility_tolerance: float = 1e-6
```

**Contract**:
- **Type**: float (must be positive)
- **Default**: 1e-6
- **Purpose**: Absolute tolerance for floating-point constraint comparisons
- **Validation**: Validated in `__post_init__` (raises ValueError if <= 0)

**Usage**:
```python
# Use default tolerance (1e-6)
config = ExecutionConfig()

# Custom tolerance for strict problems
strict_config = ExecutionConfig(feasibility_tolerance=1e-9)

# Custom tolerance for numerical stability
loose_config = ExecutionConfig(feasibility_tolerance=1e-4)
```

---

### 4. ConstraintViolation - Public Dataclass

**Full Contract**:
```python
@dataclass
class ConstraintViolation:
    constraint_id: str
    constraint_type: str
    expected_bound: float
    actual_value: float
    violation_magnitude: float
    expression: Optional[str] = None
```

**Field Contracts**:
- `constraint_id`: Never empty string
- `constraint_type`: One of ["equality", "inequality_le", "inequality_ge"]
- `expected_bound`: Finite float (not NaN/Inf)
- `actual_value`: Finite float (not NaN/Inf)
- `violation_magnitude`: Non-negative float (>= 0.0)
- `expression`: Human-readable constraint (optional, may be None)

**Invariants**:
```python
assert violation.violation_magnitude >= 0.0

# Type-specific magnitude relationships
if violation.constraint_type == "equality":
    assert violation.violation_magnitude == abs(
        violation.actual_value - violation.expected_bound
    )
elif violation.constraint_type == "inequality_le":
    assert violation.violation_magnitude == max(
        0.0, violation.actual_value - violation.expected_bound
    )
elif violation.constraint_type == "inequality_ge":
    assert violation.violation_magnitude == max(
        0.0, violation.expected_bound - violation.actual_value
    )
```

---

## Error Handling Contract

### Never Raises

The feasibility checking feature **never raises exceptions** to user code:

```python
# These NEVER raise, even with malformed inputs
solution = model.solve(method="quantum", config=config)

# Safe to access even if checking failed
print(solution.is_feasible)  # May be "unknown" but never raises
print(solution.violated_constraints)  # Always returns list (may be empty)
```

### Error Scenarios

| Scenario | Behavior |
|----------|----------|
| Model has no constraints | `is_feasible = True`, all lists empty |
| Variable missing from solution | `is_feasible = "unknown"`, error in `feasibility_error` |
| Constraint evaluation fails | `is_feasible = "unknown"`, error in `feasibility_error` |
| Model unavailable | `is_feasible = "unknown"`, error in `feasibility_error` |
| Invalid tolerance in config | Raises `ValueError` in `ExecutionConfig.__post_init__` (before solve) |

---

## Backward Compatibility

### Existing Code Compatibility

**Guarantee**: All existing code continues to work without modification

```python
# Existing code (before feature)
solution = model.solve(method="quantum", config=config)
print(solution.objective)  # Still works
print(solution.solution)   # Still works

# New feasibility fields are ignored if not accessed
# No breaking changes to ModelSolution dataclass
```

### Optional Adoption

Users can ignore feasibility features:

```python
# Minimal usage: just check overall feasibility
if solution.is_feasible:
    use_solution(solution.solution)

# Detailed usage: examine violations
for v in solution.constraint_violations:
    print(f"{v.constraint_id} violated by {v.violation_magnitude}")
```

---

## Performance Contract

### Guarantees

1. **Completion Time**: Feasibility checking completes in <5 seconds for:
   - Up to 1,000 constraints
   - Up to 1,000 variables

2. **Memory**: O(n) additional memory where n = number of constraints
   - Each ConstraintViolation is ~100 bytes
   - Max ~100KB for 1,000 violations

3. **No Quantum Resources**: All checking happens on classical computer
   - No additional quantum circuit executions
   - No additional quantum API calls

### Non-Guarantees (Implementation Details)

- Exact runtime for <1,000 constraints (expected <100ms)
- Constraint evaluation order (unspecified)
- Thread safety of FeasibilityChecker (assume single-threaded)

---

## Testing Contract

### Test Coverage Requirements

Minimum 80% branch coverage (enforced by pytest.ini):

**Required Test Scenarios**:
1. Feasible solution (all constraints satisfied)
2. Infeasible solution (some constraints violated)
3. Unconstrained problem (no constraints)
4. Each constraint type: equality, inequality_le, inequality_ge
5. Boundary cases (values exactly at constraint bounds)
6. Floating-point tolerance edge cases
7. Error scenarios (unknown status)
8. Backward compatibility (classical solve unchanged)

**Test Data**:
```python
# Feasible solution test
def test_feasible_solution():
    model = QModel("test")
    x = model.binary_var("x")
    y = model.binary_var("y")
    model.add_constraint(x + y <= 1)

    solution = model.solve(method="quantum", config=config)
    # Assume quantum solver returns x=0, y=1

    assert solution.is_feasible is True
    assert len(solution.violated_constraints) == 0

# Infeasible solution test
def test_infeasible_solution():
    model = QModel("test")
    x = model.binary_var("x")
    y = model.binary_var("y")
    model.add_constraint(x + y <= 1)

    solution = model.solve(method="quantum", config=config)
    # Assume quantum solver returns x=1, y=1 (infeasible)

    assert solution.is_feasible is False
    assert len(solution.violated_constraints) == 1
    assert solution.constraint_violations[0].violation_magnitude == 1.0
```

---

## Summary

**Public API Changes**:
- ModelSolution: 5 new fields (all optional, backward compatible)
- ExecutionConfig: 1 new field (optional, default 1e-6)
- New public type: ConstraintViolation

**Guarantees**:
- Never raises exceptions from feasibility checking
- <5 second checking for 1,000 constraints/variables
- Backward compatible with existing code
- 80% branch coverage in tests

**Contract Principles**:
- "Define errors out of existence": return "unknown" instead of raising
- Fail gracefully: always return valid ModelSolution
- Transparent: detailed violation information available
- Opt-in: users can ignore feasibility features if desired
