# Research: Solution Feasibility Checking

**Phase**: 0 (Outline & Research)
**Date**: 2026-02-11
**Purpose**: Resolve technical unknowns and document design decisions

## Research Areas

### 1. DOcplex Constraint Evaluation

**Question**: How to programmatically evaluate DOcplex constraints against solution values?

**Decision**: Use DOcplex constraint API methods

**Rationale**:
- DOcplex constraints expose `get_left_expr()` and `get_right_expr()` methods
- Existing `model_utils.py` demonstrates pattern: iterate constraints via `model.iter_constraints()`
- Constraint expressions support `iter_terms()` to extract variable coefficients
- Can evaluate by substituting variable values into expressions and comparing with bounds

**Alternatives Considered**:
- Parse constraint strings: Too fragile, doesn't handle all constraint types
- Re-implement QUBO conversion inverse: Complex, error-prone, doesn't preserve semantics

**Implementation Pattern** (from existing code):
```python
for constraint in model.iter_constraints():
    left_expr = constraint.get_left_expr()
    right_expr = constraint.get_right_expr()
    sense = constraint.sense  # EQ, LE, GE
    # Evaluate left_expr with solution values
    # Compare to right_expr using sense
```

---

### 2. Constraint Type Support

**Question**: What constraint types must be supported and how to handle each?

**Decision**: Support all DOcplex constraint types using unified evaluation approach

**Rationale**:
- Spec requires: linear equality, linear inequality, quadratic, cardinality (FR-006)
- DOcplex provides `ComparisonType` enum: EQ, LE, GE
- All constraint types share same API (get_left_expr, get_right_expr, sense)
- Unified approach simplifies code and testing

**Constraint Type Handling**:
1. **Linear constraints**: Evaluate `sum(coef * var_value)` against bound
2. **Quadratic constraints**: DOcplex quadratic expressions support `iter_terms()` for quadratic and linear terms
3. **Cardinality constraints**: Special case of linear equality (sum x_i = k)
4. **Indicator constraints**: Edge case - document as limitation if unsupported

**Reference**: `qplex/utils/model_utils.py::get_model_constraint_info()` shows constraint type detection patterns

---

### 3. Dataclass Extension Pattern

**Question**: How to extend ModelSolution dataclass with new fields?

**Decision**: Add optional fields with default None values, populate in solve()

**Rationale**:
- Python dataclasses support field defaults
- Maintains backward compatibility (existing code ignores new fields)
- Follows ExecutionConfig pattern: fields with defaults at end
- No need for inheritance or wrapper classes (simple, Pythonic)

**Pattern**:
```python
@dataclass
class ModelSolution:
    # Existing fields
    solution: Dict[str, Any]
    objective: float
    execution_time: float
    method: str
    provider: Optional[str] = None
    backend: Optional[str] = None
    algorithm: str = "N/A"

    # New fields (with defaults for backward compatibility)
    is_feasible: Optional[bool] = None
    violated_constraints: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    constraint_violations: List['ConstraintViolation'] = field(default_factory=list)
    feasibility_error: Optional[str] = None
```

---

### 4. Floating-Point Tolerance Handling

**Question**: How to handle floating-point precision when checking constraint satisfaction?

**Decision**: Use configurable absolute tolerance with default 1e-6

**Rationale**:
- Spec clarification: user-configurable via ExecutionConfig, default 1e-6
- Absolute tolerance simpler than relative for constraint bounds
- Standard pattern: `abs(actual - expected) <= tolerance`
- Consistent with numpy.isclose() default (atol=1e-8, rtol=1e-5)

**Implementation**:
```python
def is_satisfied(actual: float, expected: float, sense: ComparisonType, tolerance: float) -> bool:
    if sense == ComparisonType.EQ:
        return abs(actual - expected) <= tolerance
    elif sense == ComparisonType.LE:
        return actual <= expected + tolerance
    elif sense == ComparisonType.GE:
        return actual >= expected - tolerance
```

**Edge Case**: Boundary values (e.g., x=5.0 for x<=5) are satisfied within tolerance

---

### 5. Constraint Identification

**Question**: How should constraints be identified in violation reports?

**Decision**: Use DOcplex constraint names, fall back to index-based naming

**Rationale**:
- DOcplex constraints have `.name` property (user-provided or auto-generated)
- If no name: use `f"constraint_{index}"` pattern
- Provides stable, human-readable identifiers
- Matches user expectations from DOcplex API

**Pattern**:
```python
for idx, constraint in enumerate(model.iter_constraints()):
    constraint_id = constraint.name if constraint.name else f"constraint_{idx}"
```

---

### 6. Error Handling Strategy

**Question**: How to handle errors during constraint checking without breaking solve()?

**Decision**: Graceful degradation with "unknown" feasibility status

**Rationale**:
- Spec clarification: return "unknown" status with error details (FR-013)
- "Define errors out of existence" principle: no exceptions from feasibility checking
- Users still get solution values even if checking fails
- Error details in `feasibility_error` property for debugging

**Error Scenarios**:
1. Missing constraint data → "unknown" + error message
2. Variable not in solution → "unknown" + error message
3. Evaluation exception (e.g., division by zero) → "unknown" + error message
4. Model unavailable → "unknown" + error message

**Implementation Pattern**:
```python
try:
    # Perform constraint checking
    is_feasible = ...
except Exception as e:
    return {
        'is_feasible': 'unknown',
        'feasibility_error': f"Constraint checking failed: {str(e)}"
    }
```

---

### 7. Performance Optimization

**Question**: How to ensure <5 second performance for 1,000 constraints?

**Decision**: Use efficient iteration, avoid redundant computations

**Rationale**:
- Single pass through constraints: O(n) where n = number of constraints
- Evaluate each constraint once (no repeated iteration)
- Cache solution variable lookups in dictionary
- Python loops over 1,000 items complete in milliseconds
- Constraint evaluation is simple arithmetic (no complex operations)

**Performance Strategy**:
1. Pre-process solution into `{var_name: value}` dict for O(1) lookup
2. Single iteration over constraints
3. Lazy property evaluation (only compute when accessed) - NOT needed, always compute
4. Avoid deep copying or serialization

**Estimated Performance**: ~10-100ms for 1,000 constraints (well under 5 second target)

---

### 8. Integration Point

**Question**: When and where should feasibility checking be triggered?

**Decision**: Automatically during solve(), populate ModelSolution fields

**Rationale**:
- Users expect immediate feedback (SC-001: "immediately upon receiving quantum results")
- Synchronous checking avoids lazy evaluation complexity
- Fits naturally in QModel.solve() after solution is obtained
- Consistent with existing flow: solve() → process results → return ModelSolution

**Integration Flow**:
```python
def solve(self, method='classical', config=ExecutionConfig()):
    # ... existing solve logic ...
    solution = self._create_solution(...)

    # NEW: Add feasibility checking for quantum method
    if method == 'quantum':
        feasibility_result = check_feasibility(
            model=self,
            solution=solution.solution,
            tolerance=config.tolerance
        )
        # Populate solution with feasibility data
        solution.is_feasible = feasibility_result['is_feasible']
        solution.violated_constraints = feasibility_result['violated_constraints']
        # ...

    return solution
```

---

## Summary

All technical unknowns resolved. Key decisions:

1. **Constraint Evaluation**: Use DOcplex constraint API (get_left_expr, get_right_expr, sense)
2. **Dataclass Extension**: Add optional fields to ModelSolution with defaults
3. **Tolerance**: Configurable via ExecutionConfig, default 1e-6, absolute comparison
4. **Error Handling**: Return "unknown" status, never raise exceptions
5. **Performance**: Single O(n) pass, pre-processed solution dict, <100ms expected
6. **Integration**: Synchronous checking in solve() for quantum method only

No blocking issues identified. Ready for Phase 1 (Design & Contracts).
