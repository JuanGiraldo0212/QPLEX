# Quickstart: Solution Feasibility Checking

**Feature**: Automatic constraint feasibility checking for quantum solutions
**Audience**: QPLEX users (data scientists, researchers)
**Time**: 5 minutes

## Overview

When you solve optimization problems on quantum devices, QPLEX now automatically checks if the returned solution satisfies your original constraints. You get immediate feedback about feasibility without manual verification.

## Basic Usage

### 1. Define Your Problem (As Usual)

```python
from qplex.model import QModel, ExecutionConfig

# Create model with constraints
model = QModel("resource_allocation")

# Add variables
x = model.binary_var("x")
y = model.binary_var("y")
z = model.binary_var("z")

# Add constraints
model.add_constraint(x + y + z == 2, "select_exactly_two")
model.add_constraint(x + 2*y <= 3, "capacity_limit")

# Set objective
model.maximize(4*x + 3*y + 2*z)
```

### 2. Solve on Quantum Device

```python
# Configure quantum execution
config = ExecutionConfig(
    provider="ibmq",
    backend="ibm_brisbane",
    algorithm="qaoa",
    shots=1024
)

# Solve (feasibility checking happens automatically)
solution = model.solve(method="quantum", config=config)
```

### 3. Check Feasibility

```python
# Simple check: is the solution valid?
if solution.is_feasible:
    print("✓ Solution satisfies all constraints!")
    print(f"Objective value: {solution.objective}")
    print(f"Solution: {solution.solution}")
else:
    print("✗ Solution violates constraints")
    print(f"Violated: {len(solution.violated_constraints)} constraint(s)")
```

**Output Example** (feasible):
```
✓ Solution satisfies all constraints!
Objective value: 7.0
Solution: {'x': 1, 'y': 1, 'z': 0}
```

**Output Example** (infeasible):
```
✗ Solution violates constraints
Violated: 1 constraint(s)
```

## Detailed Violation Analysis

### View Which Constraints Failed

```python
if not solution.is_feasible:
    print(f"Violated constraints: {solution.violated_constraints}")
    print(f"Satisfied constraints: {solution.satisfied_constraints}")
```

**Output**:
```
Violated constraints: ['capacity_limit']
Satisfied constraints: ['select_exactly_two']
```

### Examine Violation Details

```python
if not solution.is_feasible:
    for violation in solution.constraint_violations:
        print(f"Constraint: {violation.constraint_id}")
        print(f"  Type: {violation.constraint_type}")
        print(f"  Expected: {violation.expression}")
        print(f"  Actual value: {violation.actual_value}")
        print(f"  Limit: {violation.expected_bound}")
        print(f"  Violated by: {violation.violation_magnitude}")
        print()
```

**Output**:
```
Constraint: capacity_limit
  Type: inequality_le
  Expected: x + 2*y <= 3
  Actual value: 5.0
  Limit: 3.0
  Violated by: 2.0
```

## Advanced Configuration

### Custom Tolerance for Floating-Point Comparisons

```python
# Strict tolerance (1e-9) for high-precision problems
strict_config = ExecutionConfig(
    provider="ibmq",
    backend="ibm_brisbane",
    feasibility_tolerance=1e-9
)

# Loose tolerance (1e-4) for numerical stability
loose_config = ExecutionConfig(
    provider="ibmq",
    backend="ibm_brisbane",
    feasibility_tolerance=1e-4
)

solution = model.solve(method="quantum", config=strict_config)
```

**When to adjust tolerance**:
- **Tighter (1e-9)**: High-precision problems, exact integer solutions expected
- **Looser (1e-4)**: Numerical optimization, floating-point heavy, stability issues
- **Default (1e-6)**: Most problems (recommended)

### Handling Unknown Status

Sometimes feasibility checking can fail (e.g., model data unavailable, evaluation errors):

```python
if solution.is_feasible == "unknown":
    print(f"Could not verify feasibility: {solution.feasibility_error}")
    print("Solution values are still available:")
    print(solution.solution)
```

**Output**:
```
Could not verify feasibility: Variable 'temp_var_1' not found in solution
Solution values are still available:
{'x': 1, 'y': 0, 'z': 1}
```

## Complete Example

```python
from qplex.model import QModel, ExecutionConfig

def solve_and_report(model, config):
    """Solve problem and report feasibility."""
    solution = model.solve(method="quantum", config=config)

    print(f"Execution time: {solution.execution_time:.2f}s")
    print(f"Objective: {solution.objective}")
    print(f"Provider: {solution.provider}/{solution.backend}")
    print()

    # Feasibility report
    if solution.is_feasible is True:
        print("✓ FEASIBLE - All constraints satisfied")
        print(f"  Satisfied: {len(solution.satisfied_constraints)} constraint(s)")

    elif solution.is_feasible is False:
        print("✗ INFEASIBLE - Constraint violations detected")
        print(f"  Violated: {len(solution.violated_constraints)} constraint(s)")
        print(f"  Satisfied: {len(solution.satisfied_constraints)} constraint(s)")
        print()
        print("Violation details:")
        for v in solution.constraint_violations:
            print(f"  - {v.constraint_id}: off by {v.violation_magnitude:.4f}")

    elif solution.is_feasible == "unknown":
        print("⚠ UNKNOWN - Could not verify feasibility")
        print(f"  Error: {solution.feasibility_error}")

    print()
    print(f"Solution: {solution.solution}")
    return solution


# Create problem
model = QModel("knapsack")
x1 = model.binary_var("item1")
x2 = model.binary_var("item2")
x3 = model.binary_var("item3")

model.add_constraint(2*x1 + 3*x2 + 4*x3 <= 7, "weight_limit")
model.add_constraint(x1 + x2 + x3 >= 2, "min_items")
model.maximize(5*x1 + 6*x2 + 7*x3)

# Solve
config = ExecutionConfig(provider="ibmq", backend="ibm_brisbane")
solution = solve_and_report(model, config)
```

**Output**:
```
Execution time: 12.45s
Objective: 13.0
Provider: ibmq/ibm_brisbane

✓ FEASIBLE - All constraints satisfied
  Satisfied: 2 constraint(s)

Solution: {'item1': 1, 'item2': 1, 'item3': 0}
```

## Comparison: Before vs. After

### Before (Manual Checking)

```python
solution = model.solve(method="quantum", config=config)

# Manual verification (tedious, error-prone)
x_val = solution.solution['x']
y_val = solution.solution['y']

# Check each constraint manually
if x_val + y_val == 2:
    print("Constraint 1: OK")
else:
    print("Constraint 1: VIOLATED")

if x_val + 2*y_val <= 3:
    print("Constraint 2: OK")
else:
    print("Constraint 2: VIOLATED")
```

### After (Automatic)

```python
solution = model.solve(method="quantum", config=config)

# Automatic verification
if solution.is_feasible:
    print("All constraints satisfied!")
```

## Integration with Existing Code

**Backward Compatible**: Existing code continues to work without changes.

```python
# Old code (still works)
solution = model.solve(method="quantum", config=config)
print(solution.objective)  # ✓ Works as before

# New code (opt-in)
if solution.is_feasible:   # ✓ New feature, optional
    accept_solution(solution)
```

## Performance Notes

- **Overhead**: Typically <100ms for problems with <1,000 constraints
- **Scale**: Handles up to 1,000 constraints and 1,000 variables in <5 seconds
- **No Quantum Cost**: Checking runs on classical computer, no additional quantum execution

## Troubleshooting

### Issue: "unknown" Status Returned

**Cause**: Feasibility checking encountered an error

**Solution**:
1. Check `solution.feasibility_error` for details
2. Verify all variables in constraints are in solution
3. Check for numerical issues (NaN, Inf values)

### Issue: Unexpected "infeasible" Result

**Cause**: Quantum solution violates constraints (expected for QUBO approximations)

**Solution**:
1. Examine `solution.constraint_violations` to see which constraints failed
2. Increase penalty weights in QUBO conversion
3. Try different quantum algorithm or backend
4. Increase number of shots for better sampling

### Issue: Tolerance Too Strict/Loose

**Cause**: Default tolerance (1e-6) not suitable for problem

**Solution**:
```python
# Adjust via ExecutionConfig
config = ExecutionConfig(
    provider="ibmq",
    backend="ibm_brisbane",
    feasibility_tolerance=1e-4  # Looser tolerance
)
```

## Next Steps

- **Learn More**: See [API Documentation](./contracts/feasibility-api.md) for detailed contracts
- **Examples**: Check `examples/` directory for more use cases
- **Contribute**: Report issues or suggest improvements on GitHub

## Quick Reference

```python
# Check overall feasibility
solution.is_feasible  # True, False, or "unknown"

# List violated/satisfied constraints
solution.violated_constraints   # List[str]
solution.satisfied_constraints  # List[str]

# Detailed violation info
solution.constraint_violations  # List[ConstraintViolation]

# Error details (if unknown)
solution.feasibility_error      # Optional[str]

# Configure tolerance
config = ExecutionConfig(feasibility_tolerance=1e-6)
```
