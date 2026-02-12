# Implementation Plan: Solution Feasibility Checking

**Branch**: `001-feasibility-checking` | **Date**: 2026-02-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-feasibility-checking/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add automatic constraint feasibility checking to quantum solutions by evaluating original DOcplex constraints against quantum solution values. The feature extends the existing ModelSolution dataclass with feasibility properties (is_feasible, violated_constraints, satisfied_constraints, constraint_violations, feasibility_error), enabling users to immediately determine if quantum solutions satisfy problem constraints without manual verification. This addresses the core issue that QUBO conversion loses constraint semantics.

## Technical Context

**Language/Version**: Python >= 3.10 (per pyproject.toml)
**Primary Dependencies**: DOcplex (docplex.mp), dataclasses, typing
**Storage**: N/A (in-memory constraint evaluation)
**Testing**: pytest with 80% branch coverage minimum (enforced)
**Target Platform**: Python library (cross-platform)
**Project Type**: Single Python package (qplex/)
**Performance Goals**: Constraint checking completes in <5 seconds for 1,000 constraints and 1,000 variables
**Constraints**:
  - Must preserve DOcplex model constraint semantics
  - Must work with existing QModel.solve() return type (ModelSolution)
  - Configurable floating-point tolerance (default 1e-6)
  - Must handle all DOcplex constraint types (linear equality/inequality, quadratic, cardinality)
**Scale/Scope**: Up to 1,000 variables and 1,000 constraints per problem

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

No project constitution file found at `.specify/memory/constitution.md`. Proceeding with standard best practices:
- ✅ Follow existing codebase patterns (dataclasses, utils modules, test structure)
- ✅ Maintain 80% branch coverage requirement
- ✅ Use NumPy-style docstrings
- ✅ Minimize new abstractions (extend existing ModelSolution)
- ✅ Define errors out of existence (return "unknown" status instead of raising)

## Project Structure

### Documentation (this feature)

```text
specs/001-feasibility-checking/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── feasibility-api.md
├── checklists/          # From /speckit.specify
│   └── requirements.md
└── spec.md              # From /speckit.specify
```

### Source Code (repository root)

```text
qplex/
├── model/
│   ├── qmodel.py               # MODIFY: update solve() to call feasibility checker
│   ├── execution_config.py     # MODIFY: add tolerance parameter
│   └── feasibility.py          # NEW: FeasibilityChecker class, dataclasses
├── utils/
│   └── feasibility_utils.py    # NEW: constraint evaluation helpers
└── __init__.py                 # MODIFY: export new feasibility types

tests/
├── model/
│   ├── test_qmodel.py          # MODIFY: add feasibility checking tests
│   └── test_feasibility.py     # NEW: unit tests for FeasibilityChecker
└── utils/
    └── test_feasibility_utils.py  # NEW: unit tests for helpers
```

**Structure Decision**: Single project structure (Option 1). This feature extends the existing qplex package by adding feasibility checking to the model/ module (core QModel functionality) and helpers to utils/ (following the pattern used for circuit_utils, model_utils, workflow_utils). Tests mirror the source structure as per existing convention.

## Complexity Tracking

No constitution violations detected. This feature follows established patterns:
- Extends existing ModelSolution dataclass (pattern used throughout codebase)
- Adds utils module (consistent with circuit_utils, model_utils, workflow_utils)
- Uses dataclasses for data structures (ExecutionConfig, ConstraintInfo pattern)
- Minimal abstraction: single FeasibilityChecker class, no factories or complex inheritance

---

## Phase 0: Research (Complete)

**Status**: ✅ Complete

**Artifact**: [research.md](./research.md)

**Key Decisions**:
1. Use DOcplex constraint API (get_left_expr, get_right_expr, sense) for evaluation
2. Extend ModelSolution dataclass with optional fields (backward compatible)
3. Configurable absolute tolerance (default 1e-6) via ExecutionConfig
4. Graceful error handling: return "unknown" status instead of raising
5. Synchronous checking in QModel.solve() for quantum method

**Unknowns Resolved**: 8/8
- DOcplex constraint evaluation patterns ✓
- Constraint type support (linear, quadratic, cardinality) ✓
- Dataclass extension approach ✓
- Floating-point tolerance handling ✓
- Constraint identifier format ✓
- Error handling strategy ✓
- Performance optimization approach ✓
- Integration point in solve() flow ✓

---

## Phase 1: Design & Contracts (Complete)

**Status**: ✅ Complete

**Artifacts**:
- [data-model.md](./data-model.md) - Entity definitions and relationships
- [contracts/feasibility-api.md](./contracts/feasibility-api.md) - API contracts and behavior guarantees
- [quickstart.md](./quickstart.md) - User guide and examples

**Data Model**:
- **New Entities**: 3 (ConstraintViolation, FeasibilityResult, FeasibilityChecker)
- **Modified Entities**: 2 (ModelSolution, ExecutionConfig)
- **Dataclasses**: 3 (ConstraintViolation, FeasibilityResult, ModelSolution extended)
- **Classes**: 1 (FeasibilityChecker)

**API Surface**:
- QModel.solve() - Extended to populate feasibility fields for quantum method
- ModelSolution - 5 new properties (is_feasible, violated_constraints, satisfied_constraints, constraint_violations, feasibility_error)
- ExecutionConfig - 1 new field (feasibility_tolerance)
- ConstraintViolation - New public dataclass for violation details

**Contracts**:
- Never raises exceptions from feasibility checking
- <5 second performance for 1,000 constraints/variables
- Backward compatible (all new fields optional)
- "Unknown" status for error cases (define errors out of existence)

**Agent Context**: Updated CLAUDE.md with feature technologies

---

## Implementation Guidance

### Key Principles (from user input)

1. **Follow established patterns**: Mirror existing qplex structure (dataclasses, utils, test organization)
2. **Write simple, elegant Pythonic code**: Prefer composition over inheritance, use type hints, clear naming
3. **Define errors out of existence**: Return "unknown" status instead of raising exceptions

### Design Patterns to Use

**Dataclasses** (existing pattern):
```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ConstraintViolation:
    constraint_id: str
    constraint_type: str
    expected_bound: float
    actual_value: float
    violation_magnitude: float
    expression: Optional[str] = None
```

**Error Handling** (define errors out of existence):
```python
def check_feasibility(model, solution, tolerance):
    """Never raises - returns 'unknown' on errors."""
    try:
        # Perform checking
        return {'is_feasible': True, ...}
    except Exception as e:
        return {
            'is_feasible': 'unknown',
            'feasibility_error': f"Checking failed: {str(e)}"
        }
```

**NumPy-style Docstrings** (existing pattern):
```python
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

### Files to Create

**New Files** (3):
1. `qplex/model/feasibility.py` - FeasibilityChecker, ConstraintViolation, FeasibilityResult
2. `qplex/utils/feasibility_utils.py` - Helper functions for constraint evaluation
3. `tests/model/test_feasibility.py` - Unit tests for feasibility checking
4. `tests/utils/test_feasibility_utils.py` - Unit tests for helpers

**Files to Modify** (3):
1. `qplex/model/qmodel.py` - Update solve() to call feasibility checker
2. `qplex/model/execution_config.py` - Add feasibility_tolerance field
3. `qplex/__init__.py` - Export ConstraintViolation type
4. `tests/model/test_qmodel.py` - Add integration tests

### Implementation Order

**Recommended sequence** (bottom-up):
1. Create `qplex/utils/feasibility_utils.py` with constraint evaluation helpers
2. Create `qplex/model/feasibility.py` with dataclasses and FeasibilityChecker
3. Modify `qplex/model/execution_config.py` to add tolerance field
4. Extend ModelSolution dataclass in `qplex/model/qmodel.py`
5. Modify `QModel.solve()` to integrate feasibility checking
6. Update `qplex/__init__.py` exports
7. Write tests (utils, then model)

---

## Next Steps

**Current Status**: Planning complete (Phases 0-1)

**Next Command**: `/speckit.tasks`

This will generate `tasks.md` with detailed implementation tasks ordered by dependencies, ready for execution via `/speckit.implement`.

**What to Expect in tasks.md**:
- Task breakdown aligned with implementation order above
- Dependency ordering (helpers → core classes → integration → tests)
- Acceptance criteria for each task
- Estimated complexity/effort per task

---

## Planning Summary

**Phases Completed**: 2/2 (Phase 0: Research, Phase 1: Design)
**Artifacts Generated**: 5 files
- research.md (Phase 0)
- data-model.md (Phase 1)
- contracts/feasibility-api.md (Phase 1)
- quickstart.md (Phase 1)
- plan.md (this file)

**Design Quality**:
- ✅ All unknowns resolved
- ✅ Constitution gates passed
- ✅ API contracts defined
- ✅ Data model complete
- ✅ User guide written
- ✅ Agent context updated
- ✅ Follows codebase patterns
- ✅ Backward compatible
- ✅ Performance targets defined

**Ready for**: Task generation (`/speckit.tasks`) and implementation (`/speckit.implement`)
