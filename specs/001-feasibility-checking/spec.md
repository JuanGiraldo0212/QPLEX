# Feature Specification: Solution Feasibility Checking

**Feature Branch**: `001-feasibility-checking`
**Created**: 2026-02-11
**Status**: Draft
**Input**: User description: "Solution feasibility checking — After quantum solving, automatically verify which original constraints are satisfied and report violations. QUBO conversion loses constraint semantics, so users currently can't tell if a quantum solution is feasible without manual checking."

## Clarifications

### Session 2026-02-11

- Q: How should the feasibility report be structured and accessed by users? → A: Add new properties to the existing solution object (e.g., solution.is_feasible, solution.violated_constraints, solution.satisfied_constraints)
- Q: How should the system handle errors during constraint checking (e.g., missing constraint data, evaluation failures)? → A: Mark solution as "unknown" feasibility status with error details in a separate property (solution.feasibility_error)
- Q: What level of detail should violation magnitude reporting provide? → A: Absolute violation value (e.g., constraint requires <=5, got 7, violation = 2)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Get Immediate Feasibility Status (Priority: P1)

After solving an optimization problem on a quantum device, users receive a solution with a clear indication of whether the solution satisfies all original constraints.

**Why this priority**: This is the most critical value proposition. Without knowing if a solution is feasible, users cannot trust quantum results. This addresses the core problem that QUBO conversion loses constraint semantics.

**Independent Test**: Can be fully tested by solving any constrained optimization problem and checking that the returned solution object contains a feasibility status (feasible/infeasible). Delivers immediate actionable feedback to users about solution validity.

**Acceptance Scenarios**:

1. **Given** a quantum optimization problem with linear equality constraints, **When** the solver returns a solution that satisfies all constraints, **Then** the solution is marked as "feasible"
2. **Given** a quantum optimization problem with multiple constraint types, **When** the solver returns a solution that violates at least one constraint, **Then** the solution is marked as "infeasible"
3. **Given** an unconstrained optimization problem, **When** the solver returns a solution, **Then** the solution is marked as "feasible" (no constraints to violate)

---

### User Story 2 - Identify Constraint Violations (Priority: P2)

When a quantum solution is infeasible, users can see exactly which constraints were violated and by how much, enabling them to understand why the solution failed and how to adjust their problem formulation or penalty weights.

**Why this priority**: Knowing that a solution is infeasible is valuable, but understanding which specific constraints failed is essential for debugging and improving the quantum formulation. This enables iterative problem refinement.

**Independent Test**: Can be fully tested by solving a problem with known constraint violations and verifying that the violation report identifies the specific constraints and magnitude of violations. Delivers diagnostic information for problem refinement.

**Acceptance Scenarios**:

1. **Given** an infeasible quantum solution, **When** the user examines the constraint violation report, **Then** each violated constraint is listed with its name/identifier
2. **Given** a linear constraint "x + y <= 5" that is violated by a solution where x=3, y=4, **When** the user checks the violation details, **Then** the violation magnitude is reported (e.g., "violated by 2")
3. **Given** a solution that satisfies some constraints but violates others, **When** the user reviews the violation report, **Then** only the violated constraints appear in the report

---

### User Story 3 - Verify Satisfied Constraints (Priority: P3)

Users can review which constraints were successfully satisfied by the quantum solution, providing confidence in the valid aspects of the result even if other constraints were violated.

**Why this priority**: For complex problems with many constraints, knowing which constraints are satisfied helps users understand partial feasibility and make informed decisions about accepting approximate solutions or adjusting problem formulations.

**Independent Test**: Can be fully tested by solving a problem and verifying that the report lists all satisfied constraints. Delivers transparency into solution quality for complex multi-constraint problems.

**Acceptance Scenarios**:

1. **Given** a feasible quantum solution, **When** the user examines the constraint report, **Then** all constraints are listed as satisfied
2. **Given** a partially feasible solution (some constraints satisfied, some violated), **When** the user reviews satisfied constraints, **Then** each satisfied constraint is clearly marked with its satisfaction status
3. **Given** a constraint that is satisfied exactly at its boundary (e.g., "x <= 5" with x=5), **When** the user checks the report, **Then** the constraint is marked as satisfied

---

### Edge Cases

- What happens when a constraint involves decision variables that were not present in the QUBO formulation (e.g., continuous variables that were discretized)?
- How does the system handle floating-point precision issues when checking constraint satisfaction (e.g., "x == 1.0" when x = 0.999999)?
- What happens when the original model contains indicator constraints or piecewise linear constraints that were transformed during QUBO conversion?
- How does the system report violations for constraints that involve absolute values or other non-linear transformations?
- What happens when the quantum solution contains unmapped or auxiliary variables introduced during QUBO conversion?
- What happens if the original DOcplex model is no longer available or has been modified after solving?
- How does the system handle constraint expressions that cannot be evaluated (e.g., division by zero, undefined operations)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST evaluate each original constraint from the DOcplex model against the quantum solution values
- **FR-002**: System MUST report overall solution feasibility status (feasible/infeasible/unknown)
- **FR-003**: System MUST identify and list all violated constraints with their identifiers
- **FR-004**: System MUST calculate and report the absolute magnitude of constraint violations, showing the difference between the actual value and the constraint bound (e.g., for constraint x<=5 with x=7, report violation magnitude of 2)
- **FR-005**: System MUST identify and list all satisfied constraints
- **FR-006**: System MUST support all DOcplex constraint types including linear equalities, linear inequalities, quadratic constraints, and cardinality constraints
- **FR-007**: System MUST preserve original constraint semantics and bounds from the DOcplex model before QUBO conversion
- **FR-008**: System MUST map quantum solution variable values back to original DOcplex variable names
- **FR-009**: System MUST handle cases where QUBO conversion introduced auxiliary variables (exclude them from constraint checking)
- **FR-010**: System MUST use configurable tolerance for floating-point comparisons when checking constraint satisfaction, with a default value of 1e-6 that users can override via ExecutionConfig
- **FR-011**: Users MUST be able to access the feasibility report through new properties added to the solution object returned by QModel.solve(), including: is_feasible (boolean), violated_constraints (list), satisfied_constraints (list), and constraint_violations (detailed violation information)
- **FR-012**: System MUST complete constraint checking efficiently for problems with up to 1,000 variables and 1,000 constraints
- **FR-013**: System MUST handle constraint checking errors gracefully by marking the solution with "unknown" feasibility status and providing error details in the feasibility_error property, allowing users to still access solution variable values

### Key Entities

- **Constraint**: Represents an original constraint from the DOcplex model, including its type (equality/inequality), expression, bounds, and identifier
- **Solution**: Represents the result from quantum solving, including variable assignments, objective value, and feasibility information exposed as properties: is_feasible (boolean or "unknown" string), violated_constraints (list of constraint identifiers), satisfied_constraints (list of constraint identifiers), constraint_violations (list of ConstraintViolation objects), and feasibility_error (optional string containing error details when checking fails)
- **ConstraintViolation**: Represents a single constraint violation, including the constraint identifier, expected bound (the constraint limit), actual value (the computed value from the solution), and absolute magnitude of violation (the numeric difference between actual and bound)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can determine solution feasibility immediately upon receiving quantum results without manual constraint checking
- **SC-002**: System correctly identifies feasibility status for 100% of test cases across all supported constraint types
- **SC-003**: Constraint violation reports enable users to identify and fix problem formulation issues in 90% fewer iterations compared to manual debugging
- **SC-004**: Constraint checking completes in under 5 seconds for problems with up to 1,000 constraints and 1,000 variables
- **SC-005**: Users report increased confidence in quantum results, with 80% of users indicating they can now trust solution validity without external verification

## Assumptions *(mandatory)*

- The quantum solver has already completed execution and returned variable assignments
- The original DOcplex model structure and constraints are available at the time of solution checking
- Users are familiar with the DOcplex constraint syntax and can interpret constraint identifiers
- The mapping between QUBO variables and original DOcplex variables is maintained by the existing QModel infrastructure
- Standard floating-point precision (64-bit) is sufficient for constraint checking in most use cases
- Constraint checking is performed on the classical computer after quantum execution (not on quantum hardware)

## Out of Scope *(mandatory)*

- Automatic correction or repair of infeasible solutions
- Suggestions for adjusting penalty weights to improve feasibility
- Re-solving the problem with different parameters based on constraint violations
- Visualization or graphical display of constraint violations
- Constraint relaxation or modification to make infeasible solutions feasible
- Comparison of feasibility across multiple solution candidates
- Integration with external constraint satisfaction solvers
- Performance optimization for extremely large-scale problems (>10,000 constraints)

## Dependencies *(optional)*

- Existing QModel constraint storage and QUBO conversion infrastructure
- DOcplex model API for accessing constraint expressions and bounds
- Variable mapping maintained during QUBO conversion process
