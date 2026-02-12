# Tasks: Solution Feasibility Checking

**Input**: Design documents from `/specs/001-feasibility-checking/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included (pytest with 80% branch coverage required per pyproject.toml)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

QPLEX uses single project structure with `qplex/` source and `tests/` at repository root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure - no new files needed (existing QPLEX project)

- [ ] T001 Review existing ModelSolution dataclass in qplex/model/qmodel.py
- [ ] T002 Review existing ExecutionConfig dataclass in qplex/model/execution_config.py
- [ ] T003 [P] Review existing constraint utilities in qplex/utils/model_utils.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Add feasibility_tolerance field to ExecutionConfig in qplex/model/execution_config.py
- [ ] T005 [P] Create FeasibilityResult dataclass in qplex/model/feasibility.py
- [ ] T006 [P] Create ConstraintViolation dataclass in qplex/model/feasibility.py
- [ ] T007 [P] Create constraint evaluation helper functions in qplex/utils/feasibility_utils.py
- [ ] T008 Create FeasibilityChecker class skeleton in qplex/model/feasibility.py
- [ ] T009 Implement FeasibilityChecker.__init__() method in qplex/model/feasibility.py
- [ ] T010 Implement constraint identification helper in qplex/utils/feasibility_utils.py
- [ ] T011 [P] Add unit test for feasibility_tolerance validation in tests/model/test_execution_config.py
- [ ] T012 [P] Add unit tests for constraint evaluation helpers in tests/utils/test_feasibility_utils.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Get Immediate Feasibility Status (Priority: P1) 🎯 MVP

**Goal**: Users receive a solution with a clear indication of whether it satisfies all original constraints (feasible/infeasible/unknown)

**Independent Test**: Solve a quantum optimization problem and verify solution.is_feasible is True/False/"unknown"

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Write test for feasible solution in tests/model/test_feasibility.py
- [ ] T014 [P] [US1] Write test for infeasible solution in tests/model/test_feasibility.py
- [ ] T015 [P] [US1] Write test for unconstrained problem in tests/model/test_feasibility.py
- [ ] T016 [P] [US1] Write test for "unknown" error handling in tests/model/test_feasibility.py

### Implementation for User Story 1

- [ ] T017 [US1] Extend ModelSolution with is_feasible field in qplex/model/qmodel.py
- [ ] T018 [US1] Implement basic constraint evaluation in FeasibilityChecker._evaluate_constraint() in qplex/model/feasibility.py
- [ ] T019 [US1] Implement FeasibilityChecker.check() main logic for boolean feasibility in qplex/model/feasibility.py
- [ ] T020 [US1] Add error handling for "unknown" status in FeasibilityChecker.check() in qplex/model/feasibility.py
- [ ] T021 [US1] Integrate FeasibilityChecker into QModel.solve() for quantum method in qplex/model/qmodel.py
- [ ] T022 [US1] Populate ModelSolution.is_feasible from FeasibilityResult in qplex/model/qmodel.py
- [ ] T023 [P] [US1] Add integration test for QModel.solve() with feasibility checking in tests/model/test_qmodel.py
- [ ] T024 [P] [US1] Add test for floating-point tolerance handling in tests/model/test_feasibility.py

**Checkpoint**: At this point, User Story 1 should be fully functional - users can check solution.is_feasible

---

## Phase 4: User Story 2 - Identify Constraint Violations (Priority: P2)

**Goal**: Users can see exactly which constraints were violated and by how much (magnitude, bounds, actual values)

**Independent Test**: Solve a problem with known violations and verify solution.constraint_violations contains detailed violation info

### Tests for User Story 2

- [ ] T025 [P] [US2] Write test for ConstraintViolation magnitude calculation in tests/model/test_feasibility.py
- [ ] T026 [P] [US2] Write test for violated_constraints list population in tests/model/test_feasibility.py
- [ ] T027 [P] [US2] Write test for constraint_violations details in tests/model/test_feasibility.py
- [ ] T028 [P] [US2] Write test for partial feasibility (some satisfied, some violated) in tests/model/test_feasibility.py

### Implementation for User Story 2

- [ ] T029 [US2] Extend ModelSolution with violated_constraints field in qplex/model/qmodel.py
- [ ] T030 [US2] Extend ModelSolution with constraint_violations field in qplex/model/qmodel.py
- [ ] T031 [US2] Implement violation magnitude calculation in qplex/utils/feasibility_utils.py
- [ ] T032 [US2] Extend FeasibilityChecker._evaluate_constraint() to create ConstraintViolation objects in qplex/model/feasibility.py
- [ ] T033 [US2] Populate violated_constraints list in FeasibilityChecker.check() in qplex/model/feasibility.py
- [ ] T034 [US2] Populate constraint_violations list in FeasibilityChecker.check() in qplex/model/feasibility.py
- [ ] T035 [US2] Update QModel.solve() to populate violation fields in ModelSolution in qplex/model/qmodel.py
- [ ] T036 [P] [US2] Add test for different constraint types (equality, inequality_le, inequality_ge) in tests/model/test_feasibility.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work - users can see is_feasible AND violation details

---

## Phase 5: User Story 3 - Verify Satisfied Constraints (Priority: P3)

**Goal**: Users can review which constraints were successfully satisfied for transparency into solution quality

**Independent Test**: Solve a problem and verify solution.satisfied_constraints contains all satisfied constraint IDs

### Tests for User Story 3

- [ ] T037 [P] [US3] Write test for satisfied_constraints list when all constraints satisfied in tests/model/test_feasibility.py
- [ ] T038 [P] [US3] Write test for satisfied_constraints list with partial feasibility in tests/model/test_feasibility.py
- [ ] T039 [P] [US3] Write test for boundary case constraints (exactly at limit) in tests/model/test_feasibility.py

### Implementation for User Story 3

- [ ] T040 [US3] Extend ModelSolution with satisfied_constraints field in qplex/model/qmodel.py
- [ ] T041 [US3] Populate satisfied_constraints list in FeasibilityChecker.check() in qplex/model/feasibility.py
- [ ] T042 [US3] Update QModel.solve() to populate satisfied_constraints in ModelSolution in qplex/model/qmodel.py
- [ ] T043 [P] [US3] Add integration test for complete feasibility report in tests/model/test_qmodel.py

**Checkpoint**: All user stories should now be independently functional - complete feasibility checking feature

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and ensure production readiness

- [ ] T044 [P] Export ConstraintViolation in qplex/__init__.py
- [ ] T045 [P] Export FeasibilityResult in qplex/__init__.py
- [ ] T046 [P] Add NumPy-style docstrings to all new functions in qplex/model/feasibility.py
- [ ] T047 [P] Add NumPy-style docstrings to all helpers in qplex/utils/feasibility_utils.py
- [ ] T048 Add feasibility_error field to ModelSolution in qplex/model/qmodel.py
- [ ] T049 Update QModel.solve() to handle classical method (leave feasibility None) in qplex/model/qmodel.py
- [ ] T050 [P] Add edge case tests for missing variables in tests/model/test_feasibility.py
- [ ] T051 [P] Add edge case tests for transformed constraints in tests/model/test_feasibility.py
- [ ] T052 [P] Add edge case tests for auxiliary QUBO variables in tests/model/test_feasibility.py
- [ ] T053 [P] Add test for custom tolerance values in tests/model/test_feasibility.py
- [ ] T054 [P] Add performance test for 1,000 constraints in tests/model/test_feasibility.py
- [ ] T055 Validate all tests achieve 80% branch coverage via pytest
- [ ] T056 [P] Update README.md with feasibility checking examples
- [ ] T057 Code review and refactoring pass across all new code
- [ ] T058 Run quickstart.md validation scenarios

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
  - Delivers: Basic feasibility status (is_feasible)
  - MVP milestone: Just US1 provides core value

- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 fields but independently testable
  - Delivers: Violation details (violated_constraints, constraint_violations)
  - Adds diagnostic information without breaking US1

- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Builds on US1 fields but independently testable
  - Delivers: Satisfied constraints list (satisfied_constraints)
  - Adds transparency without breaking US1 or US2

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Helper functions before checker methods
- Checker implementation before QModel integration
- ModelSolution fields before population logic
- Core implementation before edge cases
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**: All 3 tasks marked [P] can run in parallel

**Phase 2 (Foundational)**:
- T005, T006, T007, T010 can run in parallel (different entities/files)
- T011, T012 can run in parallel (different test files)

**Phase 3 (User Story 1)**:
- T013-T016 (tests) can run in parallel (same file, different test methods)
- T023, T024 (tests) can run in parallel (different test files)

**Phase 4 (User Story 2)**:
- T025-T028 (tests) can run in parallel (different test methods)
- T036 (test) can run in parallel with others

**Phase 5 (User Story 3)**:
- T037-T039 (tests) can run in parallel (different test methods)
- T043 (test) can run independently

**Phase 6 (Polish)**:
- T044-T047 (exports and docs) can run in parallel (different files)
- T050-T054 (edge case tests) can run in parallel (different test methods)
- T056-T057 (docs and review) can run in parallel

**Across User Stories**: Once Phase 2 completes, US1, US2, and US3 can be worked on in parallel by different developers

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test for feasible solution in tests/model/test_feasibility.py"
Task: "Write test for infeasible solution in tests/model/test_feasibility.py"
Task: "Write test for unconstrained problem in tests/model/test_feasibility.py"
Task: "Write test for 'unknown' error handling in tests/model/test_feasibility.py"

# After tests written and failing, launch implementation tasks in sequence:
Task: "Extend ModelSolution with is_feasible field in qplex/model/qmodel.py"
Task: "Implement basic constraint evaluation in FeasibilityChecker._evaluate_constraint()"
# ... etc
```

---

## Parallel Example: User Story 2

```bash
# Launch all tests for User Story 2 together:
Task: "Write test for ConstraintViolation magnitude calculation"
Task: "Write test for violated_constraints list population"
Task: "Write test for constraint_violations details"
Task: "Write test for partial feasibility"

# Launch helper implementations in parallel:
Task: "Implement violation magnitude calculation in qplex/utils/feasibility_utils.py"
# ... etc
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (review existing code - T001-T003)
2. Complete Phase 2: Foundational (T004-T012) - CRITICAL, blocks all stories
3. Complete Phase 3: User Story 1 (T013-T024)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Users can now check if quantum solutions are feasible (core value delivered!)

**MVP Scope**: Just US1 provides the essential value proposition - knowing if a quantum solution is feasible.

### Incremental Delivery

1. Complete Setup + Foundational (T001-T012) → Foundation ready
2. Add User Story 1 (T013-T024) → Test independently → **Deploy/Demo (MVP!)**
   - Value: Users can check solution.is_feasible
3. Add User Story 2 (T025-T036) → Test independently → Deploy/Demo
   - Value: Users can see violation details (which constraints failed, by how much)
4. Add User Story 3 (T037-T043) → Test independently → Deploy/Demo
   - Value: Users can see which constraints passed (transparency)
5. Add Polish (T044-T058) → Production ready
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T012)
2. Once Foundational is done:
   - Developer A: User Story 1 (T013-T024)
   - Developer B: User Story 2 (T025-T036)
   - Developer C: User Story 3 (T037-T043)
3. Stories complete and integrate independently
4. Team reconvenes for Polish phase (T044-T058)

**Note**: In practice, sequential implementation (P1 → P2 → P3) is recommended for a single developer to ensure each story is fully validated before proceeding.

---

## Task Summary

**Total Tasks**: 58

**Breakdown by Phase**:
- Phase 1 (Setup): 3 tasks
- Phase 2 (Foundational): 9 tasks
- Phase 3 (User Story 1 - P1): 12 tasks
- Phase 4 (User Story 2 - P2): 12 tasks
- Phase 5 (User Story 3 - P3): 7 tasks
- Phase 6 (Polish): 15 tasks

**Tasks by User Story**:
- US1 (P1 - MVP): 12 tasks (8 implementation + 4 test)
- US2 (P2): 12 tasks (7 implementation + 5 test)
- US3 (P3): 7 tasks (4 implementation + 3 test)
- Foundational: 9 tasks (blocks all stories)
- Shared/Polish: 18 tasks (setup + polish)

**Parallel Opportunities**: 37 tasks marked [P] can run in parallel within their phase

**MVP Scope**: Phases 1-3 (T001-T024) = 24 tasks for minimum viable feature

**Test Coverage**: 80% branch coverage enforced by pytest configuration

---

## Notes

- [P] tasks = different files or independent test methods, no dependencies
- [Story] label (US1, US2, US3) maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD approach)
- Commit after each task or logical group (e.g., all tests for a story)
- Stop at any checkpoint to validate story independently
- Follow existing QPLEX patterns: dataclasses, NumPy-style docstrings, utils modules
- "Define errors out of existence": return "unknown" status instead of raising exceptions
- ModelSolution extension maintains backward compatibility (all new fields have defaults)
