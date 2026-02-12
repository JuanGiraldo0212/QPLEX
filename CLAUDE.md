# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QPLEX is a Python library for combinatorial optimization that executes models on both classical and quantum devices. It extends DOcplex (IBM's optimization modeling library) and supports three quantum providers: IBM Quantum (gate-based), D-Wave (quantum annealing), and AWS Braket (gate-based).

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .  # Development mode

# Run all tests (includes coverage check, 80% branch minimum enforced)
pytest

# Run a single test file
pytest tests/test_qmodel.py

# Run a single test class or method
pytest tests/test_qmodel.py::TestQModel::test_method_name

# Run tests without coverage
pytest --no-cov tests/
```

Python >= 3.10 required.

## Architecture

### Core Flow

`QModel` (extends `docplex.mp.model.Model`) is the single entry point. Users define optimization problems using DOcplex syntax, then call `solve("quantum", config)` to run on quantum hardware. The flow is:

```
QModel.solve() → QUBO conversion → AlgorithmFactory → SolverFactory → Workflow execution
```

### Key Abstractions

**QModel** (`qplex/model/qmodel.py`): Central class. Converts DOcplex models to QUBO form, routes between classical (DOcplex CPLEX) and quantum solving. `ExecutionConfig` dataclass controls all quantum execution parameters (provider, backend, algorithm, optimizer settings, shots).

**Algorithms** (`qplex/algorithms/`): Abstract `Algorithm` base class with `create_circuit()`, `update_params()`, `get_starting_point()`. Implementations: `QAOA` (parameterized problem+mixer layers) and `VQE` (variational ansatz with RY+CNOT layers). All circuits are represented as **OpenQASM 3.0 strings** — this is the universal IR between algorithms and solvers.

**Mixers** (`qplex/algorithms/mixers/`): Constraint-preserving QAOA mixing operators. `MixerFactory` auto-selects based on constraint analysis:
- `StandardMixer` — unconstrained (RX rotations)
- `CardinalityMixer` — equality sum constraints (XY-mixing)
- `PartitionMixer` — partition constraints (SWAP+RZ)
- `InequalityMixer` — inequality constraints (controlled rotations)
- `CompositeMixer` — chains multiple mixers for multi-constraint problems

**Solvers** (`qplex/solvers/`): Abstract `Solver` base class. Each provider has its own solver that converts OpenQASM3 to provider-native format:
- `IBMQSolver` — converts to Qiskit circuits, uses SamplerV2
- `DWaveSolver` — handles BQM/DQM/CQM formulations (no gate-based circuits)
- `BraketSolver` — converts to Braket Program (notably replaces `cx` → `cnot`)

**Workflows** (`qplex/workflows/`): Manage the classical optimization loop that tunes quantum circuit parameters:
- `GGAEWorkflow` — general gate-based execution (used by Braket and IBMQ non-session)
- `IBMSessionWorkflow` — keeps IBM Quantum Runtime session open for efficiency

**Factories** (`qplex/commons/`): `SolverFactory`, `AlgorithmFactory` create instances based on config. The algorithm `"qao-ansatz"` triggers `MixerFactory` for constraint-aware mixer selection.

**Utilities** (`qplex/utils/`):
- `circuit_utils.py` — substitutes theta parameter placeholders in OpenQASM3 strings
- `model_utils.py` — analyzes DOcplex model constraints to determine types (cardinality, partition, inequality)
- `workflow_utils.py` — extracts solutions from measurement counts, calculates energy

## Code Conventions

- PEP 8 style, PEP 257 NumPy-style docstrings (module, class, and method level)
- Tests use pytest with class-based organization (`TestClassName`) and Arrange/Act/Assert pattern
- Heavy use of `unittest.mock` for quantum backend mocking in tests
- Branch naming: `feature/`, `bugfix/`, `docs/`
- Squash-and-merge PR strategy

## Active Technologies
- Python >= 3.10 (per pyproject.toml) + DOcplex (docplex.mp), dataclasses, typing (001-feasibility-checking)
- N/A (in-memory constraint evaluation) (001-feasibility-checking)

## Recent Changes
- 001-feasibility-checking: Added Python >= 3.10 (per pyproject.toml) + DOcplex (docplex.mp), dataclasses, typing
