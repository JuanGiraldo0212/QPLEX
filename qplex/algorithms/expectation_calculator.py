def measure_circuit(state: str, basis: str) -> str:
    """
    Converts a quantum circuit state to its measurement circuit form
    based on the given basis.

    Parameters
    ----------
    state : str
        The initial state of the quantum circuit in OpenQASM2 format.
    basis : str
        The measurement basis, where 'I', 'X', 'Y', and 'Z' specify the
        basis for each qubit.

    Returns
    -------
    str
        The OpenQASM3 string for the measurement circuit.
    """
    measure_qc = state
    for qb, base in enumerate(basis):
        if base == 'I' or base == 'Z':
            pass
        elif base == 'X':
            measure_qc += f"h q[{qb}];"
        elif base == 'Y':
            measure_qc += f"sdg q[{qb}];"
            measure_qc += f"h q[{qb}];"

    return measure_qc


def calculate_basis_exp_val(state: str, basis: str, solver) -> float:
    """
    Calculates the expectation value for a given basis by measuring the
    circuit.

    Parameters
    ----------
    state : str
        The initial state of the quantum circuit in OpenQASM2 format.
    basis : str
        The measurement basis, where 'I', 'X', 'Y', and 'Z' specify the
        basis for each qubit.
    solver : Solver
        The solver to execute the quantum circuit and get the
        measurement results.

    Returns
    -------
    float
        The expectation value for the given basis.
    """
    exp_circuit = measure_circuit(state, basis)
    counts = solver.solve(exp_circuit)
    exp = 0.0
    for key, counts in counts.items():
        power = sum([int(bit) for i, bit in enumerate(key) if basis[i] != "I"])
        exp += (-1) ** power * counts
    return exp / solver.shots


def compute_expectation_value(state: str, operator, solver) -> float:
    """
    Computes the expectation value of an operator on a given quantum state.

    Parameters
    ----------
    state : str
        The initial state of the quantum circuit in OpenQASM2 format.
    operator : Operator
        The operator to compute the expectation value of. Should have a
        method
        `primitive.to_list()` returning the basis and coefficients.
    solver : Solver
        The solver to execute the quantum circuit and get the
        measurement results.

    Returns
    -------
    float
        The computed expectation value.
    """
    energy = 0
    n_qubits = state
    for basis, coeff in operator.primitive.to_list():
        if basis.count('I') != n_qubits:
            curr_energy = coeff * calculate_basis_exp_val(state, basis, solver)
            energy += curr_energy
        else:
            energy += coeff
    return energy.real
