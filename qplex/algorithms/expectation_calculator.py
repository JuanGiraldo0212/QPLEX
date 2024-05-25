def measure_circuit(state: str, basis: str):
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


def calculate_basis_exp_val(state: str, basis: str, solver):
    exp_circuit = measure_circuit(state, basis)
    counts = solver.solve(exp_circuit)
    exp = 0.0
    for key, counts in counts.items():
        power = sum([int(bit) for i, bit in enumerate(key) if basis[i] != "I"])
        exp += (-1) ** power * counts
    return exp / solver.shots


def compute_expectation_value(state: str, operator, solver):
    energy = 0
    n_qubits = state
    for basis, coeff in operator.primitive.to_list():
        if basis.count('I') != n_qubits:
            curr_energy = coeff * calculate_basis_exp_val(state, basis, solver)
            energy += curr_energy
        else:
            energy += coeff
    return energy.real
