from qiskit_optimization.converters import QuadraticProgramToQubo


def get_solution_from_counts(model, optimal_counts, interpret=False,
                             interpreter: QuadraticProgramToQubo=None):
    """
    Extracts the best solution from the optimal parameter counts obtained
    from a quantum algorithm's execution.

    This function takes the model and the counts of possible solutions
    (bitstrings) from a quantum run and extracts the best solution by
    selecting the most frequent bitstring and computing
    its objective value.

    Parameters
    ----------
    model: Model
        The optimization model that was solved, typically represented as a
        QUBO (Quadratic Unconstrained Binary Optimization) problem.
    optimal_counts: dict
        A dictionary of bitstrings (represented as strings of 0s and 1s)
        and their corresponding frequencies or counts from the quantum
        measurement results.
    interpret: bool
        Whether the provided results should be interpreted from expanded
        binary variables into the original model's variables.
    interpreter: QuadraticProgramToQubo
        The interpreter to interpret the binary variables.

    Returns
    -------
    dict
        A dictionary containing the best solution with the following keys:
        - 'solution': A dictionary of variable names mapped to their binary
          values (0 or 1) representing the optimal solution.
        - 'objective': The computed objective value of the best solution.
    """
    best_solution, best_count = max(optimal_counts.items(),
                                    key=lambda x: x[1])
    print(f'raw best_solution: {list(best_solution)}')
    if interpret:
        if not interpreter:
            raise ValueError("Missing interpreter")
        best_solution = [int(x) for x in best_solution]
        best_solution = interpreter.interpret(best_solution)
    print(f'interpreted best_solution: {list(best_solution)}')
    values = {}
    for i, var in enumerate(model.iter_variables()):
        values[var.name] = int(best_solution[i])

    obj_value = 0
    linear_terms = model.get_objective_expr().iter_terms()
    quadratic_terms = list(model.get_objective_expr().iter_quad_triplets())

    if len(linear_terms) > 0:
        for t in linear_terms:
            obj_value += (values[t[0].name] * t[1])

    if len(quadratic_terms) > 0:
        for t in quadratic_terms:
            obj_value += (values[t[0].name] * values[t[1].name] * t[2])

    solution = {'objective': obj_value, 'solution': values}
    return solution


def calculate_energy(counts, shots, algorithm_instance):
    """
    Calculates the energy (or cost function value) of a quantum solution.

    This function computes the average energy of the quantum measurement
    results based on the provided counts (bitstrings and their frequencies).
    The energy is computed by evaluating the objective function for each
    sampled bitstring and taking the weighted average based on the counts.

    Parameters
    ----------
    counts : dict
        A dictionary of bitstrings (represented as strings of 0s and 1s) and
        their corresponding frequencies (or counts) from the quantum
        measurement results.
    shots : int
        The total number of measurement shots, used to normalize the energy.
    algorithm_instance : Algorithm
        The instance of the quantum algorithm (e.g., QAOA or VQE) being used.
        This instance contains the QUBO model, which is used to evaluate the
        objective function of the bitstrings.

    Returns
    -------
    float
        The average energy (or cost function value) of the quantum solution,
        normalized by the total number of shots.
    """
    energy = 0
    for sample, count in counts.items():
        sample = [int(n) for n in sample]
        energy += count * algorithm_instance.qubo.objective.evaluate(sample)

    algorithm_instance.iteration += 1

    return energy / shots
