from qiskit_optimization.converters import QuadraticProgramToQubo


def get_solution_from_counts(model, optimal_counts, interpret=False,
                             interpreter: QuadraticProgramToQubo = None):
    """
    Extracts the best solution from the optimal parameter counts obtained
    from a quantum algorithm's execution.

    Parameters
    ----------
    model: Model
        The optimization model that was solved.
    optimal_counts: dict
        A dictionary of bitstrings (represented as strings of 0s and 1s)
        and their corresponding counts from the quantum measurement results.
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
    if interpret and not interpreter:
        raise ValueError("Missing interpreter")

    best_solution, _ = max(optimal_counts.items(),
                           key=lambda x: x[1])
    print(f'raw best_solution: {list(best_solution)}')
    if interpret:
        best_solution = [int(x) for x in best_solution]
        best_solution = interpreter.interpret(best_solution)
    print(f'interpreted best_solution: {list(best_solution)}')

    values = {}
    for i, var in enumerate(model.iter_variables()):
        values[var.name] = int(best_solution[i])

    obj_value = compute_objective(model, values)

    solution = {'objective': obj_value, 'solution': values}
    return solution


def compute_objective(model, values):
    """
        Computes the objective value of a solution for the provided model.

        Parameters
        ----------
        model : Model
            The optimization model that contains the linear and quadratic
            terms.
        values : dict
            A dictionary mapping variable names (as strings) to their values
            (as integers, typically 0 or 1).

        Returns
        -------
        float
            The computed objective value for the given solution.
    """
    obj_value = 0
    obj_expr = model.get_objective_expr()

    # Add linear terms
    for var, coeff in obj_expr.iter_terms():
        obj_value += values[var.name] * coeff

    # Add quadratic terms
    for var1, var2, coeff in obj_expr.iter_quad_triplets():
        obj_value += values[var1.name] * values[var2.name] * coeff

    return obj_value


def calculate_energy(counts, shots, algorithm_instance):
    """
    Calculates the energy (or cost function value) of a quantum solution.

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
