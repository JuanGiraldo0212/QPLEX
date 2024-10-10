def get_solution_from_counts(model, optimal_counts):
    """
    Extracts the best solution from the optimal parameter counts.

    Parameters
    ----------
    model: Model
        The optimization model that was solved.
    optimal_counts: dict
        A dictionary of optimal parameter counts.

    Returns
    -------
    dict
        A dictionary containing the best solution and its objective value.
    """
    best_solution, best_count = max(optimal_counts.items(),
                                    key=lambda x: x[1])
    values = {}
    for i, var in enumerate(model.iter_variables()):
        values[var.name] = int(best_solution[i])
    obj_value = 0
    linear_terms = model.get_objective_expr().iter_terms()
    quadratic_terms = list(model.get_objective_expr(

    ).iter_quad_triplets())
    if len(linear_terms) > 0:
        for t in linear_terms:
            obj_value += (values[t[0].name] * t[1])
    if len(quadratic_terms) > 0:
        for t in quadratic_terms:
            obj_value += (
                    values[t[0].name] * values[t[1].name] * t[2])
    solution = {'objective': obj_value, 'solution': values}
    return solution


def calculate_energy(counts, shots, algorithm_instance):
    energy = 0
    for sample, count in counts.items():
        sample = [int(n) for n in sample]
        energy += count * algorithm_instance.qubo.objective.evaluate(
            sample)
    algorithm_instance.iteration += 1
    return energy / shots
