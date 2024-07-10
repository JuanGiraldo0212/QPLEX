from scipy.optimize import minimize
from qplex.algorithms import QAOA, VQE
from qplex.solvers.base_solver import Solver


def ggaem_workflow(model, solver: Solver, verbose: bool, shots: int,
                   algorithm: str, optimizer: str, max_iter: int,
                   tolerance: float, ansatz: str, p: int, layers: int,
                   seed: int, penalty: float):
    current_algorithm = None
    if algorithm == "qaoa":
        current_algorithm = QAOA(model, solver, verbose, shots=shots, p=p,
                                 penalty=penalty, seed=seed)
    elif algorithm == "vqe":
        current_algorithm = VQE(model, solver, verbose, shots=shots,
                                layers=layers, penalty=penalty, seed=seed,
                                ansatz=ansatz)

    starting_point = current_algorithm.get_starting_point()
    optimization_result = minimize(fun=current_algorithm.cost_function,
                                   x0=starting_point, method=optimizer,
                                   tol=tolerance,
                                   options={'maxiter': max_iter})
    optimal_params = optimization_result.x
    qc = current_algorithm.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts


def get_ggaem_solution(model, optimal_counts):
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
                    values[t[0].name] * values[t[1].name] * t[
                2])
    solution = {'objective': obj_value, 'solution': values}
    return solution
