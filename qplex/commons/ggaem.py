from scipy.optimize import minimize
from qplex.algorithms import QAOA, VQE
from qplex.solvers.base_solver import Solver


def ggaem_workflow(model, solver: Solver, verbose: bool, shots: int,
                   algorithm: str, optimizer: str, max_iter: int,
                   tolerance: float, ansatz: str, p: int, layers: int,
                   seed: int, penalty: float):
    """
    Runs the GGAEM (Generalized Grover Adaptive Execution Method) workflow.

    Parameters
    ----------
    model: Model
        The optimization model to be solved.
    solver: Solver
        The solver to be used for solving the model.
    verbose: bool
        If True, enables verbose output.
    shots: int
        The number of shots for quantum execution.
    algorithm: str
        The quantum algorithm to use ('qaoa' or 'vqe').
    optimizer: str
        The classical optimizer to use.
    max_iter: int
        The maximum number of iterations for the optimizer.
    tolerance: float
        The tolerance for the optimizer.
    ansatz: str
        The ansatz to use in the quantum algorithm.
    p: int
        The depth of the ansatz for QAOA.
    layers: int
        The number of layers in the ansatz for VQE.
    seed: int
        The seed for the random number generator.
    penalty: float
        The penalty factor for the QUBO conversion.

    Returns
    -------
    dict
        A dictionary of optimal parameter counts.
    """
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
                    values[t[0].name] * values[t[1].name] * t[
                2])
    solution = {'objective': obj_value, 'solution': values}
    return solution
