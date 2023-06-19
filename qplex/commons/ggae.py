from qplex.algorithms import QAOA
from scipy.optimize import minimize
from qplex.solvers.base_solver import Solver


def ggae_workflow(model, solver: Solver, algorithm: str = "qaoa"):
    current_algorithm = None
    if algorithm == "qaoa":
        current_algorithm = QAOA(model, solver, p=3)

    starting_point = current_algorithm.get_starting_point()
    optimization_result = minimize(fun=current_algorithm.cost_function, x0=starting_point, method='COBYLA', tol=1e-10,
                                   options={'maxiter': 2000})
    optimal_params = optimization_result.x
    qc = current_algorithm.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts