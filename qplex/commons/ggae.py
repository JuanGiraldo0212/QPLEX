from qplex.algorithms import QAOA, VQE
from scipy.optimize import minimize
from qplex.solvers.base_solver import Solver


def ggae_workflow(model, solver: Solver, shots, algorithm: str = "qaoa"):
    current_algorithm = None
    if algorithm == "qaoa":
        current_algorithm = QAOA(model, solver, p=3, shots=shots)
    elif algorithm == "vqe":
        current_algorithm = VQE(model, solver, layers=3, shots=shots)

    starting_point = current_algorithm.get_starting_point()
    optimization_result = minimize(fun=current_algorithm.cost_function, x0=starting_point, method='COBYLA', tol=1e-6,
                                   options={'maxiter': 500})
    optimal_params = optimization_result.x
    qc = current_algorithm.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts
