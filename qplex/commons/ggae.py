from qplex.algorithms import QAOA, VQE
from scipy.optimize import minimize
from qplex.solvers.base_solver import Solver


def ggae_workflow(model, solver: Solver, shots: int, algorithm: str, optimizer: str, max_iter: int, tolerance: float,
                  ansatz: str, p: int, layers: int, seed: int, penalty: float):
    current_algorithm = None
    if algorithm == "qaoa":
        current_algorithm = QAOA(model, solver, p=p, shots=shots, penalty=penalty, seed=seed)
    elif algorithm == "vqe":
        current_algorithm = VQE(model, solver, layers=layers, shots=shots, penalty=penalty, seed=seed, ansatz=ansatz)

    starting_point = current_algorithm.get_starting_point()
    optimization_result = minimize(fun=current_algorithm.cost_function, x0=starting_point, method=optimizer,
                                   tol=tolerance, options={'maxiter': max_iter})
    optimal_params = optimization_result.x
    qc = current_algorithm.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts
