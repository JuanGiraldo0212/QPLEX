from scipy.optimize import minimize
from qplex.algorithms import QAOA, VQE
from qplex.solvers.base_solver import Solver
from qplex.commons.workflow_utils import calculate_energy

import numpy as np


def ggaem_workflow(model, solver: Solver, options):
    """
    Runs the GGAEM (Generalized Gate-Based Algorithm Execution Manager)
    workflow.

    Parameters
    ----------
    model: Model
        The optimization model to be solved.
    solver: Solver
        The solver to be used for solving the model.
    options: Options
        A dictionary containing configuration options for the workflow.

    Returns
    -------
    dict
        A dictionary of optimal parameter counts.
    """
    shots = options['shots']
    algorithm = options['algorithm']
    penalty = options['penalty']
    seed = options['seed']
    verbose = options['verbose']
    optimizer = options['optimizer']
    callback = options['callback']
    max_iter = options['max_iter']
    tolerance = options['tolerance']

    algorithm_instance = None
    if algorithm == "qaoa":
        algorithm_instance = QAOA(model, p=options['p'], penalty=penalty,
                                  seed=seed)
    elif algorithm == "vqe":
        algorithm_instance = VQE(model, layers=options['layers'],
                                 penalty=penalty,
                                 seed=seed, ansatz=options['ansatz'])

    # Remove parameters declared in the OpenQASM3 base string in preparation
    # for the managed workflow
    algorithm_instance.remove_parameters()

    def cost_function(params: np.ndarray) -> float:
        """
        Defines the cost function to be used for the classical optimization
        routine.

        This method calculates the cost based on the given parameters by
        updating the quantum circuit, solving it, and evaluating the energy.

        Parameters
        ----------
        params : np.ndarray
            The new set of parameters for the circuit.

        Returns
        -------
        float
            The cost for the current parameters.
        """
        qc = algorithm_instance.update_params(params)
        counts = solver.solve(qc)
        cost = calculate_energy(counts, shots, algorithm_instance)
        if verbose:
            print(f'\nCost = {cost}')
        return cost

    starting_point = algorithm_instance.get_starting_point()
    optimization_result = minimize(fun=cost_function,
                                   x0=starting_point, method=optimizer,
                                   callback=callback,
                                   tol=tolerance,
                                   options={'maxiter': max_iter})
    optimal_params = optimization_result.x
    qc = algorithm_instance.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts