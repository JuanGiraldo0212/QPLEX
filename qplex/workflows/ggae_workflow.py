import scipy as sp
import docplex.mp.model

import qplex.commons as commons
import qplex.utils.workflow_utils as workflow_utils
from qplex.model.execution_config import ExecutionConfig
from qplex.solvers.base_solver import Solver

import numpy as np


def ggae_workflow(model: docplex.mp.model.Model, solver: Solver, options:
ExecutionConfig):
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
    shots = options.shots
    verbose = options.verbose
    optimizer = options.optimizer
    callback = options.callback
    max_iter = options.max_iter
    tolerance = options.tolerance

    algorithm_config = commons.algorithm_factory.AlgorithmConfig(
        algorithm=commons.algorithm_factory.AlgorithmType(options.algorithm),
        penalty=options.penalty,
        seed=options.seed,
        p=options.p,
        mixer=options.mixer,
        layers=options.layers,
        ansatz=options.ansatz
    )
    algorithm_instance = (commons.algorithm_factory.AlgorithmFactory
                          .get_algorithm(model,
                                         algorithm_config))

    algorithm_instance.remove_parameters()

    def cost_function(params: np.ndarray) -> float: # pragma: no cover
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
        cost = workflow_utils.calculate_energy(counts, shots,
                                               algorithm_instance)
        if verbose:
            print(f'\nCost = {cost}')
        return cost

    starting_point = algorithm_instance.get_starting_point()
    optimization_result = sp.optimize.minimize(fun=cost_function,
                                               x0=starting_point,
                                               method=optimizer,
                                               callback=callback,
                                               tol=tolerance,
                                               options={'maxiter': max_iter})
    optimal_params = optimization_result.x
    qc = algorithm_instance.update_params(optimal_params)
    opt_counts = solver.solve(qc)

    return opt_counts
