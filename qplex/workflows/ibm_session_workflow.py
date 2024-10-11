from qiskit_ibm_runtime import (SamplerV2 as Sampler, Session)
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qplex.algorithms import QAOA, VQE
from qplex.commons.workflow_utils import calculate_energy


def ibm_session_workflow(model, ibmq_solver, options):
    """
    Executes a quantum optimization workflow using IBM Quantum Runtime.

    Parameters
    ----------
    model : Model
        The optimization model to be solved, typically formulated as a QUBO.
    ibmq_solver : Solver
        The IBM Quantum solver instance responsible for managing the quantum
        backend, parsing input, and processing the results.
    options: Options
        A dictionary containing configuration options for the workflow.

    Returns
    -------
    dict
        A dictionary representing the optimal measurement results (bitstring
        counts) obtained after optimizing the quantum circuit's parameters.
    """
    algorithm = options['algorithm']
    penalty = options['penalty']
    seed = options['seed']
    verbose = options['verbose']
    optimizer = options['optimizer']
    callback = options['callback']
    max_iter = options['max_iter']
    tolerance = options['tolerance']

    service = ibmq_solver.service
    algorithm_instance = None

    if algorithm == "qaoa":
        algorithm_instance = QAOA(model, p=options['p'], penalty=penalty,
                                  seed=seed)
    elif algorithm == "vqe":
        algorithm_instance = VQE(model, layers=options['layers'],
                                 penalty=penalty, seed=seed,
                                 ansatz=options['ansatz'])

    vqc = ibmq_solver.parse_input(algorithm_instance.circuit)
    backend = ibmq_solver.select_backend(vqc.num_qubits)
    pass_manager = generate_preset_pass_manager(backend=backend,
                                                optimization_level=ibmq_solver.optimization_level)

    starting_point = algorithm_instance.get_starting_point()

    isa_circuit = pass_manager.run(vqc)

    def cost_function(params) -> float:
        """
        Computes the cost (objective function value) for a given set
        of parameters.

        Parameters
        ----------
        params : np.ndarray
            An array of parameters for the quantum circuit.

        Returns
        -------
        float
            The computed cost (energy) for the given parameters.
        """
        counts = compute_counts(params, ibmq_solver, isa_circuit, sampler)
        cost = calculate_energy(counts, ibmq_solver.shots, algorithm_instance)
        if verbose:
            print(f'\nCost = {cost}')
        return cost

    with Session(service=service, backend=backend) as session:
        sampler = Sampler(mode=session)

        optimization_result = minimize(fun=cost_function,
                                       x0=starting_point,
                                       method=optimizer,
                                       tol=tolerance,
                                       callback=callback,
                                       options={'maxiter': max_iter})

        optimal_params = optimization_result.x

        return compute_counts(optimal_params, ibmq_solver, isa_circuit,
                              sampler)


def compute_counts(params, solver, qc, sampler):
    """
    Computes the measurement results (bitstring counts) for a given set of
    parameters.

    This function binds the parameters to the quantum circuit, runs the
    circuit using the provided sampler, and parses the raw measurement
    results into meaningful counts.

    Parameters
    ----------
    params : np.ndarray
        An array of parameters to assign to the quantum circuit.
    solver : Solver
        The solver instance responsible for running the quantum circuit and
        processing the results.
    qc : QuantumCircuit
        The transpiled quantum circuit.
    sampler : Sampler
        The sampler instance responsible for running the quantum circuit on
        the quantum backend.

    Returns
    -------
    dict
        A dictionary of bitstring counts representing the measurement results
        from the quantum execution.
    """
    params_dict = {f'theta{i}': params[i] for i in range(len(params))}
    bound_vqc = qc.assign_parameters(params_dict)

    raw_counts = solver.run(bound_vqc, sampler)

    counts = solver.parse_response(raw_counts)
    return counts
