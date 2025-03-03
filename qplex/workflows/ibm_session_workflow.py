import qiskit_ibm_runtime
import scipy.optimize
import qiskit.transpiler.preset_passmanagers
import docplex.mp.model

import qplex.commons.algorithm_factory
from qplex.model.execution_config import ExecutionConfig
import qplex.utils.workflow_utils


def run_ibm_session_workflow(model: docplex.mp.model.Model, ibmq_solver,
                             options: ExecutionConfig):
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

    service = ibmq_solver.service

    verbose = options.verbose
    optimizer = options.optimizer
    callback = options.callback
    max_iter = options.max_iter
    tolerance = options.tolerance

    algorithm_config = qplex.commons.algorithm_factory.AlgorithmConfig(
        algorithm=qplex.commons.algorithm_factory.AlgorithmType(
            options.algorithm),
        penalty=options.penalty,
        seed=options.seed,
        p=options.p,
        mixer=options.mixer,
        layers=options.layers,
        ansatz=options.ansatz
    )

    algorithm_instance = qplex.commons.algorithm_factory.AlgorithmFactory.get_algorithm(
        model,
        algorithm_config)

    vqc = ibmq_solver.parse_input(algorithm_instance.circuit)
    backend = ibmq_solver.select_backend(vqc.num_qubits)
    pass_manager = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
        backend=backend,
        optimization_level=
        ibmq_solver.optimization_level)

    starting_point = algorithm_instance.get_starting_point()

    isa_circuit = pass_manager.run(vqc)

    def cost_function(params) -> float:  # pragma: no cover
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
        cost = qplex.utils.workflow_utils.calculate_energy(counts,
                                                           ibmq_solver.shots,
                                                           algorithm_instance)
        if verbose:
            print(f'\nCost = {cost}')
        return cost

    with qiskit_ibm_runtime.Session(service=service,
                                    backend=backend) as session:
        sampler = qiskit_ibm_runtime.SamplerV2(mode=session)

        optimization_result = scipy.optimize.minimize(fun=cost_function,
                                                      x0=starting_point,
                                                      method=optimizer,
                                                      tol=tolerance,
                                                      callback=callback,
                                                      options={
                                                          'maxiter': max_iter})

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
