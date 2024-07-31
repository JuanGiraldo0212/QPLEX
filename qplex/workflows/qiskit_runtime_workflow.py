from qiskit_ibm_runtime import (QiskitRuntimeService,
                                SamplerV2 as
                                Sampler, Session, )
from qiskit.qasm3 import loads
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qplex.algorithms import QAOA, VQE
from qplex.commons.ggaem import calculate_energy


def qiskit_runtime_workflow(model, backend: str, verbose: bool, shots: int,
                            algorithm: str, optimizer: str, callback,
                            max_iter: int, tolerance: float, ansatz: str,
                            p: int, layers: int, seed: int, penalty: float):
    service = QiskitRuntimeService()
    algorithm_instance = None
    if algorithm == "qaoa":
        algorithm_instance = QAOA(model, p=p, penalty=penalty, seed=seed)
    elif algorithm == "vqe":
        algorithm_instance = VQE(model, layers=layers, penalty=penalty,
                                 seed=seed, ansatz=ansatz)

    vqc = loads(algorithm_instance.parse_to_vqc())

    if backend is None or backend == "":
        print('No backend specified. Using least busy...')
        selected_backend = service.least_busy(min_num_qubits=vqc.num_qubits)
    else:
        selected_backend = service.get_backend(backend)

    pass_manager = generate_preset_pass_manager(backend=selected_backend,
                                                optimization_level=1)
    isa_circuit = pass_manager.run(vqc)
    starting_point = algorithm_instance.get_starting_point()

    def cost_function(params, vqc, sampler) -> float:
        counts = compute_counts(params, vqc, sampler, shots)
        cost = calculate_energy(counts, shots, algorithm_instance)
        if verbose:
            print(f'\nCost = {cost}')
        return cost

    with Session(service=service, backend=selected_backend) as session:
        sampler = Sampler(mode=session)

        optimization_result = minimize(cost_function,
                                       starting_point,
                                       args=(isa_circuit, sampler),
                                       method=optimizer,
                                       tol=tolerance,
                                       callback=callback,
                                       options={'maxiter': max_iter})
        print(optimization_result)

        optimal_params = optimization_result.x
        return compute_counts(optimal_params, vqc, sampler, shots)


def compute_counts(params, vqc, sampler, shots):
    params_dict = {f'theta{i}': params[i] for i in range(len(params))}
    bound_vqc = vqc.assign_parameters(params_dict)
    pub = (bound_vqc,)
    result = sampler.run([pub], shots=shots).result()
    data = result[0].data
    bits = data.c
    raw_counts = bits.get_counts()
    counts = {}
    for sample, count in raw_counts.items():
        x = [int(bit) for bit in reversed(sample)]
        counts["".join(str(n) for n in x)] = count

    return counts
