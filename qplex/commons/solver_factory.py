from qplex.solvers import IBMQSolver
from qplex.solvers import DWaveSolver


class SolverFactory:
    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict, shots: int, backend: str):
        if provider is None:
            if quantum_api_tokens.get("dwave_token"):
                return DWaveSolver()
            if quantum_api_tokens.get("ibm_token"):
                return IBMQSolver(shots=shots, backend=backend)
            raise RuntimeError("Missing credentials for quantum provider")
        if provider == 'd-wave':
            if quantum_api_tokens.get("dwave_token") is None:
                raise RuntimeError("Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if provider == 'ibmq':
            if quantum_api_tokens.get("ibmq_token") is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return IBMQSolver(shots=shots, backend=backend)
        raise ValueError(provider)


solver_factory = SolverFactory()
