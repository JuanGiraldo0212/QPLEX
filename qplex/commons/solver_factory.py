from qplex.solvers import IBMQSolver
from qplex.solvers import DWaveSolver


class SolverFactory:
    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict):
        if provider is None:
            if quantum_api_tokens.get("dwave_token"):
                return DWaveSolver()
            if quantum_api_tokens.get("ibm_token"):
                return IBMQSolver(shots=1024)
            raise RuntimeError("Missing credentials for quantum provider")
        if provider == 'd-wave':
            if quantum_api_tokens.get("dwave_token") is None:
                raise RuntimeError("Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if provider == 'ibmq':
            if quantum_api_tokens.get("ibmq_token") is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return IBMQSolver(shots=1024)
        raise ValueError(provider)


solver_factory = SolverFactory()
