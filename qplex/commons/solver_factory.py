from qplex.solvers import IBMQSolver
from qplex.solvers import DWaveSolver
from qplex.solvers.braket_solver import BraketSolver


class SolverFactory:
    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict, shots: int, backend: str):
        dwave_token = quantum_api_tokens.get("d-wave_token")
        ibmq_token = quantum_api_tokens.get("ibmq_token")
        if provider is None:
            if dwave_token:
                return DWaveSolver()
            if ibmq_token:
                return IBMQSolver(token=ibmq_token, shots=shots, backend=backend)
            raise RuntimeError("Missing credentials for quantum provider")
        if provider == 'd-wave':
            if dwave_token is None:
                raise RuntimeError("Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if provider == 'ibmq':
            if ibmq_token is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return IBMQSolver(token=ibmq_token, shots=shots, backend=backend)
        if provider == "braket":
            return BraketSolver(shots=shots, backend=backend)
        raise ValueError(provider)


solver_factory = SolverFactory()
