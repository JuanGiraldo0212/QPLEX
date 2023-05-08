from qplex.solvers import DWaveSolver, GateBasedSolver


class SolverFactory:
    @staticmethod
    def get_solver(backend: str, quantum_api_tokens: dict):
        if backend is None:
            if quantum_api_tokens.get("dwave_token"):
                return DWaveSolver()
            if quantum_api_tokens.get("ibm_token"):
                return GateBasedSolver()
            raise RuntimeError("Missing credentials for quantum provider")
        if backend == 'd-wave':
            if quantum_api_tokens.get("dwave_token") is None:
                raise RuntimeError("Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if backend == 'ibm':
            if quantum_api_tokens.get("ibm_token") is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return GateBasedSolver()
        raise ValueError(backend)


solver_factory = SolverFactory()
