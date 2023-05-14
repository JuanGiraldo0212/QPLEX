from qplex.solvers import DWaveSolver, GateBasedSolver


class SolverFactory:
    @staticmethod
    def get_solver(provider: str, quantum_api_tokens: dict):
        if provider is None:
            if quantum_api_tokens.get("dwave_token"):
                return DWaveSolver()
            if quantum_api_tokens.get("ibm_token"):
                return GateBasedSolver()
            raise RuntimeError("Missing credentials for quantum provider")
        if provider == 'd-wave':
            if quantum_api_tokens.get("dwave_token") is None:
                raise RuntimeError("Missing credentials for the D-Wave provider")
            return DWaveSolver()
        if provider == 'ibm':
            if quantum_api_tokens.get("ibm_token") is None:
                raise RuntimeError("Missing credentials for the IBM provider")
            return GateBasedSolver()
        raise ValueError(provider)


solver_factory = SolverFactory()
