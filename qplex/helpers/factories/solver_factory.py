from qplex.solvers import DWaveSolver, GateBasedSolver


class SolverFactory:
    @staticmethod
    def get_solver(backend):
        if backend == 'd-wave':
            return DWaveSolver()
        if backend == 'ibm':
            return GateBasedSolver()
        raise ValueError(backend)


solver_factory = SolverFactory()
