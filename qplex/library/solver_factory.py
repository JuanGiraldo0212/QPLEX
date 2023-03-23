from qplex.library.dwave_solver import DWaveSolver
from qplex.library.gate_based_solver import GateBasedSolver

class SolverFactory:
    def get_solver(self, backend):
        if backend == 'd-wave':
            return DWaveSolver()
        if backend == 'ibm':
            return GateBasedSolver()
        raise ValueError(backend)
    
factory = SolverFactory()