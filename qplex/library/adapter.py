from qplex.library import dwave_solver


class Adapter:

    def __init__(self, model):
        self.model = model

    def solve(self, backend):
        solution = None
        if backend == "d-wave":
            solution = dwave_solver.solve(self.model)
        else:
            pass

        return solution

