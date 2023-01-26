from qplex.library import dwave_solver


class Adapter:

    def __init__(self, model):
        self.model = model

    def solve(self, backend):
        if backend == "d-wave":
            return dwave_solver.solve(self.model)
        if backend == "ibm":
            return None
        return None

