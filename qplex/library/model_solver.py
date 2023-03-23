from qplex.library.solver_factory import factory

class ModelSolver:
    def solve(self, model, backend):
        solver = factory.get_solver(backend)
        return solver.solve(model)