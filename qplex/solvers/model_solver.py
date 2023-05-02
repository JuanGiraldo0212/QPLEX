from qplex.helpers.factories import solver_factory


class ModelSolver:
    @staticmethod
    def solve(model, backend):
        solver = solver_factory.get_solver(backend)
        return solver.solve(model)
