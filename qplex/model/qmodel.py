from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qplex.helpers.factories.solver_factory import solver_factory


class QModel(Model):

    def __init__(self, name):
        super(QModel, self).__init__(name)
        self.job_id = None

    def build_model_dict(self) -> dict:
        return {
            'name': self.name,
            'variables': list(map(lambda var: {
                'name': var.name,
                'type': var.vartype.cplex_typecode,
                'lower_bound': var.lb,
                'upper_bound': var.ub
            }, list(self.iter_variables()))),
            'constraints': list(map(lambda constraint: constraint, list(self.iter_constraints()))),
            'objective_func': self.get_objective_expr(),
            'sense': self.objective_sense.name,
        }

    def solve(self, solver: str = 'classical', backend: str = None):
        if solver == 'classical':
            Model.solve(self)
        elif solver == 'quantum':
            model = self.build_model_dict()
            model_solver = solver_factory.get_solver(backend)
            self.set_solution(model_solver.solve(model))
        else:
            raise ValueError("Invalid value for argument 'solver'")

    def set_solution(self, result):
        solve_solution = SolveSolution(self, var_value_map=result['solution'], obj=result['objective'],
                                       name=self.name)
        Model._set_solution(self, new_solution=solve_solution)

