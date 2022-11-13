from docplex.mp.model import Model
from .request_handler import solver_request
from docplex.mp.solution import SolveSolution


class QModel(Model):

    def __init__(self, name):
        super(QModel, self).__init__(name)

    def build_model_dict(self) -> dict:
        return {
            'name': self.name,
            'variables': list(map(lambda var: {
                'name': var.name,
                'type': var.vartype.cplex_typecode,
            }, list(self.iter_variables()))),
            'constraints': list(map(lambda constraint: constraint.to_string(), list(self.iter_constraints()))),
            'objective_func': self.get_objective_expr().to_string(),
            'sense': self.objective_sense.name,
        }

    def solve(self, solver: str = 'classical', as_job: bool = False, backend: str = None):
        if solver == 'classical':
            Model.solve(self)
        elif solver == 'quantum':
            result = solver_request(model=self.build_model_dict(), as_job=as_job, backend=backend)
            solve_solution = SolveSolution(self, var_value_map=result['solution'], obj=result['objective'],
                                           name=self.name)
            Model._set_solution(self, new_solution=solve_solution)
        else:
            raise ValueError("Invalid value for argument 'hardware'")
