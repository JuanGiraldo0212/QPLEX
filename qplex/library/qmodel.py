from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qplex.library.adapter import Adapter


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
            adapter = Adapter(model)
            self.set_solution(adapter.solve(backend))
        else:
            raise ValueError("Invalid value for argument 'hardware'")

    def set_solution(self, result):
        solve_solution = SolveSolution(self, var_value_map=result['solution'], obj=result['objective'],
                                       name=self.name)
        Model._set_solution(self, new_solution=solve_solution)

    def get_status(self):
        if self.job_id is None:
            raise ValueError("Attribute job_id is None")
        response = get_status_request(job_id=self.job_id)
        if response['status'] == 'finished':
            pass
            # Call API to get solution
            # self.set_solution()
        return

