from docplex.mp.model import Model
from .request_handler import solver_request, get_status_request
from docplex.mp.solution import SolveSolution


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
            'constraints': list(map(lambda constraint: constraint.to_string(), list(self.iter_constraints()))),
            'objective_func': self.get_objective_expr().to_string(),
            'sense': self.objective_sense.name,
        }

    def solve(self, solver: str = 'classical', as_job: bool = False, backend: str = None):
        if solver == 'classical':
            Model.solve(self)
        elif solver == 'quantum':
            response = solver_request(model=self.build_model_dict(), as_job=as_job, backend=backend)
            if not as_job:
                self.set_solution(response)
            else:
                self.job_id = response['jobId']
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

