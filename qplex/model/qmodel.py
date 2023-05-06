from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qplex.commons import solver_factory
from qplex.commons import ggae_workflow
import os


class QModel(Model):

    def __init__(self, name):
        super(QModel, self).__init__(name)
        self.job_id = None
        self.quantum_api_tokens = {
            "dwave_token": os.environ.get('DWAVE_API_TOKEN'),
            "ibmq_token": os.environ.get('IBMQ_API_TOKEN'),
        }

    def solve(self, solver: str = 'classical', provider: str = None):
        if solver == 'classical':
            Model.solve(self)
        elif solver == 'quantum':
            solution = None
            model_solver = solver_factory.get_solver(provider, self.quantum_api_tokens)
            if provider == "d-wave":
                solution = model_solver.solve(self)
            else:
                optimal_counts = ggae_workflow(self, model_solver)
                best_solution, best_count = max(optimal_counts.items(), key=lambda x: x[1])
                # TODO turn migrate this code into a separate function
                values = {}
                for i, var in enumerate(self.iter_variables()):
                    values[var.name] = int(best_solution[i])
                obj_value = 0
                linear_terms = self.get_objective_expr().iter_terms()
                quadratic_terms = list(self.get_objective_expr().iter_quad_triplets())
                if len(linear_terms) > 0:
                    for t in linear_terms:
                        obj_value += (values[t[0].name] * t[1])
                if len(quadratic_terms) > 0:
                    for t in quadratic_terms:
                        obj_value += (values[t[0].name] * values[t[1].name] * t[2])
                solution = {'objective': obj_value, 'solution': values}
            self.set_solution(solution)
        else:
            raise ValueError("Invalid value for argument 'solver'")

    def set_solution(self, result):
        solve_solution = SolveSolution(self, var_value_map=result['solution'], obj=result['objective'],
                                       name=self.name)
        Model._set_solution(self, new_solution=solve_solution)

