import time

from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qplex.commons import solver_factory
from qplex.commons import ggaem_workflow, get_ggaem_solution
from qplex.model.options import Options
import os


class QModel(Model):

    def __init__(self, name):
        super(QModel, self).__init__(name)
        self.job_id = None
        self.quantum_api_tokens = {
            'd-wave_token': os.environ.get('D-WAVE_API_TOKEN'),
            'ibmq_token': os.environ.get('IBMQ_API_TOKEN'),
        }
        self.exec_time = 0
        self.method = None
        self.provider = None
        self.backend = None
        self.algorithm = 'NA'

    def solve(self, method: str = 'classical', options: Options = Options()):
        self.method = method
        t0 = time.time()
        if method == 'classical':
            Model.solve(self)
            end_time = time.time() - t0
        elif method == 'quantum':
            solver = solver_factory.get_solver(provider=options['provider'],
                                               quantum_api_tokens=
                                               self.quantum_api_tokens,
                                               shots=options['shots'],
                                               backend=options['backend'])
            if options['provider'] == 'd-wave':
                solution = solver.solve(self)
                end_time = time.time() - t0
            else:
                optimal_counts = ggaem_workflow(model=self,
                                                solver=solver,
                                                verbose=options['verbose'],
                                                shots=options['shots'],
                                                algorithm=options['algorithm'],
                                                optimizer=options['optimizer'],
                                                tolerance=options['tolerance'],
                                                max_iter=options['max_iter'],
                                                ansatz=options['ansatz'],
                                                layers=options['layers'],
                                                p=options['p'],
                                                seed=options['seed'],
                                                penalty=options['penalty'])
                end_time = time.time() - t0
                self.algorithm = options['algorithm']
                solution = get_ggaem_solution(self, optimal_counts)
            self.provider = options['provider']
            self.backend = options['backend']
            self.set_solution(solution)
        else:
            raise ValueError("Invalid value for argument 'method'. Must be "
                             "'classical' or 'quantum'")

        self.exec_time = end_time

    def set_solution(self, result):
        solve_solution = SolveSolution(self,
                                       var_value_map=result['solution'],
                                       obj=result['objective'],
                                       name=self.name)
        Model._set_solution(self, new_solution=solve_solution)

    def print_solution(self, print_zeros=False,
                       solution_header_fmt=None,
                       var_value_fmt=None,
                       **kwargs):
        print(f"method: {self.method}")
        print(f"algorithm: {self.algorithm}")
        print(f"provider: "
              f"{self.provider if self.provider is not None else 'N/A'}")
        print(f"backend: "
              f"{self.backend if self.backend is not None else 'N/A'}")
        print(f"execution time: {round(self.exec_time, 2)} seconds")
        super(QModel, self).print_solution(print_zeros,
                                           solution_header_fmt,
                                           var_value_fmt,
                                           **kwargs)
