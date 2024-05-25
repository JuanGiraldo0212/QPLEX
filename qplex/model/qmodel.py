import time

from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qplex.commons import solver_factory
from qplex.commons import ggaem_workflow, get_ggaem_solution
import os


class QModel(Model):

    def __init__(self, name):
        super(QModel, self).__init__(name)
        self.job_id = None
        self.quantum_api_tokens = {
            "d-wave_token": os.environ.get('D-WAVE_API_TOKEN'),
            "ibmq_token": os.environ.get('IBMQ_API_TOKEN'),
        }
        self.exec_time = 0
        self.method = None
        self.provider = None
        self.backend = None

    def solve(self,
              method: str = 'classical',
              provider: str = None,
              backend: str = None,
              algorithm: str = "qaoa",
              ansatz: str = None,
              p: int = 2,
              layers: int = 2,
              optimizer: str = "COBYLA",
              tolerance: float = 1e-10,
              max_iter: int = 1000,
              penalty: float = None,
              shots: int = 1024,
              seed: int = 1):

        t0 = time.time()
        if method == 'classical':
            Model.solve(self)
            end_time = time.time() - t0
        elif method == 'quantum':
            solver = solver_factory.get_solver(provider=provider,
                                               quantum_api_tokens=
                                               self.quantum_api_tokens,
                                               shots=shots,
                                               backend=backend)
            if provider == "d-wave":
                solution = solver.solve(self)
                end_time = time.time() - t0
            else:
                optimal_counts = ggaem_workflow(model=self,
                                                solver=solver,
                                                shots=shots,
                                                algorithm=algorithm,
                                                optimizer=optimizer,
                                                tolerance=tolerance,
                                                max_iter=max_iter,
                                                ansatz=ansatz,
                                                layers=layers,
                                                p=p,
                                                seed=seed,
                                                penalty=penalty)
                end_time = time.time() - t0
                solution = get_ggaem_solution(self, optimal_counts)
            self.method = method
            self.provider = provider
            self.backend = backend
            self.set_solution(solution)
        else:
            raise ValueError("Invalid value for argument 'method'")

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
        print(f"method:"
              f" {self.method if self.method is not None else 'classical'}")
        print(f"provider: "
              f"{self.provider if self.provider is not None else 'N/A'}")
        print(f"backend: "
              f"{self.backend if self.backend is not None else 'N/A'}")
        print(f"execution time: {round(self.exec_time, 2)} seconds")
        super(QModel, self).print_solution(print_zeros,
                                           solution_header_fmt,
                                           var_value_fmt,
                                           **kwargs)
