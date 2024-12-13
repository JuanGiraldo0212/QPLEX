import time

from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qplex.commons import solver_factory
from qplex.commons.workflow_utils import get_solution_from_counts
from qplex.workflows import ibm_session_workflow, ggaem_workflow
from qplex.model.options import Options
from qplex.commons.model_utils import get_model_type, get_var_type
from qplex.model.constants import VAR_TYPE
import os


class QModel(Model):
    """
    Creates an instance of a QModel.

    Parameters
    ----------
    name: str
        The name of the model.

    Returns
    ----------
    An instance of a QModel.
    """

    def __init__(self, name):
        self.docplex_model = super(QModel, self).__init__(name)
        self.quantum_api_tokens = {
            'd-wave_token': os.environ.get('D-WAVE_API_TOKEN'),
            'ibmq_token': os.environ.get('IBMQ_API_TOKEN'),
        }
        self.exec_time = 0
        self.method = None
        self.provider = None
        self.backend = None
        self.algorithm = 'NA'
        self.converter = None
        self.type = None

    def get_qubo(self, penalty: float | None = None) -> QuadraticProgram:
        """
        Returns the QUBO encoding of this problem.

        Parameters
        ----------
        penalty: float, optional
            The penalty factor for the QUBO conversion.

        Returns
        -------
        QuadraticProgram
            The QUBO encoding of this problem.
        """
        quadratic_program = from_docplex_mp(self)
        self.converter = QuadraticProgramToQubo(penalty=penalty)
        return self.converter.convert(quadratic_program)

    def solve(self, method: str = 'classical', options: Options = Options()):
        """
        Solves the model using the specified method and Options.

        Parameters
        ----------
        method: str, optional
            The method to solve the model, either 'classical' or
            'quantum'.
        options: Options, optional
            The options for solving the model.

        Raises
        ------
        ValueError
            If the method argument is not 'classical' or 'quantum'.
        """
        self.method = method
        self.type = get_model_type(self)
        t0 = time.time()
        if method == 'classical':
            Model.solve(self)
            end_time = time.time() - t0
        elif method == 'quantum':
            solver = solver_factory.get_solver(provider=options['provider'],
                                               quantum_api_tokens=
                                               self.quantum_api_tokens,
                                               shots=options['shots'],
                                               backend=options['backend'],
                                               provider_options=options[
                                                   'provider_options'])
            if options['provider'] == 'd-wave':
                solution = solver.solve(self)
                end_time = time.time() - t0
                self.algorithm = 'N/A'
            else:
                if options['provider'] == 'ibmq' and options['workflow'] == \
                        'ibm_session':
                    optimal_counts = ibm_session_workflow(model=self,
                                                          ibmq_solver=
                                                          solver,
                                                          options=options)
                else:
                    optimal_counts = ggaem_workflow(model=self,
                                                    solver=solver,
                                                    options=options)
                end_time = time.time() - t0
                self.algorithm = options['algorithm']
                if get_var_type(self) == VAR_TYPE['I']:
                    solution = get_solution_from_counts(self,
                                                        optimal_counts,
                                                        interpret=True,
                                                        interpreter=
                                                        self.converter)
                else:
                    solution = get_solution_from_counts(self, optimal_counts)
            self.provider = options['provider']
            self.backend = solver.backend
            self.set_solution(solution)
        else:
            raise ValueError("Invalid value for argument 'method'. Must be "
                             "'classical' or 'quantum'")

        self.exec_time = end_time

    def set_solution(self, result):
        """
        Sets the solution for the model. Used internally by the QModel's
        solve method.

        Parameters
        ----------
        result: dict
            The solution dictionary containing 'solution' and
            'objective'.
        """
        solve_solution = SolveSolution(self,
                                       var_value_map=result['solution'],
                                       obj=result['objective'],
                                       name=self.name)
        Model._set_solution(self, new_solution=solve_solution)

    def print_solution(self, print_zeros=False,
                       solution_header_fmt=None,
                       var_value_fmt=None,
                       **kwargs):
        """
        Prints the solution of the model. Must be called after solve().

        Parameters
        ----------
        print_zeros: bool, optional
            Whether to print variables with zero values.
        solution_header_fmt: str, optional
            Format string for the solution header.
        var_value_fmt: str, optional
            Format string for variable values.
        kwargs: dict
            Additional keyword arguments.
        """
        print("\nResults")
        print("----------")
        print(f"Method: {self.method}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Provider: "
              f"{self.provider if self.provider is not None else 'N/A'}")
        print(f"Backend: {self.backend}")
        print(f"Execution time: {round(self.exec_time, 2)} seconds")
        super(QModel, self).print_solution(print_zeros,
                                           solution_header_fmt,
                                           var_value_fmt,
                                           **kwargs)
        print("----------")
