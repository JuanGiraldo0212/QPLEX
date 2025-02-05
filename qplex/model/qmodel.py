import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qplex.commons import SolverFactory
from qplex.commons.solver_factory import ProviderConfig, ProviderType
from qplex.utils.workflow_utils import get_solution_from_counts
from qplex.workflows import ibm_session_workflow, ggae_workflow
from qplex.model.execution_config import ExecutionConfig
import os


class SolveMethod(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"


@dataclass
class ModelSolution:
    solution: Dict[str, Any]
    objective: float
    execution_time: float
    method: SolveMethod
    provider: Optional[str] = None
    backend: Optional[str] = None
    algorithm: str = "N/A"


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
        super(QModel, self).__init__(name)
        self.quantum_api_tokens: Dict[str, Optional[str]] = {
            'd-wave_token': os.environ.get('D-WAVE_API_TOKEN'),
            'ibmq_token': os.environ.get('IBMQ_API_TOKEN'),
        }

        self._qmodel_solution: Optional[ModelSolution] = None

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
        mod = from_docplex_mp(self)
        converter = QuadraticProgramToQubo(penalty=penalty)
        return converter.convert(mod)

    def solve(self, method: str = 'classical',
              config: ExecutionConfig = ExecutionConfig()):
        """
        Solves the model using the specified method and Options.

        Parameters
        ----------
        method: str, optional
            The method to solve the model, either 'classical' or
            'quantum'.
        config: Options, optional
            The options for solving the model.

        Raises
        ------
        ValueError
            If the method argument is not 'classical' or 'quantum'.
        """

        solve_method = SolveMethod(method)
        start_time = time.time()

        if solve_method == SolveMethod.CLASSICAL:
            super().solve()
            solution = self._create_solution(
                execution_time=time.time() - start_time,
                method=solve_method
            )
        elif solve_method == SolveMethod.QUANTUM:
            provider_config = ProviderConfig(
                shots=config.shots,
                backend=config.backend,
                provider_options=config.provider_options
            )
            solver = SolverFactory.get_solver(
                provider=ProviderType(config.provider),
                quantum_api_tokens=
                self.quantum_api_tokens,
                config=provider_config)
            if config.provider == 'd-wave':
                result = solver.solve(self)
            elif config.provider == "ibmq" and config.workflow == "ibm_session":
                optimal_counts = ibm_session_workflow(model=self,
                                                      ibmq_solver=
                                                      solver,
                                                      options=config)
                result = get_solution_from_counts(model=self,
                                                  optimal_counts=optimal_counts)
            else:
                optimal_counts = ggae_workflow(model=self,
                                               solver=solver,
                                               options=config)
                result = get_solution_from_counts(model=self,
                                                  optimal_counts=
                                                  optimal_counts)

            solution = self._create_solution(
                execution_time=time.time() - start_time,
                method=solve_method,
                provider=config.provider,
                backend=config.backend,
                algorithm=config.algorithm,
                result=result
            )

        else:
            raise ValueError("Invalid value for argument 'method'. Must be "
                             "'classical' or 'quantum'")

        self._set_solution(solution)

    def _create_solution(
            self,
            execution_time: float,
            method: SolveMethod,
            provider: Optional[str] = None,
            backend: Optional[str] = None,
            algorithm: str = "N/A",
            result: Optional[Dict] = None
    ) -> ModelSolution:
        """Creates a ModelSolution object from the solve results."""
        if method == SolveMethod.CLASSICAL:
            solution = {var.name: var.solution_value for var in
                        self.iter_variables()}
            objective = self.objective_value
        else:
            solution = result['solution']
            objective = result['objective']

        return ModelSolution(
            solution=solution,
            objective=objective,
            execution_time=execution_time,
            method=method,
            provider=provider,
            backend=backend,
            algorithm=algorithm
        )

    def _set_solution(self, solution: ModelSolution):
        """
        Sets the solution for the model. Used internally by the QModel's
        solve method.

        Parameters
        ----------
        solution: ModelSolution
            The ModelSolution containing 'solution' and
            'objective'.
        """
        self._qmodel_solution = solution
        solve_solution = SolveSolution(model=self,
                                       var_value_map=solution.solution,
                                       obj=solution.objective,
                                       name=self.name)
        super()._set_solution(new_solution=solve_solution)

    def print_solution(self, print_zeros: bool = False,
                       solution_header_fmt: Optional[str] = None,
                       var_value_fmt: Optional[str] = None,
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
        if not self._qmodel_solution:
            print("No solution available. Please run solve() first.")
            return

        print("\nResults")
        print("----------")
        print(f"Method: {self._qmodel_solution.method.value}")
        print(f"Algorithm: {self._qmodel_solution.algorithm}")
        print(f"Provider: {self._qmodel_solution.provider or 'N/A'}")
        print(f"Backend: {self._qmodel_solution.backend or 'N/A'}")
        print(
            f"Execution time: {round(self._qmodel_solution.execution_time, 2)} "
            f"seconds")

        super().print_solution(
            print_zeros=print_zeros,
            solution_header_fmt=solution_header_fmt,
            var_value_fmt=var_value_fmt,
            **kwargs
        )
        print("----------")
