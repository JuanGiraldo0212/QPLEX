from typing import Dict, Any

from qplex.model.constants import VAR_TYPE

from dwave.system import (LeapHybridCQMSampler, LeapHybridBQMSampler,
                          LeapHybridDQMSampler, DWaveSampler,
                          EmbeddingComposite, )
from dimod import (ConstrainedQuadraticModel, QuadraticModel,
                   DiscreteQuadraticModel, BinaryQuadraticModel, )
from qplex.solvers.base_solver import Solver


class DWaveSolver(Solver):
    """
    A solver for D-Wave quantum systems capable of handling various model
    types, including constrained quadratic models (CQM), discrete quadratic
    models (DQM), and binary quadratic models (BQM).
    """

    def __init__(self, token, backend='hybrid_solver', time_limit=None,
                 num_reads=100, topology='pegasus'):
        super().__init__()
        self.token = token
        self.backend = backend
        self.presolver = None
        self.original_cqm = None
        self.time_limit = time_limit
        self.num_reads = num_reads
        self.topology = topology

    def solve(self, model) -> Dict[str, Any]:
        """
        Solve the given problem formulation using a D-Wave solver.

        Parameters
        ----------
        model : QModel
            The model to be solved, which includes quantum API tokens and
            the problem specification.

        Returns
        -------
        dict
            A dictionary containing the solution with 'objective' and
            'solution' keys.
        """
        parsed_model, model_type = self.parse_input(model)

        sampler = self.select_backend(parsed_model, model_type)

        # QPU requested and model is not constrained nor contains integer
        # variables
        if self.backend == 'd-wave_sampler' and model_type not in (VAR_TYPE[
                                                                       'C'],
                                                                   VAR_TYPE[
                                                                       'I']):

            sampleset = sampler.sample(parsed_model,
                                       num_reads=self.num_reads,
                                       label=model.name)

        # Hybrid solver requested or QPU requested but model is not
        # immediately compatible with the QUBO form.
        else:
            sampling_methods = {
                VAR_TYPE['C']: lambda: sampler.sample_cqm(
                    parsed_model, time_limit=self.time_limit, label=model.name
                ).filter(lambda row: row.is_feasible),
                VAR_TYPE['I']: lambda: sampler.sample_dqm(
                    parsed_model, time_limit=self.time_limit, label=model.name
                ),
            }

            # Use the appropriate sampling method or fall back to the default
            sampleset = (
                sampling_methods[model_type]()
                if model_type in sampling_methods
                else sampler.sample(parsed_model, time_limit=self.time_limit,
                                    label=model.name)
            )

        # Extract the best solution and format the response
        best = sampleset.first
        return self.parse_response(best)

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse the response from the D-Wave solver to extract solution.

        Parameters
        ----------
        response : Any
            The raw response from the D-Wave solver.

        Returns
        -------
        dict
            A dictionary with 'objective' and 'solution' keys.
        """
        objective = abs(response.energy)
        solution = response.sample

        result = {'objective': float(objective), 'solution': solution}
        return result

    def parse_input(self, model) -> (Any, str):
        """
        Convert the input model into a D-Wave compatible model.

        Parameters
        ----------
        model : QModel
            The model to be parsed, which includes problem
            constraints and objectives.

        Returns
        -------
        tuple
            A tuple containing the parsed model and the model type as
            a string.
        """
        if len(list(model.iter_constraints())) > 0:
            model_type = VAR_TYPE['C']
            obj = self.parse_objective(model, QuadraticModel())
            parsed_model = ConstrainedQuadraticModel()
            parsed_model.set_objective(obj)
            for constraint in model.iter_constraints():
                const_qm = self.parse_constraint(constraint)
                sense = constraint.sense.operator_symbol
                rhs = constraint.right_expr.constant
                parsed_model.add_constraint(const_qm, sense=sense, rhs=rhs,
                                            label=constraint.lpt_name)
        else:
            model_type = VAR_TYPE['B']
            if any(VAR_TYPE[var.vartype.cplex_typecode] == VAR_TYPE['I'] for
                   var in model.iter_variables()):
                model_type = VAR_TYPE['I']
            parsed_model = DiscreteQuadraticModel() if model_type == VAR_TYPE[
                'I'] else BinaryQuadraticModel(vartype='BINARY')
            parsed_model = self.parse_objective(model, parsed_model)

        return parsed_model, model_type

    def parse_objective(self, model, parsed_model) -> Any:
        """
        Convert the objective function into the format required by D-Wave.

        Parameters
        ----------
        model : QModel
            The original model containing the objective function.
        parsed_model:
            The D-Wave model to which the objective function  will be added.

        Returns
        -------
        Any
            The D-Wave model with the objective function set.
        """
        if type(parsed_model) is BinaryQuadraticModel:
            for var in model.iter_variables():
                parsed_model.add_variable(var.name)
        else:
            for var in model.iter_variables():
                parsed_model.add_variable(VAR_TYPE[var.vartype.cplex_typecode],
                                          var.name, lower_bound=var.lb,
                                          upper_bound=var.ub)

        linear_terms = model.get_objective_expr().iter_terms()
        sense_multiplier = 1 if model.objective_sense.name == "Minimize" \
            else -1

        for term in linear_terms:
            parsed_model.set_linear(term[0].name, term[1] * sense_multiplier)

        for term in model.get_objective_expr().iter_quad_triplets():
            parsed_model.set_quadratic(term[0].name, term[1].name,
                                       term[2] * sense_multiplier)

        return parsed_model

    def parse_constraint(self, constraint) -> QuadraticModel:
        """
        Convert a constraint into a D-Wave compatible QuadraticModel.

        Parameters
        ----------
        constraint : Any
            The constraint to be converted.

        Returns
        -------
        QuadraticModel
            The D-Wave model representing the constraint.
        """
        const_qm = QuadraticModel()
        for var in constraint.iter_variables():
            const_qm.add_variable(VAR_TYPE[var.vartype.cplex_typecode],
                                  var.name, lower_bound=var.lb,
                                  upper_bound=var.ub)

        expr = constraint.left_expr

        for term in expr.iter_terms():
            const_qm.set_linear(term[0].name, term[1])

        for term in expr.iter_quad_triplets():
            const_qm.set_quadratic(term[0].name, term[1].name, term[2])

        return const_qm

    def select_backend(self, parsed_model, model_type) -> Any:
        """
        Select the appropriate backend for the given model type.
        """
        hybrid_samplers = {
            VAR_TYPE['C']: LeapHybridCQMSampler,
            VAR_TYPE['I']: LeapHybridDQMSampler,
        }

        if self.backend == 'hybrid_solver':
            sampler_class = hybrid_samplers.get(model_type,
                                                LeapHybridBQMSampler)
            return sampler_class(token=self.token)

        elif self.backend == 'd-wave_sampler':
            # User requested a DWaveSampler, but model is not QUBO-compatible.
            if model_type in hybrid_samplers:
                print(
                    "The selected backend requires a QUBO-compatible model, "
                    "but the given model contains constraints or discrete "
                    "variables.\nSwitching to an appropriate Hybrid Solver to "
                    "handle this model type..."
                )
                sampler_class = hybrid_samplers[model_type]
                return sampler_class(token=self.token)

            qpu = DWaveSampler(solver=dict(topology__type=self.topology),
                               token=self.token)
            print(
                f"Selected {qpu.solver.name} with {len(qpu.nodelist)} qubits.")
            return EmbeddingComposite(qpu)

        raise ValueError(f"Unsupported backend: {self.backend}")
