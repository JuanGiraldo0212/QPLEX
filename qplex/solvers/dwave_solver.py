from typing import Dict, Any

from qplex.model.constants import VAR_TYPE
from dwave.system import (LeapHybridCQMSampler, LeapHybridBQMSampler,
                          LeapHybridDQMSampler, )
from dimod import (ConstrainedQuadraticModel, QuadraticModel,
                   DiscreteQuadraticModel, BinaryQuadraticModel, )
from qplex.solvers.base_solver import Solver


class DWaveSolver(Solver):

    def solve(self, model) -> dict:
        token = model.quantum_api_tokens.get("d-wave_token")
        parsed_model, model_type = self.parse_input(model)
        if model_type == VAR_TYPE['C']:
            sampler = LeapHybridCQMSampler(token=token)
            sampleset = sampler.sample_cqm(parsed_model,
                                           label=model.name).filter(
                lambda row: row.is_feasible)
        elif model_type == VAR_TYPE['I']:
            sampler = LeapHybridDQMSampler(token=token)
            sampleset = sampler.sample_dqm(parsed_model, label=model.name)
        else:
            sampler = LeapHybridBQMSampler(token=token)
            sampleset = sampler.sample(parsed_model, label=model.name)
        best = sampleset.first
        response = self.parse_response(best)
        return response

    def parse_response(self, response: Any) -> Dict:
        objective = abs(response.energy)
        solution = response.sample

        result = {'objective': float(objective), 'solution': solution}

        return result

    def parse_input(self, model) -> any:

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
                'I'] else BinaryQuadraticModel(
                vartype='BINARY')
            parsed_model = self.parse_objective(model, parsed_model)

        return parsed_model, model_type

    def parse_objective(self, model, parsed_model):
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

    def parse_constraint(self, constraint):
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

    def select_backend(self, model) -> str:
        pass
