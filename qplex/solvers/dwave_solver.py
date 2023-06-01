from typing import Dict, Any

from docplex.mp.linear import LinearExpr

from qplex.model.constants import VAR_TYPE
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, QuadraticModel
from qplex.solvers.base_solver import Solver


class DWaveSolver(Solver):

    def solve(self, model) -> dict:
        token = model.quantum_api_tokens.get("dwave_token")
        sampler = LeapHybridCQMSampler(token=token)
        cqm = self.parse_input(model)
        sampleset = sampler.sample_cqm(cqm, label=model.name)
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
        best = feasible_sampleset.first
        response = self.parse_response(best)
        return response

    def parse_response(self, response: Any) -> Dict:
        objective = abs(response.energy)
        solution = response.sample

        result = {'objective': float(objective), 'solution': solution}

        return result

    def parse_input(self, model) -> ConstrainedQuadraticModel:

        cqm = ConstrainedQuadraticModel()
        obj = QuadraticModel()

        for var in model.iter_variables():
            obj.add_variable(VAR_TYPE[var.vartype.cplex_typecode], var.name, lower_bound=var.lb,
                             upper_bound=var.ub)

        if type(model.get_objective_expr()) is LinearExpr:
            terms = model.get_objective_expr().iter_terms()
            for t in terms:
                value = t[1] if model.objective_sense.name == "Minimize" else t[1] * -1
                obj.set_linear(t[0].name, value)
        else:
            linear_terms = model.get_objective_expr().iter_terms()
            quadratic_terms = list(model.get_objective_expr().iter_quad_triplets())
            if len(linear_terms) > 0:
                for t in linear_terms:
                    value = t[1] if model.objective_sense.name == "Minimize" else t[1] * -1
                    obj.set_linear(t[0].name, value)
            if len(quadratic_terms) > 0:
                for t in quadratic_terms:
                    value = t[2] if model.objective_sense.name == "Minimize" else t[2] * -1
                    obj.set_quadratic(t[0].name, t[1].name, value)

        cqm.set_objective(obj)

        for const in list(map(lambda constraint: constraint, list(model.iter_constraints()))):
            const_qm = QuadraticModel()
            for var in model.iter_variables():
                const_qm.add_variable(VAR_TYPE[var.vartype.cplex_typecode], var.name, lower_bound=var.lb,
                                      upper_bound=var.ub)
            expr = const.left_expr
            value = const.right_expr.constant
            sen = const.sense.operator_symbol
            if type(expr) is LinearExpr:
                terms = expr.iter_terms()
                for t in terms:
                    const_qm.set_linear(t[0].name, t[1])
            else:
                linear_terms = expr.iter_terms()
                quadratic_terms = list(expr.iter_quad_triplets())
                if len(linear_terms) > 0:
                    for t in linear_terms:
                        const_qm.set_linear(t[0].name, t[1])
                if len(quadratic_terms) > 0:
                    for t in quadratic_terms:
                        const_qm.set_quadratic(t[0].name, t[1].name, t[2])
            cqm.add_constraint(const_qm, sense=sen, rhs=value, label=const.lpt_name)

        return cqm

    def select_backend(self, model) -> str:
        pass
