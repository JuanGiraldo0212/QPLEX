from typing import Union

from docplex.mp.linear import LinearExpr
from docplex.mp.quad import QuadExpr

from .constants import VAR_TYPE
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, QuadraticModel
import os



def run(model):
    token = os.environ.get('DWAVE_API_TOKEN')
    sampler = LeapHybridCQMSampler(token=token)
    cqm = parse_model(model)
    sampleset = sampler.sample_cqm(cqm, label=model['name'])
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    best = feasible_sampleset.first

    return parse_response(best)


def parse_response(response):
    objective = response.energy
    solution = response.items()

    ans = {'objective': objective, 'solution': solution}

    return ans


def parse_model(model) -> ConstrainedQuadraticModel:
    variables = list(map(lambda var: {
        'name': var.name,
        'type': var.vartype.cplex_typecode,
    }, list(model.iter_variables())))
    constraints = list(model.iter_constraints())
    objective_func = model.get_objective_expr()
    sense = model.objective_sense.name

    cqm = ConstrainedQuadraticModel()
    obj = QuadraticModel()

    for var in variables:
        obj.add_variable(VAR_TYPE[var['type']], var['name'])

    if type(objective_func) is LinearExpr:
        terms = objective_func.iter_terms()
        for t in terms:
            value = t[1] if sense == "Minimize" else t[1] * -1
            obj.set_linear(t[0].name, value)
    else:
        linear_terms = objective_func.iter_terms()
        quadratic_terms = list(objective_func.iter_quad_triplets())
        if len(linear_terms) > 0:
            for t in linear_terms:
                value = t[1] if sense == "Minimize" else t[1] * -1
                obj.set_linear(t[0].name, value)
        if len(quadratic_terms) > 0:
            for t in quadratic_terms:
                value = t[2] if sense == "Minimize" else t[2] * -1
                obj.set_quadratic(t[0].name, t[1].name, value)

    cqm.set_objective(obj)

    for const in constraints:
        const_qm = QuadraticModel()
        for var in variables:
            const_qm.add_variable(VAR_TYPE[var['type']], var['name'])
        expr = const.left_expr
        value = const.right_expr.constant
        sen = const.sense.operator_symbol
        if type(expr) is LinearExpr:
            terms = objective_func.iter_terms()
            for t in terms:
                const_qm.set_linear(t[0].name, t[1])
        else:
            linear_terms = objective_func.iter_terms()
            quadratic_terms = list(objective_func.iter_quad_triplets())
            if len(linear_terms) > 0:
                for t in linear_terms:
                    const_qm.set_linear(t[0].name, t[1])
            if len(quadratic_terms) > 0:
                for t in quadratic_terms:
                    const_qm.set_quadratic(t[0].name, t[1].name, t[2])
        cqm.add_constraint(const_qm, sense=sen, rhs=value, label=const.lpt_name)

    return cqm




