from qplex.model.constants import VAR_TYPE


def get_model_type(model):
    if len(list(model.iter_constraints())) > 0:
        return VAR_TYPE['C']
    if any(VAR_TYPE[var.vartype.cplex_typecode] == VAR_TYPE['I'] for
           var in model.iter_variables()):
        return VAR_TYPE['I']
    return VAR_TYPE['B']


def get_var_type(model):
    if any(VAR_TYPE[var.vartype.cplex_typecode] == VAR_TYPE['I'] for
           var in model.iter_variables()):
        return VAR_TYPE['I']
    return VAR_TYPE['B']
