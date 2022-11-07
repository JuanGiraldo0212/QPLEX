from docplex.mp.model import Model
from .request_handler import solver_request


class QModel(Model):
    def __init__(self, name):
        super(QModel, self).__init__(name)

    def solve(self, solver: str = 'classical'):
        # Use this information to create a dictionary with all the information about the model, then pass it to the
        # solver request. Must include, name, variables (with types), constraints and objective function.
        for v in self.iter_variables():
            print(f"{v}")
        # for c in self.iter_constraints():
        #     print(c)
        # print(self.get_objective_expr())
        # print(self.objective_sense.name)
        if solver is None:
            raise ValueError("Missing value for argument 'hardware'")
        if solver == 'classical':
            Model.solve(self)
        elif solver == 'quantum':
            model = {
                'name': 'peti',

            }
            print(solver_request())
        else:
            raise ValueError("Invalid value for argument 'hardware'")
