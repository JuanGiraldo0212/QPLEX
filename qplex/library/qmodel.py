from docplex.mp.model import Model
from .request_handler import solver_request


class QModel(Model):
    def __init__(self, name):
        super(QModel, self).__init__(name)
        
    def solve(self):
        # Use this information to create a dictionary with all the information about the model, then pass it to the
        # solver request. Must include, name, variables (with types), constraints and objective function.
        # for v in self.iter_variables():
        #     print(f"{v}")
        # for c in self.iter_constraints():
        #     print(c)
        # print(self.get_objective_expr())
        # print(self.objective_sense.name)

        print(solver_request())

        Model.solve(self)


