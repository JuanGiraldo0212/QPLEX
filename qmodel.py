from docplex.mp.model import Model


class QModel(Model):
    def __init__(self, name):
        super(QModel, self).__init__(name)
        
    def solve(self):
        print("Test")
        for v in self.iter_variables():
            print(f"{v}")
        for c in self.iter_constraints():
            print(c)
        print(self.get_objective_expr())
        print(self.objective_sense.name)

        Model.solve(self)

