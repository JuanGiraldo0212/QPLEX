from qplex.library.qmodel import QModel
import requests


def main():
    w = [4, 2, 5, 4, 5, 1, 3, 5]
    v = [10, 5, 18, 12, 15, 1, 2, 8]
    c = 15
    n = len(w)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n, name="x")
    knapsack_model.add_constraint(sum(w[i]*x[i] for i in range(n)) <= c)
    obj_fn = sum(v[i]*x[i] for i in range(n))
    knapsack_model.set_objective('max', obj_fn)
    knapsack_model.solve(solver='lol')
    # print(knapsack_model.objective_value)
    # print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
