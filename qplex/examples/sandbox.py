from qplex.model import QModel


def main():
    w = [4, 2, 5, 4, 5, 1, 3, 5]
    v = [10, 5, 18, 12, 15, 1, 2, 8]
    bonus = [[0],
             [3, 0],
             [9, 5, 0],
             [2, 3, 9, 0],
             [6, 5, 9, 8, 0],
             [5, 6, 9, 1, 2, 0],
             [9, 1, 6, 2, 1, 5, 0],
             [6, 9, 2, 3, 5, 4, 8, 0]]
    c = 15
    n = len(w)
    knapsack_model = QModel('knapsack')
    x = knapsack_model.binary_var_list(n, name="x")
    knapsack_model.add_constraint(sum(w[i] * x[i] for i in range(n)) <= c)
    # obj_fn = sum(v[i] * x[i] for i in range(n))
    obj_fn = sum((v[i] * x[i] + bonus[i][j] * x[i] * x[j]) for i in range(n) for j in range(n) if i > j)
    knapsack_model.set_objective('max', obj_fn)
    # knapsack_model.solve()
    knapsack_model.solve('quantum', backend='d-wave')
    print(knapsack_model.objective_value)
    print(knapsack_model.print_solution())


if __name__ == '__main__':
    main()
