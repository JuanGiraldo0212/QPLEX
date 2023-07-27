<p align="center">
  <img src="assets/QPLEX_logo.png?raw=true" alt="QPLEX"/>
</p>

⚠️ **This library is currently under development.**

`QPLEX` is an open-source Python library that enables developers to implement a general optimization model and execute it seamlessly on multiple quantum devices using different quantum algorithms. Our solution automatically handles the adaptation of the optimization model to the specific instructions of the target quantum device's SDK. The library supports unconstrained and constrained problems, as well as binary, discrete, and continuous variables. QPLEX uses `DOcplex` as a modeling API allowing the creation of optimization models using the exact same syntax as this base library. 

The motivation and technical details about this project are presented in the following paper: https://arxiv.org/abs/2307.14308

## Getting Started

The `QModel` module provides all the necessary functionality for creating and executing a combinatorial optimization problem. The following example illustrates how to build and run the knapsack problem with QPLEX.

```python3
from qplex import QModel
# Problem definition
weights = [4, 2, 5, 4, 5, 1, 3, 5]
values = [10, 5, 18, 12, 15, 1, 2, 8]
max_weight = 15
n = len(values)

# Build the model
knapsack_model = QModel('knapsack')
x = knapsack_model.binary_var_list(n, name="x")
knapsack_model.add_constraint(
    sum(weights[i] * x[i] for i in range(n)
        ) <= max_weight)
obj_fn = sum(values[i] * x[i] for i in range(n))
knapsack_model.set_objective('max', obj_fn)
# Code for solving the knapsack problem
# with quantum resources
knapsack_model.solve(solver='quantum')

# Code for solving the knapsack problem
# with classical resources
knapsack_model.solve(solver='classical')
```
It is worth noting that the syntax used by QPLEX is exactly the same as the one used by DOcplex. This facilitates the integration of our solution into classical workflows.

QPLEX provides a new functionality within the `solve` method; it allows the user to specify how the problem will be executed, either through classical or quantum resources. If nothing is passed to the method, classical resources will be used. 

When calling the calling the `solve` method with the "quantum" solver as shown above, the software will automatically select the most appropriate algorithm and quantum hardware backend for the defined optimization model. However, the users have the option to specify which algorithm, provider, and backend they want to use for the execution.

## Supported Algorithms

QPLEX currently supports the following algorithms:

- Quantum Annealing (QA)
- The Variational Quantum Eigensolver (VQE)
- The Quantum Approximate Optimization Algorithm (QAOA)

Each of these has its own set of hyperparameters that can be passed as kwargs to the `solve` method. (Documentation pending)

```python3
params = {p: 2, shots: 2048, optimizer: "COBYLA"}
knapsack_model.solve(solver='quantum', algorithm='QAOA', params)
```

Algorithms to be supported in the future:
- The Quantum Alternating Operator Ansatz (QAOAnsatz)
- Warm-start QAOA (WSQAOA)
- CVaR QAOA (CVar)

## Supported Quantum Providers

The quantum providers supported by the library are:
- D-Wave
- IBMQ
- AWS Braket (In progress)

Each of these has its own set of backends (i.e., quantum computers) and can be selected by providing a value to the "backend" parameter in the `solve` method. If no argument is specified, the library will automatically select the backend with the shortest queue and enough qubits to handle the formulation.

```python3
knapsack_model.solve(solver='quantum', provider='IBMQ', backend='bogota')
```

## Contributing

If you are interested in contributing, please check the issues page and select the one you want to address. Afterward, fork the repository and create a new branch with the issue's number. Make sure to push all you changes in a single commit with a descriptive message. If the issue description is not clear, feel free to create a comment requesting more information.

### Feature requests, bug reports and feedback comments are highly appreciated! 😃
