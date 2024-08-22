<p align="center">
  <img src="assets/icon.svg?raw=true" alt="QPLEX"/>
</p>

⚠️ **This library is currently under development.**

`QPLEX` is an open-source Python library that enables developers to implement combinatorial optimization models and execute them seamlessly on multiple classical and quantum devices (using different quantum algorithms). Our solution automatically handles the adaptation of the optimization model to the specific instructions of the target quantum device's SDK. The library supports unconstrained and constrained problems, as well as binary, discrete, and continuous variables. QPLEX uses `DOcplex` as a modeling API allowing the creation of optimization models using the exact same syntax as this base library. 

The motivation and technical details about this project are presented in the following paper: https://arxiv.org/abs/2307.14308

## Getting Started

QPLEX is not a self-managed solution, hence each user has to bring their own quantum provider access token in order to use quantum devices. All tokens must be associated with environment variables within the system.

For using devices from IBM Quantum the following env variable has to be created: `IBMQ_API_TOKEN`

For using devices from D-Wave the following env variable has to be created: `D-WAVE_API_TOKEN`

In Amazon Braket's case the user must have the AWS CLI configured properly and should have access to the AWS Braket service in the cloud. Follow <a href="https://aws.amazon.com/braket/getting-started/" target="_blank">this link</a> for more information.

The `QModel` module provides all the necessary functionality for creating and executing a combinatorial optimization problem. The following example illustrates how to build and run the knapsack problem with QPLEX. More examples are provided within the "examples" folder.

```python3
from qplex import QModel
from qplex.model import Options
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
```
It is worth noting that the syntax used by QPLEX is exactly the same as the one used by DOcplex. This facilitates the integration of our solution into classical workflows.

QPLEX provides a new functionality within the `solve` method; it allows the user to specify how the problem will be executed, either through classical or quantum resources. If nothing is passed to the method, classical resources will be used. 

When calling the `solve` method with "quantum" resources, the software will automatically select the most appropriate algorithm and quantum hardware backend for the defined optimization model. However, the users have the option to specify, as kwargs, which algorithm, provider, and backend they want to use for the execution.

The following is the complete list of kwargs supported for the solve method:

| Argument  | Description                                | Default value  |
|-----------|--------------------------------------------|----------------|
| provider  | The quantum hardware provider              | "d-wave"       |
| backend   | The specific quantum device                | Calculated     |
| algorithm | The quantum algorithm to use               | "qaoa"         |
| ansatz    | The ansatz circuit for VQE                 | Layered Ansatz |
| p         | The p value for the QAOA algorithm         | 2              |
| layers    | The number of layers for the VQE algorithm | 2              |
| optimizer | The classical optimizer                    | "cobyla"       |
| tolerance | The tolerance value for the optimizer      | 1e − 10        |
| max_iter  | The maximum number of optimizer iterations | 1000           |
| penalty   | The penalty constant used for the QUBO     | Calculated     |
| shots     | The total number of shots                  | 1024           |
| seed      | The execution random seed                  | 1              |

## Supported Algorithms

QPLEX currently supports the following algorithms:

- Quantum Annealing (QA)
- The Variational Quantum Eigensolver (VQE)
- The Quantum Approximate Optimization Algorithm (QAOA)

Each of these has its own set of hyperparameters that can be passed as kwargs to the `solve` method. Refer to the previous table for the complete list.

```python3
execution_params = {
        "provider": "braket",
        "backend": "simulator",
        "algorithm": "qaoa",
        "p": 4,
        "max_iter": 500,
        "shots": 10000
    }
knapsack_model.solve(solver='quantum', Options(**execution_params))
```

Algorithms to be supported in the future:
- The Quantum Alternating Operator Ansatz (QAOAnsatz)
- Warm-start QAOA (WSQAOA)
- CVaR QAOA (CVar)

## Supported Quantum Providers

The quantum providers supported by the library are:
- D-Wave
- IBMQ
- AWS Braket

Each of these has its own set of backends (i.e., quantum computers) and can be selected by providing a value to the "backend" key-word argument in the `solve` method. If no argument is specified, the library will automatically select the backend with the shortest queue and enough qubits to handle the formulation. Additionally, it is possible to use the argument `simulator` as a backend to use each provider's local simulator.

```python3
knapsack_model.solve(solver='quantum', provider='ibmq', backend='ibm_perth')
```

```python3
knapsack_model.solve(solver='quantum', provider='braket', backend='device/qpu/ionq/Harmony')
```

## Contributing

If you are interested in contributing, please check the issues page and select the one you want to address. Afterward, fork the repository and create a new branch with the issue's number. Make sure to push all you changes in a single commit with a descriptive message. If the issue description is not clear, feel free to create a comment requesting more information.

### Feature requests, bug reports and feedback comments are highly appreciated! 😃
