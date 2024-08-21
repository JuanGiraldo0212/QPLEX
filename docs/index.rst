Welcome to the QPLEX documentation
===================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api

Introduction
------------
QPLEX is an open-source Python library that enables developers to implement combinatorial optimization models and execute them seamlessly on multiple classical and quantum devices (using different quantum algorithms).

Our solution automatically handles the adaptation of the optimization model to the specific instructions of the target quantum device's SDK.

The library supports unconstrained and constrained problems, as well as binary, discrete, and continuous variables. QPLEX uses DOcplex as a modeling API allowing the creation of optimization models using the exact same syntax as this base library.

Installation
------------
.. code-block:: console

   pip install your-library

Quick Start
-----------
A brief example of how to use your library:

.. code-block:: python

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
   execution_params = {
        "provider": "braket",
        "backend": "simulator",
        "algorithm": "qaoa",
        "p": 4,
        "max_iter": 500,
        "shots": 10000
    }
   knapsack_model.solve(solver='quantum', Options(**execution_params))


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
