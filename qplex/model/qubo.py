import numpy as np
import re


def parse_hamiltonian(hamiltonian_str):
    """
    Parse the Hamiltonian string to extract linear and quadratic terms.

    Args:
        hamiltonian_str (str): String representation of the QUBO
        Hamiltonian.

    Returns:
        QUBO: QUBO object with parsed linear and quadratic terms.
    """
    linear = {}
    quadratic = {}
    max_index = -1

    # Regex for matching terms like "72.0*Z_{0}Z_{1}" and "-252.0*Z_{0}"
    pattern = re.compile(r'([+-]?\d+\.\d+)\*Z_\{(\d+)\}(?:Z_\{(\d+)\})?')

    for match in pattern.finditer(hamiltonian_str):
        coeff = float(match.group(1))
        i = int(match.group(2))
        j = match.group(3)

        max_index = max(max_index, i)
        if j is not None:
            j = int(j)
            max_index = max(max_index, j)
            quadratic[(i, j)] = coeff
        else:
            linear[i] = coeff

    num_vars = max_index + 1
    return QUBO(linear, quadratic, num_vars)


class QUBO:
    """
    Class to represent a QUBO (Quadratic Unconstrained Binary Optimization)
    problem.

    Attributes:
        linear (dict): Linear terms of the QUBO.
        quadratic (dict): Quadratic terms of the QUBO.
        num_vars (int): Number of binary variables in the QUBO.
    """

    def __init__(self, linear, quadratic, num_vars):
        """
        Initialize the QUBO object with linear and quadratic terms.

        Args:
            linear (dict): Linear terms of the QUBO.
            quadratic (dict): Quadratic terms of the QUBO.
            num_vars (int): Number of binary variables in the QUBO.
        """
        self.linear = linear
        self.quadratic = quadratic
        self.num_vars = num_vars

    def get_num_binary_vars(self):
        """
        Get the number of binary variables in the QUBO.

        Returns:
            int: Number of binary variables.
        """
        return self.num_vars

    def linear_to_array(self):
        """
        Convert the linear terms of the QUBO to a numpy array.

        Returns:
            np.ndarray: Array of linear terms.
        """
        linear_array = np.zeros(self.num_vars)
        for i, coeff in self.linear.items():
            linear_array[i] = coeff
        return linear_array

    def quadratic_to_array(self):
        """
        Convert the quadratic terms of the QUBO to a numpy array.

        Returns:
            np.ndarray: 2D array of quadratic terms.
        """
        quadratic_array = np.zeros((self.num_vars, self.num_vars))
        for (i, j), coeff in self.quadratic.items():
            quadratic_array[i, j] = coeff
            quadratic_array[j, i] = coeff
        return quadratic_array

    def evaluate(self, sample):
        """
        Evaluate the QUBO energy for a given binary sample.

        Args:
            sample (list[int]): Binary sample to evaluate.

        Returns:
            float: Energy of the sample.
        """
        energy = 0.0
        for i, coeff in self.linear.items():
            energy += coeff * sample[i]
        for (i, j), coeff in self.quadratic.items():
            energy += coeff * sample[i] * sample[j]
        return energy
