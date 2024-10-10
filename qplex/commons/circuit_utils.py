import re
import numpy as np


def replace_params(circuit: str, params: np.ndarray) -> str:
    """
    Replaces parameter placeholders in a quantum circuit string with actual
    values.

    This function takes a quantum circuit in the form of a string and replaces
    placeholders of the form 'thetaX' (where X is an integer index) with the
    corresponding values from a provided array of parameters. This is
    commonly used in
    parameterized quantum circuits for variational algorithms like VQE and
    QAOA, where gates
    depend on tunable parameters.

    Parameters:
    ----------
    circuit : str
        The quantum circuit as a string in which parameter placeholders of
        the form 'thetaX' (e.g., theta0, theta1) need to be replaced by
        actual numerical values.

    params : np.ndarray
        A 1-dimensional numpy array containing the numerical parameter
        values. Each index in the array corresponds to a specific placeholder
        in the circuit string. For example, params[0] replaces 'theta0',
        params[1] replaces 'theta1', and so on.

    Returns:
    -------
    str
        The quantum circuit string with all placeholders replaced by the
        corresponding values from the `params` array.

    Raises:
    ------
    IndexError
        If the circuit string contains a 'thetaX' placeholder where X is
        greater than or equal to the length of the `params` array, this will
        raise an IndexError because no corresponding parameter is available.

    Example:
    -------
    Given a circuit:
        circuit = "ry(theta0) q[0];\nrz(theta1) q[1];\n"
    And the parameters:
        params = np.array([0.5, 1.2])
    The result will be:
        "ry(0.5) q[0];\nrz(1.2) q[1];\n"
    """

    def replacer(match):
        param_index = int(match.group(1))
        return str(params[param_index])

    return re.sub(r'theta(\d+)', replacer, circuit)
