from qplex import QModel
from qplex.model.options import Options
import numpy as np


def generate_data(image_size, noise_level):
    true_phase = np.linspace(0, 4 * np.pi, image_size ** 2).reshape(
        (image_size, image_size))
    np.random.seed(42)  # For reproducibility
    noisy_phase = true_phase + noise_level * np.random.randn(image_size,
                                                             image_size)

    # Wrap phase to [-pi, pi]
    wrapped_phase = np.mod(noisy_phase + np.pi, 2 * np.pi) - np.pi

    print("True Phase:\n", true_phase)
    print("Noisy Phase:\n", noisy_phase)
    print("Wrapped Phase:\n", wrapped_phase)

    neighbors = []
    for i in range(image_size):
        for j in range(image_size):
            if i > 0:  # Top neighbor
                neighbors.append(((i, j), (i - 1, j)))
            if i < image_size - 1:  # Bottom neighbor
                neighbors.append(((i, j), (i + 1, j)))
            if j > 0:  # Left neighbor
                neighbors.append(((i, j), (i, j - 1)))
            if j < image_size - 1:  # Right neighbor
                neighbors.append(((i, j), (i, j + 1)))

    # Constants for the cost function
    W_st = {}  # Weights for neighbors
    a_st = {}  # Phase differences
    a_s = {}  # Individual pixel constants
    omega_s = {}  # Weights for individual pixels

    for (s, t) in neighbors:
        W_st[(s, t)] = 1  # Uniform weights
        diff = wrapped_phase[s] - wrapped_phase[t]
        a_st[(s, t)] = -np.round(diff / (2 * np.pi))

    for i in range(image_size):
        for j in range(image_size):
            a_s[(i, j)] = np.round(wrapped_phase[i, j] / (2 * np.pi))
            omega_s[(i, j)] = 1  # Uniform weights

    print("Neighbor Weights (W_st):", W_st)
    print("Neighbor Constants (a_st):", a_st)
    print("Pixel Constants (a_s):", a_s)

    return W_st, a_st, a_s, omega_s


def model_phase_unwrapping(image_size: int, W_st: dict, a_st: dict,
                           omega_s: dict, a_s: dict) -> QModel:
    # Create QPLEX model
    phase_model = QModel("phase_unwrapping")

    # Define binary variables for each pixel and bit
    binary_vars = {
        (i, j): phase_model.binary_var_list(2, name=f"x_{i}_{j}")
        for i in range(image_size) for j in range(image_size)
    }

    # Neighbor constraints
    for (s, t), weight in W_st.items():
        i1, j1 = s
        i2, j2 = t
        # Add quadratic term for neighbors
        phase_model.add_constraint(
            sum(2 ** b * binary_vars[(i2, j2)][b] for b in range(2)) -
            sum(2 ** b * binary_vars[(i1, j1)][b] for b in range(2)) -
            a_st[(s, t)] <= 0
        )

    # Pixel constraints
    for (i, j), weight in omega_s.items():
        phase_model.add_constraint(
            sum(2 ** b * binary_vars[(i, j)][b] for b in range(2)) - a_s[
                (i, j)] <= 0
        )

    # Objective function
    obj_fn = sum(
        weight * (sum(2 ** b * binary_vars[(i2, j2)][b] for b in range(2)) -
                  sum(2 ** b * binary_vars[(i1, j1)][b] for b in range(2)) -
                  a_st[(s, t)]) ** 2
        for (s, t), weight in W_st.items()
    )
    obj_fn += sum(
        weight * (sum(2 ** b * binary_vars[(i, j)][b] for b in range(2)) - a_s[
            (i, j)]) ** 2
        for (i, j), weight in omega_s.items()
    )
    phase_model.set_objective("min", obj_fn)

    return phase_model


def main():
    # Example setup for a small 4x4 image
    image_size = 4
    W_st, a_st, a_s, omega_s = generate_data(image_size, noise_level=0.1)

    # Create QPLEX model
    phase_model = model_phase_unwrapping(image_size, W_st, a_st, omega_s, a_s)

    # Define execution parameters
    execution_params = {
        "provider": "d-wave",
        "backend": "advantage",
        "verbose": True,
        "penalty": 10,
        "algorithm": "quantum_annealing",
        "shots": 1024,
        # "provider_options": {
        #     "annealing_time": 20,
        # }
    }

    # Solve the problem using QPLEX
    phase_model.solve("quantum", Options(**execution_params))

    # Print the solution
    print(phase_model.print_solution())


if __name__ == "__main__":
    main()
