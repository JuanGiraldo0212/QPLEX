import typing
import enum
import dataclasses

import qplex.algorithms
import qplex.algorithms.mixers
import qplex.algorithms.mixers.mixer_factory
import qplex.utils.model_utils


class AlgorithmType(enum.Enum):
    """Supported gate-based quantum algorithms.

    This enum defines the gate-based quantum algorithms that can be used to
    solve optimization problems in QPLEX. It currently supports QAOA (Quantum
    Approximate Optimization Algorithm), QAOAnsatz (Quantum Alternating
    Operator Ansatz), and VQE (Variational Quantum Eigensolver).
    """
    QAOA = "qaoa"
    QAO_ANSATZ = "qao-ansatz"
    VQE = "vqe"


@dataclasses.dataclass
class AlgorithmConfig:
    """Configuration dataclass for gate-based quantum algorithms.

    This class encapsulates the parameters needed to configure and instantiate
    quantum algorithms for combinatorial optimization problems.

    Attributes:
        algorithm (AlgorithmType): The type of quantum algorithm to use.
        penalty (float): The penalty coefficient for constraint violations
            in the Hamiltonian.
        seed (int): Random seed for reproducibility of results.
        p (int | None): Number of layers in the QAOA circuit. Only used for
            QAOA and QAOAnsatz algorithms.
        mixer (Any | None): Custom mixer operator for the QAOA framework.
            Only used for QAOA and QAOAnsatz algorithms.
        layers (int | None): Number of layers in the VQE ansatz circuit.
            Only used for VQE algorithm.
        ansatz (Any | None): Custom ansatz circuit for VQE. If None, uses the
            default hardware-efficient ansatz.
    """
    algorithm: AlgorithmType
    penalty: float
    seed: int
    p: typing.Optional[int] = None  # for the QAOA framework
    mixer: typing.Optional[typing.Any] = None  # for the QAOA framework
    layers: typing.Optional[int] = None  # for VQE
    ansatz: typing.Optional[typing.Any] = None  # for VQE


class AlgorithmFactory:
    """Factory class for creating quantum algorithm instances.

    This class implements the Factory pattern to instantiate appropriate
    quantum algorithms based on the provided configuration. It handles the
    creation of QAOA, QAOAnsatz, and VQE algorithm objects with their
    respective parameters.

    The factory determines the appropriate mixer operator for optimization
    problems when using QAOAnsatz, and ensures all necessary parameters are
    properly configured for each algorithm type.
    """

    @classmethod
    def get_algorithm(cls, model,
                      config: AlgorithmConfig) -> qplex.algorithms.Algorithm:
        """Creates and returns a quantum algorithm instance based on configuration.

        Parameters:
        -----------
        model: The optimization model to be solved, typically a DOcplex model
            instance containing the problem formulation.
        config (AlgorithmConfig): Configuration object specifying the algorithm
            type and its parameters.

        Returns:
        --------
        Algorithm: An instance of the specified quantum algorithm configured
            according to the provided parameters.

        Raises:
        -------
        ValueError: If the specified algorithm type is not supported.
        """
        model_constraint_info = qplex.utils.model_utils.get_model_constraint_info(
            model)

        if config.algorithm == AlgorithmType.QAOA:
            mixer = qplex.algorithms.mixers.StandardMixer()
            return qplex.algorithms.QAOA(model, p=config.p, seed=config.seed,
                                         penalty=config.penalty, mixer=mixer)

        elif config.algorithm == AlgorithmType.QAO_ANSATZ:
            mixer = config.mixer if config.mixer else (
                qplex.algorithms.mixers.mixer_factory.MixerFactory.get_mixer(
                    model_constraint_info))
            return qplex.algorithms.QAOA(model, p=config.p, seed=config.seed,
                                         penalty=config.penalty, mixer=mixer)

        elif config.algorithm == AlgorithmType.VQE:
            return qplex.algorithms.VQE(model, layers=config.layers,
                                        penalty=config.penalty,
                                        seed=config.seed, ansatz=config.ansatz)

        raise ValueError(f"Algorithm not supported: {config.algorithm}")
