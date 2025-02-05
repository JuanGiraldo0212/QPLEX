from enum import Enum
from typing import List

from qplex.algorithms.mixers.cardinality_mixer import CardinalityMixer
from qplex.algorithms.mixers.standard_mixer import StandardMixer
from qplex.algorithms.mixers.quantum_mixer import QuantumMixer
from qplex.algorithms.mixers.partition_mixer import PartitionMixer
from qplex.algorithms.mixers.composite_mixer import CompositeMixer
from qplex.algorithms.mixers.inequality_mixer import InequalityMixer
from qplex.utils.model_utils import ConstraintInfo


class ConstraintType(Enum):
    UNCONSTRAINED = "unconstrained"
    CARDINALITY = "cardinality"
    PARTITION = "partition"
    INEQUALITY = "inequality"
    MULTIPLE = "multiple"


class MixerFactory:
    """Factory class for creating appropriate quantum mixers.

    Creates mixer instances based on problem constraints, supporting single
    constraint types and composite mixers for multiple constraints.
    """

    _MIXER_MAP = {
        ConstraintType.CARDINALITY: CardinalityMixer,
        ConstraintType.PARTITION: PartitionMixer,
        ConstraintType.INEQUALITY: InequalityMixer,
        ConstraintType.UNCONSTRAINED: StandardMixer
    }

    @classmethod
    def get_mixer(cls, constraint_info: ConstraintInfo) -> QuantumMixer:
        """Create appropriate mixer based on constraint information.

        Parameters
        ----------
        constraint_info : ConstraintInfo
            Problem constraint metadata

        Returns
        -------
        QuantumMixer
            Appropriate mixer instance for given constraints
        """
        if not constraint_info.type:
            return StandardMixer()

        constraints = cls._get_all_constraints(constraint_info)
        if len(constraints) > 1:
            return cls._create_composite_mixer(constraints)

        mixer_class = cls._MIXER_MAP.get(constraint_info.type, StandardMixer)
        return mixer_class()

    @classmethod
    def _get_all_constraints(cls, constraint_info: ConstraintInfo) -> List[
        ConstraintType]:
        """Extract all constraint types from constraint info.

        Parameters
        ----------
        constraint_info : ConstraintInfo
            Problem constraint metadata

        Returns
        -------
        List[ConstraintType]
            List of unique constraint types
        """
        constraints = []
        if constraint_info.type != ConstraintType.UNCONSTRAINED:
            constraints.append(constraint_info.type)

        if constraint_info.parameters:
            if 'additional_constraints' in constraint_info.parameters:
                constraints.extend(
                    constraint_info.parameters['additional_constraints'])
        return list(set(constraints))

    @classmethod
    def _create_composite_mixer(cls, constraints: List[
        ConstraintType]) -> QuantumMixer:
        """Create composite mixer for multiple constraint types.

        Parameters
        ----------
        constraints : List[ConstraintType]
            List of constraints to handle

        Returns
        -------
        QuantumMixer
            CompositeMixer instance combining appropriate mixers
        """
        mixers = []
        for constraint in constraints:
            mixer_class = cls._MIXER_MAP.get(constraint)
            if mixer_class:
                mixers.append(mixer_class())
        return CompositeMixer(mixers)
