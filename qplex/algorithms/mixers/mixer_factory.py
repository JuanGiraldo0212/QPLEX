from enum import Enum
from typing import List

from qplex.algorithms.mixers.quantum_mixer import QuantumMixer
from qplex.algorithms.mixers.standard_mixer import StandardMixer
from qplex.algorithms.mixers.xy_mixer import XYMixer
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

    @classmethod
    def _get_mixer_for_type(cls, constraint_type):
        """Look up the appropriate mixer class for the constraint type."""
        if constraint_type.value == ConstraintType.CARDINALITY.value:
            print('using xy mixer')
            return XYMixer
        else:
            print('using standard mixer')
            return StandardMixer

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

        mixer_class = cls._get_mixer_for_type(constraint_info.type)
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
        if constraint_info.type.value != ConstraintType.UNCONSTRAINED.value:
            constraints.append(constraint_info.type)

        if constraint_info.parameters:
            if 'additional_constraints' in constraint_info.parameters:
                constraints.extend(
                    constraint_info.parameters['additional_constraints'])
        return list(set(constraints))
