from unittest.mock import patch, MagicMock

from qplex.algorithms.mixers.mixer_factory import (MixerFactory,
                                                   ConstraintType,
                                                   ConstraintInfo)


class TestMixerFactory:

    @patch('qplex.algorithms.mixers.mixer_factory.StandardMixer')
    def test_get_mixer_none_type(self, mock_standard_mixer):
        mock_mixer = MagicMock()
        mock_standard_mixer.return_value = mock_mixer

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = None

        result = MixerFactory.get_mixer(constraint_info)

        mock_standard_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.StandardMixer')
    def test_get_mixer_unconstrained_type(self, mock_standard_mixer):
        mock_mixer = MagicMock()
        mock_standard_mixer.return_value = mock_mixer

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.UNCONSTRAINED
        constraint_info.parameters = {}

        result = MixerFactory.get_mixer(constraint_info)

        mock_standard_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.CardinalityMixer')
    def test_get_mixer_cardinality_type(self, mock_cardinality_mixer):
        mock_mixer = MagicMock()
        mock_cardinality_mixer.return_value = mock_mixer

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.CARDINALITY
        constraint_info.parameters = {}

        result = MixerFactory.get_mixer(constraint_info)

        mock_cardinality_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.PartitionMixer')
    def test_get_mixer_partition_type(self, mock_partition_mixer):
        mock_mixer = MagicMock()
        mock_partition_mixer.return_value = mock_mixer

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.PARTITION
        constraint_info.parameters = {}

        result = MixerFactory.get_mixer(constraint_info)

        mock_partition_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.InequalityMixer')
    def test_get_mixer_inequality_type(self, mock_inequality_mixer):
        mock_mixer = MagicMock()
        mock_inequality_mixer.return_value = mock_mixer

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.INEQUALITY
        constraint_info.parameters = {}

        result = MixerFactory.get_mixer(constraint_info)

        mock_inequality_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.StandardMixer')
    def test_get_mixer_unknown_type(self, mock_standard_mixer):
        mock_mixer = MagicMock()
        mock_standard_mixer.return_value = mock_mixer

        unknown_type = MagicMock()

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = unknown_type
        constraint_info.parameters = {}

        result = MixerFactory.get_mixer(constraint_info)

        mock_standard_mixer.assert_called_once()
        assert result == mock_mixer

    @patch('qplex.algorithms.mixers.mixer_factory.CompositeMixer')
    @patch('qplex.algorithms.mixers.mixer_factory.CardinalityMixer')
    @patch('qplex.algorithms.mixers.mixer_factory.PartitionMixer')
    def test_get_mixer_multiple_constraints(self, mock_partition_mixer,
                                            mock_cardinality_mixer,
                                            mock_composite_mixer):
        mock_cardinality = MagicMock()
        mock_partition = MagicMock()
        mock_composite = MagicMock()

        mock_cardinality_mixer.return_value = mock_cardinality
        mock_partition_mixer.return_value = mock_partition
        mock_composite_mixer.return_value = mock_composite

        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.CARDINALITY
        constraint_info.parameters = {
            'additional_constraints': [ConstraintType.PARTITION]
        }

        result = MixerFactory.get_mixer(constraint_info)

        mock_cardinality_mixer.assert_called_once()
        mock_partition_mixer.assert_called_once()
        mock_composite_mixer.assert_called_once()

        call_args = mock_composite_mixer.call_args[0][0]
        assert len(call_args) == 2
        assert mock_cardinality in call_args
        assert mock_partition in call_args

        assert result == mock_composite

    def test_get_all_constraints_single(self):
        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.CARDINALITY
        constraint_info.parameters = {}

        result = MixerFactory._get_all_constraints(constraint_info)

        assert len(result) == 1
        assert ConstraintType.CARDINALITY in result

    def test_get_all_constraints_multiple(self):
        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.CARDINALITY
        constraint_info.parameters = {
            'additional_constraints': [
                ConstraintType.PARTITION,
                ConstraintType.INEQUALITY
            ]
        }

        result = MixerFactory._get_all_constraints(constraint_info)

        result_set = set(result)
        assert len(result_set) == 3
        assert ConstraintType.CARDINALITY in result_set
        assert ConstraintType.PARTITION in result_set
        assert ConstraintType.INEQUALITY in result_set

    def test_get_all_constraints_duplicate(self):
        constraint_info = MagicMock(spec=ConstraintInfo)
        constraint_info.type = ConstraintType.CARDINALITY
        constraint_info.parameters = {
            'additional_constraints': [
                ConstraintType.CARDINALITY,  # Duplicate
                ConstraintType.PARTITION
            ]
        }

        result = MixerFactory._get_all_constraints(constraint_info)

        result_set = set(result)
        assert len(result_set) == 2
        assert ConstraintType.CARDINALITY in result_set
        assert ConstraintType.PARTITION in result_set

    @patch('qplex.algorithms.mixers.mixer_factory.CompositeMixer')
    @patch('qplex.algorithms.mixers.mixer_factory.CardinalityMixer')
    @patch('qplex.algorithms.mixers.mixer_factory.PartitionMixer')
    def test_create_composite_mixer(self, mock_partition_mixer,
                                    mock_cardinality_mixer,
                                    mock_composite_mixer):
        mock_cardinality = MagicMock()
        mock_partition = MagicMock()
        mock_composite = MagicMock()

        mock_cardinality_mixer.return_value = mock_cardinality
        mock_partition_mixer.return_value = mock_partition
        mock_composite_mixer.return_value = mock_composite

        constraints = [
            ConstraintType.CARDINALITY,
            ConstraintType.PARTITION
        ]

        result = MixerFactory._create_composite_mixer(constraints)

        mock_cardinality_mixer.assert_called_once()
        mock_partition_mixer.assert_called_once()
        mock_composite_mixer.assert_called_once_with(
            [mock_cardinality, mock_partition])

        assert result == mock_composite
