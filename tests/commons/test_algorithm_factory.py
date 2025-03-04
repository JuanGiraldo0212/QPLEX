import pytest
from unittest.mock import Mock, patch, MagicMock

from qplex.commons.algorithm_factory import AlgorithmFactory, AlgorithmType, \
    AlgorithmConfig
from qplex.utils.model_utils import ConstraintInfo


class TestAlgorithmFactory:
    @pytest.fixture
    def mock_model(self):
        """Fixture for a mocked optimization model"""
        return Mock()

    @patch('qplex.utils.model_utils.get_model_constraint_info')
    @patch('qplex.algorithms.QAOA')
    @patch('qplex.algorithms.mixers.StandardMixer')
    def test_get_algorithm_qaoa(self, mock_standard_mixer_class,
                                mock_qaoa_class,
                                _, mock_model):
        """
        Test that get_algorithm returns a QAOA instance with StandardMixer
        for QAOA algorithm type
        """
        mock_standard_mixer = Mock()
        mock_standard_mixer_class.return_value = mock_standard_mixer

        mock_qaoa = Mock()
        mock_qaoa_class.return_value = mock_qaoa

        config = AlgorithmConfig(
            algorithm=AlgorithmType.QAOA,
            penalty=1.0,
            seed=42,
            p=2
        )

        result = AlgorithmFactory.get_algorithm(mock_model, config)

        mock_standard_mixer_class.assert_called_once()

        mock_qaoa_class.assert_called_once_with(
            mock_model, p=2, seed=42, penalty=1.0, mixer=mock_standard_mixer
        )

        assert result is mock_qaoa

    @patch('qplex.utils.model_utils.get_model_constraint_info')
    @patch('qplex.algorithms.QAOA')
    @patch('qplex.algorithms.mixers.mixer_factory.MixerFactory.get_mixer')
    def test_get_algorithm_qao_ansatz(self, mock_get_mixer, mock_qaoa_class,
                                      mock_get_model_constraint_info,
                                      mock_model):
        """
        Test that get_algorithm returns a QAOA instance with custom mixer for
        QAO_ANSATZ algorithm type
        """
        mock_constraint_info = Mock(spec=ConstraintInfo)
        mock_get_model_constraint_info.return_value = mock_constraint_info

        mock_mixer = Mock()
        mock_get_mixer.return_value = mock_mixer

        mock_qaoa = Mock()
        mock_qaoa_class.return_value = mock_qaoa

        config = AlgorithmConfig(
            algorithm=AlgorithmType.QAO_ANSATZ,
            penalty=1.0,
            seed=42,
            p=2
        )

        result = AlgorithmFactory.get_algorithm(mock_model, config)

        mock_get_model_constraint_info.assert_called_once_with(mock_model)

        mock_get_mixer.assert_called_once_with(mock_constraint_info)

        mock_qaoa_class.assert_called_once_with(
            mock_model, p=2, seed=42, penalty=1.0, mixer=mock_mixer
        )

        assert result is mock_qaoa

    @patch('qplex.utils.model_utils.get_model_constraint_info')
    @patch('qplex.algorithms.QAOA')
    def test_get_algorithm_qao_ansatz_with_custom_mixer(self,
                                                        mock_qaoa_class, _,
                                                        mock_model):
        """
        Test that get_algorithm uses provided mixer for QAO_ANSATZ when
        supplied
        """
        mock_qaoa = Mock()
        mock_qaoa_class.return_value = mock_qaoa

        custom_mixer = Mock()

        config = AlgorithmConfig(
            algorithm=AlgorithmType.QAO_ANSATZ,
            penalty=1.0,
            seed=42,
            p=2,
            mixer=custom_mixer
        )

        result = AlgorithmFactory.get_algorithm(mock_model, config)

        mock_qaoa_class.assert_called_once_with(
            mock_model, p=2, seed=42, penalty=1.0, mixer=custom_mixer
        )

        assert result is mock_qaoa

    @patch('qplex.algorithms.VQE')
    @patch('qplex.utils.model_utils.get_model_constraint_info')
    def test_get_algorithm_vqe(self, mock_get_model_constraint_info,
                               mock_vqe_class, mock_model):
        """
        Test that get_algorithm returns a VQE instance for VQE algorithm
        type
        """
        mock_vqe = Mock()
        mock_vqe_class.return_value = mock_vqe

        mock_get_model_constraint_info.return_value = Mock()

        custom_ansatz = "custom_ansatz"

        config = AlgorithmConfig(
            algorithm=AlgorithmType.VQE,
            penalty=1.0,
            seed=42,
            layers=3,
            ansatz=custom_ansatz
        )

        result = AlgorithmFactory.get_algorithm(mock_model, config)

        mock_vqe_class.assert_called_once_with(
            mock_model, layers=3, penalty=1.0, seed=42, ansatz=custom_ansatz
        )

        assert result is mock_vqe

    @patch('qplex.utils.model_utils.get_model_constraint_info')
    def test_get_algorithm_unsupported(self, _, mock_model):
        """
        Test that get_algorithm raises ValueError for unsupported algorithm
        types
        """
        mock_algorithm_type = Mock()
        mock_algorithm_type.__str__ = lambda x: "unsupported_algorithm"

        config = AlgorithmConfig(
            algorithm=mock_algorithm_type,
            penalty=1.0,
            seed=42
        )

        with pytest.raises(ValueError, match="Algorithm not supported: .*"):
            AlgorithmFactory.get_algorithm(mock_model, config)
