from unittest.mock import patch, MagicMock
from qplex.algorithms.mixers.composite_mixer import CompositeMixer
from qplex.algorithms.mixers.standard_mixer import StandardMixer
from qplex.algorithms.mixers.cardinality_mixer import CardinalityMixer


class TestCompositeMixer:
    """Test suite for the CompositeMixer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_mixer1 = MagicMock()
        self.mock_mixer2 = MagicMock()
        self.composite_mixer = CompositeMixer(
            [self.mock_mixer1, self.mock_mixer2])

        self.standard_mixer = StandardMixer()
        self.cardinality_mixer = CardinalityMixer()
        self.real_composite_mixer = CompositeMixer(
            [self.standard_mixer, self.cardinality_mixer])

    def test_initialization_empty_mixers(self):
        """Test initialization with an empty list of mixers."""
        empty_mixer = CompositeMixer([])
        assert empty_mixer.mixers == []

    def test_generate_circuit_calls_component_mixers(self):
        """Test that generate_circuit calls each component mixer's generate_circuit method."""
        n_qubits = 3
        theta = "beta[0]"

        self.mock_mixer1.generate_circuit.return_value = ["instruction1;",
                                                          "instruction2;"]
        self.mock_mixer2.generate_circuit.return_value = ["instruction3;",
                                                          "instruction4;"]

        circuit = self.composite_mixer.generate_circuit(n_qubits, theta)

        self.mock_mixer1.generate_circuit.assert_called_once_with(n_qubits,
                                                                  theta)
        self.mock_mixer2.generate_circuit.assert_called_once_with(n_qubits,
                                                                  theta)

        assert circuit == ["instruction1;", "instruction2;", "instruction3;",
                           "instruction4;"]

    def test_generate_circuit_empty_mixers(self):
        """Test generate_circuit with an empty list of mixers."""
        empty_mixer = CompositeMixer([])
        circuit = empty_mixer.generate_circuit(3, "beta[0]")
        assert circuit == []

    def test_generate_circuit_with_different_parameters(self):
        """Test generate_circuit with different n_qubits and theta values."""
        test_cases = [
            (0, "beta[0]"),
            (1, "beta[0]"),
            (5, "pi/2"),
            (10, "beta[2] * gamma[3]")
        ]

        for n_qubits, theta in test_cases:
            self.composite_mixer.generate_circuit(n_qubits, theta)

            self.mock_mixer1.generate_circuit.assert_any_call(n_qubits, theta)
            self.mock_mixer2.generate_circuit.assert_any_call(n_qubits, theta)

    @patch('qplex.algorithms.mixers.composite_mixer.QuantumMixer')
    def test_integration_with_real_mixers(self, _):
        """Integration test with real StandardMixer and CardinalityMixer instances."""
        n_qubits = 2
        theta = "beta[0]"

        standard_circuit = self.standard_mixer.generate_circuit(n_qubits,
                                                                theta)
        cardinality_circuit = self.cardinality_mixer.generate_circuit(n_qubits,
                                                                      theta)
        expected_circuit = standard_circuit + cardinality_circuit

        actual_circuit = self.real_composite_mixer.generate_circuit(n_qubits,
                                                                    theta)

        assert actual_circuit == expected_circuit
