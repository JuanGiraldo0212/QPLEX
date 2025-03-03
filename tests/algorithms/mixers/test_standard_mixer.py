from qplex.algorithms.mixers.standard_mixer import StandardMixer


class TestStandardMixer:
    """Test suite for the StandardMixer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mixer = StandardMixer()

    def test_generate_circuit_small(self):
        """Test circuit generation with a small number of qubits."""
        n_qubits = 3
        theta = "beta[0]"

        expected_circuit = [
            "rx(2 * beta[0]) q[0];",
            "rx(2 * beta[0]) q[1];",
            "rx(2 * beta[0]) q[2];"
        ]

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == expected_circuit
        assert len(circuit) == n_qubits

    def test_generate_circuit_large(self):
        """Test circuit generation with a larger number of qubits."""
        n_qubits = 10
        theta = "beta[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert len(circuit) == n_qubits
        for i in range(n_qubits):
            assert circuit[i] == f"rx(2 * {theta}) q[{i}];"

    def test_generate_circuit_edge_case_zero_qubits(self):
        """Test circuit generation with zero qubits (edge case)."""
        n_qubits = 0
        theta = "beta[0]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == []

    def test_generate_circuit_with_complex_parameter(self):
        """Test circuit generation with a complex theta parameter."""
        n_qubits = 2
        theta = "pi/4 + beta[0]*gamma[1]"

        expected_circuit = [
            f"rx(2 * {theta}) q[0];",
            f"rx(2 * {theta}) q[1];"
        ]

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == expected_circuit
