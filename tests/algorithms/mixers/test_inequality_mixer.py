from qplex.algorithms.mixers.inequality_mixer import InequalityMixer


class TestInequalityMixer:
    """Test suite for the InequalityMixer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mixer = InequalityMixer()

    def test_generate_circuit_small(self):
        """Test circuit generation with a small number of qubits."""
        n_qubits = 2
        theta = "beta[0]"

        expected_circuit = [
            "cx q[0], q[1];",
            "rz(beta[0]) q[1];",
            "cx q[0], q[1];"
        ]

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == expected_circuit
        assert len(circuit) == 3

    def test_generate_circuit_medium(self):
        """Test circuit generation with a medium number of qubits."""
        n_qubits = 4
        theta = "beta[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        expected_pairs = 3
        expected_instructions = expected_pairs * 3
        assert len(circuit) == expected_instructions

        instruction_index = 0
        for i in range(n_qubits - 1):
            assert circuit[instruction_index] == f"cx q[{i}], q[{i + 1}];"
            assert circuit[instruction_index + 1] == f"rz({theta}) q[{i + 1}];"
            assert circuit[instruction_index + 2] == f"cx q[{i}], q[{i + 1}];"
            instruction_index += 3

    def test_generate_circuit_edge_case_one_qubit(self):
        """Test circuit generation with one qubit (edge case)."""
        n_qubits = 1
        theta = "beta[0]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == []

    def test_generate_circuit_edge_case_zero_qubits(self):
        """Test circuit generation with zero qubits (edge case)."""
        n_qubits = 0
        theta = "beta[0]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == []

    def test_generate_circuit_with_complex_parameter(self):
        """Test circuit generation with a complex theta parameter."""
        n_qubits = 3
        theta = "pi/4 + beta[0]*gamma[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert "rz(pi/4 + beta[0]*gamma[1]) q[1];" in circuit
        assert "rz(pi/4 + beta[0]*gamma[1]) q[2];" in circuit
