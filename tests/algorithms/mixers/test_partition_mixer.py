from qplex.algorithms.mixers.partition_mixer import PartitionMixer


class TestPartitionMixer:
    """Test suite for the PartitionMixer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mixer = PartitionMixer()

    def test_generate_circuit_small(self):
        """Test circuit generation with a small number of qubits."""
        n_qubits = 2
        theta = "beta[0]"

        expected_circuit = [
            "swap q[0], q[1];",
            "rz(beta[0]) q[0];",
            "rz(beta[0]) q[1];"
        ]

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert circuit == expected_circuit
        assert len(circuit) == 3

    def test_generate_circuit_medium_even_qubits(self):
        """Test circuit generation with a medium even number of qubits."""
        n_qubits = 4
        theta = "beta[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        expected_pairs = 2
        expected_instructions = expected_pairs * 3
        assert len(circuit) == expected_instructions

        pair_indices = [(0, 1), (2, 3)]
        instruction_index = 0
        for i, j in pair_indices:
            assert circuit[instruction_index] == f"swap q[{i}], q[{j}];"
            assert circuit[instruction_index + 1] == f"rz({theta}) q[{i}];"
            assert circuit[instruction_index + 2] == f"rz({theta}) q[{j}];"
            instruction_index += 3

    def test_generate_circuit_medium_odd_qubits(self):
        """Test circuit generation with a medium odd number of qubits."""
        n_qubits = 5
        theta = "beta[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        expected_pairs = 2
        expected_instructions = expected_pairs * 3
        assert len(circuit) == expected_instructions

        for instruction in circuit:
            assert "q[4]" not in instruction

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
        n_qubits = 2
        theta = "pi/4 + beta[0]*gamma[1]"

        circuit = self.mixer.generate_circuit(n_qubits, theta)

        assert f"rz({theta}) q[0];" in circuit
        assert f"rz({theta}) q[1];" in circuit
