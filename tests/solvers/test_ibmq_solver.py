import pytest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit
from qplex.solvers.ibmq_solver import IBMQSolver


class TestIBMQSolver:
    """Test suite for IBMQSolver class."""

    @patch('qiskit_ibm_runtime.QiskitRuntimeService')
    def setup_method(self, _, mock_service):
        """Set up test fixtures before each test."""
        self.token = "fake_token"
        self.shots = 1024
        self.backend = "ibm_test"
        self.optimization_level = 1

        self.mock_service = MagicMock()
        mock_service.return_value = self.mock_service

        with patch('qiskit_ibm_runtime.QiskitRuntimeService.save_account'):
            self.solver = IBMQSolver(
                token=self.token,
                shots=self.shots,
                backend=self.backend,
                optimization_level=self.optimization_level
            )

    @patch('qiskit_ibm_runtime.QiskitRuntimeService.save_account')
    @patch('qiskit_ibm_runtime.QiskitRuntimeService')
    def test_initialization(self, mock_service, mock_save_account):
        """Test initialization of IBMQSolver with parameters."""
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance

        solver = IBMQSolver(
            token=self.token,
            shots=self.shots,
            backend=self.backend,
            optimization_level=self.optimization_level
        )

        # Check initialization
        assert solver.shots == self.shots
        assert solver._backend == self.backend
        assert solver.optimization_level == self.optimization_level
        assert solver.service == mock_service_instance

        # Check QiskitRuntimeService.save_account was called
        mock_save_account.assert_called_once_with(
            channel="ibm_quantum",
            token=self.token,
            overwrite=True
        )

    @patch('qiskit_ibm_runtime.QiskitRuntimeService.save_account')
    @patch('qiskit_ibm_runtime.QiskitRuntimeService')
    def test_initialization_default_backend(self, mock_service, _):
        """Test initialization with None backend."""
        mock_service.return_value = MagicMock()

        solver = IBMQSolver(
            token=self.token,
            shots=self.shots,
            backend=None,
            optimization_level=self.optimization_level
        )

        assert solver._backend == ''

    def test_backend_property(self):
        """Test the backend property returns the _backend value."""
        assert self.solver.backend == self.solver._backend

    def test_parse_input(self):
        """Test parse_input converts OpenQASM string to QuantumCircuit."""
        test_circuit = """
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        measure q;
        """

        with patch('qiskit.qasm3.loads') as mock_loads:
            mock_qc = MagicMock(spec=QuantumCircuit)
            mock_loads.return_value = mock_qc

            result = self.solver.parse_input(test_circuit)

            # Check that qasm3.loads was called with the right string
            expected_circuit = """
        OPENQASM 3.0;
        include "stdgates.inc";
        """ + test_circuit
            mock_loads.assert_called_once_with(expected_circuit)
            assert result == mock_qc

    def test_parse_input_with_empty_circuit(self):
        """Test parse_input with an empty circuit string."""
        test_circuit = ""

        with patch('qiskit.qasm3.loads') as mock_loads:
            mock_qc = MagicMock(spec=QuantumCircuit)
            mock_loads.return_value = mock_qc

            result = self.solver.parse_input(test_circuit)

            # Check that qasm3.loads was called with the right string
            expected_circuit = """
        OPENQASM 3.0;
        include "stdgates.inc";
        """
            mock_loads.assert_called_once_with(expected_circuit)
            assert result == mock_qc

    def test_parse_input_with_complex_circuit(self):
        """Test parse_input with a complex circuit including parameters."""
        test_circuit = """
        input float[64] theta;
        qubit[3] q;
        h q[0];
        rx(theta) q[1];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q;
        """

        with patch('qiskit.qasm3.loads') as mock_loads:
            mock_qc = MagicMock(spec=QuantumCircuit)
            mock_loads.return_value = mock_qc

            result = self.solver.parse_input(test_circuit)

            expected_circuit = """
        OPENQASM 3.0;
        include "stdgates.inc";
        """ + test_circuit
            mock_loads.assert_called_once_with(expected_circuit)
            assert result == mock_qc

    @patch('qiskit.qasm3.loads')
    def test_parse_input_handles_qasm3_errors(self, mock_loads):
        """Test parse_input handles errors from qasm3.loads."""
        mock_loads.side_effect = Exception("Invalid QASM3 syntax")

        test_circuit = """
        qubit[2 q;  # Syntax error
        h q[0];
        """

        with pytest.raises(Exception) as excinfo:
            self.solver.parse_input(test_circuit)

        assert "Invalid QASM3 syntax" in str(excinfo.value)

    def test_parse_response(self):
        """Test parse_response converts backend counts to expected format."""
        response = {
            '00': 500,
            '01': 100,
            '10': 300,
            '11': 124
        }

        parsed = self.solver.parse_response(response)

        expected = {
            '00': 500,
            '10': 100,
            '01': 300,
            '11': 124
        }

        assert parsed == expected

    @patch('qiskit_aer.AerSimulator')
    def test_select_backend_simulator(self, mock_aer_simulator):
        """Test select_backend with simulator backend."""
        self.solver._backend = 'simulator'
        mock_simulator = MagicMock()
        mock_aer_simulator.return_value = mock_simulator

        result = self.solver.select_backend(2)

        mock_aer_simulator.assert_called_once()
        assert result == mock_simulator

    def test_select_backend_specified(self):
        """Test select_backend with a specified backend."""
        self.solver._backend = 'ibm_test'
        mock_backend = MagicMock()
        self.mock_service.backend.return_value = mock_backend

        result = self.solver.select_backend(2)

        self.mock_service.backend.assert_called_once_with('ibm_test')
        assert result == mock_backend

    def test_select_backend_least_busy(self):
        """Test select_backend with empty string (least busy)."""
        self.solver._backend = ''
        mock_backend = MagicMock()
        self.mock_service.least_busy.return_value = mock_backend

        result = self.solver.select_backend(5)

        self.mock_service.least_busy.assert_called_once_with(min_num_qubits=5)
        assert result == mock_backend

    @patch.object(IBMQSolver, 'parse_input')
    @patch.object(IBMQSolver, 'select_backend')
    @patch.object(IBMQSolver, 'parse_response')
    @patch(
        'qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager')
    def test_solve_with_simulator(self, mock_pass_manager, mock_parse_response,
                                  mock_select_backend, mock_parse_input):
        """Test solve method with simulator backend."""
        test_circuit = "qubit[2] q; h q[0]; cx q[0], q[1]; measure q;"

        mock_qc = MagicMock()
        mock_qc.num_qubits = 2
        mock_parse_input.return_value = mock_qc

        mock_backend = MagicMock()
        mock_select_backend.return_value = mock_backend

        mock_pm = MagicMock()
        mock_pass_manager.return_value = mock_pm

        mock_isa_circuit = MagicMock()
        mock_pm.run.return_value = mock_isa_circuit

        mock_result = MagicMock()
        mock_backend.run.return_value.result.return_value = mock_result
        mock_result.get_counts.return_value = {'00': 500, '11': 524}

        mock_parse_response.return_value = {'00': 500, '11': 524}

        self.solver._backend = 'simulator'
        result = self.solver.solve(test_circuit)

        mock_parse_input.assert_called_once_with(test_circuit)
        mock_select_backend.assert_called_once_with(2)
        mock_pass_manager.assert_called_once_with(
            backend=mock_backend,
            optimization_level=self.optimization_level
        )
        mock_pm.run.assert_called_once_with(mock_qc)
        mock_backend.run.assert_called_once_with(mock_isa_circuit)
        mock_backend.run.return_value.result.assert_called_once()
        mock_result.get_counts.assert_called_once()
        mock_parse_response.assert_called_once_with({'00': 500, '11': 524})
        assert result == {'00': 500, '11': 524}

    @patch.object(IBMQSolver, 'parse_input')
    @patch.object(IBMQSolver, 'select_backend')
    @patch.object(IBMQSolver, 'run')
    @patch.object(IBMQSolver, 'parse_response')
    @patch(
        'qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager')
    @patch('qiskit_ibm_runtime.SamplerV2')
    def test_solve_with_ibmq_backend(self, mock_sampler_v2, mock_pass_manager,
                                     mock_parse_response, mock_run,
                                     mock_select_backend, mock_parse_input):
        """Test solve method with IBMQ backend."""
        test_circuit = "qubit[2] q; h q[0]; cx q[0], q[1]; measure q;"

        mock_qc = MagicMock()
        mock_qc.num_qubits = 2
        mock_parse_input.return_value = mock_qc

        mock_backend = MagicMock()
        mock_select_backend.return_value = mock_backend

        mock_pm = MagicMock()
        mock_pass_manager.return_value = mock_pm

        mock_isa_circuit = MagicMock()
        mock_pm.run.return_value = mock_isa_circuit

        mock_sampler = MagicMock()
        mock_sampler_v2.return_value = mock_sampler

        mock_run.return_value = {'00': 500, '11': 524}

        mock_parse_response.return_value = {'00': 500, '11': 524}

        self.solver._backend = 'ibm_test'
        result = self.solver.solve(test_circuit)

        mock_parse_input.assert_called_once_with(test_circuit)
        mock_select_backend.assert_called_once_with(2)
        mock_pass_manager.assert_called_once_with(
            backend=mock_backend,
            optimization_level=self.optimization_level
        )
        mock_pm.run.assert_called_once_with(mock_qc)
        mock_sampler_v2.assert_called_once_with(mock_backend)
        mock_run.assert_called_once_with(mock_isa_circuit, mock_sampler)
        mock_parse_response.assert_called_once_with({'00': 500, '11': 524})
        assert result == {'00': 500, '11': 524}

    def test_run(self):
        """Test run method executes circuit and processes results."""
        mock_qc = MagicMock()
        mock_sampler = MagicMock()

        mock_result = MagicMock()
        mock_sampler.run.return_value.result.return_value = [mock_result]

        mock_data = MagicMock()
        mock_result.data = mock_data

        mock_bits = MagicMock()
        mock_data.c = mock_bits

        mock_bits.get_counts.return_value = {'00': 500, '11': 524}

        result = self.solver.run(mock_qc, mock_sampler)

        mock_sampler.run.assert_called_once_with([(mock_qc,)],
                                                 shots=self.shots)
        mock_sampler.run.return_value.result.assert_called_once()
        mock_bits.get_counts.assert_called_once()
        assert result == {'00': 500, '11': 524}
