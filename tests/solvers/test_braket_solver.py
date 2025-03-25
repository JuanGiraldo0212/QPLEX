import pytest
from unittest.mock import Mock, patch

from qplex.solvers.braket_solver import BraketSolver


class TestBraketSolver:
    @pytest.fixture
    def mock_solver(self):
        """Fixture for a mocked BraketSolver instance"""
        with patch('braket.aws.AwsDevice'), \
                patch('braket.devices.LocalSimulator'), \
                patch('braket.ir.openqasm.Program'):
            solver = BraketSolver(
                shots=1000,
                backend="test_backend",
                device_parameters={"param1": "value1"}
            )

            return solver

    def test_parse_input(self, mock_solver):
        """
        Test parse_input correctly converts a circuit string to an
        OpenQASMProgram
        """
        circuit = """
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """

        mock_program = Mock()
        mock_program_class = Mock(return_value=mock_program)

        with patch('braket.ir.openqasm.Program', mock_program_class):
            result = mock_solver.parse_input(circuit)

            mock_program_class.assert_called_once()
            call_args = mock_program_class.call_args[1]
            assert "source" in call_args
            assert "OPENQASM 3.0;" in call_args["source"]
            assert "cnot" in call_args["source"]
            assert "cx" not in call_args["source"]

            assert result == mock_program

    def test_parse_response(self, mock_solver):
        """Test parse_response correctly extracts measurement counts"""
        mock_response = Mock()
        mock_response.measurement_counts = {"00": 500, "11": 500}

        result = mock_solver.parse_response(mock_response)

        assert result == {"00": 500, "11": 500}

    def test_select_backend_aws_device(self, mock_solver):
        """Test select_backend returns an AwsDevice when backend is not 'simulator'"""
        mock_solver._backend = "aws/device"

        mock_aws_device = Mock()
        mock_aws_device_class = Mock(return_value=mock_aws_device)

        with patch('braket.aws.AwsDevice', mock_aws_device_class):
            result = mock_solver.select_backend(2)

            mock_aws_device_class.assert_called_once_with(
                "arn:aws:braket:::aws/device")

            assert result == mock_aws_device

    def test_select_backend_local_simulator(self, mock_solver):
        """Test select_backend returns a LocalSimulator when backend is 'simulator'"""
        mock_solver._backend = "simulator"

        mock_local_simulator = Mock()
        mock_local_simulator_class = Mock(return_value=mock_local_simulator)

        with patch('braket.devices.LocalSimulator',
                   mock_local_simulator_class):
            result = mock_solver.select_backend(2)

            mock_local_simulator_class.assert_called_once_with(
                backend="braket_sv")

            assert result == mock_local_simulator

    def test_initialization_with_device_parameters(self):
        """Test that the solver correctly initializes with provider-specific
        device parameters"""
        custom_device_params = {
            "disableQubitRewiring": True,
            "ionq": {
                "noiseMitigation": {
                    "zne": {
                        "noiseScalingFactors": [1.0, 3.0, 5.0]
                    }
                }
            }
        }

        with patch('braket.aws.AwsDevice'), \
                patch('braket.devices.LocalSimulator'), \
                patch('braket.ir.openqasm.Program'):
            solver = BraketSolver(
                shots=1000,
                backend="test_backend",
                device_parameters=custom_device_params
            )

            assert solver.device_parameters == custom_device_params
            assert solver.device_parameters["disableQubitRewiring"] is True
            assert "ionq" in solver.device_parameters
            assert "noiseMitigation" in solver.device_parameters["ionq"]
