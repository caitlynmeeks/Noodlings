# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Quantum API - Scriptable quantum computation interface.
#
#   Provides REAL quantum computation for ScriptedFacets: - Q...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.quantum_api
# PURPOSE:  Quantum Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   QuantumAPI
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import random
import time
from typing import Dict, Any, Optional, List

# IBM Quantum imports
try:
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    QiskitRuntimeService = None
    SamplerV2 = None


class QuantumAPI:
    """
    Quantum computation API for scripting.

    Provides REAL quantum mechanics via IBM Quantum hardware.
    Falls back to simulation when offline or no API key configured.

    The cat's fate is determined by actual quantum collapse!
    """

    def __init__(self, auto_connect: bool = False):
        """
        Initialize QuantumAPI.

        Args:
            auto_connect: If True, automatically connect to IBM Quantum
                         using IBM_QUANTUM_API_KEY from environment
        """
        self._measurement_count = 0
        self._backend = 'simulator'  # 'simulator' or 'ibmq'
        self._ibm_api_key = None
        self._ibm_service = None
        self._ibm_backend = None
        self._last_job_id = None

        if auto_connect:
            self.connect_from_env()

    def connect_from_env(self) -> Dict[str, Any]:
        """
        Connect to IBM Quantum using API key from environment.

        Looks for IBM_QUANTUM_API_KEY in environment variables or .env file.

        Returns:
            Connection status dict
        """
        api_key = os.environ.get('IBM_QUANTUM_API_KEY')

        if not api_key:
            # Try loading from .env file
            try:
                from pathlib import Path
                env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
                if env_path.exists():
                    with open(env_path) as f:
                        for line in f:
                            if line.startswith('IBM_QUANTUM_API_KEY='):
                                api_key = line.split('=', 1)[1].strip()
                                break
            except Exception:
                pass

        if api_key:
            return self.set_backend('ibmq', api_key)
        else:
            return {
                'connected': False,
                'backend': 'simulator',
                'error': 'IBM_QUANTUM_API_KEY not found in environment'
            }

    def measure_qubit(self, shots: int = 1) -> Dict[str, Any]:
        """
        Measure a qubit in superposition.

        Creates a qubit in |psi> = (|0> + |1>)/sqrt(2) superposition
        and collapses it via measurement.

        When backend is 'ibmq', uses REAL IBM Quantum hardware!
        The qubit physically exists in superposition until measured.

        Args:
            shots: Number of measurements (default 1 for single definite outcome)

        Returns:
            Dict with:
                - 'result': 0 or 1 (collapsed state)
                - 'probability': The random value that determined collapse
                - 'shots': Number of measurements performed
                - 'counts': Dict of {0: count, 1: count} if shots > 1
                - 'measurement_id': Unique ID for this measurement
                - 'backend': 'ibmq' or 'simulator'
                - 'job_id': IBM job ID (if using real quantum)

        Example (JavaScript):
            var q = context.noodle.quantum.measure_qubit();
            if (q.result == 0) {
                // Collapsed to |0>
            }
        """
        self._measurement_count += 1

        # Use real IBM Quantum if configured
        if self._backend == 'ibmq' and self._ibm_service is not None:
            return self._measure_qubit_ibmq(shots)

        # Fallback to simulation
        return self._measure_qubit_simulated(shots)

    def _measure_qubit_simulated(self, shots: int) -> Dict[str, Any]:
        """Simulated quantum measurement (fallback)."""
        # High-entropy seed
        random.seed(int(time.time_ns()) ^ self._measurement_count ^ id(self))

        if shots == 1:
            prob = random.random()
            result = 0 if prob < 0.5 else 1

            return {
                'result': result,
                'probability': prob,
                'shots': 1,
                'counts': {0: 1 if result == 0 else 0, 1: 1 if result == 1 else 0},
                'measurement_id': f"qm_{self._measurement_count}",
                'state': '|0>' if result == 0 else '|1>',
                'backend': 'simulator',
                'job_id': None
            }
        else:
            counts = {0: 0, 1: 0}
            for _ in range(shots):
                result = 0 if random.random() < 0.5 else 1
                counts[result] += 1

            dominant = 0 if counts[0] >= counts[1] else 1

            return {
                'result': dominant,
                'probability': counts[0] / shots,
                'shots': shots,
                'counts': counts,
                'measurement_id': f"qm_{self._measurement_count}",
                'state': '|0>' if dominant == 0 else '|1>',
                'backend': 'simulator',
                'job_id': None
            }

    def _measure_qubit_ibmq(self, shots: int) -> Dict[str, Any]:
        """
        REAL quantum measurement on IBM Quantum hardware.

        Creates a quantum circuit with a Hadamard gate (superposition)
        and measures the qubit. The result is determined by actual
        quantum mechanics - not pseudo-random numbers!
        """
        if not QISKIT_AVAILABLE:
            print("[QuantumAPI] Qiskit not available, falling back to simulator")
            return self._measure_qubit_simulated(shots)

        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Build quantum circuit: |0> -> H -> measure
            # This creates true |+> = (|0> + |1>)/sqrt(2) superposition
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Hadamard gate creates superposition
            qc.measure(0, 0)  # Collapse the wavefunction!

            # Transpile circuit to target hardware's native gate set
            # IBM's newer quantum computers use SX, RZ, CX basis gates
            pm = generate_preset_pass_manager(backend=self._ibm_backend, optimization_level=1)
            transpiled_qc = pm.run(qc)

            # Run on real quantum hardware
            sampler = SamplerV2(self._ibm_backend)
            job = sampler.run([transpiled_qc], shots=shots)
            self._last_job_id = job.job_id()

            print(f"[QuantumAPI] Job submitted: {self._last_job_id}")
            print(f"[QuantumAPI] Waiting for quantum computer...")

            # Wait for results from the quantum computer
            result = job.result()

            # Extract counts from the result
            # New Qiskit format: result[0].data.<classical_register>.get_counts()
            pub_result = result[0]

            # Get the classical register name (usually 'c' or 'meas')
            data = pub_result.data
            # Find the classical register with measurement data
            counts_raw = None
            for attr in dir(data):
                if not attr.startswith('_'):
                    try:
                        reg_data = getattr(data, attr)
                        if hasattr(reg_data, 'get_counts'):
                            counts_raw = reg_data.get_counts()
                            break
                    except Exception:
                        pass

            if counts_raw is None:
                # Fallback: try direct access
                counts_raw = {'0': 0, '1': 0}

            # Convert to our format (keys might be '0' or '1')
            counts = {0: counts_raw.get('0', 0), 1: counts_raw.get('1', 0)}

            # Determine result
            if shots == 1:
                result_val = 0 if counts[0] > 0 else 1
                prob = 0.5  # True quantum - we don't know the "probability"
            else:
                result_val = 0 if counts[0] >= counts[1] else 1
                prob = counts[0] / shots

            return {
                'result': result_val,
                'probability': prob,
                'shots': shots,
                'counts': counts,
                'measurement_id': f"qm_{self._measurement_count}",
                'state': '|0>' if result_val == 0 else '|1>',
                'backend': 'ibmq',
                'job_id': self._last_job_id,
                'backend_name': self._ibm_backend.name if self._ibm_backend else None
            }

        except Exception as e:
            print(f"[QuantumAPI] IBM Quantum error: {e}, falling back to simulator")
            return self._measure_qubit_simulated(shots)

    def schrodingers_cat(self) -> Dict[str, Any]:
        """
        Perform Schrodinger's Cat experiment.

        Uses REAL quantum mechanics (if connected to IBM Quantum):
        - Cat exists in quantum superposition (alive AND sassy ghost)
        - Single qubit measurement determines outcome
        - |0> = Cat is ALIVE, |1> = Cat is a SASSY GHOST

        Returns:
            Dict with:
                - 'is_alive': True if |0>, False if |1>
                - 'measurement': The quantum random value (0.0 to 1.0)
                - 'state': '|0>' or '|1>'
                - 'outcome': 'alive' or 'ghost'
                - 'cat_name': 'Schrodinger' or 'Quantum Whiskers'
                - 'description': Narrative description
                - 'backend': 'ibmq' or 'simulator'
                - 'job_id': IBM job ID (if real quantum)
                - 'backend_name': Name of quantum computer used

        Example (JavaScript):
            var cat = context.noodle.quantum.schrodingers_cat();
            if (cat.is_alive) {
                console.log("The cat emerges alive!");
            } else {
                console.log("A sassy ghost cat floats out!");
            }
        """
        # Perform quantum measurement - REAL if connected to IBM Quantum!
        q = self.measure_qubit(shots=1)

        is_alive = q['result'] == 0

        return {
            'is_alive': is_alive,
            'measurement': q['probability'],
            'state': q['state'],
            'outcome': 'alive' if is_alive else 'ghost',
            'cat_name': 'Schrodinger' if is_alive else 'Quantum Whiskers',
            'description': (
                "The wavefunction collapsed to |0> - the cat is ALIVE! "
                "A joyful cartoon cat bounds out of the box!"
            ) if is_alive else (
                "The wavefunction collapsed to |1> - the cat is a SASSY GHOST! "
                "An adorable, glowing ghost-cat floats out, ready for fun!"
            ),
            'recipe': 'schrodinger_alive_cat' if is_alive else 'schrodinger_ghost_cat',
            'facet_assembly': 'schrodinger_alive_cat' if is_alive else 'schrodinger_ghost_cat',
            # Propagate quantum backend info
            'backend': q.get('backend'),
            'job_id': q.get('job_id'),
            'backend_name': q.get('backend_name')
        }

    def execute_canvas(self, canvas_path: str, inputs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a Neural Canvas (.nncanvas) file.

        Loads the canvas, runs test execution, and returns results.

        Args:
            canvas_path: Path to .nncanvas file (relative to facet_assemblies/charm_networks/)
            inputs: Optional input values to override NUMBER_INPUT nodes

        Returns:
            Dict with:
                - 'success': True if execution succeeded
                - 'node_outputs': Dict of node_id -> outputs
                - 'error': Error message if failed
                - 'execution_time_ms': Execution time

        Example (JavaScript):
            var result = context.noodle.quantum.execute_canvas(
                "tutorials/08_schrodingers_cat.nncanvas"
            );
            if (result.success) {
                // Process node_outputs
            }
        """
        try:
            # Resolve path
            from pathlib import Path

            # Try relative to charm_networks
            base_path = Path(__file__).parent.parent.parent / 'facet_assemblies' / 'charm_networks'
            full_path = base_path / canvas_path

            if not full_path.exists():
                # Try as absolute path
                full_path = Path(canvas_path)

            if not full_path.exists():
                return {
                    'success': False,
                    'error': f"Canvas not found: {canvas_path}",
                    'node_outputs': {},
                    'execution_time_ms': 0
                }

            # Load and execute canvas
            from noodlestudio.core.neural_canvas.neural_graph import NeuralGraph
            from noodlestudio.core.neural_canvas.test_executor import CanvasTestExecutor
            from noodlestudio.core.neural_canvas.neural_node import NodeType

            graph = NeuralGraph.from_json(str(full_path))

            # Apply input overrides if provided
            if inputs:
                for node_id, node in graph.nodes.items():
                    if node.type == NodeType.NUMBER_INPUT:
                        if node.name in inputs:
                            node.params['value'] = inputs[node.name]

            # Execute
            executor = CanvasTestExecutor(graph)
            result = executor.execute()

            # Convert tensors to lists for JSON serialization
            outputs = {}
            for node_id, node_out in result.node_outputs.items():
                outputs[node_id] = {}
                for key, value in node_out.items():
                    if hasattr(value, 'tolist'):
                        outputs[node_id][key] = value.tolist()
                    else:
                        outputs[node_id][key] = value

            return {
                'success': result.success,
                'node_outputs': outputs,
                'error': result.error if not result.success else None,
                'execution_time_ms': result.execution_time_ms,
                'graph_name': graph.name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'node_outputs': {},
                'execution_time_ms': 0
            }

    def entangle(self, qubit_count: int = 2) -> Dict[str, Any]:
        """
        Create entangled qubits.

        Simulates Bell state entanglement: when measured, entangled qubits
        show correlated results regardless of measurement order.

        Args:
            qubit_count: Number of qubits to entangle (default 2)

        Returns:
            Dict with:
                - 'results': List of measurement results (all same due to entanglement)
                - 'correlation': 1.0 (perfect correlation)
                - 'state': '|00...>' or '|11...>'

        Example (JavaScript):
            var ent = context.noodle.quantum.entangle(3);
            // All 3 qubits will measure to same value (correlated)
        """
        # All entangled qubits collapse together
        random.seed(int(time.time_ns()) ^ id(self))
        shared_result = 0 if random.random() < 0.5 else 1

        results = [shared_result] * qubit_count
        state = '|' + ''.join(str(r) for r in results) + '>'

        return {
            'results': results,
            'correlation': 1.0,
            'state': state,
            'qubit_count': qubit_count,
            'description': (
                f"All {qubit_count} qubits collapsed to {shared_result} - "
                "quantum entanglement maintains correlation!"
            )
        }

    def set_backend(self, backend: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Set quantum backend.

        Args:
            backend: 'simulator' or 'ibmq'
            api_key: IBM Quantum API key (required for 'ibmq')

        Returns:
            Dict with connection status and available backends

        Example (JavaScript):
            var status = context.noodle.quantum.set_backend('ibmq', 'your-api-key');
            console.log(status.connected);  // true if connected
        """
        self._backend = backend
        self._ibm_api_key = api_key

        if backend == 'ibmq' and api_key:
            return self._connect_ibmq(api_key)

        return {
            'connected': False,
            'backend': 'simulator',
            'message': 'Using simulator (no API key provided)'
        }

    def _connect_ibmq(self, api_key: str) -> Dict[str, Any]:
        """
        Connect to IBM Quantum service.

        Saves credentials and selects the least busy backend.
        """
        if not QISKIT_AVAILABLE:
            return {
                'connected': False,
                'backend': 'simulator',
                'error': 'Qiskit not installed. Run: pip install qiskit qiskit-ibm-runtime'
            }

        try:
            # Save account (overwrites if exists)
            # Note: channel changed from 'ibm_quantum' to 'ibm_quantum_platform' in 2024
            QiskitRuntimeService.save_account(
                channel='ibm_quantum_platform',
                token=api_key,
                overwrite=True
            )

            # Connect to service
            self._ibm_service = QiskitRuntimeService(channel='ibm_quantum_platform')

            # Get least busy backend (or use a simulator for testing)
            # For free accounts, we use ibm_brisbane, ibm_kyoto, etc.
            backends = self._ibm_service.backends(
                simulator=False,
                operational=True,
                min_num_qubits=1
            )

            if backends:
                # Sort by queue length and pick least busy
                self._ibm_backend = self._ibm_service.least_busy(
                    simulator=False,
                    operational=True,
                    min_num_qubits=1
                )
                backend_name = self._ibm_backend.name
                num_qubits = self._ibm_backend.num_qubits
            else:
                # Fallback to simulator if no real backends available
                self._ibm_backend = self._ibm_service.backend('ibmq_qasm_simulator')
                backend_name = 'ibmq_qasm_simulator'
                num_qubits = 32

            return {
                'connected': True,
                'backend': backend_name,
                'num_qubits': num_qubits,
                'message': f'Connected to {backend_name} ({num_qubits} qubits)',
                'available_backends': [b.name for b in backends] if backends else []
            }

        except Exception as e:
            self._ibm_service = None
            self._ibm_backend = None
            return {
                'connected': False,
                'backend': 'simulator',
                'error': str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get quantum measurement statistics.

        Returns:
            Dict with measurement count and backend info
        """
        stats = {
            'measurement_count': self._measurement_count,
            'backend': self._backend,
            'has_ibm_key': self._ibm_api_key is not None,
            'connected': self._ibm_service is not None,
            'last_job_id': self._last_job_id
        }

        if self._ibm_backend:
            stats['backend_name'] = self._ibm_backend.name
            stats['num_qubits'] = self._ibm_backend.num_qubits

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with method placeholders
        """
        return {
            'measure_qubit': '__quantum_measure_qubit__',
            'schrodingers_cat': '__quantum_schrodingers_cat__',
            'execute_canvas': '__quantum_execute_canvas__',
            'entangle': '__quantum_entangle__',
            'set_backend': '__quantum_set_backend__',
            'get_stats': '__quantum_get_stats__'
        }

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
