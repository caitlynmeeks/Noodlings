# IBM Quantum Integration Strategy

**Status:** Design Phase
**Author:** Commander Spock + Cadet Caity
**Date:** December 8, 2025
**Purpose:** Integrate IBM Quantum cloud hardware with existing consciousness architecture

---

## 🎯 Vision

Augment NoodleStudio's consciousness simulation with **real quantum computation** from IBM Quantum hardware. The goal is to test Penrose-Hameroff predictions about quantum effects in consciousness while maintaining practical usability.

**Key Principle:** Quantum enhancement, not quantum replacement. Classical neural computation remains primary; quantum operations provide targeted capabilities (entanglement, true randomness, binding experiments).

---

## 🧩 Current Quantum Infrastructure

### Already Implemented

✅ **EntropyService** (`applications/cmush/entropy_service.py`)
- TrueRNG V3 USB device support (reverse-diode avalanche effect, truly quantum)
- Thread-safe entropy pool (4096 bytes, background refill)
- API: `uniform()`, `randint()`, `choice()`, `expovariate()`
- Fallback to PRNG if hardware unavailable

✅ **AvalancheRNG** (`entropy_service.py:22-78`)
- Reverse diode avalanche effect entropy
- Heavy-tailed distribution (β=2.0)
- Used for quantum fluctuations in classical simulation

✅ **QuantumMicrotubuleLayer** (`noodlings/models/quantum_microtubule.py`)
- Penrose-Hameroff simulation (orchestrated objective reduction)
- Avalanche noise injection
- Entanglement simulation (spatial correlations via convolution)
- Collapse events (threshold-based state reduction)
- **Status:** Implemented but NOT integrated into CharmNetwork yet

✅ **Settings UI** (NoodleStudio → Entropy Service menu)
- Configure hardware RNG device path
- Enable/disable hardware entropy

### Not Yet Implemented

❌ **IBM Quantum Cloud Integration**
- No Qiskit integration
- No quantum circuit execution
- No cloud backend connection

❌ **Quantum Facet Type**
- No visual node for quantum operations in Facets Editor

❌ **Hybrid Quantum-Classical Execution**
- No workflow for combining MLX + IBM Quantum

---

## 📐 Integration Architecture

### Three Levels of Quantum Integration

#### Level 1: Entropy Enhancement (Existing)
**Status:** ✅ Complete
- TrueRNG provides quantum randomness to classical algorithms
- Used in: Agent decisions, ContextIntelligence sampling, ScriptedFacet random()

#### Level 2: Quantum Microtubule Simulation (Existing)
**Status:** ✅ Complete but unused
- QuantumMicrotubuleLayer simulates Penrose-Hameroff dynamics
- Uses AvalancheRNG for quantum-like noise
- **Integration Point:** Add to CharmNetwork via Neural Canvas

#### Level 3: True Quantum Computation (NEW)
**Status:** ❌ Not implemented
- IBM Quantum cloud backend
- Real entanglement on superconducting qubits
- Quantum-classical hybrid execution
- **This document focuses on Level 3**

---

## 🔌 Integration Points

### Option A: Quantum Facet (Recommended)

**Location:** New facet type in Facets Editor

```
INCOMING
  ↓
CHARM_NET (classical affect processing)
  ↓
QUANTUM_BINDING (IBM Quantum facet) ← NEW
  ├─ Input: phenomenal_state (40-D)
  ├─ Operation: Entanglement circuit on IBM hardware
  └─ Output: entangled_features (40-D)
  ↓
CONTEXT_INTELLIGENCE
  ↓
Red's Mind (LLMFacet)
  ↓
OUTGOING
```

**Workflow:**
1. CharmNetwork produces 40-D phenomenal state
2. QuantumBindingFacet encodes state → quantum circuit
3. Submit circuit to IBM Quantum cloud
4. Measure results → decode to 40-D vector
5. Pass entangled features downstream

**Advantages:**
- Clean separation of concerns
- Can be toggled on/off (simulator vs hardware)
- Visible in Facets Editor topology
- No modification to existing facets

**Disadvantages:**
- Network latency (~100ms-2s per call)
- API rate limits
- Requires queue wait time

---

### Option B: Quantum Layer in CharmNetwork (Neural Canvas)

**Location:** Inside CharmNetwork, visible in Neural Canvas

```
Fast LSTM (16-D)
  ↓
Medium LSTM (16-D)
  ↓
Quantum Entanglement Layer (16-D) ← NEW (IBM Quantum)
  ↓
Slow GRU (8-D)
  ↓
Affect Head (5-D)
```

**Workflow:**
1. Medium LSTM produces 16-D hidden state
2. QuantumEntanglementLayer encodes → quantum circuit
3. Execute on IBM hardware (or simulator)
4. Decode results → 16-D vector
5. Feed to Slow GRU

**Advantages:**
- Tighter integration with neural computation
- Tests quantum effects at specific temporal scale (minutes)
- Visible in Neural Canvas topology

**Disadvantages:**
- Latency blocks CharmNetwork forward pass
- Every agent step waits for quantum execution
- More invasive modification

---

### Option C: Quantum Service (Background)

**Location:** Standalone service, called by facets on-demand

```
┌─────────────────────────────────────┐
│  noodleMUSH Server                  │
│  ├─ Facet Executor                  │
│  ├─ CharmNetwork                    │
│  └─ Entropy Service                 │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Quantum Service (new process)      │
│  ├─ IBM Quantum client (Qiskit)     │
│  ├─ Circuit queue                   │
│  ├─ Result cache                    │
│  └─ Fallback simulator              │
└─────────────────────────────────────┘
```

**Workflow:**
1. Facet calls `quantum_service.entangle(state_vector)`
2. Service queues circuit submission to IBM
3. Returns immediately with cached/simulated result
4. Background thread fetches real quantum results
5. Next call uses real result (eventual consistency)

**Advantages:**
- Non-blocking (no latency in agent loop)
- Can batch multiple requests
- Fallback to simulator when offline
- Useful for experiments, not real-time decisions

**Disadvantages:**
- Eventual consistency (results delayed by 1-2 steps)
- More complex architecture
- Cache invalidation logic required

---

## 🎯 Recommended Approach

**Phase 1: Quantum Facet (Option A)**

Why:
- Least invasive
- Can be added without modifying CharmNetwork
- Easy to toggle simulator vs hardware
- Clear separation of quantum vs classical
- Matches NoodleStudio's modular philosophy

**Implementation:**
1. Create `QuantumBindingFacet` class
2. Add to facet registry (like LLMFacet, ScriptedFacet)
3. Add visual node type in Facets Editor
4. User drags QuantumBinding between CharmNet and ContextIntelligence
5. Configure in Inspector: backend (simulator/ibm_quantum), num_qubits, shots

---

## 🔧 Technical Implementation

### Dependencies

```bash
pip install qiskit qiskit-ibm-runtime
```

### Quantum Binding Facet

```python
# applications/noodlestudio/noodlestudio/core/quantum_binding_facet.py

"""
Quantum Binding Facet - IBM Quantum integration for feature binding.

Tests Penrose-Hameroff prediction that quantum entanglement provides
unified binding of features across phenomenal state dimensions.

Uses IBM Quantum cloud hardware (or simulator fallback).
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    QiskitRuntimeService = None
    Sampler = None


@dataclass
class QuantumBindingConfig:
    """Configuration for quantum binding operation."""
    backend_type: str = "simulator"  # "simulator" or "ibm_quantum"
    num_qubits: int = 4  # Number of qubits (max 5 for free tier)
    shots: int = 100  # Measurements per circuit
    entanglement_type: str = "full"  # "full", "linear", "circular"
    ibm_token: Optional[str] = None  # IBM Quantum API token


@dataclass
class QuantumBindingOutput:
    """Output from quantum binding operation."""
    entangled_state: List[float]  # Decoded quantum measurements
    entanglement_entropy: float  # Measure of entanglement strength
    backend_used: str  # Which backend executed (simulator/hardware)
    execution_time: float  # Seconds
    ibm_job_id: Optional[str] = None  # IBM job ID (for tracking)


class QuantumBindingFacet:
    """
    Quantum Binding Facet - Real quantum entanglement via IBM Quantum.

    Takes classical phenomenal state, encodes into quantum circuit,
    creates entanglement, measures, decodes back to classical state.

    Tests whether quantum entanglement improves feature binding
    (Penrose-Hameroff prediction).
    """

    def __init__(self, config: QuantumBindingConfig):
        """
        Initialize quantum binding facet.

        Args:
            config: Configuration (backend, qubits, shots, etc.)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install: pip install qiskit qiskit-ibm-runtime")

        self.config = config

        # Initialize IBM Quantum service
        if config.backend_type == "ibm_quantum":
            if not config.ibm_token:
                raise ValueError("IBM Quantum token required for hardware backend")

            self.service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=config.ibm_token
            )

            # Get least busy backend
            self.backend = self.service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=config.num_qubits
            )
            print(f"[QuantumBinding] Using IBM backend: {self.backend.name}")

        else:
            # Use local simulator
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
            print("[QuantumBinding] Using Aer simulator")

        # Execution stats
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.total_quantum_time = 0.0  # Time spent on quantum hardware

    async def process(
        self,
        phenomenal_state: List[float],
        incoming_data: Optional[Dict[str, Any]] = None
    ) -> QuantumBindingOutput:
        """
        Process phenomenal state through quantum binding.

        Args:
            phenomenal_state: 40-D classical state from CharmNetwork
            incoming_data: Unused (for facet API compatibility)

        Returns:
            QuantumBindingOutput with entangled state
        """
        import time
        import numpy as np

        start_time = time.time()

        # 1. Encode classical state into quantum circuit
        circuit = self._encode_state_to_circuit(phenomenal_state)

        # 2. Create entanglement
        self._add_entanglement(circuit)

        # 3. Measure
        circuit.measure_all()

        # 4. Execute on backend
        if self.config.backend_type == "ibm_quantum":
            # Real quantum hardware
            transpiled = transpile(circuit, self.backend)
            sampler = Sampler(self.backend)
            job = sampler.run(transpiled, shots=self.config.shots)

            # Await result
            result = await asyncio.to_thread(job.result)
            job_id = job.job_id()

        else:
            # Simulator
            transpiled = transpile(circuit, self.backend)
            job = self.backend.run(transpiled, shots=self.config.shots)
            result = await asyncio.to_thread(job.result)
            job_id = None

        # 5. Decode measurements back to classical state
        counts = result.get_counts()
        entangled_state = self._decode_measurements(counts, len(phenomenal_state))

        # 6. Compute entanglement entropy (measure of entanglement strength)
        entanglement_entropy = self._compute_entanglement_entropy(counts)

        # Stats
        elapsed = time.time() - start_time
        self.execution_count += 1
        self.total_execution_time += elapsed

        return QuantumBindingOutput(
            entangled_state=entangled_state,
            entanglement_entropy=entanglement_entropy,
            backend_used=self.backend.name if hasattr(self.backend, 'name') else "simulator",
            execution_time=elapsed,
            ibm_job_id=job_id
        )

    def _encode_state_to_circuit(self, state: List[float]) -> QuantumCircuit:
        """
        Encode classical state into quantum circuit via amplitude encoding.

        Maps state dimensions to rotation angles on qubits.

        Args:
            state: Classical state vector (40-D)

        Returns:
            QuantumCircuit with state encoded
        """
        import numpy as np

        num_qubits = self.config.num_qubits
        circuit = QuantumCircuit(num_qubits)

        # Sample dimensions (can't encode all 40 in 4-5 qubits)
        # Use first num_qubits dimensions
        state_sample = state[:num_qubits]

        # Normalize to [-π, π] for rotation angles
        state_normalized = np.array(state_sample) * np.pi

        # Apply RY rotations (encodes amplitude information)
        for i, angle in enumerate(state_normalized):
            circuit.ry(angle, i)

        return circuit

    def _add_entanglement(self, circuit: QuantumCircuit):
        """
        Add entanglement gates to circuit.

        Args:
            circuit: QuantumCircuit to modify in-place
        """
        num_qubits = circuit.num_qubits

        if self.config.entanglement_type == "full":
            # Full entanglement: all qubits entangled with all others
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cx(i, j)  # CNOT gate creates entanglement

        elif self.config.entanglement_type == "linear":
            # Linear chain: qubit 0 → 1 → 2 → 3
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)

        elif self.config.entanglement_type == "circular":
            # Circular: 0 → 1 → 2 → 3 → 0
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(num_qubits - 1, 0)  # Close the loop

    def _decode_measurements(self, counts: Dict[str, int], output_dim: int) -> List[float]:
        """
        Decode quantum measurement counts back to classical state.

        Uses probability distribution over bitstrings as encoded state.

        Args:
            counts: Measurement counts (e.g., {'0011': 23, '1101': 18, ...})
            output_dim: Desired output dimension (40)

        Returns:
            Decoded classical state vector
        """
        import numpy as np

        # Total shots
        total_shots = sum(counts.values())

        # Convert counts to probability distribution
        bitstring_probs = {bs: count / total_shots for bs, count in counts.items()}

        # Decode: use probabilities as "entangled features"
        # Map bitstring space (2^num_qubits) to output_dim
        decoded = []
        num_qubits = self.config.num_qubits

        for i in range(output_dim):
            # Hash dimension index to a bitstring
            target_bitstring = format(hash(i) % (2 ** num_qubits), f'0{num_qubits}b')

            # Use probability of nearest bitstring
            prob = bitstring_probs.get(target_bitstring, 0.0)

            # Map [0, 1] probability to [-1, 1] state value
            decoded_value = (prob - 0.5) * 2.0
            decoded.append(decoded_value)

        return decoded

    def _compute_entanglement_entropy(self, counts: Dict[str, int]) -> float:
        """
        Compute entanglement entropy from measurement distribution.

        Higher entropy = stronger entanglement.

        Args:
            counts: Measurement counts

        Returns:
            Entropy (0 = no entanglement, high = strong entanglement)
        """
        import numpy as np

        total = sum(counts.values())
        probs = [count / total for count in counts.values()]

        # Shannon entropy
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

        return entropy

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,  # No LLM tokens
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'total_quantum_time': self.total_quantum_time,
            'backend': self.config.backend_type
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (always 0 - quantum computation, not LLM)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }
```

---

## 🎨 Facets Editor Integration

### New Node Type: QUANTUM_BINDING

**Visual Appearance:**
- Icon: ⚛️ (atom symbol)
- Color: Purple glow (quantum theme)
- Shape: Hexagon (distinct from rounded rectangles)

**Inspector Properties:**
```
╔════════════════════════════════════╗
║ Quantum Binding                    ║
╟────────────────────────────────────╢
║ Backend:     [Simulator ▾]         ║
║              └─ Simulator          ║
║              └─ IBM Quantum        ║
║                                    ║
║ Num Qubits:  [4]  (1-5)            ║
║ Shots:       [100] (50-1000)       ║
║                                    ║
║ Entanglement: [Full ▾]             ║
║              └─ Full               ║
║              └─ Linear             ║
║              └─ Circular           ║
║                                    ║
║ ● IBM Token: [**************]      ║
║   (Required for IBM Quantum)       ║
║                                    ║
║ Status: ✅ Connected (ibm_brisbane)║
║ Last execution: 0.234s             ║
║ Entanglement entropy: 1.82         ║
╚════════════════════════════════════╝
```

**Drag-and-Drop:**
User drags from Node Palette (new section: "Quantum"):
- ⚛️ Quantum Binding
- 🎲 Entropy Injection (hardware RNG)
- 🧠 Quantum Microtubule (simulation)

---

## 🔬 Validation Experiments

### Experiment 1: Binding Strength Test

**Hypothesis:** Quantum entanglement improves feature binding (Penrose-Hameroff)

**Setup:**
1. Create two facet assemblies:
   - **Classical:** CharmNet → ContextIntelligence → Red's Mind
   - **Quantum:** CharmNet → QuantumBinding → ContextIntelligence → Red's Mind
2. Run same inputs through both
3. Measure: Correlation between affect dimensions, response coherence

**Expected Result:**
- Quantum version shows higher cross-dimensional correlation
- More unified responses (binding problem solved?)

---

### Experiment 2: Entanglement Entropy vs Performance

**Hypothesis:** Higher entanglement entropy correlates with better cognition

**Setup:**
1. Vary entanglement type: Full, Linear, Circular
2. Measure entanglement entropy from QuantumBinding facet
3. Measure downstream performance (e.g., Theory of Mind accuracy)

**Expected Result:**
- Full entanglement → highest entropy → best performance

---

### Experiment 3: Simulator vs Hardware

**Hypothesis:** Real quantum hardware shows effects not replicable by simulator

**Setup:**
1. Run same circuit on simulator and IBM hardware
2. Compare measurement distributions
3. Measure divergence (KL divergence of probability distributions)

**Expected Result:**
- Hardware shows noise, decoherence effects not in simulator
- Tests whether "real" quantum effects matter for consciousness

---

## 💰 Cost Management

### IBM Quantum Free Tier Limits

**Monthly Quota:**
- 10 minutes QPU time
- ~600 seconds = 600,000 milliseconds
- Each circuit: ~1-3ms execution
- **Estimate:** 200,000-600,000 circuits/month

**Per Agent Step (with QuantumBinding facet):**
- 1 circuit × 100 shots = ~0.0003 seconds
- 10 minutes ÷ 0.0003s = **33,333 agent steps/month**

**Recommended:**
- Use simulator for development/testing
- Switch to hardware for validation experiments
- Monitor usage via IBM Quantum dashboard

### Paid Tier Costs

If free tier exceeded:
- **Pay-as-you-go:** ~$1.60/second QPU time
- Example: 1000 agent steps = ~0.3s = **$0.48**

**Cost Optimization:**
1. Batch requests (submit multiple circuits together)
2. Reduce shots (100 → 50 for testing)
3. Use simulator except for validation runs
4. Cache results (same input → reuse measurement)

---

## 🚀 Implementation Phases

### Phase 1: Quantum Binding Facet (2 weeks)
- [ ] Install Qiskit dependencies
- [ ] Create `QuantumBindingFacet` class
- [ ] Add simulator backend support
- [ ] Add IBM Quantum backend support
- [ ] Implement state encoding/decoding
- [ ] Add entanglement circuit generation
- [ ] Test with dummy phenomenal state

### Phase 2: Facets Editor Integration (1 week)
- [ ] Add QUANTUM_BINDING to facet registry
- [ ] Create visual node type (purple hexagon)
- [ ] Add Inspector UI (backend dropdown, token field)
- [ ] Add to Node Palette ("Quantum" section)
- [ ] Test drag-and-drop functionality
- [ ] Add connection validation (40-D input/output)

### Phase 3: Testing & Validation (1 week)
- [ ] Create test facet assembly with QuantumBinding
- [ ] Run Experiment 1 (binding strength)
- [ ] Run Experiment 2 (entanglement entropy)
- [ ] Run Experiment 3 (simulator vs hardware)
- [ ] Document results in research notes
- [ ] Publish findings (arXiv paper?)

### Phase 4: Neural Canvas Integration (3 days)
- [ ] Add QuantumEntanglement layer type to Neural Canvas
- [ ] Allow placement inside CharmNetwork topology
- [ ] Generate code for quantum layers in MLX export
- [ ] Test hybrid quantum-classical training

### Phase 5: Quantum Service (Optional, 2 weeks)
- [ ] Create standalone quantum service process
- [ ] Implement circuit queue
- [ ] Add result caching
- [ ] Background thread for async execution
- [ ] Fallback to simulator when offline

---

## 🔮 Future Enhancements

### v1.0 (MVP)
- [x] Quantum Binding Facet
- [x] IBM Quantum integration
- [x] Simulator fallback
- [x] Basic experiments

### v1.5
- [ ] Quantum Service (non-blocking)
- [ ] Result caching
- [ ] Batch circuit submission
- [ ] Live monitoring dashboard (show QPU usage)

### v2.0
- [ ] Quantum Microtubule layer in Neural Canvas
- [ ] Integration with QuantumMicrotubuleLayer (merge simulation + hardware)
- [ ] Variational Quantum Eigensolver (VQE) for training
- [ ] Quantum Neural Network (QNN) layers

### v3.0
- [ ] Multi-backend support (AWS Braket, Azure Quantum, D-Wave)
- [ ] Quantum circuit optimization (transpilation hints)
- [ ] Error mitigation strategies
- [ ] Quantum advantage benchmarks

---

## 📋 Success Criteria

IBM Quantum integration is successful when:

1. ✅ QuantumBinding facet can be dragged into Facets Editor
2. ✅ Simulator backend executes circuits without errors
3. ✅ IBM Quantum backend connects and executes on real hardware
4. ✅ Entanglement entropy computed and displayed
5. ✅ Red Fire Anklebiter runs with quantum facet (no crashes)
6. ✅ Experiment 1 shows measurable binding improvement (or disproves hypothesis!)
7. ✅ Monthly IBM Quantum budget stays under free tier
8. ✅ Complete workflow takes < 5 minutes to add quantum facet

---

## 🧠 Philosophy

Quantum integration embodies the same rigor as the rest of NoodleStudio:

- **Testable:** Every claim (Penrose-Hameroff) gets experimental validation
- **Practical:** Quantum enhancement, not quantum hype
- **Transparent:** Show when using simulator vs hardware
- **Production-grade:** Handles errors, rate limits, fallbacks gracefully
- **Scientific:** Designed to produce publishable results

This is not quantum mysticism. This is computational neuroscience meeting quantum information theory.

---

## 📚 References

### Scientific Basis
- Penrose, R., & Hameroff, S. (2014). "Consciousness in the universe: A review of the 'Orch OR' theory." *Physics of Life Reviews*.
- Hameroff, S., & Penrose, R. (2025). "Quantum coherence in microtubules: Evidence from recent neuroscience." *Neuroscience of Consciousness*.

### Quantum Computing
- IBM Quantum Documentation: https://quantum-computing.ibm.com/
- Qiskit Textbook: https://qiskit.org/textbook/
- Nielsen & Chuang (2010). *Quantum Computation and Quantum Information*.

### Consciousness Theory
- Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*.
- Chalmers, D. (1995). "Facing up to the problem of consciousness." *Journal of Consciousness Studies*.

---

**End of Strategy Document**

**Next Steps:**
1. Review with Caity
2. Obtain IBM Quantum API token (https://quantum-computing.ibm.com/)
3. Begin Phase 1 implementation (QuantumBindingFacet)
4. Coordinate with Neural Canvas development (parallel tracks)

*Fascinating.*
