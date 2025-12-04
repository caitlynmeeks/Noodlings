"""
Quantum Charm Network - Consciousness with Microtubule Dynamics

Hybrid retrofit of trained Charm Network (Phase 4) with quantum microtubule layers.

Architecture:
    Affect Input (5-D)
        ↓
    Fast LSTM (16-D) ← classical membrane dynamics, seconds
        ↓
    🔬 MT Layer 1 ← quantum modulation
        ↓
    Medium LSTM (16-D) ← classical membrane dynamics, minutes
        ↓
    🔬 MT Layer 2 ← quantum modulation
        ↓
    Slow GRU (8-D) ← classical membrane dynamics, hours/days
        ↓
    🔬 MT Layer 3 ← quantum modulation
        ↓
    Phenomenal State (40-D) → Affect Head (5-D)

Key features:
- Preserves trained LSTM/GRU weights
- Adds quantum effects at temporal boundaries
- Tunable quantum contribution (0.0 = classical, 1.0 = full quantum)
- Tracks collapse events across all MT layers
- Compatible with existing CharmNetworkFacet interface

Author: Commander Spock + Cadet Caity
Date: December 1, 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
import time

# Handle imports for both module and standalone usage
try:
    from .noodling_phase4 import NoodlingModelPhase4
    from .quantum_microtubule import QuantumMicrotubuleLayer
except ImportError:
    from noodling_phase4 import NoodlingModelPhase4
    from quantum_microtubule import QuantumMicrotubuleLayer


class QuantumCharmNetwork(nn.Module):
    """
    Quantum-augmented hierarchical affective consciousness.

    Wraps existing trained Phase 4 model with quantum microtubule layers.
    """

    def __init__(
        self,
        base_model: NoodlingModelPhase4,
        # Quantum parameters
        quantum_contribution: float = 0.3,
        collapse_threshold: float = 1.2,
        coherence_time: int = 10,
        entanglement_range: int = 3,
        noise_scale: float = 0.15,
        enable_quantum: bool = True
    ):
        """
        Initialize quantum charm network.

        Args:
            base_model: Trained Phase 4 model (preserved weights)
            quantum_contribution: Weight of MT output (0.0-1.0)
            collapse_threshold: Magnitude threshold for collapse
            coherence_time: Steps before forced decoherence
            entanglement_range: Spatial correlation range
            noise_scale: Quantum noise amplitude
            enable_quantum: Toggle quantum effects on/off
        """
        super().__init__()

        self.base_model = base_model
        self.quantum_contribution = quantum_contribution
        self.enable_quantum = enable_quantum

        # Extract dimensions from base model
        fast_hidden = base_model.fast_hidden
        medium_hidden = base_model.medium_hidden
        slow_hidden = base_model.slow_hidden

        # Initialize 3 MT layers (one after each temporal stage)
        if enable_quantum:
            self.mt_layer_fast = QuantumMicrotubuleLayer(
                input_dim=fast_hidden,
                hidden_dim=fast_hidden,
                collapse_threshold=collapse_threshold,
                coherence_time=coherence_time,
                entanglement_range=entanglement_range,
                noise_scale=noise_scale
            )

            self.mt_layer_medium = QuantumMicrotubuleLayer(
                input_dim=medium_hidden,
                hidden_dim=medium_hidden,
                collapse_threshold=collapse_threshold,
                coherence_time=coherence_time,
                entanglement_range=entanglement_range,
                noise_scale=noise_scale
            )

            self.mt_layer_slow = QuantumMicrotubuleLayer(
                input_dim=slow_hidden,
                hidden_dim=slow_hidden,
                collapse_threshold=collapse_threshold,
                coherence_time=coherence_time,
                entanglement_range=entanglement_range,
                noise_scale=noise_scale
            )
        else:
            self.mt_layer_fast = None
            self.mt_layer_medium = None
            self.mt_layer_slow = None

        # MT states (persistent across calls)
        self.mt_state_fast = None
        self.mt_state_medium = None
        self.mt_state_slow = None

        # Step counter for coherence tracking
        self.step = 0

    def __call__(
        self,
        affect_sequence: mx.array,
        h_fast: mx.array,
        c_fast: mx.array,
        h_medium: mx.array,
        c_medium: mx.array,
        h_slow: mx.array,
        # Social cognition params (pass through to base model)
        linguistic_features: Optional[mx.array] = None,
        context_features: Optional[mx.array] = None,
        other_agent_ids: Optional[List[str]] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass with quantum microtubule modulation.

        Args:
            affect_sequence: [batch, seq_len, 5] affect inputs
            h_fast, c_fast: Fast LSTM state
            h_medium, c_medium: Medium LSTM state
            h_slow: Slow GRU state
            linguistic_features: Social cognition features
            context_features: Social context
            other_agent_ids: Other agents in interaction

        Returns:
            Dictionary containing:
                - All outputs from base model
                - quantum_collapses: [fast, medium, slow] collapse flags
                - quantum_contribution: Current quantum weighting
                - timing_ms: Dict with timing breakdown
        """
        # ⏱️ Start timing
        t_start = time.perf_counter()
        timing = {}

        batch_size = affect_sequence.shape[0]

        # Initialize MT states if needed
        if self.enable_quantum:
            if self.mt_state_fast is None:
                self.mt_state_fast = mx.zeros((batch_size, self.base_model.fast_hidden))
            if self.mt_state_medium is None:
                self.mt_state_medium = mx.zeros((batch_size, self.base_model.medium_hidden))
            if self.mt_state_slow is None:
                self.mt_state_slow = mx.zeros((batch_size, self.base_model.slow_hidden))

        # Run base model first (classical dynamics)
        t_base_start = time.perf_counter()
        base_output = self.base_model(
            affect_sequence,
            h_fast, c_fast,
            h_medium, c_medium,
            h_slow,
            linguistic_features=linguistic_features,
            context_features=context_features,
            other_agent_ids=other_agent_ids
        )
        mx.eval(base_output['actual_state'])  # Force evaluation for accurate timing
        timing['base_model_ms'] = (time.perf_counter() - t_base_start) * 1000

        # Extract phenomenal state from base model
        phenomenal_state = base_output['actual_state']  # [batch, 40]

        # If quantum enabled, add MT modulation to each temporal component
        if self.enable_quantum:
            t_quantum_start = time.perf_counter()

            # Split phenomenal state into temporal components
            fast_state = phenomenal_state[:, :16]      # [batch, 16]
            medium_state = phenomenal_state[:, 16:32]  # [batch, 16]
            slow_state = phenomenal_state[:, 32:40]    # [batch, 8]

            # 🔬 MT Layer 1: Quantum modulation of fast component
            t_mt1_start = time.perf_counter()
            mt_output_fast, self.mt_state_fast, collapse_fast = self.mt_layer_fast(
                fast_state, self.mt_state_fast, self.step
            )
            fast_modulated = fast_state + self.quantum_contribution * mt_output_fast
            mx.eval(fast_modulated)
            timing['mt_fast_ms'] = (time.perf_counter() - t_mt1_start) * 1000

            # 🔬 MT Layer 2: Quantum modulation of medium component
            t_mt2_start = time.perf_counter()
            mt_output_medium, self.mt_state_medium, collapse_medium = self.mt_layer_medium(
                medium_state, self.mt_state_medium, self.step
            )
            medium_modulated = medium_state + self.quantum_contribution * mt_output_medium
            mx.eval(medium_modulated)
            timing['mt_medium_ms'] = (time.perf_counter() - t_mt2_start) * 1000

            # 🔬 MT Layer 3: Quantum modulation of slow component
            t_mt3_start = time.perf_counter()
            mt_output_slow, self.mt_state_slow, collapse_slow = self.mt_layer_slow(
                slow_state, self.mt_state_slow, self.step
            )
            slow_modulated = slow_state + self.quantum_contribution * mt_output_slow
            mx.eval(slow_modulated)
            timing['mt_slow_ms'] = (time.perf_counter() - t_mt3_start) * 1000

            # Recombine into quantum-modulated phenomenal state
            phenomenal_state_quantum = mx.concatenate([fast_modulated, medium_modulated, slow_modulated], axis=-1)
            mx.eval(phenomenal_state_quantum)
            timing['quantum_total_ms'] = (time.perf_counter() - t_quantum_start) * 1000
        else:
            phenomenal_state_quantum = phenomenal_state
            collapse_fast = collapse_medium = collapse_slow = False
            timing['quantum_total_ms'] = 0.0

        # Increment step counter
        self.step += 1

        # ⏱️ Total timing
        timing['total_ms'] = (time.perf_counter() - t_start) * 1000

        # 📊 Compute metrics (approximate FLOPs)
        compute_metrics = self._estimate_compute(batch_size, affect_sequence.shape[1])

        # Update base output with quantum-modulated state
        base_output['actual_state'] = phenomenal_state_quantum
        base_output['quantum_collapses'] = [collapse_fast, collapse_medium, collapse_slow]
        base_output['quantum_contribution'] = self.quantum_contribution
        base_output['quantum_enabled'] = self.enable_quantum
        base_output['timing_ms'] = timing  # Add timing metrics
        base_output['compute_metrics'] = compute_metrics  # Add compute metrics

        return base_output

    def _estimate_compute(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        Estimate computational cost in FLOPs and equivalent tokens.

        For LLM comparison: ~6 FLOPs per parameter per token for inference.
        GPT-3.5 (175B params) @ 1 token ≈ 1 TFLOPs

        Returns:
            Dict with:
                - total_flops: Total floating point operations
                - mflops: MegaFLOPs (millions)
                - gflops: GigaFLOPs (billions)
                - token_equivalent: Equivalent GPT-3.5 tokens (rough comparison)
                - params_count: Total parameter count
        """
        # Base model parameters
        fast_params = self.base_model.fast_hidden * (5 + self.base_model.fast_hidden) * 4  # LSTM gates
        medium_params = self.base_model.medium_hidden * (self.base_model.fast_hidden + self.base_model.medium_hidden) * 4
        slow_params = self.base_model.slow_hidden * (self.base_model.medium_hidden + self.base_model.slow_hidden) * 3  # GRU
        affect_head_params = 40 * 5  # Phenomenal state to affect

        total_params = fast_params + medium_params + slow_params + affect_head_params  # ~54K params

        # FLOPs per forward pass (approximate)
        # LSTM: 4 * (input_size + hidden_size) * hidden_size * 2 (matmul + activation)
        fast_flops = 4 * (5 + self.base_model.fast_hidden) * self.base_model.fast_hidden * 2 * seq_len
        medium_flops = 4 * (self.base_model.fast_hidden + self.base_model.medium_hidden) * self.base_model.medium_hidden * 2 * seq_len
        slow_flops = 3 * (self.base_model.medium_hidden + self.base_model.slow_hidden) * self.base_model.slow_hidden * 2 * seq_len
        affect_flops = 40 * 5 * 2  # Affect head linear

        base_flops = (fast_flops + medium_flops + slow_flops + affect_flops) * batch_size

        # Quantum MT layer overhead (if enabled)
        if self.enable_quantum:
            # Each MT layer: attention + FFN + modulation
            mt_flops_per_layer = self.base_model.fast_hidden * self.base_model.fast_hidden * 4 * batch_size
            quantum_flops = mt_flops_per_layer * 3  # 3 MT layers
        else:
            quantum_flops = 0

        total_flops = base_flops + quantum_flops

        # Token equivalent (very rough comparison)
        # GPT-3.5: ~175B params, ~6 FLOPs/param/token = 1050 GFLOPs/token
        gpt35_flops_per_token = 1050e9  # 1.05 TFLOPs
        token_equivalent = total_flops / gpt35_flops_per_token

        return {
            'total_flops': total_flops,
            'mflops': total_flops / 1e6,
            'gflops': total_flops / 1e9,
            'token_equivalent': token_equivalent,
            'params_count': int(total_params),
            'vs_gpt35_tokens': f"{token_equivalent:.6f} tokens",
            'description': f"~{total_params/1000:.1f}K params, {total_flops/1e6:.2f} MFLOPs"
        }

    def reset_quantum_state(self):
        """Reset MT states and step counter (e.g., between conversations)."""
        self.mt_state_fast = None
        self.mt_state_medium = None
        self.mt_state_slow = None
        self.step = 0

        if self.enable_quantum:
            self.mt_layer_fast.reset_coherence()
            self.mt_layer_medium.reset_coherence()
            self.mt_layer_slow.reset_coherence()

    def get_quantum_stats(self) -> Dict:
        """
        Get quantum collapse statistics across all MT layers.

        Returns:
            Dictionary with per-layer collapse stats
        """
        if not self.enable_quantum:
            return {'quantum_enabled': False}

        return {
            'quantum_enabled': True,
            'quantum_contribution': self.quantum_contribution,
            'step': self.step,
            'fast_layer': self.mt_layer_fast.get_collapse_stats(),
            'medium_layer': self.mt_layer_medium.get_collapse_stats(),
            'slow_layer': self.mt_layer_slow.get_collapse_stats()
        }

    def set_quantum_contribution(self, contribution: float):
        """
        Dynamically adjust quantum contribution.

        Args:
            contribution: 0.0 (pure classical) to 1.0 (full quantum)
        """
        self.quantum_contribution = max(0.0, min(1.0, contribution))


if __name__ == "__main__":
    """Test quantum charm network."""

    print("Testing QuantumCharmNetwork...")

    # Initialize base model (Phase 4)
    base_model = NoodlingModelPhase4(
        affect_dim=5,
        fast_hidden=16,
        medium_hidden=16,
        slow_hidden=8,
        predictor_hidden=64,
        use_vae=False,
        memory_capacity=100,
        use_theory_of_mind=True,
        use_relationship_model=True
    )

    # Wrap with quantum layers
    quantum_model = QuantumCharmNetwork(
        base_model=base_model,
        quantum_contribution=0.3,
        collapse_threshold=1.2,
        enable_quantum=True
    )

    # Test input
    affect_seq = mx.random.normal((1, 1, 5)) * 0.5
    h_fast = mx.zeros((1, 16))
    c_fast = mx.zeros((1, 16))
    h_medium = mx.zeros((1, 16))
    c_medium = mx.zeros((1, 16))
    h_slow = mx.zeros((1, 8))

    # Run for 20 steps
    collapse_counts = {'fast': 0, 'medium': 0, 'slow': 0}

    for step in range(20):
        output = quantum_model(
            affect_seq, h_fast, c_fast, h_medium, c_medium, h_slow
        )

        # Update hidden states
        h_fast = output['h_fast']
        c_fast = output['c_fast']
        h_medium = output['h_medium']
        c_medium = output['c_medium']
        h_slow = output['h_slow']

        # Track collapses
        collapses = output['quantum_collapses']
        if collapses[0]:
            collapse_counts['fast'] += 1
        if collapses[1]:
            collapse_counts['medium'] += 1
        if collapses[2]:
            collapse_counts['slow'] += 1

    # Report
    stats = quantum_model.get_quantum_stats()
    print(f"\nQuantum Charm Network Test Results:")
    print(f"  Steps: 20")
    print(f"  Quantum contribution: {quantum_model.quantum_contribution}")
    print(f"\nCollapse events:")
    print(f"  Fast layer: {collapse_counts['fast']} ({collapse_counts['fast']/20*100:.0f}%)")
    print(f"  Medium layer: {collapse_counts['medium']} ({collapse_counts['medium']/20*100:.0f}%)")
    print(f"  Slow layer: {collapse_counts['slow']} ({collapse_counts['slow']/20*100:.0f}%)")
    print(f"\nPhenomenal state shape: {output['actual_state'].shape}")
    print(f"Surprise metric: {float(output['surprise'][0]):.4f}")
    print(f"\nTest complete! Quantum consciousness operational.")
