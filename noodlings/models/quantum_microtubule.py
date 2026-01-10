# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Quantum Microtubule Layer
#
#   This is the building block for quantum-inspired neural layers.
#   Microtubules are tiny protein tubes inside every cell, including
#   neurons. Penrose and Hameroff proposed they might be where
#   quantum effects influence brain activity.
#
#   This layer adds four quantum-like effects to neural processing:
#
#   1. QUANTUM NOISE: Real randomness (from avalanche RNG hardware)
#      injected like "quantum fluctuations." Unlike pseudorandom
#      numbers, this is genuinely unpredictable.
#
#   2. ENTANGLEMENT: Nearby dimensions become correlated. When one
#      changes, its neighbors tend to change too - like quantum
#      entanglement creating "spooky action at a distance."
#
#   3. COLLAPSE: When the state gets too extreme, it suddenly
#      "collapses" to a definite value - like quantum wavefunction
#      collapse when measured.
#
#   4. DECOHERENCE: Quantum effects fade over time due to
#      environmental interaction. The system becomes more
#      "classical" unless something refreshes the coherence.
#
#   These effects can be tuned or disabled to study their impact.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   noodlings.models.quantum_microtubule
# PURPOSE:  Quantum-inspired neural layer with collapse dynamics
# LAYER:    Core / Models
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   QuantumMicrotubuleLayer  Adds quantum effects to neural state
#
# DEPENDENCIES:
#   mlx.core                 Apple Silicon tensor operations
#   mlx.nn                   Neural network layers
#   entropy_service          Avalanche RNG for true randomness
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Quantum Microtubule Layer for Consciousness Models

Implements biologically-inspired quantum dynamics based on:
- Orchestrated Objective Reduction (Penrose-Hameroff)
- Microtubule quantum coherence (Neuroscience of Consciousness, 2025)
- Avalanche effect randomness (quantum breakdown statistics)

Adds to standard LSTM/GRU:
1. Quantum noise injection (avalanche-distributed)
2. Nonlocal entanglement (spatial correlations via convolution)
3. Orchestrated collapse (sudden state reduction when threshold exceeded)
4. Coherence decay (decoherence over time)

Author: Commander Spock + Cadet Caity
Date: December 1, 2025
"""

import sys
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple

# Add cmush path for entropy service
sys.path.append(os.path.join(os.path.dirname(__file__), '../../applications/cmush'))
from entropy_service import get_entropy_service


class QuantumMicrotubuleLayer(nn.Module):
    """
    Quantum microtubule dynamics layer.

    Augments classical neural activations with:
    - Avalanche RNG noise (quantum fluctuations)
    - Nonlocal entanglement (spatial correlations)
    - Orchestrated reduction (collapse events)
    - Coherence tracking (decay over time)

    Architecture:
        Classical LSTM/GRU output → MT integration → Quantum evolution →
        Entanglement → Orchestrated collapse → Modulated output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        collapse_threshold: float = 0.5,
        coherence_time: int = 10,
        entanglement_range: int = 3,
        noise_scale: float = 0.1,
        use_collapse: bool = True,
        use_entanglement: bool = True
    ):
        """
        Initialize quantum microtubule layer.

        Args:
            input_dim: Input dimension (from LSTM/GRU output)
            hidden_dim: Hidden dimension (MT state size)
            collapse_threshold: Magnitude threshold for collapse events
            coherence_time: Steps before forced decoherence
            entanglement_range: Spatial correlation range (for multi-dimensional states)
            noise_scale: Amplitude of quantum noise
            use_collapse: Enable orchestrated reduction
            use_entanglement: Enable nonlocal correlations
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.collapse_threshold = collapse_threshold
        self.coherence_time = coherence_time
        self.entanglement_range = entanglement_range
        self.noise_scale = noise_scale
        self.use_collapse = use_collapse
        self.use_entanglement = use_entanglement

        # Integration weights (receive classical neural signals)
        self.W_integrate = nn.Linear(input_dim, hidden_dim)

        # Collapse projection weights (map to output)
        self.W_collapse = nn.Linear(hidden_dim, hidden_dim)

        # Entropy service for quantum randomness
        entropy_service = get_entropy_service()
        self.avalanche_rng = entropy_service.create_avalanche_rng(beta=2.0)

        # Entanglement kernel (Gaussian spatial correlation)
        if use_entanglement:
            self.entanglement_kernel = self._create_entanglement_kernel()
        else:
            self.entanglement_kernel = None

        # Coherence tracking
        self.coherence_counter = 0
        self.last_collapse_step = 0

        # Statistics
        self.total_collapses = 0
        self.collapse_history = []

    def _create_entanglement_kernel(self) -> mx.array:
        """
        Create Gaussian entanglement kernel for spatial correlations.

        Returns:
            1D Gaussian kernel for convolution
        """
        kernel_size = 2 * self.entanglement_range + 1
        x = np.arange(kernel_size) - self.entanglement_range
        gaussian = np.exp(-x**2 / (2 * self.entanglement_range**2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        return mx.array(gaussian, dtype=mx.float32)

    def _apply_entanglement(self, quantum_state: mx.array) -> mx.array:
        """
        Apply nonlocal entanglement via convolution.

        Creates spatial correlations across state dimensions.

        Args:
            quantum_state: [batch, hidden_dim] or [batch, seq, hidden_dim] state vector

        Returns:
            Entangled state with spatial correlations
        """
        if not self.use_entanglement or self.entanglement_kernel is None:
            return quantum_state

        # Handle both 2D and 3D inputs (squeeze if needed)
        original_shape = quantum_state.shape
        if len(original_shape) == 3:
            batch_size, seq_len, dim = original_shape
            quantum_state = quantum_state.reshape(batch_size * seq_len, dim)
        elif len(original_shape) == 2:
            batch_size, dim = original_shape
        else:
            return quantum_state  # Unsupported shape

        # For 1D state vector, treat dimensions as spatial locations
        # Convolve each batch item separately
        effective_batch_size = quantum_state.shape[0]
        entangled_states = []
        for i in range(effective_batch_size):
            state_1d = quantum_state[i:i+1, :]  # [1, dim]

            # Pad for convolution (reflect boundary conditions)
            padded = mx.pad(state_1d, ((0, 0), (self.entanglement_range, self.entanglement_range)), mode='edge')

            # Manual convolution (MLX doesn't have conv1d yet)
            # Slide kernel across padded state
            kernel_size = len(self.entanglement_kernel)
            entangled = mx.zeros((1, dim))

            for j in range(dim):
                window = padded[0, j:j+kernel_size]
                entangled[0, j] = mx.sum(window * self.entanglement_kernel)

            entangled_states.append(entangled)

        result = mx.concatenate(entangled_states, axis=0)

        # Restore original shape if 3D
        if len(original_shape) == 3:
            result = result.reshape(original_shape)

        return result

    def _check_collapse(self, quantum_state: mx.array, step: int) -> Tuple[mx.array, bool]:
        """
        Check for orchestrated reduction (collapse event).

        Collapse occurs when:
        1. State magnitude exceeds threshold
        2. Avalanche RNG triggers (probability based on magnitude)

        Args:
            quantum_state: [batch, hidden_dim] quantum state
            step: Current timestep

        Returns:
            (collapsed_state, did_collapse)
        """
        if not self.use_collapse:
            return quantum_state, False

        # Compute state magnitude (L2 norm)
        state_magnitude = mx.sqrt(mx.sum(quantum_state**2, axis=-1, keepdims=True))  # [batch, 1]

        # Collapse probability (sigmoid of magnitude - threshold)
        collapse_prob = 1.0 / (1.0 + mx.exp(-10.0 * (state_magnitude - self.collapse_threshold)))

        # Avalanche RNG trigger (per-batch)
        batch_size = quantum_state.shape[0]
        random_triggers = self.avalanche_rng.generate_positive(shape=(batch_size, 1))
        random_triggers_mx = mx.array(random_triggers, dtype=mx.float32)

        # Collapse if probability > random threshold
        should_collapse = collapse_prob > random_triggers_mx  # [batch, 1]

        # Collapse: project to eigenvector (here, normalize to unit sphere)
        normalized_state = quantum_state / (state_magnitude + 1e-8)
        collapsed_state = mx.where(
            should_collapse,
            normalized_state,  # Collapsed: definite state
            quantum_state      # Not collapsed: superposition continues
        )

        # Track collapses
        did_collapse = bool(mx.any(should_collapse).item())
        if did_collapse:
            self.total_collapses += 1
            self.last_collapse_step = step
            self.collapse_history.append(step)

        return collapsed_state, did_collapse

    def _apply_decoherence(self, quantum_state: mx.array) -> mx.array:
        """
        Apply environmental decoherence.

        Quantum coherence decays over time, reducing quantum effects.

        Args:
            quantum_state: Current quantum state

        Returns:
            Decohered state
        """
        # Coherence decay factor (exponential)
        decay = np.exp(-self.coherence_counter / self.coherence_time)

        # Mix quantum state with zero (loss of coherence)
        decohered = quantum_state * decay

        return decohered

    def __call__(
        self,
        classical_input: mx.array,
        mt_state: Optional[mx.array] = None,
        step: int = 0
    ) -> Tuple[mx.array, mx.array, bool]:
        """
        Forward pass through quantum microtubule layer.

        Args:
            classical_input: [batch, input_dim] from LSTM/GRU
            mt_state: [batch, hidden_dim] previous MT state (or None)
            step: Current timestep (for coherence tracking)

        Returns:
            (mt_output, new_mt_state, did_collapse)
        """
        batch_size = classical_input.shape[0]

        # Initialize MT state if needed
        if mt_state is None:
            mt_state = mx.zeros((batch_size, self.hidden_dim))

        # 1. Integration: Receive classical neural signal (like calcium influx)
        calcium_signal = self.W_integrate(classical_input)

        # 2. Quantum noise injection (avalanche-distributed fluctuations)
        quantum_noise_np = self.avalanche_rng.generate(shape=(batch_size, self.hidden_dim))
        quantum_noise = mx.array(quantum_noise_np, dtype=mx.float32) * self.noise_scale

        # 3. Coherent evolution: superposition of previous state + signal + noise
        coherent_state = mt_state + calcium_signal + quantum_noise

        # 4. Nonlocal entanglement: spatial correlations
        entangled_state = self._apply_entanglement(coherent_state)

        # 5. Orchestrated reduction: collapse events
        collapsed_state, did_collapse = self._check_collapse(entangled_state, step)

        # 6. Decoherence: quantum effects decay over time
        self.coherence_counter += 1
        if did_collapse or self.coherence_counter >= self.coherence_time:
            self.coherence_counter = 0  # Reset after collapse or timeout

        decohered_state = self._apply_decoherence(collapsed_state)

        # 7. Output modulation: affect classical neural transmission
        mt_output = mx.tanh(self.W_collapse(decohered_state))

        return mt_output, decohered_state, did_collapse

    def reset_coherence(self):
        """Reset coherence tracking (e.g., between conversations)."""
        self.coherence_counter = 0
        self.last_collapse_step = 0

    def get_collapse_stats(self) -> dict:
        """
        Get collapse event statistics.

        Returns:
            Dictionary with total collapses, rate, recent history
        """
        return {
            'total_collapses': self.total_collapses,
            'last_collapse_step': self.last_collapse_step,
            'recent_collapses': self.collapse_history[-10:] if self.collapse_history else []
        }


if __name__ == "__main__":
    """Test quantum microtubule layer."""

    print("Testing QuantumMicrotubuleLayer...")

    # Initialize layer
    mt_layer = QuantumMicrotubuleLayer(
        input_dim=16,
        hidden_dim=16,
        collapse_threshold=1.2,  # Tuned for ~20-30% collapse rate
        coherence_time=10,
        entanglement_range=3,
        noise_scale=0.15
    )

    # Test input (simulating LSTM output)
    classical_input = mx.random.normal((1, 16)) * 0.5  # Scale to reasonable range

    # Run for 50 steps to see pattern
    mt_state = None
    collapses = []

    for step in range(50):
        mt_output, mt_state, did_collapse = mt_layer(classical_input, mt_state, step)
        if did_collapse:
            collapses.append(step)

    collapse_rate = (len(collapses) / 50) * 100

    print(f"\nTotal collapses: {mt_layer.total_collapses} out of 50 steps ({collapse_rate:.1f}%)")
    print(f"Collapse steps: {collapses}")
    print(f"MT output shape: {mt_output.shape}")
    print(f"\nQuantum dynamics:")
    print(f"  - State evolves in superposition between collapses")
    print(f"  - Collapse rate ~{collapse_rate:.0f}% (optimal: 20-40%)")
    print(f"  - Entanglement correlates nearby dimensions")
    print(f"  - Coherence decays every {mt_layer.coherence_time} steps")
    print(f"\nTest complete!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
