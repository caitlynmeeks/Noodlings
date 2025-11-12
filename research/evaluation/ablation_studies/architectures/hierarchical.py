"""
Hierarchical Architecture - Multi-timescale without Observers

Fast + Medium + Slow layers with different learning rates, but no observer loops.
This is the core Noodlings architecture without integrated information.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple
from .base import AblationArchitecture


class HierarchicalArchitecture(AblationArchitecture):
    """
    Hierarchical: Fast + Medium + Slow (no observers).

    Three-level hierarchy with timescale separation via learning rates:
    - Fast (16-D): immediate affective reactions
    - Medium (16-D): conversational dynamics
    - Slow (8-D): personality/disposition

    Parameters: ~4.5K
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__(affect_dim)

        # Fast layer: responds to immediate affect
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=16)
        self.c_fast = mx.zeros((1, 16))

        # Medium layer: integrates fast layer over time
        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)
        self.c_medium = mx.zeros((1, 16))

        # Slow layer: models long-term personality
        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Predictor network
        joint_dim = 16 + 16 + 8  # 40
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim)
        )

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Process through hierarchical layers."""
        # Ensure correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]  # (1, 1, 5)
        elif affect.ndim == 2:
            affect = affect[:, None, :]  # (batch, 1, 5)

        # Fast layer: immediate response to affect
        h_fast_seq, c_fast_seq = self.fast_lstm(
            affect, hidden=self.h_fast, cell=self.c_fast
        )
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium layer: integrates fast layer
        h_med_seq, c_med_seq = self.medium_lstm(
            self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium
        )
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow layer: long-term personality model
        h_slow_seq = self.slow_gru(
            self.h_medium[:, None, :], hidden=self.h_slow
        )
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Calculate surprise
        current_state = self.get_phenomenal_state()
        predicted_state = self.predictor(current_state)
        surprise = float(mx.mean((predicted_state - current_state) ** 2))

        return current_state, surprise

    def reset_states(self):
        """Reset all layer states."""
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def architecture_description(self) -> str:
        return (
            "Hierarchical: Fast (16) + Medium (16) + Slow (8) layers.\n"
            f"Parameters: ~{self.parameter_count():,}\n"
            "Purpose: Core architecture showing timescale separation."
        )
