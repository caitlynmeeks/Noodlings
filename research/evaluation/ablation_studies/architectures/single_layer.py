"""
Single Layer Architecture - One LSTM

A single LSTM layer to show the benefit of hierarchy.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple
from .base import AblationArchitecture


class SingleLayerArchitecture(AblationArchitecture):
    """
    Single Layer: One LSTM (no hierarchy).

    Uses a single 40-D LSTM to match the total dimensionality of
    the hierarchical model, but without timescale separation.

    Parameters: ~6.7K
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__(affect_dim)

        # Single LSTM with 40-D hidden state (matching total hierarchy size)
        self.lstm = nn.LSTM(input_size=affect_dim, hidden_size=40)

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 40)
        )

        # Hidden state (we'll split it for compatibility with metrics)
        self.h_all = mx.zeros((1, 40))
        self.c_all = mx.zeros((1, 40))

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Process through single LSTM."""
        # Ensure correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]  # (1, 1, 5)
        elif affect.ndim == 2:
            affect = affect[:, None, :]  # (batch, 1, 5)

        # Run LSTM
        h_seq, c_seq = self.lstm(affect, hidden=self.h_all, cell=self.c_all)
        self.h_all = h_seq[:, -1, :].reshape(1, 40)
        self.c_all = c_seq[:, -1, :].reshape(1, 40)

        # Split state for compatibility (not real hierarchy, just split)
        self.h_fast = self.h_all[:, :16]
        self.h_medium = self.h_all[:, 16:32]
        self.h_slow = self.h_all[:, 32:40]

        # Calculate surprise (prediction error)
        current_state = self.h_all
        predicted_state = self.predictor(current_state)
        surprise = float(mx.mean((predicted_state - current_state) ** 2))

        phenomenal_state = self.get_phenomenal_state()
        return phenomenal_state, surprise

    def reset_states(self):
        """Reset LSTM states."""
        self.h_all = mx.zeros((1, 40))
        self.c_all = mx.zeros((1, 40))
        self.h_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def architecture_description(self) -> str:
        return (
            "Single Layer: One LSTM (40-D hidden state).\n"
            f"Parameters: ~{self.parameter_count():,}\n"
            "Purpose: Show benefit of hierarchical timescales."
        )
