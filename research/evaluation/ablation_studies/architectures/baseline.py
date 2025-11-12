"""
Baseline Architecture - No Temporal Model

This is the simplest baseline: no learning, just returns zeros.
Used to establish floor performance.
"""

import mlx.core as mx
from typing import Tuple
from .base import AblationArchitecture


class BaselineArchitecture(AblationArchitecture):
    """
    Baseline: No temporal model.

    Returns zero states and zero surprise.
    This establishes the floor - how bad can it get?

    Parameters: 0
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__(affect_dim)
        # Dummy predictor for metrics compatibility
        self.predictor = lambda x: x

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Always return zeros."""
        # Keep states at zero
        phenomenal_state = self.get_phenomenal_state()
        surprise = 0.0
        return phenomenal_state, surprise

    def reset_states(self):
        """Reset to zeros (they're already zero)."""
        self.h_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def architecture_description(self) -> str:
        return (
            "Baseline: No temporal model. Returns zero states.\n"
            "Parameters: 0\n"
            "Purpose: Establish floor performance."
        )
