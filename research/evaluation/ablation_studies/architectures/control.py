"""
Control Architecture - Random States

Returns random states to prove that structure matters, not just parameters.
"""

import mlx.core as mx
from typing import Tuple
from .base import AblationArchitecture


class ControlArchitecture(AblationArchitecture):
    """
    Control: Random temporal states.

    Returns random states to show that having parameters isn't enough -
    you need structured learning.

    Parameters: 0 (no trainable params, just randomness)
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__(affect_dim)
        # Dummy predictor for metrics compatibility
        self.predictor = lambda x: mx.random.normal(x.shape) * 0.1

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Return random states."""
        # Generate random states
        self.h_fast = mx.random.normal((1, 16)) * 0.1
        self.h_medium = mx.random.normal((1, 16)) * 0.1
        self.h_slow = mx.random.normal((1, 8)) * 0.1

        phenomenal_state = self.get_phenomenal_state()
        surprise = float(mx.random.uniform(0, 1))

        return phenomenal_state, surprise

    def reset_states(self):
        """Reset to random states."""
        self.h_fast = mx.random.normal((1, 16)) * 0.1
        self.h_medium = mx.random.normal((1, 16)) * 0.1
        self.h_slow = mx.random.normal((1, 8)) * 0.1

    def architecture_description(self) -> str:
        return (
            "Control: Random temporal states.\n"
            "Parameters: 0\n"
            "Purpose: Prove that structure matters (not just having parameters)."
        )
