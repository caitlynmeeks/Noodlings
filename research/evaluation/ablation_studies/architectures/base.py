"""
Base Architecture Interface for Ablation Studies

All ablation variants must implement this interface for fair comparison.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod


class AblationArchitecture(nn.Module, ABC):
    """
    Base class for all ablation study architectures.

    All variants must implement:
    - __call__(affect) -> (phenomenal_state, surprise)
    - reset_states() -> None
    - get_phenomenal_state() -> mx.array (40-D)
    - parameter_count() -> int
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__()
        self.affect_dim = affect_dim

        # All architectures must have these attributes for metrics
        self.h_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    @abstractmethod
    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """
        Process affect input and return phenomenal state + surprise.

        Args:
            affect: (batch, 5) or (5,) affect vector

        Returns:
            phenomenal_state: (1, 40) full state vector
            surprise: scalar float
        """
        pass

    @abstractmethod
    def reset_states(self):
        """Reset all hidden states to zero."""
        pass

    def get_phenomenal_state(self) -> mx.array:
        """
        Return full 40-D phenomenal state.

        Returns:
            mx.array: (1, 40) concatenated [h_fast, h_medium, h_slow]
        """
        return mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

    def parameter_count(self) -> int:
        """
        Return total number of trainable parameters.

        Returns:
            int: Total parameter count
        """
        total = 0
        for name, param in self.parameters().items():
            if 'trainable' in name:
                for p in param:
                    total += p.size
        return total

    def architecture_name(self) -> str:
        """Return human-readable architecture name."""
        return self.__class__.__name__

    def architecture_description(self) -> str:
        """Return detailed description of this architecture."""
        return "Base architecture (should be overridden)"
