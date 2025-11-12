"""
Dense Observers Architecture - 150 Observer Loops

Same as Phase 4 but with 2x observer density to test if more observers
lead to better integrated information or just overfitting.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, List
from .phase4_observers import Phase4Architecture


class DenseObserversArchitecture(Phase4Architecture):
    """
    Dense Observers: Hierarchical + 150 Observer Loops.

    Same structure as Phase 4 but with double the observers:
    - Level 0: 100 observers (was 50)
    - Level 1: 40 observers (was 20)
    - Level 2: 10 observers (was 5)

    Parameters: ~264K (2x Phase 4)
    """

    def _create_observer_hierarchy(self) -> List[nn.Sequential]:
        """Create dense hierarchical observer networks (150 total)."""
        observers = []

        # Level 0: 100 observers (watch main predictor)
        for i in range(100):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 1: 40 observers (watch level 0)
        for i in range(40):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 2: 10 observers (watch level 1)
        for i in range(10):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        return observers

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Process through hierarchical layers with dense observers."""
        # Same as Phase4, but with adjusted observer indices

        # Ensure correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]  # (1, 1, 5)
        elif affect.ndim == 2:
            affect = affect[:, None, :]  # (batch, 1, 5)

        # Fast layer
        h_fast_seq, c_fast_seq = self.fast_lstm(
            affect, hidden=self.h_fast, cell=self.c_fast
        )
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium layer
        h_med_seq, c_med_seq = self.medium_lstm(
            self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium
        )
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow layer
        h_slow_seq = self.slow_gru(
            self.h_medium[:, None, :], hidden=self.h_slow
        )
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Get current state
        current_state = self.get_phenomenal_state()

        # Main predictor
        predicted_state = self.predictor(current_state)

        # Dense observer cascade
        observer_errors = []

        # Level 0: 100 observers
        for i in range(100):
            obs_pred = self.observers[i](predicted_state)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 1: 40 observers
        for i in range(100, 140):
            level0_sample = self.observers[i % 100](predicted_state)
            obs_pred = self.observers[i](level0_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 2: 10 observers
        for i in range(140, 150):
            level1_sample = self.observers[100 + (i % 40)](predicted_state)
            obs_pred = self.observers[i](level1_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Total surprise
        main_error = mx.mean((predicted_state - current_state) ** 2)
        observer_error = mx.mean(mx.array(observer_errors))
        surprise = float(main_error + 0.1 * observer_error)

        return current_state, surprise

    def architecture_description(self) -> str:
        return (
            "Dense Observers: Hierarchical + 150 Observer Loops (2x density).\n"
            f"Parameters: ~{self.parameter_count():,}\n"
            "Purpose: Test if more observers improve Φ or cause overfitting."
        )
