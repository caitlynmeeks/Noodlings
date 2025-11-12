"""
Phase 4 Architecture - Hierarchical with 75 Observer Loops

The full Noodlings Phase 4 architecture with integrated information via
meta-observer networks creating closed causal loops.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, List
from .base import AblationArchitecture


class Phase4Architecture(AblationArchitecture):
    """
    Phase 4: Hierarchical + 75 Observer Loops.

    Three-level hierarchy PLUS 75 hierarchical observer networks that
    watch the main network's predictions, creating closed causal loops
    and high integrated information (Φ).

    Observer structure:
    - Level 0: 50 observers watching main predictor
    - Level 1: 20 observers watching level 0
    - Level 2: 5 observers watching level 1

    Parameters: ~132K
    """

    def __init__(self, affect_dim: int = 5):
        super().__init__(affect_dim)

        # Core hierarchy (same as hierarchical architecture)
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=16)
        self.c_fast = mx.zeros((1, 16))

        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)
        self.c_medium = mx.zeros((1, 16))

        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Main predictor
        joint_dim = 16 + 16 + 8  # 40
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim)
        )

        # Observer networks (75 total)
        self.observers = self._create_observer_hierarchy()

    def _create_observer_hierarchy(self) -> List[nn.Sequential]:
        """Create hierarchical observer networks."""
        observers = []

        # Level 0: 50 observers (watch main predictor)
        for i in range(50):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 1: 20 observers (watch level 0)
        for i in range(20):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 2: 5 observers (watch level 1)
        for i in range(5):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        return observers

    def __call__(self, affect: mx.array) -> Tuple[mx.array, float]:
        """Process through hierarchical layers with observers."""
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

        # Observer cascade (creates integrated information)
        observer_errors = []

        # Level 0 observers (watch main prediction)
        for i in range(50):
            obs_pred = self.observers[i](predicted_state)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 1 observers (watch level 0)
        for i in range(50, 70):
            # Sample from level 0 predictions
            level0_sample = self.observers[i % 50](predicted_state)
            obs_pred = self.observers[i](level0_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 2 observers (watch level 1)
        for i in range(70, 75):
            # Sample from level 1 predictions
            level1_sample = self.observers[50 + (i % 20)](predicted_state)
            obs_pred = self.observers[i](level1_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Total surprise includes main + observer prediction errors
        main_error = mx.mean((predicted_state - current_state) ** 2)
        observer_error = mx.mean(mx.array(observer_errors))
        surprise = float(main_error + 0.1 * observer_error)  # Weight observer contribution

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
            "Phase 4: Hierarchical + 75 Observer Loops.\n"
            f"Parameters: ~{self.parameter_count():,}\n"
            "Purpose: Full architecture with integrated information (Φ)."
        )
