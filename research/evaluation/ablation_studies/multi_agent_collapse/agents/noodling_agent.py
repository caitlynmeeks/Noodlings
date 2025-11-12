#!/usr/bin/env python3
"""
NoodlingAgent - Multi-timescale Agent for Hierarchical Collapse Experiments

An agent with fast/medium/slow layers (just like your consciousness architecture!)
designed to test whether observer loops prevent collapse in multi-agent systems.

Architecture mirrors your Phase 4 noodlings:
- Fast layer (LSTM, 8-D): Immediate reactions (updates every step)
- Medium layer (LSTM, 8-D): Recent history (updates every step, slower learning)
- Slow layer (GRU, 4-D): Long-term traits (updates every step, very slow learning)

Author: Noodlings Project
Date: November 2025
Purpose: Testing General Hierarchical Collapse Principle
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class NoodlingAgent(nn.Module):
    """
    A multi-timescale agent inspired by consciousness architecture.

    Has fast (immediate), medium (contextual), and slow (trait) layers
    that maintain hierarchical separation - unless they collapse!
    """

    def __init__(
        self,
        agent_id: str,
        input_dim: int = 10,
        fast_hidden: int = 8,
        medium_hidden: int = 8,
        slow_hidden: int = 4,
        action_dim: int = 3,
        message_dim: int = 8,
        fast_lr: float = 1e-3,
        medium_lr: float = 5e-4,
        slow_lr: float = 1e-4
    ):
        """
        Initialize a NoodlingAgent.

        Args:
            agent_id: Unique identifier
            input_dim: Environmental observation dimension
            fast_hidden: Fast layer hidden size (immediate reactions)
            medium_hidden: Medium layer hidden size (context)
            slow_hidden: Slow layer hidden size (personality)
            action_dim: Number of possible actions
            message_dim: Dimension of messages to other agents
            fast_lr: Learning rate for fast layer (high for rapid adaptation)
            medium_lr: Learning rate for medium layer (moderate)
            slow_lr: Learning rate for slow layer (low for stability)
        """
        super().__init__()

        self.agent_id = agent_id
        self.input_dim = input_dim
        self.fast_hidden = fast_hidden
        self.medium_hidden = medium_hidden
        self.slow_hidden = slow_hidden
        self.state_dim = fast_hidden + medium_hidden + slow_hidden  # 20-D
        self.action_dim = action_dim
        self.message_dim = message_dim

        # Learning rates (different timescales!)
        self.fast_lr = fast_lr
        self.medium_lr = medium_lr
        self.slow_lr = slow_lr

        # Three-level hierarchy (THE CORE ARCHITECTURE)
        self.fast_lstm = nn.LSTM(input_size=input_dim, hidden_size=fast_hidden)
        self.c_fast = mx.zeros((1, fast_hidden))

        self.medium_lstm = nn.LSTM(input_size=fast_hidden, hidden_size=medium_hidden)
        self.c_medium = mx.zeros((1, medium_hidden))

        self.slow_gru = nn.GRU(input_size=medium_hidden, hidden_size=slow_hidden)

        # Action policy (maps phenomenal state to action)
        self.action_head = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # Message generator (communicates with other agents)
        self.message_head = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, message_dim)
        )

        # Predictor (predicts own next state - for learning)
        self.predictor = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.state_dim)
        )

        # State history for HSI calculation (THE KEY METRIC!)
        self.fast_history = deque(maxlen=200)
        self.medium_history = deque(maxlen=200)
        self.slow_history = deque(maxlen=200)
        self.phenomenal_history = deque(maxlen=200)

        # Hidden states
        self.h_fast = mx.zeros((1, fast_hidden))
        self.h_medium = mx.zeros((1, medium_hidden))
        self.h_slow = mx.zeros((1, slow_hidden))

        # Metrics
        self.step_count = 0
        self.total_reward = 0.0
        self.prediction_errors = []

    def reset_states(self):
        """Reset hidden states (like starting a new conversation)."""
        self.h_fast = mx.zeros((1, self.fast_hidden))
        self.c_fast = mx.zeros((1, self.fast_hidden))
        self.h_medium = mx.zeros((1, self.medium_hidden))
        self.c_medium = mx.zeros((1, self.medium_hidden))
        self.h_slow = mx.zeros((1, self.slow_hidden))

        self.fast_history.clear()
        self.medium_history.clear()
        self.slow_history.clear()
        self.phenomenal_history.clear()

        self.step_count = 0
        self.total_reward = 0.0
        self.prediction_errors.clear()

    def get_phenomenal_state(self) -> mx.array:
        """
        Get current full internal state (the agent's 'experience').

        This is what the agent 'feels' - combines all timescales.
        """
        return mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=-1)

    def perceive_and_update(
        self,
        observation: mx.array,
        messages: List[mx.array] = None,
        observer_correction: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Process observation through hierarchical layers.

        This is the CORE of the multi-timescale architecture!

        Args:
            observation: Environmental input [batch, input_dim]
            messages: Messages from other agents (incorporated into observation)
            observer_correction: Correction signal from observer (if present)

        Returns:
            phenomenal_state: Current full state [batch, state_dim]
            action_logits: Action probabilities [batch, action_dim]
            message: Message to send [batch, message_dim]
        """
        # Ensure correct shape
        if observation.ndim == 1:
            observation = observation[None, :]  # Add batch dim

        # Incorporate messages into observation (simple concatenation + projection)
        if messages:
            # Average all incoming messages
            avg_message = mx.mean(mx.stack(messages), axis=0)
            # Expand observation with message information
            # (In real implementation, might want attention mechanism here)
            observation = observation  # Keep simple for now

        # Add sequence dimension for LSTMs
        obs_seq = observation[:, None, :]  # [batch, 1, input_dim]

        # === FAST LAYER: Immediate reactions ===
        h_fast_seq, c_fast_seq = self.fast_lstm(
            obs_seq, hidden=self.h_fast, cell=self.c_fast
        )
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, self.fast_hidden)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, self.fast_hidden)

        # === MEDIUM LAYER: Contextual processing ===
        h_fast_input = self.h_fast[:, None, :]
        h_med_seq, c_med_seq = self.medium_lstm(
            h_fast_input, hidden=self.h_medium, cell=self.c_medium
        )
        self.h_medium = h_med_seq[:, -1, :].reshape(1, self.medium_hidden)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, self.medium_hidden)

        # === SLOW LAYER: Trait-level processing ===
        h_med_input = self.h_medium[:, None, :]
        h_slow_seq = self.slow_gru(h_med_input, hidden=self.h_slow)
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, self.slow_hidden)

        # Get full phenomenal state
        phenomenal_state = self.get_phenomenal_state()

        # === OBSERVER CORRECTION (THE KEY MECHANISM!) ===
        if observer_correction is not None:
            # Apply correction (like your observer loop injection!)
            # This modulates the phenomenal state based on observer prediction error
            injection_strength = 0.1  # Same as your observer loops
            phenomenal_state = phenomenal_state + injection_strength * observer_correction

            # Update hidden states to reflect correction
            # (Split corrected state back into layers)
            corrected_fast = phenomenal_state[:, :self.fast_hidden]
            corrected_medium = phenomenal_state[:, self.fast_hidden:self.fast_hidden+self.medium_hidden]
            corrected_slow = phenomenal_state[:, self.fast_hidden+self.medium_hidden:]

            self.h_fast = corrected_fast
            self.h_medium = corrected_medium
            self.h_slow = corrected_slow

        # Store in history for HSI calculation
        self.fast_history.append(float(mx.mean(mx.abs(self.h_fast))))
        self.medium_history.append(float(mx.mean(mx.abs(self.h_medium))))
        self.slow_history.append(float(mx.mean(mx.abs(self.h_slow))))
        self.phenomenal_history.append(phenomenal_state[0].tolist())

        # Generate action and message
        action_logits = self.action_head(phenomenal_state)
        message = self.message_head(phenomenal_state)

        self.step_count += 1

        return phenomenal_state, action_logits, message

    def predict_next_state(self, current_state: mx.array) -> mx.array:
        """
        Predict what the agent's next state will be.

        This is used by observers to generate corrections!
        """
        return self.predictor(current_state)

    def calculate_surprise(self, predicted_state: mx.array, actual_state: mx.array) -> float:
        """
        Calculate prediction error (surprise).

        This drives learning - minimize surprise!
        """
        error = mx.mean((predicted_state - actual_state) ** 2)
        return float(error)

    def calculate_hsi(self, window: int = 100) -> Dict[str, float]:
        """
        Calculate Hierarchical Separation Index (HSI).

        This is THE KEY METRIC - measures if timescales are separated or collapsed!

        Returns:
            Dictionary with HSI values and interpretation
        """
        if len(self.fast_history) < window:
            return {
                'slow/fast': float('nan'),
                'slow/medium': float('nan'),
                'medium/fast': float('nan'),
                'interpretation': 'Not enough data'
            }

        # Get recent history
        fast_recent = list(self.fast_history)[-window:]
        medium_recent = list(self.medium_history)[-window:]
        slow_recent = list(self.slow_history)[-window:]

        # Calculate variances (how much does each layer change?)
        fast_var = np.var(fast_recent)
        medium_var = np.var(medium_recent)
        slow_var = np.var(slow_recent)

        # Calculate HSI ratios
        hsi_slow_fast = slow_var / fast_var if fast_var > 1e-10 else float('nan')
        hsi_slow_medium = slow_var / medium_var if medium_var > 1e-10 else float('nan')
        hsi_medium_fast = medium_var / fast_var if fast_var > 1e-10 else float('nan')

        # Interpretation (same thresholds as your work!)
        if hsi_slow_fast < 0.3:
            interpretation = "Good separation: hierarchical timescales present"
        elif hsi_slow_fast < 1.0:
            interpretation = "Moderate separation: some hierarchy"
        else:
            interpretation = "Poor separation: layers change at similar rates (COLLAPSE!)"

        return {
            'slow/fast': hsi_slow_fast,
            'slow/medium': hsi_slow_medium,
            'medium/fast': hsi_medium_fast,
            'fast_var': fast_var,
            'medium_var': medium_var,
            'slow_var': slow_var,
            'interpretation': interpretation
        }

    def update_from_reward(self, reward: float):
        """
        Update agent based on reward received.

        For now, just track it. Full RL training could be added later.
        """
        self.total_reward += reward

    def get_state_statistics(self) -> Dict:
        """Get statistics about agent's internal states over time."""
        if len(self.phenomenal_history) < 10:
            return {'error': 'Not enough history'}

        # Convert to numpy for analysis
        states = np.array(list(self.phenomenal_history))

        return {
            'mean_state': np.mean(states, axis=0).tolist(),
            'std_state': np.std(states, axis=0).tolist(),
            'state_norm': float(np.mean(np.linalg.norm(states, axis=1))),
            'steps': self.step_count,
            'total_reward': self.total_reward
        }

    def __repr__(self):
        return f"NoodlingAgent(id={self.agent_id}, steps={self.step_count}, HSI={self.calculate_hsi().get('slow/fast', 'N/A')})"


# Test code (runs if file executed directly)
if __name__ == '__main__':
    print("="*70)
    print("NOODLING AGENT TEST")
    print("="*70)
    print()

    # Create agent
    agent = NoodlingAgent(agent_id="test_agent_001")
    print(f"✓ Created agent: {agent.agent_id}")
    print(f"  Architecture: {agent.fast_hidden}-{agent.medium_hidden}-{agent.slow_hidden}")
    print(f"  State dimension: {agent.state_dim}")
    print()

    # Run test episode
    print("Running test episode (100 steps)...")
    print()

    for step in range(100):
        # Random observation (simulating environment)
        obs = mx.random.normal((agent.input_dim,))

        # Process observation
        state, action_logits, message = agent.perceive_and_update(obs)

        # Fake reward
        reward = np.random.uniform(-1, 1)
        agent.update_from_reward(reward)

        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            hsi = agent.calculate_hsi()
            print(f"  Step {step+1:3d}: HSI(slow/fast) = {hsi['slow/fast']:.4f} - {hsi['interpretation']}")

    print()
    print("="*70)
    print("FINAL STATE ANALYSIS")
    print("="*70)

    # Final HSI
    hsi = agent.calculate_hsi()
    print(f"\nHierarchical Separation Index:")
    print(f"  Slow/Fast:   {hsi['slow/fast']:.4f}")
    print(f"  Slow/Medium: {hsi['slow/medium']:.4f}")
    print(f"  Medium/Fast: {hsi['medium/fast']:.4f}")
    print(f"\n  → {hsi['interpretation']}")

    # State statistics
    stats = agent.get_state_statistics()
    print(f"\nAgent Statistics:")
    print(f"  Total steps:  {stats['steps']}")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  State norm:   {stats['state_norm']:.4f}")

    print()
    print("✓ Agent test complete!")
    print()
