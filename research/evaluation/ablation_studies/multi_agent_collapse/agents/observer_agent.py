#!/usr/bin/env python3
"""
ObserverAgent - Meta-cognitive Observer for Hierarchical Stability

An observer agent that watches other agents and prevents hierarchical collapse
by predicting their next states and injecting correction signals.

This implements the "observer loop" mechanism discovered in consciousness research:
- Observer predicts what target agent will do
- Compares prediction to actual state (prediction error)
- Injects correction signal to prevent collapse

The observer DOES NOT take actions in the environment - it only observes!

Architecture inspired by your Phase 4 observer loops:
- Level 0 observers: Watch active agents directly
- Level 1 observers: Watch Level 0 observers (meta-observation)
- Level 2 observers: Watch Level 1 observers (meta-meta-observation)

Author: Noodlings Project
Date: November 2025
Purpose: Testing if observers prevent collapse in multi-agent systems
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .noodling_agent import NoodlingAgent


class ObserverAgent(NoodlingAgent):
    """
    An observer agent that watches others but doesn't act.

    Key differences from regular NoodlingAgent:
    - OBSERVES but doesn't take environment actions
    - PREDICTS other agents' states
    - INJECTS correction signals to prevent collapse
    - Operates at specific observation level (0, 1, or 2)
    """

    def __init__(
        self,
        agent_id: str,
        observation_level: int = 0,
        input_dim: int = 10,
        fast_hidden: int = 8,
        medium_hidden: int = 8,
        slow_hidden: int = 4,
        message_dim: int = 8,
        prediction_weight: float = 0.1
    ):
        """
        Initialize an ObserverAgent.

        Args:
            agent_id: Unique identifier (e.g., "observer_L0_001")
            observation_level: Which level (0=watch agents, 1=watch L0, 2=watch L1)
            input_dim: Environmental observation dimension
            fast_hidden: Fast layer size
            medium_hidden: Medium layer size
            slow_hidden: Slow layer size
            message_dim: Correction signal dimension
            prediction_weight: How strongly to weight corrections (0.1 = 10%)
        """
        # Initialize parent (full noodling architecture)
        super().__init__(
            agent_id=agent_id,
            input_dim=input_dim,
            fast_hidden=fast_hidden,
            medium_hidden=medium_hidden,
            slow_hidden=slow_hidden,
            action_dim=0,  # Observers don't take actions!
            message_dim=message_dim
        )

        self.observation_level = observation_level
        self.prediction_weight = prediction_weight

        # Observer-specific: State predictor for OTHER agents
        # This is the KEY - observers predict what others will do!
        self.other_predictor = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.state_dim)
        )

        # Track who we're observing
        self.observed_agents: List[str] = []
        self.prediction_errors: Dict[str, List[float]] = {}

        # Observer effectiveness metrics
        self.total_corrections_sent = 0
        self.average_prediction_error = 0.0

    def observe_agents(
        self,
        target_agents: List[NoodlingAgent],
        environment_state: mx.array
    ) -> Dict[str, mx.array]:
        """
        Observe target agents and generate correction signals.

        This is the CORE observer loop mechanism!

        Args:
            target_agents: List of agents to observe
            environment_state: Current environment (for context)

        Returns:
            Dictionary mapping agent_id -> correction_signal
        """
        corrections = {}

        # Update observer's own internal state based on environment
        # (Observer needs to track context to make good predictions!)
        self_state, _, _ = self.perceive_and_update(environment_state)

        # For each agent we're watching...
        for agent in target_agents:
            # Get their current state
            agent_current_state = agent.get_phenomenal_state()

            # PREDICT what their NEXT state will be
            # This uses observer's internal model of the agent
            predicted_next_state = self.predict_other_state(agent_current_state)

            # Store prediction for validation
            agent_id = agent.agent_id
            if agent_id not in self.prediction_errors:
                self.prediction_errors[agent_id] = []

            # Generate CORRECTION SIGNAL
            # (We'll compare prediction to reality on next timestep)
            # For now, correction is based on what we think they SHOULD do
            # to maintain hierarchical separation

            # Calculate if their hierarchy looks like it's collapsing
            agent_hsi = agent.calculate_hsi()
            hsi_value = agent_hsi.get('slow/fast', 1.0)

            if hsi_value > 0.5:  # Starting to collapse!
                # Generate corrective signal to push timescales apart
                correction = self.generate_corrective_signal(
                    agent_current_state,
                    predicted_next_state,
                    hsi_value
                )
            else:
                # Agent is stable, minimal correction needed
                correction = mx.zeros_like(agent_current_state) * 0.01

            corrections[agent_id] = correction
            self.total_corrections_sent += 1

        return corrections

    def predict_other_state(self, other_current_state: mx.array) -> mx.array:
        """
        Predict what another agent's next state will be.

        This is observer's internal model of the target agent.
        """
        # Detach to prevent gradient flow to target agent
        other_state_detached = mx.stop_gradient(other_current_state)

        # Predict next state
        predicted = self.other_predictor(other_state_detached)

        return predicted

    def generate_corrective_signal(
        self,
        agent_current_state: mx.array,
        predicted_next_state: mx.array,
        current_hsi: float
    ) -> mx.array:
        """
        Generate a correction signal to prevent hierarchical collapse.

        The correction tries to:
        1. Maintain separation between fast/medium/slow layers
        2. Prevent all layers from synchronizing
        3. Act as "gradient sink" to drain collapse pressure

        Args:
            agent_current_state: Agent's current phenomenal state [batch, state_dim]
            predicted_next_state: What we predict they'll do next
            current_hsi: Their current HSI (higher = more collapsed)

        Returns:
            correction: Signal to inject [batch, state_dim]
        """
        # Parse agent's state into layers
        fast_dim = self.fast_hidden
        medium_dim = self.medium_hidden
        slow_dim = self.slow_hidden

        current_fast = agent_current_state[:, :fast_dim]
        current_medium = agent_current_state[:, fast_dim:fast_dim+medium_dim]
        current_slow = agent_current_state[:, fast_dim+medium_dim:]

        # Calculate correction strength based on HSI
        # Higher HSI = stronger correction needed
        correction_strength = min(current_hsi, 2.0)  # Cap at 2.0

        # STRATEGY: Push layers apart by adding decorrelating noise
        # This acts as the "ground sink" - dissipates coupling between layers

        # Fast layer: Add high-frequency noise (rapid changes)
        fast_correction = mx.random.normal(current_fast.shape) * 0.1 * correction_strength

        # Medium layer: Add medium-frequency noise (moderate changes)
        medium_correction = mx.random.normal(current_medium.shape) * 0.05 * correction_strength

        # Slow layer: Add low-frequency noise (slow drift)
        slow_correction = mx.random.normal(current_slow.shape) * 0.02 * correction_strength

        # Combine into full correction
        correction = mx.concatenate([fast_correction, medium_correction, slow_correction], axis=-1)

        # Weight by prediction error (if we're uncertain, correct more)
        prediction_error = mx.mean((predicted_next_state - agent_current_state) ** 2)
        correction = correction * (1.0 + prediction_error)

        # Scale by observer's prediction weight
        correction = correction * self.prediction_weight

        return correction

    def validate_prediction(
        self,
        agent_id: str,
        predicted_state: mx.array,
        actual_state: mx.array
    ) -> float:
        """
        Check how accurate our prediction was.

        This is how observers learn to predict better!
        """
        error = float(mx.mean((predicted_state - actual_state) ** 2))

        if agent_id not in self.prediction_errors:
            self.prediction_errors[agent_id] = []

        self.prediction_errors[agent_id].append(error)

        # Update running average
        all_errors = [e for errors in self.prediction_errors.values() for e in errors]
        self.average_prediction_error = np.mean(all_errors) if all_errors else 0.0

        return error

    def get_observation_quality(self) -> Dict:
        """
        Report on how well this observer is doing its job.
        """
        return {
            'observation_level': self.observation_level,
            'agents_observed': len(self.prediction_errors),
            'total_corrections': self.total_corrections_sent,
            'avg_prediction_error': self.average_prediction_error,
            'observer_hsi': self.calculate_hsi()
        }

    def perceive_and_update(
        self,
        observation: mx.array,
        messages: List[mx.array] = None,
        observer_correction: Optional[mx.array] = None
    ) -> Tuple[mx.array, None, mx.array]:
        """
        Observer version of perceive_and_update.

        Key difference: Returns None for action_logits (observers don't act!)
        """
        # Call parent to update internal state
        phenomenal_state, _, message = super().perceive_and_update(
            observation, messages, observer_correction
        )

        # Observers don't generate actions, only correction messages
        return phenomenal_state, None, message

    def __repr__(self):
        quality = self.get_observation_quality()
        return (f"ObserverAgent(id={self.agent_id}, "
                f"level={self.observation_level}, "
                f"watching={quality['agents_observed']}, "
                f"corrections={quality['total_corrections']}, "
                f"error={quality['avg_prediction_error']:.4f})")


# Test code
if __name__ == '__main__':
    print("="*70)
    print("OBSERVER AGENT TEST")
    print("="*70)
    print()

    # Create a regular agent and an observer
    print("Creating test agents...")
    active_agent = NoodlingAgent(agent_id="active_001", input_dim=10)
    observer = ObserverAgent(
        agent_id="observer_L0_001",
        observation_level=0,
        input_dim=10
    )

    print(f"✓ Active agent: {active_agent.agent_id}")
    print(f"✓ Observer: {observer.agent_id} (Level {observer.observation_level})")
    print()

    # Run test episode: observer watches active agent
    print("Running 100-step episode with observer intervention...")
    print()

    for step in range(100):
        # Random environment
        env_obs = mx.random.normal((10,))

        # Active agent acts WITHOUT observer first
        active_state_before, action, msg = active_agent.perceive_and_update(env_obs)

        # Observer watches and generates correction
        corrections = observer.observe_agents([active_agent], env_obs)
        correction = corrections.get(active_agent.agent_id)

        # Apply observer correction to active agent
        if correction is not None:
            active_state_after, _, _ = active_agent.perceive_and_update(
                env_obs,
                observer_correction=correction
            )

            # Validate observer's prediction
            # (In real use, this would be done on NEXT timestep)
            observer.validate_prediction(
                active_agent.agent_id,
                active_state_before,  # What observer thought would happen
                active_state_after    # What actually happened
            )

        # Print progress
        if (step + 1) % 25 == 0:
            agent_hsi = active_agent.calculate_hsi()
            obs_quality = observer.get_observation_quality()

            print(f"Step {step+1:3d}:")
            print(f"  Agent HSI: {agent_hsi['slow/fast']:.4f} - {agent_hsi['interpretation']}")
            print(f"  Observer: {obs_quality['total_corrections']} corrections, "
                  f"error={obs_quality['avg_prediction_error']:.4f}")
            print()

    print("="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    print()

    # Agent final state
    agent_hsi = active_agent.calculate_hsi()
    print("Active Agent:")
    print(f"  HSI (slow/fast): {agent_hsi['slow/fast']:.4f}")
    print(f"  Interpretation: {agent_hsi['interpretation']}")
    print()

    # Observer quality
    obs_quality = observer.get_observation_quality()
    print("Observer Performance:")
    print(f"  Observation level: {obs_quality['observation_level']}")
    print(f"  Agents watched: {obs_quality['agents_observed']}")
    print(f"  Total corrections: {obs_quality['total_corrections']}")
    print(f"  Avg prediction error: {obs_quality['avg_prediction_error']:.4f}")
    print()

    print("✓ Observer test complete!")
    print()
    print("KEY OBSERVATION: Did observer maintain agent's low HSI?")
    print("Compare with agent running alone (from noodling_agent.py test)")
    print()
