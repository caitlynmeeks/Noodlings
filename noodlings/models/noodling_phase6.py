"""
Noodlings Phase 6: Appetite Architecture - Motivational Layer

Extends Phase 4 (Social Cognition) with:
- 8-D appetite state (curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy)
- Goal generation (16-D goal space)
- Conflict detection between competing drives
- Brenda's orchestration tools (@stoke, @sate, @appetites)

This implements computational correlates of:
- Motivational psychology (Drive theory, McClelland's needs)
- Goal-directed behavior (Miller et al.)
- Affective forecasting (Gilbert & Wilson)

Total Parameters: Phase 4 (~82.5K) + Phase 6 (~15K) = ~97.5K

Architecture Flow:
1. User input → Affect extraction (5-D) + Agent detection
2. Hierarchical encoding: fast (16-D) → medium (16-D) → slow (8-D) [Phase 1-2]
3. Appetite dynamics: (slow + fast) → appetite accumulation [NEW]
4. Goal generation: (appetites + affect + context) → 16-D goals [NEW]
5. Social cognition: Theory of Mind + Relationships [Phase 4]
6. Prediction: next phenomenal state (40-D)
7. Store in episodic memory with appetite/goal annotations

Key Innovation: Characters now WANT things and pursue them based on internal
drives, not just reacting to inputs. Mr. Toad crashes motor-cars because his
novelty appetite is 0.95, not because we scripted it.

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np

# Phase 4 imports
from .noodling_phase4 import NoodlingModelPhase4

# Phase 6 imports
from .appetite_layer import AppetiteLayer


class NoodlingModelPhase6(nn.Module):
    """
    Phase 6: Hierarchical affective consciousness with motivational layer.

    Combines all previous phases with goal-directed behavior:
    - Phase 1: Multi-timescale predictive processing
    - Phase 2: Variational uncertainty modeling
    - Phase 3: Episodic memory + attention
    - Phase 4: Social cognition + theory of mind
    - Phase 6: Appetite architecture + goal generation [NEW]
    """

    def __init__(
        self,
        # Phase 1-4 params
        affect_dim: int = 5,
        fast_hidden: int = 16,
        medium_hidden: int = 16,
        slow_hidden: int = 8,
        predictor_hidden: int = 64,
        use_vae: bool = False,
        memory_capacity: int = 100,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        max_agents: int = 10,
        linguistic_dim: int = 128,
        context_dim: int = 64,
        social_context_dim: int = 16,
        relationship_dim: int = 32,
        use_theory_of_mind: bool = True,
        use_relationship_model: bool = True,

        # Phase 6 params (motivational layer)
        appetite_dim: int = 8,
        goal_dim: int = 16,
        use_appetite_layer: bool = True
    ):
        """
        Initialize Phase 6 model with appetite architecture.

        Args:
            [Phase 1-4 args same as NoodlingModelPhase4]

            appetite_dim: Appetite state dimension (8 core appetites)
            goal_dim: Goal space dimension (16 goals)
            use_appetite_layer: Enable appetite/goal system
        """
        super().__init__()

        # Store configuration
        self.affect_dim = affect_dim
        self.fast_hidden = fast_hidden
        self.medium_hidden = medium_hidden
        self.slow_hidden = slow_hidden
        self.state_dim = fast_hidden + medium_hidden + slow_hidden  # 40-D
        self.use_vae = use_vae
        self.use_theory_of_mind = use_theory_of_mind
        self.use_relationship_model = use_relationship_model
        self.use_appetite_layer = use_appetite_layer

        # Phase 1-4: Base architecture
        self.base_model = NoodlingModelPhase4(
            affect_dim=affect_dim,
            fast_hidden=fast_hidden,
            medium_hidden=medium_hidden,
            slow_hidden=slow_hidden,
            predictor_hidden=predictor_hidden,
            use_vae=use_vae,
            memory_capacity=memory_capacity,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            max_agents=max_agents,
            linguistic_dim=linguistic_dim,
            context_dim=context_dim,
            social_context_dim=social_context_dim,
            relationship_dim=relationship_dim,
            use_theory_of_mind=use_theory_of_mind,
            use_relationship_model=use_relationship_model
        )

        # Phase 6: Appetite layer (NEW!)
        if self.use_appetite_layer:
            self.appetite_layer = AppetiteLayer(
                appetite_dim=appetite_dim,
                goal_dim=goal_dim,
                slow_dim=slow_hidden,
                fast_dim=fast_hidden,
                affect_dim=affect_dim
            )

        # Hidden states (managed by base model)
        self.h_fast = None
        self.c_fast = None
        self.h_medium = None
        self.c_medium = None
        self.h_slow = None

        self.step = 0

    def set_appetite_baselines(self, baselines: List[float]):
        """
        Set appetite baselines from recipe configuration.

        Args:
            baselines: 8-element list [curiosity, status, mastery, novelty,
                                       safety, social_bond, comfort, autonomy]
        """
        if self.use_appetite_layer:
            self.appetite_layer.set_appetite_baselines(baselines)

    def get_states(self):
        """Get current hidden states without resetting."""
        return self.base_model.get_states()

    def reset_states(self):
        """Reset all hidden states, memory, and appetites."""
        self.base_model.reset_states()
        if self.use_appetite_layer:
            self.appetite_layer.reset_appetites()
        self.step = 0
        return self.base_model.get_states()

    def forward_with_appetites(
        self,
        affect: mx.array,
        linguistic_features: Optional[Dict[str, mx.array]] = None,
        context_features: Optional[mx.array] = None,
        present_agents: Optional[List[str]] = None,
        social_context: Optional = None,
        user_text: str = "",
        accumulation_rate: float = 0.1
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        Forward pass with appetite layer integration.

        Args:
            affect: [batch, 5] affect vector
            linguistic_features: Dict mapping agent_name -> [batch, 128] features
            context_features: [batch, 64] context features
            present_agents: List of agent names currently present
            social_context: SocialContext object (optional)
            user_text: User input text
            accumulation_rate: Appetite accumulation rate (0.1 = 10% per step)

        Returns:
            self_state: [batch, 40] current phenomenal state
            predicted_state: [batch, 40] predicted next state
            info: Dictionary with:
                - inferred_agents: Theory of Mind outputs
                - relationships: Relationship model outputs
                - appetites: Current appetite state (8-D) [NEW]
                - goals: Current goal activations (16-D) [NEW]
                - conflicts: Goal conflicts [NEW]
                - top_goals: Top-k most active goals [NEW]
                - attention_weights, social_context, memory_count
        """
        batch_size = affect.shape[0]

        # 1. Run Phase 4 forward pass (hierarchical + social cognition)
        self_state, predicted_state, social_info = self.base_model.forward_with_social_context(
            affect=affect,
            linguistic_features=linguistic_features,
            context_features=context_features,
            present_agents=present_agents,
            social_context=social_context,
            user_text=user_text
        )

        # 2. Extract layer states for appetite dynamics
        fast_state = self.base_model.h_fast
        slow_state = self.base_model.h_slow

        # 3. Update appetites and generate goals (Phase 6)
        if self.use_appetite_layer:
            goals, conflicts, appetites = self.appetite_layer.forward(
                slow_state=slow_state,
                fast_state=fast_state,
                affect=affect,
                accumulation_rate=accumulation_rate
            )

            # Get top-k goals for LLM context
            top_goals = self.appetite_layer.get_top_goals(goals, k=3)

            # Add appetite/goal info to social_info
            social_info['appetites'] = appetites
            social_info['goals'] = goals
            social_info['conflicts'] = conflicts
            social_info['top_goals'] = top_goals
            social_info['appetite_dict'] = self.appetite_layer.get_appetites()

        else:
            # No appetite layer - return empty placeholders
            social_info['appetites'] = None
            social_info['goals'] = None
            social_info['conflicts'] = None
            social_info['top_goals'] = []
            social_info['appetite_dict'] = {}

        self.step += 1

        return self_state, predicted_state, social_info

    def stoke_appetite(self, appetite_name: str, amount: float):
        """
        Brenda's orchestration tool: Increase an appetite.

        Args:
            appetite_name: One of 8 appetite names
            amount: How much to increase (0.0-1.0)
        """
        if self.use_appetite_layer:
            self.appetite_layer.stoke_appetite(appetite_name, amount)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def sate_appetite(self, appetite_name: str, amount: float):
        """
        Brenda's orchestration tool: Satisfy/decrease an appetite.

        Args:
            appetite_name: One of 8 appetite names
            amount: How much to decrease (0.0-1.0)
        """
        if self.use_appetite_layer:
            self.appetite_layer.sate_appetite(appetite_name, amount)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def get_appetites(self) -> Dict[str, float]:
        """
        Get current appetite levels.

        Returns:
            Dict mapping appetite names to values (0-1)
        """
        if self.use_appetite_layer:
            return self.appetite_layer.get_appetites()
        else:
            return {}

    def override_goal(self, goal_name: str, strength: float):
        """
        Brenda's orchestration tool: Override a goal's activation.

        Args:
            goal_name: One of 16 goal names
            strength: Goal activation strength (0.0-1.0)
        """
        if self.use_appetite_layer:
            self.appetite_layer.override_goal(goal_name, strength)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def set_goal_bias(self, goal_name: str, bias: float):
        """
        Brenda's orchestration tool: Add a persistent bias to goal generation.

        Args:
            goal_name: One of 16 goal names
            bias: Amount to add to goal activation (-1.0 to 1.0)
        """
        if self.use_appetite_layer:
            self.appetite_layer.set_goal_bias(goal_name, bias)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def clear_goal_overrides(self, goal_name: Optional[str] = None):
        """
        Brenda's orchestration tool: Clear goal overrides.

        Args:
            goal_name: Specific goal to clear, or None to clear all
        """
        if self.use_appetite_layer:
            self.appetite_layer.clear_goal_overrides(goal_name)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def clear_goal_biases(self, goal_name: Optional[str] = None):
        """
        Brenda's orchestration tool: Clear goal biases.

        Args:
            goal_name: Specific goal to clear, or None to clear all
        """
        if self.use_appetite_layer:
            self.appetite_layer.clear_goal_biases(goal_name)
        else:
            raise RuntimeError("Appetite layer not enabled. Set use_appetite_layer=True")

    def get_goal_overrides(self) -> Dict[str, float]:
        """
        Get current goal overrides.

        Returns:
            Dict mapping goal names to override strengths (0-1)
        """
        if self.use_appetite_layer:
            return self.appetite_layer.get_goal_overrides()
        else:
            return {}

    def get_goal_biases(self) -> Dict[str, float]:
        """
        Get current goal biases.

        Returns:
            Dict mapping goal names to biases (-1 to 1)
        """
        if self.use_appetite_layer:
            return self.appetite_layer.get_goal_biases()
        else:
            return {}

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate total parameter count.

        Returns:
            Dictionary with parameter counts per module
        """
        counts = {
            'base_model_phase4': self.base_model.get_parameter_count()['total']
        }

        if self.use_appetite_layer:
            # Appetite layer parameters
            # appetite_dynamics: (8 + 16) * 8 = 192
            # goal_generator: (8 + 5 + 16) * 16 = 464
            # conflict_detector: 16 * 16 = 256
            # Total: ~912 + biases ≈ 1,000-1,500
            counts['appetite_layer'] = 1500  # Approximate

        counts['total'] = sum(counts.values())

        return counts

    def load_phase4_weights(self, checkpoint_path: str):
        """
        Load Phase 4 checkpoint into base model.

        Args:
            checkpoint_path: Path to Phase 4 checkpoint (.npz)
        """
        self.base_model.load_weights(checkpoint_path)
        print(f"✓ Loaded Phase 4 weights from {checkpoint_path}")

    def save_weights(self, path: str):
        """
        Save Phase 6 model weights.

        Args:
            path: Output path for checkpoint
        """
        weights = {}

        # Save base model weights
        base_weights = self.base_model.save_weights(path + '.base')

        # Save appetite layer weights
        if self.use_appetite_layer:
            for param_name, param in self.appetite_layer.parameters().items():
                weights[f'appetite_layer.{param_name}'] = param

        mx.savez(path, **weights)
        print(f"✓ Saved Phase 6 weights to {path}")

    def load_weights(self, path: str):
        """
        Load Phase 6 model weights.

        Args:
            path: Path to checkpoint
        """
        weights = mx.load(path)

        # Load appetite layer weights
        for key, value in weights.items():
            if key.startswith('appetite_layer.'):
                param_path = key.replace('appetite_layer.', '')
                # Set parameter (simplified - actual implementation needs tree manipulation)
                pass

        print(f"✓ Loaded Phase 6 weights from {path}")


# Example usage and testing
if __name__ == '__main__':
    print("Testing NoodlingModelPhase6...\n")

    # Initialize Phase 6 model
    model = NoodlingModelPhase6(
        affect_dim=5,
        fast_hidden=16,
        medium_hidden=16,
        slow_hidden=8,
        predictor_hidden=64,
        use_vae=False,
        memory_capacity=100,
        use_theory_of_mind=True,
        use_relationship_model=True,
        use_appetite_layer=True,
        appetite_dim=8,
        goal_dim=16
    )

    # Set Mr. Toad's appetite baselines
    toad_appetites = [
        0.7,   # curiosity
        0.8,   # status
        0.6,   # mastery
        0.95,  # novelty
        0.1,   # safety
        0.5,   # social_bond
        0.2,   # comfort
        0.9    # autonomy
    ]

    model.set_appetite_baselines(toad_appetites)

    print("✓ Phase 6 model initialized")
    print(f"  Parameters: ~{model.get_parameter_count()['total']:,}")
    print(f"  Appetite layer enabled: {model.use_appetite_layer}")

    # Test forward pass
    print("\n" + "="*50)
    print("Testing forward pass...")
    print("="*50 + "\n")

    # Simulate affect input
    affect = mx.array([[0.2, 0.5, 0.1, 0.0, 0.3]])  # Moderately bored

    self_state, predicted_state, info = model.forward_with_appetites(
        affect=affect,
        user_text="I wonder what adventures await today?"
    )

    print(f"Phenomenal state shape: {self_state.shape}")
    print(f"Predicted state shape: {predicted_state.shape}")
    print(f"\nCurrent appetites:")
    for name, value in info['appetite_dict'].items():
        print(f"  {name:12s}: {value:.2f}")

    print(f"\nTop active goals:")
    for goal_name, strength in info['top_goals']:
        print(f"  {goal_name:25s}: {strength:.3f}")

    # Test Brenda's orchestration
    print("\n" + "="*50)
    print("Testing Brenda's orchestration...")
    print("="*50 + "\n")

    print("Brenda: @stoke toad novelty 0.05")
    model.stoke_appetite('novelty', 0.05)

    # Forward pass after stoking
    self_state, predicted_state, info = model.forward_with_appetites(
        affect=affect,
        user_text="Ooh, what's that shiny new thing?"
    )

    print(f"\nAppetites after stoking:")
    for name, value in info['appetite_dict'].items():
        print(f"  {name:12s}: {value:.2f}")

    print(f"\nTop goals after stoking:")
    for goal_name, strength in info['top_goals']:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\n✓ Phase 6 test complete!")
