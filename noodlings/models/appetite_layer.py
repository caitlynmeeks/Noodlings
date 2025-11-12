"""
Appetite Layer - Phase 6 Motivational Architecture

Implements goal-directed behavior through accumulating appetites.
Sits alongside the slow layer and generates goals based on drives.

8 Core Appetites:
- curiosity: Drive to learn and explore
- status: Desire for recognition/prestige
- mastery: Need to excel and improve
- novelty: Craving for new experiences
- safety: Need for security/stability
- social_bond: Desire for connection
- comfort: Need for ease and pleasure
- autonomy: Drive for independence

Goals are generated when appetites reach high levels + opportunities appear.
Conflicts arise when multiple goals compete for attention.

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional


class AppetiteLayer(nn.Module):
    """
    Drives goal-directed behavior through accumulating appetites.

    Architecture:
    - Appetite state (8-D): Accumulates over time when unsatisfied
    - Appetite dynamics: Maps (slow + fast) → appetite changes
    - Goal generator: Maps (appetites + affect + context) → goals
    - Conflict detector: Identifies incompatible goals

    Integration with Phase 4:
    - Takes slow_state (8-D) and fast_state (16-D) as input
    - Outputs goals (16-D), conflicts, and current appetite state
    - Goals influence LLM prompt context ("pursue_excitement" etc.)
    """

    def __init__(
        self,
        appetite_dim: int = 8,
        goal_dim: int = 16,
        slow_dim: int = 8,
        fast_dim: int = 16,
        affect_dim: int = 5
    ):
        """
        Initialize appetite layer.

        Args:
            appetite_dim: Appetite state dimension (8 appetites)
            goal_dim: Goal space dimension (16 goals)
            slow_dim: Slow layer dimension (personality)
            fast_dim: Fast layer dimension (immediate context)
            affect_dim: Affect vector dimension
        """
        super().__init__()

        self.appetite_dim = appetite_dim
        self.goal_dim = goal_dim
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        self.affect_dim = affect_dim

        # Appetite state (persistent, accumulates over time)
        self.appetites = mx.zeros((1, appetite_dim))

        # Appetite baselines (set from recipe configuration)
        self.appetite_baselines = mx.ones((1, appetite_dim)) * 0.5

        # Map from (slow_layer + fast_layer) → appetite accumulation
        # Slow layer = personality traits that drive appetites
        # Fast layer = immediate context that triggers/sates appetites
        self.appetite_dynamics = nn.Linear(slow_dim + fast_dim, appetite_dim)

        # Map from (appetites + affect + fast_context) → goal activation
        # Goals emerge when appetites are high + opportunities present
        self.goal_generator = nn.Linear(appetite_dim + affect_dim + fast_dim, goal_dim)

        # Detect conflicts (learned: which goals negatively correlate)
        # E.g., "pursue_excitement" conflicts with "ensure_safety"
        self.conflict_detector = nn.Linear(goal_dim, goal_dim)

        # Appetite names (canonical order)
        self.appetite_names = [
            'curiosity', 'status', 'mastery', 'novelty',
            'safety', 'social_bond', 'comfort', 'autonomy'
        ]

        # Goal names (16 common goals)
        self.goal_names = [
            "explore_environment",      # curiosity-driven
            "seek_social_connection",   # social_bond-driven
            "demonstrate_competence",   # mastery + status
            "pursue_novelty",           # novelty-driven
            "ensure_safety",            # safety-driven
            "gain_status",              # status-driven
            "seek_comfort",             # comfort-driven
            "maintain_autonomy",        # autonomy-driven
            "help_friend",              # social_bond + status
            "avoid_consequences",       # safety + autonomy
            "restore_reputation",       # status + social_bond
            "learn_skill",              # curiosity + mastery
            "impress_others",           # status + novelty
            "solve_problem",            # mastery + curiosity
            "express_emotion",          # autonomy + comfort
            "achieve_goal"              # autonomy + mastery
        ]

        # Brenda's narrative control (goal overrides and biases)
        self.goal_overrides = {}  # goal_name -> strength (0-1, replaces natural generation)
        self.goal_biases = {}     # goal_name -> bias (-1 to 1, adds to natural generation)

    def set_appetite_baselines(self, baselines: List[float]):
        """
        Set appetite baselines from recipe configuration.

        Args:
            baselines: 8-element list in canonical appetite order
        """
        if len(baselines) != self.appetite_dim:
            raise ValueError(f"Expected {self.appetite_dim} appetite baselines, got {len(baselines)}")

        self.appetite_baselines = mx.array(baselines, dtype=mx.float32)[None, :]
        # Initialize appetites to baselines
        self.appetites = mx.array(self.appetite_baselines)

    def forward(
        self,
        slow_state: mx.array,
        fast_state: mx.array,
        affect: mx.array,
        accumulation_rate: float = 0.1
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Update appetites and generate goals.

        Args:
            slow_state: [batch, 8] personality/disposition
            fast_state: [batch, 16] immediate affective context
            affect: [batch, 5] current affect vector
            accumulation_rate: How quickly appetites accumulate (0.1 = 10% per step)

        Returns:
            goals: [batch, 16] active goal strengths (0-1)
            conflicts: [batch, 16] conflict tensor (values -1 to 1, negative = incompatible)
            appetites: [batch, 8] current appetite state (0-1)
        """
        # Ensure correct shapes
        if slow_state.ndim == 1:
            slow_state = slow_state[None, :]
        if fast_state.ndim == 1:
            fast_state = fast_state[None, :]
        if affect.ndim == 1:
            affect = affect[None, :]

        # 1. Calculate appetite changes based on personality + context
        # High extraversion → social_bond appetite accumulates faster
        # High boredom affect → novelty appetite spikes
        appetite_input = mx.concatenate([slow_state, fast_state], axis=1)
        appetite_change = self.appetite_dynamics(appetite_input)

        # 2. Accumulate appetites (drift toward baseline when unsatisfied)
        # If current < baseline: slowly increase
        # If current > baseline: naturally decay
        baseline_drift = (self.appetite_baselines - self.appetites) * 0.05

        self.appetites = mx.clip(
            self.appetites + (appetite_change * accumulation_rate) + baseline_drift,
            0.0,
            1.0
        )

        # 3. Generate goals from appetites + current state
        # High appetites + relevant opportunities = strong goals
        goal_input = mx.concatenate([
            self.appetites,
            affect,
            fast_state
        ], axis=1)

        goals = mx.sigmoid(self.goal_generator(goal_input))

        # 3b. Apply Brenda's narrative control (overrides and biases)
        if self.goal_overrides or self.goal_biases:
            goals_list = goals[0].tolist()

            # Apply biases first (add to natural generation)
            for goal_name, bias in self.goal_biases.items():
                if goal_name in self.goal_names:
                    idx = self.goal_names.index(goal_name)
                    goals_list[idx] = max(0.0, min(1.0, goals_list[idx] + bias))

            # Apply overrides (replace natural generation)
            for goal_name, strength in self.goal_overrides.items():
                if goal_name in self.goal_names:
                    idx = self.goal_names.index(goal_name)
                    goals_list[idx] = strength

            goals = mx.array([goals_list], dtype=mx.float32)

        # 4. Detect conflicts (which goals can't coexist)
        # Learned: "pursue_excitement" and "ensure_safety" conflict
        conflicts = mx.tanh(self.conflict_detector(goals))

        return goals, conflicts, self.appetites

    def stoke_appetite(self, appetite_name: str, amount: float):
        """
        Manually increase an appetite (Brenda's orchestration tool).

        Args:
            appetite_name: One of the 8 appetite names
            amount: How much to increase (0.0-1.0)

        Example:
            appetite_layer.stoke_appetite('novelty', 0.3)
            # Makes agent 30% more drawn to new experiences
        """
        if appetite_name not in self.appetite_names:
            raise ValueError(f"Unknown appetite: {appetite_name}. Must be one of {self.appetite_names}")

        idx = self.appetite_names.index(appetite_name)
        self.appetites = mx.array(self.appetites)  # Ensure mutable
        current = float(self.appetites[0, idx])
        new_value = min(1.0, current + amount)

        # Create new array with updated value
        appetites_list = self.appetites.tolist()[0]
        appetites_list[idx] = new_value
        self.appetites = mx.array([appetites_list], dtype=mx.float32)

    def sate_appetite(self, appetite_name: str, amount: float):
        """
        Satisfy/decrease an appetite (when goal is achieved).

        Args:
            appetite_name: One of the 8 appetite names
            amount: How much to decrease (0.0-1.0)

        Example:
            appetite_layer.sate_appetite('curiosity', 0.5)
            # Agent learned something, curiosity temporarily satisfied
        """
        if appetite_name not in self.appetite_names:
            raise ValueError(f"Unknown appetite: {appetite_name}. Must be one of {self.appetite_names}")

        idx = self.appetite_names.index(appetite_name)
        self.appetites = mx.array(self.appetites)  # Ensure mutable
        current = float(self.appetites[0, idx])
        new_value = max(0.0, current - amount)

        # Create new array with updated value
        appetites_list = self.appetites.tolist()[0]
        appetites_list[idx] = new_value
        self.appetites = mx.array([appetites_list], dtype=mx.float32)

    def override_goal(self, goal_name: str, strength: float):
        """
        Override a goal's activation (Brenda's narrative control).

        Completely replaces natural goal generation for this goal.
        Use this for strong narrative control (e.g., "Toad MUST obsess over motorcycles").

        Args:
            goal_name: One of the 16 goal names
            strength: Goal activation strength (0.0-1.0)

        Example:
            appetite_layer.override_goal('learn_skill', 0.95)
            appetite_layer.override_goal('demonstrate_competence', 0.90)
            # Toad now obsessed with mastering motorcycles!
        """
        if goal_name not in self.goal_names:
            raise ValueError(f"Unknown goal: {goal_name}. Must be one of {self.goal_names}")

        strength = max(0.0, min(1.0, strength))
        self.goal_overrides[goal_name] = strength

    def set_goal_bias(self, goal_name: str, bias: float):
        """
        Add a persistent bias to a goal's natural generation (Brenda's narrative control).

        Adds to natural goal activation rather than replacing it.
        Use this for subtle narrative influence (e.g., "Toad is slightly more cautious").

        Args:
            goal_name: One of the 16 goal names
            bias: Amount to add to goal activation (-1.0 to 1.0)

        Example:
            appetite_layer.set_goal_bias('ensure_safety', -0.3)
            appetite_layer.set_goal_bias('pursue_novelty', 0.2)
            # Toad becomes more reckless and novelty-seeking
        """
        if goal_name not in self.goal_names:
            raise ValueError(f"Unknown goal: {goal_name}. Must be one of {self.goal_names}")

        bias = max(-1.0, min(1.0, bias))
        self.goal_biases[goal_name] = bias

    def clear_goal_overrides(self, goal_name: Optional[str] = None):
        """
        Clear goal overrides (resume natural generation).

        Args:
            goal_name: Specific goal to clear, or None to clear all overrides

        Example:
            appetite_layer.clear_goal_overrides('learn_skill')  # Clear one
            appetite_layer.clear_goal_overrides()               # Clear all
        """
        if goal_name is None:
            self.goal_overrides.clear()
        elif goal_name in self.goal_overrides:
            del self.goal_overrides[goal_name]

    def clear_goal_biases(self, goal_name: Optional[str] = None):
        """
        Clear goal biases (resume natural generation).

        Args:
            goal_name: Specific goal to clear, or None to clear all biases

        Example:
            appetite_layer.clear_goal_biases('ensure_safety')  # Clear one
            appetite_layer.clear_goal_biases()                 # Clear all
        """
        if goal_name is None:
            self.goal_biases.clear()
        elif goal_name in self.goal_biases:
            del self.goal_biases[goal_name]

    def get_goal_overrides(self) -> Dict[str, float]:
        """
        Get current goal overrides.

        Returns:
            Dict mapping goal names to override strengths (0-1)
        """
        return dict(self.goal_overrides)

    def get_goal_biases(self) -> Dict[str, float]:
        """
        Get current goal biases.

        Returns:
            Dict mapping goal names to biases (-1 to 1)
        """
        return dict(self.goal_biases)

    def get_appetites(self) -> Dict[str, float]:
        """
        Get current appetite levels as dictionary.

        Returns:
            Dict mapping appetite names to values (0-1)
        """
        appetites_array = self.appetites[0].tolist()
        return {
            name: float(appetites_array[i])
            for i, name in enumerate(self.appetite_names)
        }

    def get_top_goals(self, goals: mx.array, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k most active goals.

        Args:
            goals: [batch, goal_dim] goal activation vector
            k: Number of top goals to return

        Returns:
            List of (goal_name, strength) tuples
        """
        if goals.ndim > 1:
            goals = goals[0]  # Extract batch dimension

        goals_list = goals.tolist()

        # Get indices of top-k goals
        indexed_goals = [(i, goals_list[i]) for i in range(len(goals_list))]
        indexed_goals.sort(key=lambda x: x[1], reverse=True)

        top_k = indexed_goals[:k]

        return [
            (self.goal_names[i], float(strength))
            for i, strength in top_k
        ]

    def reset_appetites(self):
        """Reset appetites to baseline values."""
        self.appetites = mx.array(self.appetite_baselines)


# Example usage and testing
if __name__ == '__main__':
    print("Testing AppetiteLayer...\n")

    # Initialize appetite layer
    appetite_layer = AppetiteLayer(
        appetite_dim=8,
        goal_dim=16,
        slow_dim=8,
        fast_dim=16,
        affect_dim=5
    )

    # Set Mr. Toad's appetite baselines (from recipe)
    toad_appetites = [
        0.7,   # curiosity
        0.8,   # status (VERY status-driven)
        0.6,   # mastery
        0.95,  # novelty (insatiable!)
        0.1,   # safety (reckless)
        0.5,   # social_bond
        0.2,   # comfort
        0.9    # autonomy
    ]

    appetite_layer.set_appetite_baselines(toad_appetites)

    print("Mr. Toad's Appetite Baselines:")
    for name, value in appetite_layer.get_appetites().items():
        print(f"  {name:12s}: {value:.2f}")

    print("\n" + "="*50)
    print("Simulating 10 timesteps...")
    print("="*50 + "\n")

    # Simulate Mr. Toad over time
    slow_state = mx.array([[0.9, 0.95, 0.7, 0.8, 0.9, 0.0, 0.0, 0.0]])  # Toad's personality

    for step in range(10):
        # Simulate varying affect (gets bored easily)
        boredom = min(1.0, 0.1 * step)
        affect = mx.array([[0.0, 0.3, 0.1, 0.0, boredom]])  # Increasing boredom

        # Random fast state (immediate context)
        fast_state = mx.random.normal((1, 16)) * 0.1

        # Forward pass
        goals, conflicts, appetites = appetite_layer.forward(
            slow_state, fast_state, affect, accumulation_rate=0.15
        )

        if step % 3 == 0:
            print(f"Step {step}:")
            print(f"  Boredom: {boredom:.2f}")
            print(f"  Appetites:")
            for name, value in appetite_layer.get_appetites().items():
                print(f"    {name:12s}: {value:.2f}")

            print(f"  Top goals:")
            for goal_name, strength in appetite_layer.get_top_goals(goals, k=3):
                print(f"    {goal_name:25s}: {strength:.3f}")
            print()

    print("="*50)
    print("Testing Brenda's orchestration tools...")
    print("="*50 + "\n")

    # Brenda stokes Toad's novelty appetite
    print("Brenda: @stoke toad novelty 0.05")
    appetite_layer.stoke_appetite('novelty', 0.05)
    print(f"  novelty: {appetite_layer.get_appetites()['novelty']:.2f}")

    # Generate goals with peaked novelty
    goals, conflicts, appetites = appetite_layer.forward(slow_state, fast_state, affect)
    print("\n  Top goals after stoking:")
    for goal_name, strength in appetite_layer.get_top_goals(goals, k=3):
        print(f"    {goal_name:25s}: {strength:.3f}")

    print("\n✓ AppetiteLayer test complete!")
