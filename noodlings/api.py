"""
Noodlings Core API - Clean Phase 4 Implementation

This is the clean, production-ready API for Noodlings agents.
Uses base Phase 4 architecture WITHOUT observer loops.

Observers were removed after rigorous scientific testing showed
they provide zero benefit (November 2025).

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, Optional, List, Tuple
import datetime
import os
import sys

# Noodlings imports
from .models.noodling_phase4 import NoodlingModelPhase4
from .models.noodling_phase6 import NoodlingModelPhase6
from .memory.social_memory import SocialContext
from .memory.hierarchical_memory import HierarchicalMemory
from .memory.semantic_memory import SemanticMemorySystem


class NoodlingAgent:
    """
    Noodlings API supporting Phase 4 (social cognition) and Phase 6 (appetites).

    Provides hierarchical affective consciousness with:
    - Multi-timescale predictive processing (fast/medium/slow)
    - Theory of Mind & social cognition
    - Episodic & semantic memory
    - Surprise-driven behavior
    - Goal-directed behavior (Phase 6 with appetites)

    Phase 4: ~82.5K parameters
    Phase 6: ~97.5K parameters (adds appetite layer)

    Example:
        agent = NoodlingAgent(
            checkpoint_path="checkpoints/phase4.npz",
            config={'memory_capacity': 100}
        )

        result = agent.perceive(
            affect_vector=[0.5, 0.3, 0.1, 0.1, 0.1],
            agent_id="user_123",
            user_text="Hello there!"
        )

        print(f"Surprise: {result['surprise']:.3f}")
        print(f"State: {result['phenomenal_state']}")
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Noodlings agent.

        Args:
            checkpoint_path: Path to Phase 4/6 checkpoint (.npz file)
            config: Configuration dict:
                - memory_capacity: Episodic memory size (default 100)
                - surprise_threshold: Response trigger (default 0.3)
                - use_vae: Enable variational encoding (default False)
                - max_agents: Max agents to track (default 10)
                - num_attention_heads: Attention heads (default 4)
                - use_phase6: Enable Phase 6 appetite architecture (default False)
                - appetite_baselines: 8-element list for appetite initialization (Phase 6 only)
        """
        # Default configuration
        self.config = {
            'memory_capacity': 100,
            'surprise_threshold': 0.3,
            'use_vae': False,
            'max_agents': 10,
            'num_attention_heads': 4,
            'use_phase6': False,  # NEW: Enable appetite architecture
            'appetite_baselines': None  # NEW: Set from recipe
        }

        if config:
            self.config.update(config)

        self.use_phase6 = self.config['use_phase6']

        # Initialize Phase 4 or Phase 6 model
        if self.use_phase6:
            self.model = NoodlingModelPhase6(
                affect_dim=5,
                fast_hidden=16,
                medium_hidden=16,
                slow_hidden=8,
                predictor_hidden=64,
                use_vae=self.config['use_vae'],
                memory_capacity=self.config['memory_capacity'],
                num_attention_heads=self.config['num_attention_heads'],
                max_agents=self.config['max_agents'],
                use_theory_of_mind=True,
                use_relationship_model=True,
                use_appetite_layer=True,
                appetite_dim=8,
                goal_dim=16
            )

            # Set appetite baselines if provided
            if self.config['appetite_baselines']:
                appetite_baselines = self.config['appetite_baselines']
                # Convert dict to list in correct order if needed
                if isinstance(appetite_baselines, dict):
                    appetite_names = ['curiosity', 'status', 'mastery', 'novelty',
                                     'safety', 'social_bond', 'comfort', 'autonomy']
                    appetite_list = [appetite_baselines.get(name, 0.0) for name in appetite_names]
                    self.model.set_appetite_baselines(appetite_list)
                else:
                    self.model.set_appetite_baselines(appetite_baselines)
        else:
            self.model = NoodlingModelPhase4(
                affect_dim=5,
                fast_hidden=16,
                medium_hidden=16,
                slow_hidden=8,
                predictor_hidden=64,
                use_vae=self.config['use_vae'],
                memory_capacity=self.config['memory_capacity'],
                num_attention_heads=self.config['num_attention_heads'],
                max_agents=self.config['max_agents'],
                use_theory_of_mind=True,
                use_relationship_model=True
            )

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.model.load_weights(checkpoint_path)
                print(f"✓ Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print(f"   Starting with random weights")

        # Initialize states
        self.model.reset_states()

        # Surprise tracking
        self.surprise_buffer = []
        self.surprise_buffer_size = 50
        self.last_surprise = 0.0
        self.predicted_state = None
        self._step = 0

        # Hierarchical memory system
        self.memory = HierarchicalMemory(
            working_capacity=20,
            episodic_capacity=200,
            surprise_threshold=0.5,
            importance_decay=0.95
        )

        # Semantic memory system
        self.semantic_memory = SemanticMemorySystem(
            max_users=self.config.get('max_agents', 10),
            consolidation_interval=50
        )

        # Conversation history
        self.conversation_history = []

        # Parameter counts
        try:
            param_counts = self.model.count_parameters()
            print(f"\nNoodlings Phase 4 initialized:")
            print(f"  Parameters: ~{param_counts.get('total', 82500):,}")
            print(f"  Memory capacity: {self.config['memory_capacity']}")
            print(f"  Surprise threshold: {self.config['surprise_threshold']}")
        except AttributeError:
            print(f"\nNoodlings Phase 4 initialized (parameter counting unavailable)")

    def perceive(
        self,
        affect_vector: List[float],
        agent_id: Optional[str] = None,
        user_text: str = "",
        present_agents: Optional[List[str]] = None
    ) -> Dict:
        """
        Process affective input through hierarchical architecture.

        Args:
            affect_vector: [valence, arousal, fear, sorrow, boredom]
            agent_id: ID of agent producing this affect
            user_text: Original user text
            present_agents: List of agent IDs currently present

        Returns:
            Dictionary containing:
                - phenomenal_state: 40-D internal state (fast+medium+slow)
                - surprise: Prediction error magnitude
                - fast_state: 16-D fast layer state
                - medium_state: 16-D medium layer state
                - slow_state: 8-D slow layer state
                - should_respond: Whether surprise exceeds threshold
                - affect_input: Input affect vector
        """
        self._step += 1

        # Convert affect to MLX array
        affect_mx = mx.array(affect_vector, dtype=mx.float32)

        if affect_mx.ndim == 1:
            affect_mx = affect_mx[None, :]  # Add batch dimension

        # Forward pass (Phase 4 or Phase 6)
        if self.use_phase6:
            self_state, predicted_state, social_info = self.model.forward_with_appetites(
                affect=affect_mx,
                present_agents=present_agents,
                user_text=user_text
            )
        else:
            self_state, predicted_state, social_info = self.model.forward_with_social_context(
                affect=affect_mx,
                present_agents=present_agents,
                user_text=user_text
            )

        # Extract phenomenal state (current self state)
        phenomenal_state = self_state
        if phenomenal_state.ndim > 1:
            phenomenal_state_np = np.array(phenomenal_state.squeeze(0))
        else:
            phenomenal_state_np = np.array(phenomenal_state)

        # Calculate surprise (prediction error)
        if self.predicted_state is not None:
            surprise = float(np.linalg.norm(phenomenal_state_np - self.predicted_state))
        else:
            surprise = 0.0

        self.last_surprise = surprise

        # Update surprise buffer
        self.surprise_buffer.append(surprise)
        if len(self.surprise_buffer) > self.surprise_buffer_size:
            self.surprise_buffer.pop(0)

        # Store current state as prediction for next step
        self.predicted_state = phenomenal_state_np.copy()

        # Determine if agent should respond
        surprise_mean = np.mean(self.surprise_buffer) if self.surprise_buffer else 0.0
        surprise_std = np.std(self.surprise_buffer) if len(self.surprise_buffer) > 1 else 1.0
        threshold = surprise_mean + self.config['surprise_threshold'] * surprise_std
        should_respond = surprise > threshold

        # Store in conversation history
        self.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'user_text': user_text,
            'affect': affect_vector,
            'surprise': surprise,
            'phenomenal_state': phenomenal_state_np.tolist(),
            'agent_id': agent_id
        })

        # Return results
        result = {
            'phenomenal_state': phenomenal_state_np,
            'surprise': surprise,
            'should_respond': should_respond,
            'fast_state': np.array(self.model.h_fast.squeeze()) if self.model.h_fast is not None else None,
            'medium_state': np.array(self.model.h_medium.squeeze()) if self.model.h_medium is not None else None,
            'slow_state': np.array(self.model.h_slow.squeeze()) if self.model.h_slow is not None else None,
            'affect_input': affect_vector,
            'step': self._step,
            'social_info': social_info  # Theory of Mind outputs
        }

        return result

    def reset_states(self):
        """Reset all internal states to zero."""
        self.model.reset_states()
        self.surprise_buffer = []
        self.predicted_state = None

    def get_states(self) -> Dict:
        """Get current internal states."""
        return {
            'fast': np.array(self.model.h_fast.squeeze()) if self.model.h_fast is not None else None,
            'medium': np.array(self.model.h_medium.squeeze()) if self.model.h_medium is not None else None,
            'slow': np.array(self.model.h_slow.squeeze()) if self.model.h_slow is not None else None,
            'surprise_buffer': self.surprise_buffer.copy(),
            'step': self._step
        }

    def get_state(self) -> Dict:
        """Alias for get_states() for backward compatibility."""
        states = self.get_states()
        states['surprise_threshold'] = self.config['surprise_threshold']
        states['surprise'] = self.last_surprise
        return states

    def save_checkpoint(self, path: str):
        """Save model weights to file."""
        self.model.save_weights(path)
        print(f"✓ Saved checkpoint: {path}")

    def get_conversation_history(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get recent conversation history.

        Args:
            n: Number of recent entries to return (None = all)

        Returns:
            List of conversation entries
        """
        if n is None:
            return self.conversation_history.copy()
        else:
            return self.conversation_history[-n:].copy()

    # Phase 6: Appetite Architecture Methods

    def stoke_appetite(self, appetite_name: str, amount: float):
        """
        Increase an appetite (Brenda's orchestration tool).

        Phase 6 only. Raises RuntimeError if Phase 6 not enabled.

        Args:
            appetite_name: One of 8 appetites (curiosity, status, mastery, novelty,
                          safety, social_bond, comfort, autonomy)
            amount: How much to increase (0.0-1.0)

        Example:
            agent.stoke_appetite('novelty', 0.3)
            # Makes agent 30% more drawn to new experiences
        """
        if not self.use_phase6:
            raise RuntimeError(
                "Phase 6 appetite architecture not enabled. "
                "Initialize with config={'use_phase6': True}"
            )

        self.model.stoke_appetite(appetite_name, amount)

    def sate_appetite(self, appetite_name: str, amount: float):
        """
        Satisfy/decrease an appetite (when goal is achieved).

        Phase 6 only. Raises RuntimeError if Phase 6 not enabled.

        Args:
            appetite_name: One of 8 appetites
            amount: How much to decrease (0.0-1.0)

        Example:
            agent.sate_appetite('curiosity', 0.5)
            # Agent learned something, curiosity temporarily satisfied
        """
        if not self.use_phase6:
            raise RuntimeError(
                "Phase 6 appetite architecture not enabled. "
                "Initialize with config={'use_phase6': True}"
            )

        self.model.sate_appetite(appetite_name, amount)

    def get_appetites(self) -> Dict[str, float]:
        """
        Get current appetite levels.

        Phase 6 only. Returns empty dict if Phase 6 not enabled.

        Returns:
            Dict mapping appetite names to values (0-1)

        Example:
            appetites = agent.get_appetites()
            print(f"Novelty: {appetites['novelty']:.2f}")
        """
        if not self.use_phase6:
            return {}

        return self.model.get_appetites()

    def get_goals(self) -> List[Tuple[str, float]]:
        """
        Get top-k most active goals.

        Phase 6 only. Returns empty list if Phase 6 not enabled.

        Returns:
            List of (goal_name, strength) tuples

        Example:
            goals = agent.get_goals()
            for goal_name, strength in goals:
                print(f"{goal_name}: {strength:.3f}")
        """
        if not self.use_phase6:
            return []

        # Goals are returned in the last perceive() call's social_info
        # For now, return empty - goals are available via perceive() result
        return []

    def get_relationships(self) -> Dict:
        """
        Get relationship information from social memory.

        Returns:
            Dict with relationship data (empty dict if no data available)
        """
        try:
            # Try Phase 6 first
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'social_memory'):
                social_memory = self.model.base_model.social_memory
            # Try Phase 4
            elif hasattr(self.model, 'social_memory'):
                social_memory = self.model.social_memory
            else:
                return {}

            relationships = {}

            # Get all tracked agents
            if hasattr(social_memory, 'agents'):
                for agent_name, agent_state in social_memory.agents.items():
                    relationships[agent_name] = {
                        'name': agent_name,
                        'last_seen': getattr(agent_state, 'last_mentioned', 0),
                        'is_present': getattr(agent_state, 'is_present', False)
                    }

            return relationships
        except Exception:
            # Silently return empty dict if any error
            return {}

    def set_goal_bias(self, goal_name: str, bias: float):
        """
        Add a persistent bias to a specific goal's activation.

        Phase 6 only. Raises RuntimeError if Phase 6 not enabled.

        Args:
            goal_name: One of the 16 goal names from AppetiteLayer
            bias: Bias amount (-1.0 to 1.0)

        Example:
            agent.set_goal_bias("seek_social_connection", 0.3)
        """
        if not self.use_phase6:
            raise RuntimeError("set_goal_bias requires Phase 6 (appetites)")

        if not hasattr(self.model, 'appetite_layer') or self.model.appetite_layer is None:
            raise RuntimeError("Appetite layer not initialized")

        self.model.appetite_layer.set_goal_bias(goal_name, bias)

    def set_goal_override(self, goal_name: str, strength: float):
        """
        Override a goal's natural activation with a fixed value.

        Phase 6 only. Raises RuntimeError if Phase 6 not enabled.

        Args:
            goal_name: One of the 16 goal names from AppetiteLayer
            strength: Fixed goal strength (0.0 to 1.0)

        Example:
            agent.set_goal_override("explore_environment", 0.8)
        """
        if not self.use_phase6:
            raise RuntimeError("set_goal_override requires Phase 6 (appetites)")

        if not hasattr(self.model, 'appetite_layer') or self.model.appetite_layer is None:
            raise RuntimeError("Appetite layer not initialized")

        self.model.appetite_layer.set_goal_override(goal_name, strength)

    def clear_goal_overrides(self, goal_name: Optional[str] = None):
        """
        Clear goal overrides (all or specific goal).

        Phase 6 only. Does nothing if Phase 6 not enabled.

        Args:
            goal_name: Specific goal to clear, or None to clear all

        Example:
            agent.clear_goal_overrides()  # Clear all
            agent.clear_goal_overrides("explore_environment")  # Clear one
        """
        if not self.use_phase6:
            return

        if not hasattr(self.model, 'appetite_layer') or self.model.appetite_layer is None:
            return

        self.model.appetite_layer.clear_goal_overrides(goal_name)


# Alias for backward compatibility
ConsilienceAgent = NoodlingAgent
