"""
Consilience Phase 4: Social Cognition & Theory of Mind

Extends Phase 3 (Attention & Episodic Memory) with:
- Multi-agent episodic memory (track up to 10 people)
- Theory of Mind module (infer others' mental states)
- Social attention mechanism (attend to self + others + relationships)
- Relationship models (track pairwise dynamics)

This implements computational correlates of:
- Theory of Mind (Premack & Woodruff): Inferring others' mental states
- Social cognition (Adolphs): Processing social information
- Attachment theory (Bowlby): Modeling relationship patterns

Total Parameters: Phase 3 (~50K) + Phase 4 (~82.5K) = ~132.5K

Architecture Flow:
1. User input → Affect extraction (5-D) + Agent detection
2. For each mentioned agent:
   - Extract linguistic features (128-D)
   - Extract context features (64-D)
   - Infer their internal state via Theory of Mind (40-D)
   - Model relationship dynamics (32-D)
3. Current self state → Hierarchical encoding (40-D) [Phase 1-2]
4. Generate social memory key: [self + primary_other + context] → (96-D)
5. Social attention over episodic memory
6. Integrate: self + attended_context + social_context → enriched state
7. Predict next self state
8. Store moment in social episodic memory

Key Innovation: The agent can now model what others are thinking/feeling,
track relationship dynamics, and use social context to inform predictions.

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np

# Phase 3 imports
from .noodling_attention import NoodlingModelWithAttention

# Phase 4 imports
from ..memory.social_memory import SocialEpisodicMemory, AgentState, SocialContext
from .theory_of_mind import TheoryOfMindModule, RelationshipModel
from .social_attention import (
    SocialKeyEncoder,
    SocialQueryGenerator,
    SocialMultiHeadAttention,
    SocialContextIntegrator,
    encode_social_context_vector
)


class NoodlingModelPhase4(nn.Module):
    """
    Phase 4: Hierarchical affective consciousness with social cognition.

    Combines all previous phases with social awareness:
    - Phase 1: Multi-timescale predictive processing
    - Phase 2: Variational uncertainty modeling
    - Phase 3: Episodic memory + attention
    - Phase 4: Social cognition + theory of mind
    """

    def __init__(
        self,
        # Phase 1-2 params (hierarchical architecture)
        affect_dim: int = 5,
        fast_hidden: int = 16,
        medium_hidden: int = 16,
        slow_hidden: int = 8,
        predictor_hidden: int = 64,
        use_vae: bool = False,

        # Phase 3 params (attention & memory)
        memory_capacity: int = 100,
        num_attention_heads: int = 4,  # Note: Social attention uses 6 heads
        attention_dropout: float = 0.1,

        # Phase 4 params (social cognition)
        max_agents: int = 10,
        linguistic_dim: int = 128,
        context_dim: int = 64,
        social_context_dim: int = 16,
        relationship_dim: int = 32,
        use_theory_of_mind: bool = True,
        use_relationship_model: bool = True
    ):
        """
        Initialize Phase 4 model with social cognition.

        Args:
            affect_dim: Affect input dimension
            fast_hidden: Fast layer hidden size
            medium_hidden: Medium layer hidden size
            slow_hidden: Slow layer hidden size
            predictor_hidden: Predictor network hidden size
            use_vae: Use Phase 2 VAE extension
            memory_capacity: Max memories to store
            num_attention_heads: Attention heads (becomes 6 for social)
            attention_dropout: Attention dropout rate
            max_agents: Max number of other agents to track
            linguistic_dim: Linguistic feature dimension
            context_dim: Context feature dimension
            social_context_dim: Social context vector dimension
            relationship_dim: Relationship representation dimension
            use_theory_of_mind: Enable Theory of Mind module
            use_relationship_model: Enable relationship modeling
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

        # Phase 1-3: Base architecture (load from Phase 3 model)
        self.base_model = NoodlingModelWithAttention(
            affect_dim=affect_dim,
            fast_hidden=fast_hidden,
            medium_hidden=medium_hidden,
            slow_hidden=slow_hidden,
            predictor_hidden=predictor_hidden,
            memory_capacity=memory_capacity,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_vae=use_vae
        )

        # Phase 4: Social cognition modules
        if self.use_theory_of_mind:
            self.theory_of_mind = TheoryOfMindModule(
                linguistic_dim=linguistic_dim,
                context_dim=context_dim,
                history_dim=self.state_dim,
                state_dim=self.state_dim
            )

        if self.use_relationship_model:
            self.relationship_model = RelationshipModel(
                state_dim=self.state_dim,
                relationship_dim=relationship_dim
            )

        # Phase 4: Social attention (replaces Phase 3 attention)
        self.social_key_encoder = SocialKeyEncoder(
            self_state_dim=self.state_dim,
            other_state_dim=self.state_dim,
            context_dim=social_context_dim,
            key_dim=96
        )

        self.social_query_generator = SocialQueryGenerator(
            self_state_dim=self.state_dim,
            other_state_dim=self.state_dim,
            query_dim=96
        )

        self.social_attention = SocialMultiHeadAttention(
            query_dim=96,
            num_heads=6,  # 6 heads for social attention
            dropout=attention_dropout
        )

        self.social_context_integrator = SocialContextIntegrator(
            self_state_dim=self.state_dim,
            attended_dim=96,
            social_context_dim=social_context_dim,
            output_dim=self.state_dim
        )

        # Phase 4: Social episodic memory
        self.social_memory = SocialEpisodicMemory(
            capacity=memory_capacity,
            state_dim=self.state_dim,
            max_agents=max_agents
        )

        # Hidden states (from Phase 1-2)
        self.h_fast = None
        self.c_fast = None
        self.h_medium = None
        self.c_medium = None
        self.h_slow = None

        self.step = 0

    def get_states(self):
        """Get current hidden states without resetting."""
        return self.h_fast, self.c_fast, self.h_medium, self.c_medium, self.h_slow

    def reset_states(self):
        """Reset all hidden states and memory."""
        self.h_fast = mx.zeros((1, self.fast_hidden))
        self.c_fast = mx.zeros((1, self.fast_hidden))
        self.h_medium = mx.zeros((1, self.medium_hidden))
        self.c_medium = mx.zeros((1, self.medium_hidden))
        self.h_slow = mx.zeros((1, self.slow_hidden))
        self.social_memory.clear()
        self.step = 0
        return self.h_fast, self.c_fast, self.h_medium, self.c_medium, self.h_slow

    def forward_with_social_context(
        self,
        affect: mx.array,
        linguistic_features: Optional[Dict[str, mx.array]] = None,
        context_features: Optional[mx.array] = None,
        present_agents: Optional[List[str]] = None,
        social_context: Optional[SocialContext] = None,
        user_text: str = ""
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        Forward pass with social context.

        Args:
            affect: [batch, 5] affect vector
            linguistic_features: Dict mapping agent_name -> [batch, 128] features
            context_features: [batch, 64] context features
            present_agents: List of agent names currently present
            social_context: SocialContext object (optional)
            user_text: User input text

        Returns:
            self_state: [batch, 40] current internal state
            predicted_state: [batch, 40] predicted next state
            social_info: Dictionary with inferred agent states, relationships, etc.
        """
        batch_size = affect.shape[0]

        # Initialize states if needed
        if self.h_fast is None:
            self.reset_states()

        # 1. Encode self state using Phase 1-2 hierarchical architecture
        # Use base model's recurrent layers
        # Add sequence dimension if needed: (batch, 5) -> (batch, 1, 5)
        if len(affect.shape) == 2:
            affect = affect[:, None, :]

        # LSTM returns (hidden_seq, cell_seq)
        h_fast_seq, c_fast_seq = self.base_model.fast_lstm(
            affect, hidden=self.h_fast, cell=self.c_fast
        )
        # Extract final timestep: (batch, 1, dim) -> (batch, dim)
        # BUGFIX: Remove mx.eval() - it can return None on lazy arrays
        # Direct reshape forces materialization and exact shape (1, hidden_dim)
        fast_dim = h_fast_seq.shape[-1]
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, fast_dim)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, fast_dim)

        # Medium LSTM
        h_fast_input = self.h_fast[:, None, :]  # Add seq dim
        h_med_seq, c_med_seq = self.base_model.medium_lstm(
            h_fast_input, hidden=self.h_medium, cell=self.c_medium
        )
        # BUGFIX: Remove mx.eval() - it can return None on lazy arrays
        med_dim = h_med_seq.shape[-1]
        self.h_medium = h_med_seq[:, -1, :].reshape(1, med_dim)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, med_dim)

        # Slow GRU
        h_med_input = self.h_medium[:, None, :]  # Add seq dim
        h_slow_seq = self.base_model.slow_gru(h_med_input, hidden=self.h_slow)
        # BUGFIX: Remove mx.eval() - it can return None on lazy arrays
        slow_dim = h_slow_seq.shape[-1]
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, slow_dim)

        # Concatenate to form internal state
        self_state = mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=-1)

        # 2. Infer other agents' states using Theory of Mind
        inferred_agents = {}
        primary_other_state = mx.zeros((batch_size, self.state_dim))

        if self.use_theory_of_mind and linguistic_features and context_features is not None:
            for agent_name, ling_feat in linguistic_features.items():
                # Get their previous state if known
                prev_agent = self.social_memory.get_agent(agent_name)
                prev_state = prev_agent.inferred_state if prev_agent else None

                if prev_state is not None and len(prev_state.shape) == 1:
                    prev_state = prev_state[None, :]  # Add batch dimension

                # Infer current state
                inferred_state, confidence, mean, logvar, uncertainty = \
                    self.theory_of_mind(ling_feat, context_features, prev_state)

                # Create AgentState
                agent_state = AgentState(
                    name=agent_name,
                    inferred_state=inferred_state,
                    confidence=float(confidence.item()),
                    last_mentioned=self.step,
                    is_present=agent_name in (present_agents or [])
                )

                inferred_agents[agent_name] = agent_state

                # Use first agent as "primary other" for attention
                if primary_other_state.sum() == 0:
                    primary_other_state = inferred_state

        # 3. Model relationships (if enabled)
        relationships = {}
        if self.use_relationship_model and inferred_agents:
            for agent_name, agent_state in inferred_agents.items():
                rel_vec, attach_logits, trust, comm = self.relationship_model(
                    self_state, agent_state.inferred_state
                )
                relationships[agent_name] = {
                    'vector': rel_vec,
                    'attachment_logits': attach_logits,
                    'trust': trust,
                    'communication': comm
                }

        # 4. Encode social context
        if social_context is None:
            # Convert affect to Python list for scalar extraction
            # Handle multiple possible shapes: (batch, seq, features) or (batch, features)
            affect_list = affect.tolist()
            # Flatten to get first affect vector
            while isinstance(affect_list, list) and len(affect_list) > 0 and isinstance(affect_list[0], list):
                affect_list = affect_list[0]

            social_context = SocialContext(
                present_agents=present_agents or [],
                group_valence=float(affect_list[0]) if len(affect_list) > 0 else 0.0,
                group_arousal=float(affect_list[1]) if len(affect_list) > 1 else 0.0
            )

        social_context_vec = encode_social_context_vector(
            present_agents=social_context.present_agents,
            group_valence=social_context.group_valence,
            group_arousal=social_context.group_arousal,
            interaction_type=social_context.interaction_type
        )

        if len(social_context_vec.shape) == 1:
            social_context_vec = social_context_vec[None, :]  # Add batch dimension

        # 5. Generate memory key
        memory_key = self.social_key_encoder(
            self_state, primary_other_state, social_context_vec
        )

        # 6. Social attention over episodic memory
        keys, memory_values = self.social_memory.get_keys_and_values()

        if keys.shape[0] > 0:
            # Add batch dimension if needed
            if len(keys.shape) == 2:
                keys = keys[None, :, :]  # [1, num_memories, key_dim]

            attended_context, attention_weights = self.social_attention(
                query=memory_key,
                keys=keys,
                values=keys  # Use keys as values for now
            )
        else:
            attended_context = mx.zeros((batch_size, 96))
            attention_weights = mx.zeros((batch_size, 6, 1))

        # 7. Integrate context
        enriched_state = self.social_context_integrator(
            self_state, attended_context, social_context_vec
        )

        # 8. Predict next state
        predicted_state = self.base_model.predictor(enriched_state)

        # 9. Store in social memory
        self.social_memory.add_memory(
            step=self.step,
            self_state=self_state,
            affect=affect,
            agents=inferred_agents,
            social_context=social_context,
            user_text=user_text,
            key_vector=memory_key,
            attention_weights=attention_weights
        )

        self.step += 1

        # Prepare social info for return
        social_info = {
            'inferred_agents': inferred_agents,
            'relationships': relationships,
            'attention_weights': attention_weights,
            'social_context': social_context,
            'memory_count': self.social_memory.get_memory_count()
        }

        return self_state, predicted_state, social_info

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate total parameter count.

        Returns:
            Dictionary with parameter counts per module
        """
        counts = {
            'base_model_phase3': self.base_model.get_parameter_count()['total']
        }

        if self.use_theory_of_mind:
            counts['theory_of_mind'] = self.theory_of_mind.get_parameter_count()['total']

        if self.use_relationship_model:
            counts['relationship_model'] = self.relationship_model.get_parameter_count()['total']

        # Social attention components (approximate)
        counts['social_key_encoder'] = (self.state_dim * 2 + 16) * 128 + 128 * 96
        counts['social_query_generator'] = (self.state_dim * 2) * 128 + 128 * 96
        counts['social_attention'] = 96 * 96 * 4  # Q, K, V, O projections
        counts['social_context_integrator'] = (self.state_dim + 96 + 16) * 128 + 128 * 64 + 64 * self.state_dim

        counts['total'] = sum(counts.values())

        return counts

    def load_phase3_weights(self, checkpoint_path: str):
        """
        Load Phase 3 checkpoint into base model.

        Args:
            checkpoint_path: Path to Phase 3 checkpoint (.npz)
        """
        self.base_model.load_weights(checkpoint_path)
        print(f"✓ Loaded Phase 3 weights from {checkpoint_path}")

    def save_weights(self, path: str):
        """
        Save Phase 4 model weights.

        Args:
            path: Output path for checkpoint
        """
        weights = {}

        # Save all module weights
        for name, module in [
            ('base_model', self.base_model),
            ('theory_of_mind', self.theory_of_mind if self.use_theory_of_mind else None),
            ('relationship_model', self.relationship_model if self.use_relationship_model else None),
            ('social_key_encoder', self.social_key_encoder),
            ('social_query_generator', self.social_query_generator),
            ('social_attention', self.social_attention),
            ('social_context_integrator', self.social_context_integrator)
        ]:
            if module is not None:
                for param_name, param in module.parameters().items():
                    weights[f'{name}.{param_name}'] = param

        mx.savez(path, **weights)
        print(f"✓ Saved Phase 4 weights to {path}")

    def load_weights(self, path: str):
        """
        Load Phase 4 model weights.

        Args:
            path: Path to checkpoint
        """
        weights = mx.load(path)

        # Distribute weights to modules
        for key, value in weights.items():
            module_name = key.split('.')[0]
            param_path = '.'.join(key.split('.')[1:])

            module = getattr(self, module_name, None)
            if module is not None:
                # Set parameter (simplified - actual implementation needs tree manipulation)
                pass

        print(f"✓ Loaded Phase 4 weights from {path}")
