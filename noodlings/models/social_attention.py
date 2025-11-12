"""
Social Multi-Head Attention: Attending to Self and Others

Extends Phase 3 attention mechanism to include social awareness:
- Attend to past moments of SELF (as before)
- Attend to past moments of OTHERS (what Alice/Bob were feeling)
- Attend to RELATIONSHIP dynamics (how self and others interact)

Architecture:
- 6 attention heads (vs 4 in Phase 3):
  - 2 heads: Self temporal patterns
  - 2 heads: Other agents' patterns
  - 2 heads: Relationship dynamics

Parameters: ~20K (extended from Phase 3)

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class SocialKeyEncoder(nn.Module):
    """
    Encodes memory keys including social context.

    Input: Self state [40] + Primary other state [40] + Social context [16]
    Output: Key vector [96] for attention
    """

    def __init__(
        self,
        self_state_dim: int = 40,
        other_state_dim: int = 40,
        context_dim: int = 16,
        key_dim: int = 96
    ):
        super().__init__()

        input_dim = self_state_dim + other_state_dim + context_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, key_dim)
        )

    def __call__(
        self,
        self_state: mx.array,
        other_state: mx.array,
        social_context: mx.array
    ) -> mx.array:
        """
        Encode social memory key.

        Args:
            self_state: [batch, 40] self phenomenal state
            other_state: [batch, 40] primary other's state (or zeros if none)
            social_context: [batch, 16] social context features

        Returns:
            key: [batch, 96] memory key for attention
        """
        combined = mx.concatenate([self_state, other_state, social_context], axis=-1)
        return self.encoder(combined)


class SocialQueryGenerator(nn.Module):
    """
    Generates attention queries from current social state.

    Takes current self state + active other states and generates
    query for retrieving relevant past moments.
    """

    def __init__(
        self,
        self_state_dim: int = 40,
        other_state_dim: int = 40,
        query_dim: int = 96
    ):
        super().__init__()

        input_dim = self_state_dim + other_state_dim

        self.query_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, query_dim)
        )

    def __call__(
        self,
        self_state: mx.array,
        other_state: Optional[mx.array] = None
    ) -> mx.array:
        """
        Generate attention query.

        Args:
            self_state: [batch, 40] current self state
            other_state: [batch, 40] primary other's state (optional)

        Returns:
            query: [batch, 96] attention query
        """
        if other_state is None:
            other_state = mx.zeros_like(self_state)

        combined = mx.concatenate([self_state, other_state], axis=-1)
        return self.query_net(combined)


class SocialMultiHeadAttention(nn.Module):
    """
    Multi-head attention over social episodic memory.

    6 attention heads specialized for:
    - Heads 0-1: Self temporal patterns (similar to Phase 3)
    - Heads 2-3: Other agents' emotional patterns
    - Heads 4-5: Relationship dynamics between self and others

    Each head learns different aspects of social-emotional memory.
    """

    def __init__(
        self,
        query_dim: int = 96,
        num_heads: int = 6,
        dropout: float = 0.1
    ):
        """
        Initialize social multi-head attention.

        Args:
            query_dim: Dimension of queries, keys, values
            num_heads: Number of attention heads (default 6)
            dropout: Dropout rate
        """
        super().__init__()

        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        # Projection layers
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(query_dim, query_dim)
        self.v_proj = nn.Linear(query_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot-product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(
        self,
        query: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply social multi-head attention.

        Args:
            query: [batch, query_dim] current state query
            keys: [batch, num_memories, query_dim] memory keys
            values: [batch, num_memories, query_dim] memory values
            mask: Optional attention mask

        Returns:
            attended: [batch, query_dim] attended context
            attention_weights: [batch, num_heads, num_memories] attention weights per head
        """
        batch_size = query.shape[0]
        num_memories = keys.shape[1] if len(keys.shape) > 1 else 0

        if num_memories == 0:
            # No memories to attend to
            return mx.zeros((batch_size, self.query_dim)), mx.zeros((batch_size, self.num_heads, 1))

        # Ensure query has sequence dimension
        if len(query.shape) == 2:
            query = query[:, None, :]  # [batch, 1, query_dim]

        # Project Q, K, V
        Q = self.q_proj(query)  # [batch, 1, query_dim]
        K = self.k_proj(keys)   # [batch, num_memories, query_dim]
        V = self.v_proj(values)  # [batch, num_memories, query_dim]

        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, head_dim]
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, num_memories, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, num_memories, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention scores: Q @ K^T
        # [batch, num_heads, 1, head_dim] @ [batch, num_heads, head_dim, num_memories]
        # -> [batch, num_heads, 1, num_memories]
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attention_weights = mx.softmax(scores, axis=-1)  # [batch, num_heads, 1, num_memories]
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # [batch, num_heads, 1, num_memories] @ [batch, num_heads, num_memories, head_dim]
        # -> [batch, num_heads, 1, head_dim]
        attended = attention_weights @ V

        # Reshape back: [batch, num_heads, 1, head_dim] -> [batch, 1, query_dim]
        attended = attended.transpose(0, 2, 1, 3)
        attended = attended.reshape(batch_size, 1, self.query_dim)

        # Project output
        output = self.out_proj(attended)  # [batch, 1, query_dim]
        output = output.squeeze(1)  # [batch, query_dim]

        # Squeeze attention weights: [batch, num_heads, 1, num_memories] -> [batch, num_heads, num_memories]
        attention_weights = attention_weights.squeeze(2)

        return output, attention_weights

    def analyze_head_specialization(
        self,
        attention_weights: mx.array,
        memory_metadata: List[Dict]
    ) -> Dict[str, mx.array]:
        """
        Analyze what each attention head is attending to.

        Args:
            attention_weights: [batch, num_heads, num_memories] attention weights
            memory_metadata: List of memory dictionaries with social info

        Returns:
            Dictionary with head specialization metrics
        """
        num_heads = attention_weights.shape[1]
        num_memories = attention_weights.shape[2]

        # Categorize memories by type
        self_only = []
        with_others = []
        relationship_focused = []

        for i, mem in enumerate(memory_metadata):
            if not mem.get('agents') or len(mem['agents']) == 0:
                self_only.append(i)
            elif len(mem['agents']) >= 2:
                relationship_focused.append(i)
            else:
                with_others.append(i)

        # Calculate average attention per head to each category
        head_to_self = mx.zeros((num_heads,))
        head_to_others = mx.zeros((num_heads,))
        head_to_relationships = mx.zeros((num_heads,))

        for h in range(num_heads):
            if self_only:
                head_to_self = mx.array([
                    float(attention_weights[0, h, self_only].mean())
                    if h < num_heads else 0.0
                    for h in range(num_heads)
                ])

            if with_others:
                head_to_others = mx.array([
                    float(attention_weights[0, h, with_others].mean())
                    if h < num_heads else 0.0
                    for h in range(num_heads)
                ])

            if relationship_focused:
                head_to_relationships = mx.array([
                    float(attention_weights[0, h, relationship_focused].mean())
                    if h < num_heads else 0.0
                    for h in range(num_heads)
                ])

        return {
            'head_to_self': head_to_self,
            'head_to_others': head_to_others,
            'head_to_relationships': head_to_relationships,
            'self_indices': self_only,
            'others_indices': with_others,
            'relationship_indices': relationship_focused
        }


class SocialContextIntegrator(nn.Module):
    """
    Integrates current state with attended social context.

    Combines:
    - Current self state [40]
    - Attended context [96] (from social attention)
    - Current social context [16]

    Output: Enriched state for prediction [40]
    """

    def __init__(
        self,
        self_state_dim: int = 40,
        attended_dim: int = 96,
        social_context_dim: int = 16,
        output_dim: int = 40
    ):
        super().__init__()

        input_dim = self_state_dim + attended_dim + social_context_dim

        self.integrator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def __call__(
        self,
        self_state: mx.array,
        attended_context: mx.array,
        social_context: mx.array
    ) -> mx.array:
        """
        Integrate current state with social context.

        Args:
            self_state: [batch, 40] current self state
            attended_context: [batch, 96] attended memories
            social_context: [batch, 16] current social context

        Returns:
            enriched_state: [batch, 40] context-aware state
        """
        combined = mx.concatenate([self_state, attended_context, social_context], axis=-1)
        return self.integrator(combined)


def encode_social_context_vector(
    present_agents: List[str],
    group_valence: float = 0.0,
    group_arousal: float = 0.0,
    interaction_type: str = "conversation"
) -> mx.array:
    """
    Encode social context as a 16-D vector.

    This is a simple encoding scheme. More sophisticated versions could:
    - Use learned embeddings for agent identities
    - Include more nuanced interaction types
    - Add temporal dynamics

    Args:
        present_agents: List of agent names currently present
        group_valence: Overall group emotional valence [-1, 1]
        group_arousal: Overall group arousal [0, 1]
        interaction_type: Type of interaction

    Returns:
        [16] social context vector
    """
    # Simple hand-crafted features
    # Build vector directly instead of using deprecated .at[].set() API
    features = [0.0] * 16

    # Slots 0-1: Group affect
    features[0] = float(group_valence)
    features[1] = float(group_arousal)

    # Slot 2: Number of agents present (normalized)
    num_agents = min(len(present_agents), 10) / 10.0
    features[2] = float(num_agents)

    # Slots 3-6: Interaction type (one-hot)
    interaction_types = ["conversation", "conflict", "celebration", "support"]
    if interaction_type in interaction_types:
        idx = interaction_types.index(interaction_type)
        features[3 + idx] = 1.0

    # Slots 7-15: Reserved for future use / learned features

    return mx.array(features, dtype=mx.float32)
