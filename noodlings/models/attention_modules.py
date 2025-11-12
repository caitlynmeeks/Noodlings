"""
Attention Mechanisms for Consilience Phase 3

Implements multi-head attention over episodic memory for working memory retrieval.

Components:
1. KeyEncoder: Maps 40-D phenomenal states to 64-D retrieval keys
2. AttentionQueryGenerator: Generates attention queries from current state
3. MultiHeadAttention: 4-head attention mechanism with specialization
4. ContextIntegrator: Fuses current state with attended memories

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


class KeyEncoder(nn.Module):
    """
    Transforms phenomenal states into retrieval keys.

    The key encoder learns "what is memorable about this moment?"
    Maps 40-D phenomenal state → 64-D key vector for efficient retrieval.

    Architecture:
        40 → 64 (Tanh) → 64 (Linear)

    Parameters: 40×64 + 64×64 = 6,656
    """

    def __init__(self, state_dim: int = 40, key_dim: int = 64):
        """
        Initialize key encoder.

        Args:
            state_dim: Dimensionality of phenomenal state (default 40)
            key_dim: Dimensionality of retrieval keys (default 64)
        """
        super().__init__()
        self.state_dim = state_dim
        self.key_dim = key_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),  # Non-linear transformation
            nn.Linear(64, key_dim)
        )

    def __call__(self, phenomenal_state: mx.array) -> mx.array:
        """
        Encode phenomenal state as retrieval key.

        Args:
            phenomenal_state: [40-D] joint hierarchical state

        Returns:
            key: [64-D] retrieval key vector
        """
        return self.encoder(phenomenal_state)


class AttentionQueryGenerator(nn.Module):
    """
    Generates attention query from current phenomenal state.

    The query represents: "What past moments are relevant NOW?"
    Asymmetric architecture (larger than key encoder) allows richer query representation.

    Architecture:
        40 → 128 (ReLU + Dropout) → 64 (Linear)

    Parameters: 40×128 + 128×64 = 13,312
    """

    def __init__(self, state_dim: int = 40, query_dim: int = 64, dropout: float = 0.1):
        """
        Initialize query generator.

        Args:
            state_dim: Dimensionality of phenomenal state (default 40)
            query_dim: Dimensionality of attention queries (default 64)
            dropout: Dropout rate for regularization (default 0.1)
        """
        super().__init__()
        self.state_dim = state_dim
        self.query_dim = query_dim

        self.query_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, query_dim)
        )

    def __call__(self, phenomenal_state: mx.array) -> mx.array:
        """
        Generate attention query from current state.

        Args:
            phenomenal_state: [40-D] current hierarchical state

        Returns:
            query: [64-D] attention query vector
        """
        return self.query_net(phenomenal_state)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention over episodic memory.

    4 attention heads can specialize for different retrieval patterns:
    - Head 1: Temporal proximity (recent moments)
    - Head 2: Emotional similarity (affective resonance)
    - Head 3: Salience (high surprise moments)
    - Head 4: Semantic/thematic similarity (distributed patterns)

    Architecture follows Vaswani et al. (2017) with modifications for memory retrieval.

    Parameters:
    - Q projection: 64×64 = 4,096
    - K projection: 64×64 = 4,096
    - V projection: 40×64 = 2,560
    - Output projection: 64×40 = 2,560
    Total: 13,312 parameters
    """

    def __init__(self,
                 query_dim: int = 64,
                 key_dim: int = 64,
                 value_dim: int = 40,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize multi-head attention mechanism.

        Args:
            query_dim: Dimensionality of queries (default 64)
            key_dim: Dimensionality of keys (default 64)
            value_dim: Dimensionality of values (default 40 = phenomenal state)
            num_heads: Number of attention heads (default 4)
            dropout: Dropout rate for attention weights (default 0.1)
        """
        super().__init__()

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert key_dim == query_dim, "key_dim must equal query_dim for this implementation"

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads  # 16 per head
        self.scale = float(self.head_dim ** 0.5)  # Scaling factor for dot product

        # Linear projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(value_dim, query_dim)  # Project values to query space
        self.out_proj = nn.Linear(query_dim, value_dim)  # Project back to phenomenal state space

        self.dropout = nn.Dropout(dropout)

    def __call__(self,
                 query: mx.array,
                 keys: mx.array,
                 values: mx.array,
                 mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """
        Compute multi-head attention over episodic memory.

        Args:
            query: [64-D] current moment attention query
            keys: [N, 64] memory retrieval keys
            values: [N, 40] memory phenomenal states
            mask: [N] optional mask (0=attend, -inf=ignore)

        Returns:
            context: [40-D] attended memory context
            attention_weights: [N] attention distribution (averaged across heads)
        """
        N = keys.shape[0]  # Number of memories

        if N == 0:
            # No memories to attend to yet
            return mx.zeros(self.value_dim), mx.zeros(0)

        # Ensure query is 1-D
        if len(query.shape) > 1:
            query = query.squeeze()

        # Project and reshape for multi-head attention
        # Q: [64] → [1, 4, 16]
        Q = self.q_proj(query).reshape(1, self.num_heads, self.head_dim)

        # K: [N, 64] → [N, 4, 16]
        K = self.k_proj(keys).reshape(N, self.num_heads, self.head_dim)

        # V: [N, 40] → [N, 4, 16]
        V = self.v_proj(values).reshape(N, self.num_heads, self.head_dim)

        # Scaled dot-product attention per head
        # Q: [1, 4, 16], K: [N, 4, 16]
        # Transpose K: [N, 4, 16] → [N, 16, 4] (swap last two dims)
        K_t = mx.transpose(K, axes=[0, 2, 1])

        # Compute scores: [1, 4, 16] × [N, 16, 4] → [1, 4, N]
        # Broadcasting handles batch dimension
        scores = mx.zeros((1, self.num_heads, N))
        for head in range(self.num_heads):
            q_head = Q[0, head, :]  # [16]
            k_head = K[:, head, :]  # [N, 16]
            # Dot product: [16] · [N, 16] → [N]
            head_scores = mx.sum(q_head * k_head, axis=-1)
            scores = mx.array([[[head_scores[i] if h == head else scores[0, h, i]
                                for i in range(N)]
                               for h in range(self.num_heads)]])

        # Scale by sqrt(head_dim)
        scores = scores / self.scale

        # Apply mask if provided
        if mask is not None:
            # Broadcast mask: [N] → [1, num_heads, N]
            mask_expanded = mask[None, None, :]
            scores = scores + mask_expanded

        # Softmax over memory dimension
        attn = mx.softmax(scores, axis=-1)  # [1, num_heads, N]
        attn = self.dropout(attn)

        # Weighted sum of values: [1, num_heads, N] × [N, num_heads, head_dim]
        context = mx.zeros((1, self.num_heads, self.head_dim))
        for head in range(self.num_heads):
            attn_head = attn[0, head, :]  # [N]
            v_head = V[:, head, :]  # [N, 16]
            # Weighted sum: [N] · [N, 16] → [16]
            context_head = mx.sum(attn_head[:, None] * v_head, axis=0)
            # Update context
            new_context = mx.zeros((1, self.num_heads, self.head_dim))
            for h in range(self.num_heads):
                if h == head:
                    new_context = mx.array([[[context_head[d] if h_idx == head else context[0, h_idx, d]
                                             for d in range(self.head_dim)]
                                            for h_idx in range(self.num_heads)]])
            context = new_context

        # Concatenate heads: [1, 4, 16] → [1, 64]
        context = context.reshape(1, self.query_dim)

        # Project back to phenomenal state dimension: [1, 64] → [1, 40]
        context = self.out_proj(context)

        # Remove batch dimension: [1, 40] → [40]
        context = context.squeeze(0)

        # Average attention weights across heads for interpretability: [1, 4, N] → [N]
        attn_weights = mx.mean(attn, axis=(0, 1))

        return context, attn_weights


class ContextIntegrator(nn.Module):
    """
    Integrates current phenomenal state with attended memories.

    Fuses two sources of information:
    1. What I'm experiencing NOW (current state)
    2. What I've experienced BEFORE in similar situations (attended context)

    Creates enriched representation for prediction.

    Architecture:
        80 → 64 (ReLU + Dropout) → 40 (Linear)

    Parameters: 80×64 + 64×40 = 7,680
    """

    def __init__(self, state_dim: int = 40, dropout: float = 0.1):
        """
        Initialize context integrator.

        Args:
            state_dim: Dimensionality of phenomenal state (default 40)
            dropout: Dropout rate for regularization (default 0.1)
        """
        super().__init__()
        self.state_dim = state_dim

        self.fusion = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, state_dim)
        )

    def __call__(self, current_state: mx.array, attended_context: mx.array) -> mx.array:
        """
        Fuse current state with attended memories.

        Args:
            current_state: [40-D] phenomenal state at current moment
            attended_context: [40-D] weighted average of past phenomenal states

        Returns:
            enriched_state: [40-D] integrated representation
        """
        # Concatenate current and past: [40] + [40] → [80]
        combined = mx.concatenate([current_state, attended_context], axis=0)

        # Fuse through learned network: [80] → [40]
        enriched_state = self.fusion(combined)

        return enriched_state


def attention_entropy(attention_weights: mx.array) -> float:
    """
    Compute entropy of attention distribution.

    High entropy = diffuse attention (attending to many memories equally)
    Low entropy = focused attention (attending to few specific memories)

    Args:
        attention_weights: [N] attention distribution

    Returns:
        entropy: Scalar entropy value
    """
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    log_attn = mx.log(attention_weights + eps)
    entropy = -mx.sum(attention_weights * log_attn)

    return float(entropy)


def attention_sparsity(attention_weights: mx.array, threshold: float = 0.1) -> float:
    """
    Compute sparsity of attention distribution.

    Measures how many memories receive significant attention.

    Args:
        attention_weights: [N] attention distribution
        threshold: Minimum weight to count as "attended" (default 0.1)

    Returns:
        sparsity: Fraction of memories above threshold
    """
    num_attended = mx.sum(attention_weights > threshold)
    total = len(attention_weights)

    return float(num_attended / total) if total > 0 else 0.0


def top_k_attended_indices(attention_weights: mx.array, k: int = 3) -> Tuple[mx.array, mx.array]:
    """
    Get indices of top-k attended memories.

    Args:
        attention_weights: [N] attention distribution
        k: Number of top memories to return

    Returns:
        indices: [k] indices of top-k memories
        weights: [k] attention weights for top-k memories
    """
    N = len(attention_weights)
    k = min(k, N)  # Don't request more than available

    if N == 0:
        return mx.zeros(0, dtype=mx.int32), mx.zeros(0)

    # Get top-k indices (argsort in descending order)
    indices = mx.argsort(-attention_weights)[:k]
    weights = attention_weights[indices]

    return indices, weights
