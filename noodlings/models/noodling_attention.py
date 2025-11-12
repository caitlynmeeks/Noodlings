"""
Consilience Phase 3: Attention Mechanisms & Episodic Memory

Extends Phase 1/2 with:
- Episodic memory buffer (last 100 moments)
- Multi-head attention over past experiences
- Context-augmented prediction
- Working memory and explicit retrieval

This implements the computational correlate of:
- Working memory (Baddeley): Episodic buffer + central executive
- Attention (James): Selective focus on relevant past experiences
- Consciousness (Baars): Global workspace with attended context

Total Parameters: ~49,320 (Phase 1/2: 9,272 + Phase 3 additions: 40,048)

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import datetime

from ..memory.episodic_memory import EpisodicMemory
from .attention_modules import (
    KeyEncoder,
    AttentionQueryGenerator,
    MultiHeadAttention,
    ContextIntegrator
)


def reparameterize(mean: mx.array, logvar: mx.array) -> mx.array:
    """
    Sample from N(mean, var) using reparameterization trick.

    Args:
        mean: Mean of distribution
        logvar: Log variance for numerical stability

    Returns:
        Sample from N(mean, exp(logvar))
    """
    std = mx.exp(0.5 * logvar)
    eps = mx.random.normal(mean.shape)
    return mean + std * eps


class TopDownPredictor(nn.Module):
    """
    Top-down prediction module for hierarchical predictive processing.

    Creates bidirectional causal loops between hierarchical layers, which
    dramatically increases integrated information (Φ) by making layers
    causally interdependent.

    Key insight: In IIT, Φ measures how much information is lost when you
    partition a system. If slow layer PREDICTS medium layer's state, you
    can't partition them without losing the prediction signal = higher Φ!

    This implements predictive coding (Rao & Ballard, 1999) where higher
    levels generate predictions about lower levels, creating prediction errors
    that drive learning and integration.
    """

    def __init__(self, source_dim: int, target_dim: int):
        """
        Initialize top-down predictor.

        Args:
            source_dim: Dimension of higher layer (e.g., slow = 8)
            target_dim: Dimension of lower layer (e.g., medium = 16)
        """
        super().__init__()
        # Two-layer MLP for non-linear prediction
        # Wider hidden layer to capture complex mappings
        hidden_dim = max(source_dim * 2, target_dim)

        self.predictor = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(hidden_dim, target_dim)
        )

    def __call__(self, h_source: mx.array) -> mx.array:
        """
        Generate top-down prediction.

        Args:
            h_source: Hidden state from higher layer [batch, source_dim]

        Returns:
            prediction: Predicted state for lower layer [batch, target_dim]
        """
        return self.predictor(h_source)


class NoodlingModelWithAttention(nn.Module):
    """
    Phase 3: Hierarchical affective consciousness with episodic memory and attention.

    Combines:
    - Phase 1: Multi-timescale predictive processing (LSTM/GRU hierarchy)
    - Phase 2 (optional): Variational uncertainty modeling (VAE)
    - Phase 3: Episodic memory + multi-head attention

    Architecture Flow:
    1. Affect → Hierarchical encoding → Phenomenal state (40-D)
    2. Optional VAE transformation → Uncertainty-aware state
    3. Generate attention query from current state
    4. Attend to episodic memory buffer
    5. Integrate current + attended context
    6. Predict next state from enriched representation
    7. Store current moment in memory for future retrieval

    Key Innovation: The agent can now explicitly "remember" and retrieve
    past experiences, not just implicitly encode them in slow layer weights.
    """

    def __init__(
        self,
        affect_dim: int = 5,
        fast_hidden: int = 16,
        medium_hidden: int = 16,
        slow_hidden: int = 8,
        predictor_hidden: int = 64,
        memory_capacity: int = 100,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_vae: bool = False
    ):
        """
        Initialize Phase 3 model with attention over episodic memory.

        Args:
            affect_dim: Dimensionality of affect input (default 5)
            fast_hidden: Fast layer hidden size (default 16)
            medium_hidden: Medium layer hidden size (default 16)
            slow_hidden: Slow layer hidden size (default 8)
            predictor_hidden: Predictor network hidden size (default 64)
            memory_capacity: Max number of moments to store (default 100)
            num_attention_heads: Number of attention heads (default 4)
            attention_dropout: Dropout rate for attention (default 0.1)
            use_vae: Whether to use Phase 2 VAE extension (default False)
        """
        super().__init__()

        # Architecture dimensions
        self.affect_dim = affect_dim
        self.fast_hidden = fast_hidden
        self.medium_hidden = medium_hidden
        self.slow_hidden = slow_hidden
        self.joint_dim = fast_hidden + medium_hidden + slow_hidden  # 40-D
        self.use_vae = use_vae
        self.step = 0

        # Phase 1: Hierarchical encoders (deterministic)
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=fast_hidden)
        self.medium_lstm = nn.LSTM(input_size=fast_hidden, hidden_size=medium_hidden)
        self.slow_gru = nn.GRU(input_size=medium_hidden, hidden_size=slow_hidden)

        # Phase 2 (optional): VAE components
        if use_vae:
            self.fast_mean_head = nn.Linear(fast_hidden, fast_hidden)
            self.fast_logvar_head = nn.Linear(fast_hidden, fast_hidden)
            self.med_mean_head = nn.Linear(medium_hidden, medium_hidden)
            self.med_logvar_head = nn.Linear(medium_hidden, medium_hidden)
            self.slow_mean_head = nn.Linear(slow_hidden, slow_hidden)
            self.slow_logvar_head = nn.Linear(slow_hidden, slow_hidden)

        # Phase 3: Episodic memory and attention
        self.episodic_memory = EpisodicMemory(
            capacity=memory_capacity,
            state_dim=self.joint_dim,
            key_dim=64
        )

        self.key_encoder = KeyEncoder(state_dim=self.joint_dim, key_dim=64)
        self.query_generator = AttentionQueryGenerator(
            state_dim=self.joint_dim,
            query_dim=64,
            dropout=attention_dropout
        )
        self.attention = MultiHeadAttention(
            query_dim=64,
            key_dim=64,
            value_dim=self.joint_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )
        self.context_integrator = ContextIntegrator(
            state_dim=self.joint_dim,
            dropout=attention_dropout
        )

        # Predictor: Uses enriched state (current + attended context)
        # Output full internal state (fast + medium + slow = 40D)
        self.state_dim = fast_hidden + medium_hidden + slow_hidden
        self.predictor = nn.Sequential(
            nn.Linear(self.joint_dim, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, self.state_dim)
        )

    def __call__(
        self,
        affect_sequence: mx.array,
        h_fast: mx.array,
        c_fast: mx.array,
        h_med: mx.array,
        c_med: mx.array,
        h_slow: mx.array,
        metadata: Optional[Dict] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass with attention over episodic memory.

        Args:
            affect_sequence: [batch, seq_len, 5] affect inputs
            h_fast, c_fast: Fast LSTM state
            h_med, c_med: Medium LSTM state
            h_slow: Slow GRU state
            metadata: Optional dict with user_text, timestamp, etc.

        Returns:
            Dictionary containing:
                - Predictions (h_fast_pred)
                - Current states (h_fast, h_med, h_slow)
                - Cell states (c_fast, c_med)
                - Phenomenal state (40-D joint representation)
                - Attended context (40-D weighted past)
                - Attention weights (N,)
                - Enriched state (current + context)
                - If VAE: mean and logvar for each layer
        """
        # 1. Hierarchical processing (Phase 1)
        # Fast layer: Immediate affective response
        h_fast_seq, c_fast_seq = self.fast_lstm(affect_sequence, hidden=h_fast, cell=c_fast)

        # Medium layer: Conversational dynamics
        h_med_seq, c_med_seq = self.medium_lstm(h_fast_seq, hidden=h_med, cell=c_med)

        # Slow layer: Long-term relationship patterns
        h_slow_seq = self.slow_gru(h_med_seq, hidden=h_slow)

        # Extract final states from sequences
        h_fast_final = h_fast_seq[:, -1, :]  # [batch, 16]
        h_med_final = h_med_seq[:, -1, :]    # [batch, 16]
        h_slow_final = h_slow_seq[:, -1, :]  # [batch, 8]

        # 2. Optional VAE transformation (Phase 2)
        vae_outputs = {}
        if self.use_vae:
            # Fast layer variational transformation
            h_fast_mean = self.fast_mean_head(h_fast_final)
            h_fast_logvar = self.fast_logvar_head(h_fast_final)
            h_fast_final = reparameterize(h_fast_mean, h_fast_logvar)

            # Medium layer variational transformation
            h_med_mean = self.med_mean_head(h_med_final)
            h_med_logvar = self.med_logvar_head(h_med_final)
            h_med_final = reparameterize(h_med_mean, h_med_logvar)

            # Slow layer variational transformation
            h_slow_mean = self.slow_mean_head(h_slow_final)
            h_slow_logvar = self.slow_logvar_head(h_slow_final)
            h_slow_final = reparameterize(h_slow_mean, h_slow_logvar)

            vae_outputs = {
                'h_fast_mean': h_fast_mean,
                'h_fast_logvar': h_fast_logvar,
                'h_med_mean': h_med_mean,
                'h_med_logvar': h_med_logvar,
                'h_slow_mean': h_slow_mean,
                'h_slow_logvar': h_slow_logvar
            }

        # 3. Current internal state (40-D joint representation)
        current_state = mx.concatenate(
            [h_fast_final, h_med_final, h_slow_final],
            axis=-1
        )  # [batch, 40]

        # Remove batch dimension for attention (assume batch=1)
        current_state = current_state.squeeze(0)  # [40]

        # 4. Attention over episodic memory (Phase 3)
        attended_context = None
        attention_weights = None
        enriched_state = current_state

        if len(self.episodic_memory) > 0:
            # Generate query: "What past moments are relevant NOW?"
            query = self.query_generator(current_state)  # [64]

            # Retrieve memory keys and values
            keys = self.episodic_memory.get_keys()      # [N, 64]
            values = self.episodic_memory.get_values()  # [N, 40]

            # Attend to relevant memories
            attended_context, attention_weights = self.attention(
                query, keys, values
            )

            # Integrate current state with attended memories
            enriched_state = self.context_integrator(
                current_state, attended_context
            )

        # 5. Predict next fast-layer state using enriched representation
        h_fast_pred = self.predictor(enriched_state)  # [16]

        # 6. Store current moment in episodic memory
        key_vector = self.key_encoder(current_state)  # [64]

        memory_entry = {
            'step': self.step,
            'timestamp': metadata.get('timestamp') if metadata else datetime.datetime.now(),
            'internal_state': current_state,
            'affect': affect_sequence[:, -1, :].squeeze(0),  # Current affect [5]
            'key_vector': key_vector,
            'user_text': metadata.get('user_text', '') if metadata else '',
            'surprise': None,  # Will be filled in after loss computation
            'attention_weights': attention_weights
        }

        self.episodic_memory.add(memory_entry)
        self.step += 1

        # 7. Prepare outputs
        outputs = {
            'h_fast': h_fast_final,
            'h_fast_pred': h_fast_pred,
            'h_med': h_med_final,
            'h_slow': h_slow_final,
            'c_fast': c_fast_seq[:, -1, :],
            'c_med': c_med_seq[:, -1, :],
            'internal_state': current_state,
            'attended_context': attended_context,
            'attention_weights': attention_weights,
            'enriched_state': enriched_state
        }

        # Add VAE outputs if enabled
        if self.use_vae:
            outputs.update(vae_outputs)

        return outputs

    def reset_states(self) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Initialize all hidden and cell states to zeros.

        Returns:
            Tuple of (h_fast, c_fast, h_med, c_med, h_slow)
        """
        h_fast = mx.zeros((1, self.fast_hidden))
        c_fast = mx.zeros((1, self.fast_hidden))
        h_med = mx.zeros((1, self.medium_hidden))
        c_med = mx.zeros((1, self.medium_hidden))
        h_slow = mx.zeros((1, self.slow_hidden))

        return h_fast, c_fast, h_med, c_med, h_slow

    def clear_memory(self) -> None:
        """
        Clear episodic memory buffer.

        Use when starting a new conversation or episode.
        """
        self.episodic_memory.clear()
        self.step = 0

    def get_memory_statistics(self) -> Dict:
        """
        Get statistics about the episodic memory buffer.

        Returns:
            Dictionary with memory statistics
        """
        return self.episodic_memory.get_statistics()

    def get_most_attended_moments(self, top_k: int = 5):
        """
        Get the most attended moments (anchor memories).

        Args:
            top_k: Number of top moments to return

        Returns:
            List of (step, importance, user_text) tuples
        """
        return self.episodic_memory.get_most_attended(top_k)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts by component
        """
        def count_module_params(module):
            total = 0
            for p in module.parameters().values():
                if isinstance(p, dict):
                    # Nested parameters (e.g., LSTM/GRU)
                    total += sum(np.prod(v.shape) for v in p.values() if hasattr(v, 'shape'))
                else:
                    # Direct parameter
                    total += np.prod(p.shape) if hasattr(p, 'shape') else 0
            return total

        counts = {
            'fast_lstm': count_module_params(self.fast_lstm),
            'medium_lstm': count_module_params(self.medium_lstm),
            'slow_gru': count_module_params(self.slow_gru),
            'key_encoder': count_module_params(self.key_encoder),
            'query_generator': count_module_params(self.query_generator),
            'attention': count_module_params(self.attention),
            'context_integrator': count_module_params(self.context_integrator),
            'predictor': count_module_params(self.predictor)
        }

        if self.use_vae:
            vae_params = (
                count_module_params(self.fast_mean_head) +
                count_module_params(self.fast_logvar_head) +
                count_module_params(self.med_mean_head) +
                count_module_params(self.med_logvar_head) +
                count_module_params(self.slow_mean_head) +
                count_module_params(self.slow_logvar_head)
            )
            counts['vae_heads'] = vae_params

        counts['total'] = sum(counts.values())

        return counts

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Alias for count_parameters() to maintain API compatibility.

        Returns:
            Dictionary with parameter counts by component
        """
        return self.count_parameters()
