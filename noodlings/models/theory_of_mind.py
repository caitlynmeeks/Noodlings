"""
Theory of Mind Module: Inferring Others' Mental States

Implements computational theory of mind for Phase 4:
- Infer internal states of other agents from observations
- Estimate confidence in inferences
- Learn patterns of how people think and feel

This module predicts what others might be experiencing based on:
- Linguistic cues (what they say)
- Behavioral patterns (what they do)
- Contextual information (situation, history)
- Relationship dynamics (how they typically interact)

Parameters: ~55K (main inference network)

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple


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


class TheoryOfMindModule(nn.Module):
    """
    Infers internal states of other agents.

    Architecture:
    - Input: Linguistic features [128] + context [64] + history [40]
    - Hidden: 256 → 128 dimensions
    - Output: Mean + logvar of 40-D internal state
    - Confidence: Separate network estimating certainty

    The module learns to predict:
    1. What is this person feeling? (inferred state)
    2. How sure am I? (confidence)

    Training uses:
    - Self-supervised: Predict user's stated feelings about others
    - Relationship patterns: Past interactions → future states
    - Proxy signals: User actions toward others
    """

    def __init__(
        self,
        linguistic_dim: int = 128,
        context_dim: int = 64,
        history_dim: int = 40,
        state_dim: int = 40,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize Theory of Mind module.

        Args:
            linguistic_dim: Dimension of linguistic features (what they said)
            context_dim: Dimension of contextual features (situation)
            history_dim: Dimension of their past state
            state_dim: Output internal state dimension
            hidden_dim: Hidden layer size
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.linguistic_dim = linguistic_dim
        self.context_dim = context_dim
        self.history_dim = history_dim
        self.state_dim = state_dim

        input_dim = linguistic_dim + context_dim + history_dim

        # Main inference network: Observations → Inferred state distribution
        self.inference_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, state_dim * 2)  # mean + logvar
        )

        # Confidence estimator: How certain am I about this inference?
        self.confidence_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # Note: No sigmoid - raw logit, apply sigmoid at inference
        )

        # Uncertainty decomposition: Why am I uncertain?
        # Helps identify: lack of data, conflicting data, high noise, fundamental uncertainty
        self.uncertainty_analyzer = nn.Sequential(
            nn.Linear(state_dim + 2, 64),  # state + (mean_logvar, var_logvar)
            nn.ReLU(),
            nn.Linear(64, 4)  # [epistemic, aleatoric, relational, contextual]
        )

    def __call__(
        self,
        linguistic_features: mx.array,
        context: mx.array,
        history: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Infer another agent's internal state.

        Args:
            linguistic_features: [batch, 128] what they said
            context: [batch, 64] situational features
            history: [batch, 40] their past inferred state (optional)

        Returns:
            inferred_state: [batch, 40] sampled internal state
            confidence: [batch, 1] how certain (0-1 after sigmoid)
            mean: [batch, 40] mean of state distribution
            logvar: [batch, 40] log variance of state distribution
            uncertainty_breakdown: [batch, 4] sources of uncertainty
        """
        batch_size = linguistic_features.shape[0]

        # Use zero history if not provided
        if history is None:
            history = mx.zeros((batch_size, self.history_dim))

        # Combine all inputs
        combined = mx.concatenate([linguistic_features, context, history], axis=-1)

        # Infer state distribution
        params = self.inference_net(combined)
        mean, logvar = mx.split(params, 2, axis=-1)

        # Sample state using reparameterization trick
        inferred_state = reparameterize(mean, logvar)

        # Estimate confidence
        confidence_logit = self.confidence_net(inferred_state)
        confidence = mx.sigmoid(confidence_logit)

        # Analyze uncertainty sources
        # Use mean and variance of the distribution as uncertainty indicators
        mean_logvar = mx.mean(logvar, axis=-1, keepdims=True)
        var_logvar = mx.var(logvar, axis=-1, keepdims=True)
        uncertainty_input = mx.concatenate([inferred_state, mean_logvar, var_logvar], axis=-1)
        uncertainty_breakdown = self.uncertainty_analyzer(uncertainty_input)

        return inferred_state, confidence, mean, logvar, uncertainty_breakdown

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate number of parameters in each component.

        Returns:
            Dictionary with parameter counts
        """
        input_dim = self.linguistic_dim + self.context_dim + self.history_dim

        inference_params = (
            input_dim * 256 + 256 +  # First layer
            256 * 128 + 128 +         # Second layer
            128 * (self.state_dim * 2) + (self.state_dim * 2)  # Output layer
        )

        confidence_params = (
            self.state_dim * 32 + 32 +  # First layer
            32 * 1 + 1                   # Output layer
        )

        uncertainty_params = (
            (self.state_dim + 2) * 64 + 64 +  # First layer
            64 * 4 + 4                         # Output layer
        )

        total = inference_params + confidence_params + uncertainty_params

        return {
            'inference_net': inference_params,
            'confidence_net': confidence_params,
            'uncertainty_analyzer': uncertainty_params,
            'total': total
        }


class RelationshipModel(nn.Module):
    """
    Models pairwise relationship dynamics.

    Learns:
    - Attachment style (secure, anxious, avoidant, disorganized)
    - Trust level (0-1, evolves over time)
    - Communication patterns
    - Conflict patterns

    This is a smaller auxiliary module (~7.5K params) that complements
    the Theory of Mind inference.
    """

    def __init__(
        self,
        state_dim: int = 40,
        relationship_dim: int = 32
    ):
        """
        Initialize relationship model.

        Args:
            state_dim: Phenomenal state dimension
            relationship_dim: Relationship vector dimension
        """
        super().__init__()

        self.state_dim = state_dim
        self.relationship_dim = relationship_dim

        # Relationship encoder: Self + Other → Relationship vector
        self.relationship_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, relationship_dim)
        )

        # Attachment style classifier (for interpretability)
        # Categories: secure, anxious-preoccupied, dismissive-avoidant, fearful-avoidant
        self.attachment_classifier = nn.Sequential(
            nn.Linear(relationship_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Logits for 4 attachment styles
            # Note: Apply softmax at inference time
        )

        # Trust estimator: How much does user trust this person?
        self.trust_estimator = nn.Sequential(
            nn.Linear(relationship_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            # Note: Apply sigmoid at inference time for [0, 1] range
        )

        # Communication pattern encoder
        # Dimensions: frequency, responsiveness, emotional_openness, conflict_style
        self.communication_pattern = nn.Sequential(
            nn.Linear(relationship_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def __call__(
        self,
        self_state: mx.array,
        other_state: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Model relationship dynamics between self and other.

        Args:
            self_state: [batch, 40] self internal state
            other_state: [batch, 40] other's internal state

        Returns:
            relationship_vector: [batch, 32] relationship representation
            attachment_logits: [batch, 4] attachment style logits
            trust: [batch, 1] trust level (0-1 after sigmoid)
            communication: [batch, 4] communication pattern features
        """
        # Combine states
        combined = mx.concatenate([self_state, other_state], axis=-1)

        # Encode relationship
        relationship_vector = self.relationship_encoder(combined)

        # Classify attachment style
        attachment_logits = self.attachment_classifier(relationship_vector)

        # Estimate trust
        trust_logit = self.trust_estimator(relationship_vector)
        trust = mx.sigmoid(trust_logit)

        # Encode communication pattern
        communication = self.communication_pattern(relationship_vector)

        return relationship_vector, attachment_logits, trust, communication

    def get_attachment_style(self, attachment_logits: mx.array) -> str:
        """
        Convert attachment logits to human-readable style.

        Args:
            attachment_logits: [4] logits from attachment_classifier

        Returns:
            Attachment style name
        """
        styles = [
            "secure",
            "anxious-preoccupied",
            "dismissive-avoidant",
            "fearful-avoidant"
        ]

        probs = mx.softmax(attachment_logits, axis=-1)
        style_idx = int(mx.argmax(probs).item())

        return styles[style_idx]

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate number of parameters.

        Returns:
            Dictionary with parameter counts
        """
        encoder_params = (
            self.state_dim * 2 * 64 + 64 +
            64 * self.relationship_dim + self.relationship_dim
        )

        attachment_params = (
            self.relationship_dim * 16 + 16 +
            16 * 4 + 4
        )

        trust_params = (
            self.relationship_dim * 16 + 16 +
            16 * 1 + 1
        )

        communication_params = (
            self.relationship_dim * 16 + 16 +
            16 * 4 + 4
        )

        total = encoder_params + attachment_params + trust_params + communication_params

        return {
            'relationship_encoder': encoder_params,
            'attachment_classifier': attachment_params,
            'trust_estimator': trust_params,
            'communication_pattern': communication_params,
            'total': total
        }


# Utility functions for linguistic feature extraction
def extract_linguistic_features(text: str, embedding_model=None) -> mx.array:
    """
    Extract linguistic features from text.

    This is a placeholder for a real linguistic feature extractor.
    In practice, this would use:
    - Sentiment analysis
    - Emotion detection
    - Linguistic style analysis
    - Topic modeling
    - Or embeddings from a small language model

    Args:
        text: Input text
        embedding_model: Optional embedding model

    Returns:
        [128] linguistic feature vector
    """
    # Placeholder: Random features
    # TODO: Replace with actual feature extraction
    return mx.random.normal((128,))


def extract_context_features(
    situation: str,
    present_agents: list,
    recent_history: list
) -> mx.array:
    """
    Extract contextual features.

    Encodes:
    - Current situation/setting
    - Who is present
    - Recent interaction history

    Args:
        situation: Description of current context
        present_agents: List of present agent names
        recent_history: Recent interaction summaries

    Returns:
        [64] context feature vector
    """
    # Placeholder: Random features
    # TODO: Replace with actual feature extraction
    return mx.random.normal((64,))
