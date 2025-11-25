"""
Emotion Classification Head for Noodlings

Maps 40-D phenomenal state to emotion categories.
Designed for fine-tuning on emotion-labeled conversation data.

Architecture:
- Input: 40-D phenomenal state (16 fast + 16 medium + 8 slow)
- Hidden: 64-D with ReLU
- Output: 10-class softmax (fear, joy, sadness, anger, love, guilt, pride, shame, curiosity, boredom)

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict


class EmotionClassificationHead(nn.Module):
    """
    Classify emotions from phenomenal state vectors.

    This module learns to map the 40-dimensional phenomenal state
    (fast + medium + slow layers) to discrete emotion categories.
    """

    # Emotion index mapping
    EMOTION_LABELS = [
        'fear', 'joy', 'sadness', 'anger', 'love',
        'guilt', 'pride', 'shame', 'curiosity', 'boredom'
    ]

    EMOTION_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}
    IDX_TO_EMOTION = {i: label for i, label in enumerate(EMOTION_LABELS)}

    def __init__(self, state_dim: int = 40, hidden_dim: int = 64, num_emotions: int = 10):
        """
        Initialize emotion classification head.

        Args:
            state_dim: Dimension of phenomenal state (default 40)
            hidden_dim: Hidden layer dimension (default 64)
            num_emotions: Number of emotion classes (default 10)
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions

        # Two-layer MLP
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)

    def __call__(self, phenomenal_state: mx.array) -> mx.array:
        """
        Forward pass: phenomenal state -> emotion logits.

        Args:
            phenomenal_state: (batch_size, 40) or (40,)

        Returns:
            logits: (batch_size, 10) or (10,) unnormalized class scores
        """
        # Ensure 2D
        if phenomenal_state.ndim == 1:
            phenomenal_state = phenomenal_state[None, :]

        # MLP forward
        h = mx.maximum(self.fc1(phenomenal_state), 0)  # ReLU
        logits = self.fc2(h)

        return logits

    def predict(self, phenomenal_state: mx.array) -> str:
        """
        Predict single emotion label.

        Args:
            phenomenal_state: (40,) vector

        Returns:
            emotion: String label (e.g., "joy", "fear")
        """
        logits = self(phenomenal_state)
        pred_idx = int(mx.argmax(logits, axis=-1))
        return self.IDX_TO_EMOTION[pred_idx]

    def predict_with_confidence(self, phenomenal_state: mx.array) -> Dict[str, float]:
        """
        Predict emotion with probability distribution.

        Args:
            phenomenal_state: (40,) vector

        Returns:
            Dict mapping emotion labels to probabilities
        """
        logits = self(phenomenal_state)
        probs = mx.softmax(logits, axis=-1)[0]  # (10,)

        return {
            label: float(probs[i])
            for i, label in enumerate(self.EMOTION_LABELS)
        }


def test_emotion_head():
    """Quick test of emotion classification head."""
    print("Testing Emotion Classification Head")
    print("=" * 60)

    # Create head
    head = EmotionClassificationHead(state_dim=40, hidden_dim=64, num_emotions=10)

    # Test input
    state = mx.random.normal((40,))

    # Forward pass
    logits = head(state)
    print(f"Input shape: {state.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Prediction
    emotion = head.predict(state)
    print(f"Predicted emotion: {emotion}")

    # Confidence
    probs = head.predict_with_confidence(state)
    print("\nProbability distribution:")
    for label, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {label:12s}: {prob:.3f}")

    # Batch test
    batch = mx.random.normal((5, 40))
    logits_batch = head(batch)
    print(f"\nBatch input: {batch.shape}")
    print(f"Batch output: {logits_batch.shape}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_emotion_head()
