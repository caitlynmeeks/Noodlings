"""
Affect Head - Continuous 5D Affect Prediction

Predicts continuous affect vectors from phenomenal states.
Trained via affect regression (Option B).

Author: Noodlings Project
Date: November 24, 2025
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Dict


class AffectHead(nn.Module):
    """
    Affect regression head: 40-D phenomenal state -> 5-D affect vector.

    Predicts continuous values instead of discrete emotion classes!
    """

    AFFECT_DIMS = ['valence', 'arousal', 'dominance', 'sorrow', 'boredom']

    def __init__(self, input_dim=40, hidden_dim=64, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, phenomenal_state):
        """
        Forward pass.

        Args:
            phenomenal_state: (batch, 40) or (40,) phenomenal state

        Returns:
            affect_pred: (batch, 5) or (5,) predicted affect vector
        """
        # Ensure batch dimension
        needs_squeeze = False
        if phenomenal_state.ndim == 1:
            phenomenal_state = phenomenal_state[None, :]
            needs_squeeze = True

        # Forward pass
        h = nn.relu(self.fc1(phenomenal_state))
        affect_pred = self.fc2(h)  # No activation - continuous regression

        # Remove batch dimension if input was 1D
        if needs_squeeze:
            affect_pred = affect_pred[0]

        return affect_pred

    def predict(self, phenomenal_state) -> Dict[str, float]:
        """
        Predict affect vector from phenomenal state.

        Args:
            phenomenal_state: (40,) or (batch, 40)

        Returns:
            Dictionary with affect dimensions
        """
        affect_pred = self(phenomenal_state)

        # Handle batch vs single prediction
        if affect_pred.ndim == 2:
            affect_pred = affect_pred[0]

        return {
            dim: float(affect_pred[i])
            for i, dim in enumerate(self.AFFECT_DIMS)
        }

    @classmethod
    def load_pretrained(cls, checkpoint_path: str = None):
        """
        Load pretrained affect head.

        Args:
            checkpoint_path: Path to .npz checkpoint
                            If None, uses default location

        Returns:
            Loaded AffectHead model
        """
        if checkpoint_path is None:
            # Default location
            checkpoint_path = Path(__file__).parent / 'affect_head_finetuned.npz'

        # Initialize model
        model = cls(input_dim=40, hidden_dim=64, output_dim=5)

        # Load weights
        model.load_weights(str(checkpoint_path))

        return model


def interpret_affect(affect: Dict[str, float]) -> str:
    """
    Human-readable interpretation of affect vector.

    Args:
        affect: Dictionary with valence, arousal, dominance, sorrow, boredom

    Returns:
        Natural language description
    """
    valence = affect['valence']
    arousal = affect['arousal']
    dominance = affect['dominance']
    sorrow = affect['sorrow']
    boredom = affect['boredom']

    # Valence interpretation
    if valence > 0.5:
        valence_str = "very positive"
    elif valence > 0.2:
        valence_str = "positive"
    elif valence > -0.2:
        valence_str = "neutral"
    elif valence > -0.5:
        valence_str = "negative"
    else:
        valence_str = "very negative"

    # Arousal interpretation
    if arousal > 0.7:
        arousal_str = "highly excited"
    elif arousal > 0.4:
        arousal_str = "energized"
    elif arousal > 0.2:
        arousal_str = "alert"
    else:
        arousal_str = "calm"

    # Dominance interpretation
    if dominance > 0.7:
        dominance_str = "dominant"
    elif dominance > 0.4:
        dominance_str = "in control"
    elif dominance > 0.2:
        dominance_str = "neutral power"
    else:
        dominance_str = "submissive"

    # Dominant secondary emotion
    secondary = []
    if sorrow > 0.5:
        secondary.append("sorrowful")
    if boredom > 0.5:
        secondary.append("bored")

    # Construct description
    parts = [valence_str, arousal_str, dominance_str]
    if secondary:
        parts.append(", ".join(secondary))

    return ", ".join(parts)


def classify_emotion_from_affect(affect: Dict[str, float]) -> str:
    """
    Classify discrete emotion from continuous affect (for backward compatibility).

    Uses PAD model heuristics based on affect dimensions.

    Args:
        affect: Dictionary with valence, arousal, dominance, sorrow, boredom

    Returns:
        Emotion label (string)
    """
    valence = affect['valence']
    arousal = affect['arousal']
    dominance = affect['dominance']
    sorrow = affect['sorrow']
    boredom = affect['boredom']

    # High sorrow overrides
    if sorrow > 0.6:
        return "sadness"

    # High boredom overrides
    if boredom > 0.6:
        return "boredom"

    # PAD-based classification
    # Fear: negative valence + high arousal + LOW dominance
    if valence < -0.3 and arousal > 0.5 and dominance < 0.3:
        return "fear"

    # Anger: negative valence + high arousal + HIGH dominance
    if valence < -0.3 and arousal > 0.5 and dominance > 0.6:
        return "anger"

    # Pride: positive valence + high arousal + HIGH dominance
    if valence > 0.5 and arousal > 0.4 and dominance > 0.6:
        return "pride"

    # Awe: positive valence + high arousal + LOW dominance
    if valence > 0.3 and arousal > 0.5 and dominance < 0.3:
        return "awe"

    # Valence-arousal quadrants (mid dominance)
    if valence > 0.3:
        if arousal > 0.5:
            return "joy"  # High arousal positive
        else:
            return "love"  # Low arousal positive
    elif valence < -0.3:
        if arousal > 0.5:
            return "anger"  # High arousal negative (default to anger if dominance unclear)
        else:
            return "sadness"  # Low arousal negative
    else:
        # Neutral valence
        if arousal > 0.5:
            return "curiosity"  # High arousal neutral
        else:
            return "boredom"  # Low arousal neutral


if __name__ == '__main__':
    # Test affect head loading
    print("Loading pretrained affect head...")
    affect_head = AffectHead.load_pretrained()

    print("Model loaded successfully!")

    # Test prediction with dummy phenomenal state
    print("\nTesting prediction with random phenomenal state...")
    phenomenal = mx.random.normal((40,))
    affect = affect_head.predict(phenomenal)

    print("\nPredicted affect:")
    for dim, val in affect.items():
        print(f"  {dim:10s}: {val:+.3f}")

    print(f"\nInterpretation: {interpret_affect(affect)}")
    print(f"Emotion (discrete): {classify_emotion_from_affect(affect)}")
