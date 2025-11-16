"""
FACS (Facial Action Coding System) Mapping for Noodlings

Maps Noodlings' 5-D affect vectors to FACS Action Units for facial animation.

Based on Paul Ekman's Facial Action Coding System:
https://en.wikipedia.org/wiki/Facial_Action_Coding_System

Author: Consilience, Inc.
Date: November 2025
"""

from typing import Dict, List, Tuple
import numpy as np


# FACS Action Units (AUs) - Anatomical basis for facial expressions
FACS_ACTION_UNITS = {
    # Upper Face
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser",
    4: "Brow Lowerer",
    5: "Upper Lid Raiser",
    6: "Cheek Raiser",
    7: "Lid Tightener",
    9: "Nose Wrinkler",

    # Lower Face
    10: "Upper Lip Raiser",
    12: "Lip Corner Puller",
    15: "Lip Corner Depressor",
    16: "Lower Lip Depressor",
    17: "Chin Raiser",
    18: "Lip Puckerer",
    20: "Lip Stretcher",
    22: "Lip Funneler",
    23: "Lip Tightener",
    24: "Lip Pressor",
    25: "Lips Part",
    26: "Jaw Drop",
    27: "Mouth Stretch",

    # Head & Eye
    51: "Head Turn Left",
    52: "Head Turn Right",
    53: "Head Up",
    54: "Head Down",
    61: "Eyes Left",
    62: "Eyes Right",
    63: "Eyes Up",
    64: "Eyes Down"
}


# Ekman's Basic Emotions → FACS Combinations
BASIC_EMOTION_FACS = {
    "happiness": [6, 12],  # Cheek Raiser + Lip Corner Puller (smile)
    "sadness": [1, 4, 15],  # Inner Brow Raise + Brow Lower + Lip Corner Depress (frown)
    "surprise": [1, 2, 5, 26],  # Brows Raised + Eyes Wide + Jaw Drop
    "fear": [1, 2, 4, 5, 20, 26],  # Raised brows + wide eyes + stretched lips + jaw drop
    "anger": [4, 5, 7, 23],  # Lowered brows + wide eyes + tight lids + tight lips
    "disgust": [9, 15],  # Nose Wrinkle + Lip Corner Depress
    "contempt": [12, 14]  # Unilateral lip corner pull (asymmetric smirk)
}


def affect_to_emotion_weights(affect: np.ndarray) -> Dict[str, float]:
    """
    Map Noodlings 5-D affect to emotion weights.

    Affect vector: [valence, arousal, fear, sorrow, boredom]
    - valence: -1 (negative) to +1 (positive)
    - arousal: 0 (calm) to 1 (excited)
    - fear: 0 (safe) to 1 (afraid)
    - sorrow: 0 (content) to 1 (sad)
    - boredom: 0 (engaged) to 1 (bored)

    Returns:
        Dict mapping emotion name to weight (0-1)
    """
    valence = float(affect[0])
    arousal = float(affect[1])
    fear_val = float(affect[2])
    sorrow = float(affect[3])
    boredom = float(affect[4])

    emotions = {}

    # Happiness: High valence, moderate arousal, low fear/sorrow
    emotions["happiness"] = max(0, valence * (1 - fear_val) * (1 - sorrow) * (1 - boredom))

    # Sadness: Low valence, high sorrow, low arousal
    emotions["sadness"] = max(0, sorrow * (1 - valence) * (1 - arousal * 0.5))

    # Surprise: High arousal, neutral valence (can be positive or negative surprise)
    emotions["surprise"] = max(0, arousal * (1 - abs(valence) * 0.3))

    # Fear: High fear, high arousal, negative valence
    emotions["fear"] = max(0, fear_val * arousal * (1 - valence * 0.5))

    # Anger: Negative valence, high arousal, low fear
    emotions["anger"] = max(0, (1 - valence) * arousal * (1 - fear_val))

    # Disgust: Negative valence, low arousal
    emotions["disgust"] = max(0, (1 - valence) * (1 - arousal) * 0.5)

    # Contempt: Subtle negative valence, low arousal, low intensity
    emotions["contempt"] = max(0, (1 - valence) * 0.3 * (1 - arousal))

    # Boredom: Not a basic emotion, but useful (low arousal, neutral valence, high boredom)
    emotions["boredom"] = max(0, boredom * (1 - arousal))

    # Normalize to sum to 1.0
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}

    return emotions


def affect_to_facs(affect: np.ndarray, threshold: float = 0.2) -> List[Tuple[int, float]]:
    """
    Convert Noodlings affect to FACS Action Units with intensities.

    Args:
        affect: 5-D affect vector [valence, arousal, fear, sorrow, boredom]
        threshold: Minimum emotion weight to include AUs (default 0.2)

    Returns:
        List of (AU code, intensity) tuples
        Example: [(6, 0.8), (12, 0.8)] = Strong smile
    """
    # Get emotion weights
    emotion_weights = affect_to_emotion_weights(affect)

    # Accumulate AU activations from each emotion
    au_activations = {}

    for emotion, weight in emotion_weights.items():
        if weight < threshold:
            continue

        # Get AUs for this emotion
        if emotion in BASIC_EMOTION_FACS:
            for au in BASIC_EMOTION_FACS[emotion]:
                if au not in au_activations:
                    au_activations[au] = 0.0
                au_activations[au] += weight

    # Convert to list of (AU, intensity) tuples
    facs_codes = []
    for au, intensity in au_activations.items():
        # Cap intensity at 1.0
        intensity = min(1.0, intensity)
        facs_codes.append((au, intensity))

    # Sort by intensity (strongest first)
    facs_codes.sort(key=lambda x: x[1], reverse=True)

    return facs_codes


def facs_to_description(facs_codes: List[Tuple[int, float]]) -> str:
    """
    Convert FACS codes to human-readable description.

    Args:
        facs_codes: List of (AU code, intensity) tuples

    Returns:
        String description of facial expression
    """
    if not facs_codes:
        return "neutral expression"

    # Get dominant AUs (intensity > 0.5)
    dominant = [au for au, intensity in facs_codes if intensity > 0.5]

    # Pattern matching for common expressions
    if set(dominant) == {6, 12}:
        return "smiling"
    elif set(dominant) == {1, 4, 15}:
        return "frowning sadly"
    elif set(dominant) >= {1, 2, 5, 26}:
        return "eyes wide with surprise, jaw dropped"
    elif set(dominant) >= {1, 2, 4, 5, 20}:
        return "eyes wide with fear, face tense"
    elif set(dominant) >= {4, 5, 7, 23}:
        return "brows furrowed angrily"
    elif set(dominant) >= {9, 15}:
        return "nose wrinkled in disgust"

    # Generic description
    au_names = [FACS_ACTION_UNITS.get(au, f"AU{au}") for au, _ in facs_codes[:3]]
    return f"expression: {', '.join(au_names[:2])}"


def format_facs_for_renderer(facs_codes: List[Tuple[int, float]]) -> Dict[str, float]:
    """
    Format FACS codes for 3D renderer consumption.

    Args:
        facs_codes: List of (AU code, intensity) tuples

    Returns:
        Dict mapping AU code to intensity, suitable for JSON serialization
    """
    return {f"AU{au}": intensity for au, intensity in facs_codes}


# Testing
if __name__ == "__main__":
    print("FACS Mapping Test\n" + "=" * 60)

    # Test Case 1: Happy
    print("\nTest 1: Happy Noodling")
    affect = np.array([0.8, 0.6, 0.1, 0.0, 0.0])  # High valence, moderate arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, fear={affect[2]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: {emotions}")

    facs = affect_to_facs(affect)
    print(f"FACS: {facs}")
    print(f"Description: {facs_to_description(facs)}")
    print(f"For renderer: {format_facs_for_renderer(facs)}")

    # Test Case 2: Sad
    print("\nTest 2: Sad Noodling")
    affect = np.array([-0.6, 0.2, 0.1, 0.8, 0.0])  # Low valence, high sorrow
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, sorrow={affect[3]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: {emotions}")

    facs = affect_to_facs(affect)
    print(f"FACS: {facs}")
    print(f"Description: {facs_to_description(facs)}")

    # Test Case 3: Surprised
    print("\nTest 3: Surprised Noodling")
    affect = np.array([0.0, 0.9, 0.3, 0.0, 0.0])  # High arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: {emotions}")

    facs = affect_to_facs(affect)
    print(f"FACS: {facs}")
    print(f"Description: {facs_to_description(facs)}")

    # Test Case 4: Fearful
    print("\nTest 4: Fearful Noodling")
    affect = np.array([-0.5, 0.8, 0.9, 0.2, 0.0])  # High fear + arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, fear={affect[2]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: {emotions}")

    facs = affect_to_facs(affect)
    print(f"FACS: {facs}")
    print(f"Description: {facs_to_description(facs)}")
