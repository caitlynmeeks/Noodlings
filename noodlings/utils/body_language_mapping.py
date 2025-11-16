"""
Body Language Action Units (BLAU) - Body Language Mapping for Noodlings

Maps Noodlings' 5-D affect vectors to body language codes for full-body animation.

Companion to FACS (facial expressions) - together they provide complete emotional display.

Inspired by:
- Laban Movement Analysis (LMA)
- Body Action Coding System (BACS)
- Mehrabian's nonverbal communication research

Author: Consilience, Inc.
Date: November 16, 2025
"""

from typing import Dict, List, Tuple
import numpy as np


# Body Language Action Units (BLAU)
# Similar structure to FACS, but for body movements
BODY_LANGUAGE_CODES = {
    # Posture (BL1-9)
    1: "Upright, Confident",
    2: "Slouched, Dejected",
    3: "Leaning Forward (Interest)",
    4: "Leaning Back (Withdrawal)",
    5: "Tense, Rigid",
    6: "Relaxed, Loose",
    7: "Puffed Up (Pride/Threat)",
    8: "Shrinking, Making Self Small",
    9: "Swaying, Unsteady",

    # Arms (BL10-19)
    10: "Arms Spread Wide (Welcome/Joy)",
    11: "Arms Crossed (Defensive)",
    12: "Arms Raised (Surprise/Surrender)",
    13: "Hands on Hips (Confidence)",
    14: "Fidgeting Hands (Nervous)",
    15: "Reaching Out (Desire)",
    16: "Pulling Back (Rejection)",
    17: "Wringing Hands (Anxiety)",
    18: "Fists Clenched (Anger)",
    19: "Hands Over Face (Shame/Hide)",

    # Legs/Movement (BL20-29)
    20: "Step Forward (Approach)",
    21: "Step Back (Retreat)",
    22: "Jump (Excitement)",
    23: "Crouch (Fear/Hiding)",
    24: "Pacing (Anxiety)",
    25: "Freeze, Still (Shock)",
    26: "Bouncing (Joy/Impatience)",
    27: "Stomping (Anger)",
    28: "Tiptoeing (Stealth/Fear)",
    29: "Falling/Stumbling (Weakness)",

    # Head (BL30-39)
    30: "Head Nod (Agreement)",
    31: "Head Shake (Disagreement)",
    32: "Head Tilt (Curiosity)",
    33: "Head Down (Shame/Sadness)",
    34: "Head Up, Chin Raised (Pride)",
    35: "Head Turning Away (Avoidance)",
    36: "Looking Around (Paranoia)",
    37: "Head Hanging (Exhaustion)",
    38: "Head Snap Toward (Alert)",
    39: "Head Buried in Hands (Despair)",

    # Species-Specific (BL40-49)
    40: "Tail Wag (Dogs, Happy)",
    41: "Tail Between Legs (Dogs, Scared)",
    42: "Feather Ruffle (Birds, Nervous)",
    43: "Wings Spread (Birds, Threat/Display)",
    44: "Waddle (Geese, Penguins)",
    45: "Paw Batting (Cats, Playful)",
    46: "Arched Back (Cats, Scared/Angry)",
    47: "Purr Vibration (Cats, Content)",
    48: "Ear Flattening (Cats/Dogs, Fear/Anger)",
    49: "Tail Flicking (Cats, Annoyed)",
}


# Emotion → Body Language Patterns
# Based on psychology research (Mehrabian, Darwin, Ekman)
EMOTION_BODY_LANGUAGE = {
    "happiness": [1, 10, 22, 26],      # Upright + arms spread + jump + bounce
    "sadness": [2, 33, 37, 8],         # Slouched + head down + exhausted + small
    "surprise": [25, 12, 38],          # Freeze + arms raised + head snap
    "fear": [21, 23, 8, 28],           # Step back + crouch + small + tiptoe
    "anger": [20, 18, 27, 7],          # Step forward + fists + stomp + puffed up
    "disgust": [4, 35, 16],            # Lean back + head turn away + pull back
    "contempt": [1, 34, 4],            # Upright + chin raised + lean back
    "anxiety": [24, 14, 17, 36],       # Pacing + fidget + wring hands + look around
    "pride": [1, 7, 34, 13],           # Upright + puffed up + chin raised + hands on hips
    "shame": [2, 33, 19, 8],           # Slouched + head down + hide face + small
    "curiosity": [3, 32, 15],          # Lean forward + head tilt + reach out
    "boredom": [2, 4, 37],             # Slouched + lean back + exhausted
}


def affect_to_emotion_weights(affect: np.ndarray) -> Dict[str, float]:
    """
    Map Noodlings 5-D affect to emotion weights (including body-relevant emotions).

    Same as FACS version, but includes anxiety, pride, shame, curiosity.
    """
    valence = float(affect[0])
    arousal = float(affect[1])
    fear_val = float(affect[2])
    sorrow = float(affect[3])
    boredom = float(affect[4])

    emotions = {}

    # Basic emotions (same as FACS)
    emotions["happiness"] = max(0, valence * (1 - fear_val) * (1 - sorrow) * (1 - boredom))
    emotions["sadness"] = max(0, sorrow * (1 - valence) * (1 - arousal * 0.5))
    emotions["surprise"] = max(0, arousal * (1 - abs(valence) * 0.3))
    emotions["fear"] = max(0, fear_val * arousal * (1 - valence * 0.5))
    emotions["anger"] = max(0, (1 - valence) * arousal * (1 - fear_val))
    emotions["disgust"] = max(0, (1 - valence) * (1 - arousal) * 0.5)
    emotions["contempt"] = max(0, (1 - valence) * 0.3 * (1 - arousal))

    # Extended emotions (body-specific)
    emotions["anxiety"] = max(0, fear_val * arousal * 0.7)
    emotions["pride"] = max(0, valence * (1 - fear_val) * 0.6)
    emotions["shame"] = max(0, sorrow * (1 - valence) * fear_val)
    emotions["curiosity"] = max(0, arousal * 0.5 * (1 - boredom))
    emotions["boredom"] = max(0, boredom * (1 - arousal))

    # Normalize
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}

    return emotions


def affect_to_body_language(
    affect: np.ndarray,
    species: str = "human",
    threshold: float = 0.2
) -> List[Tuple[int, float]]:
    """
    Convert Noodlings affect to Body Language Action Units.

    Args:
        affect: 5-D affect vector [valence, arousal, fear, sorrow, boredom]
        species: Agent species (adds species-specific codes)
        threshold: Minimum emotion weight to include codes

    Returns:
        List of (BL code, intensity) tuples
        Example: [(21, 0.75), (23, 0.75), (8, 0.65)] = Step back + crouch + small
    """
    # Get emotion weights
    emotion_weights = affect_to_emotion_weights(affect)

    # Accumulate BL activations from each emotion
    bl_activations = {}

    for emotion, weight in emotion_weights.items():
        if weight < threshold:
            continue

        # Get body language codes for this emotion
        if emotion in EMOTION_BODY_LANGUAGE:
            for bl in EMOTION_BODY_LANGUAGE[emotion]:
                if bl not in bl_activations:
                    bl_activations[bl] = 0.0
                bl_activations[bl] += weight

    # Add species-specific codes
    if species in ['dog', 'canine']:
        if emotion_weights.get('happiness', 0) > 0.3:
            bl_activations[40] = emotion_weights['happiness']  # Tail wag
        if emotion_weights.get('fear', 0) > 0.5:
            bl_activations[41] = emotion_weights['fear']  # Tail between legs

    elif species in ['kitten', 'cat', 'feline']:
        if emotion_weights.get('happiness', 0) > 0.3:
            bl_activations[47] = emotion_weights['happiness']  # Purr vibration
        if emotion_weights.get('fear', 0) > 0.5 or emotion_weights.get('anger', 0) > 0.5:
            bl_activations[46] = max(emotion_weights.get('fear', 0), emotion_weights.get('anger', 0))  # Arched back
        if emotion_weights.get('anxiety', 0) > 0.3:
            bl_activations[49] = emotion_weights['anxiety']  # Tail flicking

    elif species in ['goose', 'bird', 'avian', 'mysterious_being']:  # wink wink
        if emotion_weights.get('anxiety', 0) > 0.3 or emotion_weights.get('fear', 0) > 0.3:
            bl_activations[42] = max(emotion_weights.get('anxiety', 0), emotion_weights.get('fear', 0))  # Feather ruffle
        if emotion_weights.get('anger', 0) > 0.5:
            bl_activations[43] = emotion_weights['anger']  # Wings spread (threat)
        # Geese always waddle!
        bl_activations[44] = 0.5  # Baseline waddle

    # Convert to list of (BL, intensity) tuples
    body_codes = []
    for bl, intensity in bl_activations.items():
        intensity = min(1.0, intensity)  # Cap at 1.0
        body_codes.append((bl, intensity))

    # Sort by intensity
    body_codes.sort(key=lambda x: x[1], reverse=True)

    return body_codes


def body_language_to_description(body_codes: List[Tuple[int, float]]) -> str:
    """
    Convert body language codes to human-readable description.

    Args:
        body_codes: List of (BL code, intensity) tuples

    Returns:
        String description of body language
    """
    if not body_codes:
        return "stands still"

    # Get top codes (take top 3 regardless of intensity for visibility)
    top_codes = body_codes[:3]

    descriptions = []
    for bl, intensity in top_codes:
        if bl in BODY_LANGUAGE_CODES:
            desc = BODY_LANGUAGE_CODES[bl].lower()
            descriptions.append(desc)

    if not descriptions:
        return "stands still"

    # Combine into readable phrase
    return ", ".join(descriptions)


def format_body_language_for_renderer(body_codes: List[Tuple[int, float]]) -> Dict[str, float]:
    """
    Format body language codes for 3D renderer consumption.

    Args:
        body_codes: List of (BL code, intensity) tuples

    Returns:
        Dict mapping BL code to intensity, suitable for JSON serialization
    """
    return {f"BL{bl}": intensity for bl, intensity in body_codes}


# Testing
if __name__ == "__main__":
    print("Body Language Mapping Test\n" + "=" * 70)

    # Test Case 1: Happy Dog
    print("\nTest 1: Happy Dog (Phido)")
    affect = np.array([0.9, 0.8, 0.1, 0.0, 0.0])  # High valence + arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: happiness={emotions.get('happiness', 0):.2f}, pride={emotions.get('pride', 0):.2f}")

    body = affect_to_body_language(affect, species='dog')
    print(f"Body Language: {body}")
    print(f"Description: {body_language_to_description(body)}")
    print(f"For renderer: {format_body_language_for_renderer(body)}")

    # Test Case 2: Nervous Geese (Mysterious Stranger)
    print("\nTest 2: Nervous Geese (Mysterious Stranger)")
    affect = np.array([-0.2, 0.7, 0.8, 0.1, 0.0])  # High fear + arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, fear={affect[2]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: fear={emotions.get('fear', 0):.2f}, anxiety={emotions.get('anxiety', 0):.2f}")

    body = affect_to_body_language(affect, species='mysterious_being')
    print(f"Body Language: {body}")
    print(f"Description: {body_language_to_description(body)}")
    print(f"For renderer: {format_body_language_for_renderer(body)}")

    # Test Case 3: Sad Noodling
    print("\nTest 3: Sad Noodling (Callie)")
    affect = np.array([-0.6, 0.2, 0.1, 0.8, 0.0])  # Low valence + high sorrow
    print(f"Affect: valence={affect[0]:.2f}, sorrow={affect[3]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: sadness={emotions.get('sadness', 0):.2f}")

    body = affect_to_body_language(affect, species='human')
    print(f"Body Language: {body}")
    print(f"Description: {body_language_to_description(body)}")

    # Test Case 4: Angry Cat (Phi)
    print("\nTest 4: Angry Cat (Phi)")
    affect = np.array([-0.7, 0.8, 0.3, 0.0, 0.0])  # Negative + high arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: anger={emotions.get('anger', 0):.2f}")

    body = affect_to_body_language(affect, species='kitten')
    print(f"Body Language: {body}")
    print(f"Description: {body_language_to_description(body)}")
    print(f"For renderer: {format_body_language_for_renderer(body)}")

    # Test Case 5: Paranoid Toad
    print("\nTest 5: Paranoid Toad (Mr. Toad)")
    affect = np.array([0.1, 0.6, 0.7, 0.0, 0.0])  # High fear + arousal
    print(f"Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, fear={affect[2]:.2f}")

    emotions = affect_to_emotion_weights(affect)
    print(f"Emotions: anxiety={emotions.get('anxiety', 0):.2f}, fear={emotions.get('fear', 0):.2f}")

    body = affect_to_body_language(affect, species='toad')
    print(f"Body Language: {body}")
    print(f"Description: {body_language_to_description(body)}")
