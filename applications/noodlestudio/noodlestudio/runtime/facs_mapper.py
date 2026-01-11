# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   FACS Mapper - Affect to Facial Action Coding System to VRM Blendshapes
#
#   Pipeline:
#   1. Affect (PAD + sorrow + boredom) → Emotion weights
#   2. Emotion weights → FACS Action Units
#   3. Action Units → VRM blendshapes
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.facs_mapper
# PURPOSE:  FACS Mapper - Affect to facial expressions
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Affect:
    """
    5-dimensional affect state.

    Based on Mehrabian-Russell PAD model + Caity's extensions.
    """
    valence: float = 0.0      # -1 to +1: pleasure/displeasure
    arousal: float = 0.5      # 0 to 1: activation level
    dominance: float = 0.5    # 0 to 1: control/agency
    sorrow: float = 0.0       # 0 to 1: melancholy, grief
    boredom: float = 0.0      # 0 to 1: disengagement

    @classmethod
    def from_list(cls, values: List[float]) -> 'Affect':
        """Create from list [valence, arousal, dominance, sorrow, boredom]."""
        if len(values) >= 5:
            return cls(
                valence=values[0],
                arousal=values[1],
                dominance=values[2],
                sorrow=values[3],
                boredom=values[4]
            )
        elif len(values) >= 3:
            # Just PAD
            return cls(
                valence=values[0],
                arousal=values[1],
                dominance=values[2]
            )
        else:
            return cls()

    def to_list(self) -> List[float]:
        """Convert to list."""
        return [self.valence, self.arousal, self.dominance, self.sorrow, self.boredom]

    @classmethod
    def neutral(cls) -> 'Affect':
        """Return neutral affect state (relaxed, no expression)."""
        # Low arousal = relaxed face with minimal muscle activation
        return cls(valence=0.0, arousal=0.2, dominance=0.5, sorrow=0.0, boredom=0.0)


@dataclass
class EmotionWeights:
    """Weights for discrete emotions derived from affect."""
    happiness: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    contempt: float = 0.0
    concentration: float = 0.0
    boredom: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'happiness': self.happiness,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'contempt': self.contempt,
            'concentration': self.concentration,
            'boredom': self.boredom,
        }

    def dominant_emotion(self) -> Tuple[str, float]:
        """Return the strongest emotion and its weight."""
        emotions = self.to_dict()
        if not emotions:
            return ('neutral', 0.0)
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant


# ═══════════════════════════════════════════════════════════════════════════
# FACS Action Unit Recipes
# ═══════════════════════════════════════════════════════════════════════════

# Emotion → Action Unit mappings
# Based on Ekman's FACS research
EMOTION_AU_RECIPES: Dict[str, Dict[str, float]] = {
    'happiness': {
        'AU6': 0.8,   # Cheek raiser (Duchenne marker)
        'AU12': 1.0,  # Lip corner puller (smile)
    },
    'sadness': {
        'AU1': 0.8,   # Inner brow raiser
        'AU4': 0.3,   # Slight brow lowerer
        'AU15': 0.7,  # Lip corner depressor
        'AU17': 0.4,  # Chin raiser
    },
    'anger': {
        'AU4': 1.0,   # Brow lowerer
        'AU5': 0.3,   # Upper lid raiser (glare)
        'AU7': 0.8,   # Lid tightener
        'AU23': 0.6,  # Lip tightener
    },
    'fear': {
        'AU1': 0.9,   # Inner brow raiser
        'AU2': 0.7,   # Outer brow raiser
        'AU4': 0.3,   # Slight brow lowerer
        'AU5': 0.8,   # Upper lid raiser (wide eyes)
        'AU20': 0.6,  # Lip stretcher
        'AU25': 0.4,  # Lips part
    },
    'surprise': {
        'AU1': 0.7,   # Inner brow raiser
        'AU2': 0.9,   # Outer brow raiser
        'AU5': 0.9,   # Upper lid raiser
        'AU25': 0.6,  # Lips part
        'AU26': 0.7,  # Jaw drop
    },
    'disgust': {
        'AU9': 0.8,   # Nose wrinkler
        'AU10': 0.6,  # Upper lip raiser
        'AU4': 0.3,   # Slight brow lowerer
    },
    'contempt': {
        'AU12': 0.4,  # Asymmetric lip corner (one side)
        'AU14': 0.5,  # Dimpler (one side)
    },
    'concentration': {
        'AU4': 0.5,   # Brow lowerer
        'AU7': 0.4,   # Lid tightener
        'AU24': 0.3,  # Lip pressor
    },
    'boredom': {
        'AU43': 0.3,  # Partial eye close (heavy lids)
        'AU15': 0.2,  # Slight lip corner depressor
        'AU4': 0.1,   # Minimal brow lowerer
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# AU → VRM Blendshape Mappings
# ═══════════════════════════════════════════════════════════════════════════

# Action Unit → VRM blendshape mappings
# Each AU can contribute to multiple blendshapes with different weights
AU_TO_VRM: Dict[str, List[Tuple[str, float]]] = {
    # Eyes
    'AU5': [('Fcl_EYE_Surprised', 1.0)],      # Wide eyes
    'AU6': [('Fcl_EYE_Joy', 0.8)],            # Happy squint
    'AU7': [('Fcl_EYE_Angry', 0.7)],          # Lid tightener
    'AU43': [('Fcl_EYE_Close', 1.0)],         # Eyes closed

    # Brows
    'AU1': [('Fcl_BRW_Sorrow', 0.8)],         # Inner brow raise
    'AU2': [('Fcl_BRW_Surprised', 0.9)],      # Outer brow raise
    'AU4': [('Fcl_BRW_Angry', 0.8)],          # Brow lowerer

    # Mouth/Expression blendshapes
    'AU12': [('Fcl_ALL_Joy', 0.7), ('Fcl_MTH_Joy', 0.5)],  # Smile
    'AU15': [('Fcl_ALL_Sorrow', 0.6)],        # Frown
    'AU25': [('Fcl_MTH_A', 0.4)],             # Lips part
    'AU26': [('Fcl_MTH_A', 0.7)],             # Jaw drop

    # Composite expressions
    'AU9': [('Fcl_ALL_Angry', 0.3)],          # Nose wrinkle → partial anger
    'AU10': [('Fcl_ALL_Angry', 0.2)],         # Upper lip raise
    'AU20': [('Fcl_ALL_Surprised', 0.4)],     # Lip stretch → partial surprise
    'AU23': [('Fcl_ALL_Angry', 0.4)],         # Lip tightener
    'AU24': [('Fcl_ALL_Angry', 0.2)],         # Lip pressor
    'AU17': [('Fcl_ALL_Sorrow', 0.3)],        # Chin raiser
    'AU14': [('Fcl_ALL_Fun', 0.3)],           # Dimpler (smirk)
}

# Standard VRM blendshape names for reference
VRM_BLENDSHAPES = [
    # Full face expressions
    'Fcl_ALL_Neutral',
    'Fcl_ALL_Joy',
    'Fcl_ALL_Angry',
    'Fcl_ALL_Sorrow',
    'Fcl_ALL_Fun',
    'Fcl_ALL_Surprised',

    # Eyes
    'Fcl_EYE_Close',
    'Fcl_EYE_Close_L',
    'Fcl_EYE_Close_R',
    'Fcl_EYE_Joy',
    'Fcl_EYE_Angry',
    'Fcl_EYE_Sorrow',
    'Fcl_EYE_Surprised',

    # Brows
    'Fcl_BRW_Joy',
    'Fcl_BRW_Angry',
    'Fcl_BRW_Sorrow',
    'Fcl_BRW_Surprised',
    'Fcl_BRW_Fun',

    # Mouth (visemes)
    'Fcl_MTH_A',
    'Fcl_MTH_I',
    'Fcl_MTH_U',
    'Fcl_MTH_E',
    'Fcl_MTH_O',
    'Fcl_MTH_Joy',
    'Fcl_MTH_Angry',
    'Fcl_MTH_Sorrow',
]


# ═══════════════════════════════════════════════════════════════════════════
# FACSMapper Class
# ═══════════════════════════════════════════════════════════════════════════

class FACSMapper:
    """
    Maps affect to facial expressions via FACS Action Units.

    Pipeline:
    1. affect_to_emotions(): PAD + sorrow + boredom → emotion weights
    2. emotions_to_aus(): emotion weights → Action Unit intensities
    3. aus_to_vrm(): Action Units → VRM blendshape values

    Full pipeline: map_affect_to_vrm()
    """

    def __init__(
        self,
        emotion_recipes: Optional[Dict[str, Dict[str, float]]] = None,
        au_mappings: Optional[Dict[str, List[Tuple[str, float]]]] = None
    ):
        """
        Initialize mapper with optional custom recipes.

        Args:
            emotion_recipes: Custom emotion → AU mappings
            au_mappings: Custom AU → VRM mappings
        """
        self.emotion_recipes = emotion_recipes or EMOTION_AU_RECIPES
        self.au_mappings = au_mappings or AU_TO_VRM

    def affect_to_emotions(self, affect: Affect) -> EmotionWeights:
        """
        Convert 5D affect to discrete emotion weights.

        Based on Mehrabian-Russell PAD mappings with extensions.

        Args:
            affect: 5D affect state

        Returns:
            EmotionWeights with values 0-1
        """
        v = affect.valence
        a = affect.arousal
        d = affect.dominance
        s = affect.sorrow
        b = affect.boredom

        emotions = EmotionWeights()

        # Happiness: positive valence + moderate-high arousal
        emotions.happiness = max(0, v) * (0.5 + 0.5 * a)

        # Sadness: negative valence + low arousal + sorrow
        emotions.sadness = max(0, -v) * (1 - a) * 0.5 + s * 0.5

        # Anger: negative valence + high arousal + high dominance
        emotions.anger = max(0, -v) * a * d

        # Fear: negative valence + high arousal + low dominance
        emotions.fear = max(0, -v) * a * (1 - d)

        # Surprise: high arousal (valence-neutral)
        emotions.surprise = a * (1 - abs(v)) * 0.5

        # Disgust: negative valence + low arousal
        emotions.disgust = max(0, -v) * (1 - a) * 0.3

        # Concentration: moderate arousal + high dominance + neutral valence
        emotions.concentration = d * (1 - abs(v)) * 0.5

        # Boredom: direct mapping
        emotions.boredom = b

        # Contempt: negative valence + high dominance + low arousal
        emotions.contempt = max(0, -v) * d * (1 - a) * 0.3

        return emotions

    def emotions_to_aus(self, emotions: EmotionWeights) -> Dict[str, float]:
        """
        Convert emotion weights to Action Unit intensities.

        Blends AU recipes by emotion weights.

        Args:
            emotions: Emotion weights

        Returns:
            Dict mapping AU names (e.g., 'AU12') to intensities (0-1)
        """
        aus: Dict[str, float] = defaultdict(float)

        emotion_dict = emotions.to_dict()
        for emotion, weight in emotion_dict.items():
            if emotion in self.emotion_recipes and weight > 0.01:
                recipe = self.emotion_recipes[emotion]
                for au, intensity in recipe.items():
                    # Additive blending with max cap at 1.0
                    aus[au] = min(1.0, aus[au] + intensity * weight)

        return dict(aus)

    def aus_to_vrm(self, aus: Dict[str, float]) -> Dict[str, float]:
        """
        Convert Action Units to VRM blendshape values.

        Args:
            aus: Dict mapping AU names to intensities

        Returns:
            Dict mapping VRM blendshape names to values (0-1)
        """
        blendshapes: Dict[str, float] = defaultdict(float)

        for au, intensity in aus.items():
            if au in self.au_mappings:
                for vrm_shape, scale in self.au_mappings[au]:
                    # Additive blending with max cap
                    blendshapes[vrm_shape] = min(1.0,
                        blendshapes[vrm_shape] + intensity * scale)

        return dict(blendshapes)

    def map_affect_to_vrm(self, affect: Affect) -> Dict[str, float]:
        """
        Full pipeline: Affect → Emotions → AUs → VRM blendshapes.

        Args:
            affect: 5D affect state

        Returns:
            Dict mapping VRM blendshape names to values (0-1)
        """
        emotions = self.affect_to_emotions(affect)
        aus = self.emotions_to_aus(emotions)
        vrm = self.aus_to_vrm(aus)
        return vrm

    def map_affect_to_vrm_with_details(self, affect: Affect) -> Dict[str, any]:
        """
        Full pipeline with intermediate results for debugging/teaching.

        Args:
            affect: 5D affect state

        Returns:
            Dict with 'emotions', 'aus', 'vrm', and 'dominant_emotion'
        """
        emotions = self.affect_to_emotions(affect)
        aus = self.emotions_to_aus(emotions)
        vrm = self.aus_to_vrm(aus)
        dominant = emotions.dominant_emotion()

        return {
            'affect': affect.to_list(),
            'emotions': emotions.to_dict(),
            'aus': aus,
            'vrm': vrm,
            'dominant_emotion': dominant[0],
            'dominant_weight': dominant[1],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

# Singleton mapper instance
_default_mapper: Optional[FACSMapper] = None


def get_default_mapper() -> FACSMapper:
    """Get or create the default FACSMapper instance."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = FACSMapper()
    return _default_mapper


def affect_to_vrm(
    valence: float,
    arousal: float,
    dominance: float,
    sorrow: float = 0.0,
    boredom: float = 0.0
) -> Dict[str, float]:
    """
    Convenience function: Convert affect values to VRM blendshapes.

    Args:
        valence: -1 to +1 (pleasure/displeasure)
        arousal: 0 to 1 (activation)
        dominance: 0 to 1 (control)
        sorrow: 0 to 1 (optional)
        boredom: 0 to 1 (optional)

    Returns:
        Dict mapping VRM blendshape names to values
    """
    mapper = get_default_mapper()
    affect = Affect(valence, arousal, dominance, sorrow, boredom)
    return mapper.map_affect_to_vrm(affect)


def affect_list_to_vrm(affect_list: List[float]) -> Dict[str, float]:
    """
    Convenience function: Convert affect list to VRM blendshapes.

    Args:
        affect_list: [valence, arousal, dominance, sorrow, boredom]

    Returns:
        Dict mapping VRM blendshape names to values
    """
    mapper = get_default_mapper()
    affect = Affect.from_list(affect_list)
    return mapper.map_affect_to_vrm(affect)


# ═══════════════════════════════════════════════════════════════════════════
# Teaching Mode (for Kimii-Sensei)
# ═══════════════════════════════════════════════════════════════════════════

class TeachingMapper(FACSMapper):
    """
    Extended mapper for Kimii-Sensei's teaching mode.

    Adds features for demonstrating individual AUs and emotions.
    """

    def isolate_au(self, au: str, intensity: float = 1.0) -> Dict[str, float]:
        """
        Generate VRM blendshapes for a single AU in isolation.

        For teaching: "This is what AU12 (smile) looks like alone."

        Args:
            au: Action Unit name (e.g., 'AU12')
            intensity: How strong (0-1)

        Returns:
            VRM blendshapes for just this AU
        """
        aus = {au: intensity}
        return self.aus_to_vrm(aus)

    def isolate_emotion(self, emotion: str, intensity: float = 1.0) -> Dict[str, float]:
        """
        Generate VRM blendshapes for a single emotion.

        For teaching: "This is what pure happiness looks like."

        Args:
            emotion: Emotion name (e.g., 'happiness')
            intensity: How strong (0-1)

        Returns:
            VRM blendshapes for this emotion
        """
        if emotion not in self.emotion_recipes:
            return {}

        aus = {}
        recipe = self.emotion_recipes[emotion]
        for au, au_intensity in recipe.items():
            aus[au] = au_intensity * intensity

        return self.aus_to_vrm(aus)

    def blend_emotions(
        self,
        emotion_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Blend multiple emotions by weight.

        For teaching: "What does 70% happy + 30% surprised look like?"

        Args:
            emotion_weights: Dict mapping emotion names to weights (should sum to 1)

        Returns:
            VRM blendshapes for the blend
        """
        aus: Dict[str, float] = defaultdict(float)

        for emotion, weight in emotion_weights.items():
            if emotion in self.emotion_recipes and weight > 0.01:
                recipe = self.emotion_recipes[emotion]
                for au, intensity in recipe.items():
                    aus[au] = min(1.0, aus[au] + intensity * weight)

        return self.aus_to_vrm(dict(aus))

    def get_au_description(self, au: str) -> str:
        """Get human-readable description of an Action Unit."""
        descriptions = {
            'AU1': "Inner Brow Raiser - raises the inner part of your eyebrows (sadness, worry)",
            'AU2': "Outer Brow Raiser - raises the outer eyebrows (surprise)",
            'AU4': "Brow Lowerer - brings eyebrows down and together (anger, concentration)",
            'AU5': "Upper Lid Raiser - opens eyes wide (surprise, fear)",
            'AU6': "Cheek Raiser - squishes cheeks up, crinkles eyes (real smile!)",
            'AU7': "Lid Tightener - tenses the eyelids (anger, focus)",
            'AU9': "Nose Wrinkler - scrunches up your nose (disgust)",
            'AU10': "Upper Lip Raiser - lifts the upper lip (disgust)",
            'AU12': "Lip Corner Puller - pulls mouth corners up (smile!)",
            'AU14': "Dimpler - creates dimples (smirk)",
            'AU15': "Lip Corner Depressor - pulls mouth corners down (frown, sadness)",
            'AU17': "Chin Raiser - pushes chin up (doubt, sadness)",
            'AU20': "Lip Stretcher - stretches lips sideways (fear, tension)",
            'AU23': "Lip Tightener - presses lips together tight (anger)",
            'AU24': "Lip Pressor - presses lips together (suppression)",
            'AU25': "Lips Part - opens mouth slightly (surprise, speech)",
            'AU26': "Jaw Drop - opens mouth wide (shock, surprise)",
            'AU43': "Eyes Closed - closes the eyes (sleep, bliss, pain)",
        }
        return descriptions.get(au, f"Action Unit {au}")

    def get_all_au_descriptions(self) -> Dict[str, str]:
        """Get all AU descriptions for teaching UI."""
        return {
            au: self.get_au_description(au)
            for au in ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
                      'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24',
                      'AU25', 'AU26', 'AU43']
        }


# ═══════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== FACS Mapper Test ===\n")

    mapper = FACSMapper()

    # Test happy affect
    happy = Affect(valence=0.8, arousal=0.6, dominance=0.5, sorrow=0, boredom=0)
    result = mapper.map_affect_to_vrm_with_details(happy)

    print(f"Happy affect: {happy}")
    print(f"Dominant emotion: {result['dominant_emotion']} ({result['dominant_weight']:.2f})")
    print(f"Active AUs: {list(result['aus'].keys())}")
    print(f"VRM blendshapes: {result['vrm']}")

    print("\n---\n")

    # Test sad affect
    sad = Affect(valence=-0.6, arousal=0.2, dominance=0.3, sorrow=0.7, boredom=0)
    result = mapper.map_affect_to_vrm_with_details(sad)

    print(f"Sad affect: {sad}")
    print(f"Dominant emotion: {result['dominant_emotion']} ({result['dominant_weight']:.2f})")
    print(f"Active AUs: {list(result['aus'].keys())}")
    print(f"VRM blendshapes: {result['vrm']}")

    print("\n--- Teaching Mode ---\n")

    teaching = TeachingMapper()
    print("AU12 (smile) isolated:")
    print(teaching.isolate_au('AU12'))

    print("\n70% happy + 30% surprised:")
    print(teaching.blend_emotions({'happiness': 0.7, 'surprise': 0.3}))

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
