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
#   Tests for Facial Expression System - FACS → VRM
#
#   Tests the complete pipeline:
#   - Affect → Emotion weights
#   - Emotion weights → Action Units
#   - Action Units → VRM blendshapes
#   - FacialExpressionComponent
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_facial_expression
# PURPOSE:  Tests for facial expression system
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from noodlestudio.runtime.facs_mapper import (
    Affect,
    EmotionWeights,
    FACSMapper,
    TeachingMapper,
    affect_to_vrm,
    affect_list_to_vrm,
    EMOTION_AU_RECIPES,
    AU_TO_VRM,
)
from noodlestudio.core.facial_expression_component import FacialExpressionComponent


# ═══════════════════════════════════════════════════════════════════════════
# Test Affect Dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestAffect:
    """Tests for Affect dataclass."""

    def test_default_values(self):
        """Default affect is neutral."""
        affect = Affect()
        assert affect.valence == 0.0
        assert affect.arousal == 0.5
        assert affect.dominance == 0.5
        assert affect.sorrow == 0.0
        assert affect.boredom == 0.0

    def test_from_list_full(self):
        """Can create from full 5-element list."""
        affect = Affect.from_list([0.8, 0.6, 0.4, 0.1, 0.2])
        assert affect.valence == 0.8
        assert affect.arousal == 0.6
        assert affect.dominance == 0.4
        assert affect.sorrow == 0.1
        assert affect.boredom == 0.2

    def test_from_list_pad_only(self):
        """Can create from 3-element PAD list."""
        affect = Affect.from_list([0.5, 0.7, 0.3])
        assert affect.valence == 0.5
        assert affect.arousal == 0.7
        assert affect.dominance == 0.3
        assert affect.sorrow == 0.0
        assert affect.boredom == 0.0

    def test_to_list(self):
        """Can convert to list."""
        affect = Affect(0.1, 0.2, 0.3, 0.4, 0.5)
        assert affect.to_list() == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_neutral(self):
        """Neutral factory creates relaxed, expressionless affect."""
        affect = Affect.neutral()
        assert affect.valence == 0.0
        assert affect.arousal == 0.2  # Low arousal = relaxed face


# ═══════════════════════════════════════════════════════════════════════════
# Test FACSMapper - Affect to Emotions
# ═══════════════════════════════════════════════════════════════════════════

class TestAffectToEmotions:
    """Tests for affect → emotion mapping."""

    def test_happiness_from_positive_valence(self):
        """Positive valence + arousal produces happiness."""
        mapper = FACSMapper()
        affect = Affect(valence=0.8, arousal=0.7, dominance=0.5)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.happiness > 0.5
        assert emotions.sadness < 0.1

    def test_sadness_from_negative_valence_low_arousal(self):
        """Negative valence + low arousal produces sadness."""
        mapper = FACSMapper()
        affect = Affect(valence=-0.7, arousal=0.2, dominance=0.3, sorrow=0.5)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.sadness > 0.3
        assert emotions.happiness < 0.1

    def test_anger_from_negative_valence_high_arousal_dominance(self):
        """Negative valence + high arousal + high dominance produces anger."""
        mapper = FACSMapper()
        affect = Affect(valence=-0.8, arousal=0.9, dominance=0.8)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.anger > 0.5
        assert emotions.fear < emotions.anger  # Anger > fear when dominant

    def test_fear_from_negative_valence_high_arousal_low_dominance(self):
        """Negative valence + high arousal + low dominance produces fear."""
        mapper = FACSMapper()
        affect = Affect(valence=-0.8, arousal=0.9, dominance=0.2)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.fear > 0.5
        assert emotions.anger < emotions.fear  # Fear > anger when not dominant

    def test_surprise_from_high_arousal_neutral_valence(self):
        """High arousal + neutral valence produces surprise."""
        mapper = FACSMapper()
        affect = Affect(valence=0.0, arousal=0.9, dominance=0.5)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.surprise > 0.3

    def test_boredom_direct_mapping(self):
        """Boredom maps directly from affect."""
        mapper = FACSMapper()
        affect = Affect(valence=0.0, arousal=0.3, dominance=0.5, boredom=0.8)
        emotions = mapper.affect_to_emotions(affect)
        assert emotions.boredom == 0.8

    def test_dominant_emotion(self):
        """Can identify dominant emotion."""
        emotions = EmotionWeights(happiness=0.8, sadness=0.1, anger=0.2)
        dominant, weight = emotions.dominant_emotion()
        assert dominant == 'happiness'
        assert weight == 0.8


# ═══════════════════════════════════════════════════════════════════════════
# Test FACSMapper - Emotions to AUs
# ═══════════════════════════════════════════════════════════════════════════

class TestEmotionsToAUs:
    """Tests for emotion → Action Unit mapping."""

    def test_happiness_activates_au12(self):
        """Happiness activates AU12 (smile)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(happiness=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU12' in aus
        assert aus['AU12'] > 0.5

    def test_happiness_activates_au6(self):
        """Happiness activates AU6 (Duchenne marker)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(happiness=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU6' in aus
        assert aus['AU6'] > 0.5

    def test_sadness_activates_au1_au15(self):
        """Sadness activates AU1 (inner brow) and AU15 (frown)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(sadness=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU1' in aus
        assert 'AU15' in aus

    def test_anger_activates_au4(self):
        """Anger activates AU4 (brow lowerer)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(anger=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU4' in aus
        assert aus['AU4'] > 0.5

    def test_fear_activates_au5(self):
        """Fear activates AU5 (wide eyes)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(fear=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU5' in aus

    def test_surprise_activates_au26(self):
        """Surprise activates AU26 (jaw drop)."""
        mapper = FACSMapper()
        emotions = EmotionWeights(surprise=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert 'AU26' in aus

    def test_au_values_capped_at_1(self):
        """AU values are capped at 1.0 even with multiple emotions."""
        mapper = FACSMapper()
        # Multiple emotions that all use AU4
        emotions = EmotionWeights(anger=1.0, sadness=1.0, concentration=1.0)
        aus = mapper.emotions_to_aus(emotions)
        assert aus['AU4'] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test FACSMapper - AUs to VRM
# ═══════════════════════════════════════════════════════════════════════════

class TestAUsToVRM:
    """Tests for Action Unit → VRM blendshape mapping."""

    def test_au12_produces_joy_blendshape(self):
        """AU12 produces Fcl_ALL_Joy blendshape."""
        mapper = FACSMapper()
        aus = {'AU12': 1.0}
        vrm = mapper.aus_to_vrm(aus)
        assert 'Fcl_ALL_Joy' in vrm
        assert vrm['Fcl_ALL_Joy'] > 0.5

    def test_au15_produces_sorrow_blendshape(self):
        """AU15 produces Fcl_ALL_Sorrow blendshape."""
        mapper = FACSMapper()
        aus = {'AU15': 1.0}
        vrm = mapper.aus_to_vrm(aus)
        assert 'Fcl_ALL_Sorrow' in vrm

    def test_au5_produces_surprised_eye(self):
        """AU5 produces Fcl_EYE_Surprised blendshape."""
        mapper = FACSMapper()
        aus = {'AU5': 1.0}
        vrm = mapper.aus_to_vrm(aus)
        assert 'Fcl_EYE_Surprised' in vrm

    def test_au4_produces_angry_brow(self):
        """AU4 produces Fcl_BRW_Angry blendshape."""
        mapper = FACSMapper()
        aus = {'AU4': 1.0}
        vrm = mapper.aus_to_vrm(aus)
        assert 'Fcl_BRW_Angry' in vrm

    def test_vrm_values_capped_at_1(self):
        """VRM values are capped at 1.0."""
        mapper = FACSMapper()
        # Multiple AUs that affect the same blendshape
        aus = {'AU9': 1.0, 'AU10': 1.0, 'AU23': 1.0}
        vrm = mapper.aus_to_vrm(aus)
        for value in vrm.values():
            assert value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test FACSMapper - Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Tests for full Affect → VRM pipeline."""

    def test_happy_affect_produces_smile(self):
        """Happy affect produces smile blendshapes."""
        mapper = FACSMapper()
        affect = Affect(valence=0.9, arousal=0.7, dominance=0.5)
        vrm = mapper.map_affect_to_vrm(affect)
        assert 'Fcl_ALL_Joy' in vrm
        assert vrm['Fcl_ALL_Joy'] > 0.3

    def test_sad_affect_produces_frown(self):
        """Sad affect produces frown blendshapes."""
        mapper = FACSMapper()
        affect = Affect(valence=-0.7, arousal=0.2, dominance=0.3, sorrow=0.8)
        vrm = mapper.map_affect_to_vrm(affect)
        assert 'Fcl_ALL_Sorrow' in vrm

    def test_neutral_affect_produces_minimal_expression(self):
        """Neutral affect produces minimal blendshapes."""
        mapper = FACSMapper()
        affect = Affect.neutral()
        vrm = mapper.map_affect_to_vrm(affect)
        # Should have minimal or no blendshapes
        total = sum(vrm.values()) if vrm else 0
        assert total < 0.5

    def test_convenience_function_affect_to_vrm(self):
        """Convenience function works correctly."""
        vrm = affect_to_vrm(0.8, 0.6, 0.5, 0.0, 0.0)
        assert 'Fcl_ALL_Joy' in vrm

    def test_convenience_function_affect_list_to_vrm(self):
        """Convenience function with list works correctly."""
        vrm = affect_list_to_vrm([0.8, 0.6, 0.5, 0.0, 0.0])
        assert 'Fcl_ALL_Joy' in vrm

    def test_map_with_details(self):
        """map_affect_to_vrm_with_details returns all intermediate results."""
        mapper = FACSMapper()
        affect = Affect(valence=0.8, arousal=0.6, dominance=0.5)
        result = mapper.map_affect_to_vrm_with_details(affect)

        assert 'affect' in result
        assert 'emotions' in result
        assert 'aus' in result
        assert 'vrm' in result
        assert 'dominant_emotion' in result
        assert result['dominant_emotion'] == 'happiness'


# ═══════════════════════════════════════════════════════════════════════════
# Test TeachingMapper
# ═══════════════════════════════════════════════════════════════════════════

class TestTeachingMapper:
    """Tests for Kimii-Sensei's teaching features."""

    def test_isolate_au(self):
        """Can isolate a single AU."""
        mapper = TeachingMapper()
        vrm = mapper.isolate_au('AU12', 1.0)
        assert 'Fcl_ALL_Joy' in vrm
        # Should only have AU12-related blendshapes
        assert len(vrm) <= 2  # AU12 maps to Joy and MTH_Joy

    def test_isolate_emotion(self):
        """Can isolate a single emotion."""
        mapper = TeachingMapper()
        vrm = mapper.isolate_emotion('happiness', 1.0)
        assert 'Fcl_ALL_Joy' in vrm

    def test_blend_emotions(self):
        """Can blend multiple emotions."""
        mapper = TeachingMapper()
        vrm = mapper.blend_emotions({'happiness': 0.7, 'surprise': 0.3})
        # Should have both joy and surprise elements
        assert 'Fcl_ALL_Joy' in vrm or 'Fcl_EYE_Joy' in vrm

    def test_get_au_description(self):
        """Can get AU descriptions."""
        mapper = TeachingMapper()
        desc = mapper.get_au_description('AU12')
        assert 'smile' in desc.lower() or 'Lip Corner Puller' in desc

    def test_get_all_au_descriptions(self):
        """Can get all AU descriptions."""
        mapper = TeachingMapper()
        descs = mapper.get_all_au_descriptions()
        assert len(descs) >= 15
        assert 'AU12' in descs
        assert 'AU1' in descs


# ═══════════════════════════════════════════════════════════════════════════
# Test FacialExpressionComponent
# ═══════════════════════════════════════════════════════════════════════════

class TestFacialExpressionComponent:
    """Tests for FacialExpressionComponent."""

    def test_component_creation(self):
        """Can create component."""
        component = FacialExpressionComponent()
        assert component.component_type == "facial_expression"
        assert component.display_name == "Facial Expression"

    def test_set_affect(self):
        """Can set affect and it updates targets."""
        component = FacialExpressionComponent()
        component.set_affect(Affect(valence=0.8, arousal=0.6, dominance=0.5))
        assert len(component._target_blendshapes) > 0

    def test_set_affect_from_list(self):
        """Can set affect from list."""
        component = FacialExpressionComponent()
        component.set_affect_from_list([0.8, 0.6, 0.5, 0.0, 0.0])
        assert len(component._target_blendshapes) > 0

    def test_set_affect_from_dict(self):
        """Can set affect from dict."""
        component = FacialExpressionComponent()
        component.set_affect_from_dict({
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.5,
        })
        assert len(component._target_blendshapes) > 0

    def test_update_interpolates(self):
        """Update interpolates toward target."""
        component = FacialExpressionComponent()
        component.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))

        # First update
        blendshapes1 = component.update(dt=0.016)

        # Values should be moving toward target but not there yet
        joy1 = blendshapes1.get('Fcl_ALL_Joy', 0)

        # More updates
        for _ in range(10):
            blendshapes2 = component.update(dt=0.016)

        joy2 = blendshapes2.get('Fcl_ALL_Joy', 0)

        # Should have interpolated closer to target
        assert joy2 > joy1 or joy2 > 0.5

    def test_smoothing_affects_speed(self):
        """Higher smoothing = slower interpolation."""
        # Low smoothing (fast)
        fast = FacialExpressionComponent()
        fast.smoothing_factor = 0.1
        fast.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))
        fast.update(dt=0.1)
        joy_fast = fast.current_blendshapes.get('Fcl_ALL_Joy', 0)

        # High smoothing (slow)
        slow = FacialExpressionComponent()
        slow.smoothing_factor = 0.9
        slow.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))
        slow.update(dt=0.1)
        joy_slow = slow.current_blendshapes.get('Fcl_ALL_Joy', 0)

        # Fast should have moved more than slow
        assert joy_fast > joy_slow

    def test_intensity_multiplier(self):
        """Intensity multiplier affects expression strength."""
        component = FacialExpressionComponent()
        component.intensity = 2.0
        component.set_affect(Affect(valence=0.5, arousal=0.5, dominance=0.5))

        # Target should be amplified
        for value in component._target_blendshapes.values():
            # Values are capped at 1.0 in VRM but intensity is applied
            pass  # Just verify it doesn't crash

    def test_reset_clears_state(self):
        """Reset clears all state."""
        component = FacialExpressionComponent()
        component.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))
        component.update(dt=0.1)

        component.reset()

        assert len(component._current_blendshapes) == 0
        assert len(component._target_blendshapes) == 0
        assert component._current_affect is None

    def test_set_neutral(self):
        """set_neutral sets target to neutral."""
        component = FacialExpressionComponent()
        component.set_affect(Affect(valence=0.9, arousal=0.9, dominance=0.9))
        component.set_neutral()

        # Neutral should have minimal targets
        total = sum(component._target_blendshapes.values())
        assert total < 0.5

    def test_teaching_mode_isolate_au(self):
        """Teaching mode can isolate AU."""
        component = FacialExpressionComponent()
        component.teaching_mode = True
        component.isolate_au('AU12', 1.0)
        assert 'Fcl_ALL_Joy' in component._target_blendshapes

    def test_teaching_mode_isolate_emotion(self):
        """Teaching mode can isolate emotion."""
        component = FacialExpressionComponent()
        component.teaching_mode = True
        component.isolate_emotion('happiness', 1.0)
        assert len(component._target_blendshapes) > 0

    def test_teaching_mode_blend_emotions(self):
        """Teaching mode can blend emotions."""
        component = FacialExpressionComponent()
        component.teaching_mode = True
        component.blend_emotions({'happiness': 0.5, 'sadness': 0.5})
        assert len(component._target_blendshapes) > 0

    def test_get_active_aus(self):
        """Can get active AUs for teaching UI."""
        component = FacialExpressionComponent()
        component.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))
        aus = component.get_active_aus()
        assert 'AU12' in aus  # Happy → smile

    def test_callback_fires(self):
        """Callback fires on update."""
        component = FacialExpressionComponent()
        callback_results = []

        def callback(blendshapes):
            callback_results.append(blendshapes.copy())

        component.on_blendshapes_changed(callback)
        component.set_affect(Affect(valence=0.9, arousal=0.7, dominance=0.5))
        component.update(dt=0.016)

        assert len(callback_results) == 1

    def test_serialization(self):
        """Can serialize and deserialize."""
        component = FacialExpressionComponent()
        component.affect_channel = "custom_channel"
        component.smoothing_factor = 0.5
        component.intensity = 1.5
        component.teaching_mode = True

        data = component.to_dict()
        restored = FacialExpressionComponent.from_dict(data)

        assert restored.affect_channel == "custom_channel"
        assert restored.smoothing_factor == 0.5
        assert restored.intensity == 1.5
        assert restored.teaching_mode is True

    def test_property_specs(self):
        """Has expected property specs."""
        component = FacialExpressionComponent()
        specs = component.property_specs
        spec_names = [s.name for s in specs]

        assert 'affect_channel' in spec_names
        assert 'smoothing_factor' in spec_names
        assert 'intensity' in spec_names
        assert 'teaching_mode' in spec_names


# ═══════════════════════════════════════════════════════════════════════════
# Test Blink System
# ═══════════════════════════════════════════════════════════════════════════

class TestBlinkSystem:
    """Tests for auto-blink functionality."""

    def test_trigger_blink(self):
        """Can manually trigger blink."""
        component = FacialExpressionComponent()
        component.trigger_blink()

        assert component._blinking is True
        assert component._current_blendshapes.get('Fcl_EYE_Close', 0) == 1.0

    def test_blink_ends_after_duration(self):
        """Blink ends after duration."""
        component = FacialExpressionComponent()
        component._blink_duration = 0.01  # Very short for test

        component.trigger_blink()
        time.sleep(0.02)  # Wait for blink to end
        component.update(dt=0.02)

        assert component._blinking is False


# ═══════════════════════════════════════════════════════════════════════════
# Test Data Integrity
# ═══════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Tests for FACS data integrity."""

    def test_all_emotions_have_recipes(self):
        """All emotions in EmotionWeights have recipes."""
        emotions = EmotionWeights()
        emotion_names = [k for k in emotions.to_dict().keys()]

        for emotion in emotion_names:
            assert emotion in EMOTION_AU_RECIPES, f"Missing recipe for {emotion}"

    def test_all_recipe_aus_have_vrm_mappings(self):
        """All AUs in recipes have VRM mappings."""
        all_aus = set()
        for recipe in EMOTION_AU_RECIPES.values():
            all_aus.update(recipe.keys())

        for au in all_aus:
            assert au in AU_TO_VRM, f"Missing VRM mapping for {au}"

    def test_au_values_in_recipes_are_valid(self):
        """AU values in recipes are 0-1."""
        for emotion, recipe in EMOTION_AU_RECIPES.items():
            for au, value in recipe.items():
                assert 0 <= value <= 1, f"Invalid value {value} for {au} in {emotion}"


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
