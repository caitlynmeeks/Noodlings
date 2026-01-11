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
#   Facial Expression Component - Drive VRM avatar expressions from affect
#
#   Subscribes to affect channels and maps through FACS to VRM blendshapes.
#   Includes smoothing, auto-blink, and micro-expressions.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.facial_expression_component
# PURPOSE:  Facial Expression Component
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import random
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from .component_base import ComponentBase, ComponentCategory, PropertySpec
from ..runtime.facs_mapper import FACSMapper, Affect, TeachingMapper

logger = logging.getLogger(__name__)


class FacialExpressionComponent(ComponentBase):
    """
    Drives VRM avatar facial expressions from affect channels.

    Subscribes to the noodling's affect output and maps it through
    FACS (Facial Action Coding System) to VRM blendshapes.

    Features:
    - Smooth interpolation (no jittery expressions)
    - Auto-blink at natural intervals
    - Micro-expressions (brief involuntary emotion leaks)
    - Teaching mode (for Kimii-Sensei)
    """

    def __init__(self, entity_id: str = ""):
        super().__init__(entity_id)

        # Configuration
        self._affect_channel: str = "affect"
        self._smoothing_factor: float = 0.3  # 0 = instant, 1 = very smooth
        self._intensity: float = 1.0

        # Micro-expressions
        self._enable_micro_expressions: bool = True
        self._micro_expression_probability: float = 0.05
        self._micro_expression_duration: float = 0.1  # seconds

        # Blink settings
        self._enable_auto_blink: bool = True
        self._blink_interval_mean: float = 4.0  # seconds
        self._blink_interval_variance: float = 1.0
        self._blink_duration: float = 0.15  # seconds

        # Teaching mode (for Kimii-Sensei)
        self._teaching_mode: bool = False
        self._au_isolation_enabled: bool = False
        self._highlight_active_aus: bool = False
        self._transition_speed: float = 1.0  # Slower for teaching

        # State
        self._current_blendshapes: Dict[str, float] = {}
        self._target_blendshapes: Dict[str, float] = {}
        self._last_update_time: float = time.time()

        # Blink state
        self._blinking: bool = False
        self._blink_end_time: float = 0
        self._next_blink_time: float = time.time() + random.gauss(4.0, 1.0)

        # Micro-expression state
        self._micro_expression_active: bool = False
        self._micro_expression_emotion: str = ""
        self._micro_expression_end_time: float = 0

        # Affect state
        self._current_affect: Optional[Affect] = None
        self._suppressed_emotion: Optional[str] = None  # For micro-expression leaks

        # Mapper
        self._mapper: FACSMapper = FACSMapper()
        self._teaching_mapper: TeachingMapper = TeachingMapper()

        # Callbacks
        self._on_blendshapes_changed: Optional[Callable[[Dict[str, float]], None]] = None

    # ═══════════════════════════════════════════════════════════
    # ComponentBase Implementation
    # ═══════════════════════════════════════════════════════════

    @property
    def component_type(self) -> str:
        return "facial_expression"

    @property
    def display_name(self) -> str:
        return "Facial Expression"

    @property
    def category(self) -> ComponentCategory:
        return ComponentCategory.RENDERING

    @property
    def property_specs(self) -> List[PropertySpec]:
        return [
            PropertySpec(
                name="affect_channel",
                display_name="Affect Channel",
                property_type="string",
                default="affect",
                description="Channel to subscribe for affect updates"
            ),
            PropertySpec(
                name="smoothing_factor",
                display_name="Smoothing",
                property_type="float",
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                description="Expression smoothing (0=instant, 1=very smooth)"
            ),
            PropertySpec(
                name="intensity",
                display_name="Intensity",
                property_type="float",
                default=1.0,
                min_value=0.0,
                max_value=2.0,
                description="Expression intensity multiplier"
            ),
            PropertySpec(
                name="enable_auto_blink",
                display_name="Auto Blink",
                property_type="bool",
                default=True,
                description="Enable automatic blinking"
            ),
            PropertySpec(
                name="blink_interval_mean",
                display_name="Blink Interval",
                property_type="float",
                default=4.0,
                min_value=1.0,
                max_value=10.0,
                description="Average seconds between blinks"
            ),
            PropertySpec(
                name="enable_micro_expressions",
                display_name="Micro-Expressions",
                property_type="bool",
                default=True,
                description="Enable brief involuntary expression leaks"
            ),
            PropertySpec(
                name="teaching_mode",
                display_name="Teaching Mode",
                property_type="bool",
                default=False,
                description="Enable teaching features (Kimii-Sensei)"
            ),
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.component_type,
            "id": self._id,
            "enabled": self._enabled,
            "affect_channel": self._affect_channel,
            "smoothing_factor": self._smoothing_factor,
            "intensity": self._intensity,
            "enable_auto_blink": self._enable_auto_blink,
            "blink_interval_mean": self._blink_interval_mean,
            "blink_interval_variance": self._blink_interval_variance,
            "blink_duration": self._blink_duration,
            "enable_micro_expressions": self._enable_micro_expressions,
            "micro_expression_probability": self._micro_expression_probability,
            "micro_expression_duration": self._micro_expression_duration,
            "teaching_mode": self._teaching_mode,
            "au_isolation_enabled": self._au_isolation_enabled,
            "highlight_active_aus": self._highlight_active_aus,
            "transition_speed": self._transition_speed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FacialExpressionComponent':
        component = cls(entity_id=data.get('entity_id', ''))
        component._id = data.get('id', component._id)
        component._enabled = data.get('enabled', True)
        component._affect_channel = data.get('affect_channel', 'affect')
        component._smoothing_factor = data.get('smoothing_factor', 0.3)
        component._intensity = data.get('intensity', 1.0)
        component._enable_auto_blink = data.get('enable_auto_blink', True)
        component._blink_interval_mean = data.get('blink_interval_mean', 4.0)
        component._blink_interval_variance = data.get('blink_interval_variance', 1.0)
        component._blink_duration = data.get('blink_duration', 0.15)
        component._enable_micro_expressions = data.get('enable_micro_expressions', True)
        component._micro_expression_probability = data.get('micro_expression_probability', 0.05)
        component._micro_expression_duration = data.get('micro_expression_duration', 0.1)
        component._teaching_mode = data.get('teaching_mode', False)
        component._au_isolation_enabled = data.get('au_isolation_enabled', False)
        component._highlight_active_aus = data.get('highlight_active_aus', False)
        component._transition_speed = data.get('transition_speed', 1.0)
        return component

    # ═══════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════

    @property
    def affect_channel(self) -> str:
        return self._affect_channel

    @affect_channel.setter
    def affect_channel(self, value: str):
        self._affect_channel = value
        self._dirty = True

    @property
    def smoothing_factor(self) -> float:
        return self._smoothing_factor

    @smoothing_factor.setter
    def smoothing_factor(self, value: float):
        self._smoothing_factor = max(0.0, min(1.0, value))
        self._dirty = True

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, value)
        self._dirty = True

    @property
    def teaching_mode(self) -> bool:
        return self._teaching_mode

    @teaching_mode.setter
    def teaching_mode(self, value: bool):
        self._teaching_mode = value
        self._dirty = True

    @property
    def current_blendshapes(self) -> Dict[str, float]:
        """Get current blendshape values (read-only)."""
        return self._current_blendshapes.copy()

    @property
    def current_affect(self) -> Optional[Affect]:
        """Get current affect state."""
        return self._current_affect

    # ═══════════════════════════════════════════════════════════
    # Main Update Loop
    # ═══════════════════════════════════════════════════════════

    def update(self, dt: Optional[float] = None) -> Dict[str, float]:
        """
        Update facial expression state.

        Call this every frame to smoothly animate expressions.

        Args:
            dt: Delta time in seconds (computed if not provided)

        Returns:
            Current blendshape values
        """
        if not self._enabled:
            return self._current_blendshapes

        current_time = time.time()
        if dt is None:
            dt = current_time - self._last_update_time
        self._last_update_time = current_time

        # Clamp dt to prevent huge jumps
        dt = min(dt, 0.1)

        # Update blink
        if self._enable_auto_blink:
            self._update_blink(current_time)

        # Update micro-expression
        if self._enable_micro_expressions:
            self._update_micro_expression(current_time)

        # Smooth interpolation toward target
        self._interpolate_blendshapes(dt)

        # Notify callback
        if self._on_blendshapes_changed:
            self._on_blendshapes_changed(self._current_blendshapes)

        return self._current_blendshapes

    def set_affect(self, affect: Affect):
        """
        Set new affect state and compute target blendshapes.

        Args:
            affect: New 5D affect state
        """
        self._current_affect = affect

        # Map affect → VRM blendshapes
        self._target_blendshapes = self._mapper.map_affect_to_vrm(affect)

        # Apply intensity multiplier
        for shape in self._target_blendshapes:
            self._target_blendshapes[shape] *= self._intensity

        # Track dominant emotion for micro-expression system
        emotions = self._mapper.affect_to_emotions(affect)
        dominant, _ = emotions.dominant_emotion()
        self._suppressed_emotion = dominant

    def set_affect_from_list(self, affect_list: List[float]):
        """
        Set affect from list [valence, arousal, dominance, sorrow, boredom].

        Args:
            affect_list: 5D affect values
        """
        affect = Affect.from_list(affect_list)
        self.set_affect(affect)

    def set_affect_from_dict(self, affect_dict: Dict[str, float]):
        """
        Set affect from dictionary.

        Args:
            affect_dict: Dict with valence, arousal, dominance, sorrow, boredom
        """
        affect = Affect(
            valence=affect_dict.get('valence', 0.0),
            arousal=affect_dict.get('arousal', 0.5),
            dominance=affect_dict.get('dominance', 0.5),
            sorrow=affect_dict.get('sorrow', 0.0),
            boredom=affect_dict.get('boredom', 0.0),
        )
        self.set_affect(affect)

    # ═══════════════════════════════════════════════════════════
    # Blink System
    # ═══════════════════════════════════════════════════════════

    def _update_blink(self, current_time: float):
        """Update auto-blink state."""
        # Check if it's time to blink
        if not self._blinking and current_time >= self._next_blink_time:
            self._start_blink(current_time)

        # Check if blink is done
        if self._blinking and current_time >= self._blink_end_time:
            self._end_blink(current_time)

    def _start_blink(self, current_time: float):
        """Start a blink."""
        self._blinking = True
        self._blink_end_time = current_time + self._blink_duration

        # Apply blink blendshape
        self._current_blendshapes['Fcl_EYE_Close'] = 1.0

    def _end_blink(self, current_time: float):
        """End a blink and schedule next one."""
        self._blinking = False

        # Remove blink blendshape (will be smoothed out)
        if 'Fcl_EYE_Close' in self._target_blendshapes:
            # Keep target if expression wants eyes closed
            pass
        else:
            self._current_blendshapes['Fcl_EYE_Close'] = 0.0

        # Schedule next blink
        interval = random.gauss(self._blink_interval_mean, self._blink_interval_variance)
        self._next_blink_time = current_time + max(1.0, interval)

    def trigger_blink(self):
        """Manually trigger a blink."""
        self._start_blink(time.time())

    # ═══════════════════════════════════════════════════════════
    # Micro-Expression System
    # ═══════════════════════════════════════════════════════════

    def _update_micro_expression(self, current_time: float):
        """Update micro-expression state."""
        # Check if micro-expression is ending
        if self._micro_expression_active:
            if current_time >= self._micro_expression_end_time:
                self._end_micro_expression()
            return

        # Maybe trigger a new micro-expression
        if self._suppressed_emotion and random.random() < self._micro_expression_probability * 0.016:
            self._start_micro_expression(current_time)

    def _start_micro_expression(self, current_time: float):
        """Start a micro-expression (brief emotion leak)."""
        self._micro_expression_active = True
        self._micro_expression_emotion = self._suppressed_emotion
        self._micro_expression_end_time = current_time + self._micro_expression_duration

        # Flash the suppressed emotion at full intensity
        if self._suppressed_emotion:
            emotion_blendshapes = self._teaching_mapper.isolate_emotion(
                self._suppressed_emotion, 0.8
            )
            # Merge with current (additive)
            for shape, value in emotion_blendshapes.items():
                self._current_blendshapes[shape] = min(1.0,
                    self._current_blendshapes.get(shape, 0) + value)

        logger.debug(f"[FacialExpression] Micro-expression: {self._suppressed_emotion}")

    def _end_micro_expression(self):
        """End a micro-expression."""
        self._micro_expression_active = False
        self._micro_expression_emotion = ""

    # ═══════════════════════════════════════════════════════════
    # Interpolation
    # ═══════════════════════════════════════════════════════════

    def _interpolate_blendshapes(self, dt: float):
        """
        Smoothly interpolate current blendshapes toward target.

        Uses exponential smoothing for natural movement.
        """
        # Compute alpha based on smoothing factor and teaching mode
        base_rate = 1.0 - self._smoothing_factor
        if self._teaching_mode:
            base_rate *= self._transition_speed

        alpha = 1.0 - math.exp(-dt * 10.0 * base_rate)

        # Get all blendshapes (union of current and target)
        all_shapes = set(self._current_blendshapes.keys()) | set(self._target_blendshapes.keys())

        for shape in all_shapes:
            current = self._current_blendshapes.get(shape, 0.0)
            target = self._target_blendshapes.get(shape, 0.0)

            # Skip if blink is active and this is the eye close shape
            if self._blinking and shape == 'Fcl_EYE_Close':
                continue

            # Interpolate
            new_value = current + alpha * (target - current)

            # Clean up near-zero values
            if abs(new_value) < 0.001:
                if shape in self._current_blendshapes:
                    del self._current_blendshapes[shape]
            else:
                self._current_blendshapes[shape] = new_value

    # ═══════════════════════════════════════════════════════════
    # Teaching Mode (Kimii-Sensei)
    # ═══════════════════════════════════════════════════════════

    def isolate_au(self, au: str, intensity: float = 1.0):
        """
        Demonstrate a single Action Unit in isolation.

        For teaching: "This is what AU12 (smile) looks like alone."

        Args:
            au: Action Unit name (e.g., 'AU12')
            intensity: How strong (0-1)
        """
        self._target_blendshapes = self._teaching_mapper.isolate_au(au, intensity)

    def isolate_emotion(self, emotion: str, intensity: float = 1.0):
        """
        Demonstrate a single emotion.

        For teaching: "This is what pure happiness looks like."

        Args:
            emotion: Emotion name (e.g., 'happiness')
            intensity: How strong (0-1)
        """
        self._target_blendshapes = self._teaching_mapper.isolate_emotion(emotion, intensity)

    def blend_emotions(self, emotion_weights: Dict[str, float]):
        """
        Demonstrate an emotion blend.

        For teaching: "What does 70% happy + 30% surprised look like?"

        Args:
            emotion_weights: Dict mapping emotion names to weights
        """
        self._target_blendshapes = self._teaching_mapper.blend_emotions(emotion_weights)

    def get_active_aus(self) -> Dict[str, float]:
        """
        Get currently active Action Units.

        For teaching UI to highlight which AUs are firing.
        """
        if not self._current_affect:
            return {}

        emotions = self._mapper.affect_to_emotions(self._current_affect)
        return self._mapper.emotions_to_aus(emotions)

    def get_au_description(self, au: str) -> str:
        """Get human-readable description of an Action Unit."""
        return self._teaching_mapper.get_au_description(au)

    # ═══════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════

    def reset(self):
        """Reset to neutral expression."""
        self._current_blendshapes = {}
        self._target_blendshapes = {}
        self._current_affect = None
        self._suppressed_emotion = None
        self._blinking = False
        self._micro_expression_active = False

    def set_neutral(self):
        """Set target to neutral expression."""
        self.set_affect(Affect.neutral())

    # ═══════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════

    def on_blendshapes_changed(self, callback: Callable[[Dict[str, float]], None]):
        """
        Register callback for blendshape changes.

        The callback receives the current blendshape dict on each update.

        Args:
            callback: Function(blendshapes: Dict[str, float]) -> None
        """
        self._on_blendshapes_changed = callback


# ═══════════════════════════════════════════════════════════════════════════
# Component Registration
# ═══════════════════════════════════════════════════════════════════════════

def register_facial_expression_component():
    """Register FacialExpressionComponent with the component registry."""
    from .component_base import component_registry
    component_registry.register(FacialExpressionComponent)


# ═══════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=== FacialExpressionComponent Test ===\n")

    component = FacialExpressionComponent()

    # Set happy affect
    component.set_affect(Affect(valence=0.8, arousal=0.6, dominance=0.5))

    print("Initial target blendshapes:")
    print(component._target_blendshapes)

    # Simulate a few frames
    print("\nSimulating 10 frames...")
    for i in range(10):
        blendshapes = component.update(dt=0.016)
        if i % 3 == 0:
            joy = blendshapes.get('Fcl_ALL_Joy', 0)
            print(f"  Frame {i}: Fcl_ALL_Joy = {joy:.3f}")

    print("\nFinal blendshapes:")
    print(component.current_blendshapes)

    # Test teaching mode
    print("\n--- Teaching Mode ---")
    component.teaching_mode = True
    component.isolate_au('AU12', 1.0)
    print(f"AU12 isolated: {component._target_blendshapes}")

    component.blend_emotions({'happiness': 0.7, 'surprise': 0.3})
    print(f"70% happy + 30% surprised: {component._target_blendshapes}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
