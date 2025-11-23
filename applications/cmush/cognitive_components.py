"""
Cognitive Manifold & Transistor Implementation

Modular cognitive processing architecture where belief systems
(transistors) filter thoughts and manifolds integrate outputs.

Author: Commander Spock + Lieutenant Caitlyn
Date: November 22, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import time
import random

logger = logging.getLogger(__name__)


@dataclass
class TransistorOutput:
    """Output from a cognitive transistor."""
    transformed_text: str  # Colored/filtered thought
    salience: float        # Importance weight (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional info


class CognitiveTransistor(ABC):
    """
    Base class for cognitive transistors.

    Transistors receive input and color it based on internal state
    (beliefs, personality, mood, etc.). Each outputs a transformed
    version of the input with a salience weight.
    """

    def __init__(self):
        """Initialize transistor."""
        self.salience = 0.5  # Default importance (0.0 to 1.0)
        self.enabled = True  # Can be toggled off

    @abstractmethod
    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Process input through cognitive filter.

        Args:
            input_text: Raw perception/thought
            context: Additional context (affect, memories, etc.)

        Returns:
            TransistorOutput with transformed text and salience
        """
        pass

    def get_transistor_type(self) -> str:
        """Return transistor type identifier."""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            'type': self.get_transistor_type(),
            'salience': self.salience,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveTransistor':
        """Deserialize from dictionary."""
        instance = cls()
        instance.salience = data.get('salience', 0.5)
        instance.enabled = data.get('enabled', True)
        return instance


class CognitiveManifold:
    """
    Cognitive Manifold - Integrates multiple transistor outputs.

    Receives output from all registered transistors and synthesizes
    a coherent thought using LLM-powered blending.
    """

    def __init__(self, blending_strategy: str = "llm_weighted"):
        """
        Initialize manifold.

        Args:
            blending_strategy: "llm_weighted", "simple_concat", or "priority"
        """
        self.transistors: List[CognitiveTransistor] = []
        self.blending_strategy = blending_strategy

    def register_transistor(self, transistor: CognitiveTransistor):
        """
        Register a transistor to integrate.

        Args:
            transistor: CognitiveTransistor instance
        """
        self.transistors.append(transistor)
        logger.info(f"Registered transistor: {transistor.get_transistor_type()}")

    def unregister_transistor(self, transistor: CognitiveTransistor):
        """Remove transistor from integration."""
        if transistor in self.transistors:
            self.transistors.remove(transistor)

    def integrate(self, input_text: str, context: Dict[str, Any]) -> str:
        """
        Integrate all transistor outputs into coherent thought.

        Args:
            input_text: Raw perception/thought
            context: Additional context (affect, memories, etc.)

        Returns:
            Synthesized coherent thought
        """
        # Collect outputs from all enabled transistors
        outputs = []
        for transistor in self.transistors:
            if transistor.enabled:
                try:
                    output = transistor.process(input_text, context)
                    outputs.append(output)
                except Exception as e:
                    logger.error(f"Transistor {transistor.get_transistor_type()} failed: {e}")

        # No transistors = pass through
        if not outputs:
            return input_text

        # Synthesize using configured strategy
        if self.blending_strategy == "llm_weighted":
            return self._llm_weighted_blend(outputs, context)
        elif self.blending_strategy == "simple_concat":
            return self._simple_concatenation(outputs)
        elif self.blending_strategy == "priority":
            return self._priority_blend(outputs)
        else:
            return input_text

    def _llm_weighted_blend(
        self,
        outputs: List[TransistorOutput],
        context: Dict[str, Any]
    ) -> str:
        """
        Use LLM to blend multiple perspectives.

        Args:
            outputs: List of transistor outputs
            context: Additional context

        Returns:
            Synthesized thought
        """
        # Build prompt
        prompt = "Synthesize these cognitive perspectives into ONE coherent thought:\n\n"

        for i, output in enumerate(outputs, 1):
            prompt += f"{i}. [salience={output.salience:.2f}] {output.transformed_text}\n"

        prompt += "\nIntegrate all perspectives proportionally to salience. "
        prompt += "Higher salience = more influence. Response (one coherent sentence):"

        # Call LLM (would integrate with llm_interface.py)
        # For now, simple weighted concatenation
        return self._simple_concatenation(outputs)

    def _simple_concatenation(self, outputs: List[TransistorOutput]) -> str:
        """Simple concatenation weighted by salience."""
        # Sort by salience (highest first)
        sorted_outputs = sorted(outputs, key=lambda x: x.salience, reverse=True)

        # Concatenate high-salience outputs
        parts = [o.transformed_text for o in sorted_outputs if o.salience > 0.3]

        if not parts:
            return sorted_outputs[0].transformed_text if sorted_outputs else ""

        return " ".join(parts)

    def _priority_blend(self, outputs: List[TransistorOutput]) -> str:
        """Use only highest salience output."""
        if not outputs:
            return ""
        highest = max(outputs, key=lambda x: x.salience)
        return highest.transformed_text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'CognitiveManifold',
            'blending_strategy': self.blending_strategy,
            'transistors': [t.to_dict() for t in self.transistors]
        }


# ===== Concrete Transistor Implementations =====

class CulturalTransistor(CognitiveTransistor):
    """Colors thoughts based on cultural beliefs."""

    def __init__(self, beliefs: Optional[List[str]] = None):
        super().__init__()
        self.beliefs = beliefs or []
        self.salience = 0.8  # High influence

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through cultural lens."""
        if not self.beliefs:
            return TransistorOutput(input_text, 0.1, {})

        # Simple rule-based coloring (LLM integration later)
        colored = f"{input_text} (through lens of: {', '.join(self.beliefs[:2])})"

        return TransistorOutput(
            transformed_text=colored,
            salience=self.salience,
            metadata={'beliefs': self.beliefs}
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['beliefs'] = self.beliefs
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CulturalTransistor':
        instance = cls(beliefs=data.get('beliefs', []))
        instance.salience = data.get('salience', 0.8)
        instance.enabled = data.get('enabled', True)
        return instance


class PersonalityTransistor(CognitiveTransistor):
    """Colors thoughts based on personality traits."""

    def __init__(self, traits: Optional[Dict[str, float]] = None):
        super().__init__()
        self.traits = traits or {
            'curiosity': 0.5,
            'impulsivity': 0.5,
            'emotional_volatility': 0.5
        }
        self.salience = 0.6

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through personality lens."""
        # Find dominant trait
        dominant_trait = max(self.traits.items(), key=lambda x: x[1])
        trait_name, trait_value = dominant_trait

        if trait_value < 0.6:
            # No strong traits
            return TransistorOutput(input_text, 0.3, {})

        # Color based on dominant trait
        if trait_name == 'curiosity' and trait_value > 0.7:
            colored = f"{input_text} — I wonder why that happened?"
        elif trait_name == 'impulsivity' and trait_value > 0.7:
            colored = f"{input_text} — I should react immediately!"
        elif trait_name == 'emotional_volatility' and trait_value > 0.7:
            colored = f"{input_text} — This is overwhelming!"
        else:
            colored = input_text

        return TransistorOutput(
            transformed_text=colored,
            salience=self.salience,
            metadata={'dominant_trait': trait_name, 'value': trait_value}
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['traits'] = self.traits
        return d


class MoodTransistor(CognitiveTransistor):
    """Colors thoughts based on current emotional state."""

    def __init__(self):
        super().__init__()
        self.salience = 0.5

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through emotional lens."""
        affect = context.get('affect', [0.0, 0.0, 0.0, 0.0, 0.0])
        valence, arousal, fear, sorrow, boredom = affect

        # Determine mood coloring
        if fear > 0.6:
            colored = f"{input_text} (feeling anxious about this)"
            salience = 0.7
        elif sorrow > 0.6:
            colored = f"{input_text} (this makes me sad)"
            salience = 0.6
        elif boredom > 0.6:
            colored = f"{input_text} (not interesting)"
            salience = 0.2
        elif valence > 0.5 and arousal > 0.5:
            colored = f"{input_text} (exciting!)"
            salience = 0.6
        else:
            colored = input_text
            salience = 0.3

        return TransistorOutput(
            transformed_text=colored,
            salience=salience,
            metadata={'mood': affect}
        )


class SomaticCognitiveTransistor(CognitiveTransistor):
    """
    Somatic (bodily) sensation transistor.

    Modulates thoughts based on physical sensations:
    - Impact (hit by objects)
    - Worn items (itchy sweater, heavy armor)
    - Environment (cold room, gust of wind, humid air)
    - Touch (hot, cold, rough, soft)

    Sensitive to room environment metadata (temperature, humidity, wind).
    """

    def __init__(self):
        super().__init__()
        self.salience = 0.7  # Physical sensations are hard to ignore
        self.active_sensations = []
        self.worn_items = []
        self.last_interrupt_time = 0
        self.environment_cache = {}  # Cached room environment

    def add_sensation(
        self,
        sensation_type: str,
        intensity: float,
        duration: float = 0,
        metadata: Optional[Dict] = None
    ):
        """
        Add a bodily sensation.

        Args:
            sensation_type: "impact", "pain", "itch", "cold", "hot", "wet", etc.
            intensity: 0.0 to 1.0 (strength)
            duration: Seconds (0 = instant, >0 = sustained)
            metadata: Additional data (location, source, etc.)
        """
        self.active_sensations.append({
            'type': sensation_type,
            'intensity': intensity,
            'duration': duration,
            'start_time': time.time(),
            'metadata': metadata or {}
        })

    def update_environment(self, room_environment: Dict[str, str]):
        """
        Update environmental awareness from room.

        Called when agent enters room or environment changes.

        Args:
            room_environment: Room's environment dict
                - temperature: "freezing", "cold", "cool", "comfortable", "warm", "hot"
                - humidity: "arid", "dry", "normal", "humid", "muggy"
                - wind: "calm", "breezy", "windy", "gale"
                - weather: "clear", "rain", "snow"
        """
        self.environment_cache = room_environment

        # Check for immediate sensations from environment
        temp = room_environment.get('temperature', 'comfortable')
        humidity = room_environment.get('humidity', 'normal')
        wind = room_environment.get('wind', 'calm')

        # Temperature sensations
        if temp in ['freezing', 'cold']:
            intensity = 0.8 if temp == 'freezing' else 0.5
            self.add_sensation('cold', intensity, duration=999999)
        elif temp in ['hot', 'scorching']:
            intensity = 0.8 if temp == 'scorching' else 0.5
            self.add_sensation('hot', intensity, duration=999999)

        # Humidity sensations
        if humidity in ['muggy', 'drenched']:
            self.add_sensation('humid', 0.4, duration=999999)
        elif humidity == 'arid':
            self.add_sensation('dry', 0.3, duration=999999)

        # Wind sensations
        if wind in ['windy', 'gale']:
            intensity = 0.6 if wind == 'gale' else 0.4
            self.add_sensation('wind', intensity, duration=999999)

    def process_dynamic_event(self, event: str, event_data: Dict):
        """
        Process dynamic environmental event.

        Args:
            event: "wind_gust", "temperature_change", "rain_starts", etc.
            event_data: Event details (direction, intensity, etc.)
        """
        if event == 'wind_gust':
            # Gust of cold air from the north
            direction = event_data.get('direction', 'unknown')
            temperature = event_data.get('temperature', 'cold')
            intensity = event_data.get('intensity', 0.5)

            # Add brief sensation
            self.add_sensation(
                'wind_cold' if temperature == 'cold' else 'wind',
                intensity=intensity,
                duration=5,  # Gust lasts 5 seconds
                metadata={'direction': direction}
            )

        elif event == 'rain_starts':
            self.add_sensation('wet', 0.6, duration=999999)

        elif event == 'temperature_change':
            new_temp = event_data.get('new_temperature')
            if new_temp in ['cold', 'freezing']:
                self.add_sensation('cold', 0.6, duration=999999)

    def add_worn_item(self, item_id: str, discomfort_type: str, discomfort_level: float):
        """Add worn item causing discomfort."""
        self.worn_items.append({
            'item_id': item_id,
            'discomfort_type': discomfort_type,
            'discomfort_level': discomfort_level
        })

    def remove_worn_item(self, item_id: str):
        """Remove worn item."""
        self.worn_items = [item for item in self.worn_items if item['item_id'] != item_id]

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Filter input through bodily sensation lens.

        Physical sensations interrupt/color thought based on intensity.
        Low intensity = low salience (can deal with it).
        High intensity = high salience (dominates attention).
        """
        current_time = time.time()

        # Clean up expired sensations
        self.active_sensations = [
            s for s in self.active_sensations
            if s['duration'] == 0 or (current_time - s['start_time']) < s['duration']
        ]

        # Find strongest active sensation
        strongest = None
        if self.active_sensations:
            strongest = max(self.active_sensations, key=lambda s: s['intensity'])

        # Check worn item discomfort (periodic)
        worn_interrupt = None
        if self.worn_items and (current_time - self.last_interrupt_time) > 30:
            worst = max(self.worn_items, key=lambda i: i['discomfort_level'])
            if worst['discomfort_level'] > 0.4:
                worn_interrupt = worst
                self.last_interrupt_time = current_time

        # Generate response based on strongest sensation
        if strongest and strongest['intensity'] > 0.6:
            # High intensity - interrupts thought
            response = self._generate_sensation_response(strongest)
            colored = f"{response} ...uh, {input_text}"
            salience = min(0.9, strongest['intensity'])

        elif strongest and strongest['intensity'] > 0.3:
            # Medium intensity - colors but doesn't interrupt
            response = self._generate_sensation_response(strongest)
            colored = f"{input_text} {response}"
            salience = strongest['intensity'] * 0.6  # Lower salience

        elif worn_interrupt:
            # Sustained discomfort interrupts
            response = self._generate_discomfort_response(worn_interrupt)
            colored = f"{response} *pauses* {input_text}"
            salience = worn_interrupt['discomfort_level']

        else:
            # No significant sensations
            colored = input_text
            salience = 0.1  # Minimal somatic influence

        return TransistorOutput(
            transformed_text=colored,
            salience=salience,
            metadata={
                'active_sensations': len(self.active_sensations),
                'worn_items': len(self.worn_items),
                'strongest_sensation': strongest['type'] if strongest else None
            }
        )

    def _generate_sensation_response(self, sensation: Dict) -> str:
        """Generate response to bodily sensation."""
        sensation_type = sensation['type']
        intensity = sensation['intensity']

        # Response templates by type and intensity
        if sensation_type == 'impact' or sensation_type == 'impact_soft':
            if intensity > 0.7:
                return random.choice(["OWCH!", "OW!", "*recoils in pain*"])
            else:
                return random.choice(["Oof!", "*stumbles*", "Hey!"])

        elif sensation_type == 'pain' or sensation_type == 'impact_hard':
            return random.choice(["OUCH! THAT HURT!", "OW OW OW!", "*winces in pain*"])

        elif sensation_type == 'hot':
            if intensity > 0.7:
                return random.choice(["OUCH! HOT!", "*yanks hand back*", "BURNING!"])
            else:
                return random.choice(["Warm...", "Getting hot here", "*fans self*"])

        elif sensation_type == 'cold':
            if intensity > 0.7:
                return random.choice(["Brr! FREEZING!", "*shivers violently*", "SO COLD!"])
            else:
                return random.choice(["Bit chilly", "*shivers slightly*", "Brr"])

        elif sensation_type == 'wind' or sensation_type == 'wind_cold':
            direction = sensation['metadata'].get('direction', '')
            if direction:
                return f"*cold gust from the {direction}* Brr!"
            else:
                return "*gust of wind* Whoa!"

        elif sensation_type == 'humid':
            return random.choice(["So muggy...", "*wipes sweat*", "Air is thick"])

        elif sensation_type == 'wet':
            return random.choice(["*shakes off water*", "I'm soaked!", "Wet!"])

        else:
            return "*reacts to sensation*"

    def _generate_discomfort_response(self, worn_item: Dict) -> str:
        """Generate response to worn item discomfort."""
        discomfort_type = worn_item['discomfort_type']

        responses = {
            'itchy': ["Aagh this sweater!", "*scratches frantically*", "So itchy!", "*scratches*"],
            'tight': ["These shoes!", "*adjusts uncomfortably*", "Too tight!", "Can't breathe..."],
            'heavy': ["This armor...", "*shifts weight*", "So heavy...", "*groans*"],
            'hot': ["Too hot in this!", "*tugs at collar*", "Sweltering!", "*fans self*"],
            'cold': ["*shivers in thin clothes*", "Need warmer clothes!", "*huddles*"]
        }

        options = responses.get(discomfort_type, ["*adjusts uncomfortably*"])
        return random.choice(options)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d['active_sensations'] = self.active_sensations
        d['worn_items'] = self.worn_items
        d['environment_cache'] = self.environment_cache
        return d


class SoundEmitter:
    """
    Sound emitter component for prims.

    Emits acoustic signals that nearby Noodlings perceive through
    their Somatic Cognitive Transistor.
    """

    def __init__(
        self,
        sound_type: str = "ambient",
        decibels: float = 60.0,
        frequency: str = "medium",
        pattern: str = "continuous",
        attenuation: float = 1.0,
        enabled: bool = True
    ):
        """
        Initialize sound emitter.

        Args:
            sound_type: "siren", "music", "speech", "bells", "engine", "ambient", "alarm", "crying", "laughter"
            decibels: Volume at source (0-140 dB)
            frequency: "low", "medium", "high" (pitch)
            pattern: "continuous", "pulsing", "intermittent", "random"
            attenuation: Distance falloff rate (0.5 = slow, 2.0 = fast)
            enabled: Is emitter active
        """
        self.sound_type = sound_type
        self.decibels = decibels
        self.frequency = frequency
        self.pattern = pattern
        self.attenuation = attenuation
        self.enabled = enabled

        # Future: multimodal audio
        self.audio_file = None
        self.audio_description = None

    def get_effective_decibels(self, distance: float) -> float:
        """
        Calculate effective volume at distance.

        Args:
            distance: Distance from emitter (meters/units)

        Returns:
            Effective decibel level at that distance
        """
        if distance == 0:
            return self.decibels

        # Inverse square law with attenuation factor
        import math
        falloff = 20 * math.log10(max(1.0, distance)) * self.attenuation
        return max(0, self.decibels - falloff)

    def get_sound_description(self) -> str:
        """Get semantic description of sound."""
        descriptions = {
            'siren': "wailing siren",
            'music': "melodic music",
            'speech': "voices speaking",
            'bells': "ringing bells",
            'engine': "rumbling engine",
            'ambient': "background noise",
            'alarm': "piercing alarm",
            'laughter': "joyful laughter",
            'crying': "distressed crying",
            'howling': "mournful howling"
        }

        base_desc = descriptions.get(self.sound_type, "unknown sound")

        # Add intensity descriptor
        if self.decibels > 110:
            intensity = "deafening"
        elif self.decibels > 90:
            intensity = "very loud"
        elif self.decibels > 70:
            intensity = "loud"
        elif self.decibels > 50:
            intensity = "moderate"
        else:
            intensity = "quiet"

        return f"{intensity} {base_desc}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'SoundEmitter',
            'sound_type': self.sound_type,
            'decibels': self.decibels,
            'frequency': self.frequency,
            'pattern': self.pattern,
            'attenuation': self.attenuation,
            'enabled': self.enabled,
            'audio_file': self.audio_file
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoundEmitter':
        """Deserialize from dictionary."""
        return cls(
            sound_type=data.get('sound_type', 'ambient'),
            decibels=data.get('decibels', 60.0),
            frequency=data.get('frequency', 'medium'),
            pattern=data.get('pattern', 'continuous'),
            attenuation=data.get('attenuation', 1.0),
            enabled=data.get('enabled', True)
        )


def calculate_acoustic_salience(decibels: float, sound_type: str, context: Dict) -> float:
    """
    Calculate how much sound dominates attention.

    Args:
        decibels: Volume level
        sound_type: Type of sound
        context: Contextual factors

    Returns:
        Salience (0.0 to 1.0)
    """
    # Base salience from decibels
    if decibels > 110:
        base_salience = 0.9  # Painful
    elif decibels > 90:
        base_salience = 0.7  # Very loud
    elif decibels > 70:
        base_salience = 0.5  # Loud
    elif decibels > 50:
        base_salience = 0.3  # Moderate
    else:
        base_salience = 0.1  # Quiet

    # Sound type multipliers
    type_multipliers = {
        'siren': 1.2,      # Designed to grab attention
        'alarm': 1.2,
        'crying': 1.1,     # Hard to ignore
        'music': 0.8,      # More tolerable
        'speech': 0.9,
        'ambient': 0.7,
        'laughter': 0.7
    }

    multiplier = type_multipliers.get(sound_type, 1.0)
    salience = min(1.0, base_salience * multiplier)

    # Context adjustments
    sensitivity = context.get('acoustic_sensitivity', 1.0)
    salience *= sensitivity

    # Location context (orphanage = higher stakes)
    if context.get('location_type') == 'orphanage' and sound_type in ['siren', 'alarm']:
        salience = min(1.0, salience * 1.5)

    return salience


# ===== Dependency Resolution =====

COMPONENT_DEPENDENCIES = {
    'CognitiveTransistor': ['CognitiveManifold'],
    'CulturalTransistor': ['CognitiveManifold'],
    'PersonalityTransistor': ['CognitiveManifold'],
    'MoodTransistor': ['CognitiveManifold'],
    'MemoryTransistor': ['CognitiveManifold'],
    'SocialExpectationTransistor': ['CognitiveManifold'],
    'SomaticCognitiveTransistor': ['CognitiveManifold']
}


def check_component_dependencies(
    component_type: str,
    existing_components: List[str]
) -> List[str]:
    """
    Check if component has missing dependencies.

    Args:
        component_type: Type being added
        existing_components: List of existing component types on prim

    Returns:
        List of missing dependency types
    """
    required = COMPONENT_DEPENDENCIES.get(component_type, [])
    missing = [dep for dep in required if dep not in existing_components]
    return missing


# ===== Example Usage =====

if __name__ == '__main__':
    # Test cognitive pipeline
    print("=== COGNITIVE MANIFOLD TEST ===\n")

    # Create transistors
    cultural = CulturalTransistor(beliefs=["Logic is supreme", "Emotions are inefficient"])
    personality = PersonalityTransistor(traits={'curiosity': 0.9, 'impulsivity': 0.2})
    mood = MoodTransistor()

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="simple_concat")
    manifold.register_transistor(cultural)
    manifold.register_transistor(personality)
    manifold.register_transistor(mood)

    # Test perception
    perception = "Phi is crying because her toy broke"
    context = {
        'affect': [0.0, 0.3, 0.1, 0.0, 0.0],  # Neutral
        'memory_system': None
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nOutput: {integrated_thought}")
    print(f"\nTransistors: {len(manifold.transistors)}")
