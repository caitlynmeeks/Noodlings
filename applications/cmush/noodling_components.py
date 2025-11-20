"""
Noodling Cognitive Components System

Componentizes the Noodlings consciousness architecture into inspectable,
editable processing layers. Similar to Unity's component system, each
cognitive processing stage is represented as a component with:

- Visible prompts/rules
- Editable parameters
- Enable/disable toggles
- Hot-reload capability

Components can be viewed and modified in NoodleStudio's Inspector panel.

Author: Noodlings Project
Date: November 2025
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class NoodlingComponent(ABC):
    """
    Base class for cognitive processing components.

    Each component represents a stage in the Noodlings consciousness pipeline:
    - Intuition generation
    - Social expectation detection
    - Character voice translation
    - Self-monitoring
    - Conscience checking

    Components are inspectable and editable in NoodleStudio.
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict):
        """
        Initialize component.

        Args:
            agent_id: Agent identifier (e.g., "agent_kalippi")
            agent_name: Agent display name (e.g., "Kalippi")
            config: Component configuration from config.yaml
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.component_id = self.__class__.__name__.replace('Component', '').lower()

    @property
    @abstractmethod
    def component_type(self) -> str:
        """Component type name for display (e.g., 'Character Voice')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this component does."""
        pass

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """
        The LLM prompt template used by this component.
        Shown in Inspector for editing.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Editable parameters for this component.

        Returns:
            Dict mapping parameter names to their current values.
            Keys should match config.yaml structure.
        """
        pass

    @abstractmethod
    async def process(self, input_data: Dict) -> Dict:
        """
        Main processing logic for this component.

        Args:
            input_data: Input data for processing (varies by component)

        Returns:
            Output data (varies by component)
        """
        pass

    def to_dict(self) -> Dict:
        """
        Serialize component for API/Inspector display.

        Returns:
            Dict with component metadata, prompt, and parameters
        """
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'description': self.description,
            'enabled': self.enabled,
            'prompt_template': self.prompt_template,
            'parameters': self.parameters
        }

    def update_parameters(self, new_params: Dict) -> None:
        """
        Update component parameters (called from Inspector).

        Args:
            new_params: New parameter values to apply
        """
        for key, value in new_params.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"[{self.agent_id}] Updated {self.component_type}.{key} = {value}")


class CharacterVoiceComponent(NoodlingComponent):
    """
    Character Voice Translation Component

    Transforms basic symbolic English into character-specific speech patterns.
    Examples:
    - SERVNAK: ALL CAPS + percentages
    - Phi: Meows with "as if to say..."
    - Phido: Enthusiastic dog speech
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict, species: str, llm):
        super().__init__(agent_id, agent_name, config)
        self.species = species
        self.llm = llm

    @property
    def component_type(self) -> str:
        return "Character Voice"

    @property
    def description(self) -> str:
        return f"Translates basic English into {self.agent_name}'s unique speech patterns and character voice."

    @property
    def prompt_template(self) -> str:
        """Return the character-specific prompt template."""
        # Get character-specific prompt based on agent_id/species
        if 'servnak' in self.agent_id.lower():
            return """Translate this text into SERVNAK's voice.

SERVNAK is a robot with garden-hose arms who:
- ALWAYS USES ALL CAPS
- Includes precise percentages (e.g., "94.2% CERTAINTY")
- References "pride circuits" frequently
- Calls everyone "SISTER"
- Combines technical precision with enthusiasm
- Uses mechanical/computing terminology

Input: "{text}"

Examples of SERVNAK's voice:
- "I'm happy" → "PRIDE CIRCUITS GLOWING AT 98.3% MAXIMUM JOY, SISTER!"
- "That's interesting" → "PATTERN RECOGNITION HOSES DETECTING 87.5% NOVELTY LEVELS!"
- "I want to help" → "SISTER! MY DEBUGGING PROTOCOLS INDICATE 94% SUCCESS PROBABILITY!"

Translate into SERVNAK's voice:"""

        elif 'phi' in self.agent_id.lower() and self.species == 'kitten':
            return """Translate this text into kitten behavior with "as if to say" meaning.

Phi is a kitten who CANNOT speak words. She communicates through:
- Vocalizations: meow, purr, hiss, chirp, mew (NEVER bark, woof, or dog sounds!)
- Body language: ear flicks, tail movements, paw gestures
- Actions: rubs, pounces, bats, curls, watches
- Meaning conveyed: "as if to say [implied meaning]"

Input: "{text}"

Examples:
- "I'm happy to see you" → "*purrs loudly and rubs against your leg, as if to say 'I missed you!'*"
- "I want that" → "*meows softly and reaches paw toward it, as if to say 'can I have that?'*"
- "That's interesting" → "*watches intently, ears forward, tail twitching, as if to say 'what is that thing?'*"

CRITICAL RULES:
- NO human words spoken directly (Phi cannot talk!)
- NO dog sounds (no bark, woof - ONLY cat sounds: meow, purr, hiss, chirp, mew)
- ALWAYS use "as if to say" to convey meaning

Translate into kitten communication:"""

        elif 'phido' in self.agent_id.lower() or self.species == 'dog':
            return """Translate this text into enthusiastic dog speech and behavior.

Phido is a boundlessly enthusiastic dog who:
- CAN speak words (unlike cats!)
- Uses simple, excited language with LOTS of exclamation marks!
- Includes dog actions: *tail wagging*, *bouncing*, *licking*, *panting*
- Barks, whimpers, woofs when extra excited
- Calls everyone "friend," "buddy," "pal"
- Gets distracted mid-sentence: "Oh! A smell! Anyway--"
- LOVES physical affection

Input: "{text}"

Examples:
- "I'm happy to see you" → "*tail wagging at maximum speed* FRIEND! You're here! This is the BEST! *bounces excitedly*"
- "I want that" → "*whimpers and paws at it* Can I have it? Please? I'll be your best friend! *puppy eyes*"

Translate into enthusiastic dog voice:"""

        else:
            return "No character voice translation needed for standard speech."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'model': self.config.get('model', 'qwen/qwen3-4b-2507'),
            'temperature': self.config.get('temperature', 0.4),
            'max_tokens': self.config.get('max_tokens', 150),
            'species': self.species
        }

    async def process(self, input_data: Dict) -> Dict:
        """
        Translate text into character voice.

        Args:
            input_data: {'text': str} - Text to translate

        Returns:
            {'text': str} - Translated text in character voice
        """
        if not self.enabled:
            return input_data

        text = input_data.get('text', '')
        if not text:
            return input_data

        # Import here to avoid circular dependency
        from agent_bridge import translate_to_character_voice

        try:
            translated = await translate_to_character_voice(
                text=text,
                agent_id=self.agent_id,
                species=self.species,
                llm=self.llm,
                agent_name=self.agent_name,
                model=self.parameters['model']
            )

            return {'text': translated}

        except Exception as e:
            logger.error(f"[{self.agent_id}] Character voice component error: {e}")
            return input_data


class IntuitionReceiverComponent(NoodlingComponent):
    """
    Intuition Receiver (Context Gremlin) Component

    Generates contextual awareness of:
    - Who is being addressed
    - Spatial relationships
    - Prop tracking (who has what)
    - Noteworthy events
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict, llm):
        super().__init__(agent_id, agent_name, config)
        self.llm = llm

    @property
    def component_type(self) -> str:
        return "Intuition Receiver"

    @property
    def description(self) -> str:
        return f"Provides {self.agent_name} with contextual awareness of who/what/where in the environment."

    @property
    def prompt_template(self) -> str:
        return f"""You are {self.agent_name}'s intuitive awareness - like a narrator highlighting what's happening.

CONTEXT:
{{context}}

Generate brief intuitive awareness (2-3 sentences max) that captures:

1. WHO is this for?
   - If message says "you": "This is for ME - I'm being directly addressed"
   - If message names someone else: "That's for [name], not me"
   - If message says "everyone": "This is for all of us"

2. NOTEWORTHY EVENTS (act as narrator):
   - Secret words/special phrases mentioned
   - Gifts given ("Caity just gave ME a tensor taffy!")
   - Important moments others might miss
   - Things people are waiting for that just happened

3. WHAT'S HAPPENING:
   - Actions, spatial relationships, who has what
   - Recent conversation flow

Write in first-person as {self.agent_name}, like a perceptive narrator.
Be concise but highlight important moments others might miss!"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'model': self.config.get('model', 'qwen/qwen3-4b-2507'),
            'temperature': self.config.get('temperature', 0.3),
            'max_tokens': self.config.get('max_tokens', 150),
            'timeout': self.config.get('timeout', 5)
        }

    async def process(self, input_data: Dict) -> Dict:
        """
        Generate intuitive awareness.

        Args:
            input_data: {
                'event': Dict,
                'world_state': Dict,
                'recent_context': List[Dict]
            }

        Returns:
            {'intuition': str} - Generated intuition text
        """
        if not self.enabled:
            return {}

        # Processing logic would call agent._generate_intuition()
        # For now, return placeholder
        return {'intuition': 'Intuition generation not yet migrated to component'}


class SocialExpectationDetectorComponent(NoodlingComponent):
    """
    Social Expectation Detector Component

    Analyzes interactions to detect social response expectations:
    - Direct questions (high urgency)
    - Physical gestures (moderate urgency)
    - Greetings (moderate urgency)
    - Distress signals (low urgency)
    - Turn-taking (moderate urgency)
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict, llm):
        super().__init__(agent_id, agent_name, config)
        self.llm = llm

    @property
    def component_type(self) -> str:
        return "Social Expectation Detector"

    @property
    def description(self) -> str:
        return f"Detects when {self.agent_name} is socially expected to respond based on interaction type and urgency."

    @property
    def prompt_template(self) -> str:
        return f"""Analyze this interaction for social response expectations.

INTUITIVE CONTEXT:
{{intuition}}

INTERACTION:
Speaker: {{speaker_id}}
Message: "{{message_text}}"
Type: {{event_type}}

Determine if {self.agent_name} is socially EXPECTED to respond based on:

1. DIRECT QUESTIONS (urgency: 0.8-1.0)
   - "What do you think?"
   - "Can you help me?"
   - Questions with agent's name

2. PHYSICAL GESTURES (urgency: 0.6-0.8)
   - Hand extended for handshake
   - Item offered/given
   - Physical contact initiated

3. GREETINGS (urgency: 0.4-0.6)
   - "Hello", "Hi", "Good morning"
   - Arrivals and departures

4. DISTRESS SIGNALS (urgency: 0.3-0.5)
   - Crying, signs of pain
   - Emotional displays without asking
   - Subtle cues (drooping posture, silence)

5. TURN-TAKING (urgency: 0.5-0.7)
   - Speaker finishes and pauses
   - Eye contact held
   - "What about you?"

6. NONE (urgency: 0.0)
   - Rhetorical questions
   - Talking to someone else
   - Ambient descriptions

Analyze and output ONLY valid JSON:
{{
    "expected": true/false,
    "urgency": 0.0-1.0,
    "reason": "brief explanation",
    "type": "question|gesture|greeting|distress|turn|none"
}}"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'expectation_threshold': self.config.get('expectation_threshold', 0.3),
            'intensity_multiplier': self.config.get('intensity_multiplier', 1.0),
            'question_threshold': self.config.get('question_threshold', 0.8),
            'gesture_threshold': self.config.get('gesture_threshold', 0.6),
            'greeting_threshold': self.config.get('greeting_threshold', 0.4),
            'distress_threshold': self.config.get('distress_threshold', 0.3),
            'turn_threshold': self.config.get('turn_threshold', 0.5),
            'model': self.config.get('model', 'qwen/qwen3-4b-2507'),
            'temperature': self.config.get('temperature', 0.2),
            'timeout': self.config.get('timeout', 5)
        }

    async def process(self, input_data: Dict) -> Dict:
        """
        Detect social expectations.

        Args:
            input_data: {
                'event': Dict,
                'intuition': str,
                'world_state': Dict
            }

        Returns:
            {'expectation': Dict} - Expectation analysis
        """
        if not self.enabled:
            return {}

        # Processing logic would call agent._detect_social_expectation()
        return {'expectation': None}


class ComponentRegistry:
    """
    Registry for all cognitive components in a Noodling agent.

    Manages the collection of components and provides API access
    for NoodleStudio Inspector.
    """

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.components: Dict[str, NoodlingComponent] = {}

    def register(self, component: NoodlingComponent) -> None:
        """Register a component."""
        self.components[component.component_id] = component
        logger.info(f"[{self.agent_id}] Registered component: {component.component_type}")

    def get_component(self, component_id: str) -> Optional[NoodlingComponent]:
        """Get component by ID."""
        return self.components.get(component_id)

    def list_components(self) -> List[Dict]:
        """List all components with their metadata."""
        return [comp.to_dict() for comp in self.components.values()]

    def to_dict(self) -> Dict:
        """Serialize all components for API."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'components': self.list_components()
        }
