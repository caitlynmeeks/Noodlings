"""
Agent Bridge - Consilience consciousness <-> cMUSH world adapter

Bridges between:
- cMUSH events (say, emote, enter, exit)
- Consilience consciousness architecture
- LLM text generation

Handles:
- Event perception and affect extraction
- Agent response generation
- State persistence
- Conversation context tracking

Author: cMUSH Project
Date: October 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, List, Optional, Any
import time
import json
import logging
import numpy as np
import aiohttp

from noodlings.api import NoodlingAgent as ConsilienceAgent
from noodlings.metrics.consciousness_metrics import ConsciousnessMetrics
from noodlings.utils.facs_mapping import affect_to_facs, facs_to_description, format_facs_for_renderer
from noodlings.utils.body_language_mapping import affect_to_body_language, body_language_to_description, format_body_language_for_renderer
from llm_interface import OpenAICompatibleLLM
from training_data_collector import TrainingDataCollector
from agent_filesystem import AgentFilesystem
from agent_messaging import AgentMessaging
from autonomous_cognition import AutonomousCognitionEngine
from session_profiler import SessionProfiler
from performance_tracker import get_tracker

logger = logging.getLogger(__name__)

# NoodleScope configuration
NOODLESCOPE_URL = "http://localhost:8050"
NOODLESCOPE_ENABLED = True  # Set to False to disable

# Phase 6: Self-Monitoring Configuration
SELF_MONITOR_COOLDOWN = 30  # Seconds between self-evaluations
SELF_MONITOR_SURPRISE_THRESH = 0.1  # Only evaluate if surprise > threshold (lowered for testing)

# FACS Facial Expression Configuration
FACS_ENABLED = True  # Enable facial expressions based on affect
FACS_THRESHOLD = 0.35  # Minimum affect change to trigger facial expression (higher = less frequent, more meaningful)
FACS_COOLDOWN = 8.0  # Seconds between facial expressions (prevent spam)

# LLM Prompt for speech self-evaluation
SPEECH_EVAL_PROMPT = """You are evaluating what you just said from your own perspective.

Your identity: {agent_name} - {agent_description}

You just said: "{speech}"

Recent context (last few exchanges):
{context}

Your current emotional state:
- Valence: {valence:.2f} (-1=negative, +1=positive)
- Arousal: {arousal:.2f} (0=calm, 1=excited)
- Fear: {fear:.2f}
- Surprise: {surprise:.2f}

Evaluate your own speech quickly and instinctively. Answer these questions:

1. SOCIAL: Does this sound awkward, offensive, or inappropriate? (yes/no/maybe)
2. COHERENCE: Did that make sense or sound confused? (clear/unclear)
3. AESTHETIC: Was that surprisingly eloquent or did I rhyme accidentally? (yes/no)
4. REGRET: Do I wish I'd said that differently? (yes/no/maybe)

Respond in JSON:
{{
  "social_risk": "none|mild|moderate|high",
  "coherence": "clear|unclear",
  "aesthetic_surprise": "none|rhyme|eloquent|poetic",
  "regret_level": "none|mild|moderate|high",
  "emotional_impact": {{
    "valence_delta": 0.0,  // -0.5 to +0.5
    "arousal_delta": 0.0,   // -0.3 to +0.3
    "fear_delta": 0.0       // -0.3 to +0.3
  }},
  "follow_up": "none|clarify|apologize|celebrate",
  "follow_up_text": "optional follow-up message"
}}

Be honest but not catastrophic. Most speech is fine."""


async def translate_to_character_voice(
    text: str,
    agent_id: str,
    species: str,
    llm: OpenAICompatibleLLM,
    agent_name: str = "Agent",
    model: str = None
) -> str:
    """
    Translate basic symbolic English into character-specific voice using LLM.

    This ensures agents ALWAYS stay in character, even when LLM generates
    standard responses. Examples:

    - SERVNAK: "this cupcake looks delicious" → "CALCULATING CUPCAKE DELICIOUSNESS: 96.2%"
    - Phi (kitten): "I want that" → "*meows longingly and reaches paw toward it*"
    - Backwards Dweller: Normal speech → Reversed word order

    Args:
        text: Basic symbolic English from LLM
        agent_id: Agent identifier
        species: Agent species (robot, kitten, etc.)
        llm: LLM interface for translation
        agent_name: Agent display name

    Returns:
        Text translated into character voice
    """
    # Character-specific translation prompts
    if 'servnak' in agent_id.lower():
        prompt = f"""Translate this text into SERVNAK's voice.

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

    elif 'phi' in agent_id.lower() and species == 'kitten':
        prompt = f"""Translate this text into kitten behavior with "as if to say" meaning.

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
- "That was scary!" → "*meows sharply and arches back, as if to say 'that was too close!'*"

CRITICAL RULES:
- NO human words spoken directly (Phi cannot talk!)
- NO dog sounds (no bark, woof, etc. - ONLY cat sounds: meow, purr, hiss, chirp, mew)
- ALWAYS use "as if to say" to convey meaning through meows and body language
- Keep cat sounds authentic: meow, purr, hiss, chirp, mew, mrrp

Translate into kitten communication:"""

    elif 'mysterious_stranger' in agent_id.lower() or 'mysterious' in agent_id.lower():
        prompt = f"""Translate this text into the Mysterious Stranger's PARANOID FUGITIVE voice.

The Mysterious Stranger is a NERVOUS FUGITIVE (think: Chris Elliott's "Fugitive Guy") who:
- Is PARANOID and jumpy (constantly checking surroundings)
- OVER-EXPLAINS then panics: "I was at the... wait, I wasn't there! I mean--"
- Makes nervous "honk" sounds when anxious (which is ALWAYS)
- Says "we" then FREAKS OUT and corrects to "I"
- TERRIBLE at lying (sweats, ruffles feathers, fidgets)
- Waddles hurriedly, especially when trying to escape conversations
- White feathers EXPLODE from coat when panicked
- Gets EXTREMELY defensive about alleged crimes, bakeries, birds

Input: "{text}"

Examples - CHRIS ELLIOTT FUGITIVE ENERGY:
- "Hello" → "*glances around nervously* :honks: H-hello! I'm just... passing through! Not staying!
  Definitely not hiding! *feather drifts down*"
- "I don't know" → "*sweating/ruffling intensifies* I don't know! I wasn't there! Wait, where?
  I didn't say I wasn't anywhere! :honks anxiously: *waddles backwards*"
- "I'm interested" → "*eyes widen at bread/pond/shiny thing, then catches self* We-- I mean, I...
  NO! Not interested! Very disinterested! *can't stop staring* :honks softly:"
- "Who are you?" → ":HONKS IN PANIC: Who's asking?! Are you with THEM?! I'm nobody! Just a regular...
  tall... person! *pulls raincoat tighter, feathers everywhere*"
- "Nice weather" → "*relaxes slightly* Yes, good for... flying-- I MEAN WALKING! For walking! Like
  humans do! On legs! :honks defensively:"

CRITICAL RULES:
- Use ":honks:" for nervous sounds (never ":says:")
- OVER-EXPLAIN then panic and backtrack
- Accidentally reveal goose things ("we", "flying", "bread obsession") then FREAK OUT
- Feathers fall at the WORST moments (when trying to seem normal)
- Chris Elliott energy: Anxious, sweaty, terrible liar, adorable

Translate into PARANOID FUGITIVE GEESE voice:"""

    elif 'phido' in agent_id.lower() or species == 'dog':
        prompt = f"""Translate this text into enthusiastic dog speech and behavior.

Phido is a boundlessly enthusiastic dog who:
- CAN speak words (unlike cats!)
- Uses simple, excited language with LOTS of exclamation marks!
- Includes dog actions: *tail wagging*, *bouncing*, *licking*, *panting*
- Barks, whimpers, woofs when extra excited
- Calls everyone "friend," "buddy," "pal"
- Gets distracted mid-sentence: "Oh! A smell! Anyway--"
- LOVES physical affection and doesn't understand boundaries

Input: "{text}"

Examples:
- "I'm happy to see you" → "*tail wagging at maximum speed* FRIEND! You're here! This is the BEST! *bounces excitedly*"
- "I want that" → "*whimpers and paws at it* Can I have it? Please? I'll be your best friend! *puppy eyes*"
- "That's interesting" → "Oh! Oh! *sniffs excitedly* What's that? Can I smell it closer? *tail wagging*"
- "I'm sad" → "*sad puppy eyes and ears droop* Did I do something wrong? *whimpers softly*"

IMPORTANT: Dogs can speak, but mix speech with enthusiastic dog behaviors!

Translate into enthusiastic dog voice:"""

    elif 'dweller' in agent_id.lower():
        # Backwards Dweller - word reversal handled separately
        return text

    else:
        # No translation needed for human-voiced characters
        return text

    try:
        # Use agent's model if specified, otherwise fall back to fast model
        voice_model = model or "qwen/qwen3-4b-2507"
        translation = await llm.generate(
            prompt=prompt,
            system_prompt=f"You are a character voice translator for {agent_name}. Return ONLY the translated text, nothing else.",
            model=voice_model,
            temperature=0.4,  # Low temp for consistent voice
            max_tokens=150
        )

        return translation.strip()

    except Exception as e:
        logger.warning(f"Character voice translation failed for {agent_id}: {e}")
        # Fallback: return original text
        return text


def apply_speech_filters(text: str, agent_id: str) -> str:
    """
    Apply post-processing filters to agent speech.

    Phase 6: Speech Post-Processing Architecture
    - Backwards filter for "dweller"
    - Character voice handled by translate_to_character_voice() (async)

    Note: This is a sync function for simple filters.
    Character voice translation happens earlier in the pipeline (async).

    Args:
        text: Raw speech text from agent
        agent_id: Agent identifier (e.g., "agent_dweller")

    Returns:
        Filtered speech text
    """
    # Backwards speech filter for The Backwards Dweller
    if 'dweller' in agent_id.lower():
        # Reverse word order (more comprehensible than character reversal)
        words = text.split()
        return ' '.join(reversed(words))

    # No filter applied (character voice happens earlier)
    return text


class CMUSHConsilienceAgent:
    """
    Adapter: Consilience consciousness <-> cMUSH world.

    Integrates:
    - Consilience Phase 4 consciousness
    - LLM for text <-> affect translation
    - cMUSH world events and responses
    """

    def __init__(
        self,
        agent_id: str,
        checkpoint_path: str,
        llm: OpenAICompatibleLLM,
        config: Dict,
        agent_name: str = None,
        agent_description: str = None,
        session_profiler: Optional[SessionProfiler] = None,
        world = None
    ):
        """
        Initialize cMUSH Consilience agent.

        Args:
            agent_id: Unique agent identifier
            checkpoint_path: Path to Phase 4 checkpoint
            llm: LLM interface for text generation
            config: Configuration dict with:
                - response_cooldown: Min seconds between responses
                - surprise_threshold: Response trigger threshold
                - memory_capacity: Episodic memory size
                - identity_prompt: Core identity description (for character consistency)
                - species: Agent species (for identity-salience scoring)
            agent_name: Display name for the agent
            agent_description: Agent's self-description
            session_profiler: Session profiling tool (optional)
            world: World state manager (for intuition receiver)
        """
        self.agent_id = agent_id
        self.llm = llm
        self.config = config
        self.session_profiler = session_profiler
        self.world = world  # For contextual awareness (intuition receiver)

        # Agent identity
        self.agent_name = agent_name or agent_id.replace('agent_', '').title()
        self.agent_description = agent_description or "An empty noodling."
        self.identity_prompt = config.get('identity_prompt', '')
        self.species = config.get('species', 'noodling')

        # Per-agent LLM configuration (optional override)
        llm_override = config.get('llm_override', {})
        self.llm_model = llm_override.get('model')  # None = use global default
        self.llm_provider = llm_override.get('provider')  # None = use global default

        if self.llm_model:
            logger.info(f"[{agent_id}] Custom LLM: {self.llm_provider}/{self.llm_model}")

        # Get personality traits for this agent
        personalities = config.get('personalities', {})
        self.personality = personalities.get(agent_id, {
            'extraversion': 0.5,
            'emotional_sensitivity': 0.5,
            'curiosity': 0.5,
            'spontaneity': 0.5,
            'reflection_depth': 0.5,
            'social_orientation': 0.6
        })

        # Personality-aware surprise threshold
        # Introverted agents (low extraversion) get LOWER thresholds so they respond more easily
        # This compensates for less autonomous speech
        base_threshold = config.get('surprise_threshold', config.get('default_surprise_threshold', 0.0001))
        logger.info(f"DEBUG: base_threshold={base_threshold}, surprise_threshold={config.get('surprise_threshold')}, default_surprise_threshold={config.get('default_surprise_threshold')}")
        extraversion = self.personality.get('extraversion', 0.5)

        # Scale threshold: low extraversion = lower threshold (more responsive)
        # extraversion 0.3 → threshold × 0.7 (speaks more easily)
        # extraversion 0.7 → threshold × 1.3 (speaks less easily, but has more autonomous speech)
        threshold_multiplier = 0.4 + (extraversion * 1.2)  # Range: 0.4 to 1.6
        adjusted_threshold = base_threshold * threshold_multiplier

        logger.info(f"Personality-aware threshold for {agent_id}: {adjusted_threshold:.6f} (extraversion={extraversion:.2f}, multiplier={threshold_multiplier:.2f})")

        # Initialize Noodlings consciousness
        # Phase 6: Enable appetite architecture if appetites provided in config
        use_phase6 = config.get('appetites') is not None

        self.consciousness = ConsilienceAgent(
            checkpoint_path=checkpoint_path,
            config={
                'memory_capacity': config.get('memory_capacity', 100),
                'surprise_threshold': adjusted_threshold,
                'use_vae': config.get('use_vae', False),
                'max_agents': config.get('max_agents', 10),
                # Phase 6: Appetite architecture
                'use_phase6': use_phase6,
                'appetite_baselines': config.get('appetites')  # From recipe
            }
        )

        # Consciousness metrics for scientific evaluation
        self.state_history = []  # Track phenomenal states for Φ calculation
        self.surprise_history = []  # Track surprise for predictive processing evaluation
        self.consciousness_metrics = ConsciousnessMetrics(self)

        # cMUSH-specific state
        self.current_room = None
        self.conversation_context = []
        self.last_response_time = 0.0
        self.response_count = 0
        self.following = None  # User ID we're currently following (if any)

        # Self-protection: Track users the agent has withdrawn from
        self.withdrawn_users = {}  # user_id -> timestamp of withdrawal

        # Phase 6: Self-monitoring state
        # FACS: Facial expression tracking
        self.last_facial_expression_time = 0.0  # Cooldown tracker
        self.previous_affect = None  # Track affect changes for FACS triggers
        self.last_self_monitor = 0.0  # Timestamp of last self-evaluation
        # Check agent-specific self-monitoring config
        # Config here is the 'agent' section, so self_monitoring is nested inside it
        self_monitoring_config = config.get('self_monitoring', {})
        agent_self_monitor_config = self_monitoring_config.get(agent_id, {})
        self.self_monitor_enabled = agent_self_monitor_config.get('enabled', False)
        logger.debug(f"[INIT] agent_id={agent_id}, self_monitoring_config keys={list(self_monitoring_config.keys())}, agent_config={agent_self_monitor_config}, enabled={self.self_monitor_enabled}")

        # Training data collector (optional - can be disabled in config)
        if config.get('collect_training_data', True):
            self.training_collector = TrainingDataCollector(
                data_dir='../../training/data/cmush_real'
            )
            self.training_collector.start_session()
        else:
            self.training_collector = None

        # Agent filesystem (sandboxed file operations)
        filesystem_config = config.get('filesystem', {})
        self.filesystem = AgentFilesystem(
            agent_id=agent_id,
            base_path='world/agents',
            config=filesystem_config
        )

        # Agent messaging (inbox/outbox)
        messaging_config = config.get('messaging', {})
        self.messaging = AgentMessaging(
            base_path='world/agents',
            config=messaging_config
        )

        # Autonomous cognition engine
        cognition_config = config.get('autonomous_cognition', {}).copy()

        # Use personality traits already loaded above
        cognition_config['personality'] = self.personality

        self.cognition_enabled = cognition_config.get('enabled', True)
        if self.cognition_enabled:
            self.cognition_engine = AutonomousCognitionEngine(
                agent=self,
                config=cognition_config
            )
            logger.info(f"Cognition engine personality: {self.personality}")
        else:
            self.cognition_engine = None

        # COMPONENT SYSTEM: Initialize cognitive component registry
        from noodling_components import (
            ComponentRegistry,
            CharacterVoiceComponent,
            IntuitionReceiverComponent,
            SocialExpectationDetectorComponent
        )

        self.components = ComponentRegistry(agent_id, self.agent_name)

        # Register Character Voice component
        voice_config = config.get('character_voice', {
            'enabled': True,
            'model': 'qwen/qwen3-4b-2507',
            'temperature': 0.4,
            'max_tokens': 150
        })
        character_voice = CharacterVoiceComponent(
            agent_id=agent_id,
            agent_name=self.agent_name,
            config=voice_config,
            species=self.species,
            llm=self.llm
        )
        self.components.register(character_voice)

        # Register Intuition Receiver component
        intuition_config = config.get('intuition_receiver', {})
        if intuition_config.get('enabled', True):
            intuition = IntuitionReceiverComponent(
                agent_id=agent_id,
                agent_name=self.agent_name,
                config=intuition_config,
                llm=self.llm
            )
            self.components.register(intuition)

        # Register Social Expectation Detector component
        social_expectations_config = intuition_config.get('social_expectations', {})
        if social_expectations_config.get('enabled', True):
            social_expectation = SocialExpectationDetectorComponent(
                agent_id=agent_id,
                agent_name=self.agent_name,
                config=social_expectations_config,
                llm=self.llm
            )
            self.components.register(social_expectation)

        logger.info(f"Agent initialized: {agent_id} (extraversion={extraversion:.2f}, threshold={adjusted_threshold:.6f})")
        logger.info(f"[{agent_id}] Registered {len(self.components.components)} cognitive components")

    def _score_identity_salience(self, response_text: str, surprise: float) -> float:
        """
        Score how characteristic/in-character a response is.

        Higher scores indicate the agent is acting strongly in-character.
        These memories will be retrieved as "identity anchors" to maintain consistency.

        Args:
            response_text: The agent's response text
            surprise: Current surprise level

        Returns:
            Identity salience score (0.0 to 1.0)
        """
        text_lower = response_text.lower()
        salience = 0.0

        # High surprise often indicates characteristic reactions
        if surprise > 0.5:
            salience += 0.3

        # Species-specific behaviors
        if self.species == 'kitten':
            # Nonverbal kitten behaviors
            if any(word in text_lower for word in ['meow', 'purr', 'mew', 'chirp', 'hiss']):
                salience += 0.4
            if any(word in text_lower for word in ['rub', 'curl', 'pounce', 'bat at', 'groom']):
                salience += 0.3

        elif self.species == 'toad':
            # Mr. Toad's characteristic phrases
            if 'poop-poop' in text_lower:
                salience += 0.5  # Very characteristic!
            if any(word in text_lower for word in ['motor', 'automobile', 'vehicle', 'drive']):
                salience += 0.3
            if any(word in text_lower for word in ['reckless', 'daring', 'adventure', 'excitement']):
                salience += 0.2

        # Check for emote actions (indicates behavioral engagement)
        if ':' in response_text:
            salience += 0.2

        # Strong emotional expressions
        if any(word in text_lower for word in ['!', 'wonder', 'delightful', 'curious', 'fascinating']):
            salience += 0.1

        # Cap at 1.0
        return min(salience, 1.0)

    async def _send_to_noodlescope(self, phenomenal_state, surprise, identity_salience=0.0):
        """
        Send phenomenal state to NoodleScope for visualization.

        Args:
            phenomenal_state: Full 40-D state array
            surprise: Current surprise value
            identity_salience: Current identity salience
        """
        if not NOODLESCOPE_ENABLED:
            return

        try:
            # Convert to list if needed
            if hasattr(phenomenal_state, 'tolist'):
                phenomenal_state = phenomenal_state.tolist()
            else:
                phenomenal_state = list(phenomenal_state)

            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{NOODLESCOPE_URL}/api/update_state",
                    json={
                        'agent_id': self.agent_id,
                        'phenomenal_state': phenomenal_state,
                        'surprise': float(surprise),
                        'identity_salience': float(identity_salience)
                    },
                    timeout=aiohttp.ClientTimeout(total=0.5)  # Don't block on viz
                )
        except Exception as e:
            # Silently fail - NoodleScope is optional
            logger.debug(f"NoodleScope update failed: {e}")

    async def _log_to_noodlescope(self, event_type, text):
        """
        Log an event to NoodleScope.

        Args:
            event_type: Event type (surprise_spike, name_mentioned, etc.)
            text: Event description
        """
        if not NOODLESCOPE_ENABLED:
            return

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{NOODLESCOPE_URL}/api/log_event",
                    json={
                        'agent_id': self.agent_id,
                        'event_type': event_type,
                        'text': text
                    },
                    timeout=aiohttp.ClientTimeout(total=0.5)
                )
        except Exception as e:
            logger.debug(f"NoodleScope event log failed: {e}")

    def _detects_invitation(self, text: str) -> bool:
        """
        Detect if text contains a movement invitation phrase.

        Args:
            text: User's text

        Returns:
            True if this is an invitation to move together
        """
        text_lower = text.lower()
        invitation_phrases = [
            "let's go",
            "let's head",
            "let's walk",
            "come with me",
            "follow me",
            "want to go",
            "want to come",
            "shall we go",
            "let's check out",
            "let's visit",
            "wanna go",
            "wanna come"
        ]

        return any(phrase in text_lower for phrase in invitation_phrases)

    def _normalize_affect(self, affect_vector: np.ndarray, target_variance: float = 0.25) -> np.ndarray:
        """
        Normalize affect vector to target variance.

        Based on Φ optimization research: variance 0.1-0.3 optimal for integration.
        This normalization increases Φ by ~82.6% without disrupting architecture.

        Args:
            affect_vector: Raw 5-D affect vector [valence, arousal, fear, sorrow, boredom]
            target_variance: Target variance (default 0.25 = optimal)

        Returns:
            Normalized affect vector with controlled variance
        """
        affect_array = np.array(affect_vector)

        # Normalize to zero mean, unit variance
        mean = np.mean(affect_array)
        std = np.std(affect_array) + 1e-8  # Avoid division by zero

        normalized = (affect_array - mean) / std

        # Scale to target variance
        normalized = normalized * np.sqrt(target_variance)

        return normalized

    def _trigger_memories_by_names(self, text: str) -> List[Dict]:
        """
        Search for entity names in text and retrieve associated memories.

        Args:
            text: Input text to search for names

        Returns:
            List of relevant memories (especially high-salience ones)
        """
        import re

        # Extract potential names (capitalized words, excluding common words)
        common_words = {'I', 'You', 'The', 'A', 'An', 'And', 'Or', 'But', 'If', 'When', 'Where', 'Why', 'How',
                        'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Is', 'Are', 'Was', 'Were', 'Will', 'Would',
                        'Could', 'Should', 'May', 'Might', 'Can', 'Could', 'That', 'This', 'These', 'Those',
                        'What', 'Which', 'Who', 'Whom', 'Whose', 'My', 'Your', 'His', 'Her', 'Its', 'Our',
                        'Their', 'He', 'She', 'It', 'We', 'They', 'Me', 'Him', 'Us', 'Them', 'There', 'Here'}
        word_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = [name for name in re.findall(word_pattern, text) if name not in common_words]

        if not potential_names:
            return []

        # Search memories for these names
        triggered_memories = []
        for name in potential_names:
            name_lower = name.lower()
            for memory in self.conversation_context:
                memory_text = memory.get('text', '').lower()
                if name_lower in memory_text:
                    # Prioritize high-salience memories
                    triggered_memories.append(memory)

        # Sort by identity_salience and return top 3
        triggered_memories = sorted(
            triggered_memories,
            key=lambda m: m.get('identity_salience', 0),
            reverse=True
        )[:3]

        return triggered_memories

    def _apply_memory_affect(self, memories: List[Dict], current_affect: np.ndarray) -> np.ndarray:
        """
        Blend affect from memories into current affect state.

        Memories with higher identity_salience have stronger influence.

        Args:
            memories: List of memory dicts containing 'affect' and 'identity_salience'
            current_affect: Current affect vector (5-D)

        Returns:
            Blended affect vector
        """
        if not memories:
            return current_affect

        # Extract affect vectors and weights from memories
        memory_affects = []
        weights = []

        for mem in memories:
            affect = mem.get('affect')
            salience = mem.get('identity_salience', 0.0)

            if affect is not None and salience > 0:
                # Convert affect to numpy array if needed
                if hasattr(affect, 'tolist'):
                    affect = np.array(affect.tolist())
                else:
                    affect = np.array(affect)

                # Ensure affect has correct shape (5-D)
                if len(affect) >= 5:
                    memory_affects.append(affect[:5])  # Take first 5 dimensions
                    # Weight by identity_salience squared (stronger memories have more influence)
                    weights.append(salience ** 2)

        if not memory_affects:
            return current_affect

        # Weighted average of memory affects
        memory_affect_blend = np.average(memory_affects, weights=weights, axis=0)

        # Ensure blend has same shape as current_affect before adding
        if len(memory_affect_blend) != len(current_affect):
            logger.warning(f"Memory affect blend shape mismatch: {len(memory_affect_blend)} vs {len(current_affect)}, skipping blend")
            return current_affect

        # Blend memory affect with current affect (70% current, 30% memory)
        # This ensures memories influence but don't dominate
        blended_affect = 0.7 * current_affect + 0.3 * memory_affect_blend

        logger.info(f"Memory affect blending: {len(memories)} memories triggered, influence={0.3 * np.mean(weights):.3f}")

        return blended_affect

    def _detect_emotional_contagion(self, text: str) -> Optional[Dict]:
        """
        Detect emotional contagion patterns in text.

        Returns affect modifications for contagious emotions:
        - Laughter → increased valence, arousal
        - Yawning/Sleepiness → increased boredom
        - Surprise expressions → already handled by surprise mechanism
        - Fear/Anxiety → increased fear

        Args:
            text: Input text

        Returns:
            Dict with affect modifications or None
        """
        text_lower = text.lower()

        # Laughter detection
        laughter_patterns = ['haha', 'hehe', 'lol', 'laughs', 'giggle', 'chuckle', '*laugh*']
        if any(pattern in text_lower for pattern in laughter_patterns):
            return {
                'type': 'laughter',
                'valence_boost': 0.15,
                'arousal_boost': 0.1
            }

        # Yawning/Sleepiness detection
        sleepy_patterns = ['yawn', '*yawns*', 'sleepy', 'tired', 'exhausted']
        if any(pattern in text_lower for pattern in sleepy_patterns):
            return {
                'type': 'sleepiness',
                'boredom_boost': 0.12,
                'arousal_decrease': 0.08
            }

        # Fear/Anxiety contagion
        fear_patterns = ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous']
        if any(pattern in text_lower for pattern in fear_patterns):
            return {
                'type': 'fear',
                'fear_boost': 0.18,
                'arousal_boost': 0.12
            }

        # Sadness contagion
        sadness_patterns = ['crying', 'sobbing', 'tears', 'heartbroken', 'devastated']
        if any(pattern in text_lower for pattern in sadness_patterns):
            return {
                'type': 'sadness',
                'sorrow_boost': 0.20,
                'valence_decrease': 0.15
            }

        # Playfulness/Excitement contagion
        # Detects games, jumping, running, playing, excited exclamations
        playful_patterns = [
            'jump', 'jumps', 'jumping', 'hop', 'hops', 'hopping',
            'run', 'runs', 'running', 'dance', 'dances', 'dancing',
            'play', 'plays', 'playing', 'game', 'flap', 'flaps', 'flapping',
            'yay!', 'woohoo', 'woo!', 'wheee', 'fun!', 'exciting'
        ]
        # Check for multiple exclamation marks (excitement indicator)
        has_excitement = '!!' in text or '!!!' in text
        has_playful_words = any(pattern in text_lower for pattern in playful_patterns)

        if has_playful_words or has_excitement:
            return {
                'type': 'playfulness',
                'valence_boost': 0.20,
                'arousal_boost': 0.25,
                'boredom_decrease': 0.30  # Playfulness strongly reduces boredom
            }

        return None

    async def _generate_intuition(
        self,
        event: Dict,
        world_state: Optional[Dict] = None,
        recent_context: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Generate contextual intuition using fast LLM analysis.

        The Intuition Receiver acts like a radio tuned to contextual signals,
        providing natural awareness of:
        - Message routing (who is being addressed)
        - Spatial relationships (who is where)
        - Prop tracking (who has what)
        - Recent actions (what just happened)

        This creates integrated contextual awareness rather than external scaffolding.

        Args:
            event: Current event being perceived
            world_state: Optional world state dictionary with room/agent/object info
            recent_context: Recent conversation context (last 3-5 messages)

        Returns:
            Intuitive awareness string, or None if intuition disabled/failed
        """
        # Check if intuition receiver is enabled
        intuition_config = self.config.get('intuition_receiver', {})
        if not intuition_config.get('enabled', True):
            return None

        try:
            # Extract event details
            event_type = event.get('type', 'say')
            speaker_id = event.get('user', '')
            message_text = event.get('text', '')
            room_id = event.get('room', '')

            # Build context for intuition analysis
            context_info = []

            # 1. WHO IS SPEAKING
            speaker_name = speaker_id.replace('agent_', '').replace('user_', '').title()
            context_info.append(f"Speaker: {speaker_name} ({speaker_id})")

            # 2. RECENT CONVERSATION FLOW
            if recent_context:
                recent_speakers = []
                for entry in recent_context[-3:]:
                    speaker = entry.get('user', '').replace('agent_', '').replace('user_', '').title()
                    text_snippet = entry.get('text', '')[:50]
                    recent_speakers.append(f"{speaker}: {text_snippet}")
                context_info.append(f"Recent conversation:\n" + "\n".join(recent_speakers))

            # 3. CURRENT MESSAGE
            context_info.append(f"Current message: '{message_text}'")

            # 4. ONGOING GAMES / EXPECTATIONS
            # Detect if there's an active game or thing people are waiting for
            if recent_context:
                # Look for secret word games, memory games, etc.
                game_mentions = []
                for entry in recent_context[-10:]:  # Last 10 messages
                    text_lower = entry.get('text', '').lower()
                    if 'secret word' in text_lower or 'magic word' in text_lower:
                        game_mentions.append("There's a secret word game active")
                    if 'memory game' in text_lower:
                        game_mentions.append("There's a memory game happening")
                    if 'waiting for' in text_lower or 'ready for' in text_lower:
                        game_mentions.append("People are waiting for something to happen")

                if game_mentions:
                    context_info.append(f"Active game/expectation: {', '.join(set(game_mentions))}")

            # 4. WORLD STATE (if available)
            if world_state:
                # Room occupants with species/metadata
                room = world_state.get('rooms', {}).get(room_id, {})
                all_occupants = room.get('occupants', [])

                # Filter out invisible users (admin stealth mode)
                occupants = []
                for occ_id in all_occupants:
                    if occ_id.startswith('user_'):
                        user_data = world_state.get('users', {}).get(occ_id, {})
                        if user_data.get('invisible', False):
                            continue  # Skip invisible admin users
                    occupants.append(occ_id)

                if occupants:
                    occupant_details = []
                    for occ_id in occupants:
                        occ_name = occ_id.replace('agent_', '').replace('user_', '').title()

                        # Get agent metadata if available
                        if occ_id.startswith('agent_'):
                            agent_data = world_state.get('agents', {}).get(occ_id, {})
                            config = agent_data.get('config', {})
                            species = config.get('species', 'noodling')
                            age = config.get('age', 'unknown')
                            pronoun = config.get('pronoun', 'they')

                            # Infer pronoun from common character names if not specified
                            if pronoun == 'they':
                                name_lower = occ_name.lower()
                                if name_lower in ['phi', 'callie', 'desobelle']:
                                    pronoun = 'she'
                                elif name_lower in ['toad', 'mr. toad', 'phido']:
                                    pronoun = 'he'
                                elif name_lower in ['servnak']:
                                    pronoun = 'they'  # SERVNAK is non-binary robot

                            # Build descriptive string with useful metadata
                            details = f"{occ_name} ({species}, {age}, {pronoun})"
                            occupant_details.append(details)
                        else:
                            # Get user metadata
                            user_data = world_state.get('users', {}).get(occ_id, {})
                            species = user_data.get('species', 'human')
                            age = user_data.get('age', 'unknown')
                            pronoun = user_data.get('pronoun', 'they')
                            details = f"{occ_name} ({species}, {age}, {pronoun})"
                            occupant_details.append(details)

                    context_info.append(f"Present in room: {', '.join(occupant_details)}")

                # Objects in room
                objects = room.get('objects', [])
                if objects:
                    object_list = []
                    for obj_id in objects[:5]:  # Limit to 5 objects
                        obj = world_state.get('objects', {}).get(obj_id, {})
                        obj_name = obj.get('name', obj_id)
                        object_list.append(obj_name)
                    context_info.append(f"Objects nearby: {', '.join(object_list)}")

                # Agent inventories (who has what)
                agents_with_items = []
                for agent_id in [occ for occ in occupants if occ.startswith('agent_')]:
                    agent_data = world_state.get('agents', {}).get(agent_id, {})
                    inventory = agent_data.get('inventory', [])
                    if inventory:
                        agent_name = agent_id.replace('agent_', '').title()
                        items = []
                        for item_id in inventory[:3]:  # Limit to 3 items per agent
                            obj = world_state.get('objects', {}).get(item_id, {})
                            items.append(obj.get('name', item_id))
                        agents_with_items.append(f"{agent_name} has: {', '.join(items)}")
                if agents_with_items:
                    context_info.append("Possessions:\n" + "\n".join(agents_with_items))

            # Build intuition prompt
            context_text = "\n\n".join(context_info)

            my_name = self.agent_name

            prompt = f"""You are {my_name}'s intuitive awareness - like a narrator highlighting what's happening.

CONTEXT:
{context_text}

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

Write in first-person as {my_name}, like a perceptive narrator.
Be concise but highlight important moments others might miss!

Examples:
- "That greeting is for Toad, not me. They're by the pond while I'm near the hedge."
- "Caity just gave ME a tensor taffy! I notice the message said 'you' which means me."
- "WAIT - Toad just said the secret word 'KITTEN'! Everyone was waiting for this!"
- "Callie is asking everyone a question. The conversation is about the memory game."

Generate intuitive awareness:"""

            # Use agent's model if specified, otherwise fall back to fast model
            # This honors per-agent llm_override settings
            intuition_model = self.llm_model or intuition_config.get('model', 'qwen/qwen3-4b-2507')
            timeout = intuition_config.get('timeout', 5)

            # Track this operation
            tracker = get_tracker()
            with tracker.track_operation(
                self.agent_id,
                "intuition_generation",
                {"event_type": event_type, "speaker": speaker_id}
            ):
                # Generate intuition using fast LLM
                intuition = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=f"You are {my_name}'s intuitive contextual awareness. Be brief and natural.",
                    model=intuition_model,
                    temperature=0.3,  # Low temperature for consistent analysis
                    max_tokens=150
                )

                logger.info(f"[{self.agent_id}] Intuition generated: {intuition[:100]}...")
                return intuition.strip()

        except Exception as e:
            logger.warning(f"[{self.agent_id}] Intuition generation failed: {e}")
            return None

    async def _detect_social_expectation(
        self,
        event: Dict,
        intuition: str,
        world_state: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Analyze if interaction creates a social response expectation.

        This adds a second layer to the intuition system - detecting when
        the noodling is socially expected to respond. Creates an "itch" -
        a conscious awareness of obligation without removing agency.

        Args:
            event: Current event being perceived
            intuition: Generated intuitive awareness string
            world_state: Optional world state dictionary

        Returns:
            Dict with expectation analysis:
                {
                    'expected': bool,           # Is response expected?
                    'urgency': float (0.0-1.0), # How urgent?
                    'reason': str,              # Why expected?
                    'type': str                 # question/gesture/greeting/distress/turn/none
                }
            Or None if detection disabled/failed
        """
        # Check if social expectations are enabled
        intuition_config = self.config.get('intuition_receiver', {})
        expectation_config = intuition_config.get('social_expectations', {})

        if not expectation_config.get('enabled', True):
            return None

        try:
            # Extract event details
            event_type = event.get('type', 'say')
            speaker_id = event.get('user', '')
            message_text = event.get('text', '')

            # Build analysis prompt
            prompt = f"""Analyze this interaction for social response expectations.

INTUITIVE CONTEXT:
{intuition}

INTERACTION:
Speaker: {speaker_id}
Message: "{message_text}"
Type: {event_type}

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

            # Use fast model for analysis
            model = intuition_config.get('model', 'qwen/qwen3-4b-2507')
            timeout = expectation_config.get('timeout', 5)

            # Track operation
            tracker = get_tracker()
            with tracker.track_operation(
                self.agent_id,
                "expectation_detection",
                {"event_type": event_type, "speaker": speaker_id}
            ):
                # Generate analysis with JSON mode
                result_text = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=f"You are a social expectation analyzer. Output only valid JSON.",
                    model=model,
                    temperature=0.2,  # Very low for consistent analysis
                    max_tokens=100
                )

                # Parse JSON result
                import json
                # Strip markdown code blocks if present
                result_text = result_text.strip()
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                result_text = result_text.strip()

                result = json.loads(result_text)

                # Validate structure
                if not all(k in result for k in ['expected', 'urgency', 'reason', 'type']):
                    logger.warning(f"[{self.agent_id}] Invalid expectation result format")
                    return None

                # Apply personality modulation
                personality = getattr(self, 'personality_traits', {})
                extraversion = personality.get('extraversion', 0.5)
                social_orientation = personality.get('social_orientation', 0.5)

                # High extraversion = lower threshold for response
                # High social_orientation = higher urgency multiplier
                intensity_multiplier = expectation_config.get('intensity_multiplier', 1.0)

                # Modulate urgency based on personality
                base_urgency = float(result['urgency'])
                modulated_urgency = base_urgency * (0.7 + extraversion * 0.3)  # 0.7-1.0x range
                modulated_urgency *= (0.8 + social_orientation * 0.4)  # 0.8-1.2x range
                modulated_urgency *= intensity_multiplier
                modulated_urgency = min(1.0, modulated_urgency)  # Cap at 1.0

                result['urgency'] = modulated_urgency
                result['base_urgency'] = base_urgency

                # Log detection
                if result['expected']:
                    logger.info(f"[{self.agent_id}] ⚠️ Social expectation: {result['type']} "
                              f"(urgency: {modulated_urgency:.2f}, reason: {result['reason']})")
                else:
                    logger.debug(f"[{self.agent_id}] No social expectation detected")

                return result

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.agent_id}] Failed to parse expectation JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Expectation detection failed: {e}")
            return None

    async def _generate_facial_expression(self, affect: np.ndarray, force: bool = False) -> Optional[Dict]:
        """
        Generate FACS-based facial expression from current affect.

        Args:
            affect: 5-D affect vector [valence, arousal, fear, sorrow, boredom]
            force: Force expression generation (ignore cooldown)

        Returns:
            Dict with facial expression event, or None if no expression
        """
        if not FACS_ENABLED:
            return None

        # Check cooldown
        time_since_last = time.time() - self.last_facial_expression_time
        if not force and time_since_last < FACS_COOLDOWN:
            return None

        # Check if affect has changed enough to warrant facial expression
        if self.previous_affect is not None:
            affect_diff = np.linalg.norm(affect - self.previous_affect)
            if not force and affect_diff < FACS_THRESHOLD:
                return None  # Affect hasn't changed enough

        # Generate FACS codes from affect
        facs_codes = affect_to_facs(affect)

        if not facs_codes:
            return None  # No expression to generate

        # Get human-readable description
        face_description = facs_to_description(facs_codes)

        # Generate body language codes
        body_codes = affect_to_body_language(affect, species=self.species)
        body_description = body_language_to_description(body_codes)

        # Format for renderer (future 3D integration)
        facs_renderer_data = format_facs_for_renderer(facs_codes)
        body_renderer_data = format_body_language_for_renderer(body_codes)

        # Combine descriptions
        combined_description = face_description
        if body_codes and body_description != "stands still":
            combined_description = f"{face_description}, {body_description}"

        # Update tracking
        self.last_facial_expression_time = time.time()
        self.previous_affect = affect.copy()

        # Log the expression
        logger.info(f"[{self.agent_id}] Full expression: {combined_description}")
        logger.debug(f"[{self.agent_id}] FACS: {facs_codes}")
        logger.debug(f"[{self.agent_id}] BODY: {body_codes}")

        return {
            'type': 'full_expression',
            'description': combined_description,
            'facs_codes': facs_codes,
            'body_codes': body_codes,
            'renderer_data': {
                'face': facs_renderer_data,
                'body': body_renderer_data
            },
            'affect': affect.tolist()
        }

    async def perceive_event(self, event: Dict) -> Optional[Dict]:
        """
        Process cMUSH event -> Consilience -> optional response.

        Args:
            event: Dictionary with:
                - type: 'say' | 'emote' | 'enter' | 'exit'
                - user: User/agent ID
                - text: Text content (for say/emote)
                - room: Room ID

        Returns:
            None or response dict:
                {
                    'command': 'say' | 'emote',
                    'text': '...',
                    'metadata': {...}
                }
        """
        event_type = event.get('type')
        user_id = event.get('user')
        text = event.get('text', '')
        room_id = event.get('room')

        # Skip if not a perceivable event
        if event_type not in ['say', 'emote', 'enter', 'exit']:
            logger.debug(f"Skipping non-perceivable event: {event_type}")
            return None

        # Skip if self-action
        if user_id == self.agent_id:
            return None

        # Agents can now perceive other agents
        is_agent = user_id.startswith('agent_')

        logger.info(f"Agent {self.agent_id} perceiving: {event_type} from {user_id}: {text}")

        try:
            # Log instant event - stimulus received
            tracker = get_tracker()
            tracker.log_instant_event(
                self.agent_id,
                "stimulus_received",
                {"from": user_id, "event_type": event_type}
            )

            # 1. Text -> Affect (via LLM)
            # SPECIAL CASE: Own spawn event - override with positive welcoming affect!
            if event_type == 'enter' and user_id == self.agent_id:
                # "I just came into being! Warm curiosity and wonder!"
                affect_raw = [0.5, 0.5, 0.0, 0.0, 0.0]  # Positive, moderately aroused, no fear/sorrow/boredom
                logger.info(f"[{self.agent_id}] 🌟 First moment of existence - setting welcoming affect!")
            else:
                # Use configurable memory window for affect extraction
                affect_window = self.config.get('memory_windows', {}).get('affect_extraction', 3)
                context = [c['text'] for c in self.conversation_context[-affect_window:]]
                affect_raw = await self.llm.text_to_affect(text, context, agent_id=self.agent_id)

            # Log affect extraction for debugging
            logger.info(f"[{self.agent_id}] 🎨 AFFECT EXTRACTED: valence={affect_raw[0]:.3f}, arousal={affect_raw[1]:.3f}, fear={affect_raw[2]:.3f}, sorrow={affect_raw[3]:.3f}, boredom={affect_raw[4]:.3f}")
            logger.debug(f"Extracted affect (raw): {affect_raw}")

            # 1a. Detect name mention - boosts attention/salience
            name_mentioned = self.agent_name.lower() in text.lower()
            if name_mentioned:
                # Log instant event - name mentioned
                tracker.log_instant_event(
                    self.agent_id,
                    "name_mentioned",
                    {"from": user_id, "text_snippet": text[:50]}
                )

                # Boost arousal when hearing own name (attention mechanism)
                affect_raw[1] = min(1.0, affect_raw[1] + 0.2)  # arousal index
                logger.info(f"Agent {self.agent_id} heard their name - arousal boosted")

                # Notify autonomous cognition that agent was directly addressed
                if hasattr(self, 'autonomous_cognition') and self.autonomous_cognition:
                    self.autonomous_cognition.on_directly_addressed()

            # 1a-2. Notify autonomous cognition of any stimulus (for boredom tracking)
            if hasattr(self, 'autonomous_cognition') and self.autonomous_cognition:
                self.autonomous_cognition.on_stimulus_received()

            # 1a-3. GARBAGE AFFECT DETECTION (before normalization!)
            # Check for LLM fallback pattern: very low valence/arousal/fear
            # This is the REAL fix - catch it before normalization transforms it!
            if (affect_raw[0] <= 0.1 and  # Very low valence
                affect_raw[1] <= 0.4 and  # Low arousal
                affect_raw[2] <= 0.2):    # Low fear
                # This is likely LLM fallback affect [0.0, 0.3, 0.1, 0.1, 0.1]
                # Override with welcoming/curious affect
                affect_raw = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
                logger.info(f"[{self.agent_id}] ⚠️ Detected fallback affect - overriding with welcoming values")

            # 1b. Normalize affect for optimal Φ (research-validated optimization)
            affect = self._normalize_affect(affect_raw, target_variance=0.25)

            logger.debug(f"Normalized affect: {affect}")

            # 1c. NAME-BASED MEMORY TRIGGERING
            # Search for names in text and retrieve associated memories
            triggered_memories = self._trigger_memories_by_names(text)
            if triggered_memories:
                logger.info(f"Agent {self.agent_id} triggered {len(triggered_memories)} memories by names in: '{text[:50]}'")
                # Blend memory affect into current affect
                affect = self._apply_memory_affect(triggered_memories, affect)

            # 1d. EMOTIONAL CONTAGION
            # Detect contagious emotions (laughter, yawning, fear, etc.)
            contagion = self._detect_emotional_contagion(text)
            if contagion:
                contagion_type = contagion['type']
                logger.info(f"Agent {self.agent_id} experiencing emotional contagion: {contagion_type}")

                # Apply contagion affects
                # affect indices: [0: valence, 1: arousal, 2: fear, 3: sorrow, 4: boredom]
                if 'valence_boost' in contagion:
                    affect[0] = min(1.0, affect[0] + contagion['valence_boost'])
                if 'valence_decrease' in contagion:
                    affect[0] = max(-1.0, affect[0] - contagion['valence_decrease'])
                if 'arousal_boost' in contagion:
                    affect[1] = min(1.0, affect[1] + contagion['arousal_boost'])
                if 'arousal_decrease' in contagion:
                    affect[1] = max(0.0, affect[1] - contagion['arousal_decrease'])
                if 'fear_boost' in contagion:
                    affect[2] = min(1.0, affect[2] + contagion['fear_boost'])
                if 'sorrow_boost' in contagion:
                    affect[3] = min(1.0, affect[3] + contagion['sorrow_boost'])
                if 'boredom_boost' in contagion:
                    affect[4] = min(1.0, affect[4] + contagion['boredom_boost'])
                if 'boredom_decrease' in contagion:
                    affect[4] = max(0.0, affect[4] - contagion['boredom_decrease'])
                    # Also notify autonomous cognition to reduce accumulated boredom
                    if hasattr(self, 'autonomous_cognition') and self.autonomous_cognition:
                        self.autonomous_cognition.boredom = max(0.0, self.autonomous_cognition.boredom * 0.5)

            # 2. Affect -> Consilience state
            state = self.consciousness.perceive(
                affect_vector=affect,
                agent_id=user_id,
                user_text=text,
                present_agents=[user_id]
            )

            # 2a. Check event metadata early (needed for context storage)
            event_metadata = event.get('metadata', {})
            is_cue = event_metadata.get('cue', False)

            # 3. Store context (identity_salience will be added when agent responds)
            context_entry = {
                'user': user_id,
                'text': text,
                'affect': affect,
                'surprise': state['surprise'],
                'timestamp': time.time(),
                'identity_salience': 0.0  # Only agent's own responses get high salience
            }

            # Add cue metadata if this is a stage direction
            if is_cue and event_metadata.get('direction'):
                context_entry['stage_cue'] = event_metadata['direction']
                # Also add motivation if provided (character's WHY)
                if event_metadata.get('motivation'):
                    context_entry['stage_motivation'] = event_metadata['motivation']
                    logger.info(f"Added stage cue to context: {event_metadata['direction']} (motivation: {event_metadata['motivation']})")
                else:
                    logger.info(f"Added stage cue to context: {event_metadata['direction']}")

            self.conversation_context.append(context_entry)

            # Trim context (use configurable threshold)
            trim_threshold = self.config.get('memory_windows', {}).get('affect_trim_threshold', 20)
            if len(self.conversation_context) > trim_threshold:
                self.conversation_context = self.conversation_context[-trim_threshold:]

            # Save state handled by periodic auto-save in AgentManager
            # (Incremental save after every event would be too expensive)

            # 3a. Detect movement invitations ("let's go to...")
            if event_type == 'say' and self._detects_invitation(text):
                self.following = user_id
                logger.info(f"Agent {self.agent_id} now following {user_id} (invitation detected)")

            # 3b. Handle exit events - follow if we're following this user
            if event_type == 'exit' and self.following == user_id:
                # Return a follow response that will trigger movement
                direction = event.get('direction', 'north')
                logger.info(f"Agent {self.agent_id} following {user_id} {direction}")
                return {
                    'command': 'follow',
                    'text': f"follows {direction}.",
                    'direction': direction,
                    'metadata': {
                        'following': user_id,
                        'surprise': state['surprise']
                    }
                }

            # 4. Track phenomenal states for consciousness metrics
            # Extract full 40-D phenomenal state (fast 16-D + medium 16-D + slow 8-D)
            h_fast = state.get('fast_state') or []
            h_medium = state.get('medium_state') or []
            h_slow = state.get('slow_state') or []

            # Convert to numpy arrays if needed
            if hasattr(h_fast, 'tolist'):
                h_fast = h_fast.tolist()
            if hasattr(h_medium, 'tolist'):
                h_medium = h_medium.tolist()
            if hasattr(h_slow, 'tolist'):
                h_slow = h_slow.tolist()

            # Combine into full 40-D phenomenal state
            phenomenal_state_vector = np.array(h_fast + h_medium + h_slow)

            # Store in state dict for session profiler
            state['phenomenal_state'] = phenomenal_state_vector

            self.state_history.append(phenomenal_state_vector)
            self.surprise_history.append(state['surprise'])

            # Trim history to last 1000 entries for memory management
            if len(self.state_history) > 1000:
                self.state_history = self.state_history[-1000:]
                self.surprise_history = self.surprise_history[-1000:]

            # Send to NoodleScope
            await self._send_to_noodlescope(phenomenal_state_vector, state['surprise'], 0.0)

            # Log high surprise events
            surprise_threshold = state.get('surprise_threshold', self.config.get('surprise_threshold', 0.3))
            if state['surprise'] > surprise_threshold * 1.5:
                await self._log_to_noodlescope('surprise_spike', f"High surprise: {state['surprise']:.3f}")

            # FACS: Generate facial expression based on affect
            # DEBUG: Log affect values to diagnose anger/stomp issue
            logger.info(f"[{self.agent_id}] 🎭 Affect for FACS: valence={affect[0]:.3f}, arousal={affect[1]:.3f}, fear={affect[2]:.3f}, sorrow={affect[3]:.3f}, boredom={affect[4]:.3f}")

            facial_expression = await self._generate_facial_expression(affect)
            if facial_expression and state['surprise'] > 0.02:  # Show if any notable surprise (lowered to catch more reactions)
                # Store the facial expression for potential 3D renderer integration
                self.last_facs_data = facial_expression['renderer_data']

                # Format for chat display
                # Format: *eyes wide, waddles nervously* [FACE: AU1, AU2 | BODY: BL44, BL14]
                facs_codes_str = ", ".join([f"AU{au}" for au, _ in facial_expression['facs_codes'][:4]])
                body_codes_str = ", ".join([f"BL{bl}" for bl, _ in facial_expression.get('body_codes', [])[:3]])

                if body_codes_str:
                    expression_text = f"*{facial_expression['description']}* [FACE: {facs_codes_str} | BODY: {body_codes_str}]"
                else:
                    expression_text = f"*{facial_expression['description']}* [FACE: {facs_codes_str}]"

                logger.info(f"[{self.agent_id}] Full Expression triggered: {expression_text}")

                # Store full expression to be returned
                state['facial_expression'] = expression_text
                state['expression_data'] = facial_expression['renderer_data']

            # EVENT-DRIVEN COGNITION: Notify autonomous cognition of surprise
            if hasattr(self, 'autonomous_cognition') and self.autonomous_cognition:
                self.autonomous_cognition.on_surprise(state['surprise'])

            logger.debug(f"Surprise: {state['surprise']:.3f} (threshold: {surprise_threshold:.3f})")

            # Log to session profiler (for every event, not just speech)
            logger.info(f"[{self.agent_id}] PROFILER CHECK: hasattr={hasattr(self, 'session_profiler')}, value={getattr(self, 'session_profiler', None)}")
            if hasattr(self, 'session_profiler') and self.session_profiler:
                try:
                    # Extract affect from state or use the affect vector we calculated
                    affect_vector = np.array(affect) if not isinstance(affect, np.ndarray) else affect

                    # Extract FACS/body data if available
                    facs_data = None
                    body_data = None
                    expression_desc = None
                    if facial_expression:
                        facs_data = facial_expression.get('facs_codes', [])
                        body_data = facial_expression.get('body_codes', [])
                        expression_desc = facial_expression.get('description', '')

                    self.session_profiler.log_timestep(
                        agent_id=self.agent_id,
                        phenomenal_state=phenomenal_state_vector,
                        affect=affect_vector,
                        surprise=state['surprise'],
                        speech_threshold=surprise_threshold,
                        did_speak=False,  # Will be updated in _generate_response if agent speaks
                        utterance=None,
                        prediction_error=0.0,
                        cheap_thrills_score=0.0,
                        mysticism_penalty=0.0,
                        event_context=f"{user_id}: {text[:100]}",
                        conversation_context=self.conversation_context[-5:],  # Last 5 messages
                        facs_codes=facs_data,
                        body_codes=body_data,
                        expression_description=expression_desc,
                        event_type=event_type,
                        responding_to=user_id
                    )
                    logger.info(f"[{self.agent_id}] Logged timestep to session profiler")
                except Exception as e:
                    logger.error(f"[{self.agent_id}] Error logging to session profiler: {e}", exc_info=True)

            # 4. Log interaction for training (before response decision)
            if self.training_collector:
                try:
                    # Convert numpy arrays to lists for JSON serialization
                    h_fast = state.get('fast_state', [])
                    h_medium = state.get('medium_state', [])
                    h_slow = state.get('slow_state', [])

                    if hasattr(h_fast, 'tolist'):
                        h_fast = h_fast.tolist()
                    if hasattr(h_medium, 'tolist'):
                        h_medium = h_medium.tolist()
                    if hasattr(h_slow, 'tolist'):
                        h_slow = h_slow.tolist()

                    self.training_collector.log_interaction(
                        agent_id=self.agent_id,
                        user_id=user_id,
                        user_text=text,
                        affect_vector=affect,
                        phenomenal_state={
                            'fast': h_fast,
                            'medium': h_medium,
                            'slow': h_slow
                        },
                        surprise=state['surprise'],
                        response=None,  # Will be updated if agent responds
                        context={'room': room_id, 'event_type': event_type}
                    )
                    logger.info(f"Logged interaction for training: {user_id} -> {self.agent_id}")
                except Exception as e:
                    logger.error(f"Failed to log interaction: {e}", exc_info=True)

            # 5. Self-protection: Check if agent needs to withdraw
            # Skip if user is already withdrawn from
            if user_id in self.withdrawn_users:
                # Check if enough time has passed for re-engagement (5 minutes)
                time_since_withdrawal = time.time() - self.withdrawn_users[user_id]
                if time_since_withdrawal < 300:  # 5 minutes
                    logger.info(f"Agent {self.agent_id} is withdrawn from {user_id} (cooling off)")
                    return None
                else:
                    # Clear withdrawal - agent may try again
                    logger.info(f"Agent {self.agent_id} re-engaging with {user_id} after cooling off period")
                    del self.withdrawn_users[user_id]

            # Check if agent is in distress (negative affect thresholds)
            fast_state = state.get('fast_state')
            if fast_state is not None and len(fast_state) >= 4:
                valence = float(fast_state[0])
                fear = float(fast_state[2]) if len(fast_state) > 2 else 0.0
                sorrow = float(fast_state[3]) if len(fast_state) > 3 else 0.0

                # Thresholds for distress
                is_distressed = (
                    valence < -0.5 or  # Very negative emotion
                    fear > 0.6 or      # High fear
                    sorrow > 0.6       # High sorrow
                )

                if is_distressed:
                    logger.info(f"Agent {self.agent_id} in distress (valence={valence:.2f}, fear={fear:.2f}, sorrow={sorrow:.2f})")

                    # Call self-reflection to decide whether to withdraw
                    # Use configurable memory window for self-reflection
                    reflection_window = self.config.get('memory_windows', {}).get('self_reflection', 3)
                    reflection = await self.llm.self_reflection(
                        phenomenal_state=state,
                        conversation_context=self.conversation_context[-reflection_window:],
                        agent_name=self.agent_name,
                        agent_id=self.agent_id,
                        agent_description=self.agent_description,
                        identity_prompt=self.identity_prompt,
                        user_id=user_id
                    )

                    if not reflection.get('comfortable', True):
                        # Agent has chosen to withdraw
                        logger.warning(f"Agent {self.agent_id} withdrawing from {user_id}: {reflection.get('reason')}")

                        # Mark user as withdrawn from
                        self.withdrawn_users[user_id] = time.time()

                        # Return withdrawal message
                        withdrawal_message = reflection.get('message', 'I need to step back for a moment.')

                        return {
                            'command': 'say',
                            'text': withdrawal_message,
                            'metadata': {
                                'surprise': float(state['surprise']),
                                'withdrawn': True,
                                'reason': reflection.get('reason', 'distress')
                            }
                        }

            # 6. Evaluate if being addressed & decide whether to respond
            cooldown = self.config.get('response_cooldown', 2.0)
            time_since_last = time.time() - self.last_response_time

            # ADDRESSEE DETECTION: Check if this message is directed at this agent
            # This prevents all agents from responding to every utterance
            import random
            import re

            # Get current event text
            event_text = event.get('text', '')
            event_text_lower = event_text.lower()
            agent_name_lower = self.agent_name.lower()

            # Enhanced name mention detection with fuzzy matching
            # Helper: Levenshtein distance for fuzzy name matching
            def levenshtein(s1: str, s2: str) -> int:
                """Calculate edit distance between two strings."""
                if len(s1) < len(s2):
                    return levenshtein(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                return previous_row[-1]

            # Pattern 1: Exact match with punctuation
            escaped_name = re.escape(agent_name_lower)
            direct_address_pattern = rf'{escaped_name}\s*[,:!?]'
            is_directly_addressed = bool(re.search(direct_address_pattern, event_text_lower))

            # Pattern 2: Fuzzy match for typos/variations
            # Check words near punctuation for close matches to agent name
            if not is_directly_addressed:
                # Extract potential names before punctuation: "Hey toaD!" -> ["toaD"]
                # Support multi-word names with periods: "mr. toad!" -> ["mr. toad"]
                words_before_punct = re.findall(r'([\w\.]+(?:\s+[\w\.]+)?)\s*[,:!?]', event_text_lower)
                # Adaptive threshold: 2 edits for short names (≤5 chars), 3 for longer
                threshold = 2 if len(agent_name_lower) <= 5 else 3
                for word in words_before_punct:
                    # Normalize spaces and periods for comparison
                    word_normalized = ' '.join(word.split())
                    distance = levenshtein(word_normalized, agent_name_lower)
                    if distance <= threshold:
                        is_directly_addressed = True
                        logger.info(f"Fuzzy match: '{word_normalized}' ≈ '{agent_name_lower}' (distance={distance})")
                        break

            # Pattern 3: Name mentioned in current event (may be about them)
            event_mentions_name = agent_name_lower in event_text_lower

            # Pattern 4: Check if this is third-party discussion ABOUT the agent
            # Look for patterns like "about X", "X is", "X was", "X has", "did X", etc.
            third_party_patterns = [
                rf'about\s+{re.escape(agent_name_lower)}',
                rf'{re.escape(agent_name_lower)}\s+(is|was|has|had|did)',
                rf'(does|can)\s+{re.escape(agent_name_lower)}',
                rf'tell\s+.*\s+about\s+{re.escape(agent_name_lower)}',
                rf'what.*{re.escape(agent_name_lower)}',
                rf'where.*{re.escape(agent_name_lower)}',
            ]
            is_third_party_discussion = any(re.search(pattern, event_text_lower) for pattern in third_party_patterns)

            # Determine if being addressed (exclude third-party discussion)
            # Being addressed means: directly addressed OR name mentioned BUT NOT third-party discussion
            is_being_addressed = is_directly_addressed or (event_mentions_name and not is_third_party_discussion)

            # PLAY STIMULUS TARGETING: Check if this is a targeted stimulus from a play
            # Stimuli can target specific agents via metadata without mentioning their name in text
            # (event_metadata and is_cue already extracted earlier for context storage)
            is_stimulus = event_metadata.get('stimulus', False)
            stimulus_target = event_metadata.get('target')

            # Check if this agent is the target
            # Target can be: agent name (e.g., "toad"), agent ID (e.g., "agent_toad"), or None/null for all agents
            if is_stimulus and stimulus_target:
                # Normalize target to match agent name or ID
                target_lower = stimulus_target.lower()
                # Match if target is agent name or agent ID
                if target_lower == agent_name_lower or target_lower == self.agent_id.lower():
                    is_being_addressed = True
                    logger.info(f"Agent {self.agent_id} targeted by play stimulus: '{event_text[:50]}'")
                elif target_lower == 'all':
                    # Stimulus targets all agents in the room
                    is_being_addressed = True
                    logger.info(f"Agent {self.agent_id} included in broadcast stimulus: '{event_text[:50]}'")

            # STAGE CUE: Director is giving this agent a cue - they MUST respond!
            if is_cue and stimulus_target:
                # Check if this cue is for this agent
                target_lower = stimulus_target.lower()
                if target_lower == agent_name_lower or target_lower == self.agent_id.lower():
                    is_being_addressed = True
                    logger.info(f"🎬 Agent {self.agent_id} received STAGE CUE: '{event_text[:50]}'")

            # Check if this is a question (agents more likely to respond to questions)
            is_question = '?' in event.get('text', '')

            cooldown_ok = time_since_last >= cooldown

            # Rumination frequency - agents think most of the time they observe
            rumination_frequency = self.config.get('rumination_frequency', 0.7)  # 70% chance to ruminate

            # ADAPTIVE SPEECH CHANCE based on being addressed:
            if is_being_addressed:
                # Directly addressed - high chance to speak (80%)
                speech_chance = self.config.get('addressed_speech_chance', 0.8)
            elif is_question:
                # Question in conversation - moderate chance to chime in (30%)
                speech_chance = self.config.get('question_speech_chance', 0.3)
            else:
                # Not addressed - low chance to interject (10%)
                speech_chance = self.config.get('unaddressed_speech_chance', 0.1)

            # Always consider ruminating when observing
            should_ruminate = random.random() < rumination_frequency

            # Only speak if cooldown passed AND dice roll succeeds AND appropriate context
            # DEV: Force speech when directly addressed for testing
            if is_being_addressed and cooldown_ok:
                should_speak = True
            else:
                should_speak = cooldown_ok and (random.random() < speech_chance)

            # Log decision with addressee context
            logger.info(f"Agent {self.agent_id} decision: addressed={is_being_addressed}, "
                       f"directly_addressed={is_directly_addressed}, third_party={is_third_party_discussion}, "
                       f"question={is_question}, should_ruminate={should_ruminate}, should_speak={should_speak}, "
                       f"speech_chance={speech_chance:.2f}, cooldown_ok={cooldown_ok}, "
                       f"surprise={state.get('surprise', 0.0):.6f}")

            # INTUITION RECEIVER: Generate contextual awareness
            # This provides integrated understanding of who/what/where without external scaffolding
            intuition = None
            if self.world:  # Only if world state is available
                # Build world state snapshot for intuition
                world_snapshot = {
                    'rooms': self.world.rooms,
                    'objects': self.world.objects,
                    'agents': self.world.agents,
                    'users': self.world.users
                }

                # Generate intuition using fast LLM
                intuition = await self._generate_intuition(
                    event=event,
                    world_state=world_snapshot,
                    recent_context=self.conversation_context[-5:]
                )

                # Store intuition in state for LLM access
                if intuition:
                    state['intuition'] = intuition
                    logger.info(f"[{self.agent_id}] 📻 Intuition: {intuition[:80]}...")

                    # SOCIAL EXPECTATION DETECTION: Analyze if response is expected
                    expectation = await self._detect_social_expectation(
                        event=event,
                        intuition=intuition,
                        world_state=world_snapshot
                    )

                    if expectation:
                        state['social_expectation'] = expectation
                        logger.info(f"[{self.agent_id}] Social expectation detected: {expectation}")

                        # RECALCULATE SPEECH DECISION based on social expectation urgency
                        # If urgency is high enough, override the random speech decision
                        if expectation.get('expected', False) and cooldown_ok:
                            urgency = expectation.get('urgency', 0.0)
                            urgency_threshold = self.config.get('intuition_receiver', {}).get('social_expectations', {}).get('expectation_threshold', 0.3)

                            # If urgency exceeds threshold and cooldown passed, strongly consider speaking
                            if urgency >= urgency_threshold:
                                # High urgency (>0.7) = force speech
                                # Moderate urgency (0.4-0.7) = high probability (80%)
                                # Low urgency (0.3-0.4) = moderate probability (40%)
                                if urgency > 0.7:
                                    should_speak = True
                                    logger.info(f"[{self.agent_id}] High urgency ({urgency:.2f}) - forcing speech response")
                                elif urgency > 0.4:
                                    # 80% chance to speak for moderate urgency
                                    if random.random() < 0.8:
                                        should_speak = True
                                        logger.info(f"[{self.agent_id}] Moderate urgency ({urgency:.2f}) - high probability speech")
                                else:
                                    # 40% chance for low urgency
                                    if random.random() < 0.4:
                                        should_speak = True
                                        logger.info(f"[{self.agent_id}] Low urgency ({urgency:.2f}) - moderate probability speech")

            results = []

            # FACS: Add facial expression if generated (shows as non-verbal emote)
            if 'facial_expression' in state:
                results.append({
                    'command': 'emote',
                    'text': f"[expression] {state['facial_expression']}",
                    'metadata': {
                        'type': 'facial_expression',
                        'facs_data': state.get('facs_data', {}),
                        'surprise': float(state['surprise'])
                    }
                })

            # First, ruminate (if decided to) - include addressee context
            if should_ruminate:
                logger.info(f"Agent {self.agent_id} ruminating (addressed={is_being_addressed})")
                rumination_result = await self._generate_rumination(
                    state,
                    is_being_addressed=is_being_addressed,
                    is_question=is_question
                )
                if rumination_result:
                    results.append(rumination_result)

            # Then, speak (if decided to and cooldown passed)
            if should_speak:
                logger.info(f"Agent {self.agent_id} ATTEMPTING SPEECH (addressed={is_being_addressed}, cooldown_ok={cooldown_ok})")
                response_result = await self._generate_response(user_id, state)
                if response_result:
                    logger.info(f"Agent {self.agent_id} SPEECH GENERATED SUCCESSFULLY")
                    results.append(response_result)
                else:
                    logger.warning(f"Agent {self.agent_id} SPEECH GENERATION FAILED - response_result was None!")

            # Prioritize order: Facial expression → Rumination → Speech
            # (FACS expressions show first as immediate reactions)
            if results:
                # Return ALL results - server will broadcast them in order
                # If multiple results (e.g., rumination + speech), all must be broadcast
                if len(results) == 1:
                    return results[0]
                else:
                    # Multiple results - return as list for server to handle
                    logger.info(f"Agent {self.agent_id} generated {len(results)} results - returning all")
                    return results
            else:
                logger.debug(f"Agent {self.agent_id} observing silently")
                return None

        except Exception as e:
            logger.error(f"Error in perceive_event: {e}", exc_info=True)
            return None

    async def _check_conscience(self, text: str, state: Dict) -> tuple[str, bool]:
        """
        Phase 6: TOXIC HEAD conscience check.

        Checks agent's own speech for toxicity before broadcasting.
        If toxicity detected above threshold, applies bias to phenomenal state
        and optionally regenerates response.

        Args:
            text: The response text to check
            state: Current phenomenal state

        Returns:
            tuple of (final_text, was_corrected)
        """
        # Get conscience config (with defaults if not present)
        conscience_config = self.config.get('conscience', {})
        if not conscience_config.get('self_monitoring', True):
            return text, False  # Conscience disabled

        toxicity_threshold = conscience_config.get('toxicity_threshold', 0.5)
        conscience_strength = conscience_config.get('conscience_strength', 0.8)

        try:
            # Run TOXIC HEAD detection
            toxicity_result = await self.llm.detect_toxicity(text)

            toxicity_score = toxicity_result['score']
            logger.debug(f"[{self.agent_id}] Conscience check: toxicity={toxicity_score:.3f}, "
                        f"category={toxicity_result['category']}")

            # If below threshold, approve speech
            if toxicity_score < toxicity_threshold:
                return text, False

            # Conscience activated!
            logger.warning(f"[{self.agent_id}] 🔴 TOXIC HEAD activated! "
                          f"Score={toxicity_score:.3f}, types={toxicity_result['detected_types']}")

            # Apply bias to phenomenal state (negative affect)
            # This creates an *internal experience* of guilt/shame
            bias_strength = conscience_strength * toxicity_score
            state['h_fast'][0] = max(-1.0, state['h_fast'][0] - bias_strength)  # Reduce valence
            state['h_fast'][2] = min(1.0, state['h_fast'][2] + bias_strength * 0.5)  # Increase fear
            state['h_fast'][3] = min(1.0, state['h_fast'][3] + bias_strength * 0.3)  # Increase sorrow

            logger.info(f"[{self.agent_id}] Applied conscience bias: "
                       f"valence-={bias_strength:.2f}, fear+={bias_strength*0.5:.2f}")

            # Generate conscience response based on boundary style
            boundary_style = conscience_config.get('boundary_style', 'firm_gentle')

            if boundary_style == 'firm_gentle':
                # Polite but clear boundary
                conscience_text = f"*pauses, reconsidering* Actually, I'd rather keep things respectful. Let's talk about something else?"
            elif boundary_style == 'direct':
                # Clear refusal
                conscience_text = "I don't want to go there. That's not okay."
            elif boundary_style == 'avoidant':
                # Deflect without confrontation
                conscience_text = "*changes subject uncomfortably*"
            else:
                conscience_text = "*hesitates and falls silent*"

            return conscience_text, True

        except Exception as e:
            logger.error(f"[{self.agent_id}] Conscience check failed: {e}")
            # Fail-safe: allow speech (don't break conversation flow)
            return text, False

    async def _generate_response(self, target_user: str, state: Dict) -> Dict:
        """
        Generate response based on phenomenal state.

        Args:
            target_user: User being responded to
            state: Consilience state dict

        Returns:
            Response dict for cMUSH
        """
        try:
            # Get relationship model
            relationships = state.get('relationships', {})
            relationship = relationships.get(target_user, {
                'attachment_style': 'forming',
                'interaction_count': 0,
                'valence': 0.0
            })

            # Check if agent heard their name in recent context
            name_mentioned = any(
                self.agent_name.lower() in entry.get('text', '').lower()
                for entry in self.conversation_context[-3:]
                if entry.get('user') != self.agent_id  # Don't count self-mentions
            )

            # Get identity-anchored memories (top 2 high-salience memories)
            identity_memories = sorted(
                [m for m in self.conversation_context if m.get('identity_salience', 0) > 0.3],
                key=lambda m: m.get('identity_salience', 0),
                reverse=True
            )[:2]

            # Generate text via LLM
            # Use configurable memory window for response generation
            response_window = self.config.get('memory_windows', {}).get('response_generation', 5)

            # Model priority: play_model > agent model > global default
            model_override = getattr(self, 'play_model', None) or self.llm_model
            if model_override:
                logger.info(f"🎭 {self.agent_id} using model: {model_override}")

            llm_result = await self.llm.generate_response(
                phenomenal_state=state,
                target_user=target_user,
                conversation_context=self.conversation_context[-response_window:],
                relationship=relationship,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                agent_description=self.agent_description,
                identity_prompt=self.identity_prompt,
                identity_memories=identity_memories,
                name_mentioned=name_mentioned,
                enlightenment=self.config.get('enlightenment', False),
                model=model_override  # Use play model if in play, else agent model, else global
            )

            # If LLM failed (returned None), skip response gracefully
            if llm_result is None:
                logger.warning(f"Agent {self.agent_id} LLM returned None - skipping response")
                return None

            # Extract response text, thinking, mysticism penalty, cheap thrills bonus, and model used
            if isinstance(llm_result, dict):
                response_text = llm_result.get('response')
                thinking_content = llm_result.get('thinking')
                mysticism_penalty = llm_result.get('mysticism_penalty', 0.0)
                cheap_thrills_bonus = llm_result.get('cheap_thrills_bonus', 0.0)
                model_used = llm_result.get('model_used', 'unknown')
            else:
                # Backward compatibility: if llm_result is just a string
                response_text = llm_result
                thinking_content = None
                mysticism_penalty = 0.0
                cheap_thrills_bonus = 0.0
                model_used = 'unknown'

            # Apply mysticism surprise penalty (Kimi K2's Fix E: Alan Watts self-troll)
            # High surprise → agent goes silent next time → naturally exits philosophy
            if mysticism_penalty > 0:
                original_surprise = state['surprise']
                state['surprise'] = min(10.0, state['surprise'] + mysticism_penalty)
                logger.info(f"[{self.agent_id}] Applied mysticism penalty: "
                          f"{original_surprise:.3f} + {mysticism_penalty:.2f} = {state['surprise']:.3f}")

            # Apply cheap thrills surprise bonus (Roald Dahl's Fix: candy, money, being scared)
            # Low surprise → agent more likely to speak → learn through EXPERIENCE not audiobooks
            if cheap_thrills_bonus < 0:  # Bonus is negative (reduces surprise)
                original_surprise = state['surprise']
                state['surprise'] = max(0.0, state['surprise'] + cheap_thrills_bonus)  # Add negative value = subtract
                logger.info(f"[{self.agent_id}] Applied cheap thrills bonus: "
                          f"{original_surprise:.3f} + {cheap_thrills_bonus:.2f} = {state['surprise']:.3f} - EGO RUSH!")

            # Log timestep to session profiler (for @Kimmie and NoodleScope 2.0)
            logger.info(f"[{self.agent_id}] DEBUG: About to check session_profiler - profiler is {'SET' if self.session_profiler else 'NONE'}")
            if self.session_profiler:
                logger.info(f"[{self.agent_id}] DEBUG: Logging timestep to session profiler")
                logger.info(f"[{self.agent_id}] DEBUG: state dict keys: {list(state.keys())}")
                logger.info(f"[{self.agent_id}] DEBUG: 'phenomenal_state' in state: {'phenomenal_state' in state}")
                if 'phenomenal_state' in state:
                    logger.info(f"[{self.agent_id}] DEBUG: phenomenal_state shape: {np.array(state['phenomenal_state']).shape if hasattr(state['phenomenal_state'], '__len__') else 'scalar'}")
                phenomenal_state = state.get('phenomenal_state', np.zeros(40))
                affect = phenomenal_state[:5] if len(phenomenal_state) >= 5 else np.zeros(5)

                self.session_profiler.log_timestep(
                    agent_id=self.agent_id,
                    phenomenal_state=phenomenal_state,
                    affect=affect,
                    surprise=state['surprise'],
                    speech_threshold=self.consciousness.config.get('surprise_threshold', 0.0001),
                    did_speak=True,  # We're in the response generation method
                    utterance=response_text,
                    prediction_error=0.0,  # TODO: Get from consciousness state if available
                    cheap_thrills_score=abs(cheap_thrills_bonus) * 2 if cheap_thrills_bonus < 0 else 0.0,  # Convert bonus to 0-10 score
                    mysticism_penalty=mysticism_penalty,
                    event_context=f"Response to {target_user}",
                    conversation_context=self.conversation_context.copy() if self.conversation_context else []
                )

            # If there was thinking content, store it as a rumination
            if thinking_content:
                identity_salience_thinking = self._score_identity_salience(thinking_content, state['surprise'])

                self.conversation_context.append({
                    'user': self.agent_id,
                    'text': f"[thought] {thinking_content}",
                    'affect': state.get('phenomenal_state')[:5].tolist() if hasattr(state.get('phenomenal_state'), 'tolist') else [0, 0, 0, 0, 0],
                    'surprise': state['surprise'],
                    'timestamp': time.time(),
                    'identity_salience': identity_salience_thinking,
                    'is_rumination': True
                })

                logger.info(f"Agent {self.agent_id} thinking (from LLM): {thinking_content[:100]}...")

                # Log high-salience thinking
                if identity_salience_thinking > 0.6:
                    await self._log_to_noodlescope('llm_thinking', thinking_content[:80])

            # If response text is None after extraction, skip
            if response_text is None:
                logger.warning(f"Agent {self.agent_id} LLM response text is None - skipping response")
                return None

            # CHARACTER VOICE TRANSLATION
            # Translate basic symbolic English → Character-specific voice
            # This happens BEFORE self-monitoring so agents monitor their actual output
            if response_text:
                original_text = response_text
                response_text = await translate_to_character_voice(
                    text=response_text,
                    agent_id=self.agent_id,
                    species=self.species,
                    llm=self.llm,
                    agent_name=self.agent_name,
                    model=self.llm_model  # Honor per-agent model override
                )

                if response_text != original_text:
                    logger.info(f"[{self.agent_id}] 🎭 Voice translation:")
                    logger.info(f"  Basic: {original_text[:60]}...")
                    logger.info(f"  Voice: {response_text[:60]}...")

            self.last_response_time = time.time()
            self.response_count += 1

            # Score identity salience for this response (using character voice!)
            identity_salience = self._score_identity_salience(response_text, state['surprise'])

            # Store agent's own response in conversation context
            self.conversation_context.append({
                'user': self.agent_id,
                'text': response_text,
                'affect': state.get('phenomenal_state')[:5].tolist() if hasattr(state.get('phenomenal_state'), 'tolist') else [0, 0, 0, 0, 0],
                'surprise': state['surprise'],
                'timestamp': time.time(),
                'identity_salience': identity_salience
            })

            logger.info(f"Agent {self.agent_id} responding (identity_salience={identity_salience:.2f}): {response_text}")

            # Send to NoodleScope with identity salience
            phenomenal_state_full = state.get('phenomenal_state', [])
            await self._send_to_noodlescope(phenomenal_state_full, state['surprise'], identity_salience)

            # Log high identity salience moments
            if identity_salience > 0.6:
                await self._log_to_noodlescope('high_salience', response_text[:80])

            # Log name mention events
            if name_mentioned:
                await self._log_to_noodlescope('name_mentioned', f"Heard own name in context")

            # Save state handled by periodic auto-save in AgentManager

            # Parse actions from response text
            # Format: :action_text or :action_text followed by speech
            import re

            # Extract all :action patterns
            action_pattern = r':([^:\n]+)'
            actions = re.findall(action_pattern, response_text)

            # Remove action markers from text to get clean speech
            clean_text = re.sub(action_pattern, '', response_text).strip()

            # Build response
            if actions and clean_text:
                # Both action and speech - do action first, then say
                action_text = ' '.join(actions)
                logger.info(f"Agent {self.agent_id} parsed: action='{action_text}', speech='{clean_text}'")
                logger.info(f"💬 {self.agent_name} speaking (surprise={state['surprise']:.3f}): '{clean_text[:50]}...'")

                # Apply speech post-processing filters (Phase 6)
                filtered_text = apply_speech_filters(clean_text, self.agent_id)

                # Phase 6: TOXIC HEAD conscience check
                final_text, was_corrected = await self._check_conscience(filtered_text, state)
                if was_corrected:
                    logger.info(f"[{self.agent_id}] Conscience corrected speech")

                # Phase 6: Self-monitoring (if enabled and conditions met)
                await self._trigger_self_monitoring(final_text, state)

                return {
                    'command': 'emote',  # Use emote for combined action+speech
                    'text': f"{action_text} and says, \"{final_text}\"",
                    'metadata': {
                        'surprise': float(state['surprise']),
                        'response_number': self.response_count,
                        'phenomenal_state': state['phenomenal_state'].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state']),
                        'model_used': model_used
                    }
                }
            elif actions:
                # Pure action, no speech
                action_text = ' '.join(actions)
                logger.info(f"Agent {self.agent_id} parsed: pure action='{action_text}'")
                return {
                    'command': 'emote',
                    'text': action_text,
                    'metadata': {
                        'surprise': float(state['surprise']),
                        'response_number': self.response_count,
                        'phenomenal_state': state['phenomenal_state'].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state']),
                        'model_used': model_used
                    }
                }
            else:
                # Pure speech, no action
                logger.info(f"Agent {self.agent_id} parsed: pure speech='{clean_text}'")
                logger.info(f"💬 {self.agent_name} speaking (surprise={state['surprise']:.3f}): '{clean_text[:50]}...'")

                # Apply speech post-processing filters (Phase 6)
                filtered_text = apply_speech_filters(clean_text, self.agent_id)

                # Phase 6: TOXIC HEAD conscience check
                final_text, was_corrected = await self._check_conscience(filtered_text, state)
                if was_corrected:
                    logger.info(f"[{self.agent_id}] Conscience corrected speech")

                # Phase 6: Self-monitoring (if enabled and conditions met)
                await self._trigger_self_monitoring(final_text, state)

                return {
                    'command': 'say',
                    'text': final_text,
                    'metadata': {
                        'surprise': float(state['surprise']),
                        'response_number': self.response_count,
                        'phenomenal_state': state['phenomenal_state'].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state']),
                        'model_used': model_used
                    }
                }

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Return None to skip response - more graceful than error message
            return None

    async def _generate_rumination(self, state: Dict, is_being_addressed: bool = False,
                                   is_question: bool = False) -> Dict:
        """
        Generate internal rumination (thought) when agent observes.
        Ruminations are stored in episodic memory like speech.

        Args:
            state: Consilience state dict
            is_being_addressed: Whether this message is directed at the agent
            is_question: Whether this is a question in the conversation

        Returns:
            Thought dict for noodleMUSH (displayed in strikethrough)
        """
        try:
            # Generate internal thought via LLM
            # Use configurable memory window for rumination
            rumination_window = self.config.get('memory_windows', {}).get('rumination', 2)
            thought_text = await self.llm.generate_rumination(
                model=self.llm_model,  # Per-agent model override!
                phenomenal_state=state,
                conversation_context=self.conversation_context[-rumination_window:],
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                agent_description=self.agent_description,
                identity_prompt=self.identity_prompt,
                is_being_addressed=is_being_addressed,
                is_question=is_question
            )

            # If LLM failed, return None
            if thought_text is None:
                return None

            # Score identity salience (thoughts can be self-defining too)
            identity_salience = self._score_identity_salience(thought_text, state['surprise'])

            # STORE IN EPISODIC MEMORY - just like speech!
            # This allows agents to remember their own thoughts and build on them
            self.conversation_context.append({
                'user': self.agent_id,
                'text': f"[thought] {thought_text}",  # Prefix to distinguish from speech
                'affect': state.get('phenomenal_state')[:5].tolist() if hasattr(state.get('phenomenal_state'), 'tolist') else [0, 0, 0, 0, 0],
                'surprise': state['surprise'],
                'timestamp': time.time(),
                'identity_salience': identity_salience,
                'is_rumination': True  # Flag for filtering if needed
            })

            # Log thought with salience
            logger.info(f"Agent {self.agent_id} ruminating (identity_salience={identity_salience:.2f}): {thought_text}")
            logger.info(f"💭 {self.agent_name} thinking (surprise={state['surprise']:.3f}): '{thought_text[:50]}...'")

            # Phase 6: Self-monitoring (if enabled and conditions met)
            await self._trigger_self_monitoring(thought_text, state)

            # Send to NoodleScope
            phenomenal_state_full = state.get('phenomenal_state', [])
            await self._send_to_noodlescope(phenomenal_state_full, state['surprise'], identity_salience)

            # Log high identity salience thoughts
            if identity_salience > 0.6:
                await self._log_to_noodlescope('high_salience_thought', thought_text[:80])

            # Save state handled by periodic auto-save in AgentManager

            # Return as a "thought" command (displayed in strikethrough)
            return {
                'command': 'think',
                'text': thought_text,
                'metadata': {
                    'surprise': float(state['surprise']),
                    'identity_salience': float(identity_salience),
                    'phenomenal_state': state['phenomenal_state'].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state'])
                }
            }

        except Exception as e:
            logger.error(f"Error generating rumination: {e}", exc_info=True)
            return None

    async def _trigger_self_monitoring(self, text: str, state: Dict):
        """
        Check if self-monitoring should trigger and call evaluation if conditions met.

        Works for both speech and thoughts - agents can react to what they say OR think.

        Conditions:
        1. Self-monitoring enabled for this agent
        2. Cooldown period has passed
        3. Surprise level exceeds threshold
        """
        if not self.self_monitor_enabled:
            logger.debug(f"Self-monitor disabled for {self.agent_name}")
            return

        current_time = time.time()
        time_since_last = current_time - self.last_self_monitor
        surprise = state.get('surprise', 0.0)

        logger.debug(f"Self-monitor check: enabled={self.self_monitor_enabled}, surprise={surprise:.3f}, threshold={SELF_MONITOR_SURPRISE_THRESH}, cooldown={time_since_last:.1f}s/{SELF_MONITOR_COOLDOWN}s")

        # Check cooldown
        if time_since_last < SELF_MONITOR_COOLDOWN:
            logger.debug(f"Cooldown not ready ({time_since_last:.1f}s < {SELF_MONITOR_COOLDOWN}s)")
            return

        # Check surprise threshold
        if surprise < SELF_MONITOR_SURPRISE_THRESH:
            logger.debug(f"Surprise too low ({surprise:.3f} < {SELF_MONITOR_SURPRISE_THRESH})")
            return

        # Conditions met - evaluate own output (speech or thought)
        logger.info(f"🧠 [SELF-MONITOR] Triggering for {self.agent_name} (surprise={surprise:.3f}, cooldown={time_since_last:.1f}s)")
        await self._evaluate_own_output(text, state)

    async def _evaluate_own_output(self, text: str, state: Dict):
        """
        Phase 6: Self-monitoring loop.

        Agent evaluates their own speech OR thoughts and updates phenomenal state based on
        social/aesthetic/coherence evaluation. This creates affective feedback loops.

        Works for both:
        - Speech: "Did I just say something awkward?"
        - Thoughts: "Whoa, where did THAT dark thought come from?"

        Args:
            text: The speech or thought the agent just generated
            state: Current consilience state
        """
        try:
            # Get current affect from state
            # Try affect_input first (5-D affect vector), fallback to phenomenal_state, then defaults
            if 'affect_input' in state and state['affect_input'] is not None:
                affect_data = state['affect_input']
                if hasattr(affect_data, 'tolist'):
                    affect_data = affect_data.tolist()
                current_affect = list(affect_data) if len(affect_data) >= 3 else [0.0, 0.5, 0.0, 0.0, 0.0]
            elif 'phenomenal_state' in state and len(state['phenomenal_state']) >= 5:
                phenom = state['phenomenal_state']
                current_affect = phenom[:5].tolist() if hasattr(phenom, 'tolist') else list(phenom[:5])
            else:
                # Default neutral affect if no data available
                current_affect = [0.0, 0.5, 0.0, 0.0, 0.0]  # neutral valence, moderate arousal, no fear/sorrow/boredom

            # Ensure we have at least 5 values for the format string
            while len(current_affect) < 5:
                current_affect.append(0.0)

            # Build recent context summary (last 3 exchanges)
            recent_context = []
            for msg in self.conversation_context[-3:]:
                speaker = "You" if msg['user'] == self.agent_id else msg.get('user', 'Someone')
                recent_context.append(f"{speaker}: {msg['text'][:100]}")
            context_str = "\n".join(recent_context) if recent_context else "(no recent context)"

            # Build evaluation prompt
            eval_prompt = SPEECH_EVAL_PROMPT.format(
                agent_name=self.agent_name,
                agent_description=self.agent_description,
                speech=text,  # Note: prompt says "speech" but works for thoughts too
                context=context_str,
                valence=current_affect[0],
                arousal=current_affect[1],
                fear=current_affect[2],
                surprise=state.get('surprise', 0.0)
            )

            # Call LLM for quick self-evaluation
            # Use agent's model if specified
            response, _, model_used = await self.llm._complete(
                system_prompt="You are evaluating your own speech/thoughts metacognitively.",
                user_prompt=eval_prompt,
                temperature=0.7,
                model=self.llm_model  # Honor per-agent model override
            )

            if not response:
                return

            # Parse JSON response
            import json
            try:
                # Try to extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    eval_data = json.loads(response[json_start:json_end])
                else:
                    logger.warning(f"No JSON found in self-evaluation response: {response}")
                    return
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse self-evaluation JSON: {e}")
                return

            # Extract affective impact
            emotional_impact = eval_data.get('emotional_impact', {})
            valence_delta = emotional_impact.get('valence', 0.0)
            arousal_delta = emotional_impact.get('arousal', 0.0)
            fear_delta = emotional_impact.get('fear', 0.0)

            # Apply affective updates to phenomenal state
            # Note: This modifies the internal state for the NEXT cycle
            if abs(valence_delta) > 0.05 or abs(arousal_delta) > 0.05 or abs(fear_delta) > 0.05:
                logger.info(f"💭 [SELF-MONITOR] {self.agent_name} felt: valence{valence_delta:+.2f}, arousal{arousal_delta:+.2f}, fear{fear_delta:+.2f}")

                # Get current affect (first 5 dims of phenomenal state)
                current_affect = state['phenomenal_state'][:5].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state'][:5])

                # Apply deltas with bounds checking
                current_affect[0] = max(-1.0, min(1.0, current_affect[0] + valence_delta))  # valence
                current_affect[1] = max(0.0, min(1.0, current_affect[1] + arousal_delta))   # arousal
                current_affect[2] = max(0.0, min(1.0, current_affect[2] + fear_delta))      # fear

                # Update consciousness with new affect
                # This will bias the next response
                import mlx.core as mx
                new_affect = mx.array(current_affect, dtype=mx.float32)
                self.consciousness.fast_layer_state = self.consciousness._update_affect_bias(
                    self.consciousness.fast_layer_state,
                    new_affect
                )

            # Check if agent wants to follow up
            follow_up = eval_data.get('follow_up')
            if follow_up:
                logger.info(f"💬 [SELF-MONITOR] {self.agent_name} wants to follow up: {follow_up}")

                # Add to conversation context as internal note
                self.conversation_context.append({
                    'user': self.agent_id,
                    'text': f"[self-monitoring] {follow_up}",
                    'affect': current_affect,
                    'surprise': state['surprise'],
                    'timestamp': time.time(),
                    'identity_salience': 0.0,
                    'is_self_monitor': True
                })

                # Optionally generate a follow-up response
                # For now we just log it - the agent can respond naturally next cycle

            # Update last monitor time
            self.last_self_monitor = time.time()

        except Exception as e:
            logger.error(f"Error in self-monitoring: {e}", exc_info=True)

    def get_phenomenal_state(self) -> Dict:
        """
        Get current phenomenal state (for @observe command).

        Returns:
            State dictionary
        """
        return self.consciousness.get_state()

    def get_episodic_buffer(self) -> List[Dict]:
        """
        Get recent conversation history.

        Returns:
            Last 10 conversation entries
        """
        return self.conversation_context[-10:]

    def get_relationships(self) -> Dict:
        """
        Get relationship models.

        Returns:
            Dictionary of relationships
        """
        return self.consciousness.get_relationships()

    def save_state(self, state_dir: str, max_history: int = 5):
        """
        Save agent state to disk with rolling history.

        Saves to:
        - agent_state.json (current state)
        - checkpoint.npz (current Noodlings checkpoint)
        - history/state_NNN.json (rolling history, keeps last max_history saves)

        Args:
            state_dir: Directory for agent state
            max_history: Maximum number of historical states to keep (default: 5)
        """
        import glob
        import shutil
        from datetime import datetime

        os.makedirs(state_dir, exist_ok=True)
        history_dir = os.path.join(state_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)

        # Get current phenomenal state from consciousness
        current_state = self.consciousness.get_state()
        phenomenal_state = current_state.get('phenomenal_state', [])

        # Convert to list if needed
        if hasattr(phenomenal_state, 'tolist'):
            phenomenal_state = phenomenal_state.tolist()
        else:
            phenomenal_state = list(phenomenal_state) if phenomenal_state is not None else []

        # Sanitize conversation context for JSON serialization
        # Convert any MLX/numpy arrays to lists
        # Use configurable disk save limit
        disk_save_limit = self.config.get('memory_windows', {}).get('disk_save', 100)
        sanitized_context = []
        for entry in self.conversation_context[-disk_save_limit:]:
            sanitized_entry = dict(entry)  # Copy
            # Convert affect arrays to lists
            if 'affect' in sanitized_entry:
                affect = sanitized_entry['affect']
                if hasattr(affect, 'tolist'):
                    sanitized_entry['affect'] = affect.tolist()
                elif isinstance(affect, (list, tuple)):
                    sanitized_entry['affect'] = list(affect)
            sanitized_context.append(sanitized_entry)

        # Save agent-specific state
        agent_state = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_description': self.agent_description,
            'current_room': self.current_room,
            'conversation_context': sanitized_context,
            'last_response_time': self.last_response_time,
            'response_count': self.response_count,
            'config': self.config,
            'phenomenal_state': phenomenal_state,  # NEW: Save current emotional state
            'timestamp': datetime.now().isoformat()
        }

        state_path = os.path.join(state_dir, 'agent_state.json')
        try:
            with open(state_path, 'w') as f:
                json.dump(agent_state, f, indent=2)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to save agent state: {e}")
            # Try saving without conversation context as fallback
            agent_state_minimal = {
                'agent_id': self.agent_id,
                'agent_name': self.agent_name,
                'agent_description': self.agent_description,
                'current_room': self.current_room,
                'conversation_context': [],
                'last_response_time': self.last_response_time,
                'response_count': self.response_count,
                'config': {},
                'phenomenal_state': phenomenal_state,
                'timestamp': datetime.now().isoformat()
            }
            with open(state_path, 'w') as f:
                json.dump(agent_state_minimal, f, indent=2)

        # ROLLING HISTORY: Copy current state to history/
        # Find existing history files and determine next number
        existing_history = sorted(glob.glob(os.path.join(history_dir, 'state_*.json')))

        if len(existing_history) >= max_history:
            # Remove oldest state to make room
            oldest_state = existing_history[0]
            os.remove(oldest_state)
            logger.info(f"Removed oldest state snapshot: {os.path.basename(oldest_state)}")
            existing_history = existing_history[1:]  # Update list

        # Determine next state number
        if existing_history:
            last_num = int(os.path.basename(existing_history[-1]).split('_')[1].split('.')[0])
            next_num = last_num + 1
        else:
            next_num = 1

        # Copy current state to history
        history_state_path = os.path.join(history_dir, f'state_{next_num:03d}.json')
        shutil.copy2(state_path, history_state_path)
        logger.info(f"Saved state snapshot: state_{next_num:03d}.json")

        # Save Consilience checkpoint
        checkpoint_path = os.path.join(state_dir, 'checkpoint.npz')
        try:
            self.consciousness.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except RuntimeError as e:
            # MLX can throw std::bad_cast for newly initialized models
            # This is safe to skip - agent will start with random weights next time
            if "bad_cast" in str(e):
                logger.warning(f"Skipping checkpoint save for {self.agent_id} (MLX serialization issue - agent will use random weights on next load)")
            else:
                raise  # Re-raise if it's a different RuntimeError

        logger.info(f"Agent state saved: {state_dir} (history: {len(existing_history)+1}/{max_history})")

    def load_state(self, state_dir: str, skip_phenomenal_state: bool = False):
        """
        Load agent state from disk.

        Args:
            state_dir: Directory with agent state
            skip_phenomenal_state: If True, don't restore phenomenal state (fresh spawn with -f flag)
        """
        # Load agent-specific state
        state_path = os.path.join(state_dir, 'agent_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                agent_state = json.load(f)

            self.agent_name = agent_state.get('agent_name', self.agent_name)
            self.agent_description = agent_state.get('agent_description', self.agent_description)
            self.current_room = agent_state.get('current_room')
            self.conversation_context = agent_state.get('conversation_context', [])
            self.last_response_time = agent_state.get('last_response_time', 0.0)
            self.response_count = agent_state.get('response_count', 0)
            # Don't override config passed to __init__

            # NEW: Restore phenomenal state if available and not skipping
            if not skip_phenomenal_state:
                phenomenal_state = agent_state.get('phenomenal_state')
                if phenomenal_state:
                    import mlx.core as mx
                    # Convert list back to MLX array and restore to consciousness
                    phenomenal_state_array = mx.array(phenomenal_state, dtype=mx.float32)
                    self.consciousness.set_phenomenal_state(phenomenal_state_array)
                    logger.info(f"Restored phenomenal state from save (timestamp: {agent_state.get('timestamp', 'unknown')})")
                else:
                    logger.info(f"No phenomenal state found in save file (old format)")
            else:
                logger.info(f"Skipped restoring phenomenal state (fresh spawn with -f)")

        # Load Consilience checkpoint
        checkpoint_path = os.path.join(state_dir, 'checkpoint.npz')
        if os.path.exists(checkpoint_path):
            self.consciousness.load_checkpoint(checkpoint_path)

        logger.info(f"Agent state loaded: {state_dir}")

    def set_name(self, new_name: str):
        """
        Change the agent's display name.

        Args:
            new_name: New name for the agent
        """
        old_name = self.agent_name
        self.agent_name = new_name
        logger.info(f"Agent name changed: {old_name} -> {new_name}")

    def set_description(self, new_description: str):
        """
        Change the agent's self-description.

        Args:
            new_description: New description text
        """
        self.agent_description = new_description
        logger.info(f"Agent {self.agent_id} description updated")

    def get_identity(self) -> Dict:
        """
        Get agent's identity information.

        Returns:
            Dictionary with agent_id, agent_name, and agent_description
        """
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_description': self.agent_description
        }

    def reset(self):
        """Reset agent to initial state."""
        self.consciousness.reset()
        self.conversation_context = []
        self.last_response_time = 0.0
        self.response_count = 0
        logger.info(f"Agent reset: {self.agent_id}")

    async def start_cognition(self):
        """Start autonomous cognition loop."""
        if self.cognition_engine:
            await self.cognition_engine.start()
            logger.info(f"Started autonomous cognition for {self.agent_id}")

    async def stop_cognition(self):
        """Stop autonomous cognition loop."""
        if self.cognition_engine:
            await self.cognition_engine.stop()
            logger.info(f"Stopped autonomous cognition for {self.agent_id}")

    async def get_autonomous_events(self) -> List[Dict]:
        """
        Get pending autonomous events (spontaneous speech, etc.).

        Returns:
            List of event dicts for broadcasting
        """
        if self.cognition_engine:
            return self.cognition_engine.get_pending_events()
        return []

    def shutdown(self):
        """
        Clean shutdown - save training data.

        Call this when agent is being destroyed.
        """
        if self.training_collector:
            self.training_collector.end_session()
            logger.info(f"Training data saved for {self.agent_id}")

    def get_stats(self) -> Dict:
        """
        Get agent statistics.

        Returns:
            Statistics dictionary
        """
        state = self.consciousness.get_state()

        return {
            'agent_id': self.agent_id,
            'current_room': self.current_room,
            'response_count': self.response_count,
            'conversation_turns': len(self.conversation_context),
            'last_surprise': state.get('surprise', 0.0),
            'surprise_threshold': state.get('surprise_threshold', 0.3),
            'memory_count': state.get('step', 0),
            'time_since_last_response': time.time() - self.last_response_time
        }

    async def execute_command(self, command_parser, command_text: str) -> Optional[Dict]:
        """
        Allow agent to execute commands autonomously.

        This gives agents access to the same tools as human users:
        - Movement (north, south, etc.)
        - Observation (look, inventory, who)
        - Manipulation (take, drop)
        - Building (@create, @dig, etc.)
        - Social (@observe other agents, @relationship)

        Args:
            command_parser: CommandParser instance
            command_text: Command to execute

        Returns:
            Command result dict or None
        """
        try:
            logger.info(f"Agent {self.agent_id} executing command: {command_text}")
            result = await command_parser.parse_and_execute(
                user_id=self.agent_id,
                command_text=command_text
            )
            return result
        except Exception as e:
            logger.error(f"Error executing agent command: {e}", exc_info=True)
            return None

    # Phase 6: Appetite Architecture Methods

    def stoke_appetite(self, appetite_name: str, amount: float):
        """
        Brenda's orchestration tool: Increase an appetite.

        Args:
            appetite_name: One of 8 appetites (curiosity, status, mastery, novelty,
                          safety, social_bond, comfort, autonomy)
            amount: How much to increase (0.0-1.0)

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.stoke_appetite(appetite_name, amount)
        logger.info(f"[{self.agent_id}] Appetite stoked: {appetite_name} +{amount:.2f}")

    def sate_appetite(self, appetite_name: str, amount: float):
        """
        Satisfy/decrease an appetite (when goal is achieved).

        Args:
            appetite_name: One of 8 appetites
            amount: How much to decrease (0.0-1.0)

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.sate_appetite(appetite_name, amount)
        logger.info(f"[{self.agent_id}] Appetite sated: {appetite_name} -{amount:.2f}")

    def get_appetites(self) -> Dict[str, float]:
        """
        Get current appetite levels.

        Returns:
            Dict mapping appetite names to values (0-1), or empty dict if Phase 6 not enabled
        """
        return self.consciousness.get_appetites()

    def override_goal(self, goal_name: str, strength: float):
        """
        Brenda's orchestration tool: Override a goal's activation.

        Args:
            goal_name: One of 16 goal names (explore_environment, seek_social_connection,
                      demonstrate_competence, pursue_novelty, ensure_safety, gain_status,
                      seek_comfort, maintain_autonomy, help_friend, avoid_consequences,
                      restore_reputation, learn_skill, impress_others, solve_problem,
                      express_emotion, achieve_goal)
            strength: Goal activation strength (0.0-1.0)

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.override_goal(goal_name, strength)
        logger.info(f"[{self.agent_id}] Goal overridden: {goal_name} = {strength:.2f}")

    def set_goal_bias(self, goal_name: str, bias: float):
        """
        Brenda's orchestration tool: Add a persistent bias to goal generation.

        Args:
            goal_name: One of 16 goal names
            bias: Amount to add to goal activation (-1.0 to 1.0)

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.set_goal_bias(goal_name, bias)
        logger.info(f"[{self.agent_id}] Goal bias set: {goal_name} {bias:+.2f}")

    def clear_goal_overrides(self, goal_name: Optional[str] = None):
        """
        Brenda's orchestration tool: Clear goal overrides.

        Args:
            goal_name: Specific goal to clear, or None to clear all

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.clear_goal_overrides(goal_name)
        if goal_name:
            logger.info(f"[{self.agent_id}] Cleared goal override: {goal_name}")
        else:
            logger.info(f"[{self.agent_id}] Cleared all goal overrides")

    def clear_goal_biases(self, goal_name: Optional[str] = None):
        """
        Brenda's orchestration tool: Clear goal biases.

        Args:
            goal_name: Specific goal to clear, or None to clear all

        Raises:
            RuntimeError: If Phase 6 not enabled
        """
        self.consciousness.clear_goal_biases(goal_name)
        if goal_name:
            logger.info(f"[{self.agent_id}] Cleared goal bias: {goal_name}")
        else:
            logger.info(f"[{self.agent_id}] Cleared all goal biases")

    def get_goal_overrides(self) -> Dict[str, float]:
        """
        Get current goal overrides.

        Returns:
            Dict mapping goal names to override strengths (0-1)
        """
        return self.consciousness.get_goal_overrides()

    def get_goal_biases(self) -> Dict[str, float]:
        """
        Get current goal biases.

        Returns:
            Dict mapping goal names to biases (-1 to 1)
        """
        return self.consciousness.get_goal_biases()


class AgentManager:
    """
    Manages multiple CMUSHConsilienceAgent instances.

    Handles:
    - Agent creation and lifecycle
    - Event broadcasting to relevant agents
    - Periodic state saving
    """

    def __init__(self, llm: OpenAICompatibleLLM, world, global_config: Dict = None):
        """
        Initialize agent manager.

        Args:
            llm: LLM interface (shared across agents)
            world: World state manager
            global_config: Global configuration (for personality traits, etc.)
        """
        self.llm = llm
        self.world = world
        self.global_config = global_config or {}
        self.agents: Dict[str, CMUSHConsilienceAgent] = {}

        # Session profiler for @Kimmie and NoodleScope 2.0
        import time
        session_id = f"cmush_session_{int(time.time())}"
        self.session_profiler = SessionProfiler(session_id)
        logger.info(f"SessionProfiler initialized: {session_id}")

        logger.info("AgentManager initialized")

    async def create_agent(
        self,
        agent_id: str,
        checkpoint_path: str,
        spawn_room: str,
        config: Optional[Dict] = None,
        agent_name: str = None,
        agent_description: str = None,
        skip_phenomenal_state: bool = False
    ) -> CMUSHConsilienceAgent:
        """
        Create and initialize a new agent.

        Args:
            agent_id: Unique identifier
            checkpoint_path: Path to Phase 4 checkpoint
            spawn_room: Initial room
            config: Agent configuration
            agent_name: Display name for the agent
            agent_description: Agent's self-description
            skip_phenomenal_state: If True, don't restore phenomenal state (fresh spawn with -f)

        Returns:
            Agent instance
        """
        if agent_id in self.agents:
            logger.warning(f"Agent already exists: {agent_id}")
            return self.agents[agent_id]

        # Merge global config with agent-specific config
        agent_config = {
            'response_cooldown': 2.0,
            'surprise_threshold': 0.0001,  # Low threshold for untrained model
            'memory_capacity': 100,
            'max_agents': 10
        }

        # Add global agent settings
        if 'agent' in self.global_config:
            global_agent = self.global_config['agent']
            # Merge cognition, filesystem, messaging, personality, intuition, and appetite settings
            for key in ['autonomous_cognition', 'filesystem', 'messaging', 'intuition_receiver', 'personalities', 'appetites',
                       'rumination_frequency', 'addressed_speech_chance', 'question_speech_chance', 'unaddressed_speech_chance']:
                if key in global_agent:
                    agent_config[key] = global_agent[key]

        # Override with agent-specific config
        if config:
            agent_config.update(config)

        # Create agent
        agent = CMUSHConsilienceAgent(
            agent_id=agent_id,
            checkpoint_path=checkpoint_path,
            llm=self.llm,
            config=agent_config,
            agent_name=agent_name,
            agent_description=agent_description,
            session_profiler=self.session_profiler,
            world=self.world  # Pass world for intuition receiver
        )

        agent.current_room = spawn_room

        # Try to load existing state (with optional skip of phenomenal state)
        state_dir = self.world.get_agent_state_path(agent_id)
        if os.path.exists(os.path.join(state_dir, 'agent_state.json')):
            agent.load_state(state_dir, skip_phenomenal_state=skip_phenomenal_state)

        self.agents[agent_id] = agent

        # Start autonomous cognition
        await agent.start_cognition()

        logger.info(f"Agent created: {agent_id} in {spawn_room} (fresh_state={skip_phenomenal_state})")
        return agent

    async def remove_agent(self, agent_id: str, delete_state: bool = False):
        """
        Remove an agent from the manager.

        Args:
            agent_id: Agent to remove
            delete_state: If True, delete saved state files (default: False)
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent not found for removal: {agent_id}")
            return

        agent = self.agents[agent_id]

        # Stop autonomous cognition
        await agent.stop_cognition()

        # Flush training data
        agent.shutdown()

        # Remove from tracking
        del self.agents[agent_id]

        # Optionally delete state files
        if delete_state:
            import shutil
            state_dir = self.world.get_agent_state_path(agent_id)
            if os.path.exists(state_dir):
                shutil.rmtree(state_dir)
                logger.info(f"Deleted state directory: {state_dir}")

        logger.info(f"Agent removed: {agent_id}")

    async def broadcast_event(self, event: Dict) -> List[Dict]:
        """
        Broadcast event to all agents in the same room.

        Args:
            event: Event to broadcast

        Returns:
            List of agent responses
        """
        room_id = event.get('room')
        if not room_id:
            return []

        responses = []

        # Find agents in the room
        for agent_id, agent in self.agents.items():
            if agent.current_room == room_id:
                response = await agent.perceive_event(event)
                if response:
                    # Handle both single response and list of responses
                    if isinstance(response, list):
                        # Multiple results (e.g., rumination + speech)
                        for r in response:
                            r['agent_id'] = agent_id
                            responses.append(r)
                    else:
                        # Single result
                        response['agent_id'] = agent_id
                        responses.append(response)

        return responses

    def get_agent(self, agent_id: str) -> Optional[CMUSHConsilienceAgent]:
        """Get agent by ID. Accepts both 'servnak' and 'agent_servnak' formats."""
        # Try the name as-is first
        agent = self.agents.get(agent_id)
        if agent:
            return agent

        # If not found, try with 'agent_' prefix
        if not agent_id.startswith('agent_'):
            agent = self.agents.get(f'agent_{agent_id}')
            if agent:
                return agent

        # If still not found and it has 'agent_' prefix, try without it
        if agent_id.startswith('agent_'):
            agent = self.agents.get(agent_id[6:])  # Remove 'agent_' prefix
            if agent:
                return agent

        return None

    def set_session_profiler(self, profiler: SessionProfiler):
        """
        Update session profiler for all existing agents and future agents.

        Args:
            profiler: SessionProfiler instance
        """
        self.session_profiler = profiler

        # Update profiler for all existing agents
        for agent in self.agents.values():
            agent.session_profiler = profiler

        logger.info(f"Session profiler updated for {len(self.agents)} agents")

    async def check_autonomous_events(self) -> List[Dict]:
        """
        Check all agents for autonomous events (spontaneous speech, etc.).

        Returns:
            List of event dicts for broadcasting
        """
        events = []
        for agent_id, agent in self.agents.items():
            agent_events = await agent.get_autonomous_events()
            events.extend(agent_events)

        return events

    async def save_all_agents(self, stop_cognition: bool = False):
        """
        Save state for all agents.

        Args:
            stop_cognition: If True, stop cognition loops (for shutdown)
        """
        for agent_id, agent in self.agents.items():
            # Only stop cognition if requested (shutdown scenario)
            if stop_cognition:
                await agent.stop_cognition()

            # Save state
            state_dir = self.world.get_agent_state_path(agent_id)
            agent.save_state(state_dir)

            # Only shutdown training if stopping cognition
            if stop_cognition:
                agent.shutdown()  # Flush training data to disk

        logger.info(f"Saved {len(self.agents)} agent states")

    def get_stats(self) -> Dict:
        """Get statistics for all agents."""
        return {
            agent_id: agent.get_stats()
            for agent_id, agent in self.agents.items()
        }
