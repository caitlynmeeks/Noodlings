"""
Agent Bridge - Noodling charm <-> cMUSH world adapter

Bridges between:
- cMUSH events (say, emote, enter, exit)
- Noodling affective architecture
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
import asyncio
import numpy as np
import aiohttp

from noodlings.api import NoodlingAgent
from noodlings.memory.hierarchical_memory import HierarchicalMemory
from noodlings.models.affect_head import AffectHead, interpret_affect, classify_emotion_from_affect
from noodlings.utils.facs_mapping import affect_to_facs, facs_to_description, format_facs_for_renderer
from noodlings.utils.body_language_mapping import affect_to_body_language, body_language_to_description, format_body_language_for_renderer
from llm_interface import OpenAICompatibleLLM
from training_data_collector import TrainingDataCollector
from agent_filesystem import AgentFilesystem
from agent_messaging import AgentMessaging
# REMOVED: from autonomous_cognition import AutonomousCognitionEngine
# Replacing with continuous affect-driven cognition loop
from session_profiler import SessionProfiler
from performance_tracker import get_tracker
from affective_reinforcement import create_reinforcement
from entropy_service import get_entropy_service
from event_system import Event
from embodiment_loader import EmbodimentLoader
from agent_perception import PerceptionMixin
from agent_response import ResponseGenerationMixin
from agent_cognition import CognitionLoopMixin
from agent_state import StatePersistenceMixin

# Semantic World integration (optional - graceful fallback if not available)
try:
    from semantic_integration import get_semantic_context, get_full_context
    SEMANTIC_WORLD_AVAILABLE = True
except ImportError:
    SEMANTIC_WORLD_AVAILABLE = False
    get_semantic_context = None
    get_full_context = None

# Scene Protocol integration (optional - graceful fallback if not available)
try:
    from scene_protocol_integration import (
        SCENE_PROTOCOL_AVAILABLE,
        prepare_facet_context,
        finalize_facet_context,
        sync_agent_to_noodling,
    )
except ImportError:
    SCENE_PROTOCOL_AVAILABLE = False
    prepare_facet_context = None
    finalize_facet_context = None
    sync_agent_to_noodling = None

logger = logging.getLogger(__name__)

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
        prompt = f"""Translate this text into kitten behavior and body language.

Phi is a kitten who CANNOT speak words. She communicates through:
- Vocalizations: meow, purr, hiss, chirp, mew (NEVER bark, woof, or dog sounds!)
- Body language: ear flicks, tail movements, paw gestures
- Actions: rubs, pounces, curls, watches

Input: "{text}"

Examples:
- "I'm happy to see you" → "*purrs loudly and rubs against your leg*"
- "I want that" → "*meows softly and reaches paw toward it*"
- "That's interesting" → "*watches intently, ears forward, tail twitching*"
- "I'm curious" → "*tilts head, ears perking up*"

CRITICAL RULES:
- NO human words spoken directly (Phi cannot talk!)
- NO dog sounds (no bark, woof, etc. - ONLY cat sounds: meow, purr, hiss, chirp, mew)
- Use actions and sounds, NOT explanations
- Keep responses concise (1-3 lines max)
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
        voice_model = model or "SMALL"
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


class MemoryListWrapper:
    """
    Adapter that makes HierarchicalMemory quack like a list.

    Provides backward compatibility for conversation_context operations
    while using the sophisticated HierarchicalMemory backend.

    Handles format translation between:
    - Old format: dict with 'user', 'text', 'affect', 'surprise', 'timestamp', 'identity_salience', etc.
    - New format: MemoryEntry with user_id, user_text, affect, surprise, timestamp, importance

    Preserves extra fields (identity_salience, is_rumination, stage_direction) in side storage.
    """

    def __init__(self, hierarchical_memory: HierarchicalMemory, agent):
        """
        Initialize wrapper.

        Args:
            hierarchical_memory: The HierarchicalMemory instance to wrap
            agent: The CMUSHNoodlingAgent instance (for step counter)
        """
        self.hm = hierarchical_memory
        self.agent = agent
        # Side storage for extra fields not in MemoryEntry
        # Key: (timestamp, user_id, text_hash) -> dict of extra fields
        self._extra_fields = {}
        # Query context for semantic boosting
        self.last_query = ""

    def _make_key(self, timestamp: float, user_id: str, text: str) -> tuple:
        """Create unique key for extra fields storage."""
        # Use hash of text to keep key size reasonable
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        return (timestamp, user_id, text_hash)

    def append(self, entry_dict: Dict):
        """
        Add memory entry (old dict format).

        Converts to MemoryEntry format and adds to HierarchicalMemory.
        Stores extra fields in side storage.

        Args:
            entry_dict: Dict with keys: user, text, affect, surprise, timestamp, etc.
        """
        import mlx.core as mx

        # Extract core fields
        timestamp = entry_dict.get('timestamp', time.time())
        user_id = entry_dict['user']
        text = entry_dict['text']
        affect = entry_dict['affect']
        surprise = entry_dict.get('surprise', 0.0)

        # Capture user queries for semantic boosting
        if user_id.startswith('user_'):
            self.last_query = text

        # Validate affect array - handle empty or malformed data
        # Check for None first, then check length (avoid boolean evaluation of arrays)
        if affect is None or (hasattr(affect, '__len__') and len(affect) == 0):
            # Default neutral affect if missing or empty
            affect = [0.0, 0.0, 0.0, 0.0, 0.0]
            logger.warning(f"Empty affect vector in saved state for {user_id}, using neutral default")
        elif hasattr(affect, '__len__') and len(affect) < 5:
            # Pad to 5-D if incomplete
            # Convert to list first if it's an array
            import mlx.core as mx
            if isinstance(affect, mx.array):
                affect = affect.tolist()
            affect = list(affect) + [0.0] * (5 - len(affect))
            logger.warning(f"Incomplete affect vector ({len(entry_dict['affect'])}-D) for {user_id}, padded to 5-D")

        # Convert affect to MLX array if needed
        if not isinstance(affect, mx.array):
            affect = mx.array(affect, dtype=mx.float32)

        # Get phenomenal state from agent (if available)
        try:
            phenomenal_state = self.agent.consciousness.get_state()
        except:
            phenomenal_state = {}

        # Get step counter from agent
        step = self.agent.response_count

        # Determine if this is a response (agent's own message)
        response = text if user_id == self.agent.agent_id else None

        # Add to hierarchical memory
        self.hm.add(
            timestamp=timestamp,
            step=step,
            user_id=user_id,
            user_text=text,
            affect=affect,
            phenomenal_state=phenomenal_state,
            surprise=surprise,
            response=response
        )

        # Store extra fields
        key = self._make_key(timestamp, user_id, text)
        extra = {}
        if 'identity_salience' in entry_dict:
            extra['identity_salience'] = entry_dict['identity_salience']
        if 'is_rumination' in entry_dict:
            extra['is_rumination'] = entry_dict['is_rumination']
        if 'stage_direction' in entry_dict:
            extra['stage_direction'] = entry_dict['stage_direction']
        if 'stage_motivation' in entry_dict:
            extra['stage_motivation'] = entry_dict['stage_motivation']
        if 'is_self_monitor' in entry_dict:
            extra['is_self_monitor'] = entry_dict['is_self_monitor']

        if extra:
            self._extra_fields[key] = extra

    def _entry_to_dict(self, entry) -> Dict:
        """
        Convert MemoryEntry back to old dict format.

        Args:
            entry: MemoryEntry instance

        Returns:
            Dict in old format with extra fields restored
        """
        import mlx.core as mx

        # Convert affect to list
        if isinstance(entry.affect, mx.array):
            affect_list = entry.affect.tolist()
            # Handle squeezed arrays
            if isinstance(affect_list, list) and len(affect_list) > 0:
                if isinstance(affect_list[0], list):
                    affect_list = affect_list[0]
        else:
            affect_list = list(entry.affect)

        # Base dict
        result = {
            'user': entry.user_id,
            'text': entry.user_text,
            'affect': affect_list,
            'surprise': entry.surprise,
            'timestamp': entry.timestamp,
            'importance': entry.importance  # CRITICAL: Include actual importance score
        }

        # Restore extra fields
        key = self._make_key(entry.timestamp, entry.user_id, entry.user_text)
        if key in self._extra_fields:
            result.update(self._extra_fields[key])
        else:
            # Default values if not found
            result['identity_salience'] = 0.0

        return result

    def __getitem__(self, key):
        """
        Support indexing and slicing.

        Returns hybrid context: working memory + important episodic memories.
        This ensures loaded memories (which are all in episodic) are accessible.

        Args:
            key: int index or slice object

        Returns:
            Single dict (if int) or list of dicts (if slice)
        """
        if isinstance(key, slice):
            # Slicing: return HYBRID context (working + episodic)
            # This is critical for cross-session persistence!

            if key.start is None and key.stop is None:
                # [:] means all - return working + all episodic
                working = self.hm.retrieve_working(last_n=None)
                episodic = self.hm.episodic_memory  # All episodic memories
                # Combine: episodic first (chronological), then working (recent)
                entries = list(episodic) + working
            elif key.start is not None and key.start < 0:
                # [-N:] means last N - return STRATIFIED hybrid context
                # This balances: recent + conversations + self-monitoring + anomalies
                last_n = abs(key.start)
                working = self.hm.retrieve_working(last_n=None)

                # STRATIFIED EPISODIC RETRIEVAL
                # Separate episodic memories by type
                episodic_all = list(self.hm.episodic_memory)
                logger.info(f"[STRATIFIED] Total episodic: {len(episodic_all)}")

                # SEMANTIC BOOST: Extract keywords from recent query
                query_keywords = set()
                if self.last_query:
                    # Extract meaningful words (>3 chars, lowercase)
                    words = self.last_query.lower().split()
                    query_keywords = {w for w in words if len(w) > 3 and w.isalpha()}
                    if query_keywords:
                        logger.info(f"[SEMANTIC] Query keywords: {query_keywords}")

                # Apply semantic boost to memories containing query keywords
                SEMANTIC_BOOST = 2.0  # Multiply importance by this for matching memories
                if query_keywords:
                    for entry in episodic_all:
                        text_lower = entry.user_text.lower()
                        matches = [kw for kw in query_keywords if kw in text_lower]
                        if matches:
                            entry.importance *= SEMANTIC_BOOST
                            logger.debug(f"[SEMANTIC] Boosted memory (keywords: {matches}): {entry.user_text[:60]}...")

                self_monitoring = [e for e in episodic_all if '[self-monitoring]' in e.user_text]
                conversations = [e for e in episodic_all if '[self-monitoring]' not in e.user_text and e.user_id.startswith('user_')]
                agent_speech = [e for e in episodic_all if '[self-monitoring]' not in e.user_text and e.user_id.startswith('agent_')]
                logger.info(f"[STRATIFIED] Conversations: {len(conversations)}, Self-monitoring: {len(self_monitoring)}, Agent speech: {len(agent_speech)}")

                # Allocate slots (stratified sampling)
                # Generous allocation for rich character depth - prioritize agent's own significant statements
                episodic_slots = max(int(last_n * 1.2), 15)  # 120% of requested context, minimum 15
                conversation_slots = int(episodic_slots * 0.35)  # 35% conversations (user messages)
                speech_slots = int(episodic_slots * 0.50)  # 50% agent speech (agent's own statements) - PRIORITIZED
                selfmon_slots = episodic_slots - conversation_slots - speech_slots  # Remaining for self-monitoring
                logger.info(f"[STRATIFIED] Allocated slots - Conv: {conversation_slots}, Self-mon: {selfmon_slots}, Speech: {speech_slots}")

                # Get top entries from each category (by importance)
                top_conversations = sorted(conversations, key=lambda e: e.importance, reverse=True)[:conversation_slots]
                top_selfmon = sorted(self_monitoring, key=lambda e: e.importance, reverse=True)[:selfmon_slots]
                top_speech = sorted(agent_speech, key=lambda e: e.importance, reverse=True)[:speech_slots]
                logger.info(f"[STRATIFIED] Retrieved - Conv: {len(top_conversations)}, Self-mon: {len(top_selfmon)}, Speech: {len(top_speech)}")

                # DEBUG: Log retrieved agent_speech content
                for i, entry in enumerate(top_speech):
                    preview = entry.user_text[:80].replace('\n', ' ')
                    logger.info(f"[STRATIFIED-DEBUG] Speech[{i}]: imp={entry.importance:.4f}, text: {preview}...")

                # Combine episodic (stratified) + working (recent)
                episodic_stratified = top_conversations + top_selfmon + top_speech
                entries = list(episodic_stratified) + working
                entries = entries[-last_n:] if len(entries) > last_n else entries
            else:
                # Other slices: get all and slice in Python
                working = self.hm.retrieve_working(last_n=None)
                episodic = list(self.hm.episodic_memory)
                entries = episodic + working
                entries = entries[key]

            # Convert to dicts
            return [self._entry_to_dict(e) for e in entries]
        else:
            # Single index - use combined list
            working = self.hm.retrieve_working(last_n=None)
            episodic = list(self.hm.episodic_memory)
            entries = episodic + working

            if key < 0:
                key = len(entries) + key
            if key < 0 or key >= len(entries):
                raise IndexError("list index out of range")
            return self._entry_to_dict(entries[key])

    def __len__(self) -> int:
        """Return number of items in working memory."""
        return len(self.hm.working_memory)

    def __iter__(self):
        """Support iteration and list comprehensions."""
        entries = self.hm.retrieve_working(last_n=None)
        for entry in entries:
            yield self._entry_to_dict(entry)

    def clear(self):
        """Clear all memories."""
        self.hm.working_memory.clear()
        self.hm.episodic_memory.clear()
        self._extra_fields.clear()
        logger.info("Memory cleared (working + episodic)")

    def trim(self, max_size: int):
        """
        Trim working memory to max size.

        Note: This is largely a no-op since HierarchicalMemory
        automatically manages capacity via deque maxlen.

        Args:
            max_size: Maximum working memory size
        """
        # HierarchicalMemory already handles this via deque(maxlen=...)
        # Just log for observability
        current_size = len(self.hm.working_memory)
        if current_size > max_size:
            logger.debug(f"Trim requested to {max_size}, current size {current_size} (managed by HierarchicalMemory)")

    def load_from_list(self, memory_list: List[Dict]):
        """
        Load memories from list of dicts (for state restoration).

        Loaded memories are treated as pre-validated important memories
        and placed directly into episodic memory, bypassing importance
        threshold checks. This ensures cross-session persistence.

        Args:
            memory_list: List of memory dicts in old format
        """
        import mlx.core as mx

        self.clear()

        for entry_dict in memory_list:
            # Extract and validate data (same as append())
            timestamp = entry_dict.get('timestamp', time.time())
            user_id = entry_dict['user']
            text = entry_dict['text']
            affect = entry_dict['affect']
            surprise = entry_dict.get('surprise', 0.0)

            # Validate affect
            if affect is None or (hasattr(affect, '__len__') and len(affect) == 0):
                affect = [0.0, 0.0, 0.0, 0.0, 0.0]
            elif hasattr(affect, '__len__') and len(affect) < 5:
                if isinstance(affect, mx.array):
                    affect = affect.tolist()
                affect = list(affect) + [0.0] * (5 - len(affect))

            # Convert to MLX array
            if not isinstance(affect, mx.array):
                affect = mx.array(affect, dtype=mx.float32)

            # Get phenomenal state
            try:
                phenomenal_state = self.agent.consciousness.get_state()
            except:
                phenomenal_state = {}

            step = self.agent.response_count
            response = text if user_id == self.agent.agent_id else None

            # Compute importance (for logging/stats)
            importance = self.hm._compute_importance(surprise, affect, response)

            # Create MemoryEntry
            from noodlings.memory.hierarchical_memory import MemoryEntry
            entry = MemoryEntry(
                timestamp=timestamp,
                step=step,
                user_id=user_id,
                user_text=text,
                affect=affect,
                phenomenal_state=phenomenal_state,
                surprise=surprise,
                response=response,
                importance=importance
            )

            # Add to working memory
            self.hm.working_memory.append(entry)

            # CRITICAL: Add to episodic memory directly (bypass threshold)
            # These memories survived saving - they're pre-validated as important
            self.hm.episodic_memory.append(entry)
            self.hm.consolidations += 1

            # Store extra fields
            key = self._make_key(timestamp, user_id, text)
            extra = {}
            if 'identity_salience' in entry_dict:
                extra['identity_salience'] = entry_dict['identity_salience']
            if 'is_rumination' in entry_dict:
                extra['is_rumination'] = entry_dict['is_rumination']
            if 'stage_direction' in entry_dict:
                extra['stage_direction'] = entry_dict['stage_direction']
            if 'stage_motivation' in entry_dict:
                extra['stage_motivation'] = entry_dict['stage_motivation']
            if 'is_self_monitor' in entry_dict:
                extra['is_self_monitor'] = entry_dict['is_self_monitor']
            if extra:
                self._extra_fields[key] = extra

        logger.info(f"Loaded {len(memory_list)} memories from saved state (all preserved in episodic)")

    def copy(self) -> List[Dict]:
        """
        Return a shallow copy of working memory as list of dicts.

        Used by session profiler and other systems that need
        a snapshot of current conversation context.

        Returns:
            List of memory dicts in old format
        """
        entries = self.hm.retrieve_working(last_n=None)
        return [self._entry_to_dict(e) for e in entries]


class CMUSHNoodlingAgent(PerceptionMixin, ResponseGenerationMixin, CognitionLoopMixin, StatePersistenceMixin):
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

        # Cognitive gate locking for pause/resume
        self.cognition_paused = False
        self.cognitive_gate_id = 0  # Increments on each processing cycle
        self.pending_responses = []  # Queue for responses that arrive while paused

        # Cognition cycle tracking (for Noodle Tuner temporal alignment)
        import uuid as uuid_lib
        self.current_cycle_uuid = str(uuid_lib.uuid4())
        self.current_cycle_timestamp = time.time()
        self.cycle_in_progress = False
        self.pending_llm_calls = 0  # Count of LLM calls not yet returned

        # Step mode debugging
        self.step_mode_enabled = False
        self.step_mode_waiting = False
        self.step_mode_cycle_id = None

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

        logger.info(f"[{agent_id}] ABOUT TO INITIALIZE NoodlingAgent with checkpoint: {checkpoint_path}")
        try:
            self.consciousness = NoodlingAgent(
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
            logger.info(f"[{agent_id}] NoodlingAgent initialized successfully")
        except Exception as e:
            logger.error(f"[{agent_id}] FAILED to initialize NoodlingAgent: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # State history for analysis
        self.state_history = []  # Track phenomenal states over time
        self.surprise_history = []  # Track surprise for predictive processing evaluation

        # Affect Head - continuous 5D affect prediction (Option B)
        try:
            self.affect_head = AffectHead.load_pretrained()
            logger.info(f"[{agent_id}]  Affect Head loaded (continuous 5D prediction)")
        except Exception as e:
            logger.warning(f"[{agent_id}]   Could not load affect head: {e}")
            self.affect_head = None

        # Unity-style event system
        self.OnAffectChange = Event("OnAffectChange")
        self.OnFACSChange = Event("OnFACSChange")
        self.OnLabanChange = Event("OnLabanChange")
        self.OnSpeak = Event("OnSpeak")
        self.OnEmote = Event("OnEmote")
        self.OnAction = Event("OnAction")
        self.OnThink = Event("OnThink")
        self.OnSurpriseSpike = Event("OnSurpriseSpike")

        # Load Facet Assembly (required)
        facet_assembly_config = config.get('facet_assembly')

        if facet_assembly_config:
            # Handle both old ref format and direct string format
            if isinstance(facet_assembly_config, dict):
                facet_assembly_name = facet_assembly_config.get('ref')
            else:
                facet_assembly_name = facet_assembly_config

            if facet_assembly_name:
                try:
                    # Import facet system
                    import sys
                    noodlestudio_path = os.path.join(os.path.dirname(__file__), '../noodlestudio')
                    if noodlestudio_path not in sys.path:
                        sys.path.insert(0, noodlestudio_path)

                    from noodlestudio.core.facet_system import FacetAssembly
                    from noodlestudio.core.facet_executor import FacetExecutor

                    # Resolve assembly path - check library first, then facet_assemblies
                    noodlestudio_dir = os.path.join(os.path.dirname(__file__), '../noodlestudio')

                    if facet_assembly_name.startswith('library/'):
                        # Library template: library/empty_noodling -> library/noodlings/empty_noodling/assembly.yaml
                        template_name = facet_assembly_name.replace('library/', '')
                        assembly_path = os.path.join(
                            noodlestudio_dir,
                            'library/noodlings',
                            template_name,
                            'assembly.yaml'
                        )
                        logger.info(f"[{agent_id}] Loading library template: {template_name}")
                    else:
                        # Standard facet assembly
                        assembly_path = os.path.join(
                            noodlestudio_dir,
                            'facet_assemblies',
                            f'{facet_assembly_name}.yaml'
                        )

                    self.facet_assembly = FacetAssembly.load_yaml(assembly_path)
                    # HYBRID STRATEGY: Use 'serial' for debug, 'hybrid' for production
                    self.facet_executor = FacetExecutor(
                        llm_client=self.llm,
                        use_event_bus=True,
                        concurrency_mode='hybrid'  # 'serial' for debug, 'hybrid' for production
                    )

                    logger.info(f"[{agent_id}] Loaded facet assembly: {facet_assembly_name} ({len(self.facet_assembly.facets)} facets)")

                except Exception as e:
                    logger.error(f"[{agent_id}] Failed to load facet assembly '{facet_assembly_name}': {e}")
                    raise RuntimeError(f"Facet assembly required but failed to load: {e}")
            else:
                raise RuntimeError(f"[{agent_id}] facet_assembly config exists but no ref/name found")
        else:
            raise RuntimeError(f"[{agent_id}] No facet_assembly in config - facet assembly is required")

        # cMUSH-specific state
        self.current_room = None

        # Noodle Tuner instrumentation - store last manifold integration
        self.last_perception_text: Optional[str] = None
        self.last_manifold_output: Optional[str] = None

        # Cognition pause control (for debugging with Noodle Tuner)
        self.cognition_paused: bool = False

        # Initialize HierarchicalMemory with wrapped interface
        memory_config = config.get('memory_windows', {})
        working_capacity = memory_config.get('working_capacity', 20)
        episodic_capacity = memory_config.get('episodic_capacity', 200)
        hierarchical_memory = HierarchicalMemory(
            working_capacity=working_capacity,
            episodic_capacity=episodic_capacity,
            surprise_threshold=0.3,  # Lowered from 0.5 - consolidate more memories
            importance_decay=0.95
        )
        self.conversation_context = MemoryListWrapper(hierarchical_memory, self)
        logger.info(f"[{agent_id}] Initialized HierarchicalMemory: working={working_capacity}, episodic={episodic_capacity}")

        self.last_response_time = 0.0
        self.response_count = 0
        self.following = None  # User ID we're currently following (if any)

        # Self-protection: Track users the agent has withdrawn from
        self.withdrawn_users = {}  # user_id -> timestamp of withdrawal

        # Subconscious: Latent memory pool for symbolic insights
        self.latent_memories = []  # Symbolic images that haven't surfaced yet
        self.max_latent_memories = 10  # Keep last 10 symbolic abstractions

        # Phase 6: Self-monitoring state
        # FACS: Facial expression tracking
        self.last_facial_expression_time = 0.0  # Cooldown tracker
        self.previous_affect = None  # Track affect changes for FACS triggers

        # Phase 7: Affective Reinforcement Learning (make characters WANT their behaviors)
        reinforcement_config = self.config.get('affective_reinforcement', {})
        logger.info(f"[{agent_id}] DEBUG: self.config keys = {list(self.config.keys())}")
        logger.info(f"[{agent_id}] DEBUG: reinforcement_config = {reinforcement_config}")
        if reinforcement_config.get('enabled', False):
            reinforcement_type = reinforcement_config.get('type', 'comedy')
            intensity = reinforcement_config.get('intensity', 1.0)
            self.affective_reinforcement = create_reinforcement(
                reinforcement_type,
                enabled=True,
                intensity=intensity
            )
            logger.info(f"[{self.agent_id}]  Affective reinforcement enabled: type={reinforcement_type}, intensity={intensity}")
        else:
            self.affective_reinforcement = None
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

        # CONTINUOUS COGNITION SYSTEM (affect-driven, not timer-based)
        self.cognition_enabled = config.get('continuous_cognition', {}).get('enabled', True)
        self.cognition_task = None
        self.cognition_paused = False

        # Timing for speech cooldown (prevent spam)
        self.last_speech_time = 0.0
        self.min_speech_interval = config.get('min_speech_interval', 15.0)  # seconds

        # Continuous cognition check interval (polling frequency, not thinking frequency)
        self.cognition_check_interval = 0.5  # Check every 500ms if we should think

        logger.info(f"Continuous affect-driven cognition: {'enabled' if self.cognition_enabled else 'disabled'}")

        # COMPONENT SYSTEM: Initialize cognitive component registry (LEGACY - only for non-facet agents)
        from noodling_components import (
            ComponentRegistry,
            CharacterVoiceComponent,
            IntuitionReceiverComponent,
            SocialExpectationDetectorComponent
        )

        # Facet-based agents don't use legacy ComponentRegistry
        self.components = None

        logger.info(f"Agent initialized: {agent_id} (extraversion={extraversion:.2f}, threshold={adjusted_threshold:.6f})")

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
                'type': 'dominance',
                'dominance_boost': -0.18,  # Fear = low dominance
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
            model = intuition_config.get('model', 'SMALL')
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

                # Handle dict responses (some LLM clients return {text: ...})
                if isinstance(result_text, dict):
                    result_text = result_text.get('text', result_text.get('content', ''))
                result_text = str(result_text)

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

                # Modulate urgency based on current AFFECT STATE (not static traits!)
                # High arousal + positive valence = more socially responsive
                # Fear = social inhibition
                base_urgency = float(result['urgency'])

                # Get current affect from phenomenal state
                phenomenal = self.consilience_agent.get_current_state()
                valence = float(phenomenal[0]) if len(phenomenal) > 0 else 0.0
                arousal = float(phenomenal[1]) if len(phenomenal) > 1 else 0.5
                fear = float(phenomenal[2]) if len(phenomenal) > 2 else 0.0

                # Social activation: arousal amplifies, fear inhibits
                social_activation = arousal * (1.0 - fear * 0.5)  # 0.0-1.0 range

                # Modulate urgency
                intensity_multiplier = expectation_config.get('intensity_multiplier', 1.0)
                modulated_urgency = base_urgency * (0.6 + social_activation * 0.4)  # 0.6-1.0x range
                modulated_urgency *= intensity_multiplier
                modulated_urgency = min(1.0, modulated_urgency)  # Cap at 1.0

                result['urgency'] = modulated_urgency
                result['base_urgency'] = base_urgency
                result['social_activation'] = social_activation  # For debugging

                # Log detection
                if result['expected']:
                    logger.info(f"[{self.agent_id}]  Social expectation: {result['type']} "
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

    def set_affect_override(self, affect_dict: Dict[str, float]):
        """
        Override affect vector for next cognition cycle.

        Used by lab system to test random affect vs real affect.

        Args:
            affect_dict: Affect vector with keys:
                - valence: [-1.0, 1.0]
                - arousal: [0.0, 1.0]
                - fear: [0.0, 1.0]
                - sorrow: [0.0, 1.0]
                - boredom: [0.0, 1.0]
        """
        self.affect_override = affect_dict
        logger.debug(f"[{self.agent_id}] Affect override set: valence={affect_dict['valence']:.2f}")

    def clear_affect_override(self):
        """Clear affect override."""
        self.affect_override = None
        logger.debug(f"[{self.agent_id}] Affect override cleared")

    def get_current_affect(self) -> Dict[str, float]:
        """
        Get current affect (override if set, else predicted).

        Returns:
            Affect dictionary with valence, arousal, fear, sorrow, boredom
        """
        if hasattr(self, 'affect_override') and self.affect_override:
            return self.affect_override
        elif hasattr(self, 'affect_head') and self.affect_head:
            # Get phenomenal state
            h_fast, c_fast, h_medium, c_medium, h_slow = self.consciousness.model.get_states()
            import mlx.core as mx
            phenomenal_state = mx.concatenate([h_fast[0], h_medium[0], h_slow[0]], axis=0)

            # Predict affect (returns dict already)
            affect_dict = self.affect_head.predict(phenomenal_state)
            return affect_dict
        else:
            # Fallback: neutral affect
            return {
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.5,
                'sorrow': 0.0,
                'boredom': 0.0
            }

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
        logger.info(f" [SELF-MONITOR] Triggering for {self.agent_name} (surprise={surprise:.3f}, cooldown={time_since_last:.1f}s)")
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
            valence_delta = emotional_impact.get('valence_delta', 0.0)
            arousal_delta = emotional_impact.get('arousal_delta', 0.0)
            dominance_delta = emotional_impact.get('dominance_delta', emotional_impact.get('fear_delta', 0.0))  # Legacy support

            # Apply affective updates to phenomenal state
            # Note: This modifies the internal state for the NEXT cycle
            if abs(valence_delta) > 0.05 or abs(arousal_delta) > 0.05 or abs(dominance_delta) > 0.05:
                logger.info(f"[SELF-MONITOR] {self.agent_name} felt: valence{valence_delta:+.2f}, arousal{arousal_delta:+.2f}, dominance{dominance_delta:+.2f}")

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

    async def _evaluate_action_intention(self, thought_text: str, context: Dict) -> float:
        """
        Evaluate if an internal thought contains actionable intention.

        Returns continuous 0-1 score:
        - 0.0 = Pure observation, no action implied
        - 0.5 = Moderate intention (considering action)
        - 1.0 = Strong intention (clear plan to act)

        Examples:
        - "I notice the sunset" → 0.1 (passive observation)
        - "I wonder if I should say something" → 0.4 (weak intention)
        - "I'll roast them back with sharp sass" → 0.9 (strong intention)
        - "I want to grab that candy" → 0.8 (clear action desire)
        """
        prompt = f"""Analyze this internal thought for ACTION INTENTION.

THOUGHT:
"{thought_text}"

Does this thought contain intention to ACT (speak, move, interact)?

Rate the ACTION INTENTION on a 0-1 scale:
0.0 = Pure observation, no action (e.g., "The sky is blue")
0.3 = Weak intention, considering (e.g., "I wonder if I should...")
0.6 = Moderate intention, leaning toward action (e.g., "I want to...")
0.9 = Strong intention, clear plan (e.g., "I'll grab that", "I'm going to say...")

Look for keywords like:
- "I'll", "I'm going to", "I should", "I want to", "I need to"
- Plans, desires, impulses to act
- Intentions to speak, move, interact

Output ONLY a number between 0.0 and 1.0. No explanation."""

        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        self._increment_llm_counter()
        try:
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt="You are an action intention evaluator. Output only a number 0.0-1.0.",
                model=model,
                max_tokens=10,
                temperature=0.1
            )

            # Handle dict responses (some LLM clients return {text: ...})
            if isinstance(response, dict):
                response = response.get('text', response.get('content', ''))
            response = str(response)

            # Parse numeric response
            try:
                intention = float(response.strip())
                intention = max(0.0, min(1.0, intention))  # Clamp to 0-1
                return intention
            except ValueError:
                logger.warning(f"Could not parse intention score: {response}, defaulting to 0.3")
                return 0.3

        except Exception as e:
            logger.error(f"Action intention evaluation failed: {e}")
            return 0.3  # Default moderate intention

        finally:
            self._decrement_llm_counter()

    def _compute_self_restraint(self, state: Dict) -> float:
        """
        Compute self-restraint: the threshold for acting on internal intentions.

        Low restraint (0.1) = impulsive, disinhibited, acts on every thought
        High restraint (0.9) = cautious, filtered, rarely externalizes thoughts

        Factors:
        - Fear: High fear increases restraint (don't want to mess up)
        - Conscientiousness: High conscientiousness increases restraint (think before acting)
        - Arousal: Very high arousal can overwhelm restraint (excited/panicked)
        - Boredom: High boredom increases restraint (disengaged, don't care to act)

        Examples:
        - Drunk person: Low fear, low conscientiousness → restraint ≈ 0.1
        - Anxious person: High fear → restraint ≈ 0.8
        - Impulsive child: Low conscientiousness, low fear → restraint ≈ 0.2
        """
        # Extract affect
        phenomenal = state.get('phenomenal_state', [0]*5)
        valence = float(phenomenal[0]) if len(phenomenal) > 0 else 0.0
        arousal = float(phenomenal[1]) if len(phenomenal) > 1 else 0.5
        fear = float(phenomenal[2]) if len(phenomenal) > 2 else 0.0
        sorrow = float(phenomenal[3]) if len(phenomenal) > 3 else 0.0
        boredom = float(phenomenal[4]) if len(phenomenal) > 4 else 0.0

        # Compute restraint from PURE AFFECT (no static traits!)
        # Base restraint: starts at neutral 0.5
        base_restraint = 0.5

        # Fear increases restraint (afraid to act, social inhibition)
        fear_modifier = fear * 0.3

        # Sorrow increases restraint (withdrawn, low energy)
        sorrow_modifier = sorrow * 0.2

        # Very high arousal can overwhelm restraint (fight/flight/excitement)
        arousal_overwhelm = max(0, arousal - 0.8) * -0.5  # Only kicks in above 0.8 arousal

        # Boredom increases restraint (disengaged, don't care)
        boredom_modifier = boredom * 0.2

        # Negative valence increases restraint (unhappy = less outgoing)
        valence_modifier = -valence * 0.15 if valence < 0 else 0.0

        # Final restraint (bounded 0.05-0.95)
        restraint = base_restraint + fear_modifier + sorrow_modifier + arousal_overwhelm + boredom_modifier + valence_modifier
        restraint = max(0.05, min(0.95, restraint))

        return restraint

    def get_phenomenal_state(self) -> Dict:
        """
        Get current phenomenal state (for @observe command).

        Returns:
            State dictionary
        """
        state = self.consciousness.get_state()

        # DEBUG: Log what consciousness is returning
        logger.info(f"[{self.agent_id}] get_phenomenal_state() called")
        logger.info(f"  fast: {type(state.get('fast'))}, len={len(state.get('fast')) if state.get('fast') is not None else 'None'}")
        logger.info(f"  medium: {type(state.get('medium'))}, len={len(state.get('medium')) if state.get('medium') is not None else 'None'}")
        logger.info(f"  slow: {type(state.get('slow'))}, len={len(state.get('slow')) if state.get('slow') is not None else 'None'}")

        return state

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

    def get_memory_stats(self) -> Dict:
        """
        Get memory system statistics.

        Returns:
            Dict with working_count, episodic_count, consolidations, etc.
        """
        if hasattr(self.conversation_context, 'hm'):
            hm = self.conversation_context.hm
            return {
                'working_count': len(hm.working_memory),
                'working_capacity': hm.working_capacity,
                'episodic_count': len(hm.episodic_memory),
                'episodic_capacity': hm.episodic_capacity,
                'consolidations': hm.consolidations,
                'evictions': hm.evictions,
                'threshold': hm.surprise_threshold
            }
        return {
            'working_count': len(self.conversation_context) if hasattr(self.conversation_context, '__len__') else 0,
            'working_capacity': 0,
            'episodic_count': 0,
            'episodic_capacity': 0,
            'consolidations': 0,
            'evictions': 0,
            'threshold': 0.3
        }

    def get_working_memory(self) -> List[Dict]:
        """
        Get working memory entries.

        Returns:
            List of memory entries currently in working memory
        """
        if hasattr(self.conversation_context, 'hm'):
            hm = self.conversation_context.hm
            return [self.conversation_context._entry_to_dict(entry) for entry in hm.working_memory]
        return list(self.conversation_context) if hasattr(self.conversation_context, '__iter__') else []

    def get_episodic_memory(self, limit: int = None) -> List[Dict]:
        """
        Get episodic memory entries, sorted by importance.

        Args:
            limit: Maximum number of entries to return (None = all)

        Returns:
            List of memory entries in episodic storage, sorted by importance (highest first)
        """
        if hasattr(self.conversation_context, 'hm'):
            hm = self.conversation_context.hm
            sorted_episodic = sorted(hm.episodic_memory, key=lambda e: e.importance, reverse=True)
            if limit:
                sorted_episodic = sorted_episodic[:limit]
            return [self.conversation_context._entry_to_dict(entry) for entry in sorted_episodic]
        return []

    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search memories by text content.

        Args:
            query: Text to search for (case-insensitive)
            limit: Maximum results

        Returns:
            List of matching memory entries
        """
        query_lower = query.lower()
        results = []

        # Search working memory
        if hasattr(self.conversation_context, 'hm'):
            hm = self.conversation_context.hm
            for entry in hm.working_memory:
                if query_lower in entry.user_text.lower():
                    results.append(self.conversation_context._entry_to_dict(entry))

            # Search episodic memory
            for entry in hm.episodic_memory:
                if query_lower in entry.user_text.lower():
                    results.append(self.conversation_context._entry_to_dict(entry))

        return results[:limit]

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
        self.conversation_context.clear()
        self.last_response_time = 0.0
        self.response_count = 0
        logger.info(f"Agent reset: {self.agent_id}")

    def _parse_think_tags(self, text: str):
        """
        Parse text to separate <think> content from speech.

        Args:
            text: Raw LLM output potentially containing <think> tags

        Returns:
            Tuple of (thoughts, speech) where either can be empty string
        """
        import re

        # Extract all <think>...</think> content
        think_pattern = r'<think>(.*?)</think>'
        thoughts = re.findall(think_pattern, text, re.DOTALL)

        # Remove <think> tags and their content from text to get speech
        speech = re.sub(think_pattern, '', text, flags=re.DOTALL)

        # Clean up whitespace
        thoughts_text = ' '.join(thoughts).strip()
        speech_text = speech.strip()

        return thoughts_text, speech_text

    async def _broadcast_autonomous_speech(self, text: str):
        """
        Broadcast autonomous speech to the room.
        Parses <think> tags and sends thoughts separately from speech.

        Args:
            text: Speech text to broadcast (may contain <think> tags)
        """
        try:
            # Parse think tags
            thoughts, speech = self._parse_think_tags(text)

            # This will be picked up by get_autonomous_events()
            if not hasattr(self, '_pending_autonomous_events'):
                self._pending_autonomous_events = []

            # Broadcast thoughts first (if any)
            if thoughts:
                think_event = {
                    'type': 'think',
                    'user': self.agent_id,
                    'username': self.agent_name,
                    'room': self.current_room,
                    'text': thoughts,
                    'autonomous': True
                }
                self._pending_autonomous_events.append(think_event)

            # Then broadcast speech (if any)
            if speech:
                speech_event = {
                    'type': 'say',
                    'user': self.agent_id,
                    'username': self.agent_name,
                    'room': self.current_room,
                    'text': speech,
                    'autonomous': True
                }
                self._pending_autonomous_events.append(speech_event)

        except Exception as e:
            logger.error(f"Error broadcasting autonomous speech: {e}", exc_info=True)

    async def get_autonomous_events(self) -> List[Dict]:
        """
        Get pending autonomous events (spontaneous speech, etc.).

        Returns:
            List of event dicts for broadcasting
        """
        # Don't generate autonomous events when cognition is paused
        if self.cognition_paused:
            return []

        # Return and clear pending events
        if hasattr(self, '_pending_autonomous_events'):
            events = self._pending_autonomous_events
            self._pending_autonomous_events = []
            return events

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

    # ===== Agent Identity =====

    def GetUUID(self) -> str:
        """Get agent UUID."""
        return self.agent_id


class AgentManager:
    """
    Manages multiple CMUSHNoodlingAgent instances.

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
        self.agents: Dict[str, CMUSHNoodlingAgent] = {}

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
    ) -> CMUSHNoodlingAgent:
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

        # Add global llm settings (for lab_testing model access)
        if 'llm' in self.global_config:
            agent_config['llm'] = self.global_config['llm']

        # Override with agent-specific config
        if config:
            agent_config.update(config)

        # Create agent
        agent = CMUSHNoodlingAgent(
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
        # Take a snapshot of agents to avoid RuntimeError if dict changes during iteration
        for agent_id, agent in list(self.agents.items()):
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

    def get_agent(self, agent_id: str) -> Optional[CMUSHNoodlingAgent]:
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

