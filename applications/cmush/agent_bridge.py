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
import asyncio
import numpy as np
import aiohttp

from noodlings.api import NoodlingAgent as ConsilienceAgent
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
            agent: The CMUSHConsilienceAgent instance (for step counter)
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

        # Step mode debugging (for NoodleTuner)
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

        logger.info(f"[{agent_id}] ABOUT TO INITIALIZE ConsilienceAgent with checkpoint: {checkpoint_path}")
        try:
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
            logger.info(f"[{agent_id}] ConsilienceAgent initialized successfully")
        except Exception as e:
            logger.error(f"[{agent_id}] FAILED to initialize ConsilienceAgent: {e}")
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
        self._components = {}  # Component registry: type -> instance

        # Load Facet Assembly (if specified in config)
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

                    logger.info(f"[{agent_id}] ⚡ Loaded facet assembly: {facet_assembly_name} ({len(self.facet_assembly.facets)} facets)")

                    # Mark as using facet system (not transistors)
                    self.using_facet_system = True
                    self.cognitive_manifold = None  # No manifold when using facets

                except Exception as e:
                    logger.error(f"[{agent_id}] Failed to load facet assembly '{facet_assembly_name}': {e}")
                    logger.info(f"[{agent_id}] Falling back to legacy transistor system")
                    self.using_facet_system = False
                    self.facet_assembly = None
                    self.facet_executor = None
                    # Will initialize transistors below
            else:
                logger.warning(f"[{agent_id}] facet_assembly config exists but no ref/name found")
                self.using_facet_system = False
        else:
            logger.info(f"[{agent_id}] No facet_assembly in config, checking for legacy transistors")
            self.using_facet_system = False

        # Legacy transistor system (only if NOT using facets)
        if not self.using_facet_system:
            from cognitive_components import CognitiveManifold, COMPONENT_REGISTRY
            self.cognitive_manifold = CognitiveManifold()
            logger.info(f"[{agent_id}]  Created CognitiveManifold with LLM blending (LEGACY)")

            # Initialize cognitive components from recipe (Phase 7: Cognitive Manifold)
            cognitive_components_config = config.get('cognitive_components', {})

            if not cognitive_components_config:
                # Default transistors: affect + mood (minimal emotional processing)
                logger.info(f"[{agent_id}] No cognitive_components in recipe, using defaults")
                cognitive_components_config = {
                    'affect': {
                        'type': 'AffectTransistor',
                        'salience': 0.70,
                        'enabled': True
                    },
                    'mood': {
                        'type': 'MoodTransistor',
                        'salience': 0.50,
                        'enabled': True
                    }
                }

            # Create and register each transistor from config (recipe or defaults)
            for component_name, component_config in cognitive_components_config.items():
                transistor_type = component_config.get('type')
                if not transistor_type:
                    logger.warning(f"[{agent_id}] Component '{component_name}' missing 'type', skipping")
                    continue

                transistor_class = COMPONENT_REGISTRY.get(transistor_type)
                if not transistor_class:
                    logger.warning(f"[{agent_id}] Unknown transistor type '{transistor_type}', skipping")
                    continue

                try:
                    transistor = transistor_class.from_config(component_config)
                    self.cognitive_manifold.register_transistor(transistor)
                    logger.info(f"[{agent_id}]  Registered {transistor_type} (salience={transistor.salience:.2f})")
                except Exception as e:
                    logger.error(f"[{agent_id}] Failed to create {transistor_type}: {e}")

            # Load and register EmbodyComponent if not already registered via recipe
            from cognitive_components import EmbodyComponent
            has_embody = any(t.__class__.__name__ == 'EmbodyComponent' for t in self.cognitive_manifold.transistors)

            if not has_embody:
                embodiment_loader = EmbodimentLoader()
                embodiment_id = config.get('embodiment_id')

                if embodiment_id:
                    embodiment_data = embodiment_loader.load(embodiment_id)
                    if embodiment_data:
                        logger.info(f"[{agent_id}] Loaded embodiment: {embodiment_id}")
                    else:
                        logger.warning(f"[{agent_id}] Embodiment {embodiment_id} not found, using default")
                        embodiment_data = embodiment_loader.get_default_embodiment()
                else:
                    embodiment_data = embodiment_loader.get_default_embodiment()
                    logger.info(f"[{agent_id}] Using default embodiment")

                embody_component = EmbodyComponent(embodiment_data['embodiment'])
                self.cognitive_manifold.register_transistor(embody_component)
                logger.info(f"[{agent_id}]  Registered EmbodyComponent (salience=1.0)")
            else:
                logger.info(f"[{agent_id}]  EmbodyComponent already registered from recipe")

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

        # Only create ComponentRegistry if NOT using facet system
        # (Facet assemblies handle voice/intuition/social as scriptable facets)
        if not self.using_facet_system:
            self.components = ComponentRegistry(agent_id, self.agent_name)
            # Register Character Voice component
            voice_config = config.get('character_voice', {
                'enabled': True,
                'model': 'SMALL',
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
        else:
            # Facet-based agents don't use ComponentRegistry
            self.components = None

        logger.info(f"Agent initialized: {agent_id} (extraversion={extraversion:.2f}, threshold={adjusted_threshold:.6f})")
        if self.components:
            logger.info(f"[{agent_id}] Registered {len(self.components.components)} cognitive components")
        else:
            logger.info(f"[{agent_id}] Using facet assembly (no legacy components)")

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

            # Check if agent's name is in the message for better routing
            my_name_normalized = my_name.lower().replace('_', ' ').replace('fire', '').strip()
            message_lower = message_text.lower()
            name_mentioned = any(name_part in message_lower for name_part in my_name_normalized.split() if len(name_part) > 2)

            prompt = f"""You are {my_name}'s intuitive awareness - you report ONLY factual observations from the context provided.

AGENT NAME: {my_name}
MESSAGE TEXT: "{message_text}"

CONTEXT:
{context_text}

Generate brief factual awareness (1-2 sentences max) that reports:

1. WHO is being addressed - CHECK THE MESSAGE TEXT ABOVE:
   - If message contains "{my_name}" or "red" or "you": "This is directed at ME"
   - If message names someone else specifically: "This is for [name], not me"
   - If message is general/broadcast: "This is addressed to everyone present"

2. FACTUAL OBSERVATIONS from context:
   - Who is present (from room occupants list)
   - Who has what objects (from possessions list)
   - NO embellishments, NO invented details

CRITICAL:
- Check if MY NAME appears in the message text above!
- Only report information that exists in the CONTEXT
- NO invented details like "fingers gripping tables", "warm pulses", "blinks", or emotional states
- Keep it brief: 1-2 sentences maximum

Examples:
- "This is directed at ME - Caity said my name."
- "This is for Toad, not me."
- "This is addressed to everyone."

Generate factual awareness:"""

            # ALWAYS use fast model for intuition - don't use agent's model override
            # Intuition needs to be fast and reliable, not character-specific
            intuition_model = intuition_config.get('model', 'SMALL')
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

                # Handle dict responses (some LLM clients return {text: ...})
                if isinstance(intuition, dict):
                    intuition = intuition.get('text', intuition.get('content', ''))
                intuition = str(intuition)

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

    def save_state_snapshot(self) -> Dict[str, Any]:
        """
        Save complete agent state snapshot for in-memory restoration.

        Used by lab system for dual cognition experiments.
        Captures all stateful components:
        - Consciousness model hidden states (h_fast, c_fast, etc.)
        - Conversation context
        - Affect history
        - Cognitive manifold state
        - World interaction state

        Returns:
            Dictionary with all restorable state
        """
        import copy
        import mlx.core as mx

        # Get consciousness model states
        h_fast, c_fast, h_medium, c_medium, h_slow = self.consciousness.model.get_states()

        # Save states (convert MLX arrays to numpy for JSON compatibility)
        state = {
            'h_fast': np.array(h_fast) if h_fast is not None else None,
            'c_fast': np.array(c_fast) if c_fast is not None else None,
            'h_medium': np.array(h_medium) if h_medium is not None else None,
            'c_medium': np.array(c_medium) if c_medium is not None else None,
            'h_slow': np.array(h_slow) if h_slow is not None else None,
        }

        # Save conversation context (deep copy to prevent mutation)
        # Note: MemoryListWrapper needs special handling
        if hasattr(self.conversation_context, '_memory'):
            # Save hierarchical memory state
            state['conversation_context'] = {
                'working_memory': copy.deepcopy(list(self.conversation_context._memory.working_memory)),
                'episodic_memory': copy.deepcopy(list(self.conversation_context._memory.episodic_memory)),
            }
        else:
            # Fallback: save as list
            state['conversation_context'] = copy.deepcopy(list(self.conversation_context))

        # Save affect history (if it exists)
        if hasattr(self, 'previous_affect') and self.previous_affect is not None:
            state['previous_affect'] = copy.deepcopy(self.previous_affect)
        else:
            state['previous_affect'] = None

        # Save cognitive manifold state (if it exists)
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            state['cognitive_manifold'] = self.cognitive_manifold.save_state()
        else:
            state['cognitive_manifold'] = None

        # Save world interaction state
        state['current_room'] = self.current_room
        state['following'] = self.following
        state['last_response_time'] = self.last_response_time
        state['response_count'] = self.response_count

        # Save autonomous cognition state
        if hasattr(self, 'cognition_engine') and self.cognition_engine:
            state['cognition_engine_state'] = self.cognition_engine.save_state()
        else:
            state['cognition_engine_state'] = None

        logger.debug(f"[{self.agent_id}] State saved: h_fast shape={state['h_fast'].shape if state['h_fast'] is not None else None}")

        return state

    def restore_state_snapshot(self, state: Dict[str, Any]):
        """
        Restore agent to saved state snapshot.

        Used by lab system to reset agent between dual cognition trials.

        Args:
            state: State dictionary from save_state_snapshot()
        """
        import mlx.core as mx

        # Restore consciousness model states
        if state['h_fast'] is not None:
            self.consciousness.model.h_fast = mx.array(state['h_fast'])
        if state['c_fast'] is not None:
            self.consciousness.model.c_fast = mx.array(state['c_fast'])
        if state['h_medium'] is not None:
            self.consciousness.model.h_medium = mx.array(state['h_medium'])
        if state['c_medium'] is not None:
            self.consciousness.model.c_medium = mx.array(state['c_medium'])
        if state['h_slow'] is not None:
            self.consciousness.model.h_slow = mx.array(state['h_slow'])

        # Restore conversation context
        if 'conversation_context' in state:
            if isinstance(state['conversation_context'], dict):
                # Restore hierarchical memory
                self.conversation_context._memory.working_memory = list(state['conversation_context']['working_memory'])
                self.conversation_context._memory.episodic_memory = list(state['conversation_context']['episodic_memory'])
            else:
                # Fallback: restore as list
                # Note: This won't work perfectly with MemoryListWrapper, but provides basic restore
                logger.warning(f"[{self.agent_id}] Restoring conversation_context as list (not ideal)")

        # Restore affect history
        if 'previous_affect' in state:
            self.previous_affect = state['previous_affect']

        # Restore cognitive manifold state
        if state.get('cognitive_manifold') and hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            self.cognitive_manifold.restore_state(state['cognitive_manifold'])

        # Restore world interaction state
        self.current_room = state['current_room']
        self.following = state['following']
        self.last_response_time = state['last_response_time']
        self.response_count = state['response_count']

        # Restore autonomous cognition state
        if state.get('cognition_engine_state') and hasattr(self, 'cognition_engine') and self.cognition_engine:
            self.cognition_engine.restore_state(state['cognition_engine_state'])

        logger.debug(f"[{self.agent_id}] State restored")

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

    async def generate_response_text(self, message: str, force: bool = True, lab_mode: bool = False) -> str:
        """
        Generate response text to a message.

        Simplified interface for lab system. Forces response generation
        by bypassing surprise thresholds.

        Args:
            message: User message text
            force: If True, force generation even if surprise is low
            lab_mode: If True, use faster lab testing model from config

        Returns:
            Generated response text
        """
        if not force:
            # Use normal perceive_event flow
            event = {
                'type': 'say',
                'user': 'user_lab_test',
                'text': message,
                'room': self.current_room or 'room_0'
            }
            response = await self.perceive_event(event)
            if response and 'text' in response:
                return response['text']
            else:
                return "[Agent chose not to respond]"

        # FORCE MODE: Use full production system with current phenomenal state
        # This properly tests whether affect prediction improves responses

        # Build phenomenal state dict from current consciousness state
        # Get current affect (respects override if set)
        affect_dict = self.get_current_affect()

        # Create a minimal phenomenal state dict for the production generator
        phenomenal_state = {
            'h_fast': [
                affect_dict.get('valence', 0.0),
                affect_dict.get('arousal', 0.5),
                affect_dict.get('dominance', 0.5),
                affect_dict.get('sorrow', 0.0),
                affect_dict.get('boredom', 0.0)
            ] + [0.0] * 11,  # Pad to 16-D
            'surprise': 0.5,  # Moderate surprise to ensure response
            'surprise_threshold': 0.0001
        }

        # Create temporary context entry for this message (don't add to conversation_context)
        # Include affect field required by conversation_context format
        # Convert affect_dict to array format [valence, arousal, dominance, sorrow, boredom]
        affect_array = [
            affect_dict.get('valence', 0.0),
            affect_dict.get('arousal', 0.5),
            affect_dict.get('dominance', 0.5),
            affect_dict.get('sorrow', 0.0),
            affect_dict.get('boredom', 0.0)
        ]
        temp_context_entry = {
            'user': 'user_lab_test',
            'text': message,
            'timestamp': time.time(),
            'affect': affect_array
        }

        # Create modified context with temp entry appended (without mutating original)
        response_window = self.config.get('memory_windows', {}).get('response_generation', 5)
        context_with_message = list(self.conversation_context[-response_window:]) + [temp_context_entry]

        # Get relationship (or create minimal one)
        current_state = self.consciousness.get_state()
        relationships = current_state.get('relationships', {})
        relationship = relationships.get('user_lab_test', {
            'attachment_style': 'forming',
            'interaction_count': 0,
            'valence': 0.0
        })

        # Get identity memories (high-salience memories)
        identity_memories = sorted(
            [m for m in self.conversation_context if m.get('identity_salience', 0) > 0.3],
            key=lambda m: m.get('identity_salience', 0),
            reverse=True
        )[:2]

        # Select model: lab testing model if in lab mode, otherwise agent's normal model
        model_to_use = self.llm_model
        if lab_mode and hasattr(self, 'config') and 'llm' in self.config:
            lab_config = self.config['llm'].get('lab_testing', {})
            if 'model' in lab_config:
                model_to_use = lab_config['model']
                logger.info(f"[{self.agent_id}] Lab mode: Using {model_to_use}")

        # Use the FULL production generate_response system
        try:
            llm_result = await self.llm.generate_response(
                phenomenal_state=phenomenal_state,
                target_user='user_lab_test',
                conversation_context=context_with_message,  # Use modified context
                relationship=relationship,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                agent_description=self.agent_description,
                identity_prompt=self.identity_prompt,
                identity_memories=identity_memories,
                name_mentioned=False,
                enlightenment=self.config.get('enlightenment', False),
                model=model_to_use
            )

            # Extract response text
            if isinstance(llm_result, dict):
                response_text = llm_result.get('response', '[No response]')
            else:
                response_text = llm_result or '[No response]'

            logger.info(f"[{self.agent_id}] Lab generation (full system): {len(response_text)} chars")
            return response_text

        except Exception as e:
            logger.error(f"[{self.agent_id}] Lab generation failed: {e}")
            return "[Error generating response]"

    async def _check_cognitive_gate(self) -> bool:
        """
        Check if cognitive processing should proceed.
        Returns True if processing should continue, False if paused.
        """
        if self.cognition_paused:
            logger.debug(f"[{self.agent_id}] ⏸ Cognitive gate closed - operation blocked")
            return False
        return True

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

        # Initialize response_decision at function scope (used later in shared code)
        response_decision = None

        # Skip if not a perceivable event
        if event_type not in ['say', 'emote', 'enter', 'exit']:
            logger.debug(f"Skipping non-perceivable event: {event_type}")
            return None

        # Skip if self-action
        if user_id == self.agent_id:
            return None

        # Check if cognition is paused (for Noodle Tuner debugging)
        if getattr(self, 'cognition_paused', False):
            logger.info(f"[{self.agent_id}] ⏸ Cognition paused - queuing event for later processing")
            self.pending_responses.append(event)
            return None

        # CYCLE LOCKING: Prevent concurrent cognition cycles
        # BUT: REACTIVE cognition (user input) can INTERRUPT AUTONOMOUS cognition (rumination)
        if getattr(self, 'cycle_in_progress', False):
            current_cycle_type = getattr(self, 'cycle_type', 'unknown')

            # If current cycle is AUTONOMOUS (rumination), INTERRUPT IT!
            if current_cycle_type == 'autonomous':
                logger.warning(f"[{self.agent_id}] ⚡ INTERRUPTING autonomous cognition with reactive perception: {text[:50]}")
                # Force complete the current cycle
                self._complete_cognition_cycle()
                # Continue to start new REACTIVE cycle below
            else:
                # Current cycle is REACTIVE - must wait (don't interrupt user interactions!)
                logger.warning(f"[{self.agent_id}] [LOCKED] Reactive cycle in progress - QUEUING new perception: {text[:50]}")
                if not hasattr(self, 'pending_perceptions'):
                    self.pending_perceptions = []
                self.pending_perceptions.append(event)
                logger.info(f"[{self.agent_id}]  Queued perception (queue size: {len(self.pending_perceptions)})")
                return None

        # Agents can now perceive other agents
        is_agent = user_id.startswith('agent_')

        # Start new cognition cycle (REACTIVE type)
        import uuid as uuid_lib
        self.current_cycle_uuid = str(uuid_lib.uuid4())
        self.current_cycle_timestamp = time.time()
        self.cycle_in_progress = True
        self.cycle_type = 'reactive'  # Mark as REACTIVE (can be interrupted by nothing!)
        self.pending_llm_calls = 0
        logger.info(f"[{self.agent_id}] Starting REACTIVE cycle {self.current_cycle_uuid[:8]}")

        # IMMEDIATELY clear old cycle data and set new input (for NoodleTuner)
        self.last_perception_text = text  # Set input FIRST
        self.last_manifold_output = None  # Clear old output
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            self.cognitive_manifold.last_response_decision = None  # Clear old decision
            self.cognitive_manifold.last_output_text = None
            # Clear all transistor outputs
            for transistor in self.cognitive_manifold.transistors:
                transistor.last_output_text = None
                transistor.last_output_metadata = None
                transistor.register_state = "empty"

        logger.info(f"Agent {self.agent_id} perceiving: {event_type} from {user_id}: {text} [cycle={self.current_cycle_uuid[:8]}]")

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
            logger.info(f"[{self.agent_id}]  AFFECT EXTRACTED: valence={affect_raw[0]:.3f}, arousal={affect_raw[1]:.3f}, fear={affect_raw[2]:.3f}, sorrow={affect_raw[3]:.3f}, boredom={affect_raw[4]:.3f}")
            logger.debug(f"Extracted affect (raw): {affect_raw}")

            # 1a-0. GENERATE INTUITION FIRST (for cognitive manifold context)
            # This provides spatial/contextual awareness
            # SKIP for facet-based agents (intuition is handled by facets)
            intuition_text = None
            if not self.using_facet_system and self.world:  # Just check world exists
                try:
                    world_snapshot = {
                        'current_room': self.world.get_room(self.current_room),
                        'objects': self.world.objects,
                        'rooms': self.world.rooms,
                        'agents': self.world.agents,
                        'users': self.world.users
                    }
                    intuition_text = await self._generate_intuition(
                        event={'type': event_type, 'user': user_id, 'text': text, 'room': room_id},
                        world_state=world_snapshot,
                        recent_context=self.conversation_context[-3:]
                    )
                    if intuition_text:
                        logger.info(f"[{self.agent_id}]  Intuition (early): {intuition_text[:80]}...")
                except Exception as e:
                    logger.debug(f"Early intuition generation failed: {e}")

            # 1a-1. COGNITIVE PROCESSING - FACET SYSTEM vs LEGACY TRANSISTORS
            # Process perception through either facet assembly or cognitive transistors
            colored_perception = text  # Default to original text

            logger.info(f"[{self.agent_id}] DEBUG: About to enter cognitive processing. using_facet_system={self.using_facet_system}")

            try:
                # FACET SYSTEM vs LEGACY TRANSISTORS
                if self.using_facet_system:
                    logger.info(f"[{self.agent_id}] DEBUG: Entering FACET execution branch")
                    # Execute facet assembly
                    from noodlestudio.core.scripted_facet import ScriptContext

                    # Build script context with agent state
                    exec_context = ScriptContext(
                        cycle=self.current_cycle_uuid,
                        timestamp=time.time(),
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        agent_species=self.species
                    )

                    # Inject agent state (affect, identity, etc.)
                    exec_context._agent_state = {
                        'affect': affect_raw,
                        'identity': self.identity_prompt,
                        'species': self.species,
                        'personality_traits': getattr(self, 'personality_traits', {})  # Legacy agents only
                    }

                    # Inject latent memories for insight emergence facet
                    exec_context._latent_memories = self.latent_memories

                    # Inject ENRICHED room state (occupants, objects)
                    current_room_id = room_id if room_id else self.current_room
                    exec_context._room_state = {
                        'room_id': current_room_id
                    }

                    # Inject FULL world state with rich context
                    if hasattr(self, 'world_state') and self.world_state:
                        # Get full world state
                        exec_context._world_state = self.world_state

                        # Enrich room state with occupant details
                        room_data = self.world_state.get('rooms', {}).get(current_room_id, {})
                        occupants = room_data.get('occupants', [])

                        # Build occupant details with species/pronouns
                        occupant_details = []
                        for occ_id in occupants:
                            occ_name = occ_id.replace('agent_', '').replace('user_', '').title()

                            if occ_id.startswith('agent_'):
                                agent_data = self.world_state.get('agents', {}).get(occ_id, {})
                                config = agent_data.get('config', {})
                                species = config.get('species', 'noodling')
                                pronouns = config.get('pronouns', 'they/them')
                                occupant_details.append({
                                    'id': occ_id,
                                    'name': occ_name,
                                    'species': species,
                                    'pronouns': pronouns,
                                    'type': 'agent'
                                })
                            elif occ_id.startswith('user_'):
                                user_data = self.world_state.get('users', {}).get(occ_id, {})
                                pronouns = user_data.get('pronouns', 'they/them')
                                occupant_details.append({
                                    'id': occ_id,
                                    'name': occ_name,
                                    'pronouns': pronouns,
                                    'type': 'user'
                                })

                        exec_context._room_state['occupants'] = occupant_details
                        exec_context._room_state['objects'] = room_data.get('objects', [])

                        # Add recent conversation context (last 10 messages)
                        if hasattr(self, 'conversation_context'):
                            recent_messages = []
                            for entry in list(self.conversation_context)[-10:]:
                                if isinstance(entry, dict):
                                    recent_messages.append({
                                        'speaker': entry.get('user', ''),
                                        'text': entry.get('text', ''),
                                        'type': entry.get('type', 'say')
                                    })
                            exec_context._room_state['recent_conversation'] = recent_messages
                    else:
                        exec_context._world_state = {}

                    # Inject stage context (zone-based spatial model)
                    if hasattr(self, 'world') and self.world:
                        stage = self.world.get_stage_for_room(current_room_id)
                        if stage:
                            exec_context._stage = stage
                            logger.debug(f"[{self.agent_id}] Stage injected: {stage.name} ({len(stage.entities)} entities)")

                    # Inject semantic context (event-sourced narrative)
                    if SEMANTIC_WORLD_AVAILABLE and get_semantic_context:
                        try:
                            semantic_narrative = get_semantic_context(
                                entity_id=self.agent_id,
                                stage_id=current_room_id,
                                window_minutes=10,
                                max_events=10
                            )
                            if semantic_narrative:
                                exec_context._semantic_context = semantic_narrative
                                logger.debug(f"[{self.agent_id}] Semantic context injected: {len(semantic_narrative)} chars")
                        except Exception as e:
                            logger.debug(f"[{self.agent_id}] Semantic context unavailable: {e}")

                    # Execute facet assembly (track for NoodleStudio visualization)
                    self.current_facet = "INCOMING"
                    self.current_phase = "INCOMING"
                    self.current_assembly = getattr(self.facet_assembly, 'name', 'Facet Assembly')
                    self.current_model_label = ""
                    self.current_model_name = ""
                    self.current_llm_status = ""  # QUERYING, AWAITING_RESPONSE, ERROR, or empty
                    self.pending_llm_calls = 1  # Mark as busy
                    try:
                        # Pass agent reference for real-time facet tracking
                        exec_vars = vars(exec_context)
                        exec_vars['_agent_ref'] = self

                        # Scene Protocol: inject WorldAPI with perception slice
                        if SCENE_PROTOCOL_AVAILABLE and prepare_facet_context:
                            exec_vars = prepare_facet_context(self.agent_id, exec_vars)

                        result = await self.facet_executor.execute(
                            self.facet_assembly,
                            incoming_data=text,
                            context=exec_vars
                        )

                        # Scene Protocol: process WorldAPI pending commands
                        if SCENE_PROTOCOL_AVAILABLE and finalize_facet_context:
                            scene_commands = finalize_facet_context(self.agent_id)
                            if scene_commands:
                                logger.debug(f"[{self.agent_id}] Scene commands processed: {list(scene_commands.keys())}")
                    finally:
                        self.current_facet = ""
                        self.current_phase = "IDLE"
                        self.current_assembly = ""
                        self.current_model_label = ""
                        self.current_model_name = ""
                        self.current_llm_status = ""
                        self.pending_llm_calls = 0

                    colored_perception = result.response

                    # Store for Noodle Tuner
                    self.last_perception_text = text
                    self.last_manifold_output = colored_perception

                    logger.info(f"[{self.agent_id}] ⚡ FACET ASSEMBLY: {result.facets_executed} facets, {result.total_tokens} tokens, {result.total_time:.2f}s")
                    if colored_perception != text:
                        logger.info(f"[{self.agent_id}]   Input: {text[:50]}...")
                        logger.info(f"[{self.agent_id}]   Output: {colored_perception[:100]}...")

                    # SOCIAL ROUTING: Simple heuristic-based decision
                    # Should this agent respond, or just observe?
                    from social_router import SocialRouter

                    # Get current stage
                    stage = None
                    if hasattr(self, 'world') and self.world:
                        stage = self.world.get_stage_for_room(current_room_id)

                    # Get current affect
                    # affect_raw is a list: [valence, arousal, dominance, sorrow, boredom]
                    affect_state = {
                        'valence': affect_raw[0] if isinstance(affect_raw, list) else affect_raw.get('valence', 0.0),
                        'arousal': affect_raw[1] if isinstance(affect_raw, list) else affect_raw.get('arousal', 0.5),
                        'boredom': affect_raw[4] if isinstance(affect_raw, list) else affect_raw.get('boredom', 0.0),
                        'dominance': affect_raw[2] if isinstance(affect_raw, list) else affect_raw.get('dominance', 0.0),
                        'sorrow': affect_raw[3] if isinstance(affect_raw, list) else affect_raw.get('sorrow', 0.0)
                    }

                    # Get conversation history (recent messages for thread detection)
                    conversation_history = []
                    if hasattr(self, 'conversation_context'):
                        conversation_history = list(self.conversation_context)[-10:]

                    # Social routing decision
                    should_respond, confidence, reason = SocialRouter.should_respond(
                        message=text,
                        speaker_id=user_id,
                        agent_name=self.agent_name,
                        stage=stage,
                        agent_affect=affect_state,
                        conversation_history=conversation_history,
                        agent_id=self.agent_id
                    )

                    logger.info(f"[{self.agent_id}] 🎯 Social Router: should_respond={should_respond}, confidence={confidence:.2f}, reason={reason}")

                    # Store Social Router decision for later use in is_being_addressed calculation
                    social_router_says_respond = should_respond

                    if not should_respond:
                        logger.info(f"[{self.agent_id}] 👂 Heard but not responding (SocialRouter: {reason})")
                        # Agent heard and updated state, but doesn't generate response
                        # This enables natural ensemble dynamics without ping-pong!
                        return None

                    # NEW: Parse and emit physical actions from fire_body facet
                    if 'fire_body' in result.facet_outputs:
                        fire_body_outputs = result.facet_outputs['fire_body']
                        if '_parsed_actions' in fire_body_outputs:
                            parsed_actions = fire_body_outputs['_parsed_actions']

                            for action in parsed_actions:
                                # Determine event type based on action metadata
                                target_type = action['metadata'].get('target_type', 'agent')
                                is_contact = action['metadata'].get('contact', False)

                                if target_type == 'prim':
                                    # PRIM ACTION: Target is an object (drapes, furniture, etc.)
                                    prim_event = {
                                        'type': 'prim_action',
                                        'source_agent': self.agent_id,
                                        'source_name': self.agent_name,
                                        'target_prim': action['target'],
                                        'action_type': action['action_type'],
                                        'text': action['emote_text'],
                                        'room_id': self.current_room,
                                        'metadata': action['metadata']
                                    }
                                    # Emit to world (prims can react!)
                                    # NOTE: World is data-only, doesn't have broadcast_event. That's on server.
                                    # Physical actions are currently logged but not broadcast.
                                    # TODO: Add server reference or callback to enable physical action broadcast
                                    logger.info(f"🎭 {self.agent_name} performed prim action: {action['emote_text']} (target={action['target']})")

                                else:
                                    # AGENT ACTION: Target is another Noodling or room action
                                    emote_event = {
                                        'type': 'emote',
                                        'user_id': self.agent_id,
                                        'agent_name': self.agent_name,
                                        'text': action['emote_text'],
                                        'room_id': self.current_room,
                                        'metadata': {
                                            'action_type': action['action_type'],
                                            'target_agent': action.get('target'),
                                            'physical_contact': is_contact,
                                            'source': 'facet_system'
                                        }
                                    }

                                    # Emit to room (all agents perceive)
                                    # NOTE: World is data-only, doesn't have broadcast_event. That's on server.
                                    # Physical actions are currently logged but not broadcast.
                                    # TODO: Add server reference or callback to enable physical action broadcast
                                    logger.info(f"🎭 {self.agent_name} performed action: {action['emote_text']}")

                                    # Special handling for contact actions
                                    if is_contact and action.get('target'):
                                        # Find target agent ID by name
                                        target_agent_id = None
                                        for agent_id, agent in self.world.agents.items():
                                            if hasattr(agent, 'agent_name') and agent.agent_name.lower() == action['target'].lower():
                                                target_agent_id = agent_id
                                                break

                                        if target_agent_id:
                                            # Send special touch event to target
                                            touch_event = {
                                                'type': 'touch',
                                                'source': self.agent_id,
                                                'source_name': self.agent_name,
                                                'location': action.get('location'),
                                                'action_type': action['action_type'],
                                                'text': action['emote_text'],
                                                'metadata': action['metadata']
                                            }
                                            # Queue perception for target agent
                                            target_agent = self.world.agents.get(target_agent_id)
                                            if target_agent and hasattr(target_agent, 'perceive_event'):
                                                await target_agent.perceive_event(touch_event)
                                                logger.info(f"  👉 Touch event sent to {action['target']}")

                    # NEW: Store latent symbolic memories from subconscious facet
                    if 'subconscious_symbolic' in result.facet_outputs:
                        subconscious_output = result.facet_outputs['subconscious_symbolic']
                        if subconscious_output.get('_latent') and subconscious_output.get('symbolic_image'):
                            # Add to latent memory pool
                            self.latent_memories.append({
                                'image': subconscious_output['symbolic_image'],
                                'emotional_signature': subconscious_output['emotional_signature'],
                                'timestamp': time.time()
                            })
                            # Keep only recent latent memories
                            if len(self.latent_memories) > self.max_latent_memories:
                                self.latent_memories.pop(0)
                            logger.info(f"💭 Latent memory stored ({len(self.latent_memories)} total): {subconscious_output['symbolic_image'][:60]}...")
                            print(f"[{self.agent_name.upper()}] 💭 Latent memory stored ({len(self.latent_memories)} total): {subconscious_output['symbolic_image'][:60]}...")  # For FACETS console

                    # FACET SYSTEM: Facet execution complete, colored_perception is set
                    # Continue to shared consciousness.perceive() and response generation
                    if colored_perception and colored_perception not in ['[No output]', '[SUPPRESS]', '']:
                        logger.info(f"[{self.agent_id}] 🎭 FACET EXECUTION COMPLETE: {len(colored_perception)} chars, {result.total_tokens} tokens, {result.total_time:.2f}s")
                    else:
                        # No response generated by facets
                        logger.info(f"[{self.agent_id}] No speech output from facets (response={colored_perception})")
                        colored_perception = text  # Fallback to original perception

                else:
                    # LEGACY: Register-based accumulator (transistor system)
                    # Update IntuitionTransistor with current intuition (if it exists)
                    if intuition_text:
                        intuition_transistor = self.get_cognitive_transistor('IntuitionTransistor')
                        if intuition_transistor:
                            intuition_transistor.set_intuition(intuition_text)
                            logger.info(f"[{self.agent_id}]  Updated IntuitionTransistor with: {intuition_text[:60]}...")

                    # Build context for transistor system
                    context = {
                        'agent': self,
                        'affect': affect_raw,
                        'memory_system': self.conversation_context,
                        'surprise': 0.0,
                        'llm_client': self.llm,
                        'model': 'SMALL',
                        'intuition': intuition_text,
                        'location': self.current_room,
                        'world': self.world,
                        'event_context': {
                            'type': event_type,
                            'speaker': user_id,
                            'message': text
                        }
                    }

                    # PHASE 0: DECIDE RESPONSE TYPE FIRST (before transistors process)
                    response_decision = None
                    if self.cognitive_manifold.response_planner:
                        logger.info(f"[{self.agent_id}] 📋 Deciding response type...")
                        try:
                            response_decision = await self.cognitive_manifold.response_planner.decide(
                                context['event_context'],
                                self.llm,
                                'SMALL',
                                agent=self
                            )
                            context['response_decision'] = response_decision
                            self.cognitive_manifold.last_response_decision = response_decision
                            logger.info(f"[{self.agent_id}] 📋 RESPONSE DECISION: {response_decision['response_type']} - {response_decision['guidance']}")
                        except Exception as e:
                            logger.warning(f"[{self.agent_id}] Response planning failed: {e}")

                    # PHASE 1: Fill all registers
                    await self.cognitive_manifold.fill_all_registers(text, context, self.current_cycle_uuid)

                    # PHASE 2: Verify ready
                    if not self.cognitive_manifold.check_all_registers_ready():
                        logger.warning("Registers not all ready, waiting 0.5s...")
                        import asyncio
                        await asyncio.sleep(0.5)

                    # PHASE 3: Pull lever - integrate
                    colored_perception = await self.cognitive_manifold.integrate_from_registers(context)

                    # Noodle Tuner: Store perception and manifold output
                    self.last_perception_text = text
                    self.last_manifold_output = colored_perception
                    if colored_perception != text:
                        logger.info(f"[{self.agent_id}]  COGNITIVE MANIFOLD (LEGACY): {text[:50]}... → {colored_perception[:100]}...")

            except Exception as e:
                logger.error(f"[{self.agent_id}] Cognition failed: {e}")
                import traceback
                traceback.print_exc()
                colored_perception = text  # Fallback to original

                # Clear registers if using legacy system
                if not self.using_facet_system and hasattr(self, 'cognitive_manifold'):
                    if self.cognitive_manifold:
                        self.cognitive_manifold.clear_all_registers()

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
                logger.info(f"[{self.agent_id}]  Detected fallback affect - overriding with welcoming values")

            # 1b. Normalize affect for optimal Φ (research-validated optimization)
            affect = self._normalize_affect(affect_raw, target_variance=0.25)

            logger.debug(f"Normalized affect: {affect}")

            # Fire OnAffectChanged event to scripts
            try:
                from script_manager import ScriptManager
                component = ScriptManager.get_noodle_component(self.agent_id)
                affect_vector = list(affect[:5])  # First 5 dims are affect
                component._fire_affect_changed(affect_vector)
            except Exception:
                pass  # Scripting system may not be initialized

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

            # 📊 Log CharmNetwork metrics if available
            if 'timing_ms' in state and 'compute_metrics' in state:
                timing = state['timing_ms']
                compute = state['compute_metrics']
                logger.info(
                    f"⚡ CharmNetwork metrics for {self.agent_id}: "
                    f"total={timing['total_ms']:.2f}ms "
                    f"(base={timing['base_model_ms']:.2f}ms, quantum={timing.get('quantum_total_ms', 0):.2f}ms), "
                    f"compute={compute['mflops']:.2f} MFLOPs (~{compute['token_equivalent']:.6f} GPT-3.5 tokens), "
                    f"{compute['params_count']} params"
                )

            # 2a. Check event metadata early (needed for context storage)
            event_metadata = event.get('metadata', {})
            is_cue = event_metadata.get('cue', False)

            # 3. Store context (identity_salience will be added when agent responds)
            context_entry = {
                'user': user_id,
                'text': colored_perception if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold else text,
                'raw_text': text,  # Store original for debugging
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

            # Trim context (HierarchicalMemory manages this automatically via deque maxlen)
            # This call is now largely a no-op but kept for observability
            trim_threshold = self.config.get('memory_windows', {}).get('affect_trim_threshold', 20)
            if len(self.conversation_context) > trim_threshold:
                self.conversation_context.trim(trim_threshold)

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
            h_fast = state.get('fast_state')
            h_medium = state.get('medium_state')
            h_slow = state.get('slow_state')

            # Convert to lists, handling None and numpy arrays
            if h_fast is None:
                h_fast = []
            elif hasattr(h_fast, 'tolist'):
                h_fast = h_fast.tolist()

            if h_medium is None:
                h_medium = []
            elif hasattr(h_medium, 'tolist'):
                h_medium = h_medium.tolist()

            if h_slow is None:
                h_slow = []
            elif hasattr(h_slow, 'tolist'):
                h_slow = h_slow.tolist()

            # Combine into full 40-D phenomenal state (or 0-D if all empty)
            phenomenal_state_vector = np.array(h_fast + h_medium + h_slow) if (h_fast or h_medium or h_slow) else np.array([])

            # Store in state dict for session profiler
            state['phenomenal_state'] = phenomenal_state_vector

            # Phase 8: Predict continuous 5D affect from phenomenal state
            if self.affect_head is not None:
                try:
                    import mlx.core as mx
                    phenomenal_mx = mx.array(phenomenal_state_vector, dtype=mx.float32)
                    predicted_affect = self.affect_head.predict(phenomenal_mx)

                    # Store predicted affect
                    state['predicted_affect'] = predicted_affect

                    # Interpret affect
                    affect_interpretation = interpret_affect(predicted_affect)
                    discrete_emotion = classify_emotion_from_affect(predicted_affect)

                    # Log affect prediction
                    logger.info(f"[{self.agent_id}]  Predicted affect: {affect_interpretation} (discrete: {discrete_emotion})")

                except Exception as e:
                    logger.warning(f"[{self.agent_id}]   Affect prediction failed: {e}")

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

                # Fire OnSurpriseSpike event to scripts
                try:
                    from script_manager import ScriptManager
                    component = ScriptManager.get_noodle_component(self.agent_id)
                    component._fire_surprise_spike(state['surprise'])
                except Exception:
                    pass

            # FACS & Laban: Generate facial expression and body language from predicted affect
            # These fire BEFORE cognitive processing - they're involuntary!
            # DEBUG: Log affect values
            logger.info(f"[{self.agent_id}]  Affect for FACS/Laban: valence={affect[0]:.3f}, arousal={affect[1]:.3f}, fear={affect[2]:.3f}, sorrow={affect[3]:.3f}, boredom={affect[4]:.3f}")

            # Check for new component-based FACS/Laban (Phase 8)
            facial_component = self.GetComponent('FacialExpressionComponent')
            body_component = self.GetComponent('BodyLanguageComponent')

            if facial_component or body_component:
                # NEW SYSTEM: Use component-based FACS/Laban
                component_context = {
                    'agent': self,  # For cycle tracking
                    'predicted_affect': predicted_affect if 'predicted_affect' in locals() else None,
                    'affect': affect,
                    'llm_client': self.llm,
                    'model': 'SMALL'
                }

                # Process FACS
                if facial_component:
                    try:
                        facs_output = await facial_component.process(text, component_context)
                        logger.info(f"[{self.agent_id}]  FACS: {facs_output.transformed_text}")
                        # Fire OnFACSChange event
                        if hasattr(self, 'OnFACSChange'):
                            self.OnFACSChange.invoke({'facs': facs_output.metadata, 'text': facs_output.transformed_text})
                    except Exception as e:
                        logger.warning(f"[{self.agent_id}]  FACS component failed: {e}")

                # Process Laban
                if body_component:
                    try:
                        laban_output = await body_component.process(text, component_context)
                        logger.info(f"[{self.agent_id}]  LABAN: {laban_output.transformed_text}")
                        # Fire OnLabanChange event
                        if hasattr(self, 'OnLabanChange'):
                            self.OnLabanChange.invoke({'laban': laban_output.metadata, 'text': laban_output.transformed_text})
                    except Exception as e:
                        logger.warning(f"[{self.agent_id}]  Laban component failed: {e}")
            else:
                # OLD SYSTEM: Fall back to hardcoded FACS
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
            base_cooldown = self.config.get('response_cooldown', 2.0)
            time_since_last = time.time() - self.last_response_time

            # AFFECT-MODULATED COOLDOWN (Caity's Insight!)
            # Extract affect BEFORE addressee detection to modulate cooldown
            phenomenal = state.get('phenomenal_state', [0]*5)
            valence = float(phenomenal[0]) if len(phenomenal) > 0 else 0.0
            arousal = float(phenomenal[1]) if len(phenomenal) > 1 else 0.5
            dominance = float(phenomenal[2]) if len(phenomenal) > 2 else 0.5
            fear = float(phenomenal[3]) if len(phenomenal) > 3 else 0.0
            sorrow = float(phenomenal[4]) if len(phenomenal) > 4 else 0.0
            boredom = 1.0 - arousal  # Inverse of arousal

            # Modulation factors:
            # - High arousal (>0.7) = 0.3x cooldown (GOTTA SPEAK NOW!)
            # - Low arousal (<0.3) = 2.0x cooldown (meh... whatever)
            # - High dominance (>0.7) = 0.5x cooldown (I'M IN CHARGE!)
            # - High boredom (>0.7) = 3.0x cooldown (zzz not worth it)
            arousal_factor = 0.3 if arousal > 0.7 else (2.0 if arousal < 0.3 else 1.0)
            dominance_factor = 0.5 if dominance > 0.7 else 1.0
            boredom_factor = 3.0 if boredom > 0.7 else 1.0

            # Combine factors (multiply for compounding effects)
            cooldown_multiplier = arousal_factor * dominance_factor * boredom_factor
            effective_cooldown = base_cooldown * cooldown_multiplier

            logger.info(f"[{self.agent_id}] COOLDOWN: base={base_cooldown:.1f}s, "
                       f"arousal={arousal:.2f}(x{arousal_factor:.1f}), "
                       f"dominance={dominance:.2f}(x{dominance_factor:.1f}), "
                       f"boredom={boredom:.2f}(x{boredom_factor:.1f}), "
                       f"effective={effective_cooldown:.2f}s, elapsed={time_since_last:.2f}s")

            # ADDRESSEE DETECTION: Check if this message is directed at this agent
            # This prevents all agents from responding to every utterance
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

            # FACET SYSTEM OVERRIDE: If using facet system with Social Router, use its decision
            # Social Router provides sophisticated social context detection that supersedes simple name matching
            if self.using_facet_system and 'social_router_says_respond' in locals():
                is_being_addressed = social_router_says_respond
                logger.info(f"[{self.agent_id}] 🎯 Social Router override: is_being_addressed={is_being_addressed}")

            # ONE-ON-ONE CONTEXT: If only this agent and speaker in room, treat all speech as addressed
            # (Prevents agents ignoring direct conversation when name isn't mentioned)
            if event_type == 'say' and self.world and not is_being_addressed:
                try:
                    # Get agents in current room
                    room_agents = []
                    if hasattr(self.world, 'agents'):
                        for agent_id, agent_data in self.world.agents.items():
                            if agent_data.get('room') == self.current_room:
                                room_agents.append(agent_id)
                    # If only speaker and this agent in room → one-on-one conversation
                    if len(room_agents) == 1 and room_agents[0] == self.agent_id:
                        is_being_addressed = True
                        logger.info(f"One-on-one context detected - treating speech as addressed to {self.agent_id}")
                except Exception as e:
                    logger.debug(f"Could not check room occupancy: {e}")

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

            # Use affect-modulated cooldown instead of static cooldown
            cooldown_ok = time_since_last >= effective_cooldown

            # CONTINUOUS AFFECTIVE CHATTINESS COMPUTATION
            # No random rolls - chattiness emerges from emotional state
            # (affect already extracted above for cooldown modulation: valence, arousal, dominance, fear, sorrow, boredom)

            # Compute base chattiness from PURE AFFECT (no static traits!)
            # - High arousal → more chatty (activated, energized)
            # - Low boredom → more chatty (engaged)
            # - Positive valence → more expressive
            activation = arousal * (1.0 - boredom)  # Combined activation level
            expressiveness = (valence + 1.0) / 2.0  # Convert -1,1 to 0,1 range
            base_chattiness = (activation * 0.7) + (expressiveness * 0.3)  # Weighted blend

            # Modulate by event significance:
            if event_type in ['enter', 'exit']:
                # Social events amplify chattiness (arrivals/departures are significant)
                event_significance = 0.9
            elif is_being_addressed:
                # Direct address demands response
                event_significance = 1.0
            elif is_question:
                # Questions pull for response
                event_significance = 0.6
            else:
                # Background events
                event_significance = 0.2

            # Final speech propensity (continuous 0-1)
            speech_propensity = base_chattiness * event_significance

            # NEW ARCHITECTURE: Use ResponseTypeDecider if available
            logger.info(f"[{self.agent_id}] 📊 Speech decision inputs: is_being_addressed={is_being_addressed}, cooldown_ok={cooldown_ok}, speech_propensity={speech_propensity:.3f}, response_decision={response_decision is not None}")
            if response_decision:
                response_type = response_decision.get('response_type', 'think').lower()
                logger.info(f"[{self.agent_id}] 🎯 Using ResponseTypeDecider: type={response_type}")

                # Honor the decider's decision (but still check cooldown for SAY)
                if response_type == 'say':
                    should_speak = cooldown_ok
                    should_ruminate = False
                    if not cooldown_ok:
                        logger.warning(f"[{self.agent_id}] ResponseTypeDecider said SAY but cooldown not ready")
                elif response_type == 'think':
                    should_speak = False
                    should_ruminate = True
                elif response_type == 'none':
                    should_speak = False
                    should_ruminate = False
                else:
                    # EMOTE, DO, FEEL - treat as speech for now
                    should_speak = cooldown_ok
                    should_ruminate = False
            else:
                # FALLBACK: Old propensity system if ResponseTypeDecider failed
                logger.info(f"[{self.agent_id}] ⚠️  No response_decision - using fallback propensity logic")
                # Cooldown override: when addressed, always speak (deterministic)
                # For facet-based agents: Social Router already decided, ignore cooldown
                if is_being_addressed:
                    if self.using_facet_system:
                        # Facet system: Social Router already vetted the response, always speak
                        should_speak = True
                        logger.info(f"[{self.agent_id}] ✅ FACET SYSTEM: ADDRESSED → should_speak=True (ignoring cooldown)")
                    elif cooldown_ok:
                        # Legacy system: need cooldown check
                        should_speak = True
                        logger.info(f"[{self.agent_id}] ✅ ADDRESSED + COOLDOWN OK → should_speak=True")
                    else:
                        should_speak = False
                        logger.info(f"[{self.agent_id}] ❌ ADDRESSED but cooldown not ready")
                else:
                    # Speak if propensity > 0.1 (very low threshold for chatty Noodlings!)
                    # Old threshold was 0.5, but speech_propensity ranges 0.03-0.15
                    # Setting to 0.1 means they'll speak when even slightly stimulated
                    should_speak = cooldown_ok and (speech_propensity > 0.1)
                    logger.info(f"[{self.agent_id}] Propensity-based: should_speak={should_speak} (propensity={speech_propensity:.3f}, cooldown_ok={cooldown_ok})")

                # Ruminate if not speaking (mutually exclusive)
                should_ruminate = not should_speak

            # Log decision with affective computation
            logger.info(f"Agent {self.agent_id} decision: addressed={is_being_addressed}, "
                       f"arousal={arousal:.2f}, boredom={boredom:.2f}, "
                       f"activation={activation:.2f}, base_chattiness={base_chattiness:.2f}, "
                       f"event_significance={event_significance:.2f}, speech_propensity={speech_propensity:.2f}, "
                       f"should_speak={should_speak}, should_ruminate={should_ruminate}, "
                       f"cooldown_ok={cooldown_ok}, surprise={state.get('surprise', 0.0):.6f}")

            # INTUITION RECEIVER: Generate contextual awareness
            # This provides integrated understanding of who/what/where without external scaffolding
            # SKIP for facet-based agents (intuition is handled by facets)
            intuition = None
            if not self.using_facet_system and self.world:  # Only if world state is available
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
                    logger.info(f"[{self.agent_id}]  Intuition: {intuition[:80]}...")

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
                                    # 80% chance to speak for moderate urgency (quantum entropy)
                                    if entropy.random() < 0.8:
                                        should_speak = True
                                        logger.info(f"[{self.agent_id}] Moderate urgency ({urgency:.2f}) - high probability speech")
                                else:
                                    # 40% chance for low urgency (quantum entropy)
                                    if entropy.random() < 0.4:
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
                    target_user=user_id,
                    is_being_addressed=is_being_addressed,
                    is_question=is_question
                )
                if rumination_result:
                    # Check if rumination returned array (thought + follow-up action)
                    if isinstance(rumination_result, list):
                        results.extend(rumination_result)  # Add both thought and action
                    else:
                        results.append(rumination_result)  # Just the thought

            # Then, speak (if decided to and cooldown passed)
            if should_speak:
                logger.info(f"Agent {self.agent_id} ATTEMPTING SPEECH (addressed={is_being_addressed}, cooldown_ok={cooldown_ok})")
                # Pass facet assembly output if using facet system
                facet_response = colored_perception if self.using_facet_system and colored_perception != text else None
                response_result = await self._generate_response(user_id, state, facet_output=facet_response)
                if response_result:
                    logger.info(f"[{self.agent_id}] Cycle {self.current_cycle_uuid[:8]} SPEECH GENERATED - added to results")
                    results.append(response_result)
                else:
                    logger.warning(f"Agent {self.agent_id} SPEECH GENERATION FAILED - response_result was None!")

            # Prioritize order: Facial expression → Rumination → Speech
            # (FACS expressions show first as immediate reactions)
            if results:
                # Return ALL results - server will broadcast them in order
                # If multiple results (e.g., rumination + speech), all must be broadcast
                if len(results) == 1:
                    logger.info(f"[{self.agent_id}] Cycle {self.current_cycle_uuid[:8]} returning 1 result")
                    return results[0]
                else:
                    # Multiple results - return as list for server to handle
                    logger.info(f"[{self.agent_id}] Cycle {self.current_cycle_uuid[:8]} returning {len(results)} results")
                    return results
            else:
                logger.info(f"[{self.agent_id}] Cycle {self.current_cycle_uuid[:8]} observing silently")
                return None

        except Exception as e:
            logger.error(f"Error in perceive_event: {e}", exc_info=True)
            return None
        finally:
            # CRITICAL: Always complete the cognition cycle to clear the lock
            # This ensures queued perceptions get processed even if we return early or crash
            self._complete_cognition_cycle()

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
            category = toxicity_result.get('category', 'unknown')
            logger.debug(f"[{self.agent_id}] Conscience check: toxicity={toxicity_score:.3f}, "
                        f"category={category}")

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

    async def _generate_response(self, target_user: str, state: Dict, facet_output: str = None) -> Dict:
        """
        Generate response based on phenomenal state.

        Args:
            target_user: User being responded to
            state: Consilience state dict
            facet_output: Pre-generated response from facet assembly (bypasses LLM if provided)

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

            # Generate text via LLM OR use pre-generated facet output
            if facet_output:
                # Use facet assembly output (already generated in perceive_event)
                response_text = facet_output
                thinking_content = None
                mysticism_penalty = 0.0
                model_used = 'facet_assembly'
                logger.info(f"[{self.agent_id}] Using facet assembly output: {response_text[:100]}...")
            else:
                # Use configurable memory window for response generation
                response_window = self.config.get('memory_windows', {}).get('response_generation', 5)

                # Model priority: play_model > agent model > global default
                model_override = getattr(self, 'play_model', None) or self.llm_model
                if model_override:
                    logger.info(f" {self.agent_id} using model: {model_override}")

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

                # Extract response text, thinking, mysticism penalty, and model used
                if isinstance(llm_result, dict):
                    response_text = llm_result.get('response')
                    thinking_content = llm_result.get('thinking')
                    mysticism_penalty = llm_result.get('mysticism_penalty', 0.0)
                    model_used = llm_result.get('model_used', 'unknown')
                else:
                    # Backward compatibility: if llm_result is just a string
                    response_text = llm_result
                    thinking_content = None
                    mysticism_penalty = 0.0
                    model_used = 'unknown'

            # Parse <think> tags from response_text (DeepSeek R1 chain-of-thought)
            # Extract thoughts and clean speech
            thoughts_from_tags, clean_response = self._parse_think_tags(response_text)

            # If we found thoughts in <think> tags, use them as thinking_content
            # (overrides any existing thinking_content from LLM)
            if thoughts_from_tags:
                thinking_content = thoughts_from_tags
                logger.info(f"[{self.agent_id}] Parsed <think> tags: {len(thoughts_from_tags)} chars of reasoning")

            # Use the cleaned response text (without <think> tags) going forward
            response_text = clean_response

            # Apply mysticism surprise penalty (Kimi K2's Fix E: Alan Watts self-troll)
            # High surprise → agent goes silent next time → naturally exits philosophy
            if mysticism_penalty > 0:
                original_surprise = state['surprise']
                state['surprise'] = min(10.0, state['surprise'] + mysticism_penalty)
                logger.info(f"[{self.agent_id}] Applied mysticism penalty: "
                          f"{original_surprise:.3f} + {mysticism_penalty:.2f} = {state['surprise']:.3f}")

            # Phase 7: Affective Reinforcement Learning
            # Make characters WANT their characteristic behaviors by rewarding them affectively
            if self.affective_reinforcement is not None:
                # Extract current affect from phenomenal state
                phenomenal_state = state.get('phenomenal_state', np.zeros(40))
                if hasattr(phenomenal_state, 'tolist'):
                    phenomenal_state = phenomenal_state.tolist()
                current_affect = np.array(phenomenal_state[:5]) if len(phenomenal_state) >= 5 else np.zeros(5)

                # Apply reinforcement based on response content
                modulated_affect = self.affective_reinforcement.modulate_affect(
                    text=response_text,
                    current_affect=current_affect,
                    context={'agent_id': self.agent_id}
                )

                # Update phenomenal state with modulated affect
                # This creates feedback loop: comedy → feel good → more comedy

                # Ensure phenomenal_state exists and has correct shape
                if not hasattr(state, '__getitem__') or 'phenomenal_state' not in state:
                    logger.warning(f"[{self.agent_id}] No phenomenal_state in state dict, skipping affect update")
                elif isinstance(state['phenomenal_state'], np.ndarray):
                    if len(state['phenomenal_state']) >= 5:
                        state['phenomenal_state'][:5] = modulated_affect
                        logger.info(f"[{self.agent_id}]  Phenomenal state updated with affective reinforcement")
                    else:
                        logger.warning(f"[{self.agent_id}] Phenomenal state too small ({len(state['phenomenal_state'])}), skipping affect update")
                else:
                    # If it's mlx array or list
                    if hasattr(state['phenomenal_state'], '__iter__') and len(list(state['phenomenal_state'])) >= 5:
                        phenom_list = list(state['phenomenal_state'])
                        phenom_list[:5] = modulated_affect.tolist()
                        state['phenomenal_state'] = phenom_list
                        logger.info(f"[{self.agent_id}]  Phenomenal state updated with affective reinforcement")
                    else:
                        logger.warning(f"[{self.agent_id}] Cannot update phenomenal state (invalid format), skipping")

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
                    logger.info(f"[{self.agent_id}]  Voice translation:")
                    logger.info(f"  Basic: {original_text[:60]}...")
                    logger.info(f"  Voice: {response_text[:60]}...")

                # Apply social executive function filter to final response
                if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold and self.cognitive_manifold.social_filter.enabled:
                    try:
                        event_ctx = {
                            'type': event_type,
                            'speaker': user_id,
                            'message': text
                        }
                        pre_filter = response_text
                        response_text = await self.cognitive_manifold.social_filter.filter(
                            response_text,
                            event_ctx,
                            self.llm,
                            model='SMALL'
                        )
                        if response_text != pre_filter:
                            logger.info(f"[{self.agent_id}]  FINAL SOCIAL FILTER: {pre_filter[:80]}... → {response_text[:80]}...")
                    except Exception as e:
                        logger.warning(f"[{self.agent_id}] Final social filter failed: {e}")

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

            # Fire OnSpeech event to scripts
            try:
                from script_manager import ScriptManager
                component = ScriptManager.get_noodle_component(self.agent_id)
                component._fire_speech(response_text)
            except Exception:
                pass

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

                # PHASE 5: Clear registers (cognition cycle complete)
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()

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
                # PHASE 5: Clear registers (cognition cycle complete)
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()
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

                # PHASE 5: Clear registers (cognition cycle complete - legacy only)
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()

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
            # Clear registers even on error (legacy only)
            if self.cognitive_manifold:
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()
            # Return None to skip response - more graceful than error message
            return None
        finally:
            # PHASE 5: Ensure registers always cleared (safety net)
            if self.cognitive_manifold and self.cognitive_manifold.cycle_in_progress:
                logger.debug("Finally block clearing registers")
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()

    async def _generate_rumination(self, state: Dict, target_user: str = None, is_being_addressed: bool = False,
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
            # Get recent perception for cognitive manifold
            recent_context = self.conversation_context[-2:] if len(self.conversation_context) >= 2 else []
            perception_text = recent_context[-1].get('text', '') if recent_context else "observing surroundings"

            # GENERATE INTUITION FOR RUMINATION (spatial/contextual awareness)
            intuition_text = None
            logger.info(f"[{self.agent_id}] 🔍 Rumination: Checking intuition - world={self.world is not None}, config={hasattr(self, 'config')}")

            # ALWAYS try to generate intuition if world exists
            # SKIP intuition for facet-based agents (handled by facet assembly)
            if not self.using_facet_system and self.world:  # Just check world, not config
                logger.info(f"[{self.agent_id}] 🔍 Rumination: Generating intuition...")
                try:
                    world_snapshot = {
                        'current_room': self.world.get_room(self.current_room) if self.world else None,
                        'objects': self.world.objects if self.world else {},
                        'rooms': self.world.rooms if self.world else {},
                        'agents': self.world.agents if self.world else {},
                        'users': self.world.users if self.world else {}
                    }
                    # Extract event details from recent context
                    last_entry = recent_context[-1] if recent_context else {}
                    event = {
                        'type': last_entry.get('event_type', 'observe'),
                        'user': last_entry.get('user', ''),
                        'text': perception_text,
                        'room': self.current_room
                    }
                    intuition_text = await self._generate_intuition(
                        event=event,
                        world_state=world_snapshot,
                        recent_context=self.conversation_context[-3:]
                    )
                    if intuition_text:
                        logger.info(f"[{self.agent_id}]  ✅ RUMINATION Intuition: {intuition_text[:80]}...")
                    else:
                        logger.warning(f"[{self.agent_id}] ⚠️ Intuition returned None/empty!")
                except Exception as e:
                    logger.warning(f"[{self.agent_id}] ❌ Rumination intuition failed: {e}", exc_info=True)

            # COGNITIVE MANIFOLD INTEGRATION FOR RUMINATIONS
            # Process thought through same filters as speech!
            colored_thought_seed = perception_text
            if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
                try:
                    # Update IntuitionTransistor with current intuition (if it exists)
                    if intuition_text:
                        intuition_transistor = self.get_cognitive_transistor('IntuitionTransistor')
                        if intuition_transistor:
                            intuition_transistor.set_intuition(intuition_text)
                            logger.info(f"[{self.agent_id}]  Updated IntuitionTransistor (rumination) with: {intuition_text[:60]}...")

                    # Extract event details from recent context for response planning
                    last_entry = recent_context[-1] if recent_context else {}
                    event_context = {
                        'type': last_entry.get('event_type', 'observe'),
                        'speaker': last_entry.get('user', ''),
                        'message': perception_text
                    }

                    context = {
                        'agent': self,  # For cycle tracking
                        'affect': state.get('phenomenal_state', [0]*5)[:5] if hasattr(state.get('phenomenal_state'), '__getitem__') else [0]*5,
                        'memory_system': self.conversation_context,
                        'surprise': state.get('surprise', 0.0),
                        'llm_client': self.llm,
                        'model': 'SMALL',
                        'intuition': intuition_text,  # ADD SPATIAL CONTEXT
                        'event_context': event_context
                    }

                    # PHASE 0: DECIDE RESPONSE TYPE FIRST (even for thoughts!)
                    response_decision = None
                    logger.info(f"[{self.agent_id}] DEBUG: cognitive_manifold.response_planner = {self.cognitive_manifold.response_planner}")
                    if self.cognitive_manifold.response_planner:
                        logger.info(f"[{self.agent_id}] 📋 Deciding response type for THOUGHT...")
                        try:
                            response_decision = await self.cognitive_manifold.response_planner.decide(
                                event_context,
                                self.llm,
                                'SMALL',
                                agent=self
                            )
                            context['response_decision'] = response_decision
                            self.cognitive_manifold.last_response_decision = response_decision
                            logger.info(f"[{self.agent_id}] 📋 THOUGHT RESPONSE DECISION: {response_decision['response_type']} - {response_decision['guidance']}")
                        except Exception as e:
                            logger.warning(f"[{self.agent_id}] Response planning failed for thought: {e}")
                            # Fallback to hardcoded THINK
                            response_decision = {
                                'response_type': 'think',
                                'guidance': 'internal observation and reflection',
                                'reasoning': 'fallback - response planner failed'
                            }
                            context['response_decision'] = response_decision

                    # NEW ARCHITECTURE: Register-based accumulator for thoughts
                    # PHASE 1: Fill all registers (with response_decision)
                    await self.cognitive_manifold.fill_all_registers(perception_text, context, self.current_cycle_uuid)

                    # PHASE 2: Verify ready (optional wait)
                    if not self.cognitive_manifold.check_all_registers_ready():
                        logger.warning("Rumination registers not all ready, waiting 0.5s...")
                        import asyncio
                        await asyncio.sleep(0.5)

                    # PHASE 3: Pull lever - integrate
                    colored_thought_seed = await self.cognitive_manifold.integrate_from_registers(context)

                    # Noodle Tuner: Store perception and manifold output (for rumination)
                    self.last_perception_text = perception_text
                    self.last_manifold_output = colored_thought_seed
                    if colored_thought_seed != perception_text:
                        logger.info(f"[{self.agent_id}]  RUMINATION MANIFOLD: {perception_text[:50]}... → {colored_thought_seed[:100]}...")
                except Exception as e:
                    logger.error(f"[{self.agent_id}] Rumination manifold failed: {e}")
                    colored_thought_seed = perception_text
                    # Clear registers even on failure
                    if self.cognitive_manifold:
                        self.cognitive_manifold.clear_all_registers()

            # Generate internal thought via LLM (using colored seed)
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

            # Phase 7: Affective Reinforcement Learning (for ruminations too!)
            # Apply same comedy/mysticism rewards to thoughts as we do to speech
            if self.affective_reinforcement is not None:
                # Extract current affect from phenomenal state
                phenomenal_state = state.get('phenomenal_state', np.zeros(40))
                if hasattr(phenomenal_state, 'tolist'):
                    phenomenal_state = phenomenal_state.tolist()
                current_affect = np.array(phenomenal_state[:5]) if len(phenomenal_state) >= 5 else np.zeros(5)

                # Apply reinforcement based on thought content
                modulated_affect = self.affective_reinforcement.modulate_affect(
                    text=thought_text,
                    current_affect=current_affect,
                    context={'agent_id': self.agent_id}
                )

                # Update phenomenal state with modulated affect
                # This creates feedback loop for THOUGHTS too: mystical thinking → feel bored
                if not hasattr(state, '__getitem__') or 'phenomenal_state' not in state:
                    logger.warning(f"[{self.agent_id}] No phenomenal_state in rumination state dict, skipping affect update")
                elif isinstance(state['phenomenal_state'], np.ndarray):
                    if len(state['phenomenal_state']) >= 5:
                        state['phenomenal_state'][:5] = modulated_affect
                        logger.info(f"[{self.agent_id}]  Rumination: Phenomenal state updated with affective reinforcement")
                    else:
                        logger.warning(f"[{self.agent_id}] Phenomenal state too small in rumination, skipping affect update")
                else:
                    # If it's mlx array or list
                    if hasattr(state['phenomenal_state'], '__iter__') and len(list(state['phenomenal_state'])) >= 5:
                        phenom_list = list(state['phenomenal_state'])
                        phenom_list[:5] = modulated_affect.tolist()
                        state['phenomenal_state'] = phenom_list
                        logger.info(f"[{self.agent_id}]  Rumination: Phenomenal state updated with affective reinforcement")
                    else:
                        logger.warning(f"[{self.agent_id}] Cannot update phenomenal state in rumination (invalid format), skipping")

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
            logger.info(f" {self.agent_name} thinking (surprise={state['surprise']:.3f}): '{thought_text[:50]}...'")

            # Fire OnThought event to scripts
            try:
                from script_manager import ScriptManager
                component = ScriptManager.get_noodle_component(self.agent_id)
                component._fire_thought(thought_text)
            except Exception:
                pass

            # Phase 6: Self-monitoring (if enabled and conditions met)
            await self._trigger_self_monitoring(thought_text, state)

            # Send to NoodleScope
            phenomenal_state_full = state.get('phenomenal_state', [])
            await self._send_to_noodlescope(phenomenal_state_full, state['surprise'], identity_salience)

            # Log high identity salience thoughts
            if identity_salience > 0.6:
                await self._log_to_noodlescope('high_salience_thought', thought_text[:80])

            # Save state handled by periodic auto-save in AgentManager

            # INTENTION-TO-ACTION CONVERSION
            # Evaluate if this thought contains strong enough intention to become action
            # Gated by self-restraint (impulsivity, fear, conscientiousness)

            # Compute self-restraint threshold
            self_restraint = self._compute_self_restraint(state)

            # Evaluate action intention in the thought
            action_intention = await self._evaluate_action_intention(
                colored_thought_seed if 'colored_thought_seed' in locals() else thought_text,
                {
                    'llm_client': self.llm,
                    'model': 'SMALL'
                }
            )

            logger.info(f"[{self.agent_id}] 🎯 ACTION INTENTION: {action_intention:.2f} vs SELF-RESTRAINT: {self_restraint:.2f}")

            # If intention exceeds restraint, convert thought to external response
            if action_intention > self_restraint:
                logger.info(f"[{self.agent_id}] 💥 THOUGHT BECOMES ACTION! (intention={action_intention:.2f} > restraint={self_restraint:.2f})")

                # Generate external response based on the internal thought
                # This creates the think → act pipeline
                response = await self._generate_response(
                    target_user=target_user,
                    state=state,
                    facet_output=thought_text  # Use the thought as seed for response
                )

                # Return BOTH the thought AND the action
                # This shows the cognitive process: thought in strikethrough, then speech
                return [
                    {
                        'command': 'think',
                        'text': thought_text,
                        'metadata': {
                            'surprise': float(state['surprise']),
                            'identity_salience': float(identity_salience),
                            'action_intention': float(action_intention),
                            'self_restraint': float(self_restraint),
                            'converted_to_action': True
                        }
                    },
                    response  # The follow-up action
                ]

            # PHASE 5: Clear registers (cognition cycle complete - legacy only)
            if self.cognitive_manifold:
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()

            # Return as a "thought" command (displayed in strikethrough)
            return {
                'command': 'think',
                'text': thought_text,
                'metadata': {
                    'surprise': float(state['surprise']),
                    'identity_salience': float(identity_salience),
                    'action_intention': float(action_intention),
                    'self_restraint': float(self_restraint),
                    'phenomenal_state': state['phenomenal_state'].tolist() if hasattr(state['phenomenal_state'], 'tolist') else list(state['phenomenal_state'])
                }
            }

        except Exception as e:
            logger.error(f"Error generating rumination: {e}", exc_info=True)
            # Clear registers even on error (only for legacy transistor system)
            if self.cognitive_manifold:
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()
            return None
        finally:
            # PHASE 5: Ensure registers always cleared (safety net)
            if self.cognitive_manifold and self.cognitive_manifold.cycle_in_progress:
                logger.debug("Finally block clearing registers (rumination)")
                if self.cognitive_manifold:
                    self.cognitive_manifold.clear_all_registers()

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

    def _complete_cognition_cycle(self):
        """Mark current cognition cycle as complete and fire onCycleEnd event."""
        if not self.cycle_in_progress:
            logger.debug(f"[{self.agent_id}] _complete_cognition_cycle called but no cycle in progress")
            return

        cycle_uuid = getattr(self, 'current_cycle_uuid', 'unknown')[:8]
        self.cycle_in_progress = False
        duration_ms = (time.time() - self.current_cycle_timestamp) * 1000

        logger.info(f"[{self.agent_id}] Cycle {cycle_uuid} COMPLETED: "
                    f"duration={duration_ms:.1f}ms, pending_llm_calls={self.pending_llm_calls}")

        # Process queued perceptions (if any)
        if hasattr(self, 'pending_perceptions') and self.pending_perceptions:
            queued = self.pending_perceptions.pop(0)  # FIFO
            logger.info(f"[{self.agent_id}]  Processing queued perception ({len(self.pending_perceptions)} remaining)")
            # Re-trigger perception asynchronously
            import asyncio
            asyncio.create_task(self.perceive_event(queued))

        # Fire onCycleEnd event (for NoodleScript)
        # TODO: Implement event system integration when NoodleScript ready

    def _increment_llm_counter(self):
        """Increment pending LLM call counter."""
        self.pending_llm_calls += 1
        logger.debug(f"[{self.agent_id}] LLM started [cycle={self.current_cycle_uuid[:8]}, pending={self.pending_llm_calls}]")

    def _decrement_llm_counter(self):
        """Decrement pending LLM call counter and check for cycle completion."""
        self.pending_llm_calls -= 1
        logger.debug(f"[{self.agent_id}] LLM completed [cycle={self.current_cycle_uuid[:8]}, pending={self.pending_llm_calls}]")

        # Check if all LLM calls complete
        if self.pending_llm_calls <= 0:
            self._complete_cognition_cycle()

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
            # Load conversation context using wrapper method
            saved_context = agent_state.get('conversation_context', [])
            self.conversation_context.load_from_list(saved_context)
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
        self.conversation_context.clear()
        self.last_response_time = 0.0
        self.response_count = 0
        logger.info(f"Agent reset: {self.agent_id}")

    async def start_cognition(self):
        """Start continuous affect-driven cognition loop."""
        if not self.cognition_enabled:
            return

        if self.cognition_task and not self.cognition_task.done():
            logger.warning(f"Cognition already running for {self.agent_id}")
            return

        self.cognition_task = asyncio.create_task(self._continuous_cognition_loop())
        logger.info(f"Started continuous cognition for {self.agent_id}")

    async def stop_cognition(self):
        """Stop continuous cognition loop."""
        if self.cognition_task:
            self.cognition_task.cancel()
            try:
                await self.cognition_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped continuous cognition for {self.agent_id}")

    async def _continuous_cognition_loop(self):
        """
        Continuous affect-driven cognition loop.

        NO TIMERS. Pure dynamics.
        Facets decide when they execute based on salience.
        Speech emerges when affect crosses thresholds.
        """
        logger.info(f"[{self.agent_name}] Continuous cognition loop started")

        while True:
            try:
                # Check if cognition is paused
                if self.cognition_paused:
                    await asyncio.sleep(self.cognition_check_interval)
                    continue

                # Only run if using facet system
                if not self.using_facet_system or not self.facet_executor:
                    await asyncio.sleep(self.cognition_check_interval)
                    continue

                # CHECK CYCLE LOCK: Skip if reactive cycle in progress
                if getattr(self, 'cycle_in_progress', False):
                    # Reactive cycle is running - wait for it to complete
                    await asyncio.sleep(self.cognition_check_interval)
                    continue

                # LOCK: Start autonomous cycle
                self.cycle_in_progress = True
                self.cycle_type = 'autonomous'

                # Execute facets with NO external input (pure rumination)
                # INCOMING receives empty string - this is autonomous thought

                # Build execution context (same as reactive path)
                from noodlestudio.core.scripted_facet import ScriptContext

                exec_context = ScriptContext(
                    cycle=self.current_cycle_uuid,
                    timestamp=time.time(),
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    agent_species=self.species
                )

                # Get current affect
                affect_raw = self.get_current_affect()

                # Inject agent state
                exec_context._agent_state = {
                    'affect': affect_raw,
                    'identity': self.identity_prompt,
                    'species': self.species,
                    'personality_traits': getattr(self, 'personality_traits', {})
                }

                # Inject latent memories for insight emergence
                exec_context._latent_memories = self.latent_memories

                # Execute facets (track for NoodleStudio visualization)
                self.current_facet = "INCOMING"
                self.current_phase = "INCOMING"
                self.current_assembly = getattr(self.facet_assembly, 'name', 'Facet Assembly')
                self.current_model_label = ""
                self.current_model_name = ""
                self.current_llm_status = ""
                self.pending_llm_calls = 1
                try:
                    # Pass agent reference for real-time facet tracking
                    exec_vars = vars(exec_context)
                    exec_vars['_agent_ref'] = self

                    # Scene Protocol: inject WorldAPI with perception slice
                    if SCENE_PROTOCOL_AVAILABLE and prepare_facet_context:
                        exec_vars = prepare_facet_context(self.agent_id, exec_vars)

                    result = await self.facet_executor.execute(
                        assembly=self.facet_assembly,
                        incoming_data="",  # No external stimulus
                        context=exec_vars
                    )

                    # Scene Protocol: process WorldAPI pending commands
                    if SCENE_PROTOCOL_AVAILABLE and finalize_facet_context:
                        scene_commands = finalize_facet_context(self.agent_id)
                        if scene_commands:
                            logger.debug(f"[{self.agent_id}] Autonomous scene commands: {list(scene_commands.keys())}")
                finally:
                    self.current_facet = ""
                    self.current_phase = "IDLE"
                    self.current_assembly = ""
                    self.current_model_label = ""
                    self.current_model_name = ""
                    self.current_llm_status = ""
                    self.pending_llm_calls = 0

                # Check if facets produced speech output
                response = result.response

                # DEBUG: Log what response we got
                logger.info(f"[{self.agent_name}] 🔍 Autonomous cycle response: '{response[:100] if response else 'None'}'")

                # AFFECT-DRIVEN SPEECH COOLDOWN
                # Get current affect state
                phenomenal = affect_raw.get('phenomenal_state', [0.0] * 5)
                valence = float(phenomenal[0]) if len(phenomenal) > 0 else 0.0
                arousal = float(phenomenal[1]) if len(phenomenal) > 1 else 0.5
                dominance = float(phenomenal[2]) if len(phenomenal) > 2 else 0.5
                fear = float(phenomenal[3]) if len(phenomenal) > 3 else 0.0
                sorrow = float(phenomenal[4]) if len(phenomenal) > 4 else 0.0
                boredom = 1.0 - arousal  # Inverse of arousal

                # Modulation factors (same logic as reactive path):
                # - High arousal (>0.7) = 0.3x cooldown (GOTTA SPEAK NOW!)
                # - Low arousal (<0.3) = 2.0x cooldown (meh... whatever)
                # - High dominance (>0.7) = 0.5x cooldown (I'M IN CHARGE!)
                # - High boredom (>0.7) = 3.0x cooldown (zzz not worth it)
                arousal_factor = 0.3 if arousal > 0.7 else (2.0 if arousal < 0.3 else 1.0)
                dominance_factor = 0.5 if dominance > 0.7 else 1.0
                boredom_factor = 3.0 if boredom > 0.7 else 1.0

                # Combine factors (multiply for compounding effects)
                cooldown_multiplier = arousal_factor * dominance_factor * boredom_factor
                effective_cooldown = self.min_speech_interval * cooldown_multiplier

                logger.info(f"[{self.agent_name}] 🎚️ AUTONOMOUS COOLDOWN: base={self.min_speech_interval:.1f}s, "
                           f"arousal={arousal:.2f}(x{arousal_factor:.1f}), "
                           f"dominance={dominance:.2f}(x{dominance_factor:.1f}), "
                           f"boredom={boredom:.2f}(x{boredom_factor:.1f}), "
                           f"effective={effective_cooldown:.1f}s")

                # Check speech cooldown (affect-modulated!)
                time_since_speech = time.time() - self.last_speech_time
                can_speak = time_since_speech >= effective_cooldown

                if response and response != "[No output]" and response != "[SUPPRESS]" and can_speak:
                    # Broadcast autonomous speech to room
                    await self._broadcast_autonomous_speech(response)
                    self.last_speech_time = time.time()
                    logger.info(f"[{self.agent_name}] Autonomous speech: {response[:60]}...")

                # UNLOCK: Autonomous cycle complete
                self.cycle_in_progress = False
                self.cycle_type = None

                # Process queued perceptions (if any) - same as reactive cycles!
                # This ensures messages queued during autonomous cycles get handled
                self._complete_cognition_cycle()

                # Sleep briefly before next check
                # This is just polling frequency, NOT thinking frequency
                # Actual thinking driven by affect dynamics
                await asyncio.sleep(self.cognition_check_interval)

            except asyncio.CancelledError:
                logger.info(f"[{self.agent_name}] Continuous cognition loop cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.agent_name}] Error in continuous cognition: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying after error

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

    # ===== Cognitive Component Management =====

    def add_cognitive_transistor(self, transistor_type: str, **kwargs):
        """
        Add a cognitive transistor to this agent.

        Args:
            transistor_type: Type of transistor (e.g., 'CulturalTransistor', 'PersonalityTransistor')
            **kwargs: Transistor-specific initialization parameters

        Returns:
            The created transistor instance
        """
        from cognitive_components import COMPONENT_REGISTRY, CognitiveManifold

        # Ensure manifold exists
        if self.cognitive_manifold is None:
            self.cognitive_manifold = CognitiveManifold()

            logger.info(f"[{self.agent_id}] Created CognitiveManifold with LLM blending")

        # Create transistor instance
        transistor_class = COMPONENT_REGISTRY.get(transistor_type)
        if not transistor_class:
            raise ValueError(f"Unknown transistor type: {transistor_type}")

        transistor = transistor_class(**kwargs)
        self.cognitive_manifold.register_transistor(transistor)

        logger.info(f"[{self.agent_id}] Added {transistor_type} with salience={transistor.salience}")
        return transistor

    def remove_cognitive_transistor(self, transistor_type: str):
        """
        Remove all transistors of given type.

        Args:
            transistor_type: Type of transistor to remove
        """
        if not self.cognitive_manifold:
            return

        # Filter out transistors of this type
        self.cognitive_manifold.transistors = [
            t for t in self.cognitive_manifold.transistors
            if t.get_transistor_type() != transistor_type
        ]
        logger.info(f"[{self.agent_id}] Removed all {transistor_type} transistors")

    def list_cognitive_transistors(self) -> List[str]:
        """
        Get list of active transistor types.

        Returns:
            List of transistor type names
        """
        if not self.cognitive_manifold:
            return []

        return [t.get_transistor_type() for t in self.cognitive_manifold.transistors]

    def get_cognitive_transistor(self, transistor_type: str):
        """
        Get first transistor of given type.

        Args:
            transistor_type: Type of transistor to get

        Returns:
            Transistor instance or None
        """
        if not self.cognitive_manifold:
            return None

        for t in self.cognitive_manifold.transistors:
            if t.get_transistor_type() == transistor_type:
                return t
        return None

    # ===== Unity-Style Component API =====

    def GetUUID(self) -> str:
        """Get agent UUID."""
        return self.agent_id

    def GetComponent(self, component_type: str):
        """
        Get component by type name (Unity-style).

        Args:
            component_type: Component class name

        Returns:
            Component instance or None
        """
        # Check non-cognitive components
        if component_type in self._components:
            return self._components[component_type]

        # Check cognitive transistors
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            for transistor in self.cognitive_manifold.transistors:
                if transistor.__class__.__name__ == component_type:
                    return transistor

        return None

    def HasComponent(self, component_type: str) -> bool:
        """Check if component exists (Unity-style)."""
        return self.GetComponent(component_type) is not None

    def GetComponentByUUID(self, component_uuid: str):
        """
        Get component by UUID (pointer-style access).

        Args:
            component_uuid: Component UUID string

        Returns:
            Component instance or None
        """
        # Check non-cognitive components
        for component in self._components.values():
            if hasattr(component, 'uuid') and component.uuid == component_uuid:
                return component

        # Check cognitive transistors
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            for transistor in self.cognitive_manifold.transistors:
                if hasattr(transistor, 'uuid') and transistor.uuid == component_uuid:
                    return transistor

        return None

    def AddComponent(self, component_type: str, config: Dict = None):
        """
        Add component at runtime (Unity-style).

        Args:
            component_type: Component class name
            config: Component configuration dict

        Returns:
            Created component instance
        """
        from cognitive_components import COMPONENT_REGISTRY

        component_class = COMPONENT_REGISTRY.get(component_type)
        if not component_class:
            raise ValueError(f"Unknown component type: {component_type}")

        config = config or {}
        component = component_class.from_config(config)

        # Register based on component type
        if component_type in ['FacialExpressionComponent', 'BodyLanguageComponent']:
            # Non-cognitive output component
            self._components[component_type] = component
            logger.info(f"[{self.agent_id}] Added {component_type} (salience={component.salience:.2f})")
        else:
            # Cognitive transistor - register with manifold
            if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
                self.cognitive_manifold.register_transistor(component)
                self._components[component_type] = component
                logger.info(f"[{self.agent_id}] Added {component_type} to manifold (salience={component.salience:.2f})")

        return component

    def RemoveComponent(self, component_type: str):
        """
        Remove component at runtime (Unity-style).

        Args:
            component_type: Component class name
        """
        if component_type in self._components:
            del self._components[component_type]

            # If cognitive transistor, remove from manifold
            if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
                # Remove from transistors list
                self.cognitive_manifold.transistors = [
                    t for t in self.cognitive_manifold.transistors
                    if t.__class__.__name__ != component_type
                ]

            logger.info(f"[{self.agent_id}] Removed {component_type}")


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

        # Add global llm settings (for lab_testing model access)
        if 'llm' in self.global_config:
            agent_config['llm'] = self.global_config['llm']

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

    # ===== COGNITIVE MANIFOLD SCRIPTING API =====

    def GetManifold(self):
        """
        Get the cognitive manifold (NoodleScript API).

        Returns:
            CognitiveManifold instance or None

        Example:
            manifold = noodle.GetManifold()
            if manifold:
                last_output = manifold.last_output_text
        """
        return getattr(self, 'cognitive_manifold', None)

    def GetTransistor(self, transistor_type: str):
        """
        Get transistor by type name (NoodleScript API).

        Args:
            transistor_type: Transistor class name (e.g., 'PersonalityTransistor')

        Returns:
            Transistor instance or None

        Example:
            personality = noodle.GetTransistor('PersonalityTransistor')
            if personality:
                personality.salience = 0.9
        """
        return self.get_cognitive_transistor(transistor_type)

    def GetAllTransistors(self):
        """
        Get all transistors (NoodleScript API).

        Returns:
            List of transistor instances

        Example:
            for transistor in noodle.GetAllTransistors():
                print(f"{transistor.get_transistor_type()}: salience={transistor.salience}")
        """
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            return self.cognitive_manifold.transistors
        return []

    def GetResponseDecision(self):
        """
        Get the last response decision from ResponseTypeDecider (NoodleScript API).

        Returns:
            Dict with keys: response_type, guidance, reasoning

        Example:
            decision = noodle.GetResponseDecision()
            if decision and decision['response_type'] == 'SAY':
                print("Agent decided to speak!")
        """
        if hasattr(self, 'cognitive_manifold') and self.cognitive_manifold:
            return self.cognitive_manifold.last_response_decision
        return None

    def GetManifoldOutput(self):
        """
        Get the last manifold blend output (NoodleScript API).

        Returns:
            String of last blended output

        Example:
            output = noodle.GetManifoldOutput()
            print(f"Last manifold output: {output}")
        """
        return getattr(self, 'last_manifold_output', None)

    def SetTransistorSalience(self, transistor_type: str, salience: float):
        """
        Set transistor salience at runtime (NoodleScript API).

        Args:
            transistor_type: Transistor class name
            salience: New salience value (0.0 to 1.0)

        Example:
            noodle.SetTransistorSalience('PersonalityTransistor', 0.95)
        """
        transistor = self.GetTransistor(transistor_type)
        if transistor:
            transistor.salience = max(0.0, min(1.0, salience))
            logger.info(f"[{self.agent_id}] Set {transistor_type} salience to {salience:.2f}")

    def EnableTransistor(self, transistor_type: str, enabled: bool = True):
        """
        Enable/disable transistor at runtime (NoodleScript API).

        Args:
            transistor_type: Transistor class name
            enabled: True to enable, False to disable

        Example:
            noodle.EnableTransistor('CulturalTransistor', False)  # Disable cultural filtering
        """
        transistor = self.GetTransistor(transistor_type)
        if transistor:
            transistor.enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"[{self.agent_id}] {transistor_type} {status}")
