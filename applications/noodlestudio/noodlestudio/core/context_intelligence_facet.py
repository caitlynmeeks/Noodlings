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
#   Context Intelligence Facet - The GOD of Understanding WHO, WHAT, WHERE
#
#   Maintains persistent world model tracking: - Entity state...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.context_intelligence_facet
# PURPOSE:  context intelligence facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EntityState, ConversationThread, WorldModel, ContextIntelligenceFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Any, Optional
import json
import re
from dataclasses import dataclass, field, asdict


@dataclass
class EntityState:
    """Tracks state of an entity (Noodling) in the world."""
    name: str
    location: str = "unknown"
    posture: str = "standing"
    holding: List[str] = field(default_factory=list)
    wearing: List[str] = field(default_factory=list)
    mood: str = "neutral"
    attention_on: Optional[str] = None  # Who they're focused on
    on_entity: Optional[str] = None  # Physical contact (e.g., "perched on Caity")
    physical_contact: List[str] = field(default_factory=list)  # Who's touching them

    # Attention focus tracking (Phase 1 - Natural Social Dynamics)
    attention_focus: str = "idle"  # "deep" | "moderate" | "idle"
    attention_target: Optional[str] = None  # What they're focused on


@dataclass
class ConversationThread:
    """Tracks an ongoing conversation thread."""
    speaker: str
    addressee: str  # "you", specific name, or "everyone"
    speech_act: str  # "question", "statement", "command", "emote"
    content: str
    expects_response: bool
    turns_ago: int


class WorldModel:
    """Persistent world state tracking."""

    def __init__(self):
        self.entities: Dict[str, EntityState] = {}
        self.hidden_objects: Dict[str, List[str]] = {}  # e.g., "caity.pocket": ["mouse"]
        self.conversation_threads: List[ConversationThread] = []
        self.social_dynamics: Dict[str, float] = {}  # e.g., "caity_trusts_red": 0.8

    def update_entity(self, name: str, **kwargs):
        """Update or create entity state."""
        if name not in self.entities:
            self.entities[name] = EntityState(name=name)

        entity = self.entities[name]
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

    def add_conversation_thread(self, speaker: str, addressee: str,
                               speech_act: str, content: str,
                               expects_response: bool = False):
        """Add new conversation thread."""
        # Age existing threads
        for thread in self.conversation_threads:
            thread.turns_ago += 1

        # Add new thread
        thread = ConversationThread(
            speaker=speaker,
            addressee=addressee,
            speech_act=speech_act,
            content=content,
            expects_response=expects_response,
            turns_ago=0
        )
        self.conversation_threads.append(thread)

        # Keep only recent 10 threads
        if len(self.conversation_threads) > 10:
            self.conversation_threads = self.conversation_threads[-10:]

    def get_context_summary(self, agent_name: str) -> str:
        """Generate human-readable context summary for agent."""
        lines = []

        # Entity positions
        if self.entities:
            lines.append("ENTITY STATES:")
            for name, state in self.entities.items():
                if name.lower() == agent_name.lower():
                    continue  # Don't describe self
                desc = f"  {name}: {state.posture}"
                if state.on_entity:
                    desc += f" (on {state.on_entity})"
                if state.attention_on:
                    desc += f", focused on {state.attention_on}"
                lines.append(desc)

        # Recent conversation threads
        recent = [t for t in self.conversation_threads if t.turns_ago < 3]
        if recent:
            lines.append("\nRECENT CONVERSATION:")
            for thread in recent:
                addr = thread.addressee if thread.addressee != "you" else agent_name
                lines.append(f"  {thread.speaker} → {addr}: {thread.speech_act}")
                if thread.expects_response and thread.addressee == "you":
                    lines.append(f"    ⚠️ Expecting YOUR response!")

        return "\n".join(lines) if lines else "No context available."


class ContextIntelligenceFacet:
    """
    The GOD of context understanding.

    Sits right after INCOMING, enriches raw perception with:
    - WHO is speaking
    - WHO they're addressing
    - WHAT type of speech act
    - SOCIAL expectation (response urgency)
    - Clarified text with explicit context
    """

    def __init__(self, facet_config: Dict[str, Any], llm_client, agent_name: str):
        self.config = facet_config
        self.llm_client = llm_client
        self.agent_name = agent_name
        self.world_model = WorldModel()

        # Extract model from config (use MEDIUM label by default)
        self.model = facet_config.get('model', 'MEDIUM')
        self.max_tokens = facet_config.get('max_tokens', 512)
        self.temperature = facet_config.get('temperature', 0.3)  # Lower temp for accuracy

    async def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming perception through context intelligence.

        NOW: Spatial narrator only (no social routing).
        Social routing moved to SocialRouter in agent_bridge.py.

        Returns:
            - enriched_perception: Narrative scene description with spatial context
            - world_model_state: JSON of current world model
        """

        import logging
        logger = logging.getLogger(__name__)

        # Get agent name from context (more reliable than init-time)
        agent_name = context.get('agent_name', self.agent_name)
        self.agent_name = agent_name

        logger.info(f"[ContextIntelligence] 🧠 EXECUTE CALLED for {agent_name}")
        logger.info(f"[ContextIntelligence]   inputs={list(inputs.keys())}")
        logger.info(f"[ContextIntelligence]   context={list(context.keys())}")

        # Extract raw perception
        raw_perception = inputs.get('incoming_data', '')

        # Get stage for spatial context (if available)
        stage = context.get('_stage')

        # Get semantic context (event-sourced narrative) if available
        semantic_context = context.get('_semantic_context', '')

        logger.info(f"[ContextIntelligence]   raw_perception={raw_perception[:100] if raw_perception else 'NONE'}")
        logger.info(f"[ContextIntelligence]   stage={'present' if stage else 'absent'}")
        logger.info(f"[ContextIntelligence]   semantic_context={'present (' + str(len(semantic_context)) + ' chars)' if semantic_context else 'absent'}")

        # Build spatial scene description
        spatial_context = ""
        if stage:
            from stage_model import StageQuery
            spatial_context = StageQuery.describe_scene(stage, agent_name)
            logger.info(f"[ContextIntelligence]   spatial_context: {spatial_context[:200]}")

        # Merge semantic context with spatial context (semantic provides narrative, spatial provides structure)
        if semantic_context:
            if spatial_context:
                spatial_context = f"{semantic_context}\n\n[SPATIAL STRUCTURE]\n{spatial_context}"
            else:
                spatial_context = semantic_context

        # Build prompt for spatial narration
        logger.info(f"[ContextIntelligence] 🔨 Building narrator prompt...")
        prompt = self._build_narrator_prompt(
            raw_perception=raw_perception,
            spatial_context=spatial_context,
            world_model_summary=self.world_model.get_context_summary(agent_name),
            agent_name=agent_name
        )
        logger.info(f"[ContextIntelligence] 📝 Prompt built ({len(prompt)} chars)")

        # Call LLM for spatial narration
        logger.info(f"[ContextIntelligence] 📞 Calling LLM ({self.model})...")

        # Track activity for ambient visualization
        from .model_activity_tracker import get_model_activity_tracker
        activity_tracker = get_model_activity_tracker()
        request_id = activity_tracker.request_started(self.model)

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Handle dict responses (some LLM clients return {text: ...})
            if isinstance(response, dict):
                response = response.get('text', response.get('content', ''))
            response = str(response)

            logger.info(f"[ContextIntelligence] ✅ LLM responded ({len(response)} chars)")
        except Exception as e:
            logger.error(f"[ContextIntelligence] ❌ LLM ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            activity_tracker.request_completed(self.model, request_id)

        # Output includes BOTH original perception AND spatial context
        # This preserves what the user said while adding spatial awareness
        enriched = raw_perception
        if response.strip():
            enriched = f"{raw_perception}\n[SPATIAL CONTEXT: {response.strip()}]"

        result = {
            'enriched_perception': enriched,
            'world_model_state': self._serialize_world_model()
        }

        logger.info(f"[ContextIntelligence] 🎯 RETURNING: {len(result['enriched_perception'])} chars")

        return result

    def _build_narrator_prompt(self, raw_perception: str, spatial_context: str,
                               world_model_summary: str, agent_name: str) -> str:
        """Build LLM prompt for spatial narration (narrator-only mode)."""

        prompt = f"""SPATIAL NARRATOR - Describe the scene from {agent_name}'s perspective

You are a spatial narrator describing what {agent_name} perceives. Focus on:
- WHO is present and WHERE they are
- WHAT objects/features are nearby
- Physical relationships (proximity, containment, touch)
- Movement and actions

DO NOT:
- Decide if {agent_name} should respond (that's handled elsewhere)
- Parse addressee or social dynamics
- Make assumptions about internal states

CURRENT SCENE:
{spatial_context}

WORLD MODEL MEMORY:
{world_model_summary}

NEW PERCEPTION:
{raw_perception}

Narrate what {agent_name} observes in 1-2 sentences. Focus on concrete, spatial details."""

        return prompt

    def _build_context_prompt(self, raw_perception: str, room_occupants: List[str],
                              recent_messages: List[str], world_model_summary: str, agent_name: str) -> str:
        """Build LLM prompt for context analysis (LEGACY - replaced by narrator mode)."""

        occupants_str = "\n".join([f"  - {occ}" for occ in room_occupants]) if room_occupants else "  (none)"
        messages_str = "\n".join([f"  {msg}" for msg in recent_messages[-5:]]) if recent_messages else "  (none)"

        prompt = f"""CONTEXT INTELLIGENCE - Understanding WHO, WHAT, WHERE

You are analyzing a message for {agent_name}.

RAW PERCEPTION:
{raw_perception}

ROOM OCCUPANTS:
{occupants_str}

RECENT CONVERSATION:
{messages_str}

CURRENT WORLD MODEL:
{world_model_summary}

YOUR TASK: Extract social context and clarify ambiguity.

CRITICAL - DISTINGUISH SPEECH FROM BODY LANGUAGE:
- SPEECH contains words/greetings/questions (examples: "Hello Red", "Hi everyone", "What's up?")
  → speech_act is "statement", "question", "command", "emote", or "action"
  → addressee determined by WHO is named or implied by room context
- BODY LANGUAGE is ONLY when perception starts with "[expression]" FACS codes
  → speech_act is "body_language"
  → addressee is "observable_to_all"

ADDRESSEE REASONING (CRITICAL):
1. If agent name explicitly mentioned → addressee is "{agent_name}"
2. If specific other name mentioned → addressee is that name
3. If "everyone" or similar → addressee is "everyone"
4. If ONLY 2 entities in room + greeting/statement with no explicit addressee → BY ELIMINATION addressee is the OTHER entity
   Example: Room has [Caity, {agent_name}], message "Hello" → addressee is "{agent_name}"
5. If 3+ entities in room + no explicit addressee → addressee is "everyone" (ambiguous)
6. If unclear who is addressed → addressee is "unclear"

SPECIAL CASE - BODY LANGUAGE EVENTS:
If the perception starts with "[expression]", this is FACS body language (facial/body movements).
- Extract WHO is expressing (from the text prefix like "Red Fire Anklebiter [expression]")
- Interpret WHAT the body language means (AU codes like AU1=Inner Brow Raise = surprise/concern)
- Translate to observable description: "[OBSERVATION] Red's eyes widen and flames flicker - appears startled"
- speech_act should be "body_language"
- addressee is "observable_to_all" (body language is visible to everyone present)

FACS AU CODES (Facial Action Units):
- AU1, AU2: Brow raise (surprise, concern, fear)
- AU4: Brow lower (anger, concentration)
- AU5: Upper lid raise (surprise, fear)
- AU6, AU12: Cheek raise, lip corner pull (joy, amusement)
- AU7: Lid tighten (anger, disgust)
- AU9, AU10: Nose wrinkle, upper lip raise (disgust)
- AU15, AU17: Lip corner depress, chin raise (sadness, contempt)
- AU23, AU24: Lip tighten, press (anger, stress)
- AU25, AU26, AU27: Lips part, jaw drop, mouth stretch (surprise, fear, shock)

BODY LANGUAGE CODES:
- BL1: Upright (confidence)
- BL10, BL22: Arms spread, jump (enthusiasm, joy)
- BL12, BL25, BL38: Freeze, still, head snap (startle, alertness)
- BL18, BL20, BL27: Fists clenched, step forward, stomp (anger, aggression)
- BL44, BL14: Waddle, nervous movement (anxiety)

Answer these questions:
1. WHO is speaking/expressing? (identify by name, or "unknown" if unclear)
2. WHO are they addressing? (Options: "{agent_name}", specific name, "everyone", "observable_to_all", "unclear")
3. WHAT is the speech act? (Options: "question", "statement", "command", "emote", "action", "body_language")
4. WHAT is the social expectation? (Options: "none", "low", "medium", "high")
5. Rewrite the text to make context EXPLICIT (add "[Speaker to Addressee]" or "[OBSERVATION]" prefix)

Output ONLY valid JSON in this exact format:
{{
  "speaker": "name or unknown",
  "addressee": "{agent_name} or name or everyone or observable_to_all or unclear",
  "speech_act": "question/statement/command/emote/action/body_language",
  "social_expectation": "none/low/medium/high",
  "enriched_perception": "text with explicit context"
}}"""

        return prompt

    def _parse_context_response(self, response: str, raw_perception: str = '') -> Dict[str, Any]:
        """Parse LLM response into structured output."""

        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                result = {
                    'speaker': parsed.get('speaker', 'unknown'),
                    'addressee': parsed.get('addressee', 'unclear'),
                    'speech_act': parsed.get('speech_act', 'statement'),
                    'social_expectation': parsed.get('social_expectation', 'none'),
                    'enriched_perception': parsed.get('enriched_perception', response)
                }
                # Calculate response decision
                result['should_respond'] = self._calculate_response_need(result, raw_perception)
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: return raw response as enriched perception
        fallback = {
            'speaker': 'unknown',
            'addressee': 'unclear',
            'speech_act': 'statement',
            'social_expectation': 'none',
            'enriched_perception': response
        }
        fallback['should_respond'] = self._calculate_response_need(fallback, raw_perception)
        return fallback

    def _calculate_response_need(self, parsed: Dict[str, Any], raw_perception: str = '') -> bool:
        """
        Clean routing logic: Should THIS agent respond to this message?

        Considers:
        - Direct address (always respond if urgent)
        - Attention focus (deep focus = oblivious, idle = curious)
        - Speech act type (emotes need idle attention)
        - FALLBACK: If addressee unclear but agent name mentioned, assume direct address

        Returns True if agent should generate a response, False if just observe.
        """
        import logging
        logger = logging.getLogger(__name__)

        addressee = parsed.get('addressee', 'unclear').lower()
        social_expectation = parsed.get('social_expectation', 'none')
        speech_act = parsed.get('speech_act', 'statement')
        agent_name_lower = self.agent_name.lower()

        logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need START")
        logger.info(f"[ContextIntelligence]   raw_perception: {raw_perception[:100]}")
        logger.info(f"[ContextIntelligence]   addressee: '{addressee}'")
        logger.info(f"[ContextIntelligence]   agent_name_lower: '{agent_name_lower}'")
        logger.info(f"[ContextIntelligence]   social_expectation: '{social_expectation}'")
        logger.info(f"[ContextIntelligence]   speech_act: '{speech_act}'")

        # Get agent's current attention state
        my_state = self.world_model.entities.get(agent_name_lower, None)
        focus_level = my_state.attention_focus if my_state else 'idle'  # Default idle

        # FALLBACK: If LLM failed to parse addressee but agent name is in the text, assume direct address
        logger.info(f"[ContextIntelligence] 🔍 Checking fallback name detection...")
        if addressee == 'unclear' and raw_perception:
            # Check for various forms of agent name
            perception_lower = raw_perception.lower()
            # Generate name variants for matching
            # Names can use spaces OR underscores, so handle both
            name_variants = [
                agent_name_lower,  # Full name as-is
                agent_name_lower.replace('_', ' '),  # Convert underscores to spaces
                agent_name_lower.replace(' ', '_'),  # Convert spaces to underscores
            ]
            # Add first-word variants (e.g., "red" from "red fire anklebiter")
            if '_' in agent_name_lower:
                name_variants.append(agent_name_lower.split('_')[0])
            if ' ' in agent_name_lower:
                name_variants.append(agent_name_lower.split(' ')[0])

            # Remove duplicates while preserving order
            seen = set()
            name_variants = [x for x in name_variants if not (x in seen or seen.add(x))]
            logger.info(f"[ContextIntelligence]   name_variants: {name_variants}")
            logger.info(f"[ContextIntelligence]   perception_lower: '{perception_lower}'")
            if any(variant in perception_lower for variant in name_variants):
                logger.info(f"[ContextIntelligence] ✅ FALLBACK TRIGGERED! Name detected in text")
                addressee = agent_name_lower  # Override to direct address
                social_expectation = 'medium'  # Assume medium urgency
            else:
                logger.info(f"[ContextIntelligence] ❌ FALLBACK FAILED - no name match")
        else:
            logger.info(f"[ContextIntelligence]   Fallback skipped (addressee={addressee})")

        # Direct address with high urgency → ALWAYS respond (breaks focus)
        if addressee == agent_name_lower and social_expectation == "high":
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: True (direct + urgent)")
            return True

        # Deep focus → ignore everything except urgent direct address
        if focus_level == "deep":
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: False (deep focus)")
            return False

        # Direct address (not urgent) → respond if moderate or idle focus
        if addressee == agent_name_lower:
            result = focus_level in ["moderate", "idle"]
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: {result} (direct address, focus={focus_level})")
            return result

        # Everyone addressed + high urgency → respond if not deep focus
        if addressee == "everyone" and social_expectation in ["medium", "high"]:
            result = focus_level != "deep"
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: {result} (everyone + urgent)")
            return result

        # Observable body language → don't respond (just observe)
        if addressee == "observable_to_all":
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: False (body language)")
            return False

        # Emotes/giggles → only respond if IDLE (curiosity!)
        if speech_act in ['emote', 'action']:
            # Idle + observable social event → brief curiosity
            result = focus_level == 'idle' and social_expectation != 'none'
            logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: {result} (emote/action)")
            return result

        # Everything else → don't respond (heard but not our conversation)
        logger.info(f"[ContextIntelligence] 🎯 _calculate_response_need RESULT: False (no match)")
        return False

    def _update_world_model(self, parsed: Dict[str, Any],
                           raw_perception: str, room_occupants: List[str]):
        """Update world model based on parsed context."""

        speaker = parsed.get('speaker', 'unknown')
        addressee = parsed.get('addressee', 'unclear')
        speech_act = parsed.get('speech_act', 'statement')

        # Update conversation thread
        if speaker != 'unknown':
            expects_response = (
                speech_act == 'question' or
                speech_act == 'command' or
                parsed.get('social_expectation') in ['medium', 'high']
            )

            self.world_model.add_conversation_thread(
                speaker=speaker,
                addressee=addressee,
                speech_act=speech_act,
                content=raw_perception[:100],  # First 100 chars
                expects_response=expects_response
            )

        # Update entity awareness (who's in room)
        for occupant in room_occupants:
            if occupant.lower() != self.agent_name.lower():
                self.world_model.update_entity(occupant, location='same_room')

    def _serialize_world_model(self) -> str:
        """Serialize world model to JSON string."""
        data = {
            'entities': {name: asdict(state) for name, state in self.world_model.entities.items()},
            'conversation_threads': [asdict(t) for t in self.world_model.conversation_threads],
            'social_dynamics': self.world_model.social_dynamics
        }
        return json.dumps(data, indent=2)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
