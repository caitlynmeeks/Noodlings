"""
Context Intelligence Facet - The GOD of Understanding WHO, WHAT, WHERE

Maintains persistent world model tracking:
- Entity states (who's where, doing what)
- Object locations (what's visible vs hidden)
- Relationship dynamics (trust, annoyance)
- Conversation threads (who asked what, who's waiting)
- Temporal state (positions persist across turns)

This is the CRITICAL reasoning layer that prevents context confusion.
Uses a smarter model because this is fundamental to all cognition.
"""

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

        # Extract model from config
        self.model = facet_config.get('model', 'qwen/qwen3-14b-2507')
        self.max_tokens = facet_config.get('max_tokens', 512)
        self.temperature = facet_config.get('temperature', 0.3)  # Lower temp for accuracy

    async def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming perception through context intelligence.

        Returns:
            - speaker: Who spoke
            - addressee: Who they're talking to (you/name/everyone)
            - speech_act: Type of communication
            - social_expectation: Urgency level (none/low/medium/high)
            - enriched_perception: Text with context made explicit
            - world_model_state: JSON of current world model
        """

        import logging
        logger = logging.getLogger(__name__)

        # Get agent name from context (more reliable than init-time)
        agent_name = context.get('agent_name', self.agent_name)

        logger.info(f"[ContextIntelligence] 🧠 EXECUTE CALLED for {agent_name}")
        logger.info(f"[ContextIntelligence]   inputs={list(inputs.keys())}")
        logger.info(f"[ContextIntelligence]   context={list(context.keys())}")
        print(f"[ContextIntelligence] 🧠 EXECUTE CALLED for {agent_name}")
        print(f"[ContextIntelligence]   inputs={list(inputs.keys())}")
        print(f"[ContextIntelligence]   context={list(context.keys())}")

        # Extract raw perception
        raw_perception = inputs.get('incoming_data', '')
        room_occupants = context.get('room_occupants', [])
        recent_messages = context.get('recent_messages', [])

        print(f"[ContextIntelligence]   raw_perception={raw_perception[:100] if raw_perception else 'NONE'}")

        # Build prompt for context analysis
        print(f"[ContextIntelligence] 🔨 Building prompt...")
        prompt = self._build_context_prompt(
            raw_perception=raw_perception,
            room_occupants=room_occupants,
            recent_messages=recent_messages,
            world_model_summary=self.world_model.get_context_summary(agent_name),
            agent_name=agent_name
        )
        print(f"[ContextIntelligence] 📝 Prompt built ({len(prompt)} chars)")

        # Call LLM for context reasoning
        print(f"[ContextIntelligence] 📞 Calling LLM ({self.model})...")
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            print(f"[ContextIntelligence] ✅ LLM responded ({len(response)} chars)")
        except Exception as e:
            print(f"[ContextIntelligence] ❌ LLM ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Parse structured output
        parsed = self._parse_context_response(response)

        # Update world model based on perception
        self._update_world_model(parsed, raw_perception, room_occupants)

        # Add world model state to output
        parsed['world_model_state'] = self._serialize_world_model()

        return parsed

    def _build_context_prompt(self, raw_perception: str, room_occupants: List[str],
                              recent_messages: List[str], world_model_summary: str, agent_name: str) -> str:
        """Build LLM prompt for context analysis."""

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

    def _parse_context_response(self, response: str) -> Dict[str, Any]:
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
                result['should_respond'] = self._calculate_response_need(result)
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
        fallback['should_respond'] = self._calculate_response_need(fallback)
        return fallback

    def _calculate_response_need(self, parsed: Dict[str, Any]) -> bool:
        """
        Clean routing logic: Should THIS agent respond to this message?

        Returns True if agent should generate a response, False if just observe.
        """
        addressee = parsed.get('addressee', 'unclear').lower()
        social_expectation = parsed.get('social_expectation', 'none')
        agent_name_lower = self.agent_name.lower()

        # Direct address → ALWAYS respond
        if addressee == agent_name_lower:
            return True

        # Everyone addressed + high urgency → respond
        if addressee == "everyone" and social_expectation in ["medium", "high"]:
            return True

        # Observable body language → don't respond (just observe)
        if addressee == "observable_to_all":
            return False

        # Everything else → don't respond (heard but not our conversation)
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
