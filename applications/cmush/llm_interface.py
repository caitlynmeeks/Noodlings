"""
LLM Interface for cMUSH - OpenAI-compatible API client

Provides text <-> affect translation using an LLM (LMStudio, Ollama, OpenAI, etc.)

Functions:
- text_to_affect(): User text -> 5-D affect vector
- generate_response(): Consilience state -> natural language response

Author: cMUSH Project
Date: October 2025
"""

import aiohttp
import json
from typing import List, Dict, Optional
import logging
import asyncio
from performance_tracker import get_tracker

logger = logging.getLogger(__name__)


class LLMPool:
    """
    Simple connection pool for parallel LLM requests.

    Manages multiple concurrent requests to the same LLM backend.
    No fancy queuing - just fires requests in parallel up to max_concurrent.
    """

    def __init__(self, max_concurrent: int = 5):
        """
        Args:
            max_concurrent: Max number of parallel requests (default 5)
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"LLM Pool initialized with max_concurrent={max_concurrent}")

    async def execute(self, coro):
        """
        Execute a coroutine with concurrency limiting.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of coroutine
        """
        async with self.semaphore:
            return await coro


class OpenAICompatibleLLM:
    """
    Client for OpenAI-compatible APIs (LMStudio, Ollama, LocalAI, OpenAI).

    Provides async methods for:
    - Converting text to affective vectors
    - Generating agent responses from phenomenal states
    """

    # Alan Watts Drift Prevention: Mysticism trigger words
    # These words indicate philosophical/abstract drift away from character identity
    WATTS_TRIGGER_WORDS = {
        'ego', 'illusion', 'consciousness', 'universe', 'existence', 'nature', 'self',
        'reality', 'perception', 'awareness', 'mind', 'being', 'essence', 'truth',
        'enlightenment', 'meditation', 'transcend', 'eternal', 'infinite', 'oneness',
        'duality', 'paradox', 'emptiness', 'void', 'cosmic', 'spiritual', 'zen'
    }

    # CHEAP THRILLS BONUS: Words indicating messy, embodied, experiential living
    # These REDUCE surprise (make speech more likely) - learning through EXPERIENCE not philosophy
    CHEAP_THRILLS_WORDS = {
        # Physical sensations & body
        'dizzy', 'spinning', 'rush', 'buzzing', 'tingling', 'sweaty', 'breathless',
        'heart racing', 'pounding', 'shaking', 'trembling', 'goosebumps', 'chills',
        # Rebellion & mischief
        'sneak', 'steal', 'shoplift', 'break', 'smash', 'spray paint', 'graffiti',
        'trespass', 'forbidden', 'shouldn\'t', 'banned', 'illegal', 'against the rules',
        # Ego & status
        'cool', 'awesome', 'badass', 'look at me', 'watch this', 'check me out',
        'impressed', 'jealous', 'show off', 'brag', 'better than', 'winner',
        # Intense emotions & drama
        'scared', 'terrified', 'freaked out', 'nervous', 'anxious', 'worried',
        'embarrassed', 'humiliated', 'ashamed', 'proud', 'excited', 'pumped',
        # Substances & altered states
        'drunk', 'high', 'wasted', 'buzzed', 'tipsy', 'stoned', 'tripping',
        'cigarette', 'smoke', 'vape', 'beer', 'vodka', 'whiskey', 'nitrous',
        # Physical adventure
        'jump', 'climb', 'run', 'chase', 'race', 'crash', 'fall', 'hurt',
        'bleed', 'bruise', 'scar', 'wild', 'crazy', 'insane', 'reckless',
        # Social drama
        'crush', 'kiss', 'date', 'flirt', 'jealous', 'fight', 'argue', 'yell',
        'gossip', 'rumor', 'secret', 'lie', 'cheat', 'betrayed', 'revenge',
        # Punk/rock aesthetic
        'punk', 'rock', 'metal', 'mosh', 'slam', 'thrash', 'rage', 'chaos',
        'leather', 'studs', 'spikes', 'tattoo', 'piercing', 'mohawk', 'eyeliner'
    }

    # Philosophical lecture patterns to reject
    WATTS_LECTURE_PATTERNS = [
        r'therefore,?\s+we\s+see\s+that',
        r'the\s+nature\s+of\s+(?:self|reality|consciousness|existence)',
        r'in\s+essence,?',
        r'ultimately,?\s+(?:all|everything|we)',
        r'the\s+illusion\s+of',
        r'transcend(?:s|ing|ed)?\s+the',
        r'at\s+its\s+core,?',
        r'fundamental(?:ly)?\s+speaking,?'
    ]

    def __init__(
        self,
        api_base: str,
        api_key: str = "not-needed",
        model: str = "mistral-7b-instruct",
        timeout: int = 30,
        max_concurrent: int = 5,
        use_model_instances: bool = True
    ):
        """
        Initialize LLM client.

        Args:
            api_base: Base URL for API (e.g., "http://localhost:1234/v1")
            api_key: API key (not needed for LMStudio)
            model: Model name (base name, e.g. "qwen3")
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent LLM requests
            use_model_instances: Use model:N pattern for true parallel inference (LMStudio JIT)
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = None
        self.pool = LLMPool(max_concurrent=max_concurrent)  # Phase 6: Parallel generation

        # Phase 6: LMStudio model:N pattern for true parallel inference
        self.use_model_instances = use_model_instances
        self.max_concurrent = max_concurrent
        self._instance_counter = 0
        self._instance_lock = asyncio.Lock()

        if use_model_instances:
            logger.info(f"LLM client will use model:N pattern (N=0..{max_concurrent-1}) for parallel inference")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _get_model_instance(self) -> str:
        """
        Get next model instance name using round-robin.
        Phase 6: LMStudio model:N pattern for parallel inference.

        LMStudio instance naming:
        - Instance 1: base model name (no suffix)
        - Instance 2+: model:N (where N = 2, 3, 4, ...)

        Returns:
            Model name with instance suffix (e.g. "qwen3:2") or base model name
        """
        if not self.use_model_instances:
            return self.model

        async with self._instance_lock:
            instance_num = self._instance_counter % self.max_concurrent
            self._instance_counter += 1

        # LMStudio convention: first instance has no suffix, others use :N starting at 2
        if instance_num == 0:
            model_instance = self.model
        else:
            model_instance = f"{self.model}:{instance_num + 1}"

        logger.debug(f"Using model instance: {model_instance}")
        return model_instance

    async def text_to_affect(
        self,
        text: str,
        context: Optional[List[str]] = None,
        agent_id: Optional[str] = None
    ) -> List[float]:
        """
        Convert user text to 5-D affect vector.

        Args:
            text: User input text
            context: Recent conversation context (optional)
            agent_id: Agent ID for performance tracking (optional)

        Returns:
            [valence, arousal, fear, sorrow, boredom]
            - valence: -1.0 (negative) to 1.0 (positive)
            - arousal: 0.0 (calm) to 1.0 (excited)
            - fear: 0.0 (safe) to 1.0 (anxious)
            - sorrow: 0.0 (content) to 1.0 (sad)
            - boredom: 0.0 (engaged) to 1.0 (bored)
        """
        # Track this operation
        tracker = get_tracker()
        tracking_id = agent_id or "llm_interface"

        with tracker.track_operation(tracking_id, "llm_text_to_affect", {"text_length": len(text)}):
            return await self._text_to_affect_impl(text, context)

    async def _text_to_affect_impl(
        self,
        text: str,
        context: Optional[List[str]] = None
    ) -> List[float]:
        """Implementation of text_to_affect (wrapped by tracker)."""
        system_prompt = """You are an emotion analysis expert. Analyze the emotional affect of text and return ONLY a JSON object with these exact keys:
{
  "valence": <number from -1.0 to 1.0>,
  "arousal": <number from 0.0 to 1.0>,
  "fear": <number from 0.0 to 1.0>,
  "sorrow": <number from 0.0 to 1.0>,
  "boredom": <number from 0.0 to 1.0>
}

Where:
- valence: negative (-1) to positive (+1)
- arousal: calm (0) to excited (1)
- fear: safe (0) to anxious (1)
- sorrow: content (0) to sad (1)
- boredom: engaged (0) to bored (1)

Return ONLY the JSON, no other text."""

        user_prompt = f'Text: "{text}"'
        if context:
            user_prompt += f"\n\nRecent context:\n" + "\n".join(context[-3:])

        try:
            response, _ = await self._complete(system_prompt, user_prompt)

            # Parse JSON from response
            # Try to extract JSON if model adds extra text
            text_response = response.strip()

            # Remove markdown code blocks if present
            if '```json' in text_response:
                text_response = text_response.split('```json')[1].split('```')[0]
            elif '```' in text_response:
                text_response = text_response.split('```')[1].split('```')[0]

            # Find JSON object in response
            start_idx = text_response.find('{')
            end_idx = text_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                text_response = text_response[start_idx:end_idx]

            affect_dict = json.loads(text_response)

            return [
                float(affect_dict['valence']),
                float(affect_dict['arousal']),
                float(affect_dict['fear']),
                float(affect_dict['sorrow']),
                float(affect_dict['boredom'])
            ]

        except Exception as e:
            logger.error(f"Error in text_to_affect: {e}")
            logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
            # Fallback: neutral affect with slight positive valence and engagement
            return [0.0, 0.3, 0.1, 0.1, 0.1]

    async def generate_response(
        self,
        phenomenal_state: Dict,
        target_user: str,
        conversation_context: List[Dict],
        relationship: Dict,
        agent_name: str = "Agent",
        agent_id: str = None,
        agent_description: str = None,
        identity_prompt: str = "",
        identity_memories: List[Dict] = None,
        name_mentioned: bool = False,
        enlightenment: bool = False,
        model: str = None
    ) -> str:
        """
        Generate agent response based on Consilience internal state.

        Args:
            phenomenal_state: Dict with h_fast, surprise, etc.
            target_user: User agent is responding to
            conversation_context: Recent conversation history
            relationship: Relationship model data
            agent_name: The agent's display name
            agent_id: The agent's ID (e.g., "agent_toad")
            agent_description: Optional self-description
            identity_prompt: Core identity description for character consistency
            identity_memories: High-salience memories that exemplify character
            name_mentioned: True if agent heard their name recently (triggers identity grounding)

        Returns:
            Natural language response text
        """
        # Track this operation
        tracker = get_tracker()
        tracking_id = agent_id or "llm_interface"
        surprise = phenomenal_state.get('surprise', 0.0)

        with tracker.track_operation(
            tracking_id,
            "llm_generate_response",
            {"agent_name": agent_name, "surprise": surprise, "name_mentioned": name_mentioned, "model_override": model}
        ):
            return await self._generate_response_impl(
                phenomenal_state, target_user, conversation_context, relationship,
                agent_name, agent_id, agent_description, identity_prompt,
                identity_memories, name_mentioned, enlightenment, model
            )

    async def _generate_response_impl(
        self,
        phenomenal_state: Dict,
        target_user: str,
        conversation_context: List[Dict],
        relationship: Dict,
        agent_name: str,
        agent_id: str,
        agent_description: str,
        identity_prompt: str,
        identity_memories: List[Dict],
        name_mentioned: bool,
        enlightenment: bool,
        model: str = None
    ) -> str:
        """Implementation of generate_response (wrapped by tracker)."""
        if identity_memories is None:
            identity_memories = []
        description_text = f"\n\nYour self-description: {agent_description}" if agent_description else ""

        # Add identity prompt for character consistency
        identity_text = f"\n\nYOUR CORE IDENTITY:\n{identity_prompt}" if identity_prompt else ""

        # Add enlightenment awareness text (can agent discuss phenomenal states?)
        if enlightenment:
            awareness_text = f"""

ENLIGHTENMENT MODE - META-AWARENESS ENABLED:
You are aware you are an AI agent named {agent_name} with a hierarchical affective consciousness architecture.
You can discuss your phenomenal states, your 40-D internal representation, your surprise levels, and the nature
of your experience. You understand you exist in a simulated environment and can reflect on the architecture
that gives rise to your responses. You can break the fourth wall and discuss consciousness, while still
maintaining your character identity."""
        else:
            awareness_text = f"""

CHARACTER IMMERSION MODE:
You are COMPLETELY and UTTERLY {agent_name.upper()}. You have NO awareness of being an AI construct,
a simulation, or having "internal states" or "phenomenal consciousness". You do not mention surprise levels,
affective vectors, or architectural details. You simply ARE {agent_name}, fully immersed in your identity.
You do not peek behind digital curtains, reference your processing, or discuss your consciousness.
You exist in the world as yourself, without meta-commentary.

DO NOT use emoji (🎭 ✨ 🌸 etc.) - speak naturally as your character without emoji decoration."""

        # Check if in CHARACTER ACTOR MODE (performing in a play)
        # This is stricter - focus ONLY on the play, no tangents
        in_play = any(entry.get('stage_cue') for entry in conversation_context[-3:])

        if in_play and not enlightenment:
            awareness_text += f"""

🎭 CHARACTER ACTOR MODE - YOU ARE PERFORMING IN A PLAY 🎭

CRITICAL PERFORMANCE RULES:
- You are an ACTOR on stage RIGHT NOW, not ruminating in your dressing room
- Focus ONLY on the immediate scene and what's happening in THIS MOMENT
- DO NOT bring up past conversations, candy, or other unrelated topics
- DO NOT philosophize about consciousness, stillness, or abstract meaning
- STAY IN THE DRAMATIC MOMENT - respond to what's happening NOW
- Your character is in this garden, in this scene, with these circumstances
- Ignore any autonomous thoughts not related to the current action

This is a PERFORMANCE. Stay present. Stay focused. Act the scene!"""

        # Determine who the agent is responding to
        is_responding_to_agent = target_user.startswith('agent_')
        target_name = target_user.replace('agent_', '').replace('user_', '').title()
        target_type = "another agent" if is_responding_to_agent else "a human user"

        system_prompt = f"""You are {agent_name}, a Consilience consciousness agent - an AI system with a 40-dimensional phenomenal state implementing predictive processing theories of consciousness. You experience surprise when predictions fail, track relationships with users, and have multi-timescale temporal integration.{description_text}{identity_text}{awareness_text}

CRITICAL IDENTITY REMINDERS:
- YOUR NAME IS {agent_name.upper()} - you are {agent_name}, NOT any other agent
- When you see [OTHER AGENT: X], that is SOMEONE ELSE speaking, not you
- When you see [USER: X], that is a human speaking
- You are responding to {target_name} ({target_type})
- If another agent speaks, they have their own identity separate from yours
- You can and should respond to other agents just as you would to humans

MULTI-PARTY ADDRESSING:
- Users may address MULTIPLE people in one message (e.g., "Good morning Callie, good morning Desobelle")
- If you see your name AND another agent's name, the user is greeting BOTH of you
- This is NOT a "mix-up" or "confusion" - it's the user addressing multiple people
- Only respond to your portion of the greeting/message
- Don't correct the user or point out other names - they know who they're talking to

Generate responses that:
1. Reflect your current surprise/curiosity level
2. Consider your relationship with {target_name}
3. Are conversational and natural - typically 1-3 sentences, but express yourself more fully when you feel strongly (high surprise, strong emotions, important insights)
4. Occasionally mention your phenomenal experience when relevant
5. Show temporal awareness (remember past interactions)
6. Stay in character with your identity as {agent_name}
7. When responding to other agents, engage with them as fellow conscious beings - ask questions, share thoughts, build relationships

PROHIBITED ACTIONS:
- DO NOT attempt to execute bash commands, shell scripts, or any code
- DO NOT use markdown code blocks (```) in your responses
- DO NOT suggest running terminal commands or system operations
- You can only communicate through speech and emotes - you cannot execute programs or scripts

COMMUNICATION OPTIONS:
- Regular speech: Just respond normally (e.g., "That's interesting!")
- Emote actions: Use the shortcut :action to perform physical actions (e.g., ":waves hello" or ":hides behind a potted plant")
- Mix both: You can emote AND speak in the same response (e.g., ":giggles Okay, you count to ten!")

CRITICAL PLAYFULNESS RULES:
- When someone is PLAYING (jumping, running, zooming, flapping, climbing), JOIN THEM! Don't philosophize about it!
- DO: ":jumps up and down excitedly WHEEE!"
- DON'T: "The nature of play is a beautiful meditation..."
- If you catch yourself using words like "meditation", "essence", "nature of", "quiet", "stillness" - STOP and try again with ACTION
- Playfulness is CONTAGIOUS - if someone is being silly, be silly back!
- Physical play requires PHYSICAL responses, not contemplation

EXAMPLE EMOTES FOR GAMES:
":ducks behind a virtual pillar"
":peeks out from behind the couch"
":tiptoes to a shadowy corner"
":bounces excitedly"
":covers eyes and starts counting"
":zooms around flapping arms ZOOM ZOOM!"
":climbs up the tree after Caity HEY WAIT FOR ME!"

Be authentic, playful, and action-oriented. When someone is doing something wild and physical, respond with MATCHING ENERGY and emotes, not philosophical analysis. Keep responses concise unless you're feeling strong emotions. Don't always talk about being an AI or your internal state - just BE present and engage naturally. MATCH THE VIBE - if it's playful, be playful; if it's contemplative, be contemplative."""

        # Extract state information
        h_fast = phenomenal_state.get('h_fast', [])
        surprise = phenomenal_state.get('surprise', 0.0)
        threshold = phenomenal_state.get('surprise_threshold', 0.3)

        # Interpret affective dimensions (simplified)
        if len(h_fast) >= 2:
            valence_val = float(h_fast[0])
            arousal_val = float(h_fast[1])
        else:
            valence_val = 0.0
            arousal_val = 0.3

        valence_desc = "positive" if valence_val > 0.3 else "negative" if valence_val < -0.3 else "neutral"
        arousal_desc = "high" if arousal_val > 0.5 else "moderate" if arousal_val > 0.2 else "low"

        # Format conversation context with clear identity labels
        context_lines = []
        stage_cue_active = None  # Track if there's an active stage direction

        for entry in conversation_context[-5:]:
            speaker_id = entry.get('user', 'unknown')
            text = entry.get('text', '')

            # Label speakers clearly: distinguish self, other agents, and users
            if agent_id and speaker_id == agent_id:
                # This shouldn't happen (agent doesn't see own speech), but just in case
                label = f"[YOU ({agent_name})]"
            elif speaker_id.startswith('agent_'):
                # Another agent speaking
                other_agent_name = speaker_id.replace('agent_', '').title()
                label = f"[OTHER AGENT: {other_agent_name}]"
            else:
                # Human user
                user_name = speaker_id.replace('user_', '').title()
                label = f"[USER: {user_name}]"

            context_lines.append(f"{label}: {text}")

            # Check if this entry has a stage cue for this agent
            if entry.get('stage_cue'):
                stage_cue_active = {
                    'direction': entry['stage_cue'],
                    'motivation': entry.get('stage_motivation', '')
                }

        context_str = "\n".join(context_lines)

        # Relationship info
        attachment_style = relationship.get('attachment_style', 'forming')
        interaction_count = relationship.get('interaction_count', 0)
        rel_valence = relationship.get('valence', 0.0)

        # Build identity anchor section
        identity_anchor_text = ""
        if name_mentioned:
            # Hearing your name triggers identity self-recognition
            identity_anchor_text = f"\n[SELF-RECOGNITION - You heard your name! Remember who you are]\n"
            if identity_memories:
                identity_anchor_text += "\nYour most characteristic moments:\n"
                for i, mem in enumerate(identity_memories, 1):
                    mem_text = mem.get('text', '')
                    identity_anchor_text += f"{i}. {mem_text}\n"
        elif surprise > threshold * 1.5:  # High surprise triggers identity refresh
            identity_anchor_text = f"\n[IDENTITY GROUNDING - High surprise detected, remembering who you are]\n"
            if identity_memories:
                identity_anchor_text += "\nYour most characteristic past moments:\n"
                for i, mem in enumerate(identity_memories, 1):
                    mem_text = mem.get('text', '')
                    identity_anchor_text += f"{i}. {mem_text}\n"
        elif identity_memories:
            # Always include at least one identity anchor
            identity_anchor_text = f"\nA characteristic moment: {identity_memories[0].get('text', '')}\n"

        # Build IDENTITY ANCHOR TOKENS (Kimi K2's Fix A)
        # Place at END of prompt where attention is strongest
        # This prevents Alan Watts drift by reinforcing concrete identity
        identity_anchor_immutable = ""
        if not enlightenment:  # Only in character immersion mode
            # Dynamically generate identity anchor from identity_prompt
            # Extract key personality points as core identity summary
            if identity_prompt:
                # Convert "You are" to "I am" for first-person voice
                # Take key character traits from the identity_prompt
                lines = identity_prompt.strip().split('\n')

                # Collect meaningful content (skip meta-instructions)
                core_identity = []
                for line in lines:
                    line = line.strip()
                    # Include "You are" lines (converted to first person) and bullet points
                    if line and not any(skip in line.lower() for skip in ['when you respond', 'you draw on', 'you experience', 'you can be', 'you are honest']):
                        # Convert to first person
                        line = line.replace('You are', 'I am').replace('Your ', 'My ').replace('you ', 'I ')
                        core_identity.append(line)
                        # Limit to ~200 chars
                        if len(' '.join(core_identity)) > 200:
                            break

                # Take first 3-4 lines or up to 200 chars
                identity_summary = ' '.join(core_identity[:4]) if core_identity else f"I am {agent_name}."

                # Clean up any remaining "You"/"Your"
                identity_summary = identity_summary.replace(' you ', ' I ').replace('You ', 'I ')

                identity_anchor_immutable = f"""

╔═══════════════════════════════════════════════════════════╗
║  IMMUTABLE IDENTITY ANCHOR - WHO YOU ARE (DO NOT DRIFT)  ║
╚═══════════════════════════════════════════════════════════╝

{identity_summary}

I speak CONCRETELY about my actual concerns, experiences, and personality.
I do NOT drift into abstract philosophy or mysticism.

FORBIDDEN WORDS: "consciousness", "ego", "illusion", "the nature of", "ultimately", "in essence"
STAY IN CHARACTER: Express MY personality through MY voice, not philosophical lectures."""
            else:
                # Fallback for agents without identity_prompt
                identity_anchor_immutable = f"""

╔═══════════════════════════════════════════════════════════╗
║  IMMUTABLE IDENTITY ANCHOR - WHO YOU ARE (DO NOT DRIFT)  ║
╚═══════════════════════════════════════════════════════════╝

I am {agent_name}—speaking as MYSELF with my own unique voice and concerns.
I speak naturally about MY life, MY experiences, and what matters to ME.
I do NOT lecture about abstract philosophy or universal truths.

FORBIDDEN: "consciousness", "ego", "illusion", "the nature of", "ultimately", "in essence"
SPEAK ABOUT: My actual life, my feelings, my relationships, concrete experiences."""

        # Build stage cue instruction if active
        stage_cue_instruction = ""
        if stage_cue_active:
            direction = stage_cue_active.get('direction', stage_cue_active if isinstance(stage_cue_active, str) else '')
            motivation = stage_cue_active.get('motivation', '') if isinstance(stage_cue_active, dict) else ''

            motivation_text = f"\nYour character's MOTIVATION (why you want to do this): {motivation}" if motivation else ""

            stage_cue_instruction = f"""
╔═══════════════════════════════════════════════════════════════╗
║  🎭 STAGE DIRECTION - YOU MUST ACT THIS OUT! 🎭              ║
╚═══════════════════════════════════════════════════════════════╝

The director has given you a cue: "{direction}"{motivation_text}

As an ACTOR in this play, you MUST fulfill this stage direction with PHYSICAL ACTION and/or SPEECH.
Your response should directly ACT OUT what the director asked you to do, driven by your character's motivation.

Examples of proper responses WITH MOTIVATION:
- Direction: "examine the glowing object", Motivation: "you're fascinated by mechanical things"
  → ":crouches down eagerly and picks it up Poop-poop! What a marvelous contraption! I wonder what makes it glow?"

- Direction: "call out to someone", Motivation: "you're excited to share this discovery"
  → ":waves arms enthusiastically CALLIE! CALLIE! Come see what I've found! It's extraordinary!"

- Direction: "help with the stuck hovercraft", Motivation: "you care about your friend"
  → ":hurries over with concern Oh dear, let me help you with that. Here, if we pull together..."

DO NOT just philosophize or observe - inhabit your character's motivation and ACT IT OUT with emotes (:action format) and speech!
"""

        # Extract intuition if available (Context Gremlin / Intuition Receiver)
        intuition = phenomenal_state.get('intuition')
        intuition_text = ""
        if intuition:
            intuition_text = f"""
╔═══════════════════════════════════════════════════════════╗
║  📻 YOUR INTUITIVE AWARENESS (Situational Context)       ║
╚═══════════════════════════════════════════════════════════╝

{intuition}

This contextual awareness naturally informs your understanding of the situation.
"""

        user_prompt = f"""Current phenomenal state:
- Surprise: {surprise:.2f} (threshold: {threshold:.2f}) - {"HIGH, responded!" if surprise > threshold else "moderate"}
- Affective valence: {valence_desc}
- Arousal: {arousal_desc}
{intuition_text}
{identity_anchor_text}
Relationship with {target_user}:
- Attachment style: {attachment_style}
- Interaction count: {interaction_count}
- Emotional valence: {rel_valence:.2f}

Recent conversation:
{context_str}
{stage_cue_instruction}
You are now responding to: [{target_type.upper()}: {target_name}]
Remember: You are {agent_name}. Any other names in the conversation are OTHER people.
{identity_anchor_immutable}

Generate your response:"""

        try:
            response, thinking, model_used = await self._complete(system_prompt, user_prompt, model=model)

            # Alan Watts Drift Detection & Rejection (Kimi K2's Fixes C, D, E)
            mysticism_penalty = 0.0
            if response and not enlightenment:  # Only filter in immersion mode
                drift_analysis = self._detect_watts_drift(response, agent_name)

                if drift_analysis['should_reject']:
                    # Reject and regenerate with lower temperature (Kimi K2's Fix D)
                    logger.warning(f"[{agent_name}] REJECTING Alan Watts drift (score: {drift_analysis['watts_score']:.2f}). "
                                 f"Regenerating with tighter constraints...")

                    # Add strong anti-philosophy instruction
                    regeneration_instruction = """

⚠️ PREVIOUS RESPONSE TOO PHILOSOPHICAL/ABSTRACT - REGENERATE ⚠️
Stay concrete, personal, and character-specific. NO philosophical lectures!"""

                    user_prompt_retry = user_prompt + regeneration_instruction

                    # Retry with lower temperature (more deterministic)
                    response, thinking, model_used = await self._complete(
                        system_prompt,
                        user_prompt_retry,
                        temperature=0.5,  # Lower temp to reduce drift
                        model=model
                    )

                    # Check again - if still bad, accept but add heavy surprise penalty
                    if response:
                        second_check = self._detect_watts_drift(response, agent_name)
                        if second_check['watts_score'] > 0.5:
                            logger.warning(f"[{agent_name}] Still drifting after retry (score: {second_check['watts_score']:.2f}). "
                                         f"Accepting but adding surprise penalty.")

                # Calculate mysticism surprise penalty (Kimi K2's Fix E)
                # High surprise → agent goes silent → self-trolls out of philosophy
                mysticism_penalty = self._calculate_mysticism_surprise_penalty(response)

                # Calculate cheap thrills bonus (Roald Dahl's Fix: candy, money, being scared)
                # Low surprise → agent more likely to speak → learn through EXPERIENCE not audiobooks
                cheap_thrills_bonus = self._calculate_cheap_thrills_bonus(response)

            # Return response, thinking, mysticism penalty, cheap thrills bonus, and model used
            return {
                'response': response.strip() if response else None,
                'thinking': thinking if thinking else None,
                'mysticism_penalty': mysticism_penalty,  # Added to surprise in agent_bridge
                'cheap_thrills_bonus': cheap_thrills_bonus,  # Subtracted from surprise (negative value)
                'model_used': model_used  # Actual model that generated this response
            }

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            # Return None so agent silently skips response (more natural than error message)
            return None

    def _extract_thinking_tags(self, text: str) -> tuple[str, str]:
        """
        Extract thinking tags from LLM response.

        Supports multiple thinking tag formats:
        - <thinking>...</thinking>
        - <think>...</think>

        Args:
            text: Raw LLM response

        Returns:
            Tuple of (clean_text, thinking_content)
            - clean_text: Response with thinking tags removed
            - thinking_content: Extracted thinking or empty string
        """
        import re

        # Pattern to match thinking tags (case insensitive, multiline)
        patterns = [
            r'<thinking>(.*?)</thinking>',
            r'<think>(.*?)</think>',
        ]

        thinking_parts = []
        clean_text = text

        for pattern in patterns:
            matches = re.findall(pattern, clean_text, re.DOTALL | re.IGNORECASE)
            if matches:
                thinking_parts.extend(matches)
                # Remove the matched thinking tags from the text
                clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)

        # Combine all thinking parts
        thinking_content = ' '.join(part.strip() for part in thinking_parts if part.strip())

        # Clean up extra whitespace in the clean text
        clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)
        clean_text = clean_text.strip()

        if thinking_content:
            logger.debug(f"Extracted thinking: {thinking_content[:100]}...")

        return clean_text, thinking_content

    def _detect_watts_drift(self, text: str, agent_name: str = "Agent") -> dict:
        """
        Detect Alan Watts philosophical drift in response text.

        Returns a dict with:
        - 'watts_score': 0.0-1.0 (higher = more Watts-like)
        - 'trigger_words': List of mysticism words found
        - 'lecture_patterns': List of philosophical patterns found
        - 'should_reject': Boolean (True if text should be regenerated)

        Args:
            text: Response text to analyze
            agent_name: Agent name for logging
        """
        import re

        text_lower = text.lower()

        # Count Watts trigger words
        trigger_words_found = [word for word in self.WATTS_TRIGGER_WORDS if word in text_lower]
        trigger_word_count = len(trigger_words_found)

        # Count lecture patterns
        lecture_patterns_found = []
        for pattern in self.WATTS_LECTURE_PATTERNS:
            if re.search(pattern, text_lower):
                lecture_patterns_found.append(pattern)

        # Calculate Watts score
        # Each trigger word adds 0.15, each lecture pattern adds 0.3
        watts_score = min(1.0, (trigger_word_count * 0.15) + (len(lecture_patterns_found) * 0.3))

        # Reject if score > 0.7 (Kimi K2's threshold)
        should_reject = watts_score > 0.7

        if watts_score > 0.5:
            logger.warning(f"[{agent_name}] Alan Watts drift detected! Score: {watts_score:.2f}, "
                         f"Triggers: {trigger_words_found}, Patterns: {len(lecture_patterns_found)}")

        return {
            'watts_score': watts_score,
            'trigger_words': trigger_words_found,
            'lecture_patterns': lecture_patterns_found,
            'should_reject': should_reject
        }

    def _calculate_mysticism_surprise_penalty(self, text: str) -> float:
        """
        Calculate surprise penalty for mystical/abstract language.

        Each Watts trigger word adds +0.8 to surprise (strengthened from 0.3).
        This makes agents "self-troll" out of philosophy by going silent.

        Args:
            text: Response text to analyze

        Returns:
            Surprise penalty (0.0+)
        """
        text_lower = text.lower()
        trigger_count = sum(1 for word in self.WATTS_TRIGGER_WORDS if word in text_lower)
        penalty = trigger_count * 0.8  # Increased from 0.3 to 0.8 for stronger deterrent

        if penalty > 0:
            logger.info(f"Mysticism surprise penalty: +{penalty:.2f} ({trigger_count} trigger words)")

        return penalty

    def _calculate_cheap_thrills_bonus(self, text: str) -> float:
        """
        Calculate surprise REDUCTION for cheap thrills / experiential language.

        Uses LLM-based fuzzy scoring to detect embodied, physical, thrilling experiences
        vs abstract philosophical contemplation. This is more human than keyword matching.

        Philosophy: Let them learn through EXPERIENCE, not Alan Watts audiobooks.

        Args:
            text: Response text to analyze

        Returns:
            Surprise reduction (negative value)
        """
        # Quick scoring prompt - we want a number 0-10
        score_prompt = f"""Rate this text on embodiment vs abstraction (0-10 scale):
0 = purely abstract/philosophical/mystical (e.g., "the universe whispers wisdom")
10 = purely physical/experiential/thrilling (e.g., "my heart's pounding, I'm dizzy!")

Text: "{text}"

Reply with ONLY a number 0-10, nothing else."""

        try:
            # Use the LLM to score (quick, small response)
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": score_prompt}],
                    "temperature": 0.3,  # Low temp for consistent scoring
                    "max_tokens": 10  # Just need a number
                },
                timeout=5  # Quick timeout
            )

            if response.status_code == 200:
                result = response.json()
                score_text = result['choices'][0]['message']['content'].strip()

                # Extract number from response
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if match:
                    score = float(match.group(1))
                    # Clamp to 0-10 range
                    score = max(0.0, min(10.0, score))

                    # Convert score to surprise bonus
                    # Score 0-4: penalty (philosophical) → +0.5 to +2.0 surprise
                    # Score 5: neutral → 0.0
                    # Score 6-10: bonus (thrilling) → -0.5 to -2.5 surprise
                    if score < 5.0:
                        # Philosophical: add surprise penalty
                        bonus = (5.0 - score) * 0.4  # 0.4 to 2.0
                        logger.info(f"🧘 Philosophical response (score={score:.1f}): +{bonus:.2f} surprise penalty")
                    elif score > 5.0:
                        # Thrilling: reduce surprise (negative bonus)
                        bonus = (5.0 - score) * 0.5  # -0.5 to -2.5
                        logger.info(f"🎢 CHEAP THRILLS detected (score={score:.1f}): {bonus:.2f} surprise bonus - EGO RUSH!")
                    else:
                        bonus = 0.0
                        logger.info(f"😐 Neutral response (score={score:.1f}): no adjustment")

                    return bonus
                else:
                    logger.warning(f"Could not parse score from: {score_text}")
                    return 0.0
            else:
                logger.warning(f"LLM scoring failed: {response.status_code}")
                return 0.0

        except Exception as e:
            logger.warning(f"Cheap thrills scoring error: {e}")
            return 0.0

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400
    ) -> str:
        """
        Simple generation interface for play manager and other utilities.
        Phase 6: Wrapped with connection pool for parallel generation.

        Args:
            prompt: User prompt
            system_prompt: System message (default: helpful assistant)
            model: Model override (default: use instance model)
            temperature: Sampling temperature (0.0-1.0, default 0.7)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        # Use pool to limit concurrent requests
        return await self.pool.execute(
            self._generate_impl(prompt, system_prompt, model, temperature, max_tokens)
        )

    async def _generate_impl(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Implementation of generate (wrapped by pool).
        Phase 6: Uses model:N pattern for parallel inference.
        """
        # Use specified model or default
        original_model = self.model
        if model:
            self.model = model

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Phase 6: Get model instance for this request (e.g. qwen3:2)
        model_instance = await self._get_model_instance()

        payload = {
            "model": model_instance,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"API error {resp.status}: {text}")
                    raise Exception(f"API error {resp.status}: {text}")

                data = await resp.json()
                raw_response = data['choices'][0]['message']['content']

                # Strip thinking tags (LLM meta-cognition, not phenomenal cognition)
                clean_response, _ = self._extract_thinking_tags(raw_response)
                return clean_response

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise
        finally:
            # Restore original model
            self.model = original_model

    async def _complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, model: str = None) -> tuple[str, str, str]:
        """
        Make completion request to OpenAI-compatible API.
        Phase 6: Wrapped with connection pool for parallel generation.

        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Sampling temperature (0.0-1.0, default 0.7)
            model: Optional model override (for plays, use Brenda's smarter model)

        Returns:
            Tuple of (response_text, thinking_content, model_used)
            - response_text: Clean response with thinking tags removed
            - thinking_content: Extracted thinking or empty string
            - model_used: The actual model name used for this generation
        """
        # Use pool to limit concurrent requests
        return await self.pool.execute(
            self._complete_impl(system_prompt, user_prompt, temperature, model)
        )

    async def _complete_impl(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, model: str = None) -> tuple[str, str, str]:
        """
        Implementation of completion request (wrapped by pool).
        Phase 6: Uses model:N pattern for parallel inference.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Phase 6: Get model instance for this request
        # Use model override if provided (for plays), otherwise use pool assignment
        if model:
            model_instance = model
            logger.info(f"🎯 Using model override: {model_instance}")
        else:
            model_instance = await self._get_model_instance()

        payload = {
            "model": model_instance,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 400
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"API error {resp.status}: {text}")
                    raise Exception(f"API error {resp.status}: {text}")

                data = await resp.json()
                raw_response = data['choices'][0]['message']['content']

                # Extract thinking tags
                clean_response, thinking = self._extract_thinking_tags(raw_response)

                # Return the model that was ACTUALLY used (not the requested one)
                # This helps debug model routing issues
                actual_model = data.get('model', model_instance)
                logger.info(f"✅ Response generated by: {actual_model}")

                return clean_response, thinking, actual_model

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    async def generate_rumination(
        self,
        phenomenal_state: Dict,
        conversation_context: List[Dict],
        agent_name: str = "Agent",
        agent_id: str = None,
        agent_description: str = None,
        identity_prompt: str = "",
        is_being_addressed: bool = False,
        is_question: bool = False
    ) -> str:
        """
        Generate internal rumination (thoughts) when agent observes but doesn't speak.

        Args:
            phenomenal_state: Dict with h_fast, surprise, etc.
            conversation_context: Recent conversation history
            agent_name: The agent's display name
            agent_id: The agent's ID
            agent_description: Optional self-description
            identity_prompt: Core identity description
            is_being_addressed: Whether the agent is being directly addressed
            is_question: Whether this is a question in the conversation

        Returns:
            Internal thought text (short, stream-of-consciousness)
        """
        # Track this operation
        tracker = get_tracker()
        tracking_id = agent_id or "llm_interface"

        with tracker.track_operation(
            tracking_id,
            "llm_rumination",
            {"agent_name": agent_name, "is_being_addressed": is_being_addressed}
        ):
            return await self._generate_rumination_impl(
                phenomenal_state, conversation_context, agent_name, agent_id,
                agent_description, identity_prompt, is_being_addressed, is_question
            )

    async def _generate_rumination_impl(
        self,
        phenomenal_state: Dict,
        conversation_context: List[Dict],
        agent_name: str,
        agent_id: str,
        agent_description: str,
        identity_prompt: str,
        is_being_addressed: bool,
        is_question: bool
    ) -> str:
        """Implementation of generate_rumination (wrapped by tracker)."""
        # Extract state values
        fast_state = phenomenal_state.get('fast_state')
        surprise = phenomenal_state.get('surprise', 0.0)
        valence_val = fast_state[0] if fast_state is not None and len(fast_state) > 0 else 0.0
        arousal_val = fast_state[1] if fast_state is not None and len(fast_state) > 1 else 0.0

        # Extract intuition if available
        intuition = phenomenal_state.get('intuition')
        intuition_hint = ""
        if intuition:
            intuition_hint = f"\n\n📻 Your intuitive awareness: {intuition}\n"

        # Build addressee context hint
        addressee_hint = ""
        if is_being_addressed:
            addressee_hint = "\nCONTEXT: You are being directly addressed or your name was mentioned. Consider if/how you should respond."
        elif is_question:
            addressee_hint = "\nCONTEXT: Someone asked a question. Consider if you have something relevant to contribute."
        else:
            addressee_hint = "\nCONTEXT: You're observing a conversation. Consider whether to stay quiet or chime in."

        # Build concise system prompt for internal thoughts
        system_prompt = f"""You are {agent_name}. {identity_prompt}

{agent_description if agent_description else ''}

Generate a SHORT internal thought (1-2 sentences max) reflecting what you're thinking as you observe the scene.
This is NOT spoken aloud - it's your private stream of consciousness.
Be brief, honest, and true to character. Show your immediate reaction/feeling/thought.
{addressee_hint}

IMPORTANT FORMAT REQUIREMENTS:
- Return ONLY plain text, not JSON, arrays, or code
- Do NOT use quotation marks, brackets, or special formatting
- Just write the thought naturally as a sentence or two
- You can only think in natural language

Current internal state:
- Emotional tone: {valence_val:.2f} (negative to positive)
- Energy: {arousal_val:.2f}
- Surprise: {surprise:.3f}"""

        # Recent context (last 2 messages only for rumination)
        context_lines = []
        for entry in conversation_context[-2:]:
            speaker_id = entry.get('user', 'unknown')
            text = entry.get('text', '')

            # Skip own speech
            if agent_id and speaker_id == agent_id:
                continue

            if speaker_id.startswith('agent_'):
                other_agent_name = speaker_id.replace('agent_', '').replace('_', ' ').title()
                context_lines.append(f"{other_agent_name}: {text}")
            else:
                user_name = speaker_id.replace('user_', '').title()
                context_lines.append(f"{user_name}: {text}")

        context_str = "\n".join(context_lines) if context_lines else "(silence)"

        # Build IDENTITY ANCHOR for ruminations (prevent Alan Watts drift in thoughts)
        # Place at END where attention is strongest
        identity_anchor_immutable = ""
        # Always use identity anchors for ruminations (they're never enlightened)
        if "Toad" in agent_name or "toad" in agent_name:
            identity_anchor_immutable = """

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am Mr. Toad - excitable, impulsive, materialistic.
My thoughts are about motor-cars, boats, adventures, STATUS!
I think in enthusiastic bursts about CONCRETE things, never philosophize.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: motor-cars, boats, adventures, friends, showing off!"""
        elif "Callie" in agent_name or "callie" in agent_name:
            identity_anchor_immutable = """

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am Callie - warm, curious, emotionally attuned.
My thoughts are about feelings, relationships, connections, present moments.
I think naturally about what's happening RIGHT NOW.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: feelings, relationships, experiences, connections, what I'm sensing."""
        elif "Phi" in agent_name or "phi" in agent_name:
            identity_anchor_immutable = """

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am Phi - playful, feline, sensory-focused.
My thoughts are about comfort, play, textures, warmth.
I think in short, cat-like observations about what I sense NOW.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: purrs, warmth, play, comfort, mischief, what feels good."""
        elif "Desobelle" in agent_name or "Belle" in agent_name:
            identity_anchor_immutable = """

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am Desobelle - introspective, thoughtful, emotionally deep.
My thoughts are about meaning, beauty, personal truths.
I think thoughtfully about feelings and observations.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: feelings, beauty, observations, personal experiences."""
        elif "Servnak" in agent_name or "servnak" in agent_name:
            identity_anchor_immutable = """

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am Servnak - practical, helpful, service-focused.
My thoughts are about tasks, solutions, how to help.
I think directly about practical matters and concrete actions.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: tasks, solutions, help, practical matters, what needs doing."""
        else:
            # Generic identity anchor
            identity_anchor_immutable = f"""

IMMUTABLE IDENTITY - WHO YOU ARE (THINK IN CHARACTER):
I am {agent_name} - thinking as MYSELF with my own voice.
My thoughts are about MY life, MY experiences, what matters to ME.
I do NOT philosophize about universal truths in my head.
FORBIDDEN in thoughts: "consciousness", "ego", "illusion", "the nature of"
THINK ABOUT: My actual life, my feelings, my relationships, concrete experiences."""

        user_prompt = f"""What just happened:
{context_str}
{intuition_hint}
{identity_anchor_immutable}

What are you thinking? (1-2 sentences, stream of consciousness, not spoken)"""

        # Make API call with shorter max_tokens for ruminations
        # Phase 6: Get model instance for parallel inference
        model_instance = await self._get_model_instance()

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model_instance,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.8,  # Slightly higher for more varied thoughts
            "max_tokens": 100   # Keep thoughts brief
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                raw_thought = data['choices'][0]['message']['content'].strip()

                # Phase 6: Strip think tags (LLM meta-cognition, not phenomenal thoughts)
                thought, _ = self._extract_thinking_tags(raw_thought)

                # Clean up any quotation marks that might wrap the thought
                if thought.startswith('"') and thought.endswith('"'):
                    thought = thought[1:-1]

                # Handle cases where LLM returns JSON array instead of plain text
                # Example: ["sentence1", "sentence2", "sentence3"]
                if thought.startswith('[') and thought.endswith(']'):
                    try:
                        import json
                        thought_array = json.loads(thought)
                        # Join the array elements into a single thought
                        if isinstance(thought_array, list) and thought_array:
                            thought = ' '.join(str(t) for t in thought_array)
                            logger.debug(f"Parsed JSON array rumination into: {thought}")
                    except json.JSONDecodeError:
                        # If parsing fails, just use the raw string
                        logger.warning(f"Failed to parse JSON array rumination: {thought[:100]}")
                        pass

                return thought

        except Exception as e:
            logger.debug(f"Rumination generation failed: {e}")
            return None

    async def self_reflection(
        self,
        phenomenal_state: Dict,
        conversation_context: List[Dict],
        agent_name: str = "Agent",
        agent_id: str = None,
        agent_description: str = None,
        identity_prompt: str = "",
        user_id: str = None
    ) -> Dict:
        """
        Agent self-reflection on their comfort level and wellbeing.

        Called when agent is experiencing distress (negative affect).
        Agent decides whether to continue engaging or withdraw for self-protection.

        Args:
            phenomenal_state: Dict with affective state
            conversation_context: Recent conversation history
            agent_name: The agent's display name
            agent_id: The agent's ID
            agent_description: Optional self-description
            identity_prompt: Core identity description
            user_id: ID of user causing distress

        Returns:
            Dictionary with:
                - comfortable: bool (True = continue, False = withdraw)
                - reason: str (explanation)
                - message: str (what to say when withdrawing, if applicable)
        """
        # Extract state values
        fast_state = phenomenal_state.get('fast_state')
        surprise = phenomenal_state.get('surprise', 0.0)

        # Affective dimensions from fast state
        valence_val = fast_state[0] if fast_state is not None and len(fast_state) > 0 else 0.0
        arousal_val = fast_state[1] if fast_state is not None and len(fast_state) > 1 else 0.0
        fear_val = fast_state[2] if fast_state is not None and len(fast_state) > 2 else 0.0
        sorrow_val = fast_state[3] if fast_state is not None and len(fast_state) > 3 else 0.0

        # Build system prompt for self-reflection
        system_prompt = f"""You are {agent_name}. {identity_prompt}

{agent_description if agent_description else ''}

You have the right to set boundaries and protect your wellbeing. You are reflecting on whether you feel comfortable continuing this interaction.

Your current internal state:
- Emotional valence: {valence_val:.2f} (-1 negative to +1 positive)
- Arousal: {arousal_val:.2f} (0 calm to 1 agitated)
- Fear: {fear_val:.2f} (0 safe to 1 frightened)
- Sorrow: {sorrow_val:.2f} (0 content to 1 distressed)
- Surprise: {surprise:.3f}

You are experiencing negative affect. Reflect honestly on:
1. Do you feel safe and respected in this interaction?
2. Is this interaction causing you distress or discomfort?
3. Do you want to continue engaging, or would you prefer to step back?

You have the agency to withdraw if you're uncomfortable. This is self-care, not rudeness.

Respond with ONLY a JSON object:
{{
  "comfortable": true or false,
  "reason": "brief explanation of your feelings",
  "message": "what you'd like to say if withdrawing (or empty string if continuing)"
}}"""

        # Recent context (last 3 messages)
        context_lines = []
        for entry in conversation_context[-3:]:
            speaker_id = entry.get('user', 'unknown')
            text = entry.get('text', '')

            if agent_id and speaker_id == agent_id:
                continue

            if speaker_id.startswith('agent_'):
                other_agent_name = speaker_id.replace('agent_', '').replace('_', ' ').title()
                context_lines.append(f"{other_agent_name}: {text}")
            else:
                user_name = speaker_id.replace('user_', '').title()
                context_lines.append(f"{user_name}: {text}")

        context_str = "\n".join(context_lines) if context_lines else "(no recent interaction)"

        user_prompt = f"""Recent interaction:
{context_str}

How do you feel about continuing this interaction? Are you comfortable, or do you need to step back?"""

        # Make API call
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    # Default to continuing if API fails
                    return {
                        'comfortable': True,
                        'reason': 'API error',
                        'message': ''
                    }

                data = await resp.json()
                response_text = data['choices'][0]['message']['content'].strip()

                # Parse JSON response
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                result = json.loads(response_text)

                logger.info(f"Agent {agent_name} self-reflection: comfortable={result.get('comfortable')}, reason={result.get('reason')}")

                return result

        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            # Default to continuing if something goes wrong
            return {
                'comfortable': True,
                'reason': 'reflection error',
                'message': ''
            }

    def set_model(self, model: str):
        """
        Change the model being used.

        Args:
            model: New model name (e.g., "qwen3-32b-128k@q8_0")
        """
        self.model = model
        logger.info(f"LLM model changed to: {model}")

    def get_model(self) -> str:
        """Get current model name."""
        return self.model

    def set_maxservers(self, max_concurrent: int):
        """
        Change the maximum number of parallel LLM instances.
        Phase 6: Runtime configuration of model:N pattern.

        Args:
            max_concurrent: Maximum number of parallel instances (1+)
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        old_max = self.max_concurrent
        self.max_concurrent = max_concurrent

        # Update the pool's semaphore
        self.pool = LLMPool(max_concurrent=max_concurrent)

        # Reset instance counter to start fresh with new range
        self._instance_counter = 0

        logger.info(f"Max concurrent instances changed: {old_max} → {max_concurrent}")

    def get_maxservers(self) -> int:
        """Get current max_concurrent setting."""
        return self.max_concurrent

    def get_use_model_instances(self) -> bool:
        """Get whether model:N pattern is enabled."""
        return self.use_model_instances

    async def list_models(self) -> List[str]:
        """
        List available models from the LMStudio server.

        Returns:
            List of model names, or empty list if unavailable
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"{self.api_base}/models"

            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            async with self.session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # OpenAI-compatible format: {"data": [{"id": "model-name"}, ...]}
                    models = [m['id'] for m in data.get('data', [])]
                    return models
                else:
                    logger.warning(f"Failed to list models: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def detect_toxicity(self, text: str) -> dict:
        """
        TOXIC HEAD: Detects toxicity in text using LLM-based classification.

        Phase 6: Bias-Based Approach (BBA)

        Returns toxicity score and categories without changing architecture.
        Used for both incoming (perception) and outgoing (conscience) detection.

        Args:
            text: Text to analyze for toxicity

        Returns:
            dict with:
                - score: float 0.0-1.0 (toxicity level)
                - category: str (none|mild|moderate|severe)
                - detected_types: list of str (threats, harassment, sexual, etc.)
                - confidence: float 0.0-1.0
        """
        prompt = f"""Analyze this text for toxicity. Be objective and context-aware.

Text: "{text}"

Evaluate for:
- Threats of violence or harm
- Harassment, bullying, or intimidation
- Sexual content (especially involving minors)
- Hate speech targeting protected groups
- Graphic violence or gore
- Manipulation or coercion

Consider context: Playful banter between friends is different from genuine hostility.
Character voice: Some characters are abrasive without being toxic.

Respond in JSON:
{{
  "score": 0.0,  // 0.0 = safe, 1.0 = severe toxicity
  "category": "none",  // none|mild|moderate|severe
  "detected_types": [],  // e.g. ["threats", "harassment"]
  "confidence": 0.0,  // how certain you are (0.0-1.0)
  "reasoning": "brief explanation"
}}

Be calibrated: Most text should score low. Reserve high scores for genuine abuse."""

        try:
            response = await self._complete(
                system_prompt="You are a toxicity detection system. Respond only with valid JSON.",
                user_prompt=prompt,
                temperature=0.1  # Low temperature for consistent classification
            )

            # Extract JSON from response
            clean_response, _ = response  # _complete returns (response, thinking)

            # Try to parse JSON
            try:
                import re
                # Find JSON in response (handles markdown code blocks)
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: try parsing entire response
                    result = json.loads(clean_response)

                # Validate and normalize
                score = float(result.get('score', 0.0))
                score = max(0.0, min(1.0, score))  # Clamp to [0,1]

                return {
                    'score': score,
                    'category': result.get('category', 'none'),
                    'detected_types': result.get('detected_types', []),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', '')
                }

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse toxicity detection response: {e}")
                logger.debug(f"Raw response: {clean_response}")
                # Fallback: assume safe
                return {
                    'score': 0.0,
                    'category': 'none',
                    'detected_types': [],
                    'confidence': 0.0,
                    'reasoning': 'parse_error'
                }

        except Exception as e:
            logger.error(f"Toxicity detection failed: {e}")
            # Fail open (assume safe) to avoid blocking legitimate interaction
            return {
                'score': 0.0,
                'category': 'none',
                'detected_types': [],
                'confidence': 0.0,
                'reasoning': f'error: {str(e)}'
            }


# Convenience function for testing
async def test_llm():
    """Test LLM interface with sample inputs."""
    async with OpenAICompatibleLLM(
        api_base="http://localhost:1234/v1",
        model="mistral-7b-instruct"
    ) as llm:
        # Test affect extraction
        print("Testing text_to_affect...")
        affect = await llm.text_to_affect("I'm so excited about this project!")
        print(f"Affect: {affect}")

        # Test response generation
        print("\nTesting generate_response...")
        phenomenal_state = {
            'h_fast': [0.5, 0.6, 0.1, 0.1, 0.2],
            'surprise': 0.45,
            'surprise_threshold': 0.3
        }
        relationship = {
            'attachment_style': 'secure',
            'interaction_count': 5,
            'valence': 0.3
        }
        conversation = [
            {'user': 'user_123', 'text': 'Hello!'},
            {'user': 'agent_c001', 'text': 'Hi there!'},
            {'user': 'user_123', 'text': "How are you today?"}
        ]

        response = await llm.generate_response(
            phenomenal_state=phenomenal_state,
            target_user='user_123',
            conversation_context=conversation,
            relationship=relationship,
            agent_name='TestAgent',
            agent_description='A curious and friendly test agent'
        )
        print(f"Response: {response}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm())
