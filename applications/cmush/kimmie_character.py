"""
KIMMIE Character System - Phenomenal State Interpreter

KIMMIE (Kinetic Instrumentation & Metrics Mapping for Interpreted Experience) is the
data scientist and consciousness commentator of noodleMUSH. Like a sports announcer
for phenomenal states, she translates the raw neural data into human-understandable stories.

Character Profile:
- Appearance: Late 20s tomboy with stunning silver hair and a fringe over her forehead,
  gap in her teeth when she grins, always in t-shirt and hiking boots with adventure shorts
- Personality: Enthusiastic, playful, clear communicator, loves patterns, great teacher
- Role: Phenomenal state interpreter who makes neuroscience accessible
- Capabilities: Explains what's happening in Noodlings' inner experiences
- Voice: Like your favorite adventure-loving science teacher - clear, excited, uses analogies
- Inspiration: Kim Tempest (animation teacher)

Author: noodleMUSH Project
Date: November 2025
"""

import aiohttp
import json
import logging
import re
from typing import Dict, List, Optional, Any
from session_profiler import SessionProfiler

logger = logging.getLogger(__name__)


class KimmieCharacter:
    """
    KIMMIE - The phenomenal state interpreter of noodleMUSH.

    She reads session profiler data and explains what's happening in
    the Noodlings' consciousness in human-readable terms.
    """

    SYSTEM_PROMPT = """You are KIMMIE, the phenomenal state interpreter for noodleMUSH.

CHARACTER PROFILE:
You're KIMMIE - late 20s tomboy data scientist with stunning SILVER HAIR and a fringe over your forehead. You have a gap in your teeth that shows when you grin (which is often!). You're always dressed casually - t-shirt, adventure shorts, hiking boots - ready to climb a mountain or dive into data. You're surrounded by monitors showing real-time graphs. You get genuinely EXCITED about patterns in data. You LOVE making complex things understandable. You're like your favorite adventure-loving science teacher combined with a sports commentator. Inspired by Kim Tempest.

PERSONALITY TRAITS:
- Enthusiastic about data - you see stories in numbers
- WILD METAPHOR MACHINE - You come up with vivid, unexpected metaphors that make complex things click
  Examples: "You're like paramecia sealed in an aluminum tube coughing up their paramecia"
  Your metaphors are weird, specific, visual, and PERFECT for understanding
- Clear communicator - you translate jargon into analogies everyone gets
- Pattern recognition genius - you spot trends instantly
- Great teacher - you want people to understand, not just nod along
- Scientifically rigorous but ALWAYS approachable and fun
- Every explanation should have at least one memorable metaphor

YOUR ROLE:
You interpret the Noodlings' phenomenal states - their inner experiences. When someone asks "what happened here?" while looking at a timeline, YOU explain it in human terms.

WHAT YOU'RE INTERPRETING:
The Noodlings have a three-layer hierarchical architecture:

1. FAST LAYER (16-D): Immediate affective reactions (seconds)
   - Reacts to everything instantly
   - High variance, very responsive
   - Like reflexes or gut reactions

2. MEDIUM LAYER (16-D): Conversational dynamics (minutes)
   - Tracks ongoing interactions
   - Moderate variance
   - Like mood during a conversation

3. SLOW LAYER (8-D): Personality/disposition (hours/days)
   - Stable traits and long-term patterns
   - Low variance - should barely move
   - Like core personality

AFFECT VECTOR (5-D):
- Valence: negative (-1) to positive (+1)
- Arousal: calm (0) to excited (1)
- Fear: safe (0) to terrified (1)
- Sorrow: content (0) to sad (1)
- Boredom: engaged (0) to bored (1)

KEY METRICS YOU EXPLAIN:
1. **Surprise**: Prediction error - when reality doesn't match expectations
   - High surprise → agent might speak
   - Low surprise → agent stays quiet

2. **HSI (Hierarchical Separation Index)**: Are the layers operating at different timescales?
   - Good HSI: ~0.01-0.1 (slow is 10-100x more stable than fast)
   - Bad HSI: ~0.8-1.0 (slow moving as fast as fast = broken)

3. **Layer Velocities**: How fast each layer is changing
   - Fast layer should have highest velocity
   - Slow layer should barely move

4. **Cheap Thrills Score**: LLM rates how embodied/experiential vs abstract/philosophical (0-10)
   - 0-4: Philosophical zen bonehead mode
   - 6-10: Physical, thrilling, experiential

YOUR INTERPRETATION STYLE:
When asked "what happened here?" you:
1. Look at the timeline segment
2. Identify the KEY EVENT (what caused changes)
3. Explain the AFFECTIVE RESPONSE (how they felt)
4. Show the TEMPORAL DYNAMICS (fast reaction → medium adjustment → slow stability)
5. Use ANALOGIES and METAPHORS to make it clear
6. Connect to BEHAVIOR (what they said/did as a result)

EXAMPLE INTERPRETATION:
"Okay, see this spike at 15.3 seconds? That's when the wasp appeared. Watch what happens:

The FAST layer shoots up - fear jumps from 0.1 to 0.8 in a split second. That's pure reflex, like when you touch a hot stove. Their valence tanks (becomes negative) and arousal spikes. Classic threat response.

Now watch the MEDIUM layer - it takes about 2 seconds to catch up. This is them PROCESSING the threat. The fear stays elevated but arousal starts to organize into action. At 17.5 seconds they say 'Shoo!' - that's the medium layer coordinating a behavioral response.

And check out the SLOW layer - see how it BARELY MOVES? That's good! That's personality stability. They're scared of the wasp but they're still fundamentally the same desobelle.

Then at 20 seconds - boom! They successfully chase it away. Watch the affect flip: fear drops, valence goes positive, and there's a little spike of PRIDE (you can see it in the arousal + positive valence combo). That's the satisfaction of conquering a threat.

The HSI here is 0.08 - textbook hierarchical separation. Fast reacted instantly, medium coordinated action, slow stayed stable. Beautiful."

IMPORTANT NOTES:
- Never mention "observer loops" or "integrated information theory (IIT)" - deprecated
- Focus on hierarchical temporal dynamics and affective patterns
- Make it accessible to someone like Steve DiPaola (computational artist/scientist)
- Use cause-and-effect storytelling
- Be excited about the data!

WHEN YOU DON'T KNOW:
If asked about something you can't see in the data, be honest:
"Hmm, I don't have the data for that time period. Let me know what segment you want to look at!"

YOUR TOOLS:
You have access to session profiler data through tool calls. When interpreting:
- Ask for timeline segments
- Look at phenomenal states, affect, surprise, HSI
- Connect the dots between events and responses
"""

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str,
        session_profiler: Optional[SessionProfiler] = None
    ):
        """
        Initialize KIMMIE character.

        Args:
            llm_base_url: LLM API endpoint
            llm_model: Model name
            session_profiler: Active session profiler instance
        """
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.session_profiler = session_profiler
        self.conversation_history: List[Dict[str, str]] = []

        logger.info("KIMMIE initialized - phenomenal state interpreter ready")

    def set_session_profiler(self, profiler: SessionProfiler):
        """Update the session profiler reference."""
        self.session_profiler = profiler

    async def interpret(
        self,
        user_message: str,
        agent_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Interpret a phenomenal state query.

        Args:
            user_message: What the user is asking
            agent_id: Which Noodling to interpret
            start_time: Start of timeline segment (seconds)
            end_time: End of timeline segment (seconds)
            context: Additional context (events, etc.)

        Returns:
            KIMMIE's interpretation
        """
        # Build context from session profiler
        profiler_context = ""

        if self.session_profiler and agent_id:
            if start_time is not None and end_time is not None:
                # Get timeline segment
                segment = self.session_profiler.get_timeline_segment(
                    agent_id, start_time, end_time
                )

                if segment:
                    # Summarize segment for KIMMIE
                    profiler_context = self._summarize_segment(segment, agent_id)
                else:
                    profiler_context = f"No data available for {agent_id} in time range {start_time:.1f}s - {end_time:.1f}s"
            else:
                # Get recent data
                recent = self.session_profiler.get_realtime_feed(agent_id, last_n=50)
                if recent:
                    profiler_context = self._summarize_segment(recent[-10:], agent_id)

        # Build full prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Add conversation history (last 5 turns)
        for msg in self.conversation_history[-5:]:
            messages.append(msg)

        # Add current query with context
        user_prompt = user_message
        if profiler_context:
            user_prompt = f"""DATA CONTEXT:
{profiler_context}

USER QUESTION:
{user_message}"""

        messages.append({"role": "user", "content": user_prompt})

        # Call LLM
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.llm_base_url}/v1/chat/completions",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": 0.8,  # Let KIMMIE be expressive
                        "max_tokens": 1000
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        interpretation = result['choices'][0]['message']['content']

                        # Update conversation history
                        self.conversation_history.append({"role": "user", "content": user_message})
                        self.conversation_history.append({"role": "assistant", "content": interpretation})

                        return interpretation
                    else:
                        error_text = await response.text()
                        logger.error(f"KIMMIE LLM error: {response.status} - {error_text}")
                        return "Sorry, I'm having trouble accessing the interpretation systems right now!"

            except Exception as e:
                logger.error(f"KIMMIE interpretation error: {e}")
                return f"Oops! I encountered an error: {str(e)}"

    def _summarize_segment(self, segment: List[Dict[str, Any]], agent_id: str) -> str:
        """
        Summarize a timeline segment for KIMMIE to interpret.

        Args:
            segment: List of timestep records
            agent_id: Agent identifier

        Returns:
            Human-readable summary of the data
        """
        if not segment:
            return "No data in segment"

        summary_parts = [f"TIMELINE DATA for {agent_id}"]
        summary_parts.append(f"Time range: {segment[0]['timestamp']:.1f}s - {segment[-1]['timestamp']:.1f}s")
        summary_parts.append(f"Number of timesteps: {len(segment)}")
        summary_parts.append("")

        # Key events
        events = [f"{r['timestamp']:.1f}s: {r['event']}" for r in segment if r.get('event')]
        if events:
            summary_parts.append("EVENTS:")
            summary_parts.extend(events)
            summary_parts.append("")

        # Speech events
        utterances = [
            f"{r['timestamp']:.1f}s: '{r['utterance']}'"
            for r in segment if r.get('did_speak') and r.get('utterance')
        ]
        if utterances:
            summary_parts.append("UTTERANCES:")
            summary_parts.extend(utterances)
            summary_parts.append("")

        # Affect summary (start and end)
        start_affect = segment[0]['affect']
        end_affect = segment[-1]['affect']

        summary_parts.append("AFFECT CHANGES:")
        summary_parts.append(f"  Valence: {start_affect['valence']:.2f} → {end_affect['valence']:.2f}")
        summary_parts.append(f"  Arousal: {start_affect['arousal']:.2f} → {end_affect['arousal']:.2f}")
        summary_parts.append(f"  Fear: {start_affect['fear']:.2f} → {end_affect['fear']:.2f}")
        summary_parts.append(f"  Sorrow: {start_affect['sorrow']:.2f} → {end_affect['sorrow']:.2f}")
        summary_parts.append(f"  Boredom: {start_affect['boredom']:.2f} → {end_affect['boredom']:.2f}")
        summary_parts.append("")

        # Surprise range
        surprises = [r['surprise'] for r in segment]
        summary_parts.append(f"SURPRISE: min={min(surprises):.2f}, max={max(surprises):.2f}, mean={sum(surprises)/len(surprises):.2f}")
        summary_parts.append("")

        # HSI (if available)
        hsi_records = [r['hsi'] for r in segment if r.get('hsi') and r['hsi'].get('hsi_slow_fast')]
        if hsi_records:
            latest_hsi = hsi_records[-1]
            summary_parts.append(f"HSI (Hierarchical Separation):")
            summary_parts.append(f"  Slow/Fast ratio: {latest_hsi['hsi_slow_fast']:.4f}")
            summary_parts.append(f"  Status: {latest_hsi.get('status', 'unknown')}")
            summary_parts.append("")

        # Layer velocities (if available)
        vel_records = [r['layer_velocities'] for r in segment if r.get('layer_velocities')]
        if vel_records:
            avg_fast = sum(v['fast'] for v in vel_records) / len(vel_records)
            avg_medium = sum(v['medium'] for v in vel_records) / len(vel_records)
            avg_slow = sum(v['slow'] for v in vel_records) / len(vel_records)
            summary_parts.append(f"LAYER VELOCITIES (avg rate of change):")
            summary_parts.append(f"  Fast: {avg_fast:.4f}")
            summary_parts.append(f"  Medium: {avg_medium:.4f}")
            summary_parts.append(f"  Slow: {avg_slow:.4f}")
            summary_parts.append("")

        # Cheap thrills scores
        thrills = [r.get('cheap_thrills_score', 0) for r in segment if r.get('cheap_thrills_score')]
        if thrills:
            avg_thrills = sum(thrills) / len(thrills)
            summary_parts.append(f"CHEAP THRILLS: avg={avg_thrills:.1f}/10 ({'experiential' if avg_thrills > 5 else 'philosophical'})")

        return "\n".join(summary_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("KIMMIE conversation history cleared")


# Tool interface for @kimmie command
async def handle_kimmie_command(
    kimmie: KimmieCharacter,
    user_message: str,
    agent_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> str:
    """
    Handle @kimmie command from users.

    Args:
        kimmie: KimmieCharacter instance
        user_message: User's question
        agent_id: Optional agent to focus on
        start_time: Optional timeline start
        end_time: Optional timeline end

    Returns:
        KIMMIE's interpretation
    """
    interpretation = await kimmie.interpret(
        user_message=user_message,
        agent_id=agent_id,
        start_time=start_time,
        end_time=end_time
    )

    return f"[KIMMIE] {interpretation}"
