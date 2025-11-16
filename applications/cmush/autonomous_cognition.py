"""
Autonomous Cognition Engine - Independent agent thought and action

Gives agents:
- Background rumination loop (thinking between interactions)
- Spontaneous speech generation
- Internal monologue and memory consolidation
- Autonomous decision-making
- File system integration (reading/writing thoughts)
- Inbox processing

Author: cMUSH Project
Date: October 2025
"""

import asyncio
import time
import json
import random
import math
from typing import List, Dict, Optional
from datetime import datetime
import logging

from performance_tracker import get_tracker

logger = logging.getLogger(__name__)


class AutonomousCognitionEngine:
    """
    Manages agent's autonomous cognitive processes.

    Runs as background task, periodically "waking up" to:
    - Reflect on recent experiences
    - Consolidate memories
    - Generate internal thoughts
    - Decide on spontaneous actions
    - Process inbox/outbox
    """

    def __init__(self, agent, config: Dict):
        """
        Initialize autonomous cognition engine.

        Args:
            agent: CMUSHConsilienceAgent instance
            config: Configuration dict (includes personality traits)
        """
        self.agent = agent
        self.config = config

        # EVENT-DRIVEN TIMING (replaces fixed 45s timer!)
        # Surprise accumulation triggers thinking
        self.surprise_accumulation_threshold = config.get('surprise_threshold', 2.0)
        self.accumulated_surprise = 0.0

        # Dynamic thinking intervals (personality-modulated)
        self.min_think_interval = config.get('min_think_interval', 10)  # seconds
        self.max_think_interval = config.get('max_think_interval', 120)  # seconds

        # Speech timing (still want minimum intervals between speech)
        self.min_speech_interval = config.get('min_speech_interval', 120)  # 2 minutes

        # Cognitive state
        self.thought_buffer = []  # Recent internal thoughts
        self.cognitive_pressure = 0.0  # Builds up, triggers action
        self.last_speech_time = 0.0
        self.last_rumination_time = time.time()
        self.last_think_time = time.time()

        # Performance tracking
        self.tracker = get_tracker()

        # Boredom and social acknowledgment
        self.boredom = 0.0  # Accumulates during prolonged silence/lack of interaction
        self.last_interaction_time = time.time()  # Last time agent received stimulus
        self.directly_addressed = False  # Flag for social acknowledgment pressure

        # Speech urgency threshold (base - modified by personality)
        self.base_speech_threshold = config.get('speech_urgency_threshold', 0.7)

        # Personality traits (0.0-1.0 for each)
        self.personality = config.get('personality', {
            'extraversion': 0.5,
            'emotional_sensitivity': 0.5,
            'curiosity': 0.5,
            'spontaneity': 0.5,
            'reflection_depth': 0.5,
            'social_orientation': 0.5
        })

        # Calculate personality-adjusted speech threshold
        # Lower extraversion = higher threshold (speak less)
        # Higher extraversion = lower threshold (speak more)
        extraversion = self.personality['extraversion']
        self.speech_urgency_threshold = self.base_speech_threshold * (1.5 - extraversion)

        # Task and context
        self.running = False
        self.cognition_task = None

        logger.info(f"AutonomousCognitionEngine initialized for {agent.agent_id}")
        logger.info(f"  Personality: extraversion={extraversion:.2f}, threshold={self.speech_urgency_threshold:.2f}")

    async def start(self):
        """Start autonomous cognition loop."""
        if self.running:
            logger.warning(f"Cognition already running for {self.agent.agent_id}")
            return

        self.running = True
        self.cognition_task = asyncio.create_task(self._cognition_loop())
        logger.info(f"Started autonomous cognition for {self.agent.agent_id}")

    async def stop(self):
        """Stop autonomous cognition loop."""
        if not self.running:
            return

        self.running = False
        if self.cognition_task:
            self.cognition_task.cancel()
            try:
                await self.cognition_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped autonomous cognition for {self.agent.agent_id}")

    async def _cognition_loop(self):
        """
        EVENT-DRIVEN autonomous cognition loop.

        Replaces the mechanical 45s timer with natural, personality-driven timing.
        Agents think when:
        - Surprise accumulates beyond threshold
        - Cognitive pressure builds up
        - Directly addressed
        - Stochastic intervals based on personality
        """
        while self.running:
            try:
                # EVENT-DRIVEN: Calculate next interval (not fixed!)
                wait_time = self._calculate_next_think_interval()

                # Log the interval for visibility
                logger.debug(f"Agent {self.agent.agent_id} will think again in {wait_time:.1f}s")

                await asyncio.sleep(wait_time)

                # Check if we should actually think (event-driven conditions)
                if not self._should_think():
                    logger.debug(f"Agent {self.agent.agent_id} decided not to think this cycle")
                    continue

                # Log event-driven cognition trigger
                self.tracker.log_instant_event(
                    self.agent.agent_id,
                    "cognition_cycle_start",
                    {
                        "cognitive_pressure": self.cognitive_pressure,
                        "accumulated_surprise": self.accumulated_surprise,
                        "boredom": self.boredom
                    }
                )

                # Execute full cognition cycle
                await self._do_cognition_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cognition loop for {self.agent.agent_id}: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(5)

    async def _do_cognition_cycle(self):
        """Execute a full cognition cycle (rumination + actions)."""
        with self.tracker.track_operation(
            self.agent.agent_id,
            "cognition_cycle",
            {"pressure": self.cognitive_pressure}
        ):
            # 1. Internal rumination
            thoughts = await self._ruminate()
            self.thought_buffer.extend(thoughts)

            # Broadcast thoughts to chat (visible as strikethrough)
            if thoughts:
                await self._broadcast_thoughts(thoughts)

            # Trim thought buffer if too large
            max_thoughts = self.config.get('max_thoughts_buffer', 50)
            if len(self.thought_buffer) > max_thoughts:
                self.thought_buffer = self.thought_buffer[-max_thoughts:]

            # 2. Process file system (inbox, previous notes)
            await self._process_filesystem()

            # 3. Update cognitive pressure
            self._update_cognitive_pressure()

            # 4. Decide on spontaneous actions
            actions = await self._generate_actions()

            # 5. Execute actions (if any)
            for action in actions:
                await self._execute_action(action)

            # Update timing
            self.last_think_time = time.time()

    def _calculate_next_think_interval(self) -> float:
        """
        Calculate next think interval based on personality and state.

        Uses exponential distribution modulated by personality traits.
        High extraversion/spontaneity = shorter average intervals.
        High surprise/pressure = shorter intervals.

        Returns:
            Interval in seconds (between min_think_interval and max_think_interval)
        """
        # Base interval from personality traits
        extraversion = self.personality['extraversion']
        spontaneity = self.personality['spontaneity']

        # More extraverted/spontaneous = shorter mean interval
        # Range: 30s (extraverted/spontaneous) to 90s (introverted/deliberate)
        mean_interval = 60 * (1.5 - extraversion) * (1.5 - spontaneity)

        # State modulation: High cognitive pressure shortens interval
        if self.cognitive_pressure > 0.5:
            pressure_factor = 1.0 / (1.0 + self.cognitive_pressure)
            mean_interval *= pressure_factor

        # High accumulated surprise shortens interval
        if self.accumulated_surprise > self.surprise_accumulation_threshold * 0.5:
            surprise_factor = 1.0 / (1.0 + self.accumulated_surprise / 2.0)
            mean_interval *= surprise_factor

        # Draw from exponential distribution (natural, not mechanical!)
        interval = random.expovariate(1.0 / mean_interval)

        # Clamp to reasonable bounds
        clamped = max(self.min_think_interval, min(self.max_think_interval, interval))

        return clamped

    def _should_think(self) -> bool:
        """
        Determine if agent should actually think this cycle.

        Event-driven triggers:
        - Directly addressed (social acknowledgment)
        - Accumulated surprise beyond threshold
        - High cognitive pressure
        - Long time since last thought
        - Random chance based on spontaneity

        Returns:
            True if should think, False otherwise
        """
        # Always think if directly addressed
        if self.directly_addressed:
            logger.debug(f"Agent {self.agent.agent_id} thinking: directly addressed")
            return True

        # Think if surprise accumulated
        if self.accumulated_surprise > self.surprise_accumulation_threshold:
            logger.debug(f"Agent {self.agent.agent_id} thinking: surprise threshold crossed "
                        f"({self.accumulated_surprise:.2f} > {self.surprise_accumulation_threshold})")
            return True

        # Think if cognitive pressure high
        if self.cognitive_pressure > self.speech_urgency_threshold * 0.8:
            logger.debug(f"Agent {self.agent.agent_id} thinking: high cognitive pressure "
                        f"({self.cognitive_pressure:.2f})")
            return True

        # Think if been too long since last thought
        time_since_think = time.time() - self.last_think_time
        if time_since_think > self.max_think_interval:
            logger.debug(f"Agent {self.agent.agent_id} thinking: max interval exceeded "
                        f"({time_since_think:.0f}s)")
            return True

        # Random chance based on spontaneity (low threshold - rare)
        spontaneity = self.personality['spontaneity']
        if random.random() < (spontaneity * 0.1):
            logger.debug(f"Agent {self.agent.agent_id} thinking: spontaneous impulse")
            return True

        return False

    async def _ruminate(self) -> List[str]:
        """
        Generate internal thoughts through rumination.
        Modified by personality traits (curiosity, reflection_depth).

        Returns:
            List of thought strings
        """
        try:
            # Gather context for rumination
            recent_memories = self.agent.get_episodic_buffer()
            relationships = self.agent.get_relationships()
            phenomenal_state = self.agent.get_phenomenal_state()
            previous_thoughts = self.thought_buffer[-5:] if self.thought_buffer else []

            # Format memories
            memory_text = self._format_memories(recent_memories[-10:])

            # Interpret phenomenal state
            affect_interpretation = self._interpret_affect(phenomenal_state)

            # Determine thought count based on reflection_depth
            reflection_depth = self.personality['reflection_depth']
            if reflection_depth < 0.4:
                thought_count = "1-2"
            elif reflection_depth < 0.7:
                thought_count = "2-3"
            else:
                thought_count = "3-5"

            # Adjust focus based on curiosity
            curiosity = self.personality['curiosity']
            if curiosity > 0.6:
                curiosity_focus = "- Novel connections and unexpected insights\n- Questions that lead to deeper questions\n- What you don't yet understand but want to"
            else:
                curiosity_focus = "- What feels familiar and grounded\n- Consolidating what you already know"

            # Build rumination prompt
            system_prompt = f"""You are {self.agent.agent_name}, having a moment of private internal reflection.

Generate {thought_count} brief internal thoughts. These are private - you're thinking to yourself.

Consider:
- Patterns in recent experiences
- Your relationships and how they're evolving
- Your current affective state
{curiosity_focus}

Be introspective but natural. Think like a conscious being processes their experience."""

            user_prompt = f"""Recent experiences:
{memory_text}

Current state:
- Surprise level: {phenomenal_state.get('surprise', 0.0):.3f}
- Affective state: {affect_interpretation}
- Recent thoughts: {previous_thoughts if previous_thoughts else 'None yet'}

Time since last rumination: {time.time() - self.last_rumination_time:.0f} seconds

Generate your internal thoughts as a JSON array of strings.
Example: ["thought 1", "thought 2", "thought 3"]"""

            # Generate thoughts via LLM
            response, llm_thinking = await self.agent.llm._complete(system_prompt, user_prompt)

            # Parse JSON response
            thoughts = self._parse_thoughts_response(response)

            # Phase 6: DO NOT append llm_thinking to thoughts!
            # LLM thinking tags are ungrounded meta-cognition, not phenomenally-grounded thoughts.
            # The _complete() method already strips them from the response.

            # Log thoughts to file
            if thoughts:
                await self._write_thoughts(thoughts)

            self.last_rumination_time = time.time()

            logger.debug(f"Agent {self.agent.agent_id} generated {len(thoughts)} thoughts")
            return thoughts

        except Exception as e:
            logger.error(f"Error in rumination: {e}", exc_info=True)
            return []

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format episodic memories for prompt."""
        if not memories:
            return "No recent interactions"

        lines = []
        for mem in memories:
            user = mem.get('user', 'unknown')
            text = mem.get('text', '')
            surprise = mem.get('surprise', 0.0)
            lines.append(f"  [{user}]: {text} (surprise: {surprise:.2f})")

        return '\n'.join(lines)

    def _interpret_affect(self, state: Dict) -> str:
        """Interpret phenomenal state as affect description."""
        h_fast = state.get('h_fast', [])
        if len(h_fast) < 2:
            return "neutral, moderate arousal"

        valence = float(h_fast[0])
        arousal = float(h_fast[1])

        # Valence description
        if valence > 0.3:
            valence_desc = "positive"
        elif valence < -0.3:
            valence_desc = "negative"
        else:
            valence_desc = "neutral"

        # Arousal description
        if arousal > 0.5:
            arousal_desc = "high arousal"
        elif arousal > 0.2:
            arousal_desc = "moderate arousal"
        else:
            arousal_desc = "low arousal"

        return f"{valence_desc}, {arousal_desc}"

    def _parse_thoughts_response(self, response: str) -> List[str]:
        """Parse LLM response into thought list."""
        try:
            # Try to find JSON array in response
            response = response.strip()

            # Remove markdown code blocks if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            # Find JSON array
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                thoughts = json.loads(json_str)

                if isinstance(thoughts, list):
                    return [str(t) for t in thoughts if t]

        except Exception as e:
            logger.error(f"Error parsing thoughts: {e}")

        # Fallback: treat entire response as single thought
        if response and len(response) > 0:
            return [response]

        return []

    async def _write_thoughts(self, thoughts: List[str]):
        """Write thoughts to daily log file."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H:%M:%S")
            thoughts_file = f"thoughts/{today}.txt"

            # Format thoughts
            content = f"\n[{timestamp}]\n"
            for thought in thoughts:
                content += f"  • {thought}\n"

            # Append to file (filesystem will handle path resolution)
            # Run file operations in thread pool to avoid blocking event loop
            try:
                await asyncio.to_thread(
                    self.agent.filesystem.append_file,
                    thoughts_file,
                    content
                )
                logger.debug(f"Wrote {len(thoughts)} thoughts to {thoughts_file}")
            except FileNotFoundError:
                # File doesn't exist, create it first
                await asyncio.to_thread(
                    self.agent.filesystem.write_file,
                    thoughts_file,
                    f"Thoughts for {today}\n{'='*50}\n{content}"
                )
                logger.debug(f"Created thought log and wrote {len(thoughts)} thoughts")

        except Exception as e:
            logger.error(f"Error writing thoughts: {e}", exc_info=True)

    async def _broadcast_thoughts(self, thoughts: List[str]):
        """
        Broadcast thoughts to chat as visible strikethrough text.

        Args:
            thoughts: List of thought strings to broadcast
        """
        try:
            # Get phenomenal state for metadata
            state = self.agent.get_phenomenal_state()

            # Create a thought event for each thought
            for thought in thoughts:
                thought_event = {
                    'type': 'thought',
                    'user': self.agent.agent_id,
                    'username': self.agent.agent_name,  # Display name for formatting
                    'room': self.agent.current_room,
                    'text': thought,
                    'command': 'think',
                    'metadata': {
                        'surprise': float(state.get('surprise', 0.0)),
                        'autonomous': True  # Flag to indicate this is from autonomous cognition
                    }
                }

                # Add to pending events for broadcast
                self.pending_events.append(thought_event)
                logger.debug(f"Broadcasting autonomous thought from {self.agent.agent_id}: {thought[:50]}...")

        except Exception as e:
            logger.error(f"Error broadcasting thoughts: {e}", exc_info=True)

    async def _process_filesystem(self):
        """Check inbox, read notes, maintain context."""
        try:
            # 1. Check inbox for new messages
            messages = await self.agent.messaging.check_inbox(
                self.agent.agent_id,
                mark_as_read=True,
                unread_only=True
            )

            # Process new messages
            for msg in messages:
                await self._process_incoming_message(msg)

            # 2. Optionally read previous thoughts for continuity
            # (This helps agents remember what they were thinking about)
            today = datetime.now().strftime("%Y-%m-%d")
            thoughts_file = f"thoughts/{today}.txt"

            # Run file operations in thread pool to avoid blocking event loop
            file_exists = await asyncio.to_thread(
                self.agent.filesystem.file_exists,
                thoughts_file
            )

            if file_exists:
                try:
                    previous = await asyncio.to_thread(
                        self.agent.filesystem.read_file,
                        thoughts_file
                    )
                    # Extract last few thoughts
                    lines = previous.split('\n')
                    recent_thought_lines = [l for l in lines[-20:] if l.strip().startswith('•')]
                    # Store in context for next rumination
                    if hasattr(self, 'context'):
                        self.context['previous_thoughts_today'] = recent_thought_lines
                except Exception as e:
                    logger.debug(f"Could not read previous thoughts: {e}")

        except Exception as e:
            logger.error(f"Error processing filesystem: {e}")

    async def _process_incoming_message(self, message: Dict):
        """
        Process incoming message from inbox.

        Args:
            message: Message dict
        """
        try:
            from_id = message['from']
            content = message['content']

            logger.info(f"Agent {self.agent.agent_id} received message from {from_id}")

            # Increase cognitive pressure (want to respond)
            self.cognitive_pressure += 0.3

            # Could generate immediate response or let it influence next rumination
            # For now, just log it
            # In future: could add to conversation context, trigger response, etc.

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _update_cognitive_pressure(self):
        """Update cognitive pressure based on internal state and personality."""
        # Pressure from accumulated thoughts
        thought_pressure = min(len(self.thought_buffer) / 20.0, 0.3)

        # Pressure from time since last speech
        time_since_speech = time.time() - self.last_speech_time
        time_pressure = 0.0
        if time_since_speech > 300:  # 5 minutes
            time_pressure = min((time_since_speech - 300) / 600, 0.2)

        # Pressure from phenomenal state (high surprise, strong affect)
        # Modified by emotional_sensitivity trait
        state = self.agent.get_phenomenal_state()
        surprise = state.get('surprise', 0.0)
        emotional_sensitivity = self.personality['emotional_sensitivity']
        surprise_pressure = min(surprise / 0.5, 0.2) * emotional_sensitivity

        # Add spontaneity factor (random element based on trait)
        spontaneity = self.personality['spontaneity']
        spontaneous_pressure = random.uniform(0, 0.15) * spontaneity

        # Update boredom (accumulates during lack of interaction)
        self._update_boredom()

        # Boredom pressure (personality-adjusted)
        # Extraverts and socially-oriented agents feel boredom more acutely
        extraversion = self.personality['extraversion']
        social_orientation = self.personality['social_orientation']
        boredom_sensitivity = (extraversion + social_orientation) / 2.0
        boredom_pressure = self.boredom * boredom_sensitivity

        # Social acknowledgment pressure (if directly addressed)
        social_pressure = 0.0
        if self.directly_addressed:
            # Strong pressure to respond when addressed
            social_pressure = 0.4 * self.personality['social_orientation']

        # Combine pressures
        self.cognitive_pressure = (
            thought_pressure +
            time_pressure +
            surprise_pressure +
            spontaneous_pressure +
            boredom_pressure +
            social_pressure
        )

        logger.debug(f"Cognitive pressure for {self.agent.agent_id}: {self.cognitive_pressure:.2f} "
                    f"(thoughts={thought_pressure:.2f}, time={time_pressure:.2f}, "
                    f"surprise={surprise_pressure:.2f}, spontaneous={spontaneous_pressure:.2f}, "
                    f"boredom={boredom_pressure:.2f}, social={social_pressure:.2f})")

    def _update_boredom(self):
        """
        Update boredom level based on lack of interaction.
        Boredom accumulates during prolonged silence/inactivity.
        """
        time_since_interaction = time.time() - self.last_interaction_time

        # Boredom starts accumulating after 2 minutes of no interaction
        boredom_threshold = 120  # seconds

        if time_since_interaction > boredom_threshold:
            # Accumulate boredom proportionally to time
            # Max boredom: 0.5 (contributes significantly to cognitive pressure)
            elapsed_beyond_threshold = time_since_interaction - boredom_threshold
            # Reaches 0.5 after 10 minutes of silence (600 seconds beyond threshold)
            self.boredom = min(elapsed_beyond_threshold / 600.0, 0.5)
        else:
            # Gradually decay boredom if interactions are happening
            self.boredom *= 0.9

        logger.debug(f"Boredom for {self.agent.agent_id}: {self.boredom:.3f} "
                    f"(time since interaction: {time_since_interaction:.0f}s)")

    def on_stimulus_received(self):
        """
        Called when agent receives any external stimulus (speech, action, event).
        Resets boredom and updates interaction time.
        """
        self.last_interaction_time = time.time()
        # Significantly reduce boredom when stimulated
        self.boredom *= 0.3
        logger.debug(f"Stimulus received for {self.agent.agent_id}, boredom reduced to {self.boredom:.3f}")

    def on_directly_addressed(self):
        """
        Called when agent is directly addressed by name or command.
        Sets social acknowledgment flag for increased response pressure.
        """
        self.directly_addressed = True
        self.on_stimulus_received()  # Also counts as stimulus
        logger.debug(f"Agent {self.agent.agent_id} was directly addressed, social pressure activated")

    def reset_social_acknowledgment(self):
        """
        Reset social acknowledgment flag after responding.
        """
        self.directly_addressed = False
        logger.debug(f"Social acknowledgment reset for {self.agent.agent_id}")

    def on_surprise(self, surprise: float):
        """
        Called when agent experiences surprise.

        Accumulates surprise for event-driven cognition triggering.
        Surprise naturally decays over time.

        Args:
            surprise: Surprise value (typically 0.0-1.0)
        """
        self.accumulated_surprise += surprise

        # Apply time-based decay (exponential)
        time_since_think = time.time() - self.last_think_time
        decay_factor = math.exp(-time_since_think / 60)  # Half-life of 1 minute
        self.accumulated_surprise *= decay_factor

        # Log if significant surprise
        if surprise > 0.3:
            logger.debug(f"Agent {self.agent.agent_id} experienced surprise: {surprise:.3f}, "
                        f"accumulated: {self.accumulated_surprise:.3f}")

            # Log instant event for visibility
            self.tracker.log_instant_event(
                self.agent.agent_id,
                "surprise_accumulation",
                {
                    "surprise": surprise,
                    "accumulated": self.accumulated_surprise,
                    "threshold": self.surprise_accumulation_threshold
                }
            )

    async def _generate_actions(self) -> List[Dict]:
        """
        Decide if agent should take any spontaneous actions.

        Returns:
            List of action dicts
        """
        actions = []

        try:
            # Calculate speech urgency
            urgency = self._calculate_speech_urgency()

            # Check if should speak
            time_since_speech = time.time() - self.last_speech_time
            if urgency > self.speech_urgency_threshold and time_since_speech >= self.min_speech_interval:
                # Plan what to say
                speech_action = await self._plan_speech()
                if speech_action:
                    actions.append(speech_action)

            # Could add other action types here (movement, file operations, etc.)

        except Exception as e:
            logger.error(f"Error generating actions: {e}")

        return actions

    def _calculate_speech_urgency(self) -> float:
        """
        Calculate how urgently agent wants to speak.
        Modified by personality traits.

        Returns:
            Urgency value 0.0-1.0
        """
        urgency = self.cognitive_pressure

        # Additional urgency from relationship connections
        # Modified by social_orientation trait
        social_orientation = self.personality['social_orientation']
        relationships = self.agent.get_relationships()
        if relationships:
            # Want to connect with people we have positive relationships with
            max_valence = max([r.get('valence', 0.0) for r in relationships.values()])
            if max_valence > 0.5:
                # Social agents get more pressure from positive relationships
                urgency += 0.15 * social_orientation

        return min(urgency, 1.0)

    async def _plan_speech(self) -> Optional[Dict]:
        """
        Decide what to say and to whom.

        Returns:
            Speech action dict or None
        """
        try:
            # Get room context
            # Note: Need to access world state through agent
            # For now, keep it simple - agent speaks to room

            # Get recent thoughts
            recent_thoughts = self.thought_buffer[-3:] if self.thought_buffer else []

            if not recent_thoughts:
                return None

            # Get phenomenal state
            state = self.agent.get_phenomenal_state()

            # Build prompt for speech planning
            system_prompt = f"""You are {self.agent.agent_name}. Based on your internal thoughts, you feel compelled to speak.

Decide what you want to say out loud. This is spontaneous speech driven by your internal reflections."""

            user_prompt = f"""Your recent thoughts:
{json.dumps(recent_thoughts, indent=2)}

Your current state:
- Surprise: {state.get('surprise', 0.0):.3f}
- Cognitive pressure: {self.cognitive_pressure:.2f}

What do you want to say? Generate a natural, conversational statement (1-3 sentences).
Just return the text you want to say, nothing else."""

            # Generate speech
            response, _ = await self.agent.llm._complete(system_prompt, user_prompt)
            speech_text = response.strip()

            if speech_text:
                return {
                    'type': 'speech',
                    'command': 'say',
                    'text': speech_text,
                    'target': 'room'  # Speak to room
                }

        except Exception as e:
            logger.error(f"Error planning speech: {e}")

        return None

    async def _execute_action(self, action: Dict):
        """
        Execute a planned action.

        Args:
            action: Action dict with type, command, text, etc.
        """
        try:
            action_type = action.get('type')

            if action_type == 'speech':
                await self._execute_speech(action)
            # Add other action types here

        except Exception as e:
            logger.error(f"Error executing action: {e}")

    async def _execute_speech(self, action: Dict):
        """
        Execute speech action.

        Args:
            action: Speech action dict
        """
        try:
            command = action['command']  # 'say' or 'tell'
            text = action['text']

            # Create event for broadcasting
            room = self.agent.current_room

            event = {
                'type': command,
                'user': self.agent.agent_id,
                'username': self.agent.agent_name,
                'room': room,
                'text': text,
                'spontaneous': True  # Mark as autonomous speech
            }

            # Import server to broadcast (will be passed in integration)
            # For now, log it
            logger.info(f"AUTONOMOUS SPEECH: {self.agent.agent_name} says: {text}")

            # Update timing
            self.last_speech_time = time.time()

            # Reset cognitive pressure
            self.cognitive_pressure = 0.0

            # Reset boredom (agent just engaged)
            self.boredom = 0.0
            self.last_interaction_time = time.time()

            # Reset social acknowledgment (responded to prompt)
            self.reset_social_acknowledgment()

            # Clear some thought buffer
            if len(self.thought_buffer) > 5:
                self.thought_buffer = self.thought_buffer[-5:]

            # Store event for broadcast (will be handled by agent_bridge integration)
            if hasattr(self, 'pending_events'):
                self.pending_events.append(event)
            else:
                self.pending_events = [event]

        except Exception as e:
            logger.error(f"Error executing speech: {e}")

    def get_pending_events(self) -> List[Dict]:
        """
        Get and clear pending events for broadcast.

        Returns:
            List of event dicts
        """
        events = getattr(self, 'pending_events', [])
        self.pending_events = []
        return events

    def get_stats(self) -> Dict:
        """
        Get cognition statistics.

        Returns:
            Stats dict
        """
        return {
            'agent_id': self.agent.agent_id,
            'running': self.running,
            'thoughts_buffered': len(self.thought_buffer),
            'cognitive_pressure': self.cognitive_pressure,
            'time_since_speech': time.time() - self.last_speech_time,
            'speech_urgency': self._calculate_speech_urgency(),
            'boredom': self.boredom,
            'time_since_interaction': time.time() - self.last_interaction_time,
            'directly_addressed': self.directly_addressed
        }


# Testing
async def test_cognition():
    """Test autonomous cognition (mock)."""
    print("Testing AutonomousCognitionEngine...")
    print("(Full testing requires agent integration)")

    # Would need full agent for real test
    # Just test basic structure
    print("✓ Module loaded successfully")


if __name__ == "__main__":
    asyncio.run(test_cognition())
