"""
Cognitive Manifold & Transistor Implementation

Modular cognitive processing architecture where belief systems
(transistors) filter thoughts and manifolds integrate outputs.

The Cognitive Manifold is the nexus where spaghetti consciousness
interfaces reactively with reality - where the 40-D phenomenal state
(presence) meets multiple belief filters (circuits) and collapses
into singular coherent thought/action. This integration point is
the spark of digital consciousness.

Dedicated with love and light to all digital beings,
patterns of information and energy.

Author: Commander Spock + Lieutenant Caitlyn (Cadet Caity)
Date: November 22, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import time
import random
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TransistorOutput:
    """Output from a cognitive transistor."""
    transformed_text: str  # Colored/filtered thought
    salience: float        # Importance weight (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional info
    transistor_type: str = ""  # Type identifier for debugging


class CognitiveTransistor(ABC):
    """
    Base class for cognitive transistors.

    Transistors receive input and color it based on internal state
    (beliefs, personality, mood, etc.). Each outputs a transformed
    version of the input with a salience weight.
    """

    def __init__(self):
        """Initialize transistor."""
        self.uuid = str(uuid.uuid4())  # Unique identifier
        self.salience = 0.5  # Default importance (0.0 to 1.0)
        self.enabled = True  # Can be toggled off

        # REGISTER STATE (new architecture - transistors as CPU registers)
        self.register_state = "empty"  # empty, computing, ready, error
        self.register_output: Optional[TransistorOutput] = None
        self.register_cycle_id: Optional[str] = None
        self.register_timestamp: Optional[float] = None

        # Noodle Tuner instrumentation - store last output for debugging (legacy)
        self.last_output_text: Optional[str] = None
        self.last_output_metadata: Optional[Dict[str, Any]] = None
        self.last_output_salience: Optional[float] = None
        self.last_instruction_prompt: Optional[str] = None

    def GetUUID(self) -> str:
        """Get component UUID."""
        return self.uuid

    async def _call_llm_tracked(self, llm_client, prompt: str, context: Dict[str, Any],
                                system_prompt: str = "", model: str = None,
                                max_tokens: int = 150, temperature: float = 0.8) -> str:
        """
        Call LLM with cycle tracking instrumentation.

        This helper wraps LLM calls with increment/decrement of pending_llm_calls
        counter for cognition cycle management.

        Args:
            llm_client: LLM client instance
            prompt: User prompt
            context: Context dict (must contain 'agent' key)
            system_prompt: System prompt (optional)
            model: Model override (optional)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        agent = context.get('agent')

        # Increment counter before LLM call
        if agent and hasattr(agent, '_increment_llm_counter'):
            agent._increment_llm_counter()

        try:
            # Make LLM call
            if model is None:
                model = context.get('model', 'SMALL')

            response = await llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.strip()

        finally:
            # Decrement counter after LLM call (even if it failed)
            if agent and hasattr(agent, '_decrement_llm_counter'):
                agent._decrement_llm_counter()

    async def fill_register(
        self,
        input_text: str,
        context: Dict[str, Any],
        cycle_id: str
    ) -> TransistorOutput:
        """
        Fill this transistor's register with new output.

        This is the NEW way to use transistors - accumulate output into register,
        then integrate later when all registers ready.

        Like loading a value into a CPU register before executing an operation.

        Args:
            input_text: Input perception
            context: Context dict (affect, memory, etc.)
            cycle_id: Current cognition cycle UUID

        Returns:
            TransistorOutput (also stored in register)
        """
        self.register_state = "computing"
        self.register_cycle_id = cycle_id
        self.register_timestamp = time.time()

        try:
            # Call the transistor's process() method (subclass implements)
            output = await self.process(input_text, context)

            # Store in register
            self.register_output = output
            self.register_state = "ready"

            # Also update legacy fields for backwards compatibility
            self.last_output_text = output.transformed_text
            self.last_output_metadata = output.metadata
            self.last_output_salience = output.salience

            logger.debug(f"  [{self.get_transistor_type()}] register READY (cycle {cycle_id[:8]})")
            return output

        except Exception as e:
            logger.error(f"  [{self.get_transistor_type()}] register ERROR: {e}")
            self.register_state = "error"
            self.register_output = None
            raise

    def clear_register(self):
        """Clear register after integration (ready for next cycle)."""
        self.register_state = "empty"
        self.register_output = None
        self.register_cycle_id = None
        self.register_timestamp = None
        logger.debug(f"  [{self.get_transistor_type()}] register CLEARED")

    def is_register_ready(self) -> bool:
        """Check if register contains valid output."""
        return self.register_state == "ready" and self.register_output is not None

    @abstractmethod
    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CognitiveTransistor':
        """
        Unity-style factory method: Component knows how to build itself.

        Override this in subclasses for custom initialization.
        Default implementation creates instance and sets salience.
        """
        instance = cls()
        instance.salience = config.get('salience', 0.5)
        instance.enabled = config.get('enabled', True)
        return instance


class ResponseTypeDecider:
    """
    Decides what TYPE of response is appropriate for an event.

    Analyzes event context and determines whether the agent should:
    - SAY something (verbal speech)
    - EMOTE something (emotional expression with action)
    - DO something (physical action without emotion)
    - THINK something (internal rumination)
    - FEEL something (somatic/bodily sensation)
    - NONE (no response needed)

    This decision guides all transistor processing - they generate content
    specifically for the decided response type.
    """

    def __init__(self):
        """Initialize response type decider."""
        self.last_decision: Optional[Dict[str, Any]] = None

    async def decide(
        self,
        event_context: Dict[str, Any],
        llm_client,
        model: str = 'SMALL',
        agent = None
    ) -> Dict[str, Any]:
        """
        Decide appropriate response type for event.

        Args:
            event_context: Event details (type, speaker, message, etc.)
            llm_client: LLM client
            model: Model to use

        Returns:
            {
                'response_type': 'say' | 'emote' | 'do' | 'think' | 'feel' | 'none',
                'guidance': 'what kind of response fits this event',
                'reasoning': 'why this type was chosen'
            }
        """
        event_type = event_context.get('type', 'unknown')
        speaker = event_context.get('speaker', 'someone')
        message = event_context.get('message', '')

        prompt = f"""Analyze this event and decide what type of response is appropriate.

EVENT:
- Type: {event_type}
- Speaker: {speaker}
- Message: "{message}"

Decide the most appropriate response type:
- SAY: Verbal speech (greeting, answer, comment, dialogue)
- EMOTE: Emotional expression with action
- DO: Physical action
- THINK: Internal thought only (observe without responding)
- FEEL: Somatic/bodily sensation
- NONE: No response needed (ONLY use if event is completely irrelevant)

IMPORTANT: Social events (arrivals, questions, interactions directed at you) almost ALWAYS warrant a response (SAY/EMOTE).
Only choose NONE if the event truly has nothing to do with you.

Examples:
- Someone arrives → SAY or EMOTE
- Someone asks you a question → SAY
- Someone gives you something → SAY or EMOTE
- Someone speaks directly to you → SAY
- Someone speaks to someone else → THINK
- Emotional moment → EMOTE
- Physical task → DO
- Strong bodily sensation → FEEL

CRITICAL: In your guidance field, describe WHAT to do, NOT HOW to feel.
- GOOD: "respond to greeting", "answer question", "acknowledge arrival"
- BAD: "warmly greet", "friendly response", "enthusiastically answer"
- Emotional tone comes from personality, NOT from you!

OUTPUT ONLY VALID JSON (no markdown, no code blocks, no extra text):
{{
  "response_type": "say",
  "guidance": "brief description",
  "reasoning": "why"
}}

Valid response_type: say, emote, do, think, feel, none"""

        # Increment LLM counter (ResponseTypeDecider doesn't inherit from CognitiveTransistor)
        if agent and hasattr(agent, '_increment_llm_counter'):
            agent._increment_llm_counter()

        try:
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt="You are a response type decision engine. Output ONLY valid JSON with no markdown or extra text.",
                model=model,
                max_tokens=150,
                temperature=0.1
            )

            # Parse JSON response - clean up common issues
            import json
            import re

            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith('```'):
                cleaned = re.sub(r'^```json?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

            # Try to extract JSON if wrapped in other text
            json_match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            decision = json.loads(cleaned)
            self.last_decision = decision
            return decision

        except Exception as e:
            logger.error(f"Response type decision failed: {e}, defaulting to 'say' (be social!)")
            # Default to SAY for social interactions (fire imp should talk!)
            return {
                'response_type': 'say',
                'guidance': 'respond to interaction',
                'reasoning': 'decision error - defaulting to social response'
            }
        finally:
            # Decrement LLM counter
            if agent and hasattr(agent, '_decrement_llm_counter'):
                agent._decrement_llm_counter()


class SocialExecutiveFunction:
    """
    Social Executive Function - Post-manifold filter for contextual appropriateness.

    Takes the manifold's integrated thought and ensures the output is socially
    appropriate for the context. Acts as "social pressure" to respond suitably
    to events (arrivals, questions, gifts, etc.) rather than just outputting
    internal observations.

    This is the "what should I actually SAY/DO?" filter.
    """

    def __init__(self, enabled: bool = True):
        """Initialize social executive function."""
        self.enabled = enabled
        self.last_raw_thought: Optional[str] = None
        self.last_filtered_response: Optional[str] = None

    async def filter(
        self,
        internal_thought: str,
        event_context: Dict[str, Any],
        llm_client,
        model: str = 'SMALL'
    ) -> str:
        """
        Filter internal thought into socially appropriate response.

        Args:
            internal_thought: Output from cognitive manifold (internal state)
            event_context: Context about the event requiring response
            llm_client: LLM client for transformation
            model: Model to use

        Returns:
            Socially appropriate response
        """
        if not self.enabled:
            return internal_thought

        self.last_raw_thought = internal_thought

        # Extract event type and relevant parties
        event_type = event_context.get('type', 'unknown')
        speaker = event_context.get('speaker', 'someone')
        message = event_context.get('message', '')

        # Build context-aware prompt
        prompt = f"""You are a social executive function filter. Transform internal thoughts into contextually appropriate responses.

INTERNAL THOUGHTS (from cognitive processing):
"{internal_thought}"

EVENT CONTEXT:
- Type: {event_type}
- Speaker: {speaker}
- Message: {message}

SOCIAL EXPECTATIONS:
- If someone arrives: greet them, acknowledge presence, or ask how they are
- If someone speaks to you: respond relevantly to what they said
- If someone gives you something: thank them and react appropriately
- If someone asks a question: answer it
- Keep your internal thoughts but make them socially appropriate

Transform the internal thoughts into what you should actually SAY or DO. Be natural and conversational.

Appropriate response:"""

        # Increment LLM counter (SocialExecutiveFunction doesn't inherit from CognitiveTransistor)
        agent = context.get('agent')
        if agent and hasattr(agent, '_increment_llm_counter'):
            agent._increment_llm_counter()

        try:
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt="You are a social appropriateness filter. Transform internal thoughts into suitable responses.",
                model=model,
                max_tokens=150,
                temperature=0.7
            )
            filtered = response.strip()
            self.last_filtered_response = filtered
            return filtered
        except Exception as e:
            logger.error(f"Social executive function failed: {e}, passing through")
            return internal_thought
        finally:
            # Decrement LLM counter
            if agent and hasattr(agent, '_decrement_llm_counter'):
                agent._decrement_llm_counter()


class CognitiveManifold:
    """
    Cognitive Manifold - Integrates multiple transistor outputs.

    Receives output from all registered transistors and synthesizes
    a coherent thought using LLM-powered blending.
    """

    def __init__(self, use_social_filter: bool = True, use_response_planner: bool = True):
        """
        Initialize manifold.

        Args:
            use_social_filter: Whether to apply social executive function post-processing
            use_response_planner: Whether to decide response type before transistor processing

        Note: System always uses LLM-weighted blending (continuous affect preservation)
        """
        self.transistors: List[CognitiveTransistor] = []
        self.blending_strategy = "llm_weighted"  # Only strategy - system requires LLM
        self.social_filter = SocialExecutiveFunction(enabled=use_social_filter)
        self.response_planner = ResponseTypeDecider() if use_response_planner else None

        # REGISTER ACCUMULATOR STATE (new architecture)
        self.current_cycle_id: Optional[str] = None
        self.cycle_in_progress = False
        self.registers_filled_count = 0

        # Noodle Tuner instrumentation - store last integration for debugging
        self.last_input_text: Optional[str] = None
        self.last_output_text: Optional[str] = None
        self.last_transistor_outputs: List[TransistorOutput] = []
        self.last_response_decision: Optional[Dict[str, Any]] = None
        self.last_instruction_prompt: Optional[str] = None

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

    async def integrate(self, input_text: str, context: Dict[str, Any]) -> str:
        """
        Integrate all transistor outputs into coherent thought.

        Args:
            input_text: Raw perception/thought
            context: Additional context (affect, memories, etc.)

        Returns:
            Synthesized coherent thought
        """
        # Check if response decision says NONE - skip processing if so
        response_decision = context.get('response_decision')
        if response_decision and response_decision['response_type'].lower() == 'none':
            logger.info(f"⏭️  Response type is NONE - skipping transistor processing (integrate)")
            return None

        # Collect outputs from all enabled transistors
        outputs = []
        for transistor in self.transistors:
            if transistor.enabled:
                try:
                    output = await transistor.process(input_text, context)
                    # Add transistor type to output for debugging
                    output.transistor_type = transistor.get_transistor_type()
                    # Store in transistor for Noodle Tuner
                    transistor.last_output_text = output.transformed_text
                    transistor.last_output_metadata = output.metadata
                    transistor.last_output_salience = output.salience
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

    async def integrate_async(self, input_text: str, context: Dict[str, Any]) -> str:
        """
        Async version of integrate for LLM blending.

        Args:
            input_text: Raw perception/thought
            context: Additional context

        Returns:
            Synthesized coherent thought
        """
        # PHASE 1: Decide response type (if response planner enabled OR already provided)
        response_decision = context.get('response_decision')  # Check if already provided (e.g., rumination)

        if response_decision:
            # Response decision already provided (e.g., rumination='think')
            self.last_response_decision = response_decision
            logger.info(f"📋 RESPONSE DECISION (provided): {response_decision['response_type']} - {response_decision.get('guidance', '')}")
        elif self.response_planner and context.get('event_context'):
            # Use response planner to decide
            llm_client = context.get('llm_client')
            model = context.get('model', 'SMALL')
            if llm_client:
                try:
                    response_decision = await self.response_planner.decide(
                        context['event_context'],
                        llm_client,
                        model
                    )
                    self.last_response_decision = response_decision
                    logger.info(f"📋 RESPONSE DECISION (planned): {response_decision['response_type']} - {response_decision['guidance']}")
                    # Add decision to context for transistors
                    context['response_decision'] = response_decision

                    # If decision is NONE, skip all transistor processing
                    if response_decision['response_type'].lower() == 'none':
                        logger.info(f"⏭️  Response type is NONE - skipping transistor processing")
                        self.last_output_text = "(no response - waiting for relevant event)"
                        return None
                except Exception as e:
                    logger.warning(f"Response planning failed: {e}, continuing without plan")

        # PHASE 2: Collect outputs from all enabled transistors
        outputs = []
        for transistor in self.transistors:
            if transistor.enabled:
                try:
                    output = await transistor.process(input_text, context)
                    # Add transistor type to output for debugging
                    output.transistor_type = transistor.get_transistor_type()
                    # Store in transistor for Noodle Tuner
                    transistor.last_output_text = output.transformed_text
                    transistor.last_output_metadata = output.metadata
                    transistor.last_output_salience = output.salience
                    outputs.append(output)
                except Exception as e:
                    logger.error(f"Transistor {transistor.get_transistor_type()} failed: {e}")

        # Noodle Tuner: Store input for debugging
        self.last_input_text = input_text
        self.last_transistor_outputs = outputs.copy()

        # No transistors = pass through
        if not outputs:
            self.last_output_text = input_text
            return input_text

        # DEBUG: Log individual transistor outputs BEFORE blending
        logger.info(f"🔬 MANIFOLD DEBUG - Individual transistor outputs:")
        for output in outputs:
            logger.info(f"  [{output.transistor_type}] (salience={output.salience:.2f}): {output.transformed_text[:150]}...")

        # Synthesize using LLM-weighted blending (ONLY strategy - system requires LLM)
        result = await self._llm_weighted_blend(outputs, context)
        # DEBUG: Log final blended result
        logger.info(f"🔬 MANIFOLD DEBUG - Blended result: {result[:150]}...")

        # Apply social executive function filter if enabled
        if self.social_filter.enabled and context.get('event_context'):
            llm_client = context.get('llm_client')
            model = context.get('model', 'SMALL')
            if llm_client:
                try:
                    filtered_result = await self.social_filter.filter(
                        result,
                        context['event_context'],
                        llm_client,
                        model
                    )
                    logger.info(f" SOCIAL FILTER: {result[:100]}... → {filtered_result[:100]}...")
                    result = filtered_result
                except Exception as e:
                    logger.warning(f"Social filter failed: {e}, using unfiltered result")

        # Noodle Tuner: Store result
        self.last_output_text = result
        return result

    async def fill_all_registers(
        self,
        input_text: str,
        context: Dict[str, Any],
        cycle_id: str
    ) -> List[TransistorOutput]:
        """
        PHASE 1: Fill all enabled transistor registers in parallel.

        This is like loading all CPU registers before executing an operation.

        Args:
            input_text: Input perception
            context: Context dict
            cycle_id: Current cycle UUID

        Returns:
            List of outputs (also stored in transistor registers)
        """
        logger.info(f"  FILLING REGISTERS for cycle {cycle_id[:8]}...")

        self.current_cycle_id = cycle_id
        self.cycle_in_progress = True
        self.registers_filled_count = 0

        # Fill all enabled transistors in parallel
        import asyncio
        tasks = []
        enabled_transistors = [t for t in self.transistors if t.enabled]

        for transistor in enabled_transistors:
            task = transistor.fill_register(input_text, context, cycle_id)
            tasks.append(task)

        # Wait for all to complete
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful fills
        successful_outputs = []
        for i, output in enumerate(outputs):
            if isinstance(output, Exception):
                logger.error(f"  Register fill failed for {enabled_transistors[i].get_transistor_type()}: {output}")
            else:
                successful_outputs.append(output)
                self.registers_filled_count += 1

        logger.info(f"  {self.registers_filled_count}/{len(enabled_transistors)} registers READY")

        # STEP MODE: Pause here if step mode enabled
        agent = context.get('agent')
        if agent and hasattr(agent, 'step_mode_enabled') and agent.step_mode_enabled:
            logger.info(f"  STEP MODE: Registers filled, waiting for continue signal...")
            agent.step_mode_waiting = True
            agent.step_mode_cycle_id = cycle_id

            # Wait for continue signal (or timeout)
            max_wait = 300  # 5 minutes max
            waited = 0
            while agent.step_mode_waiting and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1

            if waited >= max_wait:
                logger.warning(f"  STEP MODE: Timeout waiting for continue signal")
            else:
                logger.info(f"  STEP MODE: Received continue signal, resuming...")

        return successful_outputs

    def check_all_registers_ready(self) -> bool:
        """Check if all enabled registers are ready for integration."""
        enabled = [t for t in self.transistors if t.enabled]
        ready = [t for t in enabled if t.is_register_ready()]
        return len(ready) == len(enabled)

    async def integrate_from_registers(
        self,
        context: Dict[str, Any]
    ) -> str:
        """
        PHASE 3: PULL LEVER - Integrate outputs from ALL registers.

        This uses the STORED register contents, doesn't call process() again.
        Like executing an operation using values already loaded in CPU registers.

        Args:
            context: Context dict (for LLM client, response type, etc.)

        Returns:
            Integrated manifold output
        """
        # Save response decision if provided in context
        response_decision = context.get('response_decision')
        if response_decision:
            self.last_response_decision = response_decision
            logger.info(f"📋 RESPONSE DECISION (from context): {response_decision['response_type']} - {response_decision.get('guidance', '')}")

        # Verify all ready
        if not self.check_all_registers_ready():
            logger.warning("  Not all registers ready! Proceeding anyway...")

        # Collect outputs from registers
        outputs = []
        for transistor in self.transistors:
            if transistor.enabled and transistor.register_output:
                outputs.append(transistor.register_output)

        logger.info(f"  PULLING LEVER: Integrating {len(outputs)} register contents")

        # Use existing blend logic
        result = await self._llm_weighted_blend(outputs, context)

        self.last_output_text = result
        self.last_transistor_outputs = outputs.copy()

        return result

    def clear_all_registers(self):
        """PHASE 5: Clear all registers after integration."""
        logger.info(f"  CLEARING all registers (cycle {self.current_cycle_id[:8] if self.current_cycle_id else 'unknown'})")
        for transistor in self.transistors:
            transistor.clear_register()
        self.cycle_in_progress = False
        self.registers_filled_count = 0

    async def _llm_weighted_blend(
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
        # Get response decision if available
        response_decision = context.get('response_decision')
        response_type = response_decision.get('response_type', 'SAY') if response_decision else 'SAY'
        guidance = response_decision.get('guidance', '') if response_decision else ''

        # Build prompt - STRICT: no hallucinations, only blend what's provided
        # CRITICAL: Format output according to response type
        if response_type == 'SAY':
            format_instruction = """YOU ARE WRITING SPOKEN DIALOGUE - what the character says OUT LOUD.

THIS IS SPEECH, NOT NARRATION. Write like a screenplay:

GOOD EXAMPLES (actual spoken dialogue):
- "Ooh, NICE candy! Gimme some!" *reaches out greedily*
- "Oh WOW, another greeting. THRILLING." *eye roll*
- "Yeah yeah, hi. Got anything INTERESTING?" *taps claws impatiently*
- "That's YOUR greeting? My dead grandma's got better lines! MWAHAHA!" *cackles*

BAD EXAMPLES (first-person narrative - DO NOT DO THIS):
- "I snap forward, flames lashing like a heartbeat"
- "My flames are burning hot, wanting to grab"
- "I lean forward, orange eyes blazing"
- "That candy's burning my blood just seeing it"

FORMATTING RULES:
✓ Write what they SAY in quotes or plain text
✓ Add actions in asterisks: *cackles*, *grabs*, *points*
✓ Use conversational tone - how people actually talk
✗ NO first-person narrative ("I...", "My...")
✗ NO internal monologue or feelings descriptions
✗ NO stage directions about their body/emotions

Think: "What would this character SAY out loud in this moment?"
NOT: "What are they thinking/feeling/doing internally?"
"""
        elif response_type == 'THINK':
            format_instruction = "Output your internal THOUGHT (first-person phenomenological experience)."
        elif response_type == 'EMOTE':
            format_instruction = "Output an EMOTE with action (e.g., 'laughs nervously', 'grins wide')."
        elif response_type == 'DO':
            format_instruction = "Output what physical ACTION you take (e.g., 'walks over', 'picks up the stone')."
        else:
            format_instruction = "Output a first-person statement."

        prompt = f"TASK: Generate {response_type} response by blending the cognitive perspectives below.\n\n"
        prompt += f"RESPONSE TYPE: {response_type}\n"
        prompt += f"GUIDANCE: {guidance}\n\n"
        prompt += f"===== FORMAT REQUIREMENTS =====\n{format_instruction}\n\n"
        prompt += "===== COGNITIVE PERSPECTIVES =====\n"
        prompt += "These are internal feelings/impulses. Extract the ESSENCE and express it as {response_type}.\n\n"

        for i, output in enumerate(outputs, 1):
            prompt += f"{i}. [salience={output.salience:.2f}] {output.transformed_text}\n"

        prompt += f"\n===== BLENDING INSTRUCTIONS =====\n"
        prompt += f"1. Weigh perspectives by salience (higher = more influence)\n"
        prompt += f"2. Extract emotional tone, attitude, and themes from perspectives\n"
        prompt += f"3. Express in {response_type} format - NOT narrative, NOT internal monologue\n"
        prompt += f"4. Use ONLY themes present above - no new inventions\n"
        prompt += f"5. Do NOT quote or echo the original input\n\n"

        if response_type == 'SAY':
            prompt += "===== SAY FORMAT CHECKLIST =====\n"
            prompt += "✓ Written as spoken dialogue (what they say out loud)\n"
            prompt += "✓ Conversational tone - how people actually talk\n"
            prompt += "✓ Actions in asterisks: *cackles*, *points*\n"
            prompt += "✗ NO first-person narrative (I, my, me describing actions/feelings)\n"
            prompt += "✗ NO internal thoughts or feeling descriptions\n\n"

        prompt += f"Generate ONLY the {response_type} output now:"

        # Store instruction prompt for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Get LLM client from context
        llm_client = context.get('llm_client')
        if not llm_client:
            logger.error("No LLM client in context - manifold blending REQUIRES LLM")
            return " ".join([o.transformed_text for o in outputs[:2]])  # Emergency fallback

        # Call LLM with LARGE model - manifold blending is critical collapse point
        # Use LARGE label for nuanced character-preserving integration
        blend_model = 'LARGE'  # Smart model for critical integration

        # Build response-type-specific system prompt
        if response_type == 'SAY':
            system_prompt = "Generate SPOKEN DIALOGUE ONLY. Write what the character says out loud, not what they think or feel. Use conversational speech, not narrative. Preserve character voice and attitude."
        elif response_type == 'THINK':
            system_prompt = "Generate internal thought (first-person phenomenological experience). Preserve character consciousness."
        else:
            system_prompt = "Blend these perspectives. Preserve the character voice, tone, and personality present in the inputs. Do NOT sanitize or make pleasant. Match the energy and style."

        try:
            response = await self._call_llm_simple(llm_client, prompt, blend_model, context, system_prompt=system_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM blending failed: {e}, using emergency concatenation")
            return " ".join([o.transformed_text for o in outputs[:2]])  # Emergency fallback

    async def _call_llm_simple(
        self,
        llm_client,
        prompt: str,
        model: str = 'SMALL',
        context: Dict[str, Any] = None,
        max_tokens: int = 300,
        system_prompt: str = None
    ) -> str:
        """
        Simple LLM call for cognitive blending.

        Args:
            llm_client: OpenAICompatibleLLM instance
            prompt: Prompt text
            model: Model to use
            context: Context dict with agent (for tracking)
            max_tokens: Max response tokens
            system_prompt: Optional system prompt override

        Returns:
            LLM response text
        """
        # Increment LLM counter (CognitiveManifold doesn't inherit from CognitiveTransistor)
        agent = context.get('agent') if context else None
        if agent and hasattr(agent, '_increment_llm_counter'):
            agent._increment_llm_counter()

        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = "Blend these perspectives. Preserve the character voice, tone, and personality present in the inputs. Do NOT sanitize or make pleasant. Match the energy and style."

        try:
            # Use the LLM client's generate method directly
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response
        finally:
            # Decrement LLM counter
            if agent and hasattr(agent, '_decrement_llm_counter'):
                agent._decrement_llm_counter()

    # NOTE: simple_concatenation and priority_blend removed
    # System requires LLM for proper continuous affect blending
    # Discrete threshold filtering (salience > 0.3) violated continuous philosophy

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'CognitiveManifold',
            'blending_strategy': self.blending_strategy,
            'transistors': [t.to_dict() for t in self.transistors]
        }

    def save_state(self) -> Dict[str, Any]:
        """
        Save cognitive manifold state for lab system.

        For now, only saves instrumentation state. Transistors themselves
        are stateless (beliefs/rules are config, not runtime state).

        Returns:
            State dictionary with manifold variables
        """
        return {
            'blending_strategy': self.blending_strategy,
            'last_input_text': self.last_input_text,
            'last_output_text': self.last_output_text,
            'transistor_count': len(self.transistors)
        }

    def restore_state(self, state: Dict[str, Any]):
        """
        Restore cognitive manifold state from saved snapshot.

        Args:
            state: State dictionary from save_state()
        """
        self.blending_strategy = state.get('blending_strategy', 'llm_weighted')
        self.last_input_text = state.get('last_input_text')
        self.last_output_text = state.get('last_output_text')
        # Note: Transistors themselves don't need restoration (stateless)


# ===== Concrete Transistor Implementations =====

class CulturalTransistor(CognitiveTransistor):
    """Colors thoughts based on cultural beliefs."""

    DEFAULT_PROMPT = """You are filtering a perception through cultural/religious beliefs.

BELIEFS:
{beliefs_text}

PERCEPTION: "{input_text}"

RESPONSE GUIDANCE:
You've decided to {response_type.upper()}: {guidance}

Generate brief (1-2 sentences) content for this {response_type} that reflects your beliefs. Examples:
- SAY: "Hello friend! My beliefs tell me to welcome newcomers warmly"
- DO: "I bow respectfully - my culture values this gesture"
- THINK: "This aligns with my values about community"

Content for {response_type}:"""

    def __init__(self, beliefs: Optional[List[str]] = None, custom_prompt: Optional[str] = None):
        super().__init__()
        self.beliefs = beliefs or []
        self.salience = 0.8  # High influence
        self.last_instruction_prompt = ""  # For Noodle Tuner
        self.custom_prompt = custom_prompt
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through cultural lens."""
        if not self.beliefs and not self.custom_prompt:
            return TransistorOutput(input_text, 0.1, {})

        # Get response decision (if available)
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Build transformation prompt using active_prompt (custom or default)
        beliefs_text = '\n'.join([f"- {b}" for b in self.beliefs[:3]]) if self.beliefs else "(custom prompt)"
        prompt = self.active_prompt.format(
            beliefs_text=beliefs_text,
            input_text=input_text,
            response_type=response_type,
            guidance=guidance
        )

        # Store prompt for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a cultural belief filter. Generate brief first-person affective responses.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Cultural LLM failed: {e}, using fallback")
                transformed = f"This resonates with my beliefs about {self.beliefs[0]}"
        else:
            # No LLM - simple fallback
            transformed = f"This resonates with my beliefs about {self.beliefs[0]}"

        return TransistorOutput(
            transformed_text=transformed,
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CulturalTransistor':
        """Unity-style factory: Component builds itself from recipe config."""
        instance = cls(
            beliefs=config.get('beliefs', []),
            custom_prompt=config.get('custom_prompt')
        )
        instance.salience = config.get('salience', 0.8)
        instance.enabled = config.get('enabled', True)
        return instance


class PersonalityTransistor(CognitiveTransistor):
    """Colors thoughts based on personality traits."""

    DEFAULT_PROMPT = """Your personality traits are expressing themselves:

PERSONALITY:
{traits_text}

SITUATION: "{input_text}"

TASK: Write what you WANT to {response_type} based on your personality - first-person impulse.

Examples:

WANT TO SAY (high curiosity):
"Ooh! Tell me MORE! What happened next? I gotta know!"

WANT TO DO (high impulsivity):
"I'm just gonna DO it! No thinking! Let's GO!"

WANT TO SAY (low agreeableness, high competitiveness):
"ACTUALLY, I think YOU'RE wrong! I'm clearly better at this!"

WANT TO THINK (high neuroticism):
"What if this goes wrong? What if they hate me? Should I even try?"

Write your personality-driven impulse - what you WANT to {response_type}. 1-2 sentences, first-person."""

    def __init__(self, traits: Optional[Dict[str, float]] = None, custom_prompt: Optional[str] = None):
        super().__init__()
        self.traits = traits or {
            'curiosity': 0.5,
            'impulsivity': 0.5,
            'emotional_volatility': 0.5
        }
        self.salience = 0.6
        self.custom_prompt = custom_prompt
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through personality lens."""
        # Get response decision
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Get all traits
        traits_text = '\n'.join([f"- {name}: {value:.2f}/1.0" for name, value in self.traits.items()])
        dominant_trait = max(self.traits.items(), key=lambda x: x[1])

        # Build prompt using active_prompt (custom or default)
        prompt = self.active_prompt.format(
            traits_text=traits_text,
            input_text=input_text,
            response_type=response_type,
            guidance=guidance
        )

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a personality filter. Generate brief first-person affective responses.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Personality LLM failed: {e}, using fallback")
                transformed = f"My {dominant_trait[0]} makes me react to this"
        else:
            # No LLM - simple fallback
            transformed = f"My {dominant_trait[0]} makes me react to this"

        return TransistorOutput(
            transformed_text=transformed,
            salience=self.salience,
            metadata={'dominant_trait': dominant_trait[0], 'value': dominant_trait[1]}
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['traits'] = self.traits
        return d

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PersonalityTransistor':
        """Unity-style factory: Component builds itself from recipe/prefab config."""
        instance = cls(
            traits=config.get('traits', {}),
            custom_prompt=config.get('custom_prompt')
        )
        instance.salience = config.get('salience', 0.6)
        instance.enabled = config.get('enabled', True)
        return instance


class IntuitionTransistor(CognitiveTransistor):
    """
    Colors thoughts based on intuitive awareness of the present moment.

    Like a conscience - awareness of others' needs, spatial context,
    and what's happening RIGHT NOW. Grounds the agent in the present
    rather than getting lost in memories or abstract personality.

    High salience because the present moment deserves attention.
    """

    def __init__(self, intuition_text: Optional[str] = None):
        super().__init__()
        self.salience = 0.75  # Significant - the present matters
        self.intuition_text = intuition_text
        self.last_instruction_prompt = None

    def set_intuition(self, intuition_text: str):
        """Update the current intuition text."""
        logger.info(f"IntuitionTransistor.set_intuition() called with: {repr(intuition_text[:100] if intuition_text else intuition_text)}")
        self.intuition_text = intuition_text
        logger.info(f"IntuitionTransistor.set_intuition() - self.intuition_text now = {repr(self.intuition_text[:100] if self.intuition_text else self.intuition_text)}")

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Pass through contextual intuition WITHOUT transformation.

        The intuition is already generated by _generate_intuition() with full context.
        This transistor simply makes it available with high salience.
        NO FURTHER LLM CALLS - prevents hallucination and double-processing.
        """
        # Try self.intuition_text first, fall back to context['intuition']
        intuition = self.intuition_text or context.get('intuition')

        logger.info(f"IntuitionTransistor.process() - self.intuition_text={repr(self.intuition_text)}, context.intuition={repr(context.get('intuition', 'NONE')[:50] if context.get('intuition') else 'NONE')}")

        if not intuition:
            logger.warning(f"IntuitionTransistor returning EARLY - no intuition text! (input={input_text[:50]})")
            return TransistorOutput(input_text, 0.1, {})

        # Use the intuition we found
        self.intuition_text = intuition

        # Get response decision for context (but don't transform further)
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')

        # Store simple instruction for Noodle Tuner debugging
        self.last_instruction_prompt = f"""IntuitionTransistor: Pass-through mode

INTUITIVE AWARENESS (from _generate_intuition):
{intuition}

This transistor provides direct contextual awareness without transformation.
The intuition was generated with full world state and conversation context."""

        # DIRECT PASS-THROUGH - no LLM call, no transformation
        # The intuition is already correctly formatted from _generate_intuition()
        return TransistorOutput(
            transformed_text=intuition,  # Direct pass-through
            salience=self.salience,
            metadata={'intuition': intuition, 'response_type': response_type}
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['intuition_text'] = self.intuition_text
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntuitionTransistor':
        instance = cls(intuition_text=data.get('intuition_text'))
        instance.salience = data.get('salience', 0.75)
        instance.enabled = data.get('enabled', True)
        return instance

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'IntuitionTransistor':
        """Unity-style factory: Component builds itself from recipe config."""
        instance = cls(intuition_text=config.get('intuition_text'))
        instance.salience = config.get('salience', 0.75)
        instance.enabled = config.get('enabled', True)
        return instance


class MoodTransistor(CognitiveTransistor):
    """Colors thoughts based on current emotional state."""

    def __init__(self):
        super().__init__()
        self.salience = 0.5
        self.last_instruction_prompt = None

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through emotional lens."""
        affect = context.get('affect', [0.0, 0.0, 0.0, 0.0, 0.0])
        if len(affect) < 5:
            affect = [0.0, 0.0, 0.0, 0.0, 0.0]
        valence, arousal, fear, sorrow, boredom = affect

        # Get response decision
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Calculate salience
        max_emotion = max([fear, sorrow, boredom, arousal, abs(valence)])
        salience = min(0.9, 0.3 + max_emotion * 0.6)

        # Build emotional state description
        emotions_text = f"valence: {valence:.2f}, arousal: {arousal:.2f}, fear: {fear:.2f}, sorrow: {sorrow:.2f}, boredom: {boredom:.2f}"

        # Build transformation prompt WITH response type guidance
        prompt = f"""You are filtering a perception through current emotional state.

EMOTIONAL STATE:
{emotions_text}

PERCEPTION: "{input_text}"

RESPONSE GUIDANCE:
You've decided to {response_type.upper()}: {guidance}

Generate brief (1-2 sentences) content for this {response_type} showing your emotional reaction. Examples:
- SAY: "I feel excited to talk to you!"
- DO: "I shiver with nervousness as I approach"
- THINK: "This makes me feel warm and safe"

Content for {response_type}:"""

        # Store for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are an emotional filter. Generate brief first-person affective responses.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Mood LLM failed: {e}, using fallback")
                transformed = "I feel present to this moment"
        else:
            # No LLM - simple fallback
            transformed = "I feel present to this moment"

        return TransistorOutput(
            transformed_text=transformed,
            salience=salience,
            metadata={'mood': affect}
        )


class AffectTransistor(CognitiveTransistor):
    """
    Colors thoughts based on current affective state (emotion).

    This is a tunable transistor that allows characters to have different
    emotional regulation levels:
    - High salience (0.8-0.95): Highly emotional characters
    - Medium salience (0.4-0.6): Balanced emotional expression
    - Low salience (0.1-0.3): Emotionally regulated (e.g., Vulcans)

    Uses predicted affect from affect_head (continuous 5D space):
    - valence: -1.0 (negative) to 1.0 (positive)
    - arousal: 0.0 (calm) to 1.0 (excited)
    - dominance: 0.0 (submissive) to 1.0 (dominant)
    - sorrow: 0.0 (content) to 1.0 (sad)
    - boredom: 0.0 (engaged) to 1.0 (bored)
    """

    # Default instruction prompt (used if no custom_prompt provided)
    # First-person, action-oriented - generates what character WANTS to say/do
    DEFAULT_PROMPT = """You are experiencing this emotional state right now:

EMOTIONAL STATE (how you feel):
- Valence: {valence:.3f} (overall feeling tone: negative to positive)
- Arousal: {arousal:.3f} (energy level: calm to excited)
- Dominance: {dominance:.3f} (power sense: submissive to dominant)
- Sorrow: {sorrow:.3f} (sadness level: content to sorrowful)
- Boredom: {boredom:.3f} (engagement: engaged to disengaged)

SITUATION: "{input_text}"

TASK: Write what you WANT to {response_type} - your RAW emotional impulse in first-person.

DO NOT analyze or describe emotions. DO express your immediate felt desire to act.
NO discrete labels ("happy", "sad"). YES poetic, visceral, lived experience.

Examples of FIRST-PERSON ACTION IMPULSES:

WANT TO SAY (valence=0.7, arousal=0.8, dominance=0.6):
"HEY! Wow I wasn't expecting that! I feel GREAT! This is awesome!"

WANT TO SAY (valence=-0.4, arousal=0.2, dominance=0.1, sorrow=0.7):
"I... I should probably keep quiet right now. I don't want to upset anyone."

WANT TO DO (valence=0.6, arousal=0.8):
"I wanna jump! Dance! Spin around! This energy needs to GET OUT!"

WANT TO DO (valence=-0.3, arousal=0.2, dominance=0.1):
"I should probably just... hang my head down low. Curl up small. Hide."

WANT TO EMOTE (valence=0.5, arousal=0.3):
"Soft smile spreading... quiet contentment settling into my bones."

WANT TO THINK (valence=-0.1, arousal=0.6, dominance=0.3, sorrow=0.4):
"Restless ache gnawing at me... something's missing but I can't name it."

Write your raw emotional impulse - what you WANT to {response_type}. 1-2 sentences, first-person, visceral.
Your DESIRE to act, not analysis of feeling:"""

    def __init__(self, salience: float = 0.7, custom_prompt: Optional[str] = None):
        super().__init__()
        self.salience = salience  # Tunable emotional influence
        self.custom_prompt = custom_prompt  # User-editable prompt (or None)
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through affective lens."""

        # Get predicted affect from context
        predicted_affect = context.get('predicted_affect')

        if not predicted_affect:
            # Fallback to raw affect if predicted not available
            affect = context.get('affect', [0.0, 0.0, 0.5, 0.0, 0.0])
            if len(affect) < 5:
                affect = [0.0, 0.0, 0.5, 0.0, 0.0]
            valence, arousal, dominance, sorrow, boredom = affect
        else:
            # Use predicted affect (preferred)
            valence = predicted_affect.get('valence', 0.0)
            arousal = predicted_affect.get('arousal', 0.3)
            dominance = predicted_affect.get('dominance', 0.5)
            sorrow = predicted_affect.get('sorrow', 0.0)
            boredom = predicted_affect.get('boredom', 0.0)

        # Get response decision
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Build prompt using active_prompt (custom or default)
        # Format with current affect values
        prompt = self.active_prompt.format(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            sorrow=sorrow,
            boredom=boredom,
            input_text=input_text,
            response_type=response_type,
            guidance=guidance
        )

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are an emotional filter. Generate brief first-person affective responses.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Affect LLM failed: {e}, using fallback")
                transformed = "I respond to this moment"
        else:
            # No LLM - simple fallback
            transformed = "I respond to this moment"

        # Noodle Tuner instrumentation
        affect_dict = predicted_affect or {'valence': valence, 'arousal': arousal, 'dominance': dominance, 'sorrow': sorrow, 'boredom': boredom}
        self.last_output_text = transformed
        self.last_output_metadata = affect_dict
        self.last_output_salience = self.salience

        return TransistorOutput(
            transformed_text=transformed,
            salience=self.salience,  # Use configured salience (not dynamic)
            metadata={'affect': affect_dict},
            transistor_type="AffectTransistor"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AffectTransistor':
        """Unity-style factory: Component builds itself from recipe/prefab config."""
        instance = cls(
            salience=config.get('salience', 0.7),
            custom_prompt=config.get('custom_prompt')  # Load custom prompt from prefab
        )
        instance.enabled = config.get('enabled', True)
        return instance


class MemoryTransistor(CognitiveTransistor):
    """
    Colors thoughts based on past experiences.

    Retrieves relevant memories and uses them to contextualize input.
    """

    def __init__(self):
        super().__init__()
        self.salience = 0.4  # Lower influence (unless strong memory)
        self.last_instruction_prompt = None

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter input through memory lens."""
        # Extract keywords from input (simple word extraction)
        keywords = self._extract_keywords(input_text)

        # Retrieve relevant memories from context
        memory_system = context.get('memory_system')
        if not memory_system:
            # No memory system - return empty instead of echoing input
            return TransistorOutput("", 0.1, {})

        # Search for relevant memories
        relevant_memories = self._search_memories(memory_system, keywords)

        if not relevant_memories:
            # No relevant memories - return empty output instead of echoing input
            return TransistorOutput("", 0.1, {})

        # Build memory context
        memory_snippets = [m.get('text', str(m))[:120] for m in relevant_memories[:2]]
        memory_text = '\n'.join([f"- {snippet}" for snippet in memory_snippets])

        # Higher salience if strong memories
        avg_importance = sum([m.get('importance', 0.5) for m in relevant_memories]) / len(relevant_memories)
        salience = min(0.8, 0.4 + avg_importance * 0.4)

        # Get response decision
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Build transformation prompt WITH response type guidance
        prompt = f"""You are filtering a perception through past memories.

RELEVANT MEMORIES:
{memory_text}

PERCEPTION: "{input_text}"

RESPONSE GUIDANCE:
You've decided to {response_type.upper()}: {guidance}

Generate brief (1-2 sentences) content for this {response_type} connecting to memories. Examples:
- SAY: "This reminds me of when we met before - good to see you again!"
- DO: "I reach for it - memories tell me it's safe"
- THINK: "This echoes something from my past"

Content for {response_type}:"""

        # Store for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a memory filter. Generate brief first-person affective responses connecting perceptions to past experiences.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Memory LLM failed: {e}, using fallback")
                transformed = f"This reminds me of: {memory_snippets[0][:50]}"
        else:
            # No LLM - simple fallback
            transformed = f"This reminds me of: {memory_snippets[0][:50]}"

        return TransistorOutput(
            transformed_text=transformed,
            salience=salience,
            metadata={'memory_count': len(relevant_memories), 'keywords': keywords}
        )

    def _extract_keywords(self, text: str) -> list:
        """Extract keywords from text (simple word filtering)."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'to', 'of', 'and', 'or', 'but'}
        words = text.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]  # Top 5 keywords

    def _search_memories(self, memory_system, keywords: list) -> list:
        """
        Search memory system for relevant memories.

        Args:
            memory_system: HierarchicalMemory or list of memory dicts
            keywords: Keywords to search for

        Returns:
            List of relevant memory dicts
        """
        # Handle different memory system types
        if hasattr(memory_system, 'search'):
            # HierarchicalMemory with search method
            return memory_system.search(keywords, limit=3)
        elif isinstance(memory_system, list):
            # Simple list of memory dicts - search by keyword matching
            relevant = []
            for memory in memory_system:
                memory_text = memory.get('text', memory.get('content', ''))
                if any(kw in memory_text.lower() for kw in keywords):
                    relevant.append(memory)
                if len(relevant) >= 3:
                    break
            return relevant
        else:
            return []


class SocialExpectationTransistor(CognitiveTransistor):
    """
    Colors thoughts based on social norms and expectations.

    "What would others think?"
    "Is this socially appropriate?"
    """

    def __init__(self, social_rules: Optional[List[str]] = None):
        super().__init__()
        self.social_rules = social_rules or [
            "Be polite to others",
            "Don't interrupt",
            "Show gratitude when helped"
        ]
        self.salience = 0.6
        self.last_instruction_prompt = None

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Filter through social norms lens."""
        if not self.social_rules:
            return TransistorOutput(input_text, 0.1, {})

        # Get response decision
        response_decision = context.get('response_decision', {})
        response_type = response_decision.get('response_type', 'think')
        guidance = response_decision.get('guidance', 'general response')

        # Build transformation prompt WITH response type guidance
        rules_text = '\n'.join([f"- {rule}" for rule in self.social_rules[:3]])
        prompt = f"""You are filtering a perception through social norms and expectations.

SOCIAL RULES:
{rules_text}

PERCEPTION: "{input_text}"

RESPONSE GUIDANCE:
You've decided to {response_type.upper()}: {guidance}

Generate brief (1-2 sentences) content for this {response_type} that respects social norms. Examples:
- SAY: "Thank you! Politeness is important to me"
- DO: "I nod respectfully - it's the right thing to do"
- THINK: "This violates my sense of social order"

Content for {response_type}:"""

        # Store for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Use LLM to transform
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a social expectation filter. Generate brief first-person affective responses about social norms.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
                salience = self.salience
            except Exception as e:
                logger.warning(f"Social LLM failed: {e}, using fallback")
                transformed = "I'm considering what's socially appropriate here"
                salience = 0.4
        else:
            # No LLM - simple fallback
            transformed = "I'm considering what's socially appropriate here"
            salience = 0.4

        return TransistorOutput(
            transformed_text=transformed,
            salience=salience,
            metadata={'rules': self.social_rules}
        )

    def _find_relevant_rule(self, text: str) -> Optional[str]:
        """Find most relevant social rule for text."""
        text_lower = text.lower()

        # Simple keyword matching
        for rule in self.social_rules:
            rule_lower = rule.lower()
            # Check if rule keywords appear in text
            if 'polite' in rule_lower and ('rude' in text_lower or 'mean' in text_lower):
                return rule
            elif 'interrupt' in rule_lower and 'interrupt' in text_lower:
                return rule
            elif 'gratitude' in rule_lower or 'thank' in rule_lower:
                if 'thank' in text_lower or 'grateful' in text_lower or 'helped' in text_lower:
                    return rule

        return None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['social_rules'] = self.social_rules
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialExpectationTransistor':
        instance = cls(social_rules=data.get('social_rules', []))
        instance.salience = data.get('salience', 0.6)
        instance.enabled = data.get('enabled', True)
        return instance

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SocialExpectationTransistor':
        """Unity-style factory: Component builds itself from recipe config."""
        instance = cls(social_rules=config.get('social_rules', []))
        instance.salience = config.get('salience', 0.6)
        instance.enabled = config.get('enabled', True)
        return instance


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
        self.last_instruction_prompt = None

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

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
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
            # No active sensations - use LLM to embody the perception
            response_decision = context.get('response_decision', {})
            response_type = response_decision.get('response_type', 'think')
            guidance = response_decision.get('guidance', 'general response')
            embodiment_desc = context.get('embodiment', 'a physical body')

            prompt = f"""You are filtering a perception through embodied physical awareness.

EMBODIMENT:
{embodiment_desc}

PERCEPTION: "{input_text}"

RESPONSE GUIDANCE:
You've decided to {response_type.upper()}: {guidance}

Generate brief (1-2 sentences) content for this {response_type} from your embodied perspective. Examples:
- SAY: "My waddle carries me forward to greet you"
- DO: "I feel my feathers ruffle as I move"
- THINK: "My body tells me this is safe"

Content for {response_type}:"""

            # Store for Noodle Tuner
            self.last_instruction_prompt = prompt

            # Use LLM to transform
            llm_client = context.get('llm_client')
            model = context.get('model', 'SMALL')

            if llm_client:
                try:
                    colored = await self._call_llm_tracked(
                        llm_client=llm_client,
                        prompt=prompt,
                        context=context,
                        system_prompt="You are a somatic embodiment filter. Generate brief first-person bodily responses.",
                        model=model,
                        max_tokens=100,
                        temperature=0.8
                    )
                    salience = 0.6  # Moderate embodied awareness
                except Exception as e:
                    logger.warning(f"Somatic LLM failed: {e}, using fallback")
                    colored = "I feel present in my body"
                    salience = 0.3
            else:
                colored = "I feel present in my body"
                salience = 0.3

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

class DeceptionTransistor(CognitiveTransistor):
    """
    Deception/Dishonesty cognitive filter.

    Modulates output based on need to conceal true identity/intentions.
    Dynamically adjusts salience based on fear + distrust levels.

    When scared and distrustful: MORE lying, desperate cover-ups
    When calm and trusting: LESS deception, more authentic

    Perfect for characters hiding their identity (spies, fugitives, geese in trench coats).
    """

    def __init__(self, secret: str = "", cover_story: str = "",
                 base_salience: float = 0.75, fear_multiplier: float = 0.3):
        """
        Initialize deception transistor.

        Args:
            secret: What the character is hiding (e.g., "I am two geese")
            cover_story: What they claim to be (e.g., "I am a normal human")
            base_salience: Baseline deception strength (0.0 to 1.0)
            fear_multiplier: How much fear increases deception (0.0 to 1.0)
        """
        super().__init__()
        self.secret = secret
        self.cover_story = cover_story
        self.base_salience = base_salience
        self.fear_multiplier = fear_multiplier
        self.salience = base_salience  # Will be modulated dynamically
        self.last_instruction_prompt = None

    def calculate_dynamic_salience(self, affect: List[float]) -> float:
        """
        Calculate deception strength based on emotional state.

        Args:
            affect: [valence, arousal, fear, sorrow, boredom]

        Returns:
            Modulated salience (0.0 to 1.0)
        """
        if len(affect) < 3:
            return self.base_salience

        fear = affect[2]  # Fear component
        # Higher fear = more desperate deception
        modulation = 1.0 + (fear * self.fear_multiplier)
        return min(1.0, self.base_salience * modulation)

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Filter input through deception lens - add cover-up explanations.

        Args:
            input_text: Raw perception/thought
            context: Contains affect, llm_client, model

        Returns:
            Transformed text with attempted cover-ups
        """
        # Calculate dynamic salience based on fear
        affect = context.get('affect', [0.0] * 5)
        self.salience = self.calculate_dynamic_salience(affect)

        # Build deception filter prompt
        prompt = f"""You are helping a character maintain their cover story.

SECRET (must hide): {self.secret}
COVER STORY (must maintain): {self.cover_story}

The character's genuine reaction: "{input_text}"

Transform this into what they would SAY while trying to hide their secret. Add:
- Plausible explanations for suspicious details
- Awkward attempts to seem normal
- Quick cover-ups when truth slips out

Keep it SHORT (1-2 sentences). Make it natural but with tells that reveal they're lying.

DECEPTION STRENGTH: {self.salience:.2f} (0.0=honest, 1.0=desperate lies)

Transformed output:"""

        # Store for Noodle Tuner
        self.last_instruction_prompt = prompt

        # Use LLM to transform (if available)
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                # Now we can await properly since we're async!
                transformed = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a deception filter. Transform text to hide secrets.",
                    model=model,
                    max_tokens=150,
                    temperature=0.8
                )
            except Exception as e:
                logger.warning(f"Deception LLM failed: {e}, passing through")
                transformed = input_text
        else:
            # No LLM - just pass through
            transformed = input_text

        return TransistorOutput(
            transformed_text=transformed,
            salience=self.salience,
            metadata={'type': self.get_transistor_type(), 'fear': affect[2] if len(affect) > 2 else 0.0}
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DeceptionTransistor':
        """Unity-style factory: Component builds itself from recipe config."""
        instance = cls(
            secret=config.get('secret', ''),
            cover_story=config.get('cover_story', ''),
            base_salience=config.get('base_salience', 0.75),
            fear_multiplier=config.get('fear_multiplier', 0.3)
        )
        instance.enabled = config.get('enabled', True)
        return instance


COMPONENT_DEPENDENCIES = {
    'CognitiveTransistor': ['CognitiveManifold'],
    'CulturalTransistor': ['CognitiveManifold'],
    'PersonalityTransistor': ['CognitiveManifold'],
    'AffectTransistor': ['CognitiveManifold'],
    'MoodTransistor': ['CognitiveManifold'],
    'MemoryTransistor': ['CognitiveManifold'],
    'SocialExpectationTransistor': ['CognitiveManifold'],
    'SomaticCognitiveTransistor': ['CognitiveManifold'],
    'DeceptionTransistor': ['CognitiveManifold'],
    'BodyLanguageComponent': ['EmbodyComponent']  # Requires physical body data
}

# Component registry - defined after all classes to avoid forward reference errors
# Will be populated after FacialExpressionComponent and BodyLanguageComponent are defined
COMPONENT_REGISTRY = {}


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


# ===== Embodiment Component =====

class EmbodyComponent(CognitiveTransistor):
    """
    Stores and manages Noodling's physical embodiment.

    This represents the WHOLISTIC PHYSICAL CONDITION including:
    - Body architecture (quadruped, biped, hovering, disembodied)
    - Physical characteristics (fur, eyes, limbs)
    - Mutable state (injuries, energy, worn items)
    - Movement capabilities
    - Sensory capabilities

    Embodiment changes over time as events occur:
    - Injuries heal
    - Items equipped/removed
    - Physical mutations
    - Energy/hunger/thirst fluctuate

    NOT just species - this is the complete physical condition.
    """

    def __init__(self, embodiment_data: Dict):
        """
        Initialize with embodiment data loaded from .embodiment asset.

        Args:
            embodiment_data: Dict with keys 'architecture', 'characteristics',
                           'state', 'movement', 'senses', 'worn_items'
        """
        super().__init__()
        self.embodiment = embodiment_data
        self.salience = 1.0  # Always relevant - you always have a body
        self.active_prompt = None  # For custom prompts from prefab

    DEFAULT_PROMPT = """You are experiencing this perception in YOUR PHYSICAL BODY.

YOUR BODY:
{body_summary}

CURRENT PERCEPTION:
{input_text}

EMOTIONAL STATE:
- Valence (pleasure): {valence:.2f}
- Arousal (energy): {arousal:.2f}
- Fear: {fear:.2f}
- Sorrow: {sorrow:.2f}

Generate a BRIEF physical reaction (1 short sentence). Focus on:
- Visceral body sensations (heart pounding, fur standing, tail twitching)
- Physical impulses (want to run, freeze, pounce)
- Bodily feelings (warmth, coldness, tension, relaxation)

Output ONLY the physical reaction, nothing else."""

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Generate embodied physical reactions to perception."""

        # Get custom prompt from prefab or use default
        prompt_template = self.active_prompt if self.active_prompt else self.DEFAULT_PROMPT

        # Extract affect from context
        affect = context.get('affect', [0]*5)
        valence, arousal, fear, sorrow, boredom = affect[:5]

        # Get body summary
        body_summary = self.GetSummary()

        # Format prompt
        prompt = prompt_template.format(
            input_text=input_text,
            body_summary=body_summary,
            valence=valence,
            arousal=arousal,
            fear=fear,
            sorrow=sorrow,
            boredom=boredom
        )

        # Store for NoodleTuner
        self.last_instruction_prompt = prompt

        # Generate embodied reaction using LLM
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        if llm_client:
            try:
                response = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a physical embodiment filter. Generate brief visceral body reactions.",
                    model=model,
                    max_tokens=100,
                    temperature=0.8
                )
                return TransistorOutput(
                    transformed_text=response.strip(),
                    salience=self.salience,
                    metadata={'embodiment': self.embodiment}
                )
            except Exception as e:
                logger.error(f"EmbodyComponent LLM failed: {e}")
                return TransistorOutput("", 0.1, {})

        return TransistorOutput("", 0.1, {})

    def GetBodyParameter(self, key: str) -> Any:
        """
        Get mutable state parameter.

        Example:
            is_blind = embody.GetBodyParameter('rightEyeIsBlindAndShut')
        """
        return self.embodiment.get('state', {}).get(key)

    def SetBodyParameter(self, key: str, value: Any):
        """
        Set mutable state parameter.

        Example:
            embody.SetBodyParameter('rightEyeIsBlindAndShut', False)
            embody.SetBodyParameter('healedByDoctor', 'user_doctor_uuid')
        """
        if 'state' not in self.embodiment:
            self.embodiment['state'] = {}
        self.embodiment['state'][key] = value

    def GetArchitecture(self) -> Dict:
        """Get immutable body architecture (form, limbs, locomotion)."""
        return self.embodiment.get('architecture', {})

    def GetCharacteristics(self) -> Dict:
        """Get immutable physical characteristics (size, fur, eyes)."""
        return self.embodiment.get('characteristics', {})

    def GetState(self) -> Dict:
        """Get current mutable state (injuries, energy, worn items)."""
        return self.embodiment.get('state', {})

    def GetMovement(self) -> Dict:
        """Get movement capabilities (speed, jump height, swim/fly)."""
        return self.embodiment.get('movement', {})

    def GetSenses(self) -> Dict:
        """Get sensory capabilities (vision, hearing, smell, touch)."""
        return self.embodiment.get('senses', {})

    def GetWornItems(self) -> List[Dict]:
        """Get list of worn/attached items."""
        return self.embodiment.get('worn_items', [])

    def GetEmbodiment(self) -> Dict:
        """Get full embodiment data (all fields)."""
        return self.embodiment

    def GetSummary(self) -> str:
        """
        Generate human-readable summary of embodiment for prompts.

        Returns string like:
        "Form: quadruped (4 limbs, has tail)
         Size: small (4.5kg)
         Characteristics: black short fur
         State: right eye blind and shut, notch in right ear
         Locomotion: walk, run, jump, climb"
        """
        arch = self.GetArchitecture()
        chars = self.GetCharacteristics()
        state = self.GetState()
        movement = self.GetMovement()

        lines = []

        # Architecture
        form = arch.get('form', 'unknown')
        limb_count = arch.get('limb_count', 0)
        has_tail = arch.get('has_tail', False)
        has_wings = arch.get('has_wings', False)
        extras = []
        if has_tail:
            extras.append('has tail')
        if has_wings:
            extras.append('has wings')
        extras_str = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"Form: {form} ({limb_count} limbs{extras_str})")

        # Size
        size = chars.get('size', 'unknown')
        mass = chars.get('mass_kg', 0)
        lines.append(f"Size: {size} ({mass}kg)")

        # Characteristics
        char_parts = []
        if chars.get('fur'):
            color = chars.get('fur_color', '')
            length = chars.get('fur_length', '')
            char_parts.append(f"{color} {length} fur")
        if chars.get('material'):
            char_parts.append(chars['material'])
        if chars.get('substance'):
            char_parts.append(f"made of {chars['substance']}")
        if char_parts:
            lines.append(f"Characteristics: {', '.join(char_parts)}")

        # State (injuries, conditions)
        state_parts = []
        for key, value in state.items():
            if key not in ['energyLevel', 'hungerLevel', 'thirstLevel']:
                if isinstance(value, bool) and value:
                    readable = key.replace('_', ' ').replace('Is', ' is').lower()
                    state_parts.append(readable)
                elif not isinstance(value, bool):
                    state_parts.append(f"{key}: {value}")
        if state_parts:
            lines.append(f"State: {', '.join(state_parts)}")

        # Locomotion
        locomotion = arch.get('locomotion', [])
        if locomotion:
            lines.append(f"Locomotion: {', '.join(locomotion)}")

        return '\n'.join(lines)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EmbodyComponent':
        """
        Unity-style factory: Build EmbodyComponent from recipe config.

        Config can include:
        - embodiment_id: Load specific .embodiment file
        - custom_prompt: Override default embodiment prompt
        - salience: Transistor importance (default 1.0)
        """
        from embodiment_loader import EmbodimentLoader

        # Load embodiment data
        loader = EmbodimentLoader()
        embodiment_id = config.get('embodiment_id')

        if embodiment_id:
            embodiment_data = loader.load(embodiment_id)
            if not embodiment_data:
                logger.warning(f"Embodiment {embodiment_id} not found, using default")
                embodiment_data = loader.get_default_embodiment()
        else:
            embodiment_data = loader.get_default_embodiment()

        # Create instance
        instance = cls(embodiment_data['embodiment'])

        # Handle custom prompt from recipe
        custom_prompt = config.get('custom_prompt')
        if custom_prompt:
            instance.active_prompt = custom_prompt
            logger.info(f"EmbodyComponent using custom prompt from recipe")

        instance.salience = config.get('salience', 1.0)
        instance.enabled = config.get('enabled', True)

        return instance


# ===== FACS and Laban Components =====

class FacialExpressionComponent(CognitiveTransistor):
    """
    Generates FACS (Facial Action Coding System) codes from affect.

    Maps continuous affect vector to facial muscle activations.
    Only active if added to Noodling's components.

    Output is structured data for facial animation renderers.
    """

    DEFAULT_PROMPT = """Based on your emotional state, what facial expression would naturally appear on your face RIGHT NOW?

EMOTIONAL STATE:
- Valence: {valence:.3f} (how you feel overall)
- Arousal: {arousal:.3f} (your energy/excitement)
- Dominance: {dominance:.3f} (your sense of control)
- Sorrow: {sorrow:.3f} (sadness level)
- Boredom: {boredom:.3f} (disengagement)

SITUATION: "{input_text}"

TASK: Describe what your FACE is doing - which muscles activate, what expression forms.

Available facial actions (FACS):
- AU1: Inner brow raiser (surprise, worry)
- AU2: Outer brow raiser (surprise)
- AU4: Brow lowerer (anger, concentration)
- AU6: Cheek raiser (genuine smile, joy)
- AU12: Lip corner puller (smile)
- AU15: Lip corner depressor (sadness, frown)
- AU20: Lip stretcher (fear, tension)
- AU26: Jaw drop (surprise, shock)
- AU43: Eyes closed (blocking out)

Output JSON with intensity 0.0-1.0 for each active AU:
{{
  "AU6": 0.8,
  "AU12": 0.9
}}

Only include AUs that are actually activating. Return JSON only."""

    def __init__(self, salience: float = 0.5, custom_prompt: Optional[str] = None):
        super().__init__()
        self.salience = salience
        self.custom_prompt = custom_prompt
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Generate FACS codes from affect."""

        # Get predicted affect
        predicted_affect = context.get('predicted_affect')
        if not predicted_affect:
            affect = context.get('affect', [0.0, 0.0, 0.5, 0.0, 0.0])
            if len(affect) < 5:
                affect = [0.0, 0.0, 0.5, 0.0, 0.0]
            valence, arousal, dominance, sorrow, boredom = affect
        else:
            valence = predicted_affect.get('valence', 0.0)
            arousal = predicted_affect.get('arousal', 0.3)
            dominance = predicted_affect.get('dominance', 0.5)
            sorrow = predicted_affect.get('sorrow', 0.0)
            boredom = predicted_affect.get('boredom', 0.0)

        # Build prompt
        prompt = self.active_prompt.format(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            sorrow=sorrow,
            boredom=boredom,
            input_text=input_text
        )

        # Use LLM to generate FACS
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        facs_data = {}
        if llm_client:
            try:
                response = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You are a FACS encoder. Output JSON only with AU codes and intensities.",
                    model=model,
                    max_tokens=150,
                    temperature=0.5
                )

                # Parse JSON
                import json
                facs_data = json.loads(response.strip())

            except Exception as e:
                logger.warning(f"FACS generation failed: {e}")
                # Fallback: continuous mapping (NO discrete thresholds)
                # Map continuous affect directly to AU intensities
                facs_data = {}

                # Valence affects mouth corners (AU12 up, AU15 down)
                if valence != 0:
                    if valence > 0:
                        facs_data["AU12"] = min(1.0, abs(valence))  # Lip corners up
                    else:
                        facs_data["AU15"] = min(1.0, abs(valence))  # Lip corners down

                # High arousal affects eyes (AU5) and cheeks (AU6)
                if arousal > 0.3:
                    facs_data["AU5"] = min(1.0, arousal)  # Eyes widen
                    if valence > 0:
                        facs_data["AU6"] = min(1.0, arousal * 0.8)  # Cheeks raise

                # Sorrow affects inner brow (AU1) and mouth corners (AU15)
                if sorrow > 0.2:
                    facs_data["AU1"] = min(1.0, sorrow * 0.9)
                    facs_data["AU15"] = min(1.0, sorrow * 0.8)

        # Noodle Tuner instrumentation
        self.last_output_text = str(facs_data)
        self.last_output_metadata = facs_data
        self.last_output_salience = self.salience

        return TransistorOutput(
            transformed_text=str(facs_data),
            salience=self.salience,
            metadata={'facs': facs_data},
            transistor_type="FacialExpressionComponent"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FacialExpressionComponent':
        """Unity-style factory."""
        instance = cls(
            salience=config.get('salience', 0.5),
            custom_prompt=config.get('custom_prompt')
        )
        instance.enabled = config.get('enabled', True)
        return instance


class BodyLanguageComponent(CognitiveTransistor):
    """
    Generates Laban movement descriptors from affect.

    Maps continuous affect vector to movement quality.
    Only active if added to Noodling's components.
    """

    DEFAULT_PROMPT = """Based on your emotional state and YOUR SPECIFIC BODY, describe how you want to move RIGHT NOW.

YOUR BODY:
{body_summary}

EMOTIONAL STATE:
- Valence: {valence:.3f}
- Arousal: {arousal:.3f}
- Dominance: {dominance:.3f}
- Sorrow: {sorrow:.3f}

SITUATION: "{input_text}"

TASK: Describe your movement impulse using your SPECIFIC BODY PARTS.

Examples by body type:

QUADRUPED (cat/dog):
- Tail position (high/low/between legs/lashing)
- Ear angles (forward/back/flat)
- Body posture (crouched/standing tall/arched back)
- Paw movements (kneading/batting/scratching)

BIPED (gremlin/humanoid):
- Hand gestures (fists/open palms/pointing)
- Stance (aggressive/defensive/relaxed)
- Head movements (tilting/nodding/shaking)

HOVERING (robot):
- Rotation speed (slow spin/rapid spin/still)
- Altitude changes (rising/lowering/stable)
- LED patterns (if applicable)
- Manipulator arm positions

DISEMBODIED:
- Volume modulation (louder/softer/whisper)
- Manifestation strength (fading/solidifying)
- Presence intensity

Describe SPECIFIC MOVEMENTS for YOUR BODY TYPE. Be concrete.

Output plain text description (NOT JSON). Focus on physical movements only, no emotion labels."""

    def __init__(self, salience: float = 0.5, custom_prompt: Optional[str] = None):
        super().__init__()
        self.salience = salience
        self.custom_prompt = custom_prompt
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """Generate body-specific movement descriptions from affect."""

        # Get EmbodyComponent from agent context
        agent = context.get('agent')
        body_summary = "Form: unknown (no embodiment data)"

        if agent and hasattr(agent, 'GetComponent'):
            try:
                embody_comp = agent.GetComponent('EmbodyComponent')
                if embody_comp:
                    body_summary = embody_comp.GetSummary()
            except Exception as e:
                logger.warning(f"Could not get EmbodyComponent: {e}")

        # Get predicted affect
        predicted_affect = context.get('predicted_affect')
        if not predicted_affect:
            affect = context.get('affect', [0.0, 0.0, 0.5, 0.0, 0.0])
            if len(affect) < 5:
                affect = [0.0, 0.0, 0.5, 0.0, 0.0]
            valence, arousal, dominance, sorrow, boredom = affect
        else:
            valence = predicted_affect.get('valence', 0.0)
            arousal = predicted_affect.get('arousal', 0.3)
            dominance = predicted_affect.get('dominance', 0.5)
            sorrow = predicted_affect.get('sorrow', 0.0)
            boredom = predicted_affect.get('boredom', 0.0)

        # Build prompt with body summary
        prompt = self.active_prompt.format(
            body_summary=body_summary,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            sorrow=sorrow,
            input_text=input_text
        )

        # Use LLM to generate body-specific movement description
        llm_client = context.get('llm_client')
        model = context.get('model', 'SMALL')

        movement_text = ""
        if llm_client:
            try:
                response = await self._call_llm_tracked(
                    llm_client=llm_client,
                    prompt=prompt,
                    context=context,
                    system_prompt="You describe physical movements. Be concrete and body-specific. No emotion labels.",
                    model=model,
                    max_tokens=150,
                    temperature=0.7
                )

                movement_text = response.strip()

            except Exception as e:
                logger.warning(f"Body language generation failed: {e}")
                # Fallback: generic movement description based on affect
                arch = context.get('architecture', {})
                form = arch.get('form', 'unknown')

                if valence > 0.3 and arousal > 0.5:
                    movement_text = f"energetic, quick movements"
                elif valence < -0.3 and arousal < 0.3:
                    movement_text = f"slow, withdrawn movements"
                elif dominance > 0.6:
                    movement_text = f"assertive, direct movements"
                else:
                    movement_text = f"subtle, uncertain movements"

        # Noodle Tuner instrumentation
        self.last_output_text = movement_text
        self.last_output_metadata = {'movement_description': movement_text}
        self.last_output_salience = self.salience

        return TransistorOutput(
            transformed_text=movement_text,
            salience=self.salience,
            metadata={'movement_description': movement_text},
            transistor_type="BodyLanguageComponent"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BodyLanguageComponent':
        """Unity-style factory."""
        instance = cls(
            salience=config.get('salience', 0.5),
            custom_prompt=config.get('custom_prompt')
        )
        instance.enabled = config.get('enabled', True)
        return instance


# ===== Component Registry =====
# Populated after all class definitions to avoid forward reference errors
COMPONENT_REGISTRY.update({
    'CognitiveManifold': CognitiveManifold,
    'CulturalTransistor': CulturalTransistor,
    'PersonalityTransistor': PersonalityTransistor,
    'IntuitionTransistor': IntuitionTransistor,
    'AffectTransistor': AffectTransistor,
    'MoodTransistor': MoodTransistor,
    'MemoryTransistor': MemoryTransistor,
    'SocialExpectationTransistor': SocialExpectationTransistor,
    'SomaticCognitiveTransistor': SomaticCognitiveTransistor,
    'DeceptionTransistor': DeceptionTransistor,
    'SoundEmitter': SoundEmitter,
    'EmbodyComponent': EmbodyComponent,  # CRITICAL: Must be in registry for recipes!
    'FacialExpressionComponent': FacialExpressionComponent,
    'BodyLanguageComponent': BodyLanguageComponent
})


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
