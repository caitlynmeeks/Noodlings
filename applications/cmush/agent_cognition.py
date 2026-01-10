# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Agent Cognition Loop
#
#   This is the "thinking engine" that runs inside every Noodling.
#   It's the continuous background process that:
#
#   - Generates intuitions and insights without external input
#   - Manages the cognition cycle (perception -> thought -> response)
#   - Decides when to speak up vs stay quiet
#   - Tracks how many LLM calls are happening
#
#   Think of it as the stream of consciousness that keeps running
#   even when nobody is talking to the Noodling. It's why they
#   might suddenly say "I just realized something!" out of nowhere.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.agent_cognition
# PURPOSE:  Continuous cognition loop for Noodlings
# LAYER:    Backend / Agent
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CognitionLoopMixin    Continuous thought process mixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Agent Cognition Loop Mixin - Continuous cognition and intuition

Extracted from agent_bridge.py for maintainability.
Contains cognition loop methods (~450 lines):
- _generate_intuition: Deep intuition/insight generation via LLM
- _complete_cognition_cycle: Cycle state management
- _increment_llm_counter / _decrement_llm_counter: LLM call tracking
- start_cognition / stop_cognition: Loop control
- _continuous_cognition_loop: Main autonomous cognition loop

This is a mixin class - CMUSHNoodlingAgent inherits from it.
"""

from typing import Dict, Optional, List
import time
import asyncio
import logging

# Scene Protocol integration (optional - graceful fallback if not available)
try:
    from scene_protocol_integration import (
        SCENE_PROTOCOL_AVAILABLE,
        prepare_facet_context,
        finalize_facet_context,
    )
except ImportError:
    SCENE_PROTOCOL_AVAILABLE = False
    prepare_facet_context = None
    finalize_facet_context = None

logger = logging.getLogger(__name__)


class CognitionLoopMixin:
    """
    Mixin class providing cognition loop methods for CMUSHNoodlingAgent.

    This mixin expects the following attributes on self:
    - agent_id, agent_name: Agent identity
    - config: Agent configuration dict
    - llm: LLM interface
    - consciousness: Noodling charm instance
    - Various cycle tracking attributes (cycle_in_progress, etc.)
    - pending_perceptions: Queue of events to process

    And methods from other mixins:
    - perceive_event() from PerceptionMixin
    - _generate_rumination() from ResponseGenerationMixin
    """

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

                # Skip if facet executor not available
                if not self.facet_executor:
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
