"""
Agent Perception Mixin - Event perception and cognitive processing

Extracted from agent_bridge.py for maintainability.
Contains the main perceive_event method (~1150 lines) which handles:
- Event validation and cycle locking
- Affect extraction from text
- Facet assembly execution
- Social routing decisions
- Response generation coordination

This is a mixin class - CMUSHNoodlingAgent inherits from it.
"""

from typing import Dict, Optional
import time
import logging

from performance_tracker import get_tracker

logger = logging.getLogger(__name__)


class PerceptionMixin:
    """
    Mixin class providing perception methods for CMUSHNoodlingAgent.

    This mixin expects the following attributes on self:
    - agent_id, agent_name, species: Agent identity
    - config: Agent configuration dict
    - llm: LLM interface for affect extraction
    - consciousness: Noodling charm instance
    - facet_assembly, facet_executor: Facet system
    - conversation_context: Recent conversation history
    - world, world_state: World reference
    - current_room: Current room ID
    - Various cycle tracking attributes (cycle_in_progress, etc.)

    And the following methods:
    - _complete_cognition_cycle()
    - _normalize_affect()
    - _trigger_memories_by_names()
    - _apply_memory_affect()
    - _detect_emotional_contagion()
    - _generate_response()
    """

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

        # IMMEDIATELY clear old cycle data and set new input
        self.last_perception_text = text  # Set input FIRST
        self.last_manifold_output = None  # Clear old output

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

            # 1a-1. COGNITIVE PROCESSING via FACET SYSTEM
            # Process perception through facet assembly
            colored_perception = text  # Default to original text

            logger.info(f"[{self.agent_id}] DEBUG: About to enter cognitive processing via facets")

            try:
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

            except Exception as e:
                logger.error(f"[{self.agent_id}] Cognition failed: {e}")
                import traceback
                traceback.print_exc()
                colored_perception = text  # Fallback to original

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
                'text': colored_perception,
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

            # Fire OnSurpriseSpike event to scripts on high surprise
            surprise_threshold = state.get('surprise_threshold', self.config.get('surprise_threshold', 0.3))
            if state['surprise'] > surprise_threshold * 1.5:
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

            # SOCIAL ROUTER OVERRIDE: Use Social Router decision if available
            # Social Router provides sophisticated social context detection that supersedes simple name matching
            if 'social_router_says_respond' in locals():
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
                # Social Router already decided, ignore cooldown
                if is_being_addressed:
                    # Social Router already vetted the response, always speak
                    should_speak = True
                    logger.info(f"[{self.agent_id}] ✅ ADDRESSED → should_speak=True (ignoring cooldown)")
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
                # Pass facet assembly output (colored_perception from facet execution)
                facet_response = colored_perception if colored_perception != text else None
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

