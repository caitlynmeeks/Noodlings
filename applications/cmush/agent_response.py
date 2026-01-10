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
#   Agent Response Generation
#
#   This is how Noodlings turn their thoughts into words. After
#   perception and cognition have processed an event, this module
#   generates what the Noodling actually says or does.
#
#   The process:
#   1. Check conscience (is this response appropriate?)
#   2. Gather context (conversation history, affect state)
#   3. Call the LLM to generate natural language
#   4. Post-process the output (clean up, format)
#
#   It also handles "rumination" - the Noodling's internal
#   monologue when thinking to themselves.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.agent_response
# PURPOSE:  Generate natural language responses
# LAYER:    Backend / Agent
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ResponseGenerationMixin    LLM-powered response generation
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Agent Response Generation Mixin - LLM response generation

Extracted from agent_bridge.py for maintainability.
Contains response generation methods (~690 lines):
- generate_response_text: Main entry point for generating agent speech
- _check_conscience: Toxicity checking before broadcast
- _generate_response: Core response generation with LLM
- _generate_rumination: Internal thought/rumination generation

This is a mixin class - CMUSHNoodlingAgent inherits from it.
"""

from typing import Dict, Optional, List
import time
import logging
import re

logger = logging.getLogger(__name__)


class ResponseGenerationMixin:
    """
    Mixin class providing response generation methods for CMUSHNoodlingAgent.

    This mixin expects the following attributes on self:
    - agent_id, agent_name, species: Agent identity
    - config: Agent configuration dict
    - llm: LLM interface
    - identity_prompt: Agent's identity/personality prompt
    - conversation_context: Recent conversation history
    - consciousness: Noodling charm instance
    - Various state tracking attributes

    And the following methods from other mixins/base class:
    - _score_identity_salience()
    - _detects_invitation()
    - Various helper methods
    """

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
            return None

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
            # Get recent context for rumination
            recent_context = self.conversation_context[-2:] if len(self.conversation_context) >= 2 else []
            perception_text = recent_context[-1].get('text', '') if recent_context else "observing surroundings"

            # Generate internal thought via LLM
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
            return None

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
