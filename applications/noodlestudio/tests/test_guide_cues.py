# ──────────────────────────────────────────────────────────────
#   Tests for Guide Cue Handler
#
#   Tests for PADState, GuideCueState, and GuideCueHandler.
#   Verifies cue reception, prompt context generation, feedback
#   reporting, and emotional state drift.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import asyncio
import time
from unittest.mock import MagicMock
import pytest

from noodlestudio.runtime.channels import ChannelBus, ChannelMessage
from noodlestudio.runtime.guide_cue_handler import (
    PADState,
    GuideCueState,
    GuideCueHandler,
    CHANNEL_CUES,
    CHANNEL_FEEDBACK,
    CHANNEL_AMBIANCE,
)


# =============================================================================
# PADState Tests
# =============================================================================

class TestPADState:
    """Tests for PADState emotional model."""

    def test_default_values(self):
        """PADState starts with neutral values."""
        pad = PADState()
        assert pad.pleasure == 0.5
        assert pad.arousal == 0.5
        assert pad.dominance == 0.5

    def test_drift_toward_partial(self):
        """Drift only affects specified dimensions."""
        pad = PADState(pleasure=0.5, arousal=0.5, dominance=0.5)
        pad.drift_toward({'pleasure': 0.8}, rate=0.5)

        # Pleasure should have moved
        assert pad.pleasure > 0.5
        assert pad.pleasure < 0.8

        # Others should be unchanged
        assert pad.arousal == 0.5
        assert pad.dominance == 0.5

    def test_drift_toward_all(self):
        """Drift affects all specified dimensions."""
        pad = PADState(pleasure=0.5, arousal=0.5, dominance=0.5)
        pad.drift_toward({'pleasure': 0.8, 'arousal': 0.3, 'dominance': 0.9}, rate=0.5)

        assert pad.pleasure > 0.5  # Moved toward 0.8
        assert pad.arousal < 0.5   # Moved toward 0.3
        assert pad.dominance > 0.5 # Moved toward 0.9

    def test_drift_clamping_pleasure(self):
        """Pleasure is clamped to [-1, 1]."""
        pad = PADState(pleasure=0.9)
        pad.drift_toward({'pleasure': 2.0}, rate=1.0)
        assert pad.pleasure == 1.0

        pad = PADState(pleasure=-0.9)
        pad.drift_toward({'pleasure': -2.0}, rate=1.0)
        assert pad.pleasure == -1.0

    def test_drift_clamping_arousal(self):
        """Arousal is clamped to [0, 1]."""
        pad = PADState(arousal=0.9)
        pad.drift_toward({'arousal': 2.0}, rate=1.0)
        assert pad.arousal == 1.0

        pad = PADState(arousal=0.1)
        pad.drift_toward({'arousal': -0.5}, rate=1.0)
        assert pad.arousal == 0.0

    def test_drift_clamping_dominance(self):
        """Dominance is clamped to [0, 1]."""
        pad = PADState(dominance=0.9)
        pad.drift_toward({'dominance': 2.0}, rate=1.0)
        assert pad.dominance == 1.0

    def test_to_dict(self):
        """PADState serializes to dict."""
        pad = PADState(pleasure=0.7, arousal=0.3, dominance=0.5)
        d = pad.to_dict()

        assert 'pleasure' in d
        assert 'arousal' in d
        assert 'dominance' in d
        assert d['pleasure'] == 0.7
        assert d['arousal'] == 0.3
        assert d['dominance'] == 0.5

    def test_from_dict(self):
        """PADState deserializes from dict."""
        d = {'pleasure': 0.2, 'arousal': 0.8, 'dominance': 0.6}
        pad = PADState.from_dict(d)

        assert pad.pleasure == 0.2
        assert pad.arousal == 0.8
        assert pad.dominance == 0.6

    def test_from_dict_partial(self):
        """PADState defaults missing values."""
        d = {'pleasure': 0.3}
        pad = PADState.from_dict(d)

        assert pad.pleasure == 0.3
        assert pad.arousal == 0.5
        assert pad.dominance == 0.5

    def test_describe_happy_energized(self):
        """Describe returns appropriate descriptors for high pleasure/arousal."""
        pad = PADState(pleasure=0.8, arousal=0.8, dominance=0.5)
        desc = pad.describe()

        assert 'happy' in desc
        assert 'energized' in desc

    def test_describe_unhappy_calm(self):
        """Describe returns appropriate descriptors for low pleasure."""
        pad = PADState(pleasure=-0.5, arousal=0.2, dominance=0.5)
        desc = pad.describe()

        assert 'unhappy' in desc
        assert 'calm' in desc

    def test_describe_confident(self):
        """Describe returns confident for high dominance."""
        pad = PADState(pleasure=0.5, arousal=0.5, dominance=0.8)
        desc = pad.describe()

        assert 'confident' in desc

    def test_describe_uncertain(self):
        """Describe returns uncertain for low dominance."""
        pad = PADState(pleasure=0.5, arousal=0.5, dominance=0.2)
        desc = pad.describe()

        assert 'uncertain' in desc

    def test_describe_neutral(self):
        """Describe returns neutral for middle values."""
        # Values must be in the "neutral" range (not triggering any descriptor)
        # pleasure: 0.0 to 0.3 (not <-0.3 and not >0.3)
        # arousal: 0.3 to 0.5 (not <0.3 and not >0.5)
        # dominance: 0.3 to 0.5 (not <0.3 and not >0.5)
        pad = PADState(pleasure=0.1, arousal=0.4, dominance=0.4)
        desc = pad.describe()

        assert desc == "neutral"


# =============================================================================
# GuideCueHandler Tests
# =============================================================================

class TestGuideCueHandler:
    """Tests for GuideCueHandler."""

    def test_initialization(self):
        """Handler initializes with correct defaults."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        assert handler.noodling_id == "guide"
        assert handler.state.mode == "passive"
        assert handler.state.current_cue is None

    def test_cue_reception(self):
        """Handler receives and stores cues."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'test_beat',
                'beat_name': 'Test Beat',
                'target_actor': 'guide',
                'direction': 'Test direction',
                'motivation': 'Test motivation',
            }
        ))

        assert handler.state.current_beat_id == 'test_beat'
        assert handler.state.mode == 'active'
        assert handler.state.current_cue is not None

    def test_cue_filtering_by_target(self):
        """Handler ignores cues for other actors."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'test_beat',
                'target_actor': 'other_actor',
            }
        ))

        assert handler.state.current_cue is None
        assert handler.state.mode == 'passive'

    def test_cue_with_no_target_accepted(self):
        """Handler accepts cues with no target specified."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'broadcast_beat',
            }
        ))

        assert handler.state.current_beat_id == 'broadcast_beat'

    def test_emotional_drift_on_cue(self):
        """Handler drifts emotional state toward cue target."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        # Start with neutral
        assert handler.state.pad.pleasure == 0.5

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'happy_beat',
                'target_actor': 'guide',
                'emotional_target': {
                    'pleasure': 0.9,
                    'arousal': 0.7,
                }
            }
        ))

        # Should have drifted toward target
        assert handler.state.pad.pleasure > 0.5
        assert handler.state.pad.arousal > 0.5

    def test_improv_zone_setup(self):
        """Handler sets up improv zone from cue."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'improv_beat',
                'target_actor': 'guide',
                'improv_zone': {
                    'topics': ['weather', 'noodlings', 'consciousness'],
                    'duration': {'max_exchanges': 3}
                }
            }
        ))

        assert handler.state.mode == 'improv'
        assert handler.state.improv_max_exchanges == 3
        assert 'weather' in handler.state.improv_topics
        assert 'noodlings' in handler.state.improv_topics

    def test_prompt_context_no_cue(self):
        """Prompt context indicates passive mode with no cue."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        ctx = handler.get_prompt_context()

        assert ctx['has_direction'] is False
        assert ctx['mode'] == 'passive'
        assert 'emotional_state' in ctx

    def test_prompt_context_with_cue(self):
        """Prompt context includes direction from cue."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        # Set up cue manually
        handler.state.current_cue = {
            'beat_name': 'Test Beat',
            'direction': 'Do the thing',
            'motivation': 'Because reasons',
            'your_action': {
                'speaks': 'Hello there',
                'blocking': 'Wave hand',
            },
        }
        handler.state.mode = 'active'

        ctx = handler.get_prompt_context()

        assert ctx['has_direction'] is True
        assert ctx['direction'] == 'Do the thing'
        assert ctx['motivation'] == 'Because reasons'
        assert ctx['suggested_dialogue'] == 'Hello there'
        assert ctx['blocking'] == 'Wave hand'

    def test_prompt_context_improv_mode(self):
        """Prompt context includes improv info in improv mode."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        handler.state.current_cue = {'beat_name': 'Improv Beat'}
        handler.state.mode = 'improv'
        handler.state.improv_topics = ['topic1', 'topic2']
        handler.state.improv_exchanges = 1
        handler.state.improv_max_exchanges = 3

        ctx = handler.get_prompt_context()

        assert ctx['mode'] == 'improv'
        assert ctx['improv_topics'] == ['topic1', 'topic2']
        assert ctx['improv_exchanges'] == 1
        assert ctx['improv_max_exchanges'] == 3

    def test_build_system_prompt_passive(self):
        """System prompt addition for passive mode."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        prompt = handler.build_system_prompt_addition()

        assert "passive mode" in prompt.lower()
        assert "be yourself" in prompt.lower()

    def test_build_system_prompt_active(self):
        """System prompt addition includes direction."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        handler.state.current_cue = {
            'beat_name': 'Intro',
            'direction': 'Welcome the visitor',
            'motivation': 'Genuinely happy to help',
            'your_action': {'speaks': 'Hey there!'},
        }
        handler.state.mode = 'active'

        prompt = handler.build_system_prompt_addition()

        assert "Director's Notes" in prompt
        assert "Intro" in prompt
        assert "Welcome the visitor" in prompt
        assert "Genuinely happy to help" in prompt
        assert "Hey there!" in prompt

    def test_feedback_reporting(self):
        """Handler reports feedback to Brenda."""
        bus = ChannelBus()
        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        handler = GuideCueHandler(bus, "guide")
        handler.state.current_beat_id = "test_beat"

        handler.report_response("My response", "User asked something?")

        assert len(received) == 1
        payload = received[0].payload
        assert payload['actor_id'] == 'guide'
        assert payload['beat_id'] == 'test_beat'
        assert payload['status'] in ['completed', 'in_progress']
        assert 'emotional_state' in payload

    def test_feedback_clears_cue_on_complete(self):
        """Completed beat clears current cue."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        handler.state.current_cue = {'beat_id': 'test'}
        handler.state.current_beat_id = 'test'
        handler.state.mode = 'active'

        handler.report_response("Done", "Thanks")

        assert handler.state.current_cue is None
        assert handler.state.mode == 'passive'

    def test_feedback_improv_not_completed_early(self):
        """Improv zone doesn't complete until exchanges reached."""
        bus = ChannelBus()
        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {'beat_id': 'improv'}
        handler.state.current_beat_id = 'improv'
        handler.state.mode = 'improv'
        handler.state.improv_exchanges = 0
        handler.state.improv_max_exchanges = 3

        handler.report_response("First response", "First question?")

        assert received[0].payload['status'] == 'in_progress'
        assert handler.state.mode == 'improv'  # Still in improv

    def test_feedback_improv_completes_at_max(self):
        """Improv zone completes when max exchanges reached."""
        bus = ChannelBus()
        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {'beat_id': 'improv'}
        handler.state.current_beat_id = 'improv'
        handler.state.mode = 'improv'
        handler.state.improv_exchanges = 2
        handler.state.improv_max_exchanges = 3

        handler.report_response("Third response", "Third question?")

        assert received[0].payload['status'] == 'completed'
        assert handler.state.mode == 'passive'

    def test_emotion_adjustment_positive(self):
        """Positive user feedback increases pleasure."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {}
        initial = handler.state.pad.pleasure

        handler.report_response("Thanks for asking!", "Thanks that was great!")

        assert handler.state.pad.pleasure > initial

    def test_emotion_adjustment_confusion(self):
        """User confusion decreases dominance."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {}
        initial = handler.state.pad.dominance

        handler.report_response("Let me clarify...", "I don't understand what you mean")

        assert handler.state.pad.dominance < initial

    def test_ambiance_affects_state(self):
        """World ambiance influences emotional state."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        initial_arousal = handler.state.pad.arousal

        bus.publish(CHANNEL_AMBIANCE, ChannelMessage(
            channel=CHANNEL_AMBIANCE,
            from_noodling="system",
            timestamp=time.time(),
            payload={'mood': 'tense', 'energy': 0.8}
        ))

        assert handler.state.pad.arousal > initial_arousal

    def test_ambiance_joyful(self):
        """Joyful ambiance increases pleasure."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        initial_pleasure = handler.state.pad.pleasure

        bus.publish(CHANNEL_AMBIANCE, ChannelMessage(
            channel=CHANNEL_AMBIANCE,
            from_noodling="system",
            timestamp=time.time(),
            payload={'mood': 'joyful'}
        ))

        assert handler.state.pad.pleasure > initial_pleasure

    def test_mode_helpers(self):
        """Mode helper methods work correctly."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        assert handler.is_expecting_cue() is True
        assert handler.has_active_direction() is False
        assert handler.get_mode() == 'passive'

        handler.state.current_cue = {'beat_id': 'test'}
        handler.state.mode = 'active'

        assert handler.is_expecting_cue() is False
        assert handler.has_active_direction() is True
        assert handler.get_mode() == 'active'

    def test_enter_passive_mode(self):
        """enter_passive_mode clears active direction."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        handler.state.current_cue = {'beat_id': 'test'}
        handler.state.mode = 'active'

        handler.enter_passive_mode()

        assert handler.state.mode == 'passive'
        assert handler.state.current_cue is None

    def test_callback_on_cue_received(self):
        """Callback fires when cue received."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        received_cues = []
        handler.on_cue_received(lambda cue: received_cues.append(cue))

        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={'beat_id': 'callback_test', 'target_actor': 'guide'}
        ))

        assert len(received_cues) == 1
        assert received_cues[0]['beat_id'] == 'callback_test'

    def test_callback_on_mode_change(self):
        """Callback fires when mode changes."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        mode_changes = []
        handler.on_mode_change(lambda mode: mode_changes.append(mode))

        handler.state.mode = 'active'
        handler.state.current_cue = {'beat_id': 'test'}

        handler.enter_passive_mode()

        assert 'passive' in mode_changes

    def test_get_state_dict(self):
        """get_state_dict returns current state."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        state = handler.get_state_dict()

        assert state['noodling_id'] == 'guide'
        assert state['mode'] == 'passive'
        assert 'pad' in state
        assert 'emotional_state' in state

    def test_notes_generation_question(self):
        """Notes include question detection."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {}

        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        handler.report_response("Response", "What is that?")

        assert "asked a question" in received[0].payload['notes'].lower()

    def test_notes_generation_detailed(self):
        """Notes include detailed response detection."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {}

        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        long_message = "a" * 150
        handler.report_response("Response", long_message)

        assert "detailed" in received[0].payload['notes'].lower()

    def test_notes_generation_brief(self):
        """Notes include brief response detection."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")
        handler.state.current_cue = {}

        received = []
        bus.subscribe(CHANNEL_FEEDBACK, lambda m: received.append(m))

        handler.report_response("Response", "ok")

        assert "brief" in received[0].payload['notes'].lower()


# =============================================================================
# Computer Use Execution Tests
# =============================================================================

class TestComputerUseCueExecution:
    """Tests for computer_use action execution triggered by cue reception."""

    def test_cue_with_computer_use_triggers_execution(self):
        """Cue with computer_use actions schedules execution on event loop."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        # Mock controller with UI element map
        controller = MagicMock()
        controller.get_ui_element_map.return_value = [
            {'name': 'Button: View Project', 'x': 100, 'y': 200},
        ]
        handler.set_computer_use_controller(controller)

        async def run_test():
            # Publish cue with computer_use actions
            bus.publish(CHANNEL_CUES, ChannelMessage(
                channel=CHANNEL_CUES,
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'cue',
                    'beat_id': 'demo_click',
                    'beat_name': 'Demo Click',
                    'target_actor': 'guide',
                    'direction': 'Click the view project button',
                    'your_action': {
                        'computer_use': [
                            {'action': 'move', 'target': 'Button: View Project'},
                            {'action': 'click', 'target': 'Button: View Project'},
                        ]
                    }
                }
            ))

            # Let the scheduled task run
            await asyncio.sleep(0)

            # Verify controller methods were called
            controller.mouse_move.assert_called_once_with(100, 200)
            controller.click.assert_called_once_with(100, 200, 'left')

        asyncio.run(run_test())

    def test_cue_without_computer_use_no_execution(self):
        """Cue without computer_use does not trigger controller."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        controller = MagicMock()
        handler.set_computer_use_controller(controller)

        async def run_test():
            bus.publish(CHANNEL_CUES, ChannelMessage(
                channel=CHANNEL_CUES,
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'cue',
                    'beat_id': 'speech_only',
                    'beat_name': 'Speech Only',
                    'target_actor': 'guide',
                    'your_action': {
                        'speaks': 'Hello there!',
                    }
                }
            ))

            await asyncio.sleep(0)

            controller.mouse_move.assert_not_called()
            controller.click.assert_not_called()

        asyncio.run(run_test())

    def test_cue_computer_use_no_controller_no_error(self):
        """Cue with computer_use but no controller does not raise."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        # No controller set - should not raise
        async def run_test():
            bus.publish(CHANNEL_CUES, ChannelMessage(
                channel=CHANNEL_CUES,
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'cue',
                    'beat_id': 'demo_click',
                    'target_actor': 'guide',
                    'your_action': {
                        'computer_use': [
                            {'action': 'click', 'target': 'Button: View Project'},
                        ]
                    }
                }
            ))

            await asyncio.sleep(0)

        asyncio.run(run_test())

        # Handler should have stored the cue
        assert handler.state.current_beat_id == 'demo_click'

    def test_cue_computer_use_no_event_loop(self):
        """Cue with computer_use outside event loop logs debug, no crash."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        controller = MagicMock()
        handler.set_computer_use_controller(controller)

        # Publish outside an event loop - should gracefully handle RuntimeError
        bus.publish(CHANNEL_CUES, ChannelMessage(
            channel=CHANNEL_CUES,
            from_noodling="brenda",
            timestamp=time.time(),
            payload={
                'type': 'cue',
                'beat_id': 'demo_click',
                'target_actor': 'guide',
                'your_action': {
                    'computer_use': [
                        {'action': 'click', 'target': 'Button: View Project'},
                    ]
                }
            }
        ))

        # Cue was still stored despite no event loop
        assert handler.state.current_beat_id == 'demo_click'

    def test_cue_computer_use_type_action(self):
        """Cue with type action executes type_text on controller."""
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        controller = MagicMock()
        controller.get_ui_element_map.return_value = []
        handler.set_computer_use_controller(controller)

        async def run_test():
            bus.publish(CHANNEL_CUES, ChannelMessage(
                channel=CHANNEL_CUES,
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'cue',
                    'beat_id': 'type_test',
                    'target_actor': 'guide',
                    'your_action': {
                        'computer_use': [
                            {'action': 'type', 'text': 'Hello world'},
                        ]
                    }
                }
            ))

            await asyncio.sleep(0)

            controller.type_text.assert_called_once_with('Hello world')

        asyncio.run(run_test())
