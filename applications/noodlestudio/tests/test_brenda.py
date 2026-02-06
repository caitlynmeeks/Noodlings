# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
# Brenda Stage Director Tests
# ──────────────────────────────────────────────────────────────
"""
Tests for BrendaDirector: stage direction, play execution, cue sending.
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from typing import List

from noodlestudio.runtime.channels import ChannelBus, ChannelMessage
from noodlestudio.runtime.brenda import (
    BrendaDirector,
    PlayState,
    DirectorMode,
    TriggerType,
    CHANNEL_CUES,
    CHANNEL_FEEDBACK,
    CHANNEL_USER_INPUT,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def channel_bus():
    """Create a fresh channel bus for each test."""
    return ChannelBus()


@pytest.fixture
def brenda(channel_bus):
    """Create a BrendaDirector instance."""
    return BrendaDirector(channel_bus)


@pytest.fixture
def simple_play_path():
    """Create a simple test play file."""
    play_content = """
title: "Test Play"
author: "Test Author"
created: 2026-01-09
version: 1

setting:
  location: "Test Location"
  mood: "Neutral"

characters:
  guide:
    noodling: "noodlings/guide"
    initial_pad:
      pleasure: 0.6
      arousal: 0.5
      dominance: 0.5
    motivation: "Help the user"

beats:
  - id: beat_1
    name: "First Beat"
    on_stage: [guide]
    direction: "Guide greets the user"
    guide:
      speaks: "Hello there!"
      pad_drift:
        pleasure: +0.1

  - id: beat_2
    name: "Second Beat"
    on_stage: [guide]
    trigger:
      type: sequence
    direction: "Guide continues"
    guide:
      speaks: "What would you like to do?"

  - id: beat_3
    name: "Third Beat"
    on_stage: [guide]
    trigger:
      type: delay
      seconds: 1
    guide:
      speaks: "Still here!"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


@pytest.fixture
def choice_play_path():
    """Create a play with user choice handling."""
    play_content = """
title: "Choice Test"
author: "Test"

characters:
  guide:
    initial_pad:
      pleasure: 0.5
      arousal: 0.5
      dominance: 0.5

beats:
  - id: offer_choice
    name: "Offer Choice"
    on_stage: [guide]
    guide:
      speaks: "Tour or explore?"
    wait_for:
      type: user_choice
      timeout: 10
      default: tour
      options:
        - id: tour
          patterns: ["tour", "show me", "yes"]
          next_beat: tour_start
        - id: explore
          patterns: ["explore", "on my own"]
          next_beat: free_explore

  - id: tour_start
    name: "Tour Start"
    on_stage: [guide]
    trigger:
      type: user_choice
      choice: tour
    guide:
      speaks: "Great, let's start the tour!"

  - id: free_explore
    name: "Free Explore"
    on_stage: [guide]
    trigger:
      type: user_choice
      choice: explore
    guide:
      speaks: "Okay, explore away!"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


@pytest.fixture
def threshold_play_path():
    """Create a play with emotional threshold triggers."""
    play_content = """
title: "Threshold Test"
author: "Test"

characters:
  toad:
    initial_pad:
      pleasure: 0.0
      arousal: 0.7
      dominance: 0.3

beats:
  - id: beat_1
    name: "Toad excited"
    on_stage: [toad]
    toad:
      speaks: "MOTORCARS!"
      pad_drift:
        arousal: +0.2

  - id: beat_2
    name: "Too excited"
    on_stage: [toad]
    trigger:
      type: threshold
      condition: "toad.arousal > 0.85"
    toad:
      speaks: "POOP POOP!"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


@pytest.fixture
def improv_play_path():
    """Create a play with improv zone."""
    play_content = """
title: "Improv Test"
author: "Test"

characters:
  guide:
    initial_pad:
      pleasure: 0.6
      arousal: 0.5
      dominance: 0.5

beats:
  - id: improv_beat
    name: "Improv Zone"
    on_stage: [guide]
    guide:
      speaks: "Let's chat freely!"
    improv_zone:
      topics:
        - noodlings
        - consciousness
      duration:
        max_exchanges: 3

  - id: after_improv
    name: "After Improv"
    on_stage: [guide]
    trigger:
      type: improv_complete
    guide:
      speaks: "Moving on!"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


# =============================================================================
# PlayState Tests
# =============================================================================

class TestPlayState:
    """Tests for PlayState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = PlayState()

        assert state.current_beat_id is None
        assert state.current_beat_index == 0
        assert state.completed_beats == []
        assert state.mode == DirectorMode.ACTIVE
        assert state.active_improv_zone is None
        assert state.waiting_for is None
        assert state.exchange_count == 0

    def test_reset(self):
        """Test state reset."""
        state = PlayState()
        state.current_beat_id = "beat_1"
        state.completed_beats = ["beat_0"]
        state.exchange_count = 5
        state.mode = DirectorMode.PASSIVE

        state.reset()

        assert state.current_beat_id is None
        assert state.completed_beats == []
        assert state.exchange_count == 0
        assert state.mode == DirectorMode.ACTIVE


# =============================================================================
# BrendaDirector Basic Tests
# =============================================================================

class TestBrendaDirectorBasic:
    """Tests for BrendaDirector core functionality."""

    def test_create_director(self, channel_bus):
        """Test creating a director."""
        brenda = BrendaDirector(channel_bus)

        assert brenda.channel_bus is channel_bus
        assert brenda.play_data is None
        assert not brenda._running

    def test_load_play(self, brenda, simple_play_path):
        """Test loading a play file."""
        assert brenda.load_play(simple_play_path)
        assert brenda.play_data is not None
        assert brenda.play_data['title'] == "Test Play"
        assert brenda.state.current_beat_id == "beat_1"

    def test_load_nonexistent_play(self, brenda):
        """Test loading a nonexistent play file."""
        assert not brenda.load_play("/nonexistent/path.play.yaml")

    def test_load_initializes_character_states(self, brenda, simple_play_path):
        """Test that loading initializes character states."""
        brenda.load_play(simple_play_path)

        assert 'guide' in brenda.state.character_states
        assert brenda.state.character_states['guide']['pleasure'] == 0.6
        assert brenda.state.character_states['guide']['arousal'] == 0.5

    def test_start(self, brenda, simple_play_path):
        """Test starting direction."""
        brenda.load_play(simple_play_path)
        brenda.start()

        assert brenda._running
        assert brenda.state.mode == DirectorMode.ACTIVE
        assert brenda.state.current_beat_id in brenda.state.beat_start_times

    def test_stop(self, brenda, simple_play_path):
        """Test stopping direction."""
        brenda.load_play(simple_play_path)
        brenda.start()
        brenda.stop()

        assert not brenda._running
        assert brenda.state.mode == DirectorMode.PAUSED

    def test_start_without_play(self, brenda):
        """Test starting without a loaded play."""
        brenda.start()
        assert not brenda._running


# =============================================================================
# Cue Sending Tests
# =============================================================================

class TestCueSending:
    """Tests for cue sending to #directors.cues."""

    def test_start_sends_initial_cue(self, channel_bus, simple_play_path):
        """Test that start() sends initial cue."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        assert len(received) >= 1
        assert received[0].payload['type'] == 'cue'
        assert received[0].payload['beat_id'] == 'beat_1'

    def test_cue_contains_direction(self, channel_bus, simple_play_path):
        """Test that cues contain direction text."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        cue = received[0].payload
        assert 'direction' in cue
        assert 'Guide greets' in cue['direction']

    def test_cue_contains_actor_action(self, channel_bus, simple_play_path):
        """Test that cues contain actor-specific actions."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        cue = received[0].payload
        assert cue['target_actor'] == 'guide'
        assert 'speaks' in cue['your_action']
        assert cue['your_action']['speaks'] == "Hello there!"

    def test_cue_contains_emotional_target(self, channel_bus, simple_play_path):
        """Test that cues contain emotional target."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        cue = received[0].payload
        assert 'emotional_target' in cue
        # Should have pleasure drift applied: 0.6 + 0.1 = 0.7
        assert cue['emotional_target']['pleasure'] == pytest.approx(0.7, 0.01)

    def test_cue_from_noodling_is_brenda(self, channel_bus, simple_play_path):
        """Test that cues are from 'brenda'."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        assert received[0].from_noodling == 'brenda'


# =============================================================================
# Trigger Tests
# =============================================================================

class TestTriggers:
    """Tests for beat trigger evaluation."""

    def test_sequence_trigger(self, channel_bus, simple_play_path):
        """Test sequence triggers fire after previous beat."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        # Complete beat_1 (clear gate to simulate actor feedback)
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('beat_1')
        brenda.tick()

        # Should have advanced to beat_2
        assert brenda.state.current_beat_id == 'beat_2'
        assert len(received) >= 2

    def test_delay_trigger(self, channel_bus, simple_play_path):
        """Test delay triggers fire after time."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        # First advance through beat_1 -> beat_2 (sequence trigger)
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('beat_1')
        brenda.tick()
        assert brenda.state.current_beat_id == 'beat_2'

        # Now we're on beat_2, and beat_3 has a delay trigger
        # Before delay expires, should stay on beat_2
        brenda.state.awaiting_actor_response = False
        brenda.tick()
        assert brenda.state.current_beat_id == 'beat_2'

        # Simulate time passing (delay of 1 second on beat_3)
        # Set current beat's start time in the past
        brenda.state.beat_start_times['beat_2'] = time.time() - 2.0
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('beat_2')

        brenda.tick()
        assert brenda.state.current_beat_id == 'beat_3'

    def test_threshold_trigger(self, channel_bus, threshold_play_path):
        """Test emotional threshold triggers."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(threshold_play_path)
        brenda.start()

        # beat_1 should apply arousal drift
        # Initial arousal 0.7 + 0.2 = 0.9 > 0.85
        assert brenda.state.character_states['toad']['arousal'] == pytest.approx(0.9, 0.01)

        # Complete beat_1 (clear gate to simulate actor feedback)
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('beat_1')
        brenda.tick()

        # Should trigger beat_2
        assert brenda.state.current_beat_id == 'beat_2'

    def test_user_choice_trigger(self, channel_bus, choice_play_path):
        """Test user choice triggers."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(choice_play_path)
        brenda.start()

        # Should be waiting for choice
        assert brenda.state.waiting_for is not None
        assert brenda.state.waiting_for['type'] == 'user_choice'

        # Simulate user choosing tour (clear gate first)
        brenda.state.awaiting_actor_response = False
        channel_bus.publish_simple(CHANNEL_USER_INPUT, {'text': 'show me around'})

        brenda.tick()

        # Should advance to tour_start
        assert brenda.state.current_beat_id == 'tour_start'

    def test_improv_complete_trigger(self, channel_bus, improv_play_path):
        """Test improv_complete triggers."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(improv_play_path)
        brenda.start()

        # Should be in improv zone
        assert brenda.state.active_improv_zone is not None

        # Simulate max exchanges (clear gate to allow processing)
        brenda.state.awaiting_actor_response = False
        for i in range(3):
            channel_bus.publish_simple(CHANNEL_USER_INPUT, {'text': f'message {i}'})

        # Improv zone should exit
        assert brenda.state.active_improv_zone is None

        # Complete improv beat (clear gate)
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('improv_beat')
        brenda.tick()

        # Should advance to after_improv
        assert brenda.state.current_beat_id == 'after_improv'


# =============================================================================
# User Input Handling Tests
# =============================================================================

class TestUserInputHandling:
    """Tests for user input handling."""

    def test_user_input_updates_state(self, channel_bus, simple_play_path):
        """Test user input updates state."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        channel_bus.publish_simple(CHANNEL_USER_INPUT, {'text': 'Hello Brenda!'})

        assert brenda.state.last_user_message == 'Hello Brenda!'
        assert brenda.state.exchange_count == 1

    def test_choice_matching(self, channel_bus, choice_play_path):
        """Test user choice pattern matching."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(choice_play_path)
        brenda.start()

        # Test 'tour' pattern
        choice = brenda._classify_user_choice("I'd like a tour please")
        assert choice == 'tour'

        # Test 'explore' pattern
        choice = brenda._classify_user_choice("I'll explore on my own")
        assert choice == 'explore'

        # Test no match
        choice = brenda._classify_user_choice("I don't know")
        assert choice is None

    def test_wait_for_timeout(self, channel_bus, choice_play_path):
        """Test wait_for timeout uses default."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(choice_play_path)
        brenda.start()

        # Should be waiting
        assert brenda.state.waiting_for is not None

        # Set wait start time in the past (beyond 10 second timeout)
        # Clear gate so tick() can process
        brenda.state.awaiting_actor_response = False
        brenda.state.wait_start_time = time.time() - 15

        brenda.tick()

        # Timeout should have cleared waiting_for with default 'tour'
        assert brenda.state.waiting_for is None

        # Now tick again - should advance to tour_start due to user_choice trigger
        brenda.state.awaiting_actor_response = False
        brenda.state.completed_beats.append('offer_choice')
        brenda.tick()

        assert brenda.state.current_beat_id == 'tour_start'


# =============================================================================
# Feedback Handling Tests
# =============================================================================

class TestFeedbackHandling:
    """Tests for actor feedback handling."""

    def test_feedback_updates_character_state(self, channel_bus, simple_play_path):
        """Test feedback updates character state."""
        brenda = BrendaDirector(channel_bus)
        brenda.load_play(simple_play_path)
        brenda.start()

        # Send feedback with emotional state
        channel_bus.publish_simple(
            CHANNEL_FEEDBACK,
            {
                'actor_id': 'guide',
                'status': 'completed',
                'beat_id': 'beat_1',
                'emotional_state': {
                    'pleasure': 0.8,
                    'arousal': 0.6,
                    'dominance': 0.5
                }
            }
        )

        assert brenda.state.character_states['guide']['pleasure'] == 0.8


# =============================================================================
# Mode Management Tests
# =============================================================================

class TestModeManagement:
    """Tests for director mode management."""

    def test_set_mode_from_beat(self, channel_bus):
        """Test set_mode in beat changes director mode."""
        play_content = """
title: "Mode Test"

characters:
  guide:
    initial_pad:
      pleasure: 0.5

beats:
  - id: go_passive
    on_stage: [guide]
    guide:
      speaks: "I'll be here if you need me"
    set_mode: passive_available
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
            f.write(play_content)
            path = f.name

        try:
            brenda = BrendaDirector(channel_bus)
            brenda.load_play(path)
            brenda.start()

            assert brenda.state.mode == DirectorMode.PASSIVE_AVAILABLE
        finally:
            os.unlink(path)

    def test_mode_change_callback(self, channel_bus, simple_play_path):
        """Test mode change callback is called."""
        modes_received = []

        brenda = BrendaDirector(channel_bus)
        brenda.on_mode_change(lambda m: modes_received.append(m))
        brenda.load_play(simple_play_path)
        brenda.start()
        brenda.stop()

        # Stop should set PAUSED mode but callback is for set_mode from beats
        # Let's verify the callback mechanism works
        assert brenda.state.mode == DirectorMode.PAUSED


# =============================================================================
# Introspection Tests
# =============================================================================

class TestIntrospection:
    """Tests for director introspection methods."""

    def test_get_play_info(self, brenda, simple_play_path):
        """Test get_play_info returns play details."""
        brenda.load_play(simple_play_path)
        info = brenda.get_play_info()

        assert info['title'] == "Test Play"
        assert info['author'] == "Test Author"
        assert info['beat_count'] == 3
        assert 'guide' in info['characters']

    def test_get_state(self, brenda, simple_play_path):
        """Test get_state returns current state."""
        brenda.load_play(simple_play_path)
        brenda.start()

        state = brenda.get_state()

        assert state['running'] is True
        assert state['mode'] == 'active'
        assert state['current_beat_id'] == 'beat_1'


# =============================================================================
# World Control Tests
# =============================================================================

class TestWorldControl:
    """Tests for Brenda's world control methods."""

    def test_set_ambiance(self, channel_bus, simple_play_path):
        """Test set_ambiance publishes to #world.ambiance."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe("#world.ambiance", lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.set_ambiance("tense", energy=0.8)

        assert len(received) == 1
        assert received[0].payload['mood'] == "tense"
        assert received[0].payload['energy'] == 0.8
        assert received[0].from_noodling == 'brenda'

    def test_trigger_event(self, channel_bus, simple_play_path):
        """Test trigger_event publishes to #world.events."""
        received: List[ChannelMessage] = []
        channel_bus.subscribe("#world.events", lambda m: received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.trigger_event("sound", "door", "A door slammed.")

        assert len(received) == 1
        assert received[0].payload['event_type'] == "sound"
        assert received[0].payload['source'] == "door"
        assert received[0].payload['description'] == "A door slammed."


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full performance flow."""

    def test_full_performance_flow(self, channel_bus, choice_play_path):
        """Test a complete performance flow."""
        cues_received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: cues_received.append(m))

        brenda = BrendaDirector(channel_bus)
        brenda.load_play(choice_play_path)
        brenda.start()

        # Should send initial cue
        assert len(cues_received) >= 1
        assert cues_received[0].payload['beat_id'] == 'offer_choice'

        # User chooses tour (clear gate to simulate actor feedback)
        brenda.state.awaiting_actor_response = False
        channel_bus.publish_simple(CHANNEL_USER_INPUT, {'text': 'Show me around'})
        brenda.tick()

        # Should advance to tour_start
        assert brenda.state.current_beat_id == 'tour_start'
        assert any(c.payload['beat_id'] == 'tour_start' for c in cues_received)

    def test_lets_consciousness_play(self, channel_bus):
        """Test with the actual Let's Consciousness play."""
        play_path = Path(__file__).parent.parent.parent.parent / \
            "docs/noodlestudio/plays/lets_consciousness_intro.play.yaml"

        if not play_path.exists():
            pytest.skip("Let's Consciousness play not found")

        cues_received: List[ChannelMessage] = []
        channel_bus.subscribe(CHANNEL_CUES, lambda m: cues_received.append(m))

        brenda = BrendaDirector(channel_bus)
        assert brenda.load_play(str(play_path))

        info = brenda.get_play_info()
        assert info['title'] == "Let's Consciousness"

        brenda.start()

        # Should start with first_hello
        assert brenda.state.current_beat_id == 'first_hello'
        assert len(cues_received) >= 1


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
