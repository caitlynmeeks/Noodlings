# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
# Play Integration Tests
# ──────────────────────────────────────────────────────────────
"""
Tests for Play execution integration:
- NoodleApp + BrendaDirector + GuideCueHandler wiring
- Direction injection into facet execution
"""

import asyncio
from unittest.mock import MagicMock
import pytest
import tempfile
import os
from pathlib import Path

from noodlestudio.runtime.app import NoodleApp
from noodlestudio.runtime.channels import ChannelBus
from noodlestudio.runtime.brenda import CHANNEL_CUES


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_play_path():
    """Create a simple test play file."""
    play_content = """
title: "Integration Test Play"
author: "Test"

characters:
  guide:
    initial_pad:
      pleasure: 0.6
      arousal: 0.5
      dominance: 0.5
    motivation: "Help the user learn"

beats:
  - id: welcome
    name: "Welcome"
    on_stage: [guide]
    direction: "Greet the user warmly"
    guide:
      speaks: "Welcome! I'm your guide."
      pad_drift:
        pleasure: +0.1
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


# =============================================================================
# NoodleApp + Director Integration Tests
# =============================================================================

class TestNoodleAppDirectorIntegration:
    """Tests for NoodleApp and BrendaDirector integration."""

    def test_load_director_creates_guide_handler(self, simple_play_path):
        """Test that load_director creates GuideCueHandler."""
        app = NoodleApp()

        # Before loading, no handler
        assert app.guide_cue_handler is None

        # Load play
        result = app.load_director(simple_play_path)
        assert result is True

        # After loading, handler exists
        assert app.guide_cue_handler is not None
        assert app.guide_cue_handler.noodling_id == "guide"

    def test_load_director_custom_actor_id(self, simple_play_path):
        """Test that load_director respects custom actor_id."""
        app = NoodleApp()

        app.load_director(simple_play_path, actor_id="toad")

        assert app.guide_cue_handler is not None
        assert app.guide_cue_handler.noodling_id == "toad"

    def test_start_performance_sends_cue_to_handler(self, simple_play_path):
        """Test that start_performance sends cue to GuideCueHandler."""
        app = NoodleApp()
        app.load_director(simple_play_path)

        # Handler should have no cue before start
        assert app.guide_cue_handler.state.current_cue is None

        # Start performance
        app.start_performance()

        # Handler should now have the cue
        assert app.guide_cue_handler.state.current_cue is not None
        assert app.guide_cue_handler.state.current_beat_id == "welcome"

    def test_get_brenda_direction_returns_direction(self, simple_play_path):
        """Test that get_brenda_direction returns direction text."""
        app = NoodleApp()
        app.load_director(simple_play_path)
        app.start_performance()

        direction = app.get_brenda_direction()

        # Should contain the beat name and direction
        assert "Director's Notes" in direction
        assert "welcome" in direction.lower() or "Welcome" in direction
        assert "Greet the user warmly" in direction

    def test_get_brenda_direction_empty_without_play(self):
        """Test that get_brenda_direction returns empty without play."""
        app = NoodleApp()

        direction = app.get_brenda_direction()

        assert direction == ""

    def test_cleanup_clears_handler(self, simple_play_path):
        """Test that cleanup clears the guide_cue_handler."""
        import asyncio

        app = NoodleApp()
        app.load_director(simple_play_path)

        # Handler exists
        assert app.guide_cue_handler is not None

        # Cleanup (use asyncio.run for Python 3.10+)
        asyncio.run(app.cleanup())

        # Handler cleared
        assert app.guide_cue_handler is None


# =============================================================================
# Direction Injection Tests
# =============================================================================

class TestDirectionInjection:
    """Tests for direction injection into execution context."""

    def test_direction_includes_motivation(self, simple_play_path):
        """Test that direction includes character motivation."""
        app = NoodleApp()
        app.load_director(simple_play_path)
        app.start_performance()

        direction = app.get_brenda_direction()

        # Should include the suggested dialogue
        assert "Welcome!" in direction

    def test_direction_includes_emotional_state(self, simple_play_path):
        """Test that direction includes emotional state."""
        app = NoodleApp()
        app.load_director(simple_play_path)
        app.start_performance()

        direction = app.get_brenda_direction()

        # Should describe emotional state
        assert "feeling" in direction.lower()

    def test_passive_mode_direction(self, simple_play_path):
        """Test direction text in passive mode."""
        app = NoodleApp()
        app.load_director(simple_play_path)

        # Don't start performance - handler in passive mode
        direction = app.get_brenda_direction()

        # Should indicate passive mode
        assert "passive" in direction.lower()


# =============================================================================
# User Input and Response Flow Tests
# =============================================================================

class TestUserInputResponseFlow:
    """Tests for user input and response reporting."""

    def test_publish_user_input(self, simple_play_path):
        """Test that publish_user_input sends to channel."""
        app = NoodleApp()
        app.load_director(simple_play_path)

        # Track messages on user input channel
        received = []
        app.channel_bus.subscribe("#user.input", lambda m: received.append(m))

        # Publish input
        app.publish_user_input("Hello there!")

        # Should have received message
        assert len(received) == 1
        assert received[0].payload['text'] == "Hello there!"

    def test_report_actor_response(self, simple_play_path):
        """Test that report_actor_response updates handler state."""
        app = NoodleApp()
        app.load_director(simple_play_path)
        app.start_performance()

        # Report a response
        app.report_actor_response(
            response="Welcome to the exhibit!",
            user_message="Hi!"
        )

        # Handler state should be updated
        assert app.guide_cue_handler.state.last_response == "Welcome to the exhibit!"
        assert app.guide_cue_handler.state.last_user_message == "Hi!"


# =============================================================================
# ComputerUse Controller Wiring Tests
# =============================================================================

class TestComputerUseWiring:
    """Tests for ComputerUseController wiring."""

    def test_set_computer_use_controller(self, simple_play_path):
        """Test that set_computer_use_controller wires to handler."""
        app = NoodleApp()
        app.load_director(simple_play_path)

        # Create mock controller
        class MockController:
            pass

        controller = MockController()

        # Wire it up
        app.set_computer_use_controller(controller)

        # Handler should have the controller
        assert app.guide_cue_handler._computer_use is controller

    def test_set_computer_use_controller_no_handler(self):
        """Test that set_computer_use_controller handles no handler gracefully."""
        app = NoodleApp()

        # No play loaded, no handler
        class MockController:
            pass

        # Should not raise
        app.set_computer_use_controller(MockController())


# =============================================================================
# Computer Use Play Integration Tests
# =============================================================================

@pytest.fixture
def computer_use_play_path():
    """Create a play file with computer_use actions."""
    play_content = """
title: "Computer Use Integration Test"
author: "Test"

characters:
  guide:
    initial_pad:
      pleasure: 0.6
      arousal: 0.5
      dominance: 0.5
    motivation: "Demonstrate the UI"

beats:
  - id: click_view_project
    name: "Click View Project"
    on_stage: [guide]
    direction: "Show the user where the View Project button is"
    guide:
      speaks: "Let me show you the View Project button."
      computer_use:
        - action: move
          target: "Button: View Project"
        - action: click
          target: "Button: View Project"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.play.yaml', delete=False) as f:
        f.write(play_content)
        path = f.name

    yield path

    os.unlink(path)


class TestComputerUsePlayIntegration:
    """Integration tests for computer_use actions through the full play chain."""

    def test_play_with_computer_use_executes_actions(self, computer_use_play_path):
        """Full chain: play -> Brenda -> cue -> GuideCueHandler -> controller."""
        app = NoodleApp()
        app.load_director(computer_use_play_path)

        # Wire mock controller
        controller = MagicMock()
        controller.get_ui_element_map.return_value = [
            {'name': 'Button: View Project', 'x': 150, 'y': 300},
        ]
        app.set_computer_use_controller(controller)

        async def run_test():
            # Start performance - this sends the first cue
            app.start_performance()

            # Let the scheduled computer_use task execute
            await asyncio.sleep(0)

            # Verify controller was driven
            controller.mouse_move.assert_called_once_with(150, 300)
            controller.click.assert_called_once_with(150, 300, 'left')

        asyncio.run(run_test())

    def test_play_with_computer_use_no_controller_safe(self, computer_use_play_path):
        """Play with computer_use but no controller does not crash."""
        app = NoodleApp()
        app.load_director(computer_use_play_path)

        # No controller set - should not raise
        async def run_test():
            app.start_performance()
            await asyncio.sleep(0)

        asyncio.run(run_test())

        # Cue was still received and stored
        assert app.guide_cue_handler.state.current_beat_id == 'click_view_project'

    def test_play_computer_use_cue_also_stores_direction(self, computer_use_play_path):
        """Computer use cue also stores direction for prompt context."""
        app = NoodleApp()
        app.load_director(computer_use_play_path)

        controller = MagicMock()
        controller.get_ui_element_map.return_value = []
        app.set_computer_use_controller(controller)

        async def run_test():
            app.start_performance()
            await asyncio.sleep(0)

        asyncio.run(run_test())

        # Direction should be available for LLM prompt
        direction = app.get_brenda_direction()
        assert "Director's Notes" in direction
        assert "View Project" in direction


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
