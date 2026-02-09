# ──────────────────────────────────────────────────────────────
#   Tests for PerformancePlayer
#
#   Tests character-by-character text delivery, punctuation pauses,
#   speaking state changes, and playback lifecycle.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import json
import pytest
from unittest.mock import MagicMock

try:
    from PyQt6.QtCore import QCoreApplication
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")


@pytest.fixture(scope="session")
def qapp():
    """Create a QCoreApplication for signal/slot testing."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _make_script(text, base_delay=35, speaking_intensity=0.7):
    """Helper to create a performance script dict."""
    chars = []
    pauses = {'.': 220, '!': 250, '?': 250, ',': 120}
    for ch in text:
        delay = pauses.get(ch, base_delay)
        if ch == ' ':
            delay = int(base_delay * 0.6)
        chars.append({'c': ch, 'd': delay})
    return {
        'type': 'performance_script',
        'text': text,
        'characters': chars,
        'speaking_intensity': speaking_intensity,
    }


# =============================================================================
# PerformancePlayer Construction
# =============================================================================

class TestPerformancePlayerConstruction:
    """Tests for PerformancePlayer initialization."""

    def test_creates_without_error(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        assert player is not None

    def test_not_playing_initially(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        assert not player.is_playing

    def test_speaking_intensity_default(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        assert player.speaking_intensity == 0.7


# =============================================================================
# Playback Tests
# =============================================================================

class TestPerformancePlayback:
    """Tests for PerformancePlayer playback mechanics."""

    def test_play_emits_characters(self, qapp):
        """Verify characterRevealed emits for each character."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        received = []
        player.characterRevealed.connect(lambda c: received.append(c))

        # Use zero delays for instant playback
        script = {
            'type': 'performance_script',
            'text': 'Hi',
            'characters': [
                {'c': 'H', 'd': 0},
                {'c': 'i', 'd': 0},
            ],
            'speaking_intensity': 0.7,
        }
        player.play(script)

        # Process events to let timers fire
        qapp.processEvents()
        qapp.processEvents()
        qapp.processEvents()
        qapp.processEvents()

        assert 'H' in received
        assert 'i' in received

    def test_play_emits_speaking_true_on_start(self, qapp):
        """Speaking state becomes True when playback starts."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        states = []
        player.speakingStateChanged.connect(lambda s: states.append(s))

        script = {
            'type': 'performance_script',
            'text': 'A',
            'characters': [{'c': 'A', 'd': 0}],
            'speaking_intensity': 0.7,
        }
        player.play(script)

        # First state should be True (speaking started)
        assert len(states) >= 1
        assert states[0] is True

    def test_empty_script_emits_finished(self, qapp):
        """Empty script emits finished immediately."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        finished_calls = []
        player.finished.connect(lambda: finished_calls.append(True))

        script = {
            'type': 'performance_script',
            'text': '',
            'characters': [],
            'speaking_intensity': 0.7,
        }
        player.play(script)
        assert len(finished_calls) == 1

    def test_stop_halts_playback(self, qapp):
        """Stop prevents further character emission."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        # Long delays so nothing fires before stop
        script = {
            'type': 'performance_script',
            'text': 'Hello',
            'characters': [{'c': c, 'd': 5000} for c in 'Hello'],
            'speaking_intensity': 0.7,
        }

        received = []
        player.characterRevealed.connect(lambda c: received.append(c))
        player.play(script)

        # First char is emitted immediately by _reveal_next
        qapp.processEvents()
        count_before = len(received)

        player.stop()
        assert not player.is_playing

        # No more chars after stop
        qapp.processEvents()
        qapp.processEvents()
        assert len(received) == count_before

    def test_speaking_intensity_from_script(self, qapp):
        """Speaking intensity is read from the script."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        script = {
            'type': 'performance_script',
            'text': 'X',
            'characters': [{'c': 'X', 'd': 0}],
            'speaking_intensity': 0.42,
        }
        player.play(script)
        assert player.speaking_intensity == 0.42


# =============================================================================
# Speaking State Tests
# =============================================================================

class TestSpeakingState:
    """Tests for speaking state transitions during playback."""

    def test_long_pause_toggles_speaking_off(self, qapp):
        """A delay >= 150ms triggers speaking state change to False."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        states = []
        player.speakingStateChanged.connect(lambda s: states.append(s))

        # Use zero delays so timers fire immediately on processEvents.
        # The period character has delay=220 which is >= threshold (150ms),
        # so it should toggle speaking off.
        script = {
            'type': 'performance_script',
            'text': 'A.',
            'characters': [
                {'c': 'A', 'd': 0},    # Zero delay - fires immediately
                {'c': '.', 'd': 220},   # Long pause - speaking goes False
            ],
            'speaking_intensity': 0.7,
        }
        player.play(script)

        # Process events: first char emits immediately in play(),
        # 0ms timer fires second char on processEvents
        for _ in range(10):
            qapp.processEvents()

        # Should have True (start), then False (at period)
        assert True in states
        assert False in states

    def test_speaking_off_after_stop(self, qapp):
        """Stop always sets speaking to False."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()

        states = []
        player.speakingStateChanged.connect(lambda s: states.append(s))

        script = {
            'type': 'performance_script',
            'text': 'ABC',
            'characters': [{'c': c, 'd': 5000} for c in 'ABC'],
            'speaking_intensity': 0.7,
        }
        player.play(script)
        player.stop()

        # Last state should be False
        assert states[-1] is False


# =============================================================================
# Performance Script Format Tests
# =============================================================================

class TestPerformanceScriptFormat:
    """Tests for the performance script dict format."""

    def test_script_has_required_fields(self):
        """Helper generates scripts with all required fields."""
        script = _make_script("Hello!")
        assert script['type'] == 'performance_script'
        assert script['text'] == 'Hello!'
        assert isinstance(script['characters'], list)
        assert script['speaking_intensity'] == 0.7

    def test_characters_have_char_and_delay(self):
        """Each character entry has 'c' and 'd' fields."""
        script = _make_script("Hi")
        for entry in script['characters']:
            assert 'c' in entry
            assert 'd' in entry

    def test_punctuation_has_longer_delay(self):
        """Punctuation characters get longer delays than regular chars."""
        script = _make_script("a.")
        a_entry = script['characters'][0]
        dot_entry = script['characters'][1]
        assert dot_entry['d'] > a_entry['d']

    def test_script_round_trips_through_json(self):
        """Script survives JSON serialization/deserialization."""
        script = _make_script("Test, string.")
        json_str = json.dumps(script)
        restored = json.loads(json_str)
        assert restored['text'] == "Test, string."
        assert len(restored['characters']) == len("Test, string.")
