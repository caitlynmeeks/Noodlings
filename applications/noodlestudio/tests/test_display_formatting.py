# ------------------------------------------------------------------
#   Display Formatting Tests
#
#   Verifies that PerformancePanel applies correct text formatting
#   for spoken vs. action segments, that format resets correctly,
#   and that the StubWindow accepts on_format_changed calls.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_display_formatting
# PURPOSE:  Commit 3 -- Display formatting for action/spoken/thought
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture
def panel(qapp, qtbot):
    from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
    p = PerformancePanel(ensemble_mode=True)
    qtbot.addWidget(p)
    return p


class TestOnFormatChanged:
    """PerformancePanel.on_format_changed() updates internal state."""

    def test_on_format_changed_updates_current_char_fmt(self, panel):
        panel.on_format_changed('action')
        assert panel._current_char_fmt == 'action'

    def test_on_format_changed_spoken(self, panel):
        panel.on_format_changed('action')
        panel.on_format_changed('spoken')
        assert panel._current_char_fmt == 'spoken'

    def test_initial_fmt_is_spoken(self, panel):
        assert panel._current_char_fmt == 'spoken'


class TestAppendCharacterFormatting:
    """append_character uses _current_char_fmt to style text."""

    def _fmt_of_last_char(self, panel):
        """Return QTextCharFormat of the most recently appended character.

        Qt's charFormat() at End position returns the format of the character
        immediately before the cursor (the last inserted character).
        """
        from PyQt6.QtGui import QTextCursor
        cursor = panel.dialogue_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        return cursor.charFormat()

    def test_spoken_char_is_not_italic(self, panel):
        """Characters in spoken mode should NOT be italic."""
        panel.begin_noodling_text('ajo', 'Ajo')
        panel.on_format_changed('spoken')
        panel.append_character('A')
        fmt = self._fmt_of_last_char(panel)
        assert not fmt.fontItalic()

    def test_action_char_is_italic(self, panel):
        """Characters in action mode should be italic."""
        panel.begin_noodling_text('ajo', 'Ajo')
        panel.on_format_changed('action')
        panel.append_character('*')
        fmt = self._fmt_of_last_char(panel)
        assert fmt.fontItalic()


class TestFormatReset:
    """Format state is reset in clear_dialogue and begin_noodling_text."""

    def test_format_resets_on_clear_dialogue(self, panel):
        panel.on_format_changed('action')
        panel.clear_dialogue()
        assert panel._current_char_fmt == 'spoken'

    def test_format_resets_on_begin_noodling_text(self, panel):
        panel.on_format_changed('action')
        panel.begin_noodling_text('ajo', 'Ajo')
        assert panel._current_char_fmt == 'spoken'


class TestStubWindowFormatChanged:
    """StubWindow must accept on_format_changed for testing."""

    def test_stub_window_accepts_format_changed(self):
        from conftest import StubWindow
        stub = StubWindow()
        stub.on_format_changed('action')
        assert stub._format_changes == ['action']

    def test_stub_window_tracks_multiple_changes(self):
        from conftest import StubWindow
        stub = StubWindow()
        stub.on_format_changed('action')
        stub.on_format_changed('spoken')
        stub.on_format_changed('action')
        assert stub._format_changes == ['action', 'spoken', 'action']
