# ──────────────────────────────────────────────────────────────
#   Tests for Name Card Feature (Phase F.3a)
#
#   When a noodling has no VRM assigned, its viewport slot shows
#   a name card (dark rectangle with centered name) instead of
#   a blank placeholder.
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_name_card
# PURPOSE:  Name Card Feature Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def _make_ensemble_window(qapp):
    """Create an ensemble-mode PerformancePanel."""
    from noodlestudio.runtime.ui.guide_performance_window import (
        PerformancePanel
    )
    window = PerformancePanel(ensemble_mode=True)
    return window


# ═══════════════════════════════════════════════════════════════
# NAME CARD DISPLAY
# ═══════════════════════════════════════════════════════════════

class TestNameCard:
    """Test name card rendering in ensemble viewport slots."""

    def test_name_card_shows_noodling_name(self, qapp):
        """Name card displays the noodling's name in the placeholder."""
        window = _make_ensemble_window(qapp)
        try:
            window.show_name_card('ajo', 'Ajo Majo')

            slot = window._get_slot('ajo')
            placeholder = window._vrm_placeholders.get(slot)

            assert placeholder is not None
            assert placeholder.text() == 'Ajo Majo'
        finally:
            window.close()

    def test_name_card_styling(self, qapp):
        """Name card has dark background and light text."""
        window = _make_ensemble_window(qapp)
        try:
            window.show_name_card('krampus', 'Krampus')

            slot = window._get_slot('krampus')
            placeholder = window._vrm_placeholders.get(slot)

            style = placeholder.styleSheet()
            assert '#D2D2D2' in style, "Name card text should be light gray"
            assert '#2a2a2a' in style, "Name card background should be dark"
        finally:
            window.close()

    def test_name_card_for_multiple_noodlings(self, qapp):
        """Multiple noodlings without VRM each get their own name card."""
        window = _make_ensemble_window(qapp)
        try:
            window.show_name_card('ajo', 'Ajo Majo')
            window.show_name_card('krampus', 'Krampus')
            window.show_name_card('juanita', 'Juanita')

            for nid, name in [('ajo', 'Ajo Majo'),
                              ('krampus', 'Krampus'),
                              ('juanita', 'Juanita')]:
                slot = window._get_slot(nid)
                placeholder = window._vrm_placeholders.get(slot)
                assert placeholder is not None
                assert placeholder.text() == name
        finally:
            window.close()

    def test_name_card_restored_after_clear(self, qapp):
        """Calling show_name_card restores the card in its slot."""
        window = _make_ensemble_window(qapp)
        try:
            window.show_name_card('ajo', 'Ajo Majo')

            slot = window._get_slot('ajo')
            placeholder = window._vrm_placeholders.get(slot)
            assert placeholder.text() == 'Ajo Majo'

            # Call again (as update_vrm would when clearing)
            window.show_name_card('ajo', 'Ajo Majo')
            placeholder = window._vrm_placeholders.get(slot)
            assert placeholder is not None
            assert placeholder.text() == 'Ajo Majo'
        finally:
            window.close()

    def test_default_placeholder_text_before_name_card(self, qapp):
        """Before show_name_card, placeholders show generic text."""
        window = _make_ensemble_window(qapp)
        try:
            for slot_key in ('left', 'center', 'right'):
                placeholder = window._vrm_placeholders.get(slot_key)
                assert placeholder is not None
                assert placeholder.text() == "No character loaded"
        finally:
            window.close()


# ═══════════════════════════════════════════════════════════════
# MANAGER INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestNameCardManagerIntegration:
    """Test that the performance manager wires name cards for no-VRM noodlings."""

    def test_update_vrm_with_empty_path_shows_name_card(self):
        """update_vrm('') should call show_name_card on the window."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager
        )

        calls = []

        class FakeWindow:
            def show_name_card(self, noodling_id, name):
                calls.append((noodling_id, name))

        class FakePerformer:
            def __init__(self, name):
                self.name = name

        manager = GuidePerformanceManager.__new__(GuidePerformanceManager)
        manager._window = FakeWindow()
        manager._performers = {'ajo': FakePerformer('Ajo Majo')}
        manager._ensemble_mode = True

        manager.update_vrm('ajo', '')

        assert len(calls) == 1
        assert calls[0] == ('ajo', 'Ajo Majo')

    def test_update_vrm_with_none_path_shows_name_card(self):
        """update_vrm(None) should call show_name_card on the window."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager
        )

        calls = []

        class FakeWindow:
            def show_name_card(self, noodling_id, name):
                calls.append((noodling_id, name))

        class FakePerformer:
            def __init__(self, name):
                self.name = name

        manager = GuidePerformanceManager.__new__(GuidePerformanceManager)
        manager._window = FakeWindow()
        manager._performers = {'krampus': FakePerformer('Krampus')}
        manager._ensemble_mode = True

        manager.update_vrm('krampus', None)

        assert len(calls) == 1
        assert calls[0] == ('krampus', 'Krampus')

    def test_update_vrm_with_unknown_performer_uses_id(self):
        """update_vrm for unknown performer falls back to noodling_id as name."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager
        )

        calls = []

        class FakeWindow:
            def show_name_card(self, noodling_id, name):
                calls.append((noodling_id, name))

        manager = GuidePerformanceManager.__new__(GuidePerformanceManager)
        manager._window = FakeWindow()
        manager._performers = {}
        manager._ensemble_mode = True

        manager.update_vrm('unknown_noodling', '')

        assert len(calls) == 1
        assert calls[0] == ('unknown_noodling', 'unknown_noodling')
