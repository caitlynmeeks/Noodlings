# ------------------------------------------------------------------
#   Expression Perception Tests
#
#   Verifies: _describe_expression() returns observable physical cues
#   (not mood labels), and _format_present_entities() includes
#   Expression and Last action lines when available.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_expression_perception
# PURPOSE:  Commit 4 -- Observable expression cues + last action
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


def _make_manager():
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
    )
    from conftest import StubMainWindow, StubWindow, FakeLLMClient
    from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

    manager = GuidePerformanceManager(StubMainWindow())
    manager._ensemble_mode = True
    manager._window = StubWindow()

    ajo = NoodlingPerformer(
        noodling_id='ajo', name='Ajo Majo', llm_client=FakeLLMClient()
    )
    krampus = NoodlingPerformer(
        noodling_id='krampus', name='Krampus', llm_client=FakeLLMClient()
    )

    manager._performers = {'ajo': ajo, 'krampus': krampus}
    manager._instance_metadata = {
        'ajo': {
            'name': 'Ajo Majo',
            'appearance': 'A small chibi axolotl with pink-lavender coloring',
        },
        'krampus': {
            'name': 'Krampus',
            'appearance': 'A seven-year-old boy with tiny horns',
        },
    }
    return manager


class TestDescribeExpression:
    """_describe_expression() returns observable body-language cues."""

    @pytest.fixture
    def manager(self):
        return _make_manager()

    def test_returns_observable_prose(self, manager):
        """Result must describe physical signals, not mood labels."""
        result = manager._describe_expression({'valence': 0.7, 'arousal': 0.6, 'dominance': 0.5})
        # Observable terms -- not emotion labels
        assert 'smile' in result or 'eyes' in result

    def test_high_valence_returns_bright_eyes_smile(self, manager):
        result = manager._describe_expression({'valence': 0.8, 'arousal': 0.5, 'dominance': 0.5})
        assert 'bright eyes' in result
        assert 'smile' in result

    def test_negative_valence_returns_jaw_tight(self, manager):
        result = manager._describe_expression({'valence': -0.7, 'arousal': 0.5, 'dominance': 0.5})
        assert 'jaw tight' in result

    def test_high_arousal_returns_animated_gestures(self, manager):
        result = manager._describe_expression({'valence': 0.0, 'arousal': 0.9, 'dominance': 0.5})
        assert 'animated gestures' in result

    def test_low_arousal_returns_still(self, manager):
        result = manager._describe_expression({'valence': 0.0, 'arousal': 0.2, 'dominance': 0.5})
        assert 'still' in result

    def test_low_dominance_returns_hunched(self, manager):
        result = manager._describe_expression({'valence': 0.0, 'arousal': 0.5, 'dominance': 0.2})
        assert 'hunched' in result

    def test_high_dominance_returns_chin_up(self, manager):
        result = manager._describe_expression({'valence': 0.0, 'arousal': 0.5, 'dominance': 0.8})
        assert 'chin up' in result

    def test_result_is_string(self, manager):
        result = manager._describe_expression({'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5})
        assert isinstance(result, str)


class TestFormatPresentEntitiesExpression:
    """_format_present_entities includes Expression and Last action when available."""

    @pytest.fixture
    def manager(self):
        return _make_manager()

    def test_expression_line_present_when_affect_available(self, manager):
        """Expression line appears when a performer has affect data."""
        manager._performers['ajo']._last_pad_values = {
            'valence': 0.7, 'arousal': 0.8, 'dominance': 0.5
        }
        result = manager._format_present_entities('krampus')
        assert 'Expression:' in result

    def test_expression_line_absent_when_no_affect(self, manager):
        """No Expression line when performer has no affect data yet."""
        # ajo._last_pad_values is None by default
        result = manager._format_present_entities('krampus')
        assert 'Expression:' not in result

    def test_last_action_line_present_when_actions_available(self, manager):
        """Last action line appears when _last_actions is non-empty."""
        manager._performers['ajo']._last_actions = ['*wiggles gill nubs*']
        result = manager._format_present_entities('krampus')
        assert 'Last action:' in result
        assert 'wiggles gill nubs' in result

    def test_last_action_absent_when_no_actions(self, manager):
        """No Last action line when _last_actions is empty."""
        manager._performers['ajo']._last_actions = []
        result = manager._format_present_entities('krampus')
        assert 'Last action:' not in result

    def test_last_action_uses_most_recent_action(self, manager):
        """Last action uses the LAST element of _last_actions."""
        manager._performers['ajo']._last_actions = [
            '*first action*', '*second action*', '*final action*'
        ]
        result = manager._format_present_entities('krampus')
        assert 'final action' in result
        assert 'first action' not in result

    def test_appearance_description_still_present(self, manager):
        """Appearance description must still appear in entities output."""
        result = manager._format_present_entities('krampus')
        assert 'chibi axolotl' in result

    def test_excludes_self(self, manager):
        """Excluded noodling must not appear in output."""
        result = manager._format_present_entities('ajo')
        assert 'Ajo Majo' not in result
        assert 'Krampus' in result
