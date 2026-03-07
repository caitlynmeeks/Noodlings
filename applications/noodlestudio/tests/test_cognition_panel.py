# ──────────────────────────────────────────────────────────────
#   Tests for Cognition Panel
# ──────────────────────────────────────────────────────────────

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def panel(qapp):
    """Create a CognitionPanel for testing."""
    from noodlestudio.panels.cognition_panel import CognitionPanel
    p = CognitionPanel()
    yield p
    p.close()


def _sample_traces():
    """Generate sample traces for testing."""
    return [
        {
            'facet_id': 'response',
            'facet_name': 'Response',
            'facet_type': 'IntuitionFacet',
            'system_prompt': 'You are a helpful guide.',
            'formatted_prompt': 'User said: hello',
            'output': 'Hi there!',
            'execution_time': 1.5,
            'token_count': 42,
            'model_label': 'response',
        },
        {
            'facet_id': 'sentiment',
            'facet_name': 'Sentiment',
            'facet_type': 'IntuitionFacet',
            'system_prompt': 'Analyze mood.',
            'formatted_prompt': 'Message: hello',
            'output': 'valence=0.6',
            'execution_time': 0.3,
            'token_count': 15,
            'model_label': 'sentiment',
        },
    ]


class TestCognitionPanelCreation:
    """Basic panel creation tests."""

    def test_panel_creates(self, panel):
        from PyQt6.QtWidgets import QWidget
        assert isinstance(panel, QWidget)

    def test_has_empty_character_dropdown(self, panel):
        assert panel._character_combo.count() == 0

    def test_auto_follow_default_on(self, panel):
        assert panel._auto_follow is True
        assert panel._auto_follow_cb.isChecked()


class TestCognitionPanelData:
    """Data population tests."""

    def test_on_turn_trace_populates_dropdown(self, panel):
        """on_turn_trace adds noodling to dropdown."""
        panel.on_turn_trace('ajo', _sample_traces(), 1, time.time())
        assert panel._character_combo.count() == 1
        assert panel._character_combo.itemText(0) == 'ajo'

    def test_on_turn_trace_populates_sections(self, panel):
        """on_turn_trace creates facet sections."""
        panel.on_turn_trace('ajo', _sample_traces(), 1, time.time())
        # Should have 2 sections + 1 stretch = 3 items in layout
        assert panel._sections_layout.count() == 3

    def test_auto_follow_switches_to_speaker(self, panel):
        """Auto-follow switches dropdown to most recent speaker."""
        panel.on_turn_trace('ajo', _sample_traces(), 1, time.time())
        panel.on_turn_trace('krampus', _sample_traces(), 2, time.time())
        assert panel._character_combo.currentText() == 'krampus'

    def test_manual_select_preserved_when_auto_follow_off(self, panel):
        """When auto-follow is off, dropdown stays on selected."""
        panel.on_turn_trace('ajo', _sample_traces(), 1, time.time())
        panel._auto_follow_cb.setChecked(False)
        panel.on_turn_trace('krampus', _sample_traces(), 2, time.time())
        assert panel._character_combo.currentText() == 'ajo'


class TestCollapsibleFacetSection:
    """Section expand/collapse and content tests."""

    def test_section_expands_collapses(self, qapp):
        from noodlestudio.panels.cognition_panel import CollapsibleFacetSection
        trace = _sample_traces()[0]
        section = CollapsibleFacetSection(trace)

        assert not section._expanded
        assert not section._body.isVisible()

        section._toggle()
        assert section._expanded
        # Body widget visible flag set (even if parent not shown)
        assert not section._body.isHidden()

        section._toggle()
        assert not section._expanded
        section.close()

    def test_section_shows_correct_text(self, qapp):
        from noodlestudio.panels.cognition_panel import CollapsibleFacetSection
        trace = _sample_traces()[0]
        section = CollapsibleFacetSection(trace)

        # Default: output view
        assert section._text_view.toPlainText() == 'Hi there!'

        section._show_view('system')
        assert section._text_view.toPlainText() == 'You are a helpful guide.'

        section._show_view('prompt')
        assert section._text_view.toPlainText() == 'User said: hello'

        section.close()


class TestCognitionTabRegistered:
    """Cognition tab exists in main window."""

    def test_cognition_tab_on_main_window(self, main_window):
        assert hasattr(main_window, 'cognition_panel')
        from noodlestudio.panels.cognition_panel import CognitionPanel
        assert isinstance(main_window.cognition_panel, CognitionPanel)
