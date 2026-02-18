# ------------------------------------------------------------------
#   Charm Network Inspector Tests (D.1.5 -- 3.5)
#
#   Verifies: inspector type detection, baseline parsing/editing,
#   depth navigation registry, depth view protocol, double-click
#   dispatch, inline inspector detection.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_charm_network_inspector
# PURPOSE:  Charm Network Inspector + Depth View Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication for all tests in this module."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_charm_facet(prompt='valence:0.7,arousal:0.5,dominance:0.4'):
    """Create a minimal CharmNetworkEMA Facet object."""
    from noodlestudio.core.facet_system import Facet
    return Facet(
        id='charm_test',
        name='Charm Network',
        facet_type='CharmNetworkEMA',
        prompt=prompt,
    )


# ======================================================================
# Baseline parsing and update helpers
# ======================================================================

class TestBaselineParsing:
    """Static helpers that parse/update the prompt-encoded baseline."""

    def test_parse_standard_baseline(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        result = InspectorPanel._parse_charm_baseline(
            'valence:0.7,arousal:0.5,dominance:0.4')
        assert result['valence'] == pytest.approx(0.7)
        assert result['arousal'] == pytest.approx(0.5)
        assert result['dominance'] == pytest.approx(0.4)

    def test_parse_negative_valence(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        result = InspectorPanel._parse_charm_baseline(
            'valence:-0.3,arousal:0.8,dominance:0.2')
        assert result['valence'] == pytest.approx(-0.3)

    def test_parse_empty_prompt(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        result = InspectorPanel._parse_charm_baseline('')
        assert result == {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}

    def test_parse_none_prompt(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        result = InspectorPanel._parse_charm_baseline(None)
        assert result == {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}

    def test_update_valence(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        updated = InspectorPanel._update_charm_baseline(
            'valence:0.7,arousal:0.5,dominance:0.4', 'valence', 0.85)
        assert 'valence:0.85' in updated
        # Other values unchanged
        assert 'arousal:0.5' in updated
        assert 'dominance:0.4' in updated

    def test_update_arousal(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        updated = InspectorPanel._update_charm_baseline(
            'valence:0.7,arousal:0.5,dominance:0.4', 'arousal', 0.80)
        assert 'arousal:0.80' in updated

    def test_update_negative_valence(self):
        from noodlestudio.panels.inspector_panel import InspectorPanel
        updated = InspectorPanel._update_charm_baseline(
            'valence:0.7,arousal:0.5,dominance:0.4', 'valence', -0.50)
        assert 'valence:-0.50' in updated

    def test_update_roundtrip(self):
        """Parse -> update -> parse must produce correct values."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        original = 'valence:0.7,arousal:0.5,dominance:0.4'
        updated = InspectorPanel._update_charm_baseline(original, 'dominance', 0.90)
        result = InspectorPanel._parse_charm_baseline(updated)
        assert result['valence'] == pytest.approx(0.7)
        assert result['arousal'] == pytest.approx(0.5)
        assert result['dominance'] == pytest.approx(0.90)


# ======================================================================
# Inspector type detection
# ======================================================================

class TestInspectorTypeDetection:
    """CharmNetworkEMA facet must get its own inspector section."""

    def test_standalone_inspector_shows_charm_section(self, qapp):
        """Selecting a CharmNetworkEMA facet must show 'Charm Network', not 'LLM Configuration'."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.widgets.collapsible_section import CollapsibleSection

        inspector = InspectorPanel()
        facet = _make_charm_facet()
        inspector._load_facet_standalone(facet)

        # Collect section titles
        section_titles = []
        for i in range(inspector.properties_layout.count()):
            widget = inspector.properties_layout.itemAt(i).widget()
            if isinstance(widget, CollapsibleSection):
                section_titles.append(widget.title_text)

        assert "Charm Network" in section_titles
        assert "LLM Configuration" not in section_titles

    def test_standalone_inspector_shows_current_state(self, qapp):
        """Inspector must include a 'Current State' section."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.widgets.collapsible_section import CollapsibleSection

        inspector = InspectorPanel()
        facet = _make_charm_facet()
        inspector._load_facet_standalone(facet)

        section_titles = []
        for i in range(inspector.properties_layout.count()):
            widget = inspector.properties_layout.itemAt(i).widget()
            if isinstance(widget, CollapsibleSection):
                section_titles.append(widget.title_text)

        assert "Current State" in section_titles

    def test_standalone_inspector_shows_parameters(self, qapp):
        """Inspector must include a collapsed 'Parameters' section."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.widgets.collapsible_section import CollapsibleSection

        inspector = InspectorPanel()
        facet = _make_charm_facet()
        inspector._load_facet_standalone(facet)

        section_titles = []
        for i in range(inspector.properties_layout.count()):
            widget = inspector.properties_layout.itemAt(i).widget()
            if isinstance(widget, CollapsibleSection):
                section_titles.append(widget.title_text)

        assert "Parameters" in section_titles

    def test_charm_network_facet_also_handled(self, qapp):
        """The older 'CharmNetworkFacet' type must also route to charm inspector."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from noodlestudio.widgets.collapsible_section import CollapsibleSection

        inspector = InspectorPanel()
        facet = Facet(
            id='cn_old', name='Old Charm', facet_type='CharmNetworkFacet',
            prompt='valence:0.5,arousal:0.5,dominance:0.5',
        )
        inspector._load_facet_standalone(facet)

        section_titles = []
        for i in range(inspector.properties_layout.count()):
            widget = inspector.properties_layout.itemAt(i).widget()
            if isinstance(widget, CollapsibleSection):
                section_titles.append(widget.title_text)

        assert "Charm Network" in section_titles
        assert "LLM Configuration" not in section_titles


# ======================================================================
# Depth navigation registry
# ======================================================================

class TestDepthNavigationRegistry:
    """CharmNetworkEMA must be registered for depth navigation."""

    def test_registry_has_charm_network_ema(self):
        from noodlestudio.panels.editors import UnifiedEditorPanel
        view_class = UnifiedEditorPanel.get_registered_view_class("CharmNetworkEMA")
        assert view_class is not None

    def test_registry_returns_correct_class(self):
        from noodlestudio.panels.editors import UnifiedEditorPanel
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        view_class = UnifiedEditorPanel.get_registered_view_class("CharmNetworkEMA")
        assert view_class is CharmNetworkDepthView


# ======================================================================
# Depth view protocol
# ======================================================================

class TestDepthViewProtocol:
    """CharmNetworkDepthView must implement DepthViewProtocol."""

    def test_has_load_data(self):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        assert hasattr(CharmNetworkDepthView, 'load_data')

    def test_has_save_data(self):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        assert hasattr(CharmNetworkDepthView, 'save_data')

    def test_has_get_breadcrumb_label(self):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        assert hasattr(CharmNetworkDepthView, 'get_breadcrumb_label')

    def test_has_has_unsaved_changes(self):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        assert hasattr(CharmNetworkDepthView, 'has_unsaved_changes')

    def test_breadcrumb_label(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        view = CharmNetworkDepthView()
        assert view.get_breadcrumb_label() == "Charm Network"

    def test_no_unsaved_changes(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        view = CharmNetworkDepthView()
        assert view.has_unsaved_changes() is False

    def test_load_data_sets_baseline(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        view = CharmNetworkDepthView()
        view.load_data('valence:0.7,arousal:0.5,dominance:0.4', {})
        assert view._baseline['valence'] == pytest.approx(0.7)
        assert view._baseline['arousal'] == pytest.approx(0.5)
        assert view._baseline['dominance'] == pytest.approx(0.4)

    def test_load_data_with_noodling_name(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import CharmNetworkDepthView
        view = CharmNetworkDepthView()
        view.load_data('valence:0.7,arousal:0.5,dominance:0.4',
                        {'noodling_name': 'Ajo Majo'})
        assert "Ajo Majo" in view._title.text()


# ======================================================================
# PAD bar widget
# ======================================================================

class TestPADBar:
    """Custom PAD bar must handle bipolar (valence) and unipolar ranges."""

    def test_valence_bar_is_bipolar(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import _PADBar
        bar = _PADBar('valence', 0.5)
        assert bar._bipolar is True

    def test_arousal_bar_is_unipolar(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import _PADBar
        bar = _PADBar('arousal', 0.5)
        assert bar._bipolar is False

    def test_dominance_bar_is_unipolar(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import _PADBar
        bar = _PADBar('dominance', 0.5)
        assert bar._bipolar is False

    def test_set_value_updates(self, qapp):
        from noodlestudio.panels.editors.charm_network_depth_view import _PADBar
        bar = _PADBar('valence', 0.0)
        bar.set_value(-0.7)
        assert bar._value == pytest.approx(-0.7)


# ======================================================================
# Inline inspector (noodling entity view)
# ======================================================================

class TestInlineInspector:
    """The inline facet properties (below dropdown) must handle CharmNetworkEMA."""

    def test_inline_charm_shows_spinboxes(self, qapp):
        """CharmNetworkEMA in inline view must show baseline spinboxes."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from PyQt6.QtWidgets import QDoubleSpinBox

        inspector = InspectorPanel()
        # Set up the container for inline properties
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        inspector.facet_properties_container = QWidget()
        inspector.facet_properties_layout = QVBoxLayout(
            inspector.facet_properties_container)

        facet = _make_charm_facet()
        inspector._load_facet_properties_inline(facet)

        # Find all double spinboxes in the properties container
        spinboxes = inspector.facet_properties_container.findChildren(
            QDoubleSpinBox)
        # Should have 3: valence, arousal, dominance
        assert len(spinboxes) == 3

    def test_inline_charm_no_model_dropdown(self, qapp):
        """CharmNetworkEMA inline view must NOT show a Model dropdown."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from PyQt6.QtWidgets import QComboBox, QWidget, QVBoxLayout

        inspector = InspectorPanel()
        inspector.facet_properties_container = QWidget()
        inspector.facet_properties_layout = QVBoxLayout(
            inspector.facet_properties_container)

        facet = _make_charm_facet()
        inspector._load_facet_properties_inline(facet)

        combos = inspector.facet_properties_container.findChildren(QComboBox)
        assert len(combos) == 0
