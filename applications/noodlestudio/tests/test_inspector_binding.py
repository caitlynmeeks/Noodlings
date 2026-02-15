# ──────────────────────────────────────────────────────────────
#   Inspector Binding Tests
#
#   Verifies: init_base_inspector() call, _bound_widgets init,
#   PropertyMeta attribute consistency, facet loading.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_inspector_binding
# PURPOSE:  Inspector Binding Tests
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


class TestInspectorBaseInit:
    """InspectorPanel must call init_base_inspector() to initialize bindings."""

    def test_inspector_has_bound_widgets_after_init(self, qapp):
        """InspectorPanel must have _bound_widgets initialized after construction."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        panel = InspectorPanel()
        assert hasattr(panel, '_bound_widgets'), "_bound_widgets not initialized"
        assert isinstance(panel._bound_widgets, dict)

    def test_inspector_has_all_base_attributes(self, qapp):
        """init_base_inspector() must set all required attributes."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        panel = InspectorPanel()
        for attr in ['property_fields', 'component_widgets', 'collapsible_states',
                     'is_loading', '_bound_widgets']:
            assert hasattr(panel, attr), f"Missing attribute: {attr}"


class TestPropertyMetaConsistency:
    """PropertyMeta attributes must match what inspector_base.py accesses."""

    def test_property_meta_has_minimum_maximum(self):
        """PropertyMeta must have minimum/maximum (not min_value/max_value)."""
        from noodlestudio.core.property_binding import PropertyMeta

        meta = PropertyMeta(name='test', prop_type=int, minimum=0, maximum=100)
        assert hasattr(meta, 'minimum'), "PropertyMeta missing 'minimum'"
        assert hasattr(meta, 'maximum'), "PropertyMeta missing 'maximum'"
        assert meta.minimum == 0
        assert meta.maximum == 100

    def test_property_meta_has_prop_type(self):
        """PropertyMeta must have prop_type (not property_type)."""
        from noodlestudio.core.property_binding import PropertyMeta

        meta = PropertyMeta(name='test', prop_type=str)
        assert hasattr(meta, 'prop_type'), "PropertyMeta missing 'prop_type'"
        assert meta.prop_type is str

    def test_property_meta_has_hidden(self):
        """PropertyMeta must have hidden attribute for inspector filtering."""
        from noodlestudio.core.property_binding import PropertyMeta

        meta = PropertyMeta(name='test', prop_type=str)
        assert hasattr(meta, 'hidden'), "PropertyMeta missing 'hidden'"
        assert meta.hidden is False  # Default

        meta_hidden = PropertyMeta(name='test', prop_type=str, hidden=True)
        assert meta_hidden.hidden is True

    def test_create_widget_for_property_uses_correct_attrs(self, qapp):
        """create_widget_for_property must use prop_type, not property_type."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.property_binding import PropertyMeta

        panel = InspectorPanel()

        class DummyObj:
            name = "test"

        meta = PropertyMeta(name='name', prop_type=str, display_name='Name')
        # Must not raise AttributeError
        widget = panel.create_widget_for_property(DummyObj(), meta)
        assert widget is not None


class TestInspectorFacetLoading:
    """Inspector must load facet properties without errors."""

    def test_inspector_loads_facet_without_error(self, qapp):
        """Selecting a facet must populate inspector without exceptions."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet

        panel = InspectorPanel()
        facet = Facet(
            id="test_facet_1",
            facet_type="LLMFacet",
            name="TestFacet",
            prompt="You are a test facet.",
        )
        # Must not raise
        panel._load_facet_standalone(facet)
        # Verify widgets were created
        assert 'name' in panel._bound_widgets, "Name widget not created"
        assert 'enabled' in panel._bound_widgets, "Enabled widget not created"
