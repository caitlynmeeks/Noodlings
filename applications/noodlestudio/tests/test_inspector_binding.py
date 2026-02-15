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
