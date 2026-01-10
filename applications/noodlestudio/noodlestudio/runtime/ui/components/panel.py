# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Panel Component
#
#   Container component with background color. Equivalent to ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.panel
# PURPOSE:  Panel Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Panel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Any, Dict, Optional

from ..component import UIComponent, register_component


@register_component
class Panel(UIComponent):
    """
    Container component with background color.

    Properties:
        background: Background color (hex string, e.g., "#1a1a1a")
        border_color: Border color (hex string, optional)
        border_width: Border width in pixels
        border_radius: Corner radius in pixels
    """

    component_type = "Panel"

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.background: str = "#2a2a2a"
        self.border_color: Optional[str] = None
        self.border_width: int = 0
        self.border_radius: int = 0

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Panel-specific properties to serialization."""
        if self.background != "#2a2a2a":
            data["background"] = self.background
        if self.border_color:
            data["border_color"] = self.border_color
        if self.border_width > 0:
            data["border_width"] = self.border_width
        if self.border_radius > 0:
            data["border_radius"] = self.border_radius

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Panel':
        """Deserialize from dictionary."""
        panel = cls(name=data.get("name", ""))

        # Apply base properties (geometry, anchors, events, bindings)
        panel._apply_base_properties(data)

        # Override geometry defaults for panels
        panel.geometry.width = data.get("width", 100)
        panel.geometry.height = data.get("height", 100)

        # Panel-specific properties
        panel.background = data.get("background", "#2a2a2a")
        panel.border_color = data.get("border_color")
        panel.border_width = data.get("border_width", 0)
        panel.border_radius = data.get("border_radius", 0)

        return panel

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
