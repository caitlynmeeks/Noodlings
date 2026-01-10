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
#   Button Component
#
#   Clickable button. Equivalent to Delphi's TButton.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.button
# PURPOSE:  Button Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Button
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
class Button(UIComponent):
    """
    Clickable button component.

    Properties:
        text: Button label text
        text_color: Text color (hex string)
        background: Background color (hex string)
        hover_background: Background color on hover
        pressed_background: Background color when pressed
        font_size: Font size in pixels
        border_radius: Corner radius in pixels

    Events:
        onClick: Triggered when button is clicked
        onHover: Triggered when mouse enters
        onLeave: Triggered when mouse leaves
    """

    component_type = "Button"

    def __init__(self, name: str = "", text: str = "Button"):
        super().__init__(name)
        self.text: str = text
        self.text_color: str = "#ffffff"
        self.background: str = "#3b82f6"  # Blue
        self.hover_background: Optional[str] = None  # Slightly lighter if not set
        self.pressed_background: Optional[str] = None  # Slightly darker if not set
        self.font_size: int = 14
        self.border_radius: int = 4

        # Default size for buttons
        self.geometry.width = 80
        self.geometry.height = 32

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Button-specific properties to serialization."""
        data["text"] = self.text

        if self.text_color != "#ffffff":
            data["text_color"] = self.text_color
        if self.background != "#3b82f6":
            data["background"] = self.background
        if self.hover_background:
            data["hover_background"] = self.hover_background
        if self.pressed_background:
            data["pressed_background"] = self.pressed_background
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.border_radius != 4:
            data["border_radius"] = self.border_radius

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Button':
        """Deserialize from dictionary."""
        button = cls(
            name=data.get("name", ""),
            text=data.get("text", "Button")
        )

        # Apply base properties (geometry, anchors, events, bindings)
        button._apply_base_properties(data)

        # Override geometry defaults for buttons
        button.geometry.width = data.get("width", 80)
        button.geometry.height = data.get("height", 32)

        # Button-specific properties
        button.text_color = data.get("text_color", "#ffffff")
        button.background = data.get("background", "#3b82f6")
        button.hover_background = data.get("hover_background")
        button.pressed_background = data.get("pressed_background")
        button.font_size = data.get("font_size", 14)
        button.border_radius = data.get("border_radius", 4)

        return button

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
