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
#   LED Indicator Component
#
#   Dashboard-style LED indicator for displaying boolean state.
#   Supports round/square shapes, glow effects, and blinking.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.led
# PURPOSE:  LED Indicator Component
# LAYER:    Studio / UI Components / Dashboard
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LED
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from enum import Enum
from typing import Any, Dict, Optional

from ..component import UIComponent, register_component


class LEDShape(Enum):
    """Shape of the LED indicator."""
    ROUND = "round"
    SQUARE = "square"


@register_component
class LED(UIComponent):
    """
    LED indicator component for dashboard displays.

    A simple on/off indicator that renders like a physical LED with
    optional glow effects and blinking animation.

    Properties:
        on: Boolean state (True = lit, False = unlit)
        color: Color when lit (hex string, default green)
        off_color: Color when unlit (hex string, dimmer by default)
        size: Diameter/side length in pixels
        shape: 'round' or 'square'
        glow: Glow intensity (0.0 = none, 1.0 = full glow)
        blink_rate: Blink interval in seconds (0 = no blink)
        label: Optional text label displayed next to LED
        label_position: 'right', 'left', 'top', 'bottom'

    Events:
        onChange: Triggered when on state changes
        onClick: Triggered when LED is clicked
    """

    component_type = "LED"

    # Common LED colors (Mercedes dashboard aesthetic)
    COLOR_GREEN = "#00ff66"
    COLOR_RED = "#ff3344"
    COLOR_YELLOW = "#ffcc00"
    COLOR_BLUE = "#3399ff"
    COLOR_ORANGE = "#ff8800"
    COLOR_WHITE = "#ffffff"

    def __init__(
        self,
        name: str = "",
        on: bool = False,
        color: str = "#00ff66",  # Bright green
        size: int = 16
    ):
        super().__init__(name)

        # Core state
        self.on: bool = on

        # Appearance
        self.color: str = color
        self.off_color: str = ""  # Empty = auto-calculate from color
        self.size: int = size
        self.shape: LEDShape = LEDShape.ROUND
        self.glow: float = 0.6  # Default glow when lit
        self.border_color: str = "#333333"
        self.border_width: int = 1

        # Animation
        self.blink_rate: float = 0.0  # Seconds between blinks (0 = no blink)

        # Label
        self.label: str = ""
        self.label_position: str = "right"  # right, left, top, bottom
        self.label_color: str = "#cccccc"
        self.label_spacing: int = 8
        self.font_size: int = 12

        # Set default geometry
        self.geometry.width = size
        self.geometry.height = size

    @property
    def value(self) -> bool:
        """Alias for on state (for consistency with other components)."""
        return self.on

    @value.setter
    def value(self, val: bool) -> None:
        self.on = val

    def toggle(self) -> bool:
        """Toggle on state and return new state."""
        self.on = not self.on
        return self.on

    def turn_on(self) -> None:
        """Turn the LED on."""
        self.on = True

    def turn_off(self) -> None:
        """Turn the LED off."""
        self.on = False

    def get_effective_off_color(self) -> str:
        """
        Get the off color, auto-calculating if not explicitly set.

        Returns a dimmed version of the on color (20% brightness).
        """
        if self.off_color:
            return self.off_color

        # Auto-calculate: dim the on color
        color = self.color.lstrip('#')
        if len(color) == 6:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            # Reduce to 20% brightness
            r = int(r * 0.2)
            g = int(g * 0.2)
            b = int(b * 0.2)
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#1a1a1a"  # Fallback dark

    def get_current_color(self) -> str:
        """Get the current display color based on state."""
        return self.color if self.on else self.get_effective_off_color()

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add LED-specific properties to serialization."""
        data["on"] = self.on
        data["color"] = self.color
        data["size"] = self.size

        if self.off_color:
            data["off_color"] = self.off_color
        if self.shape != LEDShape.ROUND:
            data["shape"] = self.shape.value
        if self.glow != 0.6:
            data["glow"] = self.glow
        if self.border_color != "#333333":
            data["border_color"] = self.border_color
        if self.border_width != 1:
            data["border_width"] = self.border_width
        if self.blink_rate != 0.0:
            data["blink_rate"] = self.blink_rate
        if self.label:
            data["label"] = self.label
            data["label_position"] = self.label_position
        if self.label_color != "#cccccc":
            data["label_color"] = self.label_color
        if self.label_spacing != 8:
            data["label_spacing"] = self.label_spacing
        if self.font_size != 12:
            data["font_size"] = self.font_size

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LED':
        """Deserialize from dictionary."""
        led = cls(
            name=data.get("name", ""),
            on=data.get("on", False),
            color=data.get("color", "#00ff66"),
            size=data.get("size", 16)
        )

        led._apply_base_properties(data)

        # Appearance
        led.off_color = data.get("off_color", "")
        shape_str = data.get("shape", "round")
        led.shape = LEDShape(shape_str) if shape_str in [s.value for s in LEDShape] else LEDShape.ROUND
        led.glow = data.get("glow", 0.6)
        led.border_color = data.get("border_color", "#333333")
        led.border_width = data.get("border_width", 1)

        # Animation
        led.blink_rate = data.get("blink_rate", 0.0)

        # Label
        led.label = data.get("label", "")
        led.label_position = data.get("label_position", "right")
        led.label_color = data.get("label_color", "#cccccc")
        led.label_spacing = data.get("label_spacing", 8)
        led.font_size = data.get("font_size", 12)

        # Geometry
        led.geometry.width = data.get("width", led.size)
        led.geometry.height = data.get("height", led.size)

        return led


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
