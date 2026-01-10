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
#   Slider Component
#
#   Numeric range slider. Equivalent to Delphi's TTrackBar.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.slider
# PURPOSE:  Slider Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Slider
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
class Slider(UIComponent):
    """
    Horizontal slider for numeric value selection.

    Properties:
        value: Current value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Step increment (0 for continuous)
        show_value: Whether to display current value
        value_format: Format string for value display (e.g., "{:.1f}")
        track_color: Color of the track background
        track_fill_color: Color of filled portion
        handle_color: Color of the draggable handle
        handle_size: Size of handle in pixels
        track_height: Height of track in pixels

    Events:
        onChange: Triggered when value changes
        onSlideStart: Triggered when user starts dragging
        onSlideEnd: Triggered when user stops dragging
    """

    component_type = "Slider"

    def __init__(
        self,
        name: str = "",
        value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 100.0
    ):
        super().__init__(name)
        self._value: float = value
        self.min_value: float = min_value
        self.max_value: float = max_value
        self.step: float = 0.0  # 0 = continuous
        self.show_value: bool = False
        self.value_format: str = "{:.0f}"
        self.track_color: str = "#3d3d3d"
        self.track_fill_color: str = "#76AF6A"  # Noodle green
        self.handle_color: str = "#cccccc"
        self.handle_hover_color: str = "#ffffff"
        self.handle_size: int = 16
        self.track_height: int = 6

        # Default size
        self.geometry.width = 200
        self.geometry.height = 24

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        """Set value, clamping to valid range."""
        # Clamp to range
        val = max(self.min_value, min(self.max_value, val))
        # Snap to step if specified
        if self.step > 0:
            val = round((val - self.min_value) / self.step) * self.step + self.min_value
            val = max(self.min_value, min(self.max_value, val))
        self._value = val

    @property
    def percentage(self) -> float:
        """Get value as percentage (0.0 - 1.0)."""
        range_size = self.max_value - self.min_value
        if range_size == 0:
            return 0.0
        return (self._value - self.min_value) / range_size

    @percentage.setter
    def percentage(self, pct: float) -> None:
        """Set value by percentage (0.0 - 1.0)."""
        pct = max(0.0, min(1.0, pct))
        self.value = self.min_value + pct * (self.max_value - self.min_value)

    @property
    def formatted_value(self) -> str:
        """Get value formatted for display."""
        try:
            return self.value_format.format(self._value)
        except (ValueError, KeyError):
            return str(self._value)

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Slider-specific properties to serialization."""
        data["value"] = self._value
        data["min_value"] = self.min_value
        data["max_value"] = self.max_value

        if self.step != 0.0:
            data["step"] = self.step
        if self.show_value:
            data["show_value"] = self.show_value
        if self.value_format != "{:.0f}":
            data["value_format"] = self.value_format
        if self.track_color != "#3d3d3d":
            data["track_color"] = self.track_color
        if self.track_fill_color != "#76AF6A":
            data["track_fill_color"] = self.track_fill_color
        if self.handle_color != "#cccccc":
            data["handle_color"] = self.handle_color
        if self.handle_hover_color != "#ffffff":
            data["handle_hover_color"] = self.handle_hover_color
        if self.handle_size != 16:
            data["handle_size"] = self.handle_size
        if self.track_height != 6:
            data["track_height"] = self.track_height

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Slider':
        """Deserialize from dictionary."""
        slider = cls(
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 100.0)
        )

        slider._apply_base_properties(data)

        slider.geometry.width = data.get("width", 200)
        slider.geometry.height = data.get("height", 24)

        slider.step = data.get("step", 0.0)
        slider.show_value = data.get("show_value", False)
        slider.value_format = data.get("value_format", "{:.0f}")
        slider.track_color = data.get("track_color", "#3d3d3d")
        slider.track_fill_color = data.get("track_fill_color", "#76AF6A")
        slider.handle_color = data.get("handle_color", "#cccccc")
        slider.handle_hover_color = data.get("handle_hover_color", "#ffffff")
        slider.handle_size = data.get("handle_size", 16)
        slider.track_height = data.get("track_height", 6)

        return slider

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
