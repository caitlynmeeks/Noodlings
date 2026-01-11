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
#   Gauge Component
#
#   Dashboard-style analog gauge for displaying numeric values.
#   Mercedes-inspired instrumentation aesthetic.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.gauge
# PURPOSE:  Gauge Component
# LAYER:    Studio / UI Components / Dashboard
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Gauge, GaugeZone
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..component import UIComponent, register_component


@dataclass
class GaugeZone:
    """
    A colored zone on the gauge arc.

    Used to indicate ranges like "safe", "warning", "danger".
    """
    start_value: float  # Zone start value
    end_value: float    # Zone end value
    color: str          # Zone color (hex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_value": self.start_value,
            "end_value": self.end_value,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GaugeZone':
        return cls(
            start_value=data.get("start_value", 0),
            end_value=data.get("end_value", 100),
            color=data.get("color", "#3d3d3d")
        )


@register_component
class Gauge(UIComponent):
    """
    Analog gauge component for dashboard displays.

    A circular dial with a rotating needle that indicates a value
    within a defined range. Supports color zones, tick marks, and
    digital value display.

    Properties:
        value: Current value to display
        min_value: Minimum value (default 0)
        max_value: Maximum value (default 100)
        start_angle: Arc start angle in degrees (default 225, bottom-left)
        sweep_angle: Arc sweep in degrees (default -270, clockwise)
        size: Diameter in pixels

        # Appearance
        background_color: Gauge face color
        arc_color: Default arc color
        arc_width: Width of the arc in pixels
        needle_color: Needle color
        needle_width: Needle width at base
        center_color: Center cap color
        center_radius: Center cap radius (0-1, fraction of size)

        # Tick marks
        major_ticks: Number of major tick marks
        minor_ticks: Number of minor ticks between majors
        tick_color: Tick mark color
        major_tick_length: Length of major ticks (fraction of radius)
        minor_tick_length: Length of minor ticks (fraction of radius)

        # Labels
        show_value: Show digital value display
        value_format: Format string for value (e.g., "{:.1f}")
        value_suffix: Suffix to append (e.g., "%", "rpm")
        label: Optional label text below value
        font_size: Font size for value display
        label_font_size: Font size for label
        text_color: Color for text elements

        # Color zones
        zones: List of GaugeZone for colored arc sections

    Events:
        onChange: Triggered when value changes
    """

    component_type = "Gauge"

    def __init__(
        self,
        name: str = "",
        value: float = 0,
        min_value: float = 0,
        max_value: float = 100,
        size: int = 120
    ):
        super().__init__(name)

        # Core value
        self.value: float = value
        self.min_value: float = min_value
        self.max_value: float = max_value

        # Arc geometry (angles in degrees, 0 = right, positive = counterclockwise)
        # Default: arc from bottom-left (225) sweeping clockwise (-270) to bottom-right
        self.start_angle: float = 225
        self.sweep_angle: float = -270  # Negative = clockwise

        # Size
        self.size: int = size

        # Appearance - Mercedes dark dashboard aesthetic
        self.background_color: str = "#1a1a1a"
        self.arc_color: str = "#3d3d3d"
        self.arc_width: int = 8
        self.needle_color: str = "#ff4444"
        self.needle_width: int = 3
        self.center_color: str = "#2a2a2a"
        self.center_radius: float = 0.15  # Fraction of gauge radius

        # Tick marks
        self.major_ticks: int = 5  # Number of major divisions
        self.minor_ticks: int = 4  # Minor ticks between each major
        self.tick_color: str = "#888888"
        self.major_tick_length: float = 0.15  # Fraction of radius
        self.minor_tick_length: float = 0.08

        # Labels
        self.show_value: bool = True
        self.value_format: str = "{:.0f}"
        self.value_suffix: str = ""
        self.label: str = ""
        self.font_size: int = 14
        self.label_font_size: int = 10
        self.text_color: str = "#cccccc"
        self.show_tick_labels: bool = True
        self.tick_label_font_size: int = 9

        # Color zones
        self.zones: List[GaugeZone] = []

        # Set default geometry
        self.geometry.width = size
        self.geometry.height = size

    def set_value(self, value: float) -> None:
        """Set gauge value, clamping to range."""
        self.value = max(self.min_value, min(self.max_value, value))

    def get_normalized_value(self) -> float:
        """Get value as 0-1 fraction of range."""
        range_size = self.max_value - self.min_value
        if range_size == 0:
            return 0
        return (self.value - self.min_value) / range_size

    def get_needle_angle(self) -> float:
        """Get needle angle in degrees based on current value."""
        normalized = self.get_normalized_value()
        return self.start_angle + (normalized * self.sweep_angle)

    def get_formatted_value(self) -> str:
        """Get formatted value string for display."""
        formatted = self.value_format.format(self.value)
        return f"{formatted}{self.value_suffix}"

    def add_zone(self, start_value: float, end_value: float, color: str) -> None:
        """Add a color zone to the gauge."""
        self.zones.append(GaugeZone(start_value, end_value, color))

    def clear_zones(self) -> None:
        """Remove all color zones."""
        self.zones.clear()

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Gauge-specific properties to serialization."""
        data["value"] = self.value
        data["min_value"] = self.min_value
        data["max_value"] = self.max_value
        data["size"] = self.size

        # Arc geometry (only if non-default)
        if self.start_angle != 225:
            data["start_angle"] = self.start_angle
        if self.sweep_angle != -270:
            data["sweep_angle"] = self.sweep_angle

        # Appearance (only if non-default)
        if self.background_color != "#1a1a1a":
            data["background_color"] = self.background_color
        if self.arc_color != "#3d3d3d":
            data["arc_color"] = self.arc_color
        if self.arc_width != 8:
            data["arc_width"] = self.arc_width
        if self.needle_color != "#ff4444":
            data["needle_color"] = self.needle_color
        if self.needle_width != 3:
            data["needle_width"] = self.needle_width
        if self.center_color != "#2a2a2a":
            data["center_color"] = self.center_color
        if self.center_radius != 0.15:
            data["center_radius"] = self.center_radius

        # Tick marks
        if self.major_ticks != 5:
            data["major_ticks"] = self.major_ticks
        if self.minor_ticks != 4:
            data["minor_ticks"] = self.minor_ticks
        if self.tick_color != "#888888":
            data["tick_color"] = self.tick_color
        if self.major_tick_length != 0.15:
            data["major_tick_length"] = self.major_tick_length
        if self.minor_tick_length != 0.08:
            data["minor_tick_length"] = self.minor_tick_length

        # Labels
        if not self.show_value:
            data["show_value"] = False
        if self.value_format != "{:.0f}":
            data["value_format"] = self.value_format
        if self.value_suffix:
            data["value_suffix"] = self.value_suffix
        if self.label:
            data["label"] = self.label
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.label_font_size != 10:
            data["label_font_size"] = self.label_font_size
        if self.text_color != "#cccccc":
            data["text_color"] = self.text_color
        if not self.show_tick_labels:
            data["show_tick_labels"] = False
        if self.tick_label_font_size != 9:
            data["tick_label_font_size"] = self.tick_label_font_size

        # Zones
        if self.zones:
            data["zones"] = [z.to_dict() for z in self.zones]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Gauge':
        """Deserialize from dictionary."""
        gauge = cls(
            name=data.get("name", ""),
            value=data.get("value", 0),
            min_value=data.get("min_value", 0),
            max_value=data.get("max_value", 100),
            size=data.get("size", 120)
        )

        gauge._apply_base_properties(data)

        # Arc geometry
        gauge.start_angle = data.get("start_angle", 225)
        gauge.sweep_angle = data.get("sweep_angle", -270)

        # Appearance
        gauge.background_color = data.get("background_color", "#1a1a1a")
        gauge.arc_color = data.get("arc_color", "#3d3d3d")
        gauge.arc_width = data.get("arc_width", 8)
        gauge.needle_color = data.get("needle_color", "#ff4444")
        gauge.needle_width = data.get("needle_width", 3)
        gauge.center_color = data.get("center_color", "#2a2a2a")
        gauge.center_radius = data.get("center_radius", 0.15)

        # Tick marks
        gauge.major_ticks = data.get("major_ticks", 5)
        gauge.minor_ticks = data.get("minor_ticks", 4)
        gauge.tick_color = data.get("tick_color", "#888888")
        gauge.major_tick_length = data.get("major_tick_length", 0.15)
        gauge.minor_tick_length = data.get("minor_tick_length", 0.08)

        # Labels
        gauge.show_value = data.get("show_value", True)
        gauge.value_format = data.get("value_format", "{:.0f}")
        gauge.value_suffix = data.get("value_suffix", "")
        gauge.label = data.get("label", "")
        gauge.font_size = data.get("font_size", 14)
        gauge.label_font_size = data.get("label_font_size", 10)
        gauge.text_color = data.get("text_color", "#cccccc")
        gauge.show_tick_labels = data.get("show_tick_labels", True)
        gauge.tick_label_font_size = data.get("tick_label_font_size", 9)

        # Zones
        zones_data = data.get("zones", [])
        gauge.zones = [GaugeZone.from_dict(z) for z in zones_data]

        # Geometry
        gauge.geometry.width = data.get("width", gauge.size)
        gauge.geometry.height = data.get("height", gauge.size)

        return gauge


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
