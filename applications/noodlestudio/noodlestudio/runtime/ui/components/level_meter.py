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
#   Level Meter Component
#
#   VU-meter style segmented bar display for showing levels.
#   Perfect for audio meters, progress indicators, and
#   affect monitoring dashboards.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.level_meter
# PURPOSE:  Level Meter Component
# LAYER:    Studio / UI Components / Dashboard
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LevelMeter, MeterZone, MeterOrientation
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..component import UIComponent, register_component


class MeterOrientation(Enum):
    """Orientation of the level meter bar."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass
class MeterZone:
    """
    A colored zone on the level meter.

    Used to indicate ranges like "safe" (green), "warning" (yellow),
    "danger" (red).
    """
    start_value: float  # Zone start value (0.0-1.0)
    end_value: float    # Zone end value (0.0-1.0)
    color: str          # Zone color (hex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_value": self.start_value,
            "end_value": self.end_value,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeterZone':
        return cls(
            start_value=data.get("start_value", 0.0),
            end_value=data.get("end_value", 1.0),
            color=data.get("color", "#33ff66")
        )


# Default VU meter zones (green -> yellow -> red)
DEFAULT_ZONES = [
    MeterZone(0.0, 0.6, "#33ff66"),    # Green: 0-60%
    MeterZone(0.6, 0.8, "#ffcc00"),    # Yellow: 60-80%
    MeterZone(0.8, 1.0, "#ff3344"),    # Red: 80-100%
]


@register_component
class LevelMeter(UIComponent):
    """
    VU-meter style level indicator component.

    A segmented or continuous bar display that shows a value
    within a 0-1 range. Supports color zones, peak hold,
    and classic audio meter aesthetics.

    Properties:
        value: Current level value (0.0 to 1.0)
        orientation: 'vertical' or 'horizontal'
        segments: Number of LED segments (0 = smooth/continuous bar)

        # Peak hold
        peak_hold: Show peak level indicator
        peak_value: Current peak value (auto-managed if peak_hold)
        peak_decay: Seconds for peak to decay to current value

        # Appearance
        background_color: Meter background color
        inactive_color: Color for unlit segments/area
        zones: List of MeterZone for colored sections
        glow: Glow intensity on lit segments (0.0 = none, 1.0 = full)

        # Sizing
        width: Width in pixels
        height: Height in pixels
        segment_gap: Gap between segments in pixels
        corner_radius: Corner radius for segments

        # Border
        border_color: Border color
        border_width: Border width in pixels

        # Labels
        show_scale: Show scale markings
        scale_color: Color for scale markings

    Events:
        onChange: Triggered when value changes

    Example:
        ```yaml
        - type: LevelMeter
          name: arousal_meter
          value: 0.7
          orientation: vertical
          segments: 10
          peak_hold: true
          zones:
            - start_value: 0.0
              end_value: 0.6
              color: "#33ff66"
            - start_value: 0.6
              end_value: 0.8
              color: "#ffcc00"
            - start_value: 0.8
              end_value: 1.0
              color: "#ff3344"
        ```
    """

    component_type = "LevelMeter"

    # Common color presets
    COLOR_GREEN = "#33ff66"
    COLOR_YELLOW = "#ffcc00"
    COLOR_RED = "#ff3344"
    COLOR_BLUE = "#3399ff"
    COLOR_CYAN = "#00ffcc"

    def __init__(
        self,
        name: str = "",
        value: float = 0.0,
        orientation: str = "vertical",
        segments: int = 10
    ):
        super().__init__(name)

        # Core value (0.0 - 1.0)
        self.value: float = max(0.0, min(1.0, value))

        # Layout
        self.orientation: str = orientation
        self.segments: int = segments  # 0 = smooth/continuous

        # Peak hold
        self.peak_hold: bool = False
        self.peak_value: float = 0.0
        self.peak_decay: float = 1.5  # Seconds to decay

        # Appearance - dark dashboard aesthetic
        self.background_color: str = "#1a1a1a"
        self.inactive_color: str = "#2a2a2a"
        self.glow: float = 0.4  # Subtle glow on lit segments

        # Color zones (default: green -> yellow -> red)
        self.zones: List[MeterZone] = list(DEFAULT_ZONES)

        # Sizing
        self._width: int = 24 if orientation == "vertical" else 120
        self._height: int = 120 if orientation == "vertical" else 24
        self.segment_gap: int = 2
        self.corner_radius: int = 2

        # Border
        self.border_color: str = "#333333"
        self.border_width: int = 1

        # Scale
        self.show_scale: bool = False
        self.scale_color: str = "#666666"

        # Set geometry
        self.geometry.width = self._width
        self.geometry.height = self._height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, val: int) -> None:
        self._width = val
        self.geometry.width = val

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, val: int) -> None:
        self._height = val
        self.geometry.height = val

    def set_value(self, value: float) -> None:
        """Set meter value, clamping to 0-1 range."""
        self.value = max(0.0, min(1.0, value))

        # Update peak if enabled
        if self.peak_hold and value > self.peak_value:
            self.peak_value = value

    def reset_peak(self) -> None:
        """Reset peak to current value."""
        self.peak_value = self.value

    def get_color_at_value(self, val: float) -> str:
        """
        Get the color for a given value based on zones.

        Args:
            val: Value between 0.0 and 1.0

        Returns:
            Hex color string for that level
        """
        for zone in self.zones:
            if zone.start_value <= val < zone.end_value:
                return zone.color
            # Handle edge case where val == 1.0 and zone.end_value == 1.0
            if val == 1.0 and zone.end_value == 1.0:
                return zone.color

        # Fallback to first zone color or default
        return self.zones[0].color if self.zones else self.COLOR_GREEN

    def get_segment_count_lit(self) -> int:
        """Get how many segments should be lit based on current value."""
        if self.segments <= 0:
            return 0
        return int(self.value * self.segments + 0.5)  # Round to nearest

    def get_peak_segment(self) -> int:
        """Get which segment the peak indicator should be on."""
        if self.segments <= 0 or not self.peak_hold:
            return -1
        return min(int(self.peak_value * self.segments + 0.5), self.segments) - 1

    def add_zone(self, start_value: float, end_value: float, color: str) -> None:
        """Add a color zone to the meter."""
        self.zones.append(MeterZone(start_value, end_value, color))

    def clear_zones(self) -> None:
        """Remove all color zones."""
        self.zones.clear()

    def set_default_zones(self) -> None:
        """Reset to default green/yellow/red zones."""
        self.zones = list(DEFAULT_ZONES)

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add LevelMeter-specific properties to serialization."""
        data["value"] = self.value
        data["orientation"] = self.orientation
        data["segments"] = self.segments

        # Peak hold
        if self.peak_hold:
            data["peak_hold"] = True
            data["peak_value"] = self.peak_value
        if self.peak_decay != 1.5:
            data["peak_decay"] = self.peak_decay

        # Appearance (only if non-default)
        if self.background_color != "#1a1a1a":
            data["background_color"] = self.background_color
        if self.inactive_color != "#2a2a2a":
            data["inactive_color"] = self.inactive_color
        if self.glow != 0.4:
            data["glow"] = self.glow

        # Sizing
        data["width"] = self._width
        data["height"] = self._height
        if self.segment_gap != 2:
            data["segment_gap"] = self.segment_gap
        if self.corner_radius != 2:
            data["corner_radius"] = self.corner_radius

        # Border
        if self.border_color != "#333333":
            data["border_color"] = self.border_color
        if self.border_width != 1:
            data["border_width"] = self.border_width

        # Scale
        if self.show_scale:
            data["show_scale"] = True
        if self.scale_color != "#666666":
            data["scale_color"] = self.scale_color

        # Zones (only if not default)
        zones_match_default = (
            len(self.zones) == 3 and
            all(
                z.start_value == d.start_value and
                z.end_value == d.end_value and
                z.color == d.color
                for z, d in zip(self.zones, DEFAULT_ZONES)
            )
        )
        if not zones_match_default:
            data["zones"] = [z.to_dict() for z in self.zones]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LevelMeter':
        """Deserialize from dictionary."""
        meter = cls(
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            orientation=data.get("orientation", "vertical"),
            segments=data.get("segments", 10)
        )

        meter._apply_base_properties(data)

        # Peak hold
        meter.peak_hold = data.get("peak_hold", False)
        meter.peak_value = data.get("peak_value", 0.0)
        meter.peak_decay = data.get("peak_decay", 1.5)

        # Appearance
        meter.background_color = data.get("background_color", "#1a1a1a")
        meter.inactive_color = data.get("inactive_color", "#2a2a2a")
        meter.glow = data.get("glow", 0.4)

        # Sizing
        meter._width = data.get("width", 24 if meter.orientation == "vertical" else 120)
        meter._height = data.get("height", 120 if meter.orientation == "vertical" else 24)
        meter.segment_gap = data.get("segment_gap", 2)
        meter.corner_radius = data.get("corner_radius", 2)

        # Border
        meter.border_color = data.get("border_color", "#333333")
        meter.border_width = data.get("border_width", 1)

        # Scale
        meter.show_scale = data.get("show_scale", False)
        meter.scale_color = data.get("scale_color", "#666666")

        # Zones
        zones_data = data.get("zones", None)
        if zones_data is not None:
            meter.zones = [MeterZone.from_dict(z) for z in zones_data]
        # else: keep default zones

        # Update geometry
        meter.geometry.width = meter._width
        meter.geometry.height = meter._height

        return meter


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
