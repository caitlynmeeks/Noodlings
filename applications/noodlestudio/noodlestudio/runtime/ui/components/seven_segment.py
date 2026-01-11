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
#   Seven-Segment Display Component
#
#   Classic digital display for numeric values. Perfect for
#   dashboards, clocks, counters, and retro instrumentation.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.seven_segment
# PURPOSE:  Seven-Segment Display Component
# LAYER:    Studio / UI Components / Dashboard
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SevenSegment
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

from ..component import UIComponent, register_component


class SegmentStyle(Enum):
    """Visual style for segment rendering."""
    CLASSIC = "classic"      # Sharp-edged classic LCD look
    ROUNDED = "rounded"      # Rounded segment ends
    SLANTED = "slanted"      # Italic/slanted segments


# Segment bit patterns for digits 0-9 and A-F
# Segments are numbered:
#    aaaa
#   f    b
#   f    b
#    gggg
#   e    c
#   e    c
#    dddd
#
# Bit order: abcdefg (MSB to LSB)
SEGMENT_PATTERNS = {
    '0': 0b1111110,  # abcdef
    '1': 0b0110000,  # bc
    '2': 0b1101101,  # abdeg
    '3': 0b1111001,  # abcdg
    '4': 0b0110011,  # bcfg
    '5': 0b1011011,  # acdfg
    '6': 0b1011111,  # acdefg
    '7': 0b1110000,  # abc
    '8': 0b1111111,  # abcdefg
    '9': 0b1111011,  # abcdfg
    'A': 0b1110111,  # abcefg
    'B': 0b0011111,  # cdefg
    'C': 0b1001110,  # adef
    'D': 0b0111101,  # bcdeg
    'E': 0b1001111,  # adefg
    'F': 0b1000111,  # aefg
    '-': 0b0000001,  # g (minus sign)
    ' ': 0b0000000,  # blank
    '_': 0b0001000,  # d (underscore)
}


@register_component
class SevenSegment(UIComponent):
    """
    Seven-segment digital display component.

    A classic LCD/LED-style numeric display with configurable digits,
    colors, and optional decimal point. Perfect for dashboards,
    counters, clocks, and retro instrumentation aesthetic.

    Properties:
        value: Numeric value to display (int or float)
        digit_count: Number of digits to show
        decimal_places: Decimal places for float values
        show_leading_zeros: Pad with zeros (e.g., "007" vs "  7")

        # Appearance
        on_color: Color of lit segments
        off_color: Color of unlit segments (dim)
        background_color: Display background
        glow: Enable glow effect on lit segments

        # Sizing
        digit_height: Height of each digit in pixels
        digit_width: Width of each digit (auto-calculated if 0)
        segment_thickness: Thickness of segments
        digit_spacing: Gap between digits
        slant_angle: Italic angle in degrees (0 = upright)

        # Options
        style: Segment style (classic, rounded, slanted)
        show_decimal_point: Show decimal point indicator
        hex_mode: Display value in hexadecimal

    Events:
        onChange: Triggered when value changes

    Example:
        ```yaml
        - type: SevenSegment
          name: counter
          digit_count: 4
          value: 42
          on_color: "#00ff00"
          show_leading_zeros: true
        ```
    """

    component_type = "SevenSegment"

    # Standard LED colors
    COLOR_RED = "#ff3333"
    COLOR_GREEN = "#33ff66"
    COLOR_BLUE = "#3399ff"
    COLOR_AMBER = "#ffaa00"
    COLOR_WHITE = "#ffffff"

    def __init__(
        self,
        name: str = "",
        value: float = 0,
        digit_count: int = 4
    ):
        super().__init__(name)

        # Value
        self.value: float = value
        self.digit_count: int = digit_count
        self.decimal_places: int = 0
        self.show_leading_zeros: bool = False

        # Appearance - classic green LCD
        self.on_color: str = "#33ff66"
        self.off_color: str = ""  # Auto-calculated if empty
        self.background_color: str = "#1a1a1a"
        self.glow: bool = True

        # Sizing
        self.digit_height: int = 40
        self.digit_width: int = 0  # Auto-calculate based on height
        self.segment_thickness: int = 0  # Auto-calculate
        self.digit_spacing: int = 4
        self.slant_angle: float = 0  # Degrees

        # Options
        self.style: str = SegmentStyle.CLASSIC.value
        self.show_decimal_point: bool = True
        self.hex_mode: bool = False

        # Calculate default geometry
        self._update_geometry()

    def _update_geometry(self) -> None:
        """Update geometry based on digit count and sizing."""
        # Auto-calculate digit width if not set
        width = self.digit_width if self.digit_width > 0 else int(self.digit_height * 0.6)

        # Total width: digits + spacing + potential decimal points
        total_width = (width * self.digit_count) + (self.digit_spacing * (self.digit_count - 1))
        if self.show_decimal_point and self.decimal_places > 0:
            total_width += self.digit_spacing  # Extra space for decimal point

        self.geometry.width = total_width + 16  # Padding
        self.geometry.height = self.digit_height + 16

    def set_value(self, value: float) -> None:
        """Set the display value."""
        self.value = value

    def get_display_string(self) -> str:
        """
        Get the string representation for display.

        Returns:
            String of characters to display, including decimal point marker.
        """
        if self.hex_mode:
            # Hexadecimal display
            int_val = int(abs(self.value))
            hex_str = format(int_val, 'X')
            if self.value < 0:
                hex_str = '-' + hex_str

            if self.show_leading_zeros:
                # Pad with zeros
                if self.value < 0:
                    hex_str = '-' + hex_str[1:].zfill(self.digit_count - 1)
                else:
                    hex_str = hex_str.zfill(self.digit_count)
            else:
                # Pad with spaces
                hex_str = hex_str.rjust(self.digit_count)

            return hex_str[:self.digit_count]

        # Decimal display
        if self.decimal_places > 0:
            # Float with decimal places
            format_str = f"{{:.{self.decimal_places}f}}"
            num_str = format_str.format(self.value)

            # Handle overflow
            parts = num_str.split('.')
            int_part = parts[0]
            dec_part = parts[1] if len(parts) > 1 else ''

            # Check if it fits
            available_int_digits = self.digit_count - self.decimal_places - 1  # -1 for decimal point
            if len(int_part.lstrip('-')) > available_int_digits:
                # Overflow - show dashes
                return '-' * self.digit_count

            if self.show_leading_zeros:
                if self.value < 0:
                    int_part = '-' + int_part[1:].zfill(available_int_digits)
                else:
                    int_part = int_part.zfill(available_int_digits)
            else:
                int_part = int_part.rjust(available_int_digits)

            return int_part + '.' + dec_part

        else:
            # Integer display
            int_val = int(self.value)
            num_str = str(int_val)

            if len(num_str.lstrip('-')) > self.digit_count:
                # Overflow
                return '-' * self.digit_count

            if self.show_leading_zeros:
                if int_val < 0:
                    num_str = '-' + num_str[1:].zfill(self.digit_count - 1)
                else:
                    num_str = num_str.zfill(self.digit_count)
            else:
                num_str = num_str.rjust(self.digit_count)

            return num_str[:self.digit_count]

    def get_segment_pattern(self, char: str) -> int:
        """
        Get the segment bit pattern for a character.

        Args:
            char: Single character to get pattern for

        Returns:
            7-bit integer where each bit represents a segment
        """
        return SEGMENT_PATTERNS.get(char.upper(), 0)

    def get_effective_off_color(self) -> str:
        """Get the off color, auto-calculating if not set."""
        if self.off_color:
            return self.off_color

        # Auto-calculate: 15% brightness of on_color
        on = self.on_color.lstrip('#')
        r = int(on[0:2], 16)
        g = int(on[2:4], 16)
        b = int(on[4:6], 16)

        r = int(r * 0.15)
        g = int(g * 0.15)
        b = int(b * 0.15)

        return f"#{r:02x}{g:02x}{b:02x}"

    def get_effective_digit_width(self) -> int:
        """Get digit width, auto-calculating if not set."""
        if self.digit_width > 0:
            return self.digit_width
        return int(self.digit_height * 0.6)

    def get_effective_segment_thickness(self) -> int:
        """Get segment thickness, auto-calculating if not set."""
        if self.segment_thickness > 0:
            return self.segment_thickness
        return max(2, int(self.digit_height * 0.12))

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add SevenSegment-specific properties to serialization."""
        data["value"] = self.value
        data["digit_count"] = self.digit_count

        if self.decimal_places != 0:
            data["decimal_places"] = self.decimal_places
        if self.show_leading_zeros:
            data["show_leading_zeros"] = True

        # Appearance (only if non-default)
        if self.on_color != "#33ff66":
            data["on_color"] = self.on_color
        if self.off_color:
            data["off_color"] = self.off_color
        if self.background_color != "#1a1a1a":
            data["background_color"] = self.background_color
        if not self.glow:
            data["glow"] = False

        # Sizing (only if non-default)
        if self.digit_height != 40:
            data["digit_height"] = self.digit_height
        if self.digit_width != 0:
            data["digit_width"] = self.digit_width
        if self.segment_thickness != 0:
            data["segment_thickness"] = self.segment_thickness
        if self.digit_spacing != 4:
            data["digit_spacing"] = self.digit_spacing
        if self.slant_angle != 0:
            data["slant_angle"] = self.slant_angle

        # Options
        if self.style != SegmentStyle.CLASSIC.value:
            data["style"] = self.style
        if not self.show_decimal_point:
            data["show_decimal_point"] = False
        if self.hex_mode:
            data["hex_mode"] = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SevenSegment':
        """Deserialize from dictionary."""
        display = cls(
            name=data.get("name", ""),
            value=data.get("value", 0),
            digit_count=data.get("digit_count", 4)
        )

        display._apply_base_properties(data)

        # Value options
        display.decimal_places = data.get("decimal_places", 0)
        display.show_leading_zeros = data.get("show_leading_zeros", False)

        # Appearance
        display.on_color = data.get("on_color", "#33ff66")
        display.off_color = data.get("off_color", "")
        display.background_color = data.get("background_color", "#1a1a1a")
        display.glow = data.get("glow", True)

        # Sizing
        display.digit_height = data.get("digit_height", 40)
        display.digit_width = data.get("digit_width", 0)
        display.segment_thickness = data.get("segment_thickness", 0)
        display.digit_spacing = data.get("digit_spacing", 4)
        display.slant_angle = data.get("slant_angle", 0)

        # Options
        display.style = data.get("style", SegmentStyle.CLASSIC.value)
        display.show_decimal_point = data.get("show_decimal_point", True)
        display.hex_mode = data.get("hex_mode", False)

        # Update geometry
        display._update_geometry()

        # Override if explicitly set
        if "width" in data:
            display.geometry.width = data["width"]
        if "height" in data:
            display.geometry.height = data["height"]

        return display


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
