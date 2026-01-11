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
#   Seven-Segment Display Tests
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
from pathlib import Path
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestSegmentPatterns:
    """Tests for segment bit patterns."""

    def test_digit_zero(self):
        """Test pattern for digit 0."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        # 0 lights segments a,b,c,d,e,f (all except g)
        assert SEGMENT_PATTERNS['0'] == 0b1111110

    def test_digit_one(self):
        """Test pattern for digit 1."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        # 1 lights segments b,c only
        assert SEGMENT_PATTERNS['1'] == 0b0110000

    def test_digit_eight(self):
        """Test pattern for digit 8."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        # 8 lights all segments
        assert SEGMENT_PATTERNS['8'] == 0b1111111

    def test_hex_a(self):
        """Test pattern for hex A."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        assert SEGMENT_PATTERNS['A'] == 0b1110111

    def test_minus_sign(self):
        """Test pattern for minus sign."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        # Minus lights only segment g
        assert SEGMENT_PATTERNS['-'] == 0b0000001

    def test_blank(self):
        """Test pattern for blank/space."""
        from noodlestudio.runtime.ui.components.seven_segment import SEGMENT_PATTERNS
        assert SEGMENT_PATTERNS[' '] == 0b0000000


class TestSevenSegment:
    """Tests for SevenSegment component."""

    def test_create_display(self):
        """Test creating a seven-segment display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(name="counter")
        assert display.name == "counter"
        assert display.component_type == "SevenSegment"
        assert display.value == 0
        assert display.digit_count == 4

    def test_create_with_value(self):
        """Test creating display with initial value."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(name="score", value=42, digit_count=3)
        assert display.value == 42
        assert display.digit_count == 3

    def test_set_value(self):
        """Test setting display value."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.set_value(123)
        assert display.value == 123

    def test_default_colors(self):
        """Test default color values."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        assert display.on_color == "#33ff66"  # Green
        assert display.off_color == ""  # Auto-calculated
        assert display.background_color == "#1a1a1a"

    def test_color_constants(self):
        """Test color constant values."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        assert SevenSegment.COLOR_RED == "#ff3333"
        assert SevenSegment.COLOR_GREEN == "#33ff66"
        assert SevenSegment.COLOR_BLUE == "#3399ff"
        assert SevenSegment.COLOR_AMBER == "#ffaa00"

    def test_glow_default_enabled(self):
        """Test that glow is enabled by default."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        assert display.glow is True


class TestDisplayString:
    """Tests for display string generation."""

    def test_integer_display(self):
        """Test integer value display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.set_value(42)
        result = display.get_display_string()
        assert result == "  42"

    def test_leading_zeros(self):
        """Test display with leading zeros."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.show_leading_zeros = True
        display.set_value(7)
        result = display.get_display_string()
        assert result == "0007"

    def test_negative_value(self):
        """Test negative value display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.set_value(-5)
        result = display.get_display_string()
        assert result == "  -5"

    def test_negative_with_leading_zeros(self):
        """Test negative value with leading zeros."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.show_leading_zeros = True
        display.set_value(-5)
        result = display.get_display_string()
        assert result == "-005"

    def test_overflow_shows_dashes(self):
        """Test that overflow shows dashes."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=3)
        display.set_value(9999)  # Too big for 3 digits
        result = display.get_display_string()
        assert result == "---"

    def test_float_display(self):
        """Test float value display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=5)
        display.decimal_places = 2
        display.set_value(3.14)
        result = display.get_display_string()
        assert result == " 3.14"

    def test_float_with_leading_zeros(self):
        """Test float with leading zeros."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=5)
        display.decimal_places = 2
        display.show_leading_zeros = True
        display.set_value(3.14)
        result = display.get_display_string()
        assert result == "03.14"

    def test_hex_mode(self):
        """Test hexadecimal display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.hex_mode = True
        display.set_value(255)
        result = display.get_display_string()
        assert result == "  FF"

    def test_hex_with_leading_zeros(self):
        """Test hex with leading zeros."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(digit_count=4)
        display.hex_mode = True
        display.show_leading_zeros = True
        display.set_value(15)
        result = display.get_display_string()
        assert result == "000F"


class TestSegmentPattern:
    """Tests for segment pattern retrieval."""

    def test_get_digit_pattern(self):
        """Test getting segment pattern for a digit."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        pattern = display.get_segment_pattern('8')
        assert pattern == 0b1111111

    def test_get_pattern_case_insensitive(self):
        """Test that pattern lookup is case insensitive."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        assert display.get_segment_pattern('a') == display.get_segment_pattern('A')
        assert display.get_segment_pattern('f') == display.get_segment_pattern('F')

    def test_unknown_char_returns_zero(self):
        """Test that unknown character returns zero pattern."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        pattern = display.get_segment_pattern('X')  # Not in pattern table
        assert pattern == 0


class TestEffectiveValues:
    """Tests for auto-calculated effective values."""

    def test_effective_off_color_auto(self):
        """Test auto-calculated off color."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.on_color = "#ff0000"  # Pure red
        off = display.get_effective_off_color()
        # Should be 15% of red
        assert off == "#260000"

    def test_effective_off_color_explicit(self):
        """Test explicit off color."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.off_color = "#333333"
        assert display.get_effective_off_color() == "#333333"

    def test_effective_digit_width_auto(self):
        """Test auto-calculated digit width."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.digit_height = 50
        display.digit_width = 0  # Auto
        # Should be 60% of height
        assert display.get_effective_digit_width() == 30

    def test_effective_digit_width_explicit(self):
        """Test explicit digit width."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.digit_width = 40
        assert display.get_effective_digit_width() == 40

    def test_effective_segment_thickness_auto(self):
        """Test auto-calculated segment thickness."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.digit_height = 50
        display.segment_thickness = 0  # Auto
        # Should be 12% of height, minimum 2
        assert display.get_effective_segment_thickness() == 6

    def test_effective_segment_thickness_minimum(self):
        """Test minimum segment thickness."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment()
        display.digit_height = 10  # Small
        display.segment_thickness = 0
        # 12% of 10 = 1.2, but minimum is 2
        assert display.get_effective_segment_thickness() == 2


class TestSevenSegmentSerialization:
    """Tests for SevenSegment serialization."""

    def test_to_dict_minimal(self):
        """Test serializing minimal display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(name="counter")
        data = display.to_dict()

        assert data["type"] == "SevenSegment"
        assert data["name"] == "counter"
        assert data["value"] == 0
        assert data["digit_count"] == 4

    def test_to_dict_full(self):
        """Test serializing display with all properties."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        display = SevenSegment(name="full", value=123, digit_count=6)
        display.decimal_places = 2
        display.show_leading_zeros = True
        display.on_color = "#ff0000"
        display.off_color = "#330000"
        display.glow = False
        display.digit_height = 60
        display.hex_mode = True

        data = display.to_dict()

        assert data["value"] == 123
        assert data["digit_count"] == 6
        assert data["decimal_places"] == 2
        assert data["show_leading_zeros"] is True
        assert data["on_color"] == "#ff0000"
        assert data["off_color"] == "#330000"
        assert data["glow"] is False
        assert data["digit_height"] == 60
        assert data["hex_mode"] is True

    def test_from_dict_minimal(self):
        """Test deserializing minimal display."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        data = {"name": "test", "value": 42}
        display = SevenSegment.from_dict(data)

        assert display.name == "test"
        assert display.value == 42
        assert display.digit_count == 4  # Default

    def test_from_dict_full(self):
        """Test deserializing display with all properties."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        data = {
            "name": "full",
            "value": 99,
            "digit_count": 3,
            "decimal_places": 1,
            "show_leading_zeros": True,
            "on_color": "#00ff00",
            "off_color": "#003300",
            "background_color": "#000000",
            "glow": False,
            "digit_height": 50,
            "digit_spacing": 8,
            "hex_mode": True
        }

        display = SevenSegment.from_dict(data)

        assert display.value == 99
        assert display.digit_count == 3
        assert display.decimal_places == 1
        assert display.show_leading_zeros is True
        assert display.on_color == "#00ff00"
        assert display.off_color == "#003300"
        assert display.glow is False
        assert display.digit_height == 50
        assert display.digit_spacing == 8
        assert display.hex_mode is True

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment

        original = SevenSegment(name="test", value=42, digit_count=5)
        original.on_color = "#ff6600"
        original.show_leading_zeros = True

        data = original.to_dict()
        restored = SevenSegment.from_dict(data)

        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.digit_count == original.digit_count
        assert restored.on_color == original.on_color
        assert restored.show_leading_zeros == original.show_leading_zeros


class TestSevenSegmentYAML:
    """Tests for SevenSegment YAML loading."""

    def test_load_from_yaml(self):
        """Test loading seven-segment from YAML."""
        from noodlestudio.runtime.ui.loader import UILoader
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: root
  children:
    - type: SevenSegment
      name: counter
      value: 42
      digit_count: 4
      on_color: "#00ff00"
      show_leading_zeros: true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_yaml = Path(tmpdir) / "ui.yaml"
            ui_yaml.write_text(yaml_content)

            loader = UILoader()
            root = loader.load_file(str(ui_yaml))

            display = root.find_by_name("counter")
            assert display is not None
            assert display.component_type == "SevenSegment"
            assert display.value == 42
            assert display.digit_count == 4
            assert display.on_color == "#00ff00"
            assert display.show_leading_zeros is True

    def test_load_hex_display(self):
        """Test loading hex mode display from YAML."""
        from noodlestudio.runtime.ui.loader import UILoader
        import tempfile

        yaml_content = """
version: 1
root:
  type: SevenSegment
  name: hex_display
  value: 255
  digit_count: 4
  hex_mode: true
  on_color: "#3399ff"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_yaml = Path(tmpdir) / "ui.yaml"
            ui_yaml.write_text(yaml_content)

            loader = UILoader()
            display = loader.load_file(str(ui_yaml))

            assert display.hex_mode is True
            assert display.get_display_string() == "  FF"


class TestSevenSegmentRendering:
    """Tests for SevenSegment rendering (requires Qt)."""

    @pytest.fixture
    def qtbot(self, qapp):
        """Create a QtBot for widget testing."""
        from pytestqt.qtbot import QtBot
        return QtBot(qapp)

    @pytest.fixture
    def qapp(self):
        """Create or get QApplication."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_render_display(self, qapp):
        """Test that display renders without error."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer

        display = SevenSegment(name="test", value=42)
        renderer = QtWidgetRenderer()

        widget = renderer.render(display)

        assert widget is not None
        # Widget should have the display reference
        assert display._widget is not None

    def test_render_different_values(self, qapp):
        """Test rendering various values."""
        from noodlestudio.runtime.ui.components.seven_segment import SevenSegment
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer

        test_values = [0, 1, 42, 99, -5, 123.45]
        renderer = QtWidgetRenderer()

        for val in test_values:
            display = SevenSegment(value=val)
            if isinstance(val, float):
                display.decimal_places = 2
            widget = renderer.render(display)
            assert widget is not None


class TestSevenSegmentRegistry:
    """Tests for SevenSegment registration."""

    def test_registered_in_registry(self):
        """Test that SevenSegment is registered."""
        from noodlestudio.runtime.ui.component import get_component_class, list_component_types

        assert "SevenSegment" in list_component_types()
        cls = get_component_class("SevenSegment")
        assert cls is not None
        assert cls.component_type == "SevenSegment"

    def test_exported_from_components(self):
        """Test that SevenSegment is exported from components module."""
        from noodlestudio.runtime.ui.components import SevenSegment, SegmentStyle

        display = SevenSegment(name="test")
        assert display.component_type == "SevenSegment"

        # Test SegmentStyle enum
        assert SegmentStyle.CLASSIC.value == "classic"
        assert SegmentStyle.ROUNDED.value == "rounded"


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
