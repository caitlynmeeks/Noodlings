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
#   LevelMeter Component Tests
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""Tests for the LevelMeter UI component."""

import pytest
import os
import sys

# Add the noodlestudio package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from noodlestudio.runtime.ui.components.level_meter import (
    LevelMeter, MeterZone, MeterOrientation, DEFAULT_ZONES
)


class TestMeterZone:
    """Tests for MeterZone dataclass."""

    def test_meter_zone_creation(self):
        """Test basic zone creation."""
        zone = MeterZone(0.0, 0.5, "#00ff00")
        assert zone.start_value == 0.0
        assert zone.end_value == 0.5
        assert zone.color == "#00ff00"

    def test_meter_zone_to_dict(self):
        """Test zone serialization."""
        zone = MeterZone(0.6, 0.8, "#ffcc00")
        data = zone.to_dict()

        assert data["start_value"] == 0.6
        assert data["end_value"] == 0.8
        assert data["color"] == "#ffcc00"

    def test_meter_zone_from_dict(self):
        """Test zone deserialization."""
        data = {
            "start_value": 0.8,
            "end_value": 1.0,
            "color": "#ff0000"
        }
        zone = MeterZone.from_dict(data)

        assert zone.start_value == 0.8
        assert zone.end_value == 1.0
        assert zone.color == "#ff0000"

    def test_meter_zone_from_dict_defaults(self):
        """Test zone deserialization with defaults."""
        zone = MeterZone.from_dict({})
        assert zone.start_value == 0.0
        assert zone.end_value == 1.0
        assert zone.color == "#33ff66"


class TestMeterOrientation:
    """Tests for MeterOrientation enum."""

    def test_vertical_value(self):
        """Test vertical enum value."""
        assert MeterOrientation.VERTICAL.value == "vertical"

    def test_horizontal_value(self):
        """Test horizontal enum value."""
        assert MeterOrientation.HORIZONTAL.value == "horizontal"


class TestLevelMeterBasics:
    """Tests for basic LevelMeter functionality."""

    def test_default_creation(self):
        """Test creating a default LevelMeter."""
        meter = LevelMeter()

        assert meter.value == 0.0
        assert meter.orientation == "vertical"
        assert meter.segments == 10
        assert meter.component_type == "LevelMeter"

    def test_creation_with_params(self):
        """Test creating LevelMeter with custom parameters."""
        meter = LevelMeter(
            name="arousal",
            value=0.75,
            orientation="horizontal",
            segments=16
        )

        assert meter.name == "arousal"
        assert meter.value == 0.75
        assert meter.orientation == "horizontal"
        assert meter.segments == 16

    def test_value_clamping_on_init(self):
        """Test that values are clamped to 0-1 range on init."""
        meter1 = LevelMeter(value=1.5)
        assert meter1.value == 1.0

        meter2 = LevelMeter(value=-0.5)
        assert meter2.value == 0.0

    def test_set_value(self):
        """Test set_value method."""
        meter = LevelMeter()
        meter.set_value(0.5)
        assert meter.value == 0.5

    def test_set_value_clamping(self):
        """Test set_value clamps values."""
        meter = LevelMeter()

        meter.set_value(1.5)
        assert meter.value == 1.0

        meter.set_value(-0.3)
        assert meter.value == 0.0

    def test_default_zones(self):
        """Test that default zones are set."""
        meter = LevelMeter()

        assert len(meter.zones) == 3
        assert meter.zones[0].start_value == 0.0
        assert meter.zones[0].end_value == 0.6
        assert meter.zones[0].color == "#33ff66"  # Green
        assert meter.zones[1].color == "#ffcc00"  # Yellow
        assert meter.zones[2].color == "#ff3344"  # Red

    def test_default_geometry_vertical(self):
        """Test default geometry for vertical orientation."""
        meter = LevelMeter(orientation="vertical")

        assert meter.width == 24
        assert meter.height == 120

    def test_default_geometry_horizontal(self):
        """Test default geometry for horizontal orientation."""
        meter = LevelMeter(orientation="horizontal")

        assert meter.width == 120
        assert meter.height == 24


class TestLevelMeterPeakHold:
    """Tests for peak hold functionality."""

    def test_peak_hold_disabled_by_default(self):
        """Test that peak hold is disabled by default."""
        meter = LevelMeter()
        assert meter.peak_hold is False
        assert meter.peak_value == 0.0

    def test_peak_value_updates_with_set_value(self):
        """Test that peak value updates when value increases."""
        meter = LevelMeter()
        meter.peak_hold = True

        meter.set_value(0.7)
        assert meter.peak_value == 0.7

        meter.set_value(0.5)  # Lower value
        assert meter.peak_value == 0.7  # Peak stays

        meter.set_value(0.9)  # Higher value
        assert meter.peak_value == 0.9  # Peak updates

    def test_reset_peak(self):
        """Test resetting peak to current value."""
        meter = LevelMeter(value=0.8)
        meter.peak_hold = True
        meter.peak_value = 0.8

        meter.set_value(0.3)
        assert meter.peak_value == 0.8  # Still high

        meter.reset_peak()
        assert meter.peak_value == 0.3  # Reset to current

    def test_peak_decay_default(self):
        """Test default peak decay value."""
        meter = LevelMeter()
        assert meter.peak_decay == 1.5


class TestLevelMeterZones:
    """Tests for zone management."""

    def test_add_zone(self):
        """Test adding a custom zone."""
        meter = LevelMeter()
        original_count = len(meter.zones)

        meter.add_zone(0.9, 1.0, "#ff00ff")

        assert len(meter.zones) == original_count + 1
        assert meter.zones[-1].start_value == 0.9
        assert meter.zones[-1].end_value == 1.0
        assert meter.zones[-1].color == "#ff00ff"

    def test_clear_zones(self):
        """Test clearing all zones."""
        meter = LevelMeter()
        assert len(meter.zones) > 0

        meter.clear_zones()
        assert len(meter.zones) == 0

    def test_set_default_zones(self):
        """Test resetting to default zones."""
        meter = LevelMeter()
        meter.clear_zones()
        assert len(meter.zones) == 0

        meter.set_default_zones()
        assert len(meter.zones) == 3

    def test_get_color_at_value_green_zone(self):
        """Test getting color in green zone."""
        meter = LevelMeter()

        color = meter.get_color_at_value(0.3)
        assert color == "#33ff66"  # Green

    def test_get_color_at_value_yellow_zone(self):
        """Test getting color in yellow zone."""
        meter = LevelMeter()

        color = meter.get_color_at_value(0.7)
        assert color == "#ffcc00"  # Yellow

    def test_get_color_at_value_red_zone(self):
        """Test getting color in red zone."""
        meter = LevelMeter()

        color = meter.get_color_at_value(0.9)
        assert color == "#ff3344"  # Red

    def test_get_color_at_value_max(self):
        """Test getting color at maximum value."""
        meter = LevelMeter()

        color = meter.get_color_at_value(1.0)
        assert color == "#ff3344"  # Red (last zone)

    def test_get_color_at_value_empty_zones(self):
        """Test fallback when zones are empty."""
        meter = LevelMeter()
        meter.clear_zones()

        color = meter.get_color_at_value(0.5)
        assert color == LevelMeter.COLOR_GREEN


class TestLevelMeterSegments:
    """Tests for segment calculations."""

    def test_get_segment_count_lit_zero(self):
        """Test segment count when value is 0."""
        meter = LevelMeter(value=0.0, segments=10)
        assert meter.get_segment_count_lit() == 0

    def test_get_segment_count_lit_full(self):
        """Test segment count when value is 1."""
        meter = LevelMeter(value=1.0, segments=10)
        assert meter.get_segment_count_lit() == 10

    def test_get_segment_count_lit_half(self):
        """Test segment count at 50%."""
        meter = LevelMeter(value=0.5, segments=10)
        assert meter.get_segment_count_lit() == 5

    def test_get_segment_count_lit_rounding(self):
        """Test segment count rounding."""
        meter = LevelMeter(value=0.45, segments=10)
        # 0.45 * 10 + 0.5 = 5.0 -> rounds to 5
        assert meter.get_segment_count_lit() == 5

        meter.set_value(0.46)
        # 0.46 * 10 + 0.5 = 5.1 -> rounds to 5
        assert meter.get_segment_count_lit() == 5

    def test_get_segment_count_lit_continuous_mode(self):
        """Test segment count in continuous mode (segments=0)."""
        meter = LevelMeter(value=0.5, segments=0)
        assert meter.get_segment_count_lit() == 0

    def test_get_peak_segment(self):
        """Test peak segment calculation."""
        meter = LevelMeter(segments=10)
        meter.peak_hold = True
        meter.peak_value = 0.8

        # 0.8 * 10 + 0.5 = 8.5 -> 8, then -1 = 7 (0-indexed)
        assert meter.get_peak_segment() == 7

    def test_get_peak_segment_max(self):
        """Test peak segment at maximum."""
        meter = LevelMeter(segments=10)
        meter.peak_hold = True
        meter.peak_value = 1.0

        # Should cap at last segment (9)
        assert meter.get_peak_segment() == 9

    def test_get_peak_segment_disabled(self):
        """Test peak segment when peak_hold is disabled."""
        meter = LevelMeter(segments=10)
        meter.peak_hold = False
        meter.peak_value = 0.8

        assert meter.get_peak_segment() == -1

    def test_get_peak_segment_continuous_mode(self):
        """Test peak segment in continuous mode."""
        meter = LevelMeter(segments=0)
        meter.peak_hold = True
        meter.peak_value = 0.8

        assert meter.get_peak_segment() == -1


class TestLevelMeterSerialization:
    """Tests for LevelMeter serialization."""

    def test_to_dict_defaults(self):
        """Test serialization with default values."""
        meter = LevelMeter(name="test_meter")
        data = meter.to_dict()

        assert data["type"] == "LevelMeter"
        assert data["name"] == "test_meter"
        assert data["value"] == 0.0
        assert data["orientation"] == "vertical"
        assert data["segments"] == 10
        assert "zones" not in data  # Default zones not serialized

    def test_to_dict_custom_values(self):
        """Test serialization with custom values."""
        meter = LevelMeter(
            name="arousal",
            value=0.65,
            orientation="horizontal",
            segments=16
        )
        meter.peak_hold = True
        meter.peak_value = 0.8
        meter.glow = 0.8
        meter.background_color = "#222222"

        data = meter.to_dict()

        assert data["value"] == 0.65
        assert data["orientation"] == "horizontal"
        assert data["segments"] == 16
        assert data["peak_hold"] is True
        assert data["peak_value"] == 0.8
        assert data["glow"] == 0.8
        assert data["background_color"] == "#222222"

    def test_to_dict_custom_zones(self):
        """Test serialization with custom zones."""
        meter = LevelMeter()
        meter.clear_zones()
        meter.add_zone(0.0, 1.0, "#0000ff")

        data = meter.to_dict()

        assert "zones" in data
        assert len(data["zones"]) == 1
        assert data["zones"][0]["color"] == "#0000ff"

    def test_from_dict_defaults(self):
        """Test deserialization with minimal data."""
        data = {"type": "LevelMeter", "name": "meter1"}
        meter = LevelMeter.from_dict(data)

        assert meter.name == "meter1"
        assert meter.value == 0.0
        assert meter.orientation == "vertical"
        assert meter.segments == 10
        assert len(meter.zones) == 3  # Default zones

    def test_from_dict_full(self):
        """Test deserialization with full data."""
        data = {
            "type": "LevelMeter",
            "name": "valence_meter",
            "value": 0.45,
            "orientation": "horizontal",
            "segments": 20,
            "peak_hold": True,
            "peak_value": 0.6,
            "peak_decay": 2.0,
            "background_color": "#111111",
            "inactive_color": "#333333",
            "glow": 0.6,
            "width": 200,
            "height": 30,
            "segment_gap": 3,
            "corner_radius": 4,
            "border_color": "#444444",
            "border_width": 2,
            "show_scale": True,
            "scale_color": "#888888",
            "zones": [
                {"start_value": 0.0, "end_value": 1.0, "color": "#00ffff"}
            ]
        }
        meter = LevelMeter.from_dict(data)

        assert meter.name == "valence_meter"
        assert meter.value == 0.45
        assert meter.orientation == "horizontal"
        assert meter.segments == 20
        assert meter.peak_hold is True
        assert meter.peak_value == 0.6
        assert meter.peak_decay == 2.0
        assert meter.background_color == "#111111"
        assert meter.inactive_color == "#333333"
        assert meter.glow == 0.6
        assert meter.width == 200
        assert meter.height == 30
        assert meter.segment_gap == 3
        assert meter.corner_radius == 4
        assert meter.border_color == "#444444"
        assert meter.border_width == 2
        assert meter.show_scale is True
        assert meter.scale_color == "#888888"
        assert len(meter.zones) == 1
        assert meter.zones[0].color == "#00ffff"

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrips correctly."""
        original = LevelMeter(
            name="roundtrip_test",
            value=0.77,
            orientation="horizontal",
            segments=12
        )
        original.peak_hold = True
        original.glow = 0.9
        original.clear_zones()
        original.add_zone(0.0, 0.5, "#00ff00")
        original.add_zone(0.5, 1.0, "#ff0000")

        data = original.to_dict()
        restored = LevelMeter.from_dict(data)

        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.orientation == original.orientation
        assert restored.segments == original.segments
        assert restored.peak_hold == original.peak_hold
        assert restored.glow == original.glow
        assert len(restored.zones) == len(original.zones)


class TestLevelMeterYAML:
    """Tests for LevelMeter YAML loading via UILoader."""

    def test_yaml_load_simple(self):
        """Test loading LevelMeter from YAML."""
        import yaml
        from noodlestudio.runtime.ui.loader import UILoader

        yaml_content = """
version: 1
root:
  type: LevelMeter
  name: test_meter
  value: 0.6
  orientation: vertical
  segments: 10
"""
        data = yaml.safe_load(yaml_content)
        loader = UILoader()
        component = loader.load_dict(data)

        assert component.component_type == "LevelMeter"
        assert component.name == "test_meter"
        assert component.value == 0.6
        assert component.segments == 10

    def test_yaml_load_with_zones(self):
        """Test loading LevelMeter with custom zones from YAML."""
        import yaml
        from noodlestudio.runtime.ui.loader import UILoader

        yaml_content = """
version: 1
root:
  type: LevelMeter
  name: custom_zones_meter
  value: 0.5
  zones:
    - start_value: 0.0
      end_value: 0.3
      color: "#00ff00"
    - start_value: 0.3
      end_value: 0.7
      color: "#ffff00"
    - start_value: 0.7
      end_value: 1.0
      color: "#ff0000"
"""
        data = yaml.safe_load(yaml_content)
        loader = UILoader()
        component = loader.load_dict(data)

        assert len(component.zones) == 3
        assert component.zones[0].end_value == 0.3
        assert component.zones[1].color == "#ffff00"

    def test_yaml_load_horizontal(self):
        """Test loading horizontal LevelMeter from YAML."""
        import yaml
        from noodlestudio.runtime.ui.loader import UILoader

        yaml_content = """
version: 1
root:
  type: LevelMeter
  name: horizontal_meter
  orientation: horizontal
  segments: 20
  width: 200
  height: 24
"""
        data = yaml.safe_load(yaml_content)
        loader = UILoader()
        component = loader.load_dict(data)

        assert component.orientation == "horizontal"
        assert component.segments == 20
        assert component.width == 200
        assert component.height == 24


class TestLevelMeterRendering:
    """Tests for LevelMeter rendering widget."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for widget tests."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_widget_creation(self, qapp):
        """Test creating LevelMeterWidget."""
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer, LevelMeterWidget

        meter = LevelMeter(name="test", value=0.5)
        renderer = QtWidgetRenderer()

        widget = LevelMeterWidget(meter, renderer)

        assert widget is not None
        assert widget.component is meter

    def test_widget_size_hint(self, qapp):
        """Test widget size hint matches component dimensions."""
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer, LevelMeterWidget
        from PyQt6.QtCore import QSize

        meter = LevelMeter(name="test")
        meter.width = 30
        meter.height = 150

        renderer = QtWidgetRenderer()
        widget = LevelMeterWidget(meter, renderer)

        hint = widget.sizeHint()
        assert hint == QSize(30, 150)

    def test_widget_update_from_component(self, qapp):
        """Test update_from_component triggers repaint."""
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer, LevelMeterWidget

        meter = LevelMeter(name="test", value=0.3)
        renderer = QtWidgetRenderer()
        widget = LevelMeterWidget(meter, renderer)

        # This should not raise
        widget.update_from_component()

    def test_render_via_renderer(self, qapp):
        """Test rendering LevelMeter via QtWidgetRenderer."""
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer, LevelMeterWidget

        meter = LevelMeter(name="rendered_meter", value=0.7)
        renderer = QtWidgetRenderer()

        widget = renderer._render_level_meter(meter, None)

        assert isinstance(widget, LevelMeterWidget)
        assert widget.objectName() == "rendered_meter"


class TestLevelMeterRegistry:
    """Tests for LevelMeter component registration."""

    def test_component_registered(self):
        """Test that LevelMeter is registered in component registry."""
        from noodlestudio.runtime.ui.component import get_component_class

        cls = get_component_class("LevelMeter")
        assert cls is LevelMeter

    def test_component_type_matches(self):
        """Test component_type class attribute."""
        assert LevelMeter.component_type == "LevelMeter"


class TestLevelMeterColorPresets:
    """Tests for color preset constants."""

    def test_color_presets_exist(self):
        """Test that color preset constants exist."""
        assert LevelMeter.COLOR_GREEN == "#33ff66"
        assert LevelMeter.COLOR_YELLOW == "#ffcc00"
        assert LevelMeter.COLOR_RED == "#ff3344"
        assert LevelMeter.COLOR_BLUE == "#3399ff"
        assert LevelMeter.COLOR_CYAN == "#00ffcc"


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
