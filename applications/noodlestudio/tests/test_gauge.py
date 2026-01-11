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
#   Tests for the Gauge Dashboard Widget
#
#   Tests the analog gauge component for dashboard displays.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_gauge
# PURPOSE:  Tests for the Gauge Dashboard Widget
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestGauge, TestGaugeZone, TestGaugeSerialization, TestGaugeRendering
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest


class TestGaugeZone:
    """Test GaugeZone dataclass."""

    def test_zone_creation(self):
        """Test GaugeZone creation."""
        from noodlestudio.runtime.ui.components.gauge import GaugeZone

        zone = GaugeZone(start_value=0, end_value=50, color="#00ff00")
        assert zone.start_value == 0
        assert zone.end_value == 50
        assert zone.color == "#00ff00"

    def test_zone_to_dict(self):
        """Test GaugeZone serialization."""
        from noodlestudio.runtime.ui.components.gauge import GaugeZone

        zone = GaugeZone(start_value=75, end_value=100, color="#ff0000")
        data = zone.to_dict()

        assert data["start_value"] == 75
        assert data["end_value"] == 100
        assert data["color"] == "#ff0000"

    def test_zone_from_dict(self):
        """Test GaugeZone deserialization."""
        from noodlestudio.runtime.ui.components.gauge import GaugeZone

        data = {"start_value": 25, "end_value": 75, "color": "#ffcc00"}
        zone = GaugeZone.from_dict(data)

        assert zone.start_value == 25
        assert zone.end_value == 75
        assert zone.color == "#ffcc00"


class TestGauge:
    """Test Gauge component class."""

    def test_gauge_creation_defaults(self):
        """Test Gauge creation with default values."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(name="test_gauge")
        assert gauge.name == "test_gauge"
        assert gauge.component_type == "Gauge"
        assert gauge.value == 0
        assert gauge.min_value == 0
        assert gauge.max_value == 100
        assert gauge.size == 120
        assert gauge.start_angle == 225
        assert gauge.sweep_angle == -270

    def test_gauge_creation_custom(self):
        """Test Gauge creation with custom values."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(name="speed", value=75, min_value=0, max_value=200, size=150)
        assert gauge.value == 75
        assert gauge.min_value == 0
        assert gauge.max_value == 200
        assert gauge.size == 150

    def test_gauge_set_value(self):
        """Test set_value method clamps to range."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(min_value=0, max_value=100)

        gauge.set_value(50)
        assert gauge.value == 50

        gauge.set_value(150)  # Over max
        assert gauge.value == 100

        gauge.set_value(-50)  # Under min
        assert gauge.value == 0

    def test_gauge_normalized_value(self):
        """Test get_normalized_value returns 0-1 fraction."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(min_value=0, max_value=100, value=50)
        assert gauge.get_normalized_value() == 0.5

        gauge.value = 0
        assert gauge.get_normalized_value() == 0.0

        gauge.value = 100
        assert gauge.get_normalized_value() == 1.0

    def test_gauge_normalized_value_custom_range(self):
        """Test normalization with custom range."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(min_value=50, max_value=150, value=100)
        assert gauge.get_normalized_value() == 0.5

    def test_gauge_normalized_value_zero_range(self):
        """Test normalization with zero range (edge case)."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(min_value=50, max_value=50, value=50)
        assert gauge.get_normalized_value() == 0

    def test_gauge_needle_angle(self):
        """Test needle angle calculation."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(min_value=0, max_value=100, value=0)
        gauge.start_angle = 225
        gauge.sweep_angle = -270

        # At value=0, needle should be at start_angle
        assert gauge.get_needle_angle() == 225

        # At value=100, needle should be at start + sweep
        gauge.value = 100
        assert gauge.get_needle_angle() == 225 + (-270)  # -45

        # At midpoint
        gauge.value = 50
        assert gauge.get_needle_angle() == 225 + (-135)  # 90

    def test_gauge_formatted_value(self):
        """Test get_formatted_value method."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(value=75.5)
        assert gauge.get_formatted_value() == "76"  # Default format is {:.0f}

        gauge.value_format = "{:.1f}"
        assert gauge.get_formatted_value() == "75.5"

        gauge.value_suffix = "%"
        assert gauge.get_formatted_value() == "75.5%"

    def test_gauge_add_zone(self):
        """Test adding color zones."""
        from noodlestudio.runtime.ui.components.gauge import Gauge, GaugeZone

        gauge = Gauge()
        assert len(gauge.zones) == 0

        gauge.add_zone(0, 50, "#00ff00")
        gauge.add_zone(50, 75, "#ffcc00")
        gauge.add_zone(75, 100, "#ff0000")

        assert len(gauge.zones) == 3
        assert gauge.zones[0].color == "#00ff00"
        assert gauge.zones[2].color == "#ff0000"

    def test_gauge_clear_zones(self):
        """Test clearing zones."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge()
        gauge.add_zone(0, 100, "#ffffff")
        assert len(gauge.zones) == 1

        gauge.clear_zones()
        assert len(gauge.zones) == 0

    def test_gauge_appearance_properties(self):
        """Test appearance property defaults."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge()
        assert gauge.background_color == "#1a1a1a"
        assert gauge.arc_color == "#3d3d3d"
        assert gauge.needle_color == "#ff4444"
        assert gauge.major_ticks == 5
        assert gauge.minor_ticks == 4

    def test_gauge_geometry_defaults(self):
        """Test gauge geometry defaults match size."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(size=150)
        assert gauge.geometry.width == 150
        assert gauge.geometry.height == 150


class TestGaugeSerialization:
    """Test Gauge serialization and deserialization."""

    def test_gauge_to_dict_minimal(self):
        """Test serializing Gauge with default values."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(name="test")
        data = gauge.to_dict()

        assert data["type"] == "Gauge"
        assert data["name"] == "test"
        assert data["value"] == 0
        assert data["min_value"] == 0
        assert data["max_value"] == 100
        assert data["size"] == 120

    def test_gauge_to_dict_full(self):
        """Test serializing Gauge with all properties."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge = Gauge(name="speedometer", value=80, min_value=0, max_value=200, size=180)
        gauge.start_angle = 180
        gauge.sweep_angle = -180
        gauge.needle_color = "#ff0000"
        gauge.value_suffix = "mph"
        gauge.label = "Speed"
        gauge.show_tick_labels = False
        gauge.add_zone(0, 60, "#00ff00")
        gauge.add_zone(60, 120, "#ffcc00")
        gauge.add_zone(120, 200, "#ff0000")

        data = gauge.to_dict()

        assert data["value"] == 80
        assert data["max_value"] == 200
        assert data["start_angle"] == 180
        assert data["sweep_angle"] == -180
        assert data["needle_color"] == "#ff0000"
        assert data["value_suffix"] == "mph"
        assert data["label"] == "Speed"
        assert data["show_tick_labels"] is False
        assert len(data["zones"]) == 3

    def test_gauge_from_dict_minimal(self):
        """Test deserializing Gauge from minimal data."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        data = {
            "type": "Gauge",
            "name": "temp_gauge"
        }

        gauge = Gauge.from_dict(data)

        assert gauge.name == "temp_gauge"
        assert gauge.value == 0
        assert gauge.min_value == 0
        assert gauge.max_value == 100

    def test_gauge_from_dict_full(self):
        """Test deserializing Gauge with all properties."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        data = {
            "type": "Gauge",
            "name": "rpm",
            "value": 3500,
            "min_value": 0,
            "max_value": 8000,
            "size": 200,
            "start_angle": 225,
            "sweep_angle": -270,
            "needle_color": "#ff6600",
            "arc_width": 12,
            "major_ticks": 8,
            "value_suffix": " RPM",
            "label": "Engine",
            "zones": [
                {"start_value": 0, "end_value": 3000, "color": "#00ff00"},
                {"start_value": 6000, "end_value": 8000, "color": "#ff0000"}
            ]
        }

        gauge = Gauge.from_dict(data)

        assert gauge.name == "rpm"
        assert gauge.value == 3500
        assert gauge.max_value == 8000
        assert gauge.size == 200
        assert gauge.needle_color == "#ff6600"
        assert gauge.arc_width == 12
        assert gauge.major_ticks == 8
        assert gauge.value_suffix == " RPM"
        assert gauge.label == "Engine"
        assert len(gauge.zones) == 2
        assert gauge.zones[0].color == "#00ff00"
        assert gauge.zones[1].end_value == 8000

    def test_gauge_round_trip(self):
        """Test serialization round-trip preserves data."""
        from noodlestudio.runtime.ui.components.gauge import Gauge

        original = Gauge(name="round_trip", value=50, max_value=200, size=180)
        original.needle_color = "#0000ff"
        original.label = "Test"
        original.add_zone(0, 100, "#00ff00")

        data = original.to_dict()
        restored = Gauge.from_dict(data)

        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.max_value == original.max_value
        assert restored.size == original.size
        assert restored.needle_color == original.needle_color
        assert restored.label == original.label
        assert len(restored.zones) == len(original.zones)


class TestGaugeYAML:
    """Test Gauge YAML integration."""

    def test_gauge_in_ui_yaml(self, tmp_path):
        """Test Gauge component in ui.yaml file."""
        from noodlestudio.runtime.ui.loader import UILoader

        ui_yaml = tmp_path / "ui.yaml"
        ui_yaml.write_text("""
version: 1
root:
  type: Panel
  name: root
  children:
    - type: Gauge
      name: speed_gauge
      properties:
        value: 65
        min_value: 0
        max_value: 120
        size: 150
        label: "Speed"
        value_suffix: " mph"
""")

        loader = UILoader()
        root = loader.load_file(str(ui_yaml))

        gauge = root.find_by_name("speed_gauge")
        assert gauge is not None
        assert gauge.component_type == "Gauge"
        assert gauge.value == 65
        assert gauge.max_value == 120
        assert gauge.label == "Speed"
        assert gauge.value_suffix == " mph"

    def test_gauge_with_zones_in_yaml(self, tmp_path):
        """Test Gauge with color zones in YAML."""
        from noodlestudio.runtime.ui.loader import UILoader

        ui_yaml = tmp_path / "ui.yaml"
        ui_yaml.write_text("""
version: 1
root:
  type: Panel
  name: root
  children:
    - type: Gauge
      name: temp_gauge
      properties:
        value: 75
        min_value: 0
        max_value: 100
        zones:
          - start_value: 0
            end_value: 30
            color: "#0066ff"
          - start_value: 30
            end_value: 70
            color: "#00ff00"
          - start_value: 70
            end_value: 100
            color: "#ff0000"
""")

        loader = UILoader()
        root = loader.load_file(str(ui_yaml))

        gauge = root.find_by_name("temp_gauge")
        assert gauge is not None
        assert len(gauge.zones) == 3
        assert gauge.zones[0].color == "#0066ff"
        assert gauge.zones[1].color == "#00ff00"
        assert gauge.zones[2].color == "#ff0000"


class TestGaugeRendering:
    """Test Gauge rendering (requires PyQt6)."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        import os
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_gauge_widget_creation(self, qapp):
        """Test GaugeWidget can be created."""
        from noodlestudio.runtime.ui.components.gauge import Gauge
        from noodlestudio.runtime.ui.renderer import GaugeWidget, QtWidgetRenderer

        gauge = Gauge(name="test", value=50, size=120)
        renderer = QtWidgetRenderer()

        widget = GaugeWidget(gauge, renderer)
        assert widget is not None
        assert widget.component == gauge

    def test_gauge_widget_minimum_size(self, qapp):
        """Test GaugeWidget sets minimum size correctly."""
        from noodlestudio.runtime.ui.components.gauge import Gauge
        from noodlestudio.runtime.ui.renderer import GaugeWidget, QtWidgetRenderer

        gauge = Gauge(size=150)
        renderer = QtWidgetRenderer()
        widget = GaugeWidget(gauge, renderer)

        assert widget.minimumWidth() == 150
        assert widget.minimumHeight() == 150

    def test_gauge_widget_set_value(self, qapp):
        """Test programmatic value change."""
        from noodlestudio.runtime.ui.components.gauge import Gauge
        from noodlestudio.runtime.ui.renderer import GaugeWidget, QtWidgetRenderer

        gauge = Gauge(value=0, max_value=100)
        renderer = QtWidgetRenderer()
        widget = GaugeWidget(gauge, renderer)

        assert gauge.value == 0
        widget.set_value(75)
        assert gauge.value == 75

    def test_gauge_widget_set_value_clamps(self, qapp):
        """Test set_value clamps to range."""
        from noodlestudio.runtime.ui.components.gauge import Gauge
        from noodlestudio.runtime.ui.renderer import GaugeWidget, QtWidgetRenderer

        gauge = Gauge(value=50, min_value=0, max_value=100)
        renderer = QtWidgetRenderer()
        widget = GaugeWidget(gauge, renderer)

        widget.set_value(150)
        assert gauge.value == 100

        widget.set_value(-50)
        assert gauge.value == 0

    def test_gauge_full_render_pipeline(self, qapp):
        """Test Gauge rendering through QtWidgetRenderer."""
        from noodlestudio.runtime.ui.components.gauge import Gauge
        from noodlestudio.runtime.ui.components.panel import Panel
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer

        panel = Panel(name="root")
        gauge = Gauge(name="meter", value=50, max_value=100, size=120)
        gauge.label = "Progress"
        gauge.add_zone(0, 50, "#00ff00")
        gauge.add_zone(50, 100, "#ff0000")
        panel.add_child(gauge)

        renderer = QtWidgetRenderer()
        widget = renderer.render(panel)

        # Check gauge widget was created
        gauge_widget = renderer.get_widget("meter")
        assert gauge_widget is not None

        # Check component reference
        gauge_comp = renderer.get_component("meter")
        assert gauge_comp is not None
        assert gauge_comp.value == 50
        assert len(gauge_comp.zones) == 2


class TestGaugeRegistry:
    """Test Gauge component registration."""

    def test_gauge_registered_in_component_system(self):
        """Test Gauge is registered via @register_component."""
        from noodlestudio.runtime.ui.component import list_component_types

        # Import Gauge to ensure it's registered
        from noodlestudio.runtime.ui.components.gauge import Gauge

        assert "Gauge" in list_component_types()

    def test_gauge_can_be_instantiated_from_registry(self):
        """Test creating Gauge from registry."""
        from noodlestudio.runtime.ui.component import get_component_class

        # Import Gauge to ensure it's registered
        from noodlestudio.runtime.ui.components.gauge import Gauge

        gauge_class = get_component_class("Gauge")
        assert gauge_class is not None

        gauge = gauge_class(name="from_registry", value=42)

        assert gauge.name == "from_registry"
        assert gauge.value == 42


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
