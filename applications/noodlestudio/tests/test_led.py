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
#   Tests for the LED Dashboard Widget
#
#   Tests the LED indicator component for dashboard displays.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_led
# PURPOSE:  Tests for the LED Dashboard Widget
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestLED, TestLEDSerialization, TestLEDRendering
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest


class TestLED:
    """Test LED component class."""

    def test_led_creation_defaults(self):
        """Test LED creation with default values."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(name="test_led")
        assert led.name == "test_led"
        assert led.component_type == "LED"
        assert led.on is False
        assert led.color == "#00ff66"
        assert led.size == 16
        assert led.glow == 0.6

    def test_led_creation_custom(self):
        """Test LED creation with custom values."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(name="red_led", on=True, color="#ff3344", size=24)
        assert led.on is True
        assert led.color == "#ff3344"
        assert led.size == 24

    def test_led_value_alias(self):
        """Test that value property is an alias for on."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED()
        led.on = True
        assert led.value is True

        led.value = False
        assert led.on is False

    def test_led_toggle(self):
        """Test LED toggle method."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(on=False)
        result = led.toggle()
        assert result is True
        assert led.on is True

        result = led.toggle()
        assert result is False
        assert led.on is False

    def test_led_turn_on_off(self):
        """Test LED turn_on and turn_off methods."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(on=False)

        led.turn_on()
        assert led.on is True

        led.turn_off()
        assert led.on is False

    def test_led_shape_enum(self):
        """Test LEDShape enum values."""
        from noodlestudio.runtime.ui.components.led import LED, LEDShape

        led = LED()
        assert led.shape == LEDShape.ROUND

        led.shape = LEDShape.SQUARE
        assert led.shape == LEDShape.SQUARE
        assert led.shape.value == "square"

    def test_led_color_constants(self):
        """Test LED color constant definitions."""
        from noodlestudio.runtime.ui.components.led import LED

        assert LED.COLOR_GREEN == "#00ff66"
        assert LED.COLOR_RED == "#ff3344"
        assert LED.COLOR_YELLOW == "#ffcc00"
        assert LED.COLOR_BLUE == "#3399ff"
        assert LED.COLOR_ORANGE == "#ff8800"
        assert LED.COLOR_WHITE == "#ffffff"

    def test_led_effective_off_color_auto(self):
        """Test automatic off color calculation."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(color="#00ff66")  # Bright green
        off_color = led.get_effective_off_color()

        # Should be dimmed to ~20%
        assert off_color.startswith("#")
        assert len(off_color) == 7

        # The off color should be much darker than the on color
        # #00ff66 at 20% should be roughly #003314
        assert off_color != "#00ff66"

    def test_led_effective_off_color_explicit(self):
        """Test explicit off color setting."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED()
        led.off_color = "#111111"
        assert led.get_effective_off_color() == "#111111"

    def test_led_current_color(self):
        """Test get_current_color based on state."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(color="#ff0000", on=False)
        assert led.get_current_color() != "#ff0000"  # Should be off color

        led.on = True
        assert led.get_current_color() == "#ff0000"  # Should be on color

    def test_led_label_properties(self):
        """Test LED label configuration."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(name="status")
        led.label = "Status"
        led.label_position = "right"
        led.label_color = "#ffffff"
        led.label_spacing = 12
        led.font_size = 14

        assert led.label == "Status"
        assert led.label_position == "right"
        assert led.label_color == "#ffffff"
        assert led.label_spacing == 12
        assert led.font_size == 14

    def test_led_blink_rate(self):
        """Test LED blink rate configuration."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED()
        assert led.blink_rate == 0.0  # No blink by default

        led.blink_rate = 0.5  # Blink every 500ms
        assert led.blink_rate == 0.5

    def test_led_geometry_defaults(self):
        """Test LED geometry defaults match size."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(size=20)
        assert led.geometry.width == 20
        assert led.geometry.height == 20


class TestLEDSerialization:
    """Test LED serialization and deserialization."""

    def test_led_to_dict_minimal(self):
        """Test serializing LED with default values."""
        from noodlestudio.runtime.ui.components.led import LED

        led = LED(name="test")
        data = led.to_dict()

        assert data["type"] == "LED"
        assert data["name"] == "test"
        assert data["on"] is False
        assert data["color"] == "#00ff66"
        assert data["size"] == 16

    def test_led_to_dict_full(self):
        """Test serializing LED with all properties."""
        from noodlestudio.runtime.ui.components.led import LED, LEDShape

        led = LED(name="warning_led", on=True, color="#ff0000", size=24)
        led.off_color = "#330000"
        led.shape = LEDShape.SQUARE
        led.glow = 0.8
        led.blink_rate = 1.0
        led.label = "Warning"
        led.label_position = "bottom"
        led.label_color = "#ffaaaa"

        data = led.to_dict()

        assert data["on"] is True
        assert data["color"] == "#ff0000"
        assert data["size"] == 24
        assert data["off_color"] == "#330000"
        assert data["shape"] == "square"
        assert data["glow"] == 0.8
        assert data["blink_rate"] == 1.0
        assert data["label"] == "Warning"
        assert data["label_position"] == "bottom"
        assert data["label_color"] == "#ffaaaa"

    def test_led_from_dict_minimal(self):
        """Test deserializing LED from minimal data."""
        from noodlestudio.runtime.ui.components.led import LED

        data = {
            "type": "LED",
            "name": "status_led"
        }

        led = LED.from_dict(data)

        assert led.name == "status_led"
        assert led.on is False
        assert led.color == "#00ff66"
        assert led.size == 16

    def test_led_from_dict_full(self):
        """Test deserializing LED with all properties."""
        from noodlestudio.runtime.ui.components.led import LED, LEDShape

        data = {
            "type": "LED",
            "name": "alert",
            "on": True,
            "color": "#ffcc00",
            "size": 20,
            "off_color": "#332200",
            "shape": "square",
            "glow": 0.9,
            "blink_rate": 0.25,
            "label": "Alert",
            "label_position": "left",
            "label_color": "#ffee00",
            "label_spacing": 10,
            "font_size": 16
        }

        led = LED.from_dict(data)

        assert led.name == "alert"
        assert led.on is True
        assert led.color == "#ffcc00"
        assert led.size == 20
        assert led.off_color == "#332200"
        assert led.shape == LEDShape.SQUARE
        assert led.glow == 0.9
        assert led.blink_rate == 0.25
        assert led.label == "Alert"
        assert led.label_position == "left"
        assert led.label_color == "#ffee00"
        assert led.label_spacing == 10
        assert led.font_size == 16

    def test_led_round_trip(self):
        """Test serialization round-trip preserves data."""
        from noodlestudio.runtime.ui.components.led import LED, LEDShape

        original = LED(name="round_trip", on=True, color="#3399ff", size=32)
        original.shape = LEDShape.SQUARE
        original.label = "Test"
        original.blink_rate = 2.0

        data = original.to_dict()
        restored = LED.from_dict(data)

        assert restored.name == original.name
        assert restored.on == original.on
        assert restored.color == original.color
        assert restored.size == original.size
        assert restored.shape == original.shape
        assert restored.label == original.label
        assert restored.blink_rate == original.blink_rate


class TestLEDYAML:
    """Test LED YAML integration."""

    def test_led_in_ui_yaml(self, tmp_path):
        """Test LED component in ui.yaml file."""
        import yaml
        from noodlestudio.runtime.ui.loader import UILoader

        ui_yaml = tmp_path / "ui.yaml"
        ui_yaml.write_text("""
version: 1
root:
  type: Panel
  name: root
  children:
    - type: LED
      name: status_led
      properties:
        "on": true
        color: "#00ff66"
        size: 20
        label: "Online"
        label_position: right
""")

        loader = UILoader()
        root = loader.load_file(str(ui_yaml))

        # Find the LED component
        led = root.find_by_name("status_led")
        assert led is not None
        assert led.component_type == "LED"
        assert led.on is True
        assert led.color == "#00ff66"
        assert led.size == 20
        assert led.label == "Online"

    def test_multiple_leds_in_panel(self, tmp_path):
        """Test multiple LED components in a panel."""
        import yaml
        from noodlestudio.runtime.ui.loader import UILoader

        ui_yaml = tmp_path / "ui.yaml"
        ui_yaml.write_text("""
version: 1
root:
  type: Panel
  name: indicator_panel
  children:
    - type: LED
      name: power_led
      properties:
        x: 10
        y: 10
        "on": true
        color: "#00ff66"
        label: "Power"
    - type: LED
      name: error_led
      properties:
        x: 10
        y: 40
        "on": false
        color: "#ff3344"
        label: "Error"
    - type: LED
      name: warning_led
      properties:
        x: 10
        y: 70
        "on": true
        color: "#ffcc00"
        blink_rate: 0.5
        label: "Warning"
""")

        loader = UILoader()
        root = loader.load_file(str(ui_yaml))

        power = root.find_by_name("power_led")
        error = root.find_by_name("error_led")
        warning = root.find_by_name("warning_led")

        assert power is not None and power.on is True
        assert error is not None and error.on is False
        assert warning is not None and warning.blink_rate == 0.5


class TestLEDRendering:
    """Test LED rendering (requires PyQt6)."""

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

    def test_led_widget_creation(self, qapp):
        """Test LEDWidget can be created."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(name="test", on=True, color="#ff0000", size=20)
        renderer = QtWidgetRenderer()

        widget = LEDWidget(led, renderer)
        assert widget is not None
        assert widget.component == led

    def test_led_widget_minimum_size(self, qapp):
        """Test LEDWidget sets minimum size correctly."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(size=24)
        renderer = QtWidgetRenderer()
        widget = LEDWidget(led, renderer)

        assert widget.minimumWidth() == 24
        assert widget.minimumHeight() == 24

    def test_led_widget_with_label_size(self, qapp):
        """Test LEDWidget accounts for label in size."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(size=16)
        led.label = "Test Label"
        led.label_position = "right"

        renderer = QtWidgetRenderer()
        widget = LEDWidget(led, renderer)

        # Widget should be wider than just the LED size
        assert widget.minimumWidth() > 16

    def test_led_widget_set_on(self, qapp):
        """Test programmatic state change."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(on=False)
        renderer = QtWidgetRenderer()
        widget = LEDWidget(led, renderer)

        assert led.on is False
        widget.set_on(True)
        assert led.on is True

    def test_led_full_render_pipeline(self, qapp):
        """Test LED rendering through QtWidgetRenderer."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.components.panel import Panel
        from noodlestudio.runtime.ui.renderer import QtWidgetRenderer

        panel = Panel(name="root")
        led = LED(name="status", on=True, color="#00ff66", size=16)
        led.label = "Status"
        panel.add_child(led)

        renderer = QtWidgetRenderer()
        widget = renderer.render(panel)

        # Check LED widget was created
        led_widget = renderer.get_widget("status")
        assert led_widget is not None

        # Check component reference
        led_comp = renderer.get_component("status")
        assert led_comp is not None
        assert led_comp.on is True

    def test_led_blink_timer_setup(self, qapp):
        """Test that blink timer is set up correctly."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(on=True)
        led.blink_rate = 0.5  # 500ms blink
        renderer = QtWidgetRenderer()
        widget = LEDWidget(led, renderer)

        assert widget._blink_timer is not None
        assert widget._blink_timer.isActive()

    def test_led_no_blink_timer_when_zero(self, qapp):
        """Test that no blink timer is created when blink_rate is 0."""
        from noodlestudio.runtime.ui.components.led import LED
        from noodlestudio.runtime.ui.renderer import LEDWidget, QtWidgetRenderer

        led = LED(on=True)
        led.blink_rate = 0.0
        renderer = QtWidgetRenderer()
        widget = LEDWidget(led, renderer)

        assert widget._blink_timer is None


class TestLEDRegistry:
    """Test LED component registration."""

    def test_led_registered_in_component_system(self):
        """Test LED is registered via @register_component."""
        from noodlestudio.runtime.ui.component import list_component_types

        # Import LED to ensure it's registered
        from noodlestudio.runtime.ui.components.led import LED

        assert "LED" in list_component_types()

    def test_led_can_be_instantiated_from_registry(self):
        """Test creating LED from registry."""
        from noodlestudio.runtime.ui.component import get_component_class

        # Import LED to ensure it's registered
        from noodlestudio.runtime.ui.components.led import LED

        led_class = get_component_class("LED")
        assert led_class is not None

        led = led_class(name="from_registry", on=True)

        assert led.name == "from_registry"
        assert led.on is True


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
