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
#   QML Widget Tests
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
import tempfile
import os
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestQMLPropertyBinding:
    """Tests for QMLPropertyBinding dataclass."""

    def test_create_binding(self):
        """Test creating a property binding."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLPropertyBinding

        binding = QMLPropertyBinding(
            property_name="value",
            channel="affect/arousal",
            direction="input"
        )

        assert binding.property_name == "value"
        assert binding.channel == "affect/arousal"
        assert binding.direction == "input"
        assert binding.value_type == "any"
        assert binding.default is None

    def test_binding_to_dict(self):
        """Test serializing a binding to dict."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLPropertyBinding

        binding = QMLPropertyBinding(
            property_name="label",
            channel="ui/label_text",
            direction="input",
            value_type="str",
            default="Hello"
        )

        data = binding.to_dict()
        assert data["property_name"] == "label"
        assert data["channel"] == "ui/label_text"
        assert data["direction"] == "input"
        assert data["value_type"] == "str"
        assert data["default"] == "Hello"

    def test_binding_from_dict(self):
        """Test deserializing a binding from dict."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLPropertyBinding

        data = {
            "property_name": "value",
            "channel": "data/value",
            "direction": "output",
            "value_type": "float",
            "default": 0.5
        }

        binding = QMLPropertyBinding.from_dict(data)
        assert binding.property_name == "value"
        assert binding.channel == "data/value"
        assert binding.direction == "output"
        assert binding.value_type == "float"
        assert binding.default == 0.5

    def test_binding_roundtrip(self):
        """Test serialization roundtrip."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLPropertyBinding

        original = QMLPropertyBinding(
            property_name="color",
            channel="theme/accent",
            direction="input",
            value_type="color",
            default="#ff0000"
        )

        data = original.to_dict()
        restored = QMLPropertyBinding.from_dict(data)

        assert restored.property_name == original.property_name
        assert restored.channel == original.channel
        assert restored.direction == original.direction
        assert restored.value_type == original.value_type
        assert restored.default == original.default


class TestQMLWidget:
    """Tests for QMLWidget component."""

    def test_create_widget(self):
        """Test creating a QML widget."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="my_gauge")
        assert widget.name == "my_gauge"
        assert widget.component_type == "QMLWidget"
        assert widget.qml_source == ""
        assert widget.qml_properties == {}
        assert widget.property_bindings == []

    def test_create_with_source(self):
        """Test creating a widget with QML source."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge", qml_source="widgets/gauge.qml")
        assert widget.qml_source == "widgets/gauge.qml"

    def test_create_with_size(self):
        """Test creating a widget with custom size."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="big_gauge", size=200)
        assert widget.geometry.width == 200
        assert widget.geometry.height == 200

    def test_set_qml_property(self):
        """Test setting QML properties."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        widget.set_qml_property("value", 0.75)
        widget.set_qml_property("label", "Arousal")

        assert widget.qml_properties["value"] == 0.75
        assert widget.qml_properties["label"] == "Arousal"

    def test_get_qml_property(self):
        """Test getting QML properties."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        widget.qml_properties["value"] = 0.5

        # Without root object, returns from local dict
        assert widget.get_qml_property("value") == 0.5
        assert widget.get_qml_property("nonexistent") is None

    def test_bind_property_to_channel(self):
        """Test binding a property to a channel."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget, QMLPropertyBinding

        widget = QMLWidget(name="gauge")
        widget.bind_property_to_channel("value", "affect/arousal", "input")

        assert len(widget.property_bindings) == 1
        binding = widget.property_bindings[0]
        assert binding.property_name == "value"
        assert binding.channel == "affect/arousal"
        assert binding.direction == "input"

    def test_bind_property_replaces_existing(self):
        """Test that binding a property replaces existing binding."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        widget.bind_property_to_channel("value", "channel1")
        widget.bind_property_to_channel("value", "channel2")

        assert len(widget.property_bindings) == 1
        assert widget.property_bindings[0].channel == "channel2"

    def test_unbind_property(self):
        """Test unbinding a property."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        widget.bind_property_to_channel("value", "channel1")
        widget.bind_property_to_channel("label", "channel2")

        widget.unbind_property("value")

        assert len(widget.property_bindings) == 1
        assert widget.property_bindings[0].property_name == "label"

    def test_get_binding_for_property(self):
        """Test getting a binding by property name."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        widget.bind_property_to_channel("value", "affect/arousal")

        binding = widget.get_binding_for_property("value")
        assert binding is not None
        assert binding.channel == "affect/arousal"

        no_binding = widget.get_binding_for_property("nonexistent")
        assert no_binding is None

    def test_error_state(self):
        """Test error state management."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="gauge")
        assert not widget.has_error
        assert widget.error_message == ""

        widget.set_error("QML file not found")
        assert widget.has_error
        assert widget.error_message == "QML file not found"

        widget.clear_error()
        assert not widget.has_error
        assert widget.error_message == ""

    def test_resolve_qml_path_absolute(self):
        """Test resolving absolute QML path."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        with tempfile.TemporaryDirectory() as tmpdir:
            qml_file = Path(tmpdir) / "test.qml"
            qml_file.write_text("Item {}")

            widget = QMLWidget(qml_source=str(qml_file))
            resolved = widget.resolve_qml_path()

            assert resolved == qml_file

    def test_resolve_qml_path_relative(self):
        """Test resolving relative QML path."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            qml_file = base / "widgets" / "test.qml"
            qml_file.parent.mkdir(parents=True)
            qml_file.write_text("Item {}")

            widget = QMLWidget(qml_source="widgets/test.qml")
            resolved = widget.resolve_qml_path(base)

            assert resolved == qml_file

    def test_resolve_qml_path_not_found(self):
        """Test resolving nonexistent QML path."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(qml_source="nonexistent.qml")
        resolved = widget.resolve_qml_path()

        assert resolved is None

    def test_fallback_text_default(self):
        """Test default fallback text."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget()
        assert widget.fallback_text == "QML Widget"
        assert widget.fallback_color == "#444444"


class TestQMLWidgetSerialization:
    """Tests for QMLWidget serialization."""

    def test_to_dict_minimal(self):
        """Test serializing minimal widget."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="my_widget")
        data = widget.to_dict()

        assert data["type"] == "QMLWidget"
        assert data["name"] == "my_widget"

    def test_to_dict_full(self):
        """Test serializing widget with all properties."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        widget = QMLWidget(name="full_widget", qml_source="gauge.qml", size=150)
        widget.qml_properties = {"value": 0.5, "label": "Test"}
        widget.bind_property_to_channel("value", "data/value")
        widget.fallback_text = "Loading..."
        widget.fallback_color = "#333333"

        data = widget.to_dict()

        assert data["qml_source"] == "gauge.qml"
        assert data["qml_properties"] == {"value": 0.5, "label": "Test"}
        assert len(data["property_bindings"]) == 1
        assert data["fallback_text"] == "Loading..."
        assert data["fallback_color"] == "#333333"

    def test_from_dict_minimal(self):
        """Test deserializing minimal widget."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        data = {"name": "test_widget", "type": "QMLWidget"}
        widget = QMLWidget.from_dict(data)

        assert widget.name == "test_widget"
        assert widget.qml_source == ""

    def test_from_dict_full(self):
        """Test deserializing widget with all properties."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        data = {
            "name": "full_widget",
            "type": "QMLWidget",
            "qml_source": "widgets/meter.qml",
            "qml_properties": {"value": 0.8, "color": "#ff0000"},
            "property_bindings": [
                {"property_name": "value", "channel": "sensor/level", "direction": "input"}
            ],
            "fallback_text": "Meter",
            "fallback_color": "#222222",
            "width": 180,
            "height": 180
        }

        widget = QMLWidget.from_dict(data)

        assert widget.qml_source == "widgets/meter.qml"
        assert widget.qml_properties["value"] == 0.8
        assert widget.qml_properties["color"] == "#ff0000"
        assert len(widget.property_bindings) == 1
        assert widget.property_bindings[0].channel == "sensor/level"
        assert widget.fallback_text == "Meter"
        assert widget.fallback_color == "#222222"
        assert widget.geometry.width == 180
        assert widget.geometry.height == 180

    def test_from_dict_legacy_bindings(self):
        """Test deserializing with legacy bindings format."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        # Legacy format: bindings: {property_name: channel}
        data = {
            "name": "legacy_widget",
            "type": "QMLWidget",
            "qml_source": "gauge.qml",
            "bindings": {
                "value": "affect/arousal",
                "label": "ui/label_text"
            }
        }

        widget = QMLWidget.from_dict(data)

        assert len(widget.property_bindings) == 2
        binding_names = {b.property_name for b in widget.property_bindings}
        assert "value" in binding_names
        assert "label" in binding_names

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        from noodlestudio.runtime.ui.components.qml_widget import QMLWidget

        original = QMLWidget(name="roundtrip", qml_source="test.qml", size=160)
        original.qml_properties = {"value": 0.3, "label": "Test"}
        original.bind_property_to_channel("value", "channel/data", "input")

        data = original.to_dict()
        restored = QMLWidget.from_dict(data)

        assert restored.name == original.name
        assert restored.qml_source == original.qml_source
        assert restored.qml_properties == original.qml_properties
        assert len(restored.property_bindings) == len(original.property_bindings)


class TestQMLWidgetYAML:
    """Tests for QMLWidget YAML loading."""

    def test_load_from_yaml(self):
        """Test loading QML widget from YAML."""
        from noodlestudio.runtime.ui.loader import UILoader

        yaml_content = """
version: 1
root:
  type: Panel
  name: root
  children:
    - type: QMLWidget
      name: my_gauge
      qml_source: "widgets/arc_gauge.qml"
      x: 10
      y: 10
      width: 150
      height: 150
      qml_properties:
        label: "Arousal"
        needleColor: "#ff0000"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_yaml = Path(tmpdir) / "ui.yaml"
            ui_yaml.write_text(yaml_content)

            loader = UILoader()
            root = loader.load_file(str(ui_yaml))

            assert root is not None
            gauge = root.find_by_name("my_gauge")
            assert gauge is not None
            assert gauge.component_type == "QMLWidget"
            assert gauge.qml_source == "widgets/arc_gauge.qml"
            assert gauge.qml_properties["label"] == "Arousal"
            assert gauge.qml_properties["needleColor"] == "#ff0000"

    def test_load_with_bindings(self):
        """Test loading QML widget with property bindings."""
        from noodlestudio.runtime.ui.loader import UILoader

        yaml_content = """
version: 1
root:
  type: QMLWidget
  name: bound_gauge
  qml_source: "gauge.qml"
  property_bindings:
    - property_name: value
      channel: "affect/arousal"
      direction: input
    - property_name: label
      channel: "ui/label"
      direction: input
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_yaml = Path(tmpdir) / "ui.yaml"
            ui_yaml.write_text(yaml_content)

            loader = UILoader()
            widget = loader.load_file(str(ui_yaml))

            assert len(widget.property_bindings) == 2
            value_binding = widget.get_binding_for_property("value")
            assert value_binding.channel == "affect/arousal"


class TestQMLEngineManager:
    """Tests for QMLEngineManager singleton."""

    def test_is_available(self):
        """Test checking QML availability."""
        from noodlestudio.runtime.ui.qml_engine_manager import QMLEngineManager

        # This will return True or False depending on PyQt6.QtQml availability
        # This method doesn't require QApplication
        available = QMLEngineManager.is_available()
        assert isinstance(available, bool)

    def test_singleton_without_init(self):
        """Test that the singleton pattern is set up correctly."""
        from noodlestudio.runtime.ui.qml_engine_manager import QMLEngineManager

        # Ensure _instance is None initially (after reset)
        QMLEngineManager._instance = None
        assert QMLEngineManager._instance is None

        # Note: We can't actually test instance() without QApplication
        # The singleton logic is tested by ensuring is_available works

    def test_reset_clears_instance(self):
        """Test that reset clears the _instance."""
        from noodlestudio.runtime.ui.qml_engine_manager import QMLEngineManager

        # Set to something non-None
        QMLEngineManager._instance = "dummy"
        QMLEngineManager.reset()

        assert QMLEngineManager._instance is None


class TestQMLTypeMapping:
    """Tests for QML type mapping."""

    def test_type_mapping(self):
        """Test QML type to Python type mapping."""
        from noodlestudio.runtime.ui.components.qml_widget import qml_type_to_python

        assert qml_type_to_python("real") == "float"
        assert qml_type_to_python("double") == "float"
        assert qml_type_to_python("int") == "int"
        assert qml_type_to_python("bool") == "bool"
        assert qml_type_to_python("string") == "str"
        assert qml_type_to_python("QString") == "str"
        assert qml_type_to_python("color") == "color"
        assert qml_type_to_python("QColor") == "color"
        assert qml_type_to_python("url") == "str"
        assert qml_type_to_python("var") == "any"
        assert qml_type_to_python("QVariant") == "any"

    def test_unknown_type(self):
        """Test mapping unknown QML type."""
        from noodlestudio.runtime.ui.components.qml_widget import qml_type_to_python

        assert qml_type_to_python("CustomType") == "any"
        assert qml_type_to_python("") == "any"


class TestQMLWidgetRegistry:
    """Tests for QMLWidget registration."""

    def test_registered_in_registry(self):
        """Test that QMLWidget is registered."""
        from noodlestudio.runtime.ui.component import get_component_class, list_component_types

        assert "QMLWidget" in list_component_types()
        cls = get_component_class("QMLWidget")
        assert cls is not None
        assert cls.component_type == "QMLWidget"

    def test_exported_from_components(self):
        """Test that QMLWidget is exported from components module."""
        from noodlestudio.runtime.ui.components import QMLWidget, QMLPropertyBinding

        widget = QMLWidget(name="test")
        assert widget.component_type == "QMLWidget"

        binding = QMLPropertyBinding(property_name="value", channel="test")
        assert binding.property_name == "value"


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
