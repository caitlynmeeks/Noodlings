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
#   QML Widget Component
#
#   Wraps QML files as native NoodleStudio UI components.
#   Enables use of thousands of QML widgets from the Qt ecosystem.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.qml_widget
# PURPOSE:  QML Widget Component
# LAYER:    Studio / UI Components / QML
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   QMLWidget, QMLPropertyBinding
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
from pathlib import Path
import logging

from ..component import UIComponent, register_component

logger = logging.getLogger(__name__)


@dataclass
class QMLPropertyBinding:
    """
    Binding between a QML property and a NoodleStudio channel.

    Attributes:
        property_name: Name of the QML property
        channel: Channel path to bind to (e.g., "affect/arousal")
        direction: "input" (channel -> QML) or "output" (QML -> channel)
        value_type: Type hint for the property value
        default: Default value if channel not connected
    """
    property_name: str
    channel: str = ""
    direction: str = "input"  # "input" or "output"
    value_type: str = "any"
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "channel": self.channel,
            "direction": self.direction,
            "value_type": self.value_type,
            "default": self.default
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QMLPropertyBinding':
        return cls(
            property_name=data.get("property_name", ""),
            channel=data.get("channel", ""),
            direction=data.get("direction", "input"),
            value_type=data.get("value_type", "any"),
            default=data.get("default")
        )


@register_component
class QMLWidget(UIComponent):
    """
    UI component that wraps a QML file.

    Enables drag-and-drop use of QML widgets from the Qt ecosystem:
    - Gauges, meters, dials from automotive dashboards
    - Industrial control panel widgets
    - Aviation instrumentation
    - Custom visualizations

    The user never sees QML code - they configure properties in
    the Inspector and bind them to channels.

    Properties:
        qml_source: Path to the .qml file (relative to project or absolute)
        qml_properties: Dictionary of property values to set on the QML root
        property_bindings: List of QMLPropertyBinding for channel connections
        fallback_text: Text to display if QML fails to load

    Events:
        onLoad: Triggered when QML loads successfully
        onError: Triggered if QML fails to load

    Example YAML:
        - type: QMLWidget
          qml_source: "widgets/arc_gauge.qml"
          qml_properties:
            label: "Arousal"
            needleColor: "#e74c3c"
            minValue: 0.0
            maxValue: 1.0
          bindings:
            value: "affect/arousal"
    """

    component_type = "QMLWidget"

    def __init__(
        self,
        name: str = "",
        qml_source: str = "",
        size: int = 120
    ):
        super().__init__(name)

        # QML source path
        self.qml_source: str = qml_source

        # Properties to set on the QML root object
        self.qml_properties: Dict[str, Any] = {}

        # Channel bindings for QML properties
        self.property_bindings: List[QMLPropertyBinding] = []

        # Fallback display if QML fails to load
        self.fallback_text: str = "QML Widget"
        self.fallback_color: str = "#444444"

        # Error state
        self.error_message: str = ""
        self.has_error: bool = False

        # Default size
        self.geometry.width = size
        self.geometry.height = size

        # Runtime state (set by renderer)
        self._root_object: Any = None
        self._discovered_properties: Dict[str, Dict[str, Any]] = {}

    def set_qml_property(self, name: str, value: Any) -> None:
        """
        Set a QML property value.

        Args:
            name: Property name (must exist in QML)
            value: Value to set
        """
        self.qml_properties[name] = value

        # If we have a live root object, update it
        if self._root_object:
            try:
                self._root_object.setProperty(name, value)
            except Exception as e:
                logger.warning(f"Failed to set QML property {name}: {e}")

    def get_qml_property(self, name: str) -> Any:
        """
        Get a QML property value.

        Args:
            name: Property name

        Returns:
            Current value, or None if not available
        """
        if self._root_object:
            try:
                return self._root_object.property(name)
            except Exception:
                pass
        return self.qml_properties.get(name)

    def bind_property_to_channel(
        self,
        property_name: str,
        channel: str,
        direction: str = "input"
    ) -> None:
        """
        Bind a QML property to a NoodleStudio channel.

        Args:
            property_name: QML property name
            channel: Channel path (e.g., "affect/arousal")
            direction: "input" (channel -> QML) or "output" (QML -> channel)
        """
        # Remove existing binding for this property
        self.property_bindings = [
            b for b in self.property_bindings
            if b.property_name != property_name
        ]

        # Add new binding
        self.property_bindings.append(QMLPropertyBinding(
            property_name=property_name,
            channel=channel,
            direction=direction
        ))

    def unbind_property(self, property_name: str) -> None:
        """Remove channel binding for a property."""
        self.property_bindings = [
            b for b in self.property_bindings
            if b.property_name != property_name
        ]

    def get_binding_for_property(self, property_name: str) -> Optional[QMLPropertyBinding]:
        """Get the channel binding for a property, if any."""
        for binding in self.property_bindings:
            if binding.property_name == property_name:
                return binding
        return None

    def get_discovered_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Get properties discovered from the QML root object.

        Returns dict of property_name -> {type, default, bindable}
        Set by the renderer after QML loads.
        """
        return self._discovered_properties

    def set_error(self, message: str) -> None:
        """Mark the widget as having an error."""
        self.has_error = True
        self.error_message = message

    def clear_error(self) -> None:
        """Clear any error state."""
        self.has_error = False
        self.error_message = ""

    def resolve_qml_path(self, base_path: Optional[Path] = None) -> Optional[Path]:
        """
        Resolve the QML source path to an absolute path.

        Args:
            base_path: Base directory for relative paths (usually project dir)

        Returns:
            Resolved Path, or None if not found
        """
        if not self.qml_source:
            return None

        # Already absolute
        path = Path(self.qml_source)
        if path.is_absolute():
            return path if path.exists() else None

        # Try relative to base_path
        if base_path:
            resolved = base_path / self.qml_source
            if resolved.exists():
                return resolved

        # Try as-is
        if path.exists():
            return path

        return None

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add QMLWidget-specific properties to serialization."""
        if self.qml_source:
            data["qml_source"] = self.qml_source

        if self.qml_properties:
            data["qml_properties"] = self.qml_properties.copy()

        if self.property_bindings:
            data["property_bindings"] = [b.to_dict() for b in self.property_bindings]

        if self.fallback_text != "QML Widget":
            data["fallback_text"] = self.fallback_text
        if self.fallback_color != "#444444":
            data["fallback_color"] = self.fallback_color

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QMLWidget':
        """Deserialize from dictionary."""
        widget = cls(
            name=data.get("name", ""),
            qml_source=data.get("qml_source", ""),
            size=data.get("size", 120)
        )

        widget._apply_base_properties(data)

        # QML properties
        widget.qml_properties = data.get("qml_properties", {}).copy()

        # Property bindings - support both old "bindings" format and new "property_bindings"
        if "property_bindings" in data:
            widget.property_bindings = [
                QMLPropertyBinding.from_dict(b)
                for b in data["property_bindings"]
            ]
        elif "bindings" in data:
            # Legacy format: bindings: {property_name: channel}
            for prop_name, channel in data["bindings"].items():
                widget.property_bindings.append(QMLPropertyBinding(
                    property_name=prop_name,
                    channel=channel,
                    direction="input"
                ))

        widget.fallback_text = data.get("fallback_text", "QML Widget")
        widget.fallback_color = data.get("fallback_color", "#444444")

        # Geometry
        widget.geometry.width = data.get("width", widget.geometry.width)
        widget.geometry.height = data.get("height", widget.geometry.height)

        return widget


# Type mapping from QML types to Python/channel types
QML_TYPE_MAP = {
    "real": "float",
    "double": "float",
    "int": "int",
    "bool": "bool",
    "string": "str",
    "QString": "str",
    "color": "color",
    "QColor": "color",
    "url": "str",
    "QUrl": "str",
    "var": "any",
    "QVariant": "any",
    "point": "point",
    "QPointF": "point",
    "size": "size",
    "QSizeF": "size",
    "rect": "rect",
    "QRectF": "rect",
}


def qml_type_to_python(qml_type: str) -> str:
    """Convert QML type name to Python/channel type."""
    return QML_TYPE_MAP.get(qml_type, "any")


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
