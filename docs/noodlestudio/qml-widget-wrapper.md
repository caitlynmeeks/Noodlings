# QML Widget Wrapper System

> Thousands of gorgeous dashboard widgets, zero QML knowledge required

## The Vision

The Qt/QML ecosystem contains thousands of production-quality dashboard widgets - gauges, meters, dials, indicators used in Mercedes cockpits, aircraft instrumentation, industrial control panels. NoodleStudio should let users drag these beautiful components onto their UI canvas and wire them to cognition channels, without ever touching QML code.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USER EXPERIENCE                                  │
│                                                                      │
│   Widget Palette          Canvas              Inspector              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│   │ ◎ ArcGauge  │     │             │     │ ArcGauge            │   │
│   │ ◉ Speedo    │ --> │   [GAUGE]   │     │                     │   │
│   │ ◐ TankLevel │     │             │     │ Value: ─○─ 0.75     │   │
│   │ ◑ LED Array │     │             │     │ Min:   0.0          │   │
│   └─────────────┘     └─────────────┘     │ Max:   1.0          │   │
│                                           │                     │   │
│   User sees NoodleStudio                  │ Channel Bindings:   │   │
│   widgets. QML is invisible.              │ ┌─────────────────┐ │   │
│                                           │ │ value ← affect/ │ │   │
│                                           │ │         arousal │ │   │
│                                           │ └─────────────────┘ │   │
│                                           └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     UNDER THE HOOD                                   │
│                                                                      │
│   QMLWidgetWrapper                                                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │   QQuickWidget ─────────────────────────┐                   │   │
│   │   │                                     │                   │   │
│   │   │  ┌─────────────────────────────┐    │                   │   │
│   │   │  │     ArcGauge.qml            │    │                   │   │
│   │   │  │                             │    │                   │   │
│   │   │  │  property real value: 0.75  │◄───┼── Channel binding │   │
│   │   │  │  property real min: 0.0     │    │                   │   │
│   │   │  │  property real max: 1.0     │    │                   │   │
│   │   │  │  property color needleColor │    │                   │   │
│   │   │  │                             │    │                   │   │
│   │   │  └─────────────────────────────┘    │                   │   │
│   │   │                                     │                   │   │
│   │   └─────────────────────────────────────┘                   │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Why This Matters

1. **Instant Library** - Thousands of widgets available immediately
2. **Production Quality** - Battle-tested in automotive, aviation, industrial
3. **Beautiful by Default** - Professional designers made these
4. **No QML Required** - Users just wire channels, we handle the rest
5. **Community Growth** - Asset store can host contributed QML widgets

## Architecture

### Core Classes

```
widgets/
├── qml_widget_wrapper.py      # Base wrapper class
├── qml_property_bridge.py     # QML ↔ Channel binding
├── qml_widget_importer.py     # .qml → widget definition
└── qml_widget_gallery.py      # Browse/preview imported widgets

core/
└── qml_engine_manager.py      # Shared QML engine (performance)
```

### QMLWidgetWrapper

The core class that makes QML widgets behave like native NoodleStudio widgets.

```python
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtCore import QUrl, pyqtProperty
from noodlestudio.widgets.widget_base import WidgetBase


class QMLWidgetWrapper(WidgetBase):
    """
    Wraps a QML component as a NoodleStudio widget.

    The user never sees QML - they see a normal widget with
    properties and channel bindings in the inspector.
    """

    def __init__(self, qml_source: str, parent=None):
        super().__init__(parent)

        # The actual QML renderer
        self._quick_widget = QQuickWidget()
        self._quick_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)

        # Load the QML
        self._quick_widget.setSource(QUrl.fromLocalFile(qml_source))

        # Get the root QML object for property access
        self._root = self._quick_widget.rootObject()

        # Extract properties and create channel bindings
        self._property_bindings: Dict[str, ChannelBinding] = {}
        self._setup_property_bindings()

        # Embed in our widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._quick_widget)

    def _setup_property_bindings(self):
        """
        Discover QML properties and create channel binding points.

        QML properties become bindable channels:
        - 'value' property → input channel
        - 'clicked' signal → output channel
        """
        if not self._root:
            return

        meta = self._root.metaObject()
        for i in range(meta.propertyCount()):
            prop = meta.property(i)
            name = prop.name()

            # Skip internal Qt properties
            if name.startswith('_') or name in ('objectName', 'parent'):
                continue

            # Create a channel binding point for this property
            prop_type = self._qml_type_to_channel_type(prop.typeName())
            self._property_bindings[name] = ChannelBinding(
                name=name,
                direction='input',  # Most properties are inputs
                value_type=prop_type,
                default=self._root.property(name)
            )

    def set_property(self, name: str, value: Any):
        """Set a QML property value (called when channel updates)."""
        if self._root and name in self._property_bindings:
            self._root.setProperty(name, value)

    def get_property(self, name: str) -> Any:
        """Get a QML property value."""
        if self._root:
            return self._root.property(name)
        return None

    def on_channel_update(self, channel: str, value: Any):
        """Called when a bound channel updates."""
        # Find which property this channel is bound to
        for prop_name, binding in self._property_bindings.items():
            if binding.channel == channel:
                self.set_property(prop_name, value)
                break

    # ─────────────────────────────────────────────────────────────
    # Widget Interface (what NoodleStudio sees)
    # ─────────────────────────────────────────────────────────────

    def get_bindable_properties(self) -> List[PropertySpec]:
        """Return properties that can be bound to channels."""
        return [
            PropertySpec(
                name=name,
                type=binding.value_type,
                default=binding.default,
                bindable=True
            )
            for name, binding in self._property_bindings.items()
        ]

    def serialize(self) -> Dict:
        """Serialize for saving."""
        return {
            'type': 'qml_widget',
            'qml_source': self._qml_source,
            'properties': {
                name: self.get_property(name)
                for name in self._property_bindings
            },
            'bindings': {
                name: binding.channel
                for name, binding in self._property_bindings.items()
                if binding.channel
            }
        }
```

### QML Property Bridge

Handles the bidirectional binding between QML properties and NoodleStudio channels.

```python
class QMLPropertyBridge(QObject):
    """
    Bridges QML properties to NoodleStudio channels.

    - When channel updates → QML property updates
    - When QML signal fires → Channel emits
    """

    value_changed = pyqtSignal(str, object)  # (property_name, new_value)

    def __init__(self, root_object: QObject, channel_manager: ChannelManager):
        super().__init__()
        self._root = root_object
        self._channels = channel_manager
        self._bindings: Dict[str, str] = {}  # property_name → channel_path

    def bind_property_to_channel(self, property_name: str, channel_path: str):
        """Bind a QML property to receive updates from a channel."""
        self._bindings[property_name] = channel_path

        # Subscribe to channel updates
        self._channels.subscribe(channel_path,
            lambda value: self._on_channel_update(property_name, value))

        # Get initial value
        current = self._channels.get(channel_path)
        if current is not None:
            self._root.setProperty(property_name, current)

    def bind_signal_to_channel(self, signal_name: str, channel_path: str):
        """Bind a QML signal to emit on a channel."""
        signal = getattr(self._root, signal_name, None)
        if signal:
            signal.connect(lambda *args: self._channels.emit(channel_path, args))

    def _on_channel_update(self, property_name: str, value: Any):
        """Called when a bound channel updates."""
        self._root.setProperty(property_name, value)
```

### QML Widget Importer

Parses .qml files and generates NoodleStudio widget definitions.

```python
class QMLWidgetImporter:
    """
    Import QML components as NoodleStudio widgets.

    Workflow:
    1. User drops .qml file into Asset Browser
    2. Importer parses QML, extracts properties
    3. Generates widget_definition.yaml
    4. Widget appears in palette with preview
    """

    def import_qml(self, qml_path: Path) -> WidgetDefinition:
        """
        Import a QML file as a widget definition.

        Returns a WidgetDefinition that can be saved and loaded.
        """
        # Parse QML to extract metadata
        metadata = self._parse_qml(qml_path)

        # Generate widget definition
        definition = WidgetDefinition(
            name=metadata.get('name', qml_path.stem),
            category='QML Widgets',
            description=metadata.get('description', f'Imported from {qml_path.name}'),
            qml_source=str(qml_path),
            properties=self._extract_properties(metadata),
            preview=self._generate_preview(qml_path),
            license=metadata.get('license', 'Unknown'),
            attribution=metadata.get('attribution', ''),
        )

        return definition

    def _parse_qml(self, qml_path: Path) -> Dict:
        """
        Parse QML file for metadata and properties.

        QML is fairly regular - we can extract:
        - Root component type
        - Property declarations
        - Signal declarations
        - Comments for metadata
        """
        content = qml_path.read_text()
        metadata = {}

        # Extract header comments for metadata
        # // @name: Arc Gauge
        # // @description: A beautiful arc gauge
        # // @license: MIT
        for line in content.split('\n'):
            if line.strip().startswith('// @'):
                key, _, value = line.strip()[4:].partition(':')
                metadata[key.strip()] = value.strip()

        # Extract property declarations
        # property real value: 0.5
        # property color needleColor: "#ff0000"
        import re
        property_pattern = r'property\s+(\w+)\s+(\w+)(?:\s*:\s*(.+?))?(?:\n|$)'
        properties = []
        for match in re.finditer(property_pattern, content):
            prop_type, prop_name, default = match.groups()
            properties.append({
                'type': prop_type,
                'name': prop_name,
                'default': default.strip() if default else None
            })
        metadata['properties'] = properties

        # Extract signals
        # signal clicked()
        # signal valueChanged(real newValue)
        signal_pattern = r'signal\s+(\w+)\s*\(([^)]*)\)'
        signals = []
        for match in re.finditer(signal_pattern, content):
            sig_name, sig_params = match.groups()
            signals.append({
                'name': sig_name,
                'params': sig_params
            })
        metadata['signals'] = signals

        return metadata

    def _extract_properties(self, metadata: Dict) -> List[PropertySpec]:
        """Convert parsed QML properties to PropertySpecs."""
        type_map = {
            'real': 'float',
            'int': 'int',
            'bool': 'bool',
            'string': 'str',
            'color': 'color',
            'url': 'str',
            'var': 'any',
        }

        return [
            PropertySpec(
                name=prop['name'],
                type=type_map.get(prop['type'], 'any'),
                default=prop['default'],
                bindable=True
            )
            for prop in metadata.get('properties', [])
        ]

    def _generate_preview(self, qml_path: Path) -> Optional[bytes]:
        """
        Render a preview image of the QML widget.

        Creates a small QQuickWidget, renders the QML,
        captures to image.
        """
        # Create offscreen renderer
        widget = QQuickWidget()
        widget.setSource(QUrl.fromLocalFile(str(qml_path)))
        widget.resize(128, 128)

        # Wait for render
        widget.show()
        QApplication.processEvents()

        # Capture
        pixmap = widget.grab()

        # Convert to bytes
        buffer = QBuffer()
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, 'PNG')

        widget.close()
        return buffer.data().data()
```

### Widget Definition YAML

When a QML widget is imported, we generate a definition file:

```yaml
# widgets/imported/arc_gauge.widget.yaml
name: Arc Gauge
category: QML Widgets / Gauges
description: A smooth arc gauge with customizable colors and range
qml_source: arc_gauge.qml
license: MIT
attribution: "Original by QtQuick Controls 2"

preview: arc_gauge_preview.png

properties:
  - name: value
    type: float
    default: 0.5
    min: 0.0
    max: 1.0
    bindable: true
    description: Current value (0-1 normalized)

  - name: minimumValue
    type: float
    default: 0.0
    bindable: true

  - name: maximumValue
    type: float
    default: 100.0
    bindable: true

  - name: needleColor
    type: color
    default: "#e74c3c"
    bindable: true

  - name: backgroundColor
    type: color
    default: "#2c3e50"
    bindable: false  # Static, not animated

  - name: label
    type: string
    default: ""
    bindable: true

signals:
  - name: clicked
    description: Emitted when gauge is clicked
    output_channel: true
```

## QML Engine Manager

For performance, we share a single QML engine across all QML widgets.

```python
class QMLEngineManager:
    """
    Manages a shared QML engine for all QML widgets.

    Benefits:
    - Shared component cache
    - Single JavaScript engine
    - Reduced memory footprint
    """

    _instance = None

    @classmethod
    def instance(cls) -> 'QMLEngineManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._engine = QQmlEngine()

        # Add import paths for common QML modules
        self._engine.addImportPath(":/qml")
        self._engine.addImportPath(str(Path(__file__).parent / "qml_modules"))

        # Register custom types
        self._register_custom_types()

    def _register_custom_types(self):
        """Register NoodleStudio types accessible from QML."""
        # Allow QML to call back into Python
        qmlRegisterType(ChannelBridge, 'NoodleStudio', 1, 0, 'ChannelBridge')

    def create_component(self, qml_path: Path) -> QQmlComponent:
        """Create a QML component from file."""
        component = QQmlComponent(self._engine, QUrl.fromLocalFile(str(qml_path)))

        if component.status() == QQmlComponent.Status.Error:
            for error in component.errors():
                logger.error(f"QML Error: {error.toString()}")
            return None

        return component

    @property
    def engine(self) -> QQmlEngine:
        return self._engine
```

## Widget Gallery

Browse and preview available QML widgets.

```python
class QMLWidgetGallery(QDialog):
    """
    Browse available QML widgets with live previews.

    ┌─────────────────────────────────────────────────────────┐
    │ QML Widget Gallery                               [X]    │
    ├─────────────────────────────────────────────────────────┤
    │ Categories          │ Widgets                           │
    │ ┌─────────────────┐ │ ┌─────────┐ ┌─────────┐          │
    │ │ ▸ Gauges        │ │ │ [    ]  │ │ [    ]  │          │
    │ │ ▸ Meters        │ │ │ Arc     │ │ Speedo  │          │
    │ │ ▸ Indicators    │ │ │ Gauge   │ │ meter   │          │
    │ │ ▸ Buttons       │ │ └─────────┘ └─────────┘          │
    │ │ ▸ Displays      │ │ ┌─────────┐ ┌─────────┐          │
    │ └─────────────────┘ │ │ [    ]  │ │ [    ]  │          │
    │                     │ │ Tank    │ │ Compass │          │
    │ [Import QML...]     │ │ Level   │ │         │          │
    │                     │ └─────────┘ └─────────┘          │
    ├─────────────────────┴───────────────────────────────────┤
    │ Preview: Arc Gauge                                      │
    │ ┌─────────────────────────────────────────────────────┐ │
    │ │                                                     │ │
    │ │                    ◠◡◠                              │ │
    │ │                   ╱   ╲                             │ │
    │ │                  │  ●  │   ← Live preview           │ │
    │ │                   ╲   ╱     with animation          │ │
    │ │                    ‾‾‾                              │ │
    │ │                                                     │ │
    │ └─────────────────────────────────────────────────────┘ │
    │ License: MIT │ Author: Qt Project                       │
    ├─────────────────────────────────────────────────────────┤
    │                              [Cancel]  [Add to Palette] │
    └─────────────────────────────────────────────────────────┘
    ```
    """

    widget_selected = pyqtSignal(WidgetDefinition)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QML Widget Gallery")
        self.setMinimumSize(800, 600)
        self._setup_ui()
        self._load_widgets()
```

## Example QML Widgets

### Arc Gauge

```qml
// arc_gauge.qml
// @name: Arc Gauge
// @description: Smooth animated arc gauge
// @license: MIT
// @category: Gauges

import QtQuick 2.15
import QtQuick.Shapes 1.15

Item {
    id: root
    width: 200
    height: 200

    // Bindable properties (become NoodleStudio channels)
    property real value: 0.5
    property real minimumValue: 0.0
    property real maximumValue: 1.0
    property color needleColor: "#e74c3c"
    property color arcColor: "#3498db"
    property color backgroundColor: "#2c3e50"
    property string label: ""

    // Computed
    property real normalizedValue: (value - minimumValue) / (maximumValue - minimumValue)
    property real angle: -135 + normalizedValue * 270

    // Background arc
    Shape {
        anchors.fill: parent
        ShapePath {
            strokeWidth: 8
            strokeColor: root.backgroundColor
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - 10
                radiusY: root.height / 2 - 10
                startAngle: -135
                sweepAngle: 270
            }
        }
    }

    // Value arc
    Shape {
        anchors.fill: parent
        ShapePath {
            strokeWidth: 8
            strokeColor: root.arcColor
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - 10
                radiusY: root.height / 2 - 10
                startAngle: -135
                sweepAngle: root.normalizedValue * 270
            }
        }

        Behavior on sweepAngle {
            NumberAnimation { duration: 200; easing.type: Easing.OutQuad }
        }
    }

    // Needle
    Rectangle {
        width: 4
        height: root.height / 2 - 20
        radius: 2
        color: root.needleColor
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.verticalCenter
        transformOrigin: Item.Bottom
        rotation: root.angle

        Behavior on rotation {
            NumberAnimation { duration: 200; easing.type: Easing.OutQuad }
        }
    }

    // Center cap
    Rectangle {
        width: 16
        height: 16
        radius: 8
        color: root.needleColor
        anchors.centerIn: parent
    }

    // Label
    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        text: root.label
        color: "#ecf0f1"
        font.pixelSize: 14
    }
}
```

### LED Indicator

```qml
// led_indicator.qml
// @name: LED Indicator
// @description: Glowing LED with on/off state
// @license: MIT
// @category: Indicators

import QtQuick 2.15
import QtQuick.Effects

Item {
    id: root
    width: 32
    height: 32

    property bool active: false
    property color onColor: "#2ecc71"
    property color offColor: "#7f8c8d"
    property string label: ""

    Rectangle {
        id: led
        anchors.centerIn: parent
        width: 24
        height: 24
        radius: 12
        color: root.active ? root.onColor : root.offColor

        Behavior on color {
            ColorAnimation { duration: 150 }
        }
    }

    // Glow effect when active
    MultiEffect {
        source: led
        anchors.fill: led
        visible: root.active
        blurEnabled: true
        blur: 0.5
        blurMax: 32
        colorization: 1.0
        colorizationColor: root.onColor
    }

    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: led.bottom
        anchors.topMargin: 4
        text: root.label
        color: "#bdc3c7"
        font.pixelSize: 10
    }
}
```

## Integration with NoodleStudio

### UI Canvas Integration

QML widgets appear in the widget palette alongside native widgets:

```yaml
# Widget palette categories
categories:
  - name: Layout
    widgets: [Panel, Spacer, Divider]

  - name: Input
    widgets: [Button, Slider, TextField]

  - name: Display
    widgets: [Label, Image, RichText]

  - name: QML Widgets        # ← New category
    subcategories:
      - name: Gauges
        widgets: [ArcGauge, CircularGauge, LinearGauge]
      - name: Indicators
        widgets: [LED, StatusLight, ProgressArc]
      - name: Meters
        widgets: [Speedometer, VUMeter, LevelMeter]
```

### Channel Binding in Inspector

When a QML widget is selected, the inspector shows its properties with channel binding UI:

```
┌─────────────────────────────────────┐
│ Arc Gauge                           │
├─────────────────────────────────────┤
│ Value                               │
│ ┌─────────────────────────────────┐ │
│ │ ─────○───── 0.75                │ │  ← Direct value edit
│ └─────────────────────────────────┘ │
│ [⚡] Bind to channel...             │  ← Channel binding button
│     └─► affect/arousal              │  ← Currently bound to
│                                     │
│ Range                               │
│ Min: [0.0    ]  Max: [1.0    ]     │
│                                     │
│ Appearance                          │
│ Needle: [████████] #e74c3c          │
│ Arc:    [████████] #3498db          │
│                                     │
│ Label                               │
│ [Arousal Level          ]           │
│ [⚡] Bind to channel...             │
└─────────────────────────────────────┘
```

### Example: Affect Dashboard

Wire up QML widgets to show a noodling's affect state:

```yaml
# ui/affect_dashboard.ui.yaml
type: Panel
title: "Affect Monitor"
children:
  - type: qml_widget
    qml: arc_gauge.qml
    properties:
      label: "Valence"
      needleColor: "#3498db"
      minimumValue: -1.0
      maximumValue: 1.0
    bindings:
      value: "noodling/affect/valence"

  - type: qml_widget
    qml: arc_gauge.qml
    properties:
      label: "Arousal"
      needleColor: "#e74c3c"
    bindings:
      value: "noodling/affect/arousal"

  - type: qml_widget
    qml: led_indicator.qml
    properties:
      label: "Active"
      onColor: "#2ecc71"
    bindings:
      active: "noodling/is_thinking"
```

## Licensing & Asset Store

### License Tracking

Every imported QML widget must track its license:

```python
class WidgetLicense:
    """Track license information for imported widgets."""

    # Permissive licenses (can distribute freely)
    PERMISSIVE = ['MIT', 'BSD-2-Clause', 'BSD-3-Clause', 'Apache-2.0', 'ISC', 'Unlicense']

    # Copyleft (must include source)
    COPYLEFT = ['GPL-2.0', 'GPL-3.0', 'LGPL-2.1', 'LGPL-3.0']

    # Qt-specific
    QT_LICENSES = ['LicenseRef-Qt-Commercial', 'GPL-3.0', 'LGPL-3.0']
```

### Asset Store Metadata

For the future asset store, QML widgets include:

```yaml
# asset_store/arc_gauge/manifest.yaml
name: Arc Gauge
version: 1.2.0
author: "NoodleStudio Community"
license: MIT
license_url: https://opensource.org/licenses/MIT

description: |
  A beautiful animated arc gauge perfect for displaying
  normalized values. Smooth animations, customizable colors.

tags: [gauge, dashboard, indicator, animated]

preview_images:
  - preview_light.png
  - preview_dark.png
  - preview_animated.gif

dependencies:
  qt_modules: [QtQuick, QtQuick.Shapes]
  min_qt_version: "6.2.0"

stats:
  downloads: 1247
  rating: 4.8
  ratings_count: 89
```

## Performance Considerations

### Lazy Loading

Don't load QML widgets until needed:

```python
class LazyQMLWidget(WidgetBase):
    """Load QML only when widget becomes visible."""

    def __init__(self, qml_source: str, parent=None):
        super().__init__(parent)
        self._qml_source = qml_source
        self._loaded = False
        self._quick_widget = None

    def showEvent(self, event):
        if not self._loaded:
            self._load_qml()
        super().showEvent(event)

    def _load_qml(self):
        self._quick_widget = QQuickWidget()
        self._quick_widget.setSource(QUrl.fromLocalFile(self._qml_source))
        self._loaded = True
```

### Render Caching

For static widgets, cache the rendered output:

```python
def _update_cache(self):
    """Cache widget render for performance."""
    if not self._needs_update:
        return

    self._cached_pixmap = self._quick_widget.grab()
    self._needs_update = False
```

## QML Input Facets

### The Insight

QML widgets aren't just for display - many are **input devices**. MIDI controllers, joysticks, touch surfaces, barcode scanners. These become **sensor facets** that feed channels into the cognition system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QML AS FACETS                                     │
│                                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │ MIDI Knob   │     │  Joystick   │     │  QR Scanner │          │
│   │ (qmlmidi)   │     │ (QJoystick) │     │  (SCodes)   │          │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘          │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │              QMLInputFacet Wrapper                       │      │
│   │  QML signals ──► Facet outputs ──► Channels             │      │
│   └─────────────────────────────────────────────────────────┘      │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                 COGNITION SYSTEM                         │      │
│   │                                                          │      │
│   │   MIDI CC → arousal    Joystick → attention   QR → input │      │
│   │                                                          │      │
│   └─────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### QMLInputFacet Base Class

```python
class QMLInputFacet(FacetBase):
    """
    Wraps a QML input widget as a sensor facet.

    QML signals become facet outputs that feed channels.

    Use cases:
    - MIDI controller → affect dimensions
    - Joystick → attention direction
    - Barcode scanner → entity lookup
    - Microphone level → arousal
    - Accelerometer → physical state
    """

    facet_type = "qml_input"

    def __init__(self, qml_source: str, signal_mappings: Dict[str, str]):
        """
        Args:
            qml_source: Path to .qml file
            signal_mappings: Map QML signals to output names
                             {"noteOn": "midi_note", "ccChange": "midi_cc"}
        """
        super().__init__()
        self._qml_source = qml_source
        self._signal_mappings = signal_mappings

        # Create hidden QML widget (doesn't need to be visible)
        self._quick_widget = QQuickWidget()
        self._quick_widget.setSource(QUrl.fromLocalFile(qml_source))
        self._root = self._quick_widget.rootObject()

        # Connect QML signals to facet outputs
        self._connect_signals()

    def _connect_signals(self):
        """Wire QML signals to facet output emissions."""
        if not self._root:
            return

        for qml_signal, output_name in self._signal_mappings.items():
            signal = getattr(self._root, qml_signal, None)
            if signal:
                # Create closure to capture output_name
                def make_handler(name):
                    def handler(*args):
                        # Normalize args to single value or dict
                        if len(args) == 1:
                            value = args[0]
                        else:
                            value = args
                        self.emit_output(name, value)
                    return handler

                signal.connect(make_handler(output_name))

    def get_outputs(self) -> List[OutputSpec]:
        """Declare outputs based on signal mappings."""
        return [
            OutputSpec(name=output_name, type='any')
            for output_name in self._signal_mappings.values()
        ]


class MIDIFacet(QMLInputFacet):
    """
    MIDI input as a facet.

    Perfect for live performance, interactive installations,
    or just using hardware knobs to tweak cognition parameters.

    Outputs:
    - note: int (0-127)
    - velocity: float (0-1 normalized)
    - cc: Dict[int, float] (CC number → value)
    - pitch_bend: float (-1 to 1)
    """

    facet_type = "midi_input"

    def __init__(self, device_name: str = None):
        # Use qmlmidi under the hood
        qml = self._generate_midi_qml(device_name)
        super().__init__(
            qml_source=qml,
            signal_mappings={
                'noteOn': 'note_on',
                'noteOff': 'note_off',
                'controlChange': 'cc',
                'pitchBend': 'pitch_bend',
            }
        )

    def _generate_midi_qml(self, device: str) -> str:
        """Generate QML that wraps qmlmidi."""
        return f'''
        import QtQuick 2.15
        import Midi 1.0

        Item {{
            signal noteOn(int note, int velocity)
            signal noteOff(int note)
            signal controlChange(int cc, int value)
            signal pitchBend(int value)

            MidiInput {{
                id: midiIn
                deviceName: "{device or ''}"

                onNoteOn: parent.noteOn(note, velocity)
                onNoteOff: parent.noteOff(note)
                onControlChange: parent.controlChange(control, value)
                onPitchBend: parent.pitchBend(value)
            }}
        }}
        '''


class JoystickFacet(QMLInputFacet):
    """
    Gamepad/joystick as a facet.

    Control noodling attention with a game controller!

    Outputs:
    - left_stick: (x, y) tuple, -1 to 1
    - right_stick: (x, y) tuple
    - triggers: (left, right) tuple, 0 to 1
    - buttons: Dict[str, bool]
    """

    facet_type = "joystick_input"

    # ... similar pattern using QJoysticks
```

### Facet Definition YAML

```yaml
# facets/midi_input.facet.yaml
name: MIDI Input
type: qml_input
category: Hardware / Input
description: |
  Receive MIDI input from controllers, keyboards, or DAWs.
  Map knobs, faders, and keys to cognition parameters.

qml_source: qml/midi_input.qml
qml_dependencies:
  - Midi 1.0  # qmlmidi module

outputs:
  - name: note_on
    type: tuple
    description: "(note: 0-127, velocity: 0-127)"

  - name: note_off
    type: int
    description: "Note number that was released"

  - name: cc
    type: tuple
    description: "(cc_number: 0-127, value: 0-127)"

  - name: pitch_bend
    type: int
    description: "Pitch bend value (-8192 to 8191)"

properties:
  - name: device
    type: string
    default: ""
    description: "MIDI device name (empty = first available)"

  - name: channel
    type: int
    default: -1
    description: "MIDI channel filter (-1 = all channels)"

# Example usage in assembly
example: |
  # Map MIDI CC #1 (mod wheel) to arousal
  - source: midi_input.cc
    filter: "cc[0] == 1"  # CC #1 only
    transform: "cc[1] / 127.0"  # Normalize to 0-1
    target: affect.arousal
```

## Ecosystem Value Extraction

### The Full Picture

The QML ecosystem gives us FIVE categories of reusable components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 QML ECOSYSTEM → NOODLESTUDIO                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DISPLAY WIDGETS          →  UI Canvas Components                 │
│     Gauges, meters, LEDs         Show cognition state                │
│     Charts, graphs               Visualize data                      │
│     Automotive dashboards        Professional aesthetics             │
│                                                                      │
│  2. INPUT WIDGETS            →  Sensor Facets                        │
│     MIDI (qmlmidi)               Hardware control                    │
│     Joysticks (QJoysticks)       Game controllers                    │
│     Barcode/QR (SCodes)          Object identification               │
│     Virtual keyboard             Text input                          │
│     Touch surfaces               Gesture input                       │
│                                                                      │
│  3. DATA VISUALIZATION       →  Inspection Tools                     │
│     Node editors                 Facet assembly viz                  │
│     Network graphs               Charm network viz                   │
│     3D viewers (QuickVtk)        Spatial cognition                   │
│     Plotting (qnite, Chart.qml)  Time series analysis                │
│                                                                      │
│  4. EFFECTS & ANIMATION      →  Visual Polish                        │
│     Particle systems             Cognitive "sparks"                  │
│     Shaders                      Custom rendering                    │
│     Transitions                  State change feedback               │
│                                                                      │
│  5. EMBEDDED APPLICATIONS    →  Power User Tools                     │
│     Media players                Content playback                    │
│     File browsers                Asset management                    │
│     Terminal emulators           Debug/scripting                     │
│     (Theoretically: DesQ)        (A desktop in your app 😂)          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Asset Store Categories

When we launch the asset store, QML components slot into existing categories:

```yaml
asset_store_categories:
  widgets:
    - Gauges & Meters
    - Indicators & LEDs
    - Charts & Graphs
    - Buttons & Controls
    - Automotive
    - Industrial
    - Aviation
    - Medical

  facets:
    - Input / MIDI
    - Input / Game Controllers
    - Input / Sensors
    - Input / Camera & Vision
    - Output / DMX & Lighting
    - Output / Serial & Hardware

  themes:
    - Dark Mode
    - Light Mode
    - Automotive HMI
    - Industrial SCADA
    - Retro / Skeuomorphic

  complete_dashboards:
    - Affect Monitor
    - Cognition Debugger
    - Performance Metrics
    - Training Dashboard
```

### Developer Value Propositions

**For Widget Creators:**
```
"Turn your QML skills into income. Create beautiful gauges,
import them into NoodleStudio, sell on the asset store."
```

**For Hardware Integrators:**
```
"Already have QML code for your MIDI controller / custom sensor?
Wrap it as a facet, connect it to cognition in minutes."
```

**For UI Designers:**
```
"Access thousands of production-quality components.
Mercedes dashboard gauges. Aviation instruments. Industrial HMIs.
Drag, drop, wire to channels. Ship."
```

**For Researchers:**
```
"Visualize cognitive state in real-time with professional
dashboards. Export publication-ready charts. Record sessions."
```

### Import Sources

Where to find QML components:

| Source | Quality | License | Notes |
|--------|---------|---------|-------|
| [RoniaKit](https://github.com/Roniasoft/RoniaKit) | ⭐⭐⭐⭐⭐ | Apache 2.0 | Best gauge library |
| [QtQuickCarGauges](https://github.com/lemirep/QtQuickCarGauges) | ⭐⭐⭐⭐ | Permissive | Drop-in gauges |
| [Qt-HMI-Display-UI](https://github.com/cppqtdev/Qt-HMI-Display-UI) | ⭐⭐⭐⭐⭐ | Open | Full automotive dashboard |
| [Fluid](https://github.com/lirios/fluid) | ⭐⭐⭐⭐ | MPL 2.0 | Material Design |
| [qmlmidi](https://github.com/jarnoh/qmlmidi) | ⭐⭐⭐ | MIT | MIDI I/O |
| [QJoysticks](https://github.com/alex-spataru/QJoysticks) | ⭐⭐⭐⭐ | MIT | Gamepad input |
| [nodeeditor](https://github.com/paceholder/nodeeditor) | ⭐⭐⭐⭐⭐ | BSD | Node graph editing |
| [QuickVtk](https://github.com/qCring/QuickVtk) | ⭐⭐⭐⭐ | BSD | 3D visualization |
| Qt Marketplace | ⭐⭐⭐⭐⭐ | Commercial | Professional components |

### The "Batteries Included" Bundle

Ship NoodleStudio with a curated set of QML widgets pre-imported:

```
noodlestudio/
└── resources/
    └── qml_widgets/
        ├── gauges/
        │   ├── arc_gauge.qml
        │   ├── circular_gauge.qml
        │   ├── linear_gauge.qml
        │   └── speedometer.qml
        ├── indicators/
        │   ├── led.qml
        │   ├── status_light.qml
        │   └── level_meter.qml
        ├── charts/
        │   ├── line_chart.qml
        │   ├── bar_chart.qml
        │   └── pie_chart.qml
        └── input/
            ├── midi_input.qml
            ├── joystick_input.qml
            └── virtual_keyboard.qml
```

Users get gorgeous dashboards out of the box. Power users import more from the ecosystem.

## The Wild Possibilities

### What's Technically Possible

Since QQuickWidget runs full QML, these are all *technically* feasible:

| Component | Sanity Level | Use Case |
|-----------|--------------|----------|
| Automotive gauge | ✅ Sane | Affect dashboards |
| MIDI controller | ✅ Sane | Live performance |
| Node editor | ✅ Sane | Visual facet editing |
| Video player | 🤔 Maybe | Media playback facet |
| Terminal emulator | 🤔 Maybe | Debug console |
| Web browser | 😅 Pushing it | Web content facet |
| File manager | 😅 Pushing it | Asset browser alt |
| DesQ desktop | 🤪 Chaos | Your noodling runs Linux |

### The DesQ Easter Egg

```python
# DO NOT SHIP THIS. But it would work.
class DesktopEnvironmentWidget(QMLWidgetWrapper):
    """
    Embed an entire desktop environment in a panel.

    Your noodling can have its own desktop. With windows.
    And a taskbar. This is not a good idea but it IS possible.
    """

    def __init__(self):
        super().__init__("desq/shell.qml")
        logger.warning("You're embedding a desktop environment. Why?")
```

*We're not going to do this. But the fact that we COULD is delightful.*

## Future Enhancements

1. **QML Editor** - Edit QML directly in NoodleStudio for power users
2. **Widget Converter** - Convert native widgets to QML for sharing
3. **Theme System** - Apply consistent styling across all QML widgets
4. **Animation Editor** - Modify QML animations visually
5. **Remote Widget Library** - Download widgets from asset store on demand
6. **Custom QML Types** - Register NoodleStudio types for use in QML
7. **QML Facet Wizard** - GUI for wrapping QML input widgets as facets
8. **Live Preview** - See channel data flowing through QML widgets in real-time
9. **Performance Profiler** - Identify slow QML components

## Complete Example: Chord Mood Analyzer

A real-world app showing QML input facets, QML display widgets, and native components working together.

### The App

**Chord Mood** analyzes piano chord progressions for emotional sentiment, mapping them to PAD (Pleasure-Arousal-Dominance) values. A pianist plays, the app analyzes the harmonic content, and a noodling discusses the emotional qualities of the music.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     FACET ASSEMBLY                               │   │
│   │                                                                  │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │   │
│   │   │ MIDIInputFacet│───▶│ChordAnalyzer │───▶│SentimentMapper│     │   │
│   │   │ (QML: qmlmidi)│    │   Facet      │    │    Facet      │     │   │
│   │   └──────────────┘    └──────────────┘    └───────┬───────┘     │   │
│   │         │                                         │              │   │
│   │         │ notes[]                                 │ PAD values   │   │
│   │         │ velocity[]                              ▼              │   │
│   │         │                              ┌──────────────────┐      │   │
│   │         │                              │ Output Channels  │      │   │
│   │         │                              │ • affect/valence │      │   │
│   │         │                              │ • affect/arousal │      │   │
│   │         │                              │ • affect/dominance│     │   │
│   │         │                              │ • chord/current  │      │   │
│   │         │                              │ • mood/label     │      │   │
│   │         │                              └────────┬─────────┘      │   │
│   └─────────┼───────────────────────────────────────┼────────────────┘   │
│             │                                       │                    │
│             ▼                                       ▼                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        UI CANVAS                                 │   │
│   │                                                                  │   │
│   │  ┌────────────────────┐  ┌────────────────────────────────────┐ │   │
│   │  │ CHAT (Native)      │  │ DASHBOARD (QML Widgets)            │ │   │
│   │  │                    │  │                                    │ │   │
│   │  │ ChatHistory        │  │  ┌────────┐┌────────┐┌────────┐   │ │   │
│   │  │ ┌────────────────┐ │  │  │ P 0.82 ││ A 0.65 ││ D 0.41 │   │ │   │
│   │  │ │ Agent: That Dm7│ │  │  │  ▓▓▓▓  ││  ▓▓▓░  ││  ▓▓░░  │   │ │   │
│   │  │ │ to G7 creates  │ │  │  │ Valence││ Arousal││Dominanc│   │ │   │
│   │  │ │ tension...     │ │  │  └────────┘└────────┘└────────┘   │ │   │
│   │  │ └────────────────┘ │  │                                    │ │   │
│   │  │                    │  │  ┌──────────────────────────────┐  │ │   │
│   │  │ TextInput          │  │  │ CHORD: Dm7 → G7 → Cmaj7      │  │ │   │
│   │  │ ┌────────────────┐ │  │  │ MOOD:  Longing → Resolution  │  │ │   │
│   │  │ │ Why does that  │ │  │  └──────────────────────────────┘  │ │   │
│   │  │ │ sound sad?     │ │  │                                    │ │   │
│   │  │ └────────────────┘ │  │  ┌──────────────────────────────┐  │ │   │
│   │  │ [Send]             │  │  │ ◉ Yamaha P-125 Connected     │  │ │   │
│   │  └────────────────────┘  │  └──────────────────────────────┘  │ │   │
│   │                          └────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Facet Assembly

```yaml
# assemblies/chord_analyzer.facet.yaml
name: Chord Mood Analyzer
description: Analyzes piano chords for emotional content

facets:
  # MIDI input via QML wrapper
  - type: qml_input
    id: midi_in
    qml_source: "qml/midi_input.qml"
    properties:
      device: "Yamaha P-125"  # or empty for first available
    outputs:
      - note_on    # (note: int, velocity: int)
      - note_off   # (note: int)
      - sustain    # (value: int) pedal

  # Chord detection (custom Python facet)
  - type: python
    id: chord_analyzer
    script: "scripts/chord_analyzer.py"
    inputs:
      notes: "midi_in.note_on"
    outputs:
      - chord_name      # "Dm7", "G7", "Cmaj7"
      - chord_notes     # [62, 65, 69, 72]
      - chord_quality   # "minor7", "dominant7", "major7"
      - progression     # ["Dm7", "G7", "Cmaj7"]

  # Sentiment mapping (chord qualities → PAD)
  - type: python
    id: sentiment_mapper
    script: "scripts/sentiment_mapper.py"
    inputs:
      chord: "chord_analyzer.chord_name"
      quality: "chord_analyzer.chord_quality"
      progression: "chord_analyzer.progression"
    outputs:
      - valence     # -1 to 1 (minor=negative, major=positive)
      - arousal     # 0 to 1 (dissonance, tempo of changes)
      - dominance   # 0 to 1 (root motion strength)
      - mood_label  # "melancholic", "triumphant", "tense"

# Publish to world channels
outputs:
  - source: sentiment_mapper.valence
    channel: affect/valence
  - source: sentiment_mapper.arousal
    channel: affect/arousal
  - source: sentiment_mapper.dominance
    channel: affect/dominance
  - source: chord_analyzer.chord_name
    channel: chord/current
  - source: chord_analyzer.progression
    channel: chord/progression
  - source: sentiment_mapper.mood_label
    channel: mood/label
```

### The UI Layout

```yaml
# ui/main.ui.yaml
type: Panel
layout: horizontal
children:

  # ─────────────────────────────────────────────────────────
  # LEFT PANEL: Chat Interface (Native Delphi-style widgets)
  # ─────────────────────────────────────────────────────────
  - type: Panel
    width: 320
    layout: vertical
    padding: 8
    children:
      - type: Label
        text: "Chord Mood Analyst"
        style: heading

      - type: ChatHistory
        id: chat
        flex: 1
        channel: "chat/messages"

      - type: TextInput
        id: user_input
        placeholder: "Ask about the music..."

      - type: Button
        text: "Send"
        onClick:
          action: send_to_noodling
          target: mood_analyst
          message: "{user_input.value}"

  # ─────────────────────────────────────────────────────────
  # RIGHT PANEL: Sexy QML Dashboard
  # ─────────────────────────────────────────────────────────
  - type: Panel
    flex: 1
    layout: vertical
    padding: 16
    children:

      # PAD Meters Row (QML VU meters from RoniaKit or similar)
      - type: Panel
        layout: horizontal
        height: 180
        spacing: 16
        children:
          - type: QMLWidget
            qml_source: "qml/widgets/vu_meter.qml"
            flex: 1
            properties:
              label: "Valence"
              color: "#3498db"
              minValue: -1.0
              maxValue: 1.0
              showValue: true
            bindings:
              value: "affect/valence"

          - type: QMLWidget
            qml_source: "qml/widgets/vu_meter.qml"
            flex: 1
            properties:
              label: "Arousal"
              color: "#e74c3c"
              minValue: 0.0
              maxValue: 1.0
            bindings:
              value: "affect/arousal"

          - type: QMLWidget
            qml_source: "qml/widgets/vu_meter.qml"
            flex: 1
            properties:
              label: "Dominance"
              color: "#2ecc71"
              minValue: 0.0
              maxValue: 1.0
            bindings:
              value: "affect/dominance"

      # Chord progression display (custom QML)
      - type: QMLWidget
        qml_source: "qml/widgets/chord_display.qml"
        height: 120
        bindings:
          currentChord: "chord/current"
          progression: "chord/progression"
          moodLabel: "mood/label"

      # Piano roll visualization (QML)
      - type: QMLWidget
        qml_source: "qml/widgets/piano_roll.qml"
        flex: 1
        bindings:
          notes: "midi_in/active_notes"

      # MIDI connection status (Native LED + Label)
      - type: Panel
        layout: horizontal
        height: 32
        children:
          - type: LED
            size: 12
            color: "#2ecc71"
            bindings:
              active: "midi/connected"

          - type: Label
            bindings:
              text: "midi/device_name"
            style: caption
```

### Component Breakdown

| Component | Type | Purpose |
|-----------|------|---------|
| `MIDIInputFacet` | **QML Input Facet** | Receives MIDI from keyboard via qmlmidi |
| `ChordAnalyzer` | **Python Facet** | Detects chord names from note clusters |
| `SentimentMapper` | **Python Facet** | Maps chord qualities to PAD values |
| `ChatHistory` | **Native Widget** | Conversation with the analyst noodling |
| `TextInput` | **Native Widget** | User message input |
| `Button` | **Native Widget** | Send message |
| `VU Meters` | **QML Widget** | Sexy PAD value display |
| `ChordDisplay` | **QML Widget** | Shows progression and mood label |
| `PianoRoll` | **QML Widget** | Visualizes active notes |
| `LED` | **Native Widget** | MIDI connection indicator |
| `Label` | **Native Widget** | Device name display |

### The Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ QML INPUT FACETS                                                │
│   • Receive external data (MIDI, joystick, sensors)            │
│   • Live in Facet Assemblies                                    │
│   • Output to channels                                          │
│   • User never sees QML - just configures device in inspector   │
├─────────────────────────────────────────────────────────────────┤
│ QML DISPLAY WIDGETS                                             │
│   • Show data beautifully (gauges, meters, visualizations)      │
│   • Live on UI Canvas                                           │
│   • Bind to channels for live updates                           │
│   • Imported from ecosystem (RoniaKit, automotive, etc.)        │
├─────────────────────────────────────────────────────────────────┤
│ NATIVE WIDGETS                                                  │
│   • Reliable Delphi-style basics (Button, TextInput, Chat)      │
│   • Always work, no QML dependency                              │
│   • Match our dark monochromatic theme                          │
│   • The foundation everything else builds on                    │
└─────────────────────────────────────────────────────────────────┘

All three layers communicate through CHANNELS.
```

### Why This Is Hard Elsewhere

Building this app in other frameworks:

| Framework | Difficulty | Why |
|-----------|------------|-----|
| Unity | Hard | No built-in MIDI, QML, or chat UI |
| Unreal | Very Hard | Blueprint spaghetti, C++ for MIDI |
| Web (React) | Medium | WebMIDI exists but no AI integration |
| Pure Python | Hard | GUI + MIDI + AI glue code nightmare |
| **NoodleStudio** | **Easy** | Drag facets, wire channels, done |

The magic is that MIDI input, chord analysis, sentiment mapping, AI chat, and sexy visualizations all speak the same language: **channels**.

---

## References

- [Qt Quick Overview](https://doc.qt.io/qt-6/qtquick-index.html)
- [QML Reference](https://doc.qt.io/qt-6/qmlreference.html)
- [Qt Quick Controls](https://doc.qt.io/qt-6/qtquickcontrols-index.html)
- [PyQt6 QQuickWidget](https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtquickwidgets/qquickwidget.html)

---

*"The best interface is no interface... but if you must have one, make it gorgeous."*

Made with love by Caity & Claude
