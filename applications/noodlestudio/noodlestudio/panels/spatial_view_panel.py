"""
Spatial View Panel - Qt Quick 3D visualization of stage zones.

Renders zones as wireframe spheres in 3D space with connections shown as lines.
Click a zone to select it. Orbit camera to explore.

Controls:
  - W: Toggle wireframe/solid rendering
  - T: Toggle ghost mode (transparent, no occlusion)
  - F: Focus on selected zone
  - A: Frame all zones (snaps to top-down view)
  - Alt+LMB: Tumble (orbit camera)
  - Alt+MMB: Track (pan camera)
  - Alt+RMB: Dolly (zoom)
  - Scroll: Zoom (with limits)
  - Click: Select zone
  - Right-click: Context menu (add/delete/rename zones)

Author: Caitlyn + Claude
Date: December 18, 2025
"""

import os
import yaml
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QMenu, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QObject, QTimer
from PyQt6.QtGui import QColor, QKeyEvent, QShortcut, QKeySequence

# Qt Quick imports for 3D
from PyQt6.QtQuick import QQuickView
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtQml import QQmlApplicationEngine, qmlRegisterType


@dataclass
class ZoneData:
    """Zone information for 3D rendering."""
    id: str
    name: str
    center: List[float]  # [x, y, z]
    radius: float
    falloff: float
    shape: str  # sphere, cylinder, box
    color: QColor
    exits: Dict[str, str]  # direction -> zone_id
    file_path: str = ""  # Path to zone yaml file
    description: str = ""
    perception: Dict[str, Any] = field(default_factory=dict)
    ambient: Dict[str, Any] = field(default_factory=dict)


class ZoneModel(QObject):
    """
    Bridge object that exposes zone data to QML.

    Registered as a context property so QML can access zone positions,
    colors, and connections.
    """

    zonesChanged = pyqtSignal()
    zoneSelected = pyqtSignal(str, dict)  # zone_id, zone_data

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zones: Dict[str, ZoneData] = {}
        self._connections: List[tuple] = []  # [(from_id, to_id), ...]
        self._selected_zone: Optional[str] = None
        self._stage_path: Optional[str] = None

    @property
    def selected_zone(self) -> Optional[str]:
        return self._selected_zone

    @property
    def zones(self) -> Dict[str, ZoneData]:
        return self._zones

    @pyqtSlot(result=list)
    def getZones(self) -> list:
        """Return list of zone data for QML consumption."""
        zones = []
        for zone_id, zone in self._zones.items():
            zones.append({
                'id': zone.id,
                'name': zone.name,
                'x': zone.center[0],
                'y': zone.center[1],
                'z': zone.center[2],
                'radius': zone.radius,
                'color': zone.color.name(),
                'selected': zone.id == self._selected_zone
            })
        return zones

    @pyqtSlot(result=list)
    def getConnections(self) -> list:
        """Return list of zone connections for QML line rendering."""
        connections = []
        for from_id, to_id in self._connections:
            if from_id in self._zones and to_id in self._zones:
                from_zone = self._zones[from_id]
                to_zone = self._zones[to_id]
                connections.append({
                    'from_x': from_zone.center[0],
                    'from_y': from_zone.center[1],
                    'from_z': from_zone.center[2],
                    'to_x': to_zone.center[0],
                    'to_y': to_zone.center[1],
                    'to_z': to_zone.center[2]
                })
        return connections

    @pyqtSlot(str, result=dict)
    def getZoneData(self, zone_id: str) -> dict:
        """Get full zone data for Inspector panel."""
        if zone_id not in self._zones:
            return {}
        zone = self._zones[zone_id]
        return {
            'type': 'zone',
            'id': zone.id,
            'name': zone.name,
            'center': zone.center,
            'radius': zone.radius,
            'falloff': zone.falloff,
            'shape': zone.shape,
            'exits': zone.exits,
            'description': zone.description,
            'perception': zone.perception,
            'ambient': zone.ambient,
            'file_path': zone.file_path
        }

    @pyqtSlot(str)
    def selectZone(self, zone_id: str):
        """Called from QML when a zone sphere is clicked."""
        self._selected_zone = zone_id
        if zone_id in self._zones:
            self.zoneSelected.emit(zone_id, self.getZoneData(zone_id))
        self.zonesChanged.emit()

    def get_selected_zone_data(self) -> Optional[ZoneData]:
        """Get the currently selected zone's data."""
        if self._selected_zone and self._selected_zone in self._zones:
            return self._zones[self._selected_zone]
        return None

    def get_scene_bounds(self) -> tuple:
        """Calculate bounding box of all zones."""
        if not self._zones:
            return (0, 0, 0), (100, 50, 100)

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for zone in self._zones.values():
            r = zone.radius
            min_x = min(min_x, zone.center[0] - r)
            max_x = max(max_x, zone.center[0] + r)
            min_y = min(min_y, zone.center[1] - r)
            max_y = max(max_y, zone.center[1] + r)
            min_z = min(min_z, zone.center[2] - r)
            max_z = max(max_z, zone.center[2] + r)

        return (min_x, min_y, min_z), (max_x, max_y, max_z)

    def load_stage(self, stage_path: str):
        """Load zones from a stage folder."""
        self._zones.clear()
        self._connections.clear()
        self._stage_path = stage_path
        self._selected_zone = None

        # Define color palette for zones
        palette = [
            QColor("#4A90A4"),  # Teal
            QColor("#E07B53"),  # Coral
            QColor("#8B7355"),  # Brown
            QColor("#6B8E4E"),  # Olive
            QColor("#9B6B9E"),  # Purple
            QColor("#C4A35A"),  # Gold
            QColor("#5B8BA0"),  # Steel blue
            QColor("#A0522D"),  # Sienna
            QColor("#708090"),  # Slate
            QColor("#BC8F8F"),  # Rosy brown
            QColor("#5F9EA0"),  # Cadet blue
            QColor("#D2691E"),  # Chocolate
        ]
        color_index = 0

        # Load stage.yaml for zone graph
        stage_yaml = os.path.join(stage_path, "stage.yaml")
        zone_graph = {}
        if os.path.exists(stage_yaml):
            try:
                with open(stage_yaml, 'r') as f:
                    stage_data = yaml.safe_load(f) or {}
                    zone_graph = stage_data.get('zone_graph', {})
            except Exception as e:
                print(f"Error loading stage.yaml: {e}")

        # Load zones from Zones/*.zone.yaml
        zones_dir = os.path.join(stage_path, "Zones")
        if os.path.exists(zones_dir):
            for filename in sorted(os.listdir(zones_dir)):
                if filename.endswith(".zone.yaml"):
                    zone_path = os.path.join(zones_dir, filename)
                    try:
                        with open(zone_path, 'r') as f:
                            zone_data = yaml.safe_load(f) or {}

                        zone_id = zone_data.get('id', filename.replace('.zone.yaml', ''))
                        spatial = zone_data.get('spatial', {})
                        text = zone_data.get('text', {})

                        zone = ZoneData(
                            id=zone_id,
                            name=zone_data.get('name', zone_id),
                            center=spatial.get('center', [0, 0, 0]),
                            radius=spatial.get('radius', 10),
                            falloff=spatial.get('falloff', 5),
                            shape=spatial.get('shape', 'sphere'),
                            color=palette[color_index % len(palette)],
                            exits=text.get('exits', {}),
                            file_path=zone_path,
                            description=text.get('description', ''),
                            perception=zone_data.get('perception', {}),
                            ambient=zone_data.get('ambient', {})
                        )
                        self._zones[zone_id] = zone
                        color_index += 1

                        # Build connections from exits
                        for direction, target_id in zone.exits.items():
                            # Avoid duplicate connections
                            if (target_id, zone_id) not in self._connections:
                                self._connections.append((zone_id, target_id))

                    except Exception as e:
                        print(f"Error loading zone {filename}: {e}")

        # Also add connections from zone_graph (if stage.yaml has it)
        for from_zone, targets in zone_graph.items():
            for to_zone in targets:
                if (from_zone, to_zone) not in self._connections and (to_zone, from_zone) not in self._connections:
                    self._connections.append((from_zone, to_zone))

        self.zonesChanged.emit()


class SpatialViewPanel(QWidget):
    """
    3D spatial view of stage zones using Qt Quick 3D.

    Shows zones as colored spheres at their center positions,
    with lines connecting zones that have exits to each other.

    Shortcuts:
        F: Focus on selected zone
        A: Frame all zones

    Signals:
        zoneSelected: Emitted when user clicks a zone (zone_id, zone_data)
    """

    zoneSelected = pyqtSignal(str, dict)

    # Camera limits
    MIN_ZOOM_DISTANCE = 5
    MAX_ZOOM_DISTANCE = 20000  # Very large for big scenes
    CAMERA_NEAR_CLIP = 0.1
    CAMERA_FAR_CLIP = 50000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = None
        self.current_stage = None

        # Zone data bridge for QML
        self.zone_model = ZoneModel(self)
        self.zone_model.zoneSelected.connect(self._on_zone_selected)

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # F = Focus on selected
        self.focus_shortcut = QShortcut(QKeySequence("F"), self)
        self.focus_shortcut.activated.connect(self._focus_selected)

        # A = Frame all (top-down)
        self.frame_all_shortcut = QShortcut(QKeySequence("A"), self)
        self.frame_all_shortcut.activated.connect(self._fit_all)

        # W = Toggle wireframe
        self.wireframe_shortcut = QShortcut(QKeySequence("W"), self)
        self.wireframe_shortcut.activated.connect(self._toggle_wireframe)

        # T = Toggle ghost mode
        self.ghost_shortcut = QShortcut(QKeySequence("T"), self)
        self.ghost_shortcut.activated.connect(self._toggle_ghost)

    def _setup_ui(self):
        """Build the UI with toolbar and 3D view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-bottom: 1px solid #1A1A1A;
            }
            QPushButton {
                background-color: transparent;
                color: #CCCCCC;
                border: none;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3E3E3E;
            }
            QPushButton:pressed {
                background-color: #4A4A4A;
            }
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: 1px solid #555;
                padding: 2px 6px;
                border-radius: 2px;
                min-width: 100px;
            }
            QComboBox:hover {
                border: 1px solid #777;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 6px;
            }
            QLabel {
                color: #888;
                padding: 0 4px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 0, 4, 0)
        toolbar_layout.setSpacing(4)

        # Stage selector
        stage_label = QLabel("Stage:")
        toolbar_layout.addWidget(stage_label)

        self.stage_selector = QComboBox()
        self.stage_selector.currentTextChanged.connect(self._on_stage_changed)
        toolbar_layout.addWidget(self.stage_selector)

        toolbar_layout.addSpacing(8)

        # Zone selector
        zone_label = QLabel("Zone:")
        toolbar_layout.addWidget(zone_label)

        self.zone_selector = QComboBox()
        self.zone_selector.setMinimumWidth(140)
        self.zone_selector.currentTextChanged.connect(self._on_zone_dropdown_changed)
        toolbar_layout.addWidget(self.zone_selector)

        toolbar_layout.addStretch()

        # View controls
        self.fit_btn = QPushButton("Fit All (A)")
        self.fit_btn.clicked.connect(self._fit_all)
        toolbar_layout.addWidget(self.fit_btn)

        self.focus_btn = QPushButton("Focus (F)")
        self.focus_btn.clicked.connect(self._focus_selected)
        toolbar_layout.addWidget(self.focus_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_view)
        toolbar_layout.addWidget(self.reset_btn)

        layout.addWidget(toolbar)

        # Qt Quick 3D view - MUST expand to fill available space
        self.quick_widget = QQuickWidget()
        self.quick_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self.quick_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.quick_widget.setMinimumSize(200, 200)
        from PyQt6.QtWidgets import QSizePolicy
        self.quick_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set context properties before loading QML
        self.quick_widget.rootContext().setContextProperty("zoneModel", self.zone_model)
        self.quick_widget.rootContext().setContextProperty("minZoomDistance", self.MIN_ZOOM_DISTANCE)
        self.quick_widget.rootContext().setContextProperty("maxZoomDistance", self.MAX_ZOOM_DISTANCE)
        self.quick_widget.rootContext().setContextProperty("cameraNearClip", self.CAMERA_NEAR_CLIP)
        self.quick_widget.rootContext().setContextProperty("cameraFarClip", self.CAMERA_FAR_CLIP)

        # Load QML
        qml_path = os.path.join(os.path.dirname(__file__), "../qml/SpatialView.qml")

        # Always recreate QML to get latest version
        self._create_qml_file(qml_path)

        self.quick_widget.setSource(QUrl.fromLocalFile(qml_path))

        # Check for QML errors
        if self.quick_widget.status() == QQuickWidget.Status.Error:
            errors = self.quick_widget.errors()
            for error in errors:
                print(f"QML Error: {error.toString()}")
            # Show fallback
            fallback = QLabel("QML Error - check console")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: #FF6B6B; background: #1E1E1E;")
            layout.addWidget(fallback)
        else:
            layout.addWidget(self.quick_widget)

        # Enable context menu on the quick widget
        self.quick_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.quick_widget.customContextMenuRequested.connect(self._show_context_menu)

        # Status bar
        self.status_label = QLabel("No stage loaded | F: Focus | A: Frame All | Right-click: Options")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2D2D2D;
                color: #888;
                padding: 4px 8px;
                font-size: 11px;
                border-top: 1px solid #1A1A1A;
            }
        """)
        layout.addWidget(self.status_label)

    def _create_qml_file(self, qml_path: str):
        """Create the QML file for the 3D view."""
        os.makedirs(os.path.dirname(qml_path), exist_ok=True)

        qml_content = '''import QtQuick
import QtQuick3D
import QtQuick3D.Helpers

Rectangle {
    id: root
    color: "#1E1E1E"
    focus: true

    // Camera state - top-down by default
    property real cameraDistance: 250
    property real cameraYaw: 0
    property real cameraPitch: -90  // Top-down view
    property vector3d cameraTarget: Qt.vector3d(0, 0, -80)

    // Render mode: false = solid, true = wireframe
    property bool wireframeMode: true  // Default to wireframe
    // Ghost mode: shows backfaces (no occlusion), everything translucent
    property bool ghostMode: false

    // Limits from Python (fallbacks if not set)
    property real minDistance: minZoomDistance || 5
    property real maxDistance: maxZoomDistance || 20000
    property real nearClip: cameraNearClip || 0.1
    property real farClip: cameraFarClip || 50000

    // 3D View - FAST, CHEAP rendering
    View3D {
        id: view3d
        anchors.fill: parent

        environment: SceneEnvironment {
            clearColor: "#1A1A1A"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA

            // Actual wireframe rendering - shows triangle edges
            debugSettings: DebugSettings {
                wireframeEnabled: wireframeMode
            }
        }

        // Orbit camera
        PerspectiveCamera {
            id: camera
            position: updateCameraPosition()
            eulerRotation: Qt.vector3d(cameraPitch, cameraYaw, 0)
            clipFar: farClip
            clipNear: nearClip

            function updateCameraPosition() {
                var yawRad = cameraYaw * Math.PI / 180
                var pitchRad = cameraPitch * Math.PI / 180

                return Qt.vector3d(
                    cameraTarget.x + cameraDistance * Math.cos(pitchRad) * Math.sin(yawRad),
                    cameraTarget.y + cameraDistance * Math.sin(-pitchRad),
                    cameraTarget.z + cameraDistance * Math.cos(pitchRad) * Math.cos(yawRad)
                )
            }
        }

        // Single light
        DirectionalLight {
            eulerRotation.x: -35
            eulerRotation.y: -25
            color: "#FFFFFF"
            brightness: 1.0
            ambientColor: "#404040"
        }

        // Ground plane
        Model {
            visible: !ghostMode
            source: "#Rectangle"
            scale: Qt.vector3d(50, 50, 1)
            eulerRotation.x: -90
            y: -1
            materials: DefaultMaterial {
                diffuseColor: "#252525"
                lighting: DefaultMaterial.NoLighting
            }
        }

        // Zone boxes - simple cubes
        Repeater3D {
            id: zoneRepeater
            model: zoneModel ? zoneModel.getZones() : []

            delegate: Node {
                id: zoneNode
                property var zoneData: modelData
                property real s: zoneData.radius
                position: Qt.vector3d(zoneData.x, zoneData.y, zoneData.z)

                // Zone cube
                Model {
                    source: "#Cube"
                    scale: Qt.vector3d(s, s, s)
                    materials: DefaultMaterial {
                        // Wireframe mode: gray so lines stand out
                        // Solid mode: use zone color
                        diffuseColor: wireframeMode ? "#444444" : (zoneData.selected ? "#FFFFFF" : zoneData.color)
                        opacity: ghostMode ? 0.2 : (zoneData.selected ? 1.0 : 0.8)
                        lighting: DefaultMaterial.NoLighting
                    }
                }

                // Selection highlight - yellow glow
                Model {
                    visible: zoneData.selected
                    source: "#Cube"
                    scale: Qt.vector3d(s * 1.12, s * 1.12, s * 1.12)
                    materials: DefaultMaterial {
                        diffuseColor: "#FFFF00"
                        opacity: 0.3
                        lighting: DefaultMaterial.NoLighting
                    }
                }
            }
        }
    }

    // 2D overlay for connection lines - DISABLED (projection math needs work)
    // TODO: Fix 3D-to-2D projection or use 3D line geometry instead
    Canvas {
        id: connectionCanvas
        anchors.fill: parent
        visible: false  // Hidden until projection is fixed

        property var connections: zoneModel ? zoneModel.getConnections() : []

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
        }

        Connections {
            target: zoneModel
            function onZonesChanged() {
                connectionCanvas.connections = zoneModel.getConnections()
            }
        }
    }

    // Mouse interaction - Unity/Maya style controls
    // Alt+LMB: Tumble (orbit)
    // Alt+MMB: Track (pan)
    // Alt+RMB: Dolly (zoom)
    // Scroll: Zoom
    // LMB (no modifier): Select
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        hoverEnabled: true

        property point lastPos
        property bool dragging: false

        onPressed: function(mouse) {
            lastPos = Qt.point(mouse.x, mouse.y)
            dragging = false
            root.forceActiveFocus()
        }

        onReleased: function(mouse) {
            // Only select if not dragging and no Alt modifier
            if (!dragging && mouse.button === Qt.LeftButton && !(mouse.modifiers & Qt.AltModifier)) {
                pickZoneAt(mouse.x, mouse.y)
            }
            dragging = false
        }

        onPositionChanged: function(mouse) {
            if (!(mouse.buttons & (Qt.LeftButton | Qt.RightButton | Qt.MiddleButton))) return

            var dx = mouse.x - lastPos.x
            var dy = mouse.y - lastPos.y

            if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
                dragging = true
            }

            lastPos = Qt.point(mouse.x, mouse.y)

            var hasAlt = mouse.modifiers & Qt.AltModifier

            // Alt + LMB: Tumble (orbit)
            if ((mouse.buttons & Qt.LeftButton) && hasAlt) {
                cameraYaw += dx * 0.5
                cameraPitch = Math.max(-89.9, Math.min(89.9, cameraPitch - dy * 0.5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
            }
            // Alt + MMB: Track (pan)
            else if ((mouse.buttons & Qt.MiddleButton) && hasAlt) {
                panCamera(dx, dy)
            }
            // MMB without Alt: Also pan (convenience)
            else if ((mouse.buttons & Qt.MiddleButton) && !hasAlt) {
                panCamera(dx, dy)
            }
            // Alt + RMB: Dolly (zoom)
            else if ((mouse.buttons & Qt.RightButton) && hasAlt) {
                var dollyAmount = (dx + dy) * 0.5
                var factor = dollyAmount > 0 ? 1.02 : 0.98
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, cameraDistance * Math.pow(factor, Math.abs(dollyAmount))))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
            }
            // RMB without Alt: Context menu handled by Qt (no drag action)
        }

        function panCamera(dx, dy) {
            var panSpeed = cameraDistance * 0.002
            var yawRad = cameraYaw * Math.PI / 180
            var pitchRad = cameraPitch * Math.PI / 180

            // Calculate right and up vectors based on camera orientation
            var rightX = Math.cos(yawRad)
            var rightZ = -Math.sin(yawRad)

            // For top-down view, up is forward (Z axis)
            // For perspective view, up is Y axis mixed with forward
            var upX, upY, upZ
            if (Math.abs(cameraPitch) > 80) {
                // Near top-down: up maps to Z movement
                upX = Math.sin(yawRad) * Math.sign(cameraPitch)
                upY = 0
                upZ = Math.cos(yawRad) * Math.sign(cameraPitch)
            } else {
                // Perspective: up maps to Y with some forward
                upX = 0
                upY = 1
                upZ = 0
            }

            cameraTarget.x -= dx * rightX * panSpeed
            cameraTarget.z -= dx * rightZ * panSpeed
            cameraTarget.x -= dy * upX * panSpeed
            cameraTarget.y += dy * upY * panSpeed
            cameraTarget.z -= dy * upZ * panSpeed

            camera.position = camera.updateCameraPosition()
            // connectionCanvas.requestPaint() - disabled
        }

        onWheel: function(wheel) {
            var factor = wheel.angleDelta.y > 0 ? 0.9 : 1.1
            cameraDistance = Math.max(minDistance, Math.min(maxDistance, cameraDistance * factor))
            camera.position = camera.updateCameraPosition()
            // connectionCanvas.requestPaint() - disabled
        }

        function pickZoneAt(screenX, screenY) {
            var zones = zoneModel ? zoneModel.getZones() : []
            var closestZone = null
            var closestDist = 40

            for (var i = 0; i < zones.length; i++) {
                var zone = zones[i]
                var screenPos = connectionCanvas.project3DTo2D(zone.x, zone.y, zone.z)
                if (!screenPos) continue

                var dist = Math.sqrt(
                    Math.pow(screenX - screenPos.x, 2) +
                    Math.pow(screenY - screenPos.y, 2)
                )
                if (dist < closestDist) {
                    closestDist = dist
                    closestZone = zone
                }
            }

            if (closestZone && zoneModel) {
                zoneModel.selectZone(closestZone.id)
            }
        }
    }

    // Keyboard handling
    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_F) {
            focusSelected()
            event.accepted = true
        } else if (event.key === Qt.Key_A) {
            fitAll()
            event.accepted = true
        } else if (event.key === Qt.Key_W) {
            toggleWireframe()
            event.accepted = true
        } else if (event.key === Qt.Key_T) {
            toggleGhost()
            event.accepted = true
        }
    }

    // Render mode indicator
    Row {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.margins: 8
        spacing: 6

        Rectangle {
            width: modeText.width + 12
            height: 20
            radius: 3
            color: "#2A2A2A"

            Text {
                id: modeText
                anchors.centerIn: parent
                text: wireframeMode ? "WIRE" : "SOLID"
                color: wireframeMode ? "#88AAFF" : "#888888"
                font.pixelSize: 10
                font.bold: true
            }
        }

        Rectangle {
            visible: ghostMode
            width: ghostText.width + 12
            height: 20
            radius: 3
            color: "#2A2A2A"

            Text {
                id: ghostText
                anchors.centerIn: parent
                text: "X-RAY"
                color: "#88FF88"
                font.pixelSize: 10
                font.bold: true
            }
        }
    }

    // Platform-specific modifier key name
    property string modKey: Qt.platform.os === "osx" ? "⌥" : "Alt"

    // Help text
    Text {
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 8
        text: "W: Wireframe | T: Ghost | " + modKey + "+LMB: Tumble | " + modKey + "+MMB: Track | Scroll: Dolly | A: Top"
        color: "#555555"
        font.pixelSize: 10
    }

    // Model update listener
    Connections {
        target: zoneModel
        function onZonesChanged() {
            zoneRepeater.model = zoneModel.getZones()
        }
    }

    // API functions callable from Python
    function resetView() {
        cameraYaw = 0
        cameraPitch = -25
        cameraDistance = 150
        cameraTarget = Qt.vector3d(0, 0, 0)
        camera.position = camera.updateCameraPosition()
        // connectionCanvas.requestPaint() - disabled
    }

    function fitAll() {
        var zones = zoneModel ? zoneModel.getZones() : []
        if (zones.length === 0) {
            resetView()
            return
        }

        var minX = Infinity, maxX = -Infinity
        var minY = Infinity, maxY = -Infinity
        var minZ = Infinity, maxZ = -Infinity

        for (var i = 0; i < zones.length; i++) {
            var z = zones[i]
            var r = z.radius
            minX = Math.min(minX, z.x - r)
            maxX = Math.max(maxX, z.x + r)
            minY = Math.min(minY, z.y - r)
            maxY = Math.max(maxY, z.y + r)
            minZ = Math.min(minZ, z.z - r)
            maxZ = Math.max(maxZ, z.z + r)
        }

        var centerX = (minX + maxX) / 2
        var centerY = (minY + maxY) / 2
        var centerZ = (minZ + maxZ) / 2

        // For top-down view, extent is max of X and Z spans
        var extentX = maxX - minX
        var extentZ = maxZ - minZ
        var extent = Math.max(extentX, extentZ)

        // Calculate distance needed to see the whole scene (rough FOV calculation)
        // For 45 degree FOV, distance ~= extent / tan(22.5) ~= extent * 2.4
        var neededDistance = extent * 1.2

        cameraTarget = Qt.vector3d(centerX, centerY, centerZ)
        cameraDistance = Math.max(minDistance, Math.min(maxDistance, neededDistance))
        cameraYaw = 0
        cameraPitch = -90  // Snap to top-down view
        camera.position = camera.updateCameraPosition()
        // connectionCanvas.requestPaint() - disabled

        console.log("fitAll: center=(" + centerX + "," + centerY + "," + centerZ + ") extent=" + extent + " dist=" + cameraDistance)
    }

    function toggleWireframe() {
        wireframeMode = !wireframeMode
    }

    function toggleGhost() {
        ghostMode = !ghostMode
    }

    function focusSelected() {
        var zones = zoneModel ? zoneModel.getZones() : []
        for (var i = 0; i < zones.length; i++) {
            if (zones[i].selected) {
                var z = zones[i]
                cameraTarget = Qt.vector3d(z.x, z.y, z.z)
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, z.radius * 5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
                return
            }
        }
    }

    function focusOnZone(zoneId) {
        var zones = zoneModel ? zoneModel.getZones() : []
        for (var i = 0; i < zones.length; i++) {
            if (zones[i].id === zoneId) {
                var z = zones[i]
                cameraTarget = Qt.vector3d(z.x, z.y, z.z)
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, z.radius * 5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
                return
            }
        }
    }

    // Auto-fit on first load
    Component.onCompleted: {
        Qt.callLater(fitAll)
    }
}
'''
        with open(qml_path, 'w') as f:
            f.write(qml_content)

    def _on_zone_selected(self, zone_id: str, zone_data: dict):
        """Handle zone selection from QML."""
        # Update zone dropdown
        self.zone_selector.blockSignals(True)
        index = self.zone_selector.findData(zone_id)
        if index >= 0:
            self.zone_selector.setCurrentIndex(index)
        self.zone_selector.blockSignals(False)

        # Update status
        zone_name = zone_data.get('name', zone_id)
        self.status_label.setText(f"Selected: {zone_name} | F: Focus | A: Frame All")

        # Emit to parent (for Inspector)
        self.zoneSelected.emit(zone_id, zone_data)

    def set_project_manager(self, project_manager):
        """Set project manager reference and populate stage list."""
        self.project_manager = project_manager
        self._populate_stages()

    def _populate_stages(self):
        """Populate stage selector from project."""
        self.stage_selector.blockSignals(True)
        self.stage_selector.clear()
        self.zone_selector.blockSignals(True)
        self.zone_selector.clear()

        if not self.project_manager or not self.project_manager.is_project_open():
            self.stage_selector.addItem("(No project open)")
            self.stage_selector.blockSignals(False)
            self.zone_selector.blockSignals(False)
            return

        stages = self.project_manager.list_stages()
        for stage_name in stages:
            self.stage_selector.addItem(stage_name)

        if stages:
            self.current_stage = stages[0]
            self._load_stage(stages[0])

        self.stage_selector.blockSignals(False)
        self.zone_selector.blockSignals(False)

    def _populate_zone_selector(self):
        """Populate zone dropdown from loaded zones."""
        self.zone_selector.blockSignals(True)
        self.zone_selector.clear()

        self.zone_selector.addItem("(Select zone)", "")

        for zone_id, zone in sorted(self.zone_model.zones.items(), key=lambda x: x[1].name):
            self.zone_selector.addItem(zone.name, zone_id)

        self.zone_selector.blockSignals(False)

    def _on_stage_changed(self, stage_name: str):
        """Handle stage selection change."""
        if stage_name and stage_name != "(No project open)":
            self.current_stage = stage_name
            self._load_stage(stage_name)

    def _on_zone_dropdown_changed(self, zone_name: str):
        """Handle zone dropdown selection - selects but does NOT auto-focus."""
        zone_id = self.zone_selector.currentData()
        if zone_id:
            self.zone_model.selectZone(zone_id)
            # Don't auto-focus - user must press F to focus

    def _load_stage(self, stage_name: str):
        """Load zones from selected stage."""
        if not self.project_manager:
            return

        stage_path = self.project_manager.get_stage_path(stage_name)
        if not stage_path:
            self.status_label.setText(f"Stage not found: {stage_name}")
            return

        self.zone_model.load_stage(stage_path)
        self._populate_zone_selector()

        zone_count = len(self.zone_model.zones)
        conn_count = len(self.zone_model._connections)
        self.status_label.setText(f"{zone_count} zones, {conn_count} connections | F: Focus | A: Frame All")

        # Auto-fit view on load
        QTimer.singleShot(100, self._fit_all)

    def _reset_view(self):
        """Reset camera to default position."""
        root = self.quick_widget.rootObject()
        if root:
            root.resetView()

    def _fit_all(self):
        """Fit all zones in view (top-down)."""
        root = self.quick_widget.rootObject()
        if root:
            root.fitAll()

    def _focus_selected(self):
        """Focus camera on selected zone."""
        root = self.quick_widget.rootObject()
        if root:
            root.focusSelected()

    def _toggle_wireframe(self):
        """Toggle between wireframe and solid rendering."""
        root = self.quick_widget.rootObject()
        if root:
            root.toggleWireframe()

    def _toggle_ghost(self):
        """Toggle ghost mode (transparent, no occlusion)."""
        root = self.quick_widget.rootObject()
        if root:
            root.toggleGhost()

    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2a2a2a;
                border: 1px solid #404040;
                padding: 4px;
            }
            QMenu::item {
                color: #d2d2d2;
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #404040;
            }
            QMenu::separator {
                height: 1px;
                background: #404040;
                margin: 4px 8px;
            }
        """)

        # View actions
        menu.addAction("Fit All (A)", self._fit_all)
        menu.addAction("Focus Selected (F)", self._focus_selected)
        menu.addAction("Reset View", self._reset_view)

        menu.addSeparator()

        # Zone actions
        menu.addAction("Add Zone...", self._add_zone)

        selected = self.zone_model.selected_zone
        if selected:
            zone = self.zone_model.zones.get(selected)
            if zone:
                menu.addAction(f"Rename '{zone.name}'...", self._rename_zone)
                menu.addAction(f"Edit '{zone.name}'...", self._edit_zone)
                menu.addSeparator()
                menu.addAction(f"Delete '{zone.name}'", self._delete_zone)

        menu.exec(self.quick_widget.mapToGlobal(pos))

    def _add_zone(self):
        """Add a new zone to the stage."""
        if not self.project_manager or not self.current_stage:
            QMessageBox.warning(self, "No Stage", "Please open a project and stage first.")
            return

        name, ok = QInputDialog.getText(self, "Add Zone", "Zone name:")
        if not ok or not name:
            return

        # Generate zone ID from name
        zone_id = name.lower().replace(' ', '_').replace('-', '_')

        # Check for duplicates
        if zone_id in self.zone_model.zones:
            QMessageBox.warning(self, "Duplicate", f"Zone '{zone_id}' already exists.")
            return

        # Create zone file
        stage_path = self.project_manager.get_stage_path(self.current_stage)
        zones_dir = os.path.join(stage_path, "Zones")
        os.makedirs(zones_dir, exist_ok=True)

        zone_path = os.path.join(zones_dir, f"{zone_id}.zone.yaml")

        zone_data = {
            'name': name,
            'id': zone_id,
            'spatial': {
                'center': [0, 0, 0],
                'radius': 15.0,
                'falloff': 5.0,
                'shape': 'sphere'
            },
            'text': {
                'description': f'A new zone called {name}.',
                'features': [],
                'exits': {}
            },
            'perception': {
                'visibility': 20.0,
                'audibility': 20.0,
                'lighting': 'natural'
            },
            'ambient': {
                'sounds': [],
                'mood': 'neutral',
                'temperature': 'pleasant'
            }
        }

        with open(zone_path, 'w') as f:
            yaml.dump(zone_data, f, default_flow_style=False, sort_keys=False)

        # Reload stage
        self._load_stage(self.current_stage)

        # Select the new zone
        self.zone_model.selectZone(zone_id)

    def _rename_zone(self):
        """Rename the selected zone."""
        selected = self.zone_model.selected_zone
        if not selected:
            return

        zone = self.zone_model.zones.get(selected)
        if not zone:
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Zone", "New name:", text=zone.name
        )
        if not ok or not new_name or new_name == zone.name:
            return

        # Update zone file
        try:
            with open(zone.file_path, 'r') as f:
                zone_data = yaml.safe_load(f)

            zone_data['name'] = new_name

            with open(zone.file_path, 'w') as f:
                yaml.dump(zone_data, f, default_flow_style=False, sort_keys=False)

            # Reload
            self._load_stage(self.current_stage)
            self.zone_model.selectZone(selected)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename zone: {e}")

    def _edit_zone(self):
        """Open zone for editing in Inspector."""
        selected = self.zone_model.selected_zone
        if selected:
            zone_data = self.zone_model.getZoneData(selected)
            self.zoneSelected.emit(selected, zone_data)

    def _delete_zone(self):
        """Delete the selected zone."""
        selected = self.zone_model.selected_zone
        if not selected:
            return

        zone = self.zone_model.zones.get(selected)
        if not zone:
            return

        reply = QMessageBox.question(
            self,
            "Delete Zone",
            f"Delete zone '{zone.name}'?\n\nThis will delete the zone file permanently.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            os.remove(zone.file_path)
            self._load_stage(self.current_stage)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete zone: {e}")

    def refresh(self):
        """Refresh the view from current stage."""
        if self.current_stage:
            self._load_stage(self.current_stage)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_F:
            self._focus_selected()
            event.accept()
        elif event.key() == Qt.Key.Key_A:
            self._fit_all()
            event.accept()
        elif event.key() == Qt.Key.Key_W:
            self._toggle_wireframe()
            event.accept()
        elif event.key() == Qt.Key.Key_T:
            self._toggle_ghost()
            event.accept()
        else:
            super().keyPressEvent(event)
