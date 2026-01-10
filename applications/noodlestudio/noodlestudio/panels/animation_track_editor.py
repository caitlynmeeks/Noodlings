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
#   Animation Track Editor - Curve editor for affect and pose tracks
#
#   A Maya Graph Editor-style panel for editing keyframed ani...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.animation_track_editor
# PURPOSE:  Animation Track Editor
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   KeyframeItem, TangentHandleItem, CurveScene, CurveView, ChannelListWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QScrollArea, QFrame, QSlider, QCheckBox,
    QTreeWidget, QTreeWidgetItem, QMenu, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem,
    QToolBar, QSpinBox, QDoubleSpinBox, QComboBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QLineF,
    pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QFont, QPainter, QColor, QPen, QBrush,
    QPainterPath, QKeySequence, QAction
)

import sys
sys.path.append('../..')


# =============================================================================
# Channel Colors (consistent with rest of NoodleStudio)
# =============================================================================

AFFECT_COLORS = {
    'valence': QColor('#E91E63'),     # Pink
    'arousal': QColor('#FF5722'),     # Orange
    'dominance': QColor('#9C27B0'),   # Purple
    'boredom': QColor('#607D8B'),     # Blue-gray
    'sorrow': QColor('#3F51B5'),      # Indigo
}

MUSCLE_COLORS = {
    # Body
    'Spine': QColor('#4CAF50'),       # Green
    'Chest': QColor('#8BC34A'),       # Light green
    'UpperChest': QColor('#CDDC39'),  # Lime
    'Neck': QColor('#FFEB3B'),        # Yellow
    'Head': QColor('#FFC107'),        # Amber

    # Eyes/Jaw
    'LeftEye': QColor('#00BCD4'),     # Cyan
    'RightEye': QColor('#00ACC1'),    # Darker cyan
    'Jaw': QColor('#FF9800'),         # Orange

    # Arms
    'LeftShoulder': QColor('#2196F3'),   # Blue
    'LeftArm': QColor('#1976D2'),        # Darker blue
    'LeftForeArm': QColor('#1565C0'),    # Even darker
    'LeftHand': QColor('#0D47A1'),       # Navy

    'RightShoulder': QColor('#9C27B0'),  # Purple
    'RightArm': QColor('#7B1FA2'),       # Darker purple
    'RightForeArm': QColor('#6A1B9A'),   # Even darker
    'RightHand': QColor('#4A148C'),      # Deep purple

    # Legs
    'LeftUpperLeg': QColor('#009688'),   # Teal
    'LeftLowerLeg': QColor('#00796B'),   # Darker teal
    'LeftFoot': QColor('#00695C'),       # Even darker
    'LeftToes': QColor('#004D40'),       # Deep teal

    'RightUpperLeg': QColor('#795548'),  # Brown
    'RightLowerLeg': QColor('#6D4C41'),  # Darker brown
    'RightFoot': QColor('#5D4037'),      # Even darker
    'RightToes': QColor('#4E342E'),      # Deep brown
}

BLENDSHAPE_COLORS = {
    'happy': QColor('#FFEB3B'),
    'angry': QColor('#F44336'),
    'sad': QColor('#3F51B5'),
    'relaxed': QColor('#4CAF50'),
    'surprised': QColor('#FF9800'),
    'blink': QColor('#9E9E9E'),
}


def get_channel_color(channel_name: str) -> QColor:
    """Get color for a channel by name."""
    # Check affect colors
    if channel_name in AFFECT_COLORS:
        return AFFECT_COLORS[channel_name]

    # Check muscle colors by prefix
    for prefix, color in MUSCLE_COLORS.items():
        if channel_name.startswith(prefix):
            return color

    # Check blend shapes
    for prefix, color in BLENDSHAPE_COLORS.items():
        if channel_name.startswith(prefix):
            return color

    # Default gray
    return QColor('#888888')


# =============================================================================
# Keyframe Graphics Items
# =============================================================================

class KeyframeItem(QGraphicsEllipseItem):
    """
    Draggable keyframe point on the curve.

    Can be selected and moved. Shows tangent handles when selected.
    """

    def __init__(self, channel_name: str, keyframe_index: int,
                 x: float, y: float, parent=None):
        size = 10
        super().__init__(-size/2, -size/2, size, size, parent)
        self.setPos(x, y)

        self.channel_name = channel_name
        self.keyframe_index = keyframe_index

        # Appearance
        color = get_channel_color(channel_name)
        self.setBrush(QBrush(color))
        self.setPen(QPen(color.darker(150), 1))

        # Selection
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Cursor
        self.setCursor(Qt.CursorShape.SizeAllCursor)

        # Tangent handles (created when selected)
        self.in_handle: Optional['TangentHandleItem'] = None
        self.out_handle: Optional['TangentHandleItem'] = None

    def itemChange(self, change, value):
        """Handle position changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if self.isSelected():
                self._show_tangent_handles()
            else:
                self._hide_tangent_handles()

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Notify editor of position change
            scene = self.scene()
            if scene and hasattr(scene, 'keyframeMoved'):
                scene.keyframeMoved.emit(self)

        return super().itemChange(change, value)

    def _show_tangent_handles(self):
        """Create and show tangent handles."""
        # TODO: Get actual tangent values from keyframe
        if self.in_handle is None:
            self.in_handle = TangentHandleItem(self, is_in=True)
            self.scene().addItem(self.in_handle)

        if self.out_handle is None:
            self.out_handle = TangentHandleItem(self, is_in=False)
            self.scene().addItem(self.out_handle)

    def _hide_tangent_handles(self):
        """Remove tangent handles."""
        if self.in_handle:
            self.scene().removeItem(self.in_handle)
            self.in_handle = None
        if self.out_handle:
            self.scene().removeItem(self.out_handle)
            self.out_handle = None


class TangentHandleItem(QGraphicsEllipseItem):
    """
    Tangent handle for bezier curve editing.

    Connected to parent keyframe with a line.
    """

    def __init__(self, keyframe: KeyframeItem, is_in: bool, parent=None):
        size = 6
        super().__init__(-size/2, -size/2, size, size, parent)

        self.keyframe = keyframe
        self.is_in = is_in

        # Position offset from keyframe
        offset = -30 if is_in else 30
        self.setPos(keyframe.pos().x() + offset, keyframe.pos().y())

        # Appearance
        self.setBrush(QBrush(QColor('#FFFFFF')))
        self.setPen(QPen(QColor('#888888'), 1))

        # Behavior
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Line to keyframe
        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(QColor('#666666'), 1, Qt.PenStyle.DashLine))
        self._update_line()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_line()
        return super().itemChange(change, value)

    def _update_line(self):
        """Update line connecting handle to keyframe."""
        if self.line.scene() is None and self.scene():
            self.scene().addItem(self.line)
        self.line.setLine(QLineF(self.keyframe.pos(), self.pos()))


# =============================================================================
# Curve Scene
# =============================================================================

class CurveScene(QGraphicsScene):
    """
    Scene for animation curves.

    Manages curve paths, keyframes, and grid.
    """

    keyframeMoved = pyqtSignal(object)      # KeyframeItem
    keyframeAdded = pyqtSignal(str, float, float)  # channel, time, value
    keyframeDeleted = pyqtSignal(str, int)  # channel, index

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene dimensions
        self.time_range = (0.0, 10.0)  # seconds
        self.value_range = (-1.0, 1.0)  # normalized

        # Pixels per unit
        self.time_scale = 100.0   # pixels per second
        self.value_scale = 100.0  # pixels per unit

        # Grid lines
        self.grid_lines: List[QGraphicsLineItem] = []

        # Curve paths per channel
        self.curve_paths: Dict[str, QGraphicsPathItem] = {}

        # Keyframe items per channel
        self.keyframe_items: Dict[str, List[KeyframeItem]] = {}

        # Visible channels
        self.visible_channels: Set[str] = set()

        # Playhead
        self.playhead_line: Optional[QGraphicsLineItem] = None
        self.playhead_time: float = 0.0

        # Setup
        self._create_grid()
        self._create_playhead()

    def _create_grid(self):
        """Create background grid."""
        # Clear old grid
        for line in self.grid_lines:
            self.removeItem(line)
        self.grid_lines.clear()

        # Calculate bounds
        x_min = self.time_range[0] * self.time_scale
        x_max = self.time_range[1] * self.time_scale
        y_min = -self.value_range[1] * self.value_scale  # Flip Y
        y_max = -self.value_range[0] * self.value_scale

        # Vertical lines (time)
        for t in range(int(self.time_range[0]), int(self.time_range[1]) + 1):
            x = t * self.time_scale
            line = QGraphicsLineItem(x, y_min, x, y_max)
            if t == 0:
                line.setPen(QPen(QColor('#444444'), 1))
            else:
                line.setPen(QPen(QColor('#2a2a2a'), 1))
            self.addItem(line)
            self.grid_lines.append(line)

        # Horizontal lines (value)
        for v in range(-10, 11):  # -1.0 to 1.0 in 0.1 steps
            y = -v * 0.1 * self.value_scale
            line = QGraphicsLineItem(x_min, y, x_max, y)
            if v == 0:
                line.setPen(QPen(QColor('#444444'), 1))
            elif v % 5 == 0:
                line.setPen(QPen(QColor('#333333'), 1))
            else:
                line.setPen(QPen(QColor('#222222'), 1))
            self.addItem(line)
            self.grid_lines.append(line)

        # Set scene rect
        self.setSceneRect(x_min - 50, y_min - 50, x_max - x_min + 100, y_max - y_min + 100)

    def _create_playhead(self):
        """Create playhead line."""
        y_min = -self.value_range[1] * self.value_scale - 50
        y_max = -self.value_range[0] * self.value_scale + 50

        self.playhead_line = QGraphicsLineItem(0, y_min, 0, y_max)
        self.playhead_line.setPen(QPen(QColor('#E91E63'), 2))
        self.playhead_line.setZValue(1000)  # Always on top
        self.addItem(self.playhead_line)

    def set_playhead(self, time: float):
        """Move playhead to time."""
        self.playhead_time = time
        x = time * self.time_scale
        if self.playhead_line:
            self.playhead_line.setLine(x, self.playhead_line.line().y1(),
                                       x, self.playhead_line.line().y2())

    def time_to_x(self, t: float) -> float:
        """Convert time to X coordinate."""
        return t * self.time_scale

    def value_to_y(self, v: float) -> float:
        """Convert value to Y coordinate (flipped)."""
        return -v * self.value_scale

    def x_to_time(self, x: float) -> float:
        """Convert X coordinate to time."""
        return x / self.time_scale

    def y_to_value(self, y: float) -> float:
        """Convert Y coordinate to value (flipped)."""
        return -y / self.value_scale

    def add_channel(self, channel_name: str, keyframes: List[Tuple[float, float]],
                    interpolation: str = 'linear'):
        """
        Add or update a channel with keyframes.

        Args:
            channel_name: Channel name
            keyframes: List of (time, value) tuples
            interpolation: 'linear', 'bezier', 'step'
        """
        self.visible_channels.add(channel_name)
        color = get_channel_color(channel_name)

        # Remove old curve path
        if channel_name in self.curve_paths:
            self.removeItem(self.curve_paths[channel_name])

        # Remove old keyframe items
        if channel_name in self.keyframe_items:
            for item in self.keyframe_items[channel_name]:
                if item.in_handle:
                    self.removeItem(item.in_handle)
                if item.out_handle:
                    self.removeItem(item.out_handle)
                self.removeItem(item)

        # Create curve path
        path = QPainterPath()
        if keyframes:
            sorted_kfs = sorted(keyframes, key=lambda k: k[0])

            # Start at first keyframe
            x0 = self.time_to_x(sorted_kfs[0][0])
            y0 = self.value_to_y(sorted_kfs[0][1])
            path.moveTo(x0, y0)

            # Draw segments
            for i in range(1, len(sorted_kfs)):
                x1 = self.time_to_x(sorted_kfs[i][0])
                y1 = self.value_to_y(sorted_kfs[i][1])

                if interpolation == 'step':
                    path.lineTo(x1, y0)  # Horizontal to x1
                    path.lineTo(x1, y1)  # Vertical to y1
                elif interpolation == 'bezier':
                    # Simple cubic bezier (could use actual tangents)
                    cx = (x0 + x1) / 2
                    path.cubicTo(cx, y0, cx, y1, x1, y1)
                else:  # linear
                    path.lineTo(x1, y1)

                x0, y0 = x1, y1

        # Add curve path item
        path_item = QGraphicsPathItem(path)
        path_item.setPen(QPen(color, 2))
        path_item.setZValue(10)
        self.addItem(path_item)
        self.curve_paths[channel_name] = path_item

        # Create keyframe items
        self.keyframe_items[channel_name] = []
        for i, (t, v) in enumerate(sorted(keyframes, key=lambda k: k[0])):
            x = self.time_to_x(t)
            y = self.value_to_y(v)
            kf_item = KeyframeItem(channel_name, i, x, y)
            kf_item.setZValue(100)
            self.addItem(kf_item)
            self.keyframe_items[channel_name].append(kf_item)

    def remove_channel(self, channel_name: str):
        """Remove a channel from display."""
        self.visible_channels.discard(channel_name)

        if channel_name in self.curve_paths:
            self.removeItem(self.curve_paths[channel_name])
            del self.curve_paths[channel_name]

        if channel_name in self.keyframe_items:
            for item in self.keyframe_items[channel_name]:
                if item.in_handle:
                    self.removeItem(item.in_handle)
                if item.out_handle:
                    self.removeItem(item.out_handle)
                self.removeItem(item)
            del self.keyframe_items[channel_name]

    def set_channel_visibility(self, channel_name: str, visible: bool):
        """Show/hide a channel."""
        if channel_name in self.curve_paths:
            self.curve_paths[channel_name].setVisible(visible)
        if channel_name in self.keyframe_items:
            for item in self.keyframe_items[channel_name]:
                item.setVisible(visible)

        if visible:
            self.visible_channels.add(channel_name)
        else:
            self.visible_channels.discard(channel_name)


# =============================================================================
# Curve View
# =============================================================================

class CurveView(QGraphicsView):
    """
    View for the curve scene with zoom and pan.
    """

    def __init__(self, scene: CurveScene, parent=None):
        super().__init__(scene, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setStyleSheet("background-color: #1a1a1a; border: none;")

        # Pan state
        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        """Handle pan start."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle panning."""
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle pan end."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Add keyframe on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            scene = self.scene()
            if isinstance(scene, CurveScene):
                t = scene.x_to_time(scene_pos.x())
                v = scene.y_to_value(scene_pos.y())
                # Signal to add keyframe (editor will handle channel selection)
                scene.keyframeAdded.emit("", t, v)
        super().mouseDoubleClickEvent(event)


# =============================================================================
# Channel List Widget
# =============================================================================

class ChannelListWidget(QTreeWidget):
    """
    Tree widget showing available channels with visibility checkboxes.
    """

    channelVisibilityChanged = pyqtSignal(str, bool)  # channel, visible

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setHeaderHidden(True)
        self.setIndentation(15)
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                border: none;
                color: #cccccc;
                font-family: Monaco;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 3px;
            }
            QTreeWidget::item:selected {
                background-color: #3a3a3a;
            }
        """)

        # Channel items by name
        self.channel_items: Dict[str, QTreeWidgetItem] = {}

    def add_category(self, name: str) -> QTreeWidgetItem:
        """Add a category (Affect, Muscles, BlendShapes)."""
        item = QTreeWidgetItem(self, [name])
        item.setFont(0, QFont("Monaco", 10, QFont.Weight.Bold))
        item.setExpanded(True)
        return item

    def add_channel(self, category: QTreeWidgetItem, channel_name: str,
                    visible: bool = True):
        """Add a channel under a category."""
        item = QTreeWidgetItem(category, [channel_name])

        # Color indicator
        color = get_channel_color(channel_name)
        item.setForeground(0, QBrush(color))

        # Checkbox for visibility
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)

        self.channel_items[channel_name] = item

    def itemChanged(self, item, column):
        """Handle visibility checkbox changes."""
        channel_name = item.text(0)
        if channel_name in self.channel_items:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.channelVisibilityChanged.emit(channel_name, visible)


# =============================================================================
# Timeline Ruler
# =============================================================================

class TimelineRuler(QWidget):
    """
    Horizontal ruler showing time markers.
    """

    timeClicked = pyqtSignal(float)  # time in seconds

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedHeight(24)
        self.setStyleSheet("background-color: #1a1a1a;")

        self.time_range = (0.0, 10.0)
        self.time_scale = 100.0
        self.offset = 0.0

        self.markers: List[Tuple[float, str]] = []  # (time, name)

    def set_view(self, view: CurveView):
        """Connect to view for scroll sync."""
        view.horizontalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_scroll(self, value):
        """Update offset when view scrolls."""
        self.offset = value
        self.update()

    def paintEvent(self, event):
        """Draw time markers."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor('#1a1a1a'))

        # Time labels
        painter.setPen(QColor('#888888'))
        painter.setFont(QFont("Monaco", 9))

        for t in range(int(self.time_range[0]), int(self.time_range[1]) + 1):
            x = t * self.time_scale - self.offset
            if 0 <= x <= self.width():
                painter.drawText(int(x) + 3, 16, f"{t}s")
                painter.drawLine(int(x), 18, int(x), 24)

        # Markers
        painter.setPen(QColor('#E91E63'))
        for marker_time, marker_name in self.markers:
            x = marker_time * self.time_scale - self.offset
            if 0 <= x <= self.width():
                painter.drawLine(int(x), 0, int(x), 24)

    def mousePressEvent(self, event):
        """Seek on click."""
        t = (event.position().x() + self.offset) / self.time_scale
        self.timeClicked.emit(t)


# =============================================================================
# Playback Controls
# =============================================================================

class PlaybackControls(QWidget):
    """
    Transport controls: play, pause, stop, seek.
    """

    playClicked = pyqtSignal()
    pauseClicked = pyqtSignal()
    stopClicked = pyqtSignal()
    timeChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)

        # Style for buttons
        btn_style = """
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                color: #ccc;
                padding: 4px 8px;
                font-family: Monaco;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(btn_style)
        self.stop_btn.clicked.connect(self.stopClicked)
        layout.addWidget(self.stop_btn)

        # Play/Pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet(btn_style)
        self.play_btn.clicked.connect(self._on_play_click)
        layout.addWidget(self.play_btn)

        self.is_playing = False

        # Time display
        self.time_label = QLabel("0.00s")
        self.time_label.setStyleSheet("color: #E91E63; font-family: Monaco; font-size: 12px;")
        self.time_label.setMinimumWidth(60)
        layout.addWidget(self.time_label)

        # Duration display
        layout.addWidget(QLabel("/"))
        self.duration_label = QLabel("10.00s")
        self.duration_label.setStyleSheet("color: #888; font-family: Monaco; font-size: 12px;")
        layout.addWidget(self.duration_label)

        layout.addStretch()

        # Speed control
        layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setFixedWidth(60)
        self.speed_spin.setStyleSheet("background-color: #2a2a2a; color: #ccc;")
        layout.addWidget(self.speed_spin)

        # Loop checkbox
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setStyleSheet("color: #888;")
        layout.addWidget(self.loop_check)

    def _on_play_click(self):
        if self.is_playing:
            self.pauseClicked.emit()
            self.play_btn.setText("Play")
        else:
            self.playClicked.emit()
            self.play_btn.setText("Pause")
        self.is_playing = not self.is_playing

    def set_time(self, t: float):
        """Update time display."""
        self.time_label.setText(f"{t:.2f}s")

    def set_duration(self, d: float):
        """Update duration display."""
        self.duration_label.setText(f"{d:.2f}s")

    def set_playing(self, playing: bool):
        """Update play/pause button state."""
        self.is_playing = playing
        self.play_btn.setText("Pause" if playing else "Play")


# =============================================================================
# Main Editor Panel
# =============================================================================

class AnimationTrackEditor(QWidget):
    """
    Main animation track editor panel.

    Maya Graph Editor-style curve editing for affect and pose tracks.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Animation Track Editor")

        # Current track data
        self.current_track = None
        self.current_track_type = None  # 'affect' or 'pose'
        self.current_file_path = None

        # Playback state
        self.is_playing = False
        self.current_time = 0.0
        self.duration = 10.0

        self._setup_ui()
        self._setup_playback_timer()

    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main splitter (channel list | curve editor)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Channel list (left)
        self.channel_list = ChannelListWidget()
        self.channel_list.setMinimumWidth(150)
        self.channel_list.setMaximumWidth(250)
        self.channel_list.channelVisibilityChanged.connect(self._on_channel_visibility)
        splitter.addWidget(self.channel_list)

        # Curve editor (right)
        curve_container = QWidget()
        curve_layout = QVBoxLayout(curve_container)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_layout.setSpacing(0)

        # Timeline ruler
        self.ruler = TimelineRuler()
        self.ruler.timeClicked.connect(self._on_ruler_click)
        curve_layout.addWidget(self.ruler)

        # Curve scene and view
        self.curve_scene = CurveScene()
        self.curve_scene.keyframeMoved.connect(self._on_keyframe_moved)
        self.curve_scene.keyframeAdded.connect(self._on_keyframe_added)

        self.curve_view = CurveView(self.curve_scene)
        self.ruler.set_view(self.curve_view)
        curve_layout.addWidget(self.curve_view, 1)

        splitter.addWidget(curve_container)
        splitter.setSizes([180, 600])

        layout.addWidget(splitter, 1)

        # Playback controls (bottom)
        self.playback = PlaybackControls()
        self.playback.playClicked.connect(self._on_play)
        self.playback.pauseClicked.connect(self._on_pause)
        self.playback.stopClicked.connect(self._on_stop)
        layout.addWidget(self.playback)

    def _create_toolbar(self) -> QToolBar:
        """Create toolbar with file and edit actions."""
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333;
                spacing: 4px;
                padding: 2px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                color: #ccc;
                padding: 4px 8px;
                font-family: Monaco;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #3a3a3a;
            }
        """)

        # File actions
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._on_open)
        toolbar.addAction(open_action)

        import_fbx_action = QAction("Import FBX", self)
        import_fbx_action.triggered.connect(self._on_import_fbx)
        toolbar.addAction(import_fbx_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._on_save)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Edit actions
        add_kf_action = QAction("Add Keyframe", self)
        add_kf_action.triggered.connect(self._on_add_keyframe)
        toolbar.addAction(add_kf_action)

        del_kf_action = QAction("Delete Keyframe", self)
        del_kf_action.triggered.connect(self._on_delete_keyframe)
        toolbar.addAction(del_kf_action)

        toolbar.addSeparator()

        # Interpolation
        toolbar.addWidget(QLabel("Interp:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Linear", "Bezier", "Step", "Hermite"])
        self.interp_combo.setStyleSheet("background-color: #2a2a2a; color: #ccc;")
        self.interp_combo.currentTextChanged.connect(self._on_interp_changed)
        toolbar.addWidget(self.interp_combo)

        return toolbar

    def _setup_playback_timer(self):
        """Create timer for playback updates."""
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._on_playback_tick)
        self.playback_timer.setInterval(16)  # ~60fps

        self._last_tick_time = 0.0

    # -------------------------------------------------------------------------
    # Track Loading
    # -------------------------------------------------------------------------

    def load_affect_track(self, file_path: str):
        """Load an affect track file."""
        try:
            from noodlestudio.core.affect_track import AffectTrack

            track = AffectTrack.load_yaml(file_path)
            self.current_track = track
            self.current_track_type = 'affect'
            self.current_file_path = file_path
            self.duration = track.duration

            self._populate_from_affect_track(track)
            self.playback.set_duration(track.duration)

            # Set markers
            self.ruler.markers = [(m.time, m.name) for m in track.markers]
            self.ruler.update()

            print(f"[AnimTrackEditor] Loaded affect track: {track.name}")

        except Exception as e:
            print(f"[AnimTrackEditor] Failed to load affect track: {e}")

    def load_pose_track(self, file_path: str):
        """Load a pose track file."""
        try:
            from noodlestudio.core.pose_track import PoseTrack

            track = PoseTrack.load_yaml(file_path)
            self.current_track = track
            self.current_track_type = 'pose'
            self.current_file_path = file_path
            self.duration = track.duration

            self._populate_from_pose_track(track)
            self.playback.set_duration(track.duration)

            # Set markers
            self.ruler.markers = [(m.time, m.name) for m in track.markers]
            self.ruler.update()

            print(f"[AnimTrackEditor] Loaded pose track: {track.name}")

        except Exception as e:
            print(f"[AnimTrackEditor] Failed to load pose track: {e}")

    def _populate_from_affect_track(self, track):
        """Populate UI from affect track data."""
        self.channel_list.clear()
        self.curve_scene.curve_paths.clear()
        self.curve_scene.keyframe_items.clear()

        # Update scene time range
        self.curve_scene.time_range = (0.0, max(10.0, track.duration + 1))
        self.curve_scene._create_grid()

        # Add affect category
        affect_cat = self.channel_list.add_category("Affect")

        for channel_name, channel in track.channels.items():
            # Add to channel list
            self.channel_list.add_channel(affect_cat, channel_name)

            # Add to scene
            keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
            self.curve_scene.add_channel(
                channel_name, keyframes, channel.interpolation.value
            )

    def _populate_from_pose_track(self, track):
        """Populate UI from pose track data."""
        self.channel_list.clear()
        self.curve_scene.curve_paths.clear()
        self.curve_scene.keyframe_items.clear()

        # Update scene time range
        self.curve_scene.time_range = (0.0, max(10.0, track.duration + 1))
        self.curve_scene._create_grid()

        # Add muscles category
        if track.muscles:
            muscles_cat = self.channel_list.add_category("Muscles")
            for channel_name, channel in track.muscles.items():
                self.channel_list.add_channel(muscles_cat, channel_name)
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(
                    channel_name, keyframes, channel.interpolation.value
                )

        # Add blend shapes category
        if track.blendshapes:
            bs_cat = self.channel_list.add_category("BlendShapes")
            for channel_name, channel in track.blendshapes.items():
                self.channel_list.add_channel(bs_cat, channel_name)
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(channel_name, keyframes, 'linear')

    # -------------------------------------------------------------------------
    # Slots
    # -------------------------------------------------------------------------

    def _on_channel_visibility(self, channel: str, visible: bool):
        """Handle channel visibility toggle."""
        self.curve_scene.set_channel_visibility(channel, visible)

    def _on_ruler_click(self, time: float):
        """Handle timeline ruler click."""
        self.current_time = max(0, min(time, self.duration))
        self.curve_scene.set_playhead(self.current_time)
        self.playback.set_time(self.current_time)

    def _on_keyframe_moved(self, kf_item: KeyframeItem):
        """Handle keyframe moved in scene."""
        # Convert position back to time/value
        new_time = self.curve_scene.x_to_time(kf_item.pos().x())
        new_value = self.curve_scene.y_to_value(kf_item.pos().y())

        # Update track data
        if self.current_track and self.current_track_type == 'affect':
            channel = self.current_track.channels.get(kf_item.channel_name)
            if channel and kf_item.keyframe_index < len(channel.keyframes):
                channel.keyframes[kf_item.keyframe_index].time = new_time
                channel.keyframes[kf_item.keyframe_index].value = new_value
                channel.keyframes.sort(key=lambda k: k.time)

        # Redraw curve
        self._refresh_channel(kf_item.channel_name)

    def _on_keyframe_added(self, channel: str, time: float, value: float):
        """Handle keyframe added via double-click."""
        # If no channel specified, use first visible
        if not channel and self.curve_scene.visible_channels:
            channel = list(self.curve_scene.visible_channels)[0]

        if not channel:
            return

        # Clamp value
        value = max(-1.0, min(1.0, value))

        # Add to track
        if self.current_track:
            if self.current_track_type == 'affect':
                self.current_track.add_keyframe(channel, time, value)
            elif self.current_track_type == 'pose':
                self.current_track.add_muscle_keyframe(channel, time, value)

        self._refresh_channel(channel)

    def _refresh_channel(self, channel_name: str):
        """Refresh a channel's curve display."""
        if not self.current_track:
            return

        if self.current_track_type == 'affect':
            channel = self.current_track.channels.get(channel_name)
            if channel:
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(
                    channel_name, keyframes, channel.interpolation.value
                )

        elif self.current_track_type == 'pose':
            channel = self.current_track.muscles.get(channel_name)
            if channel:
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(
                    channel_name, keyframes, channel.interpolation.value
                )

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _on_play(self):
        """Start playback."""
        self.is_playing = True
        self._last_tick_time = time.time()
        self.playback_timer.start()
        self.playback.set_playing(True)

    def _on_pause(self):
        """Pause playback."""
        self.is_playing = False
        self.playback_timer.stop()
        self.playback.set_playing(False)

    def _on_stop(self):
        """Stop and reset playback."""
        self.is_playing = False
        self.playback_timer.stop()
        self.current_time = 0.0
        self.curve_scene.set_playhead(0.0)
        self.playback.set_time(0.0)
        self.playback.set_playing(False)

    def _on_playback_tick(self):
        """Update playback on timer tick."""
        now = time.time()
        delta = (now - self._last_tick_time) * self.playback.speed_spin.value()
        self._last_tick_time = now

        self.current_time += delta

        # Check for end
        if self.current_time >= self.duration:
            if self.playback.loop_check.isChecked():
                self.current_time = 0.0
            else:
                self.current_time = self.duration
                self._on_pause()

        self.curve_scene.set_playhead(self.current_time)
        self.playback.set_time(self.current_time)

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    def _on_open(self):
        """Open file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Animation Track",
            "",
            "Animation Tracks (*.affecttrack *.posetrack *.noodletrack);;All Files (*)"
        )

        if file_path:
            if file_path.endswith('.affecttrack'):
                self.load_affect_track(file_path)
            elif file_path.endswith('.posetrack'):
                self.load_pose_track(file_path)

    def _on_save(self):
        """Save current track."""
        if not self.current_track:
            return

        if not self.current_file_path:
            self._on_save_as()
            return

        try:
            self.current_track.save_yaml(self.current_file_path)
            print(f"[AnimTrackEditor] Saved: {self.current_file_path}")
        except Exception as e:
            print(f"[AnimTrackEditor] Save failed: {e}")

    def _on_save_as(self):
        """Save As dialog."""
        if not self.current_track:
            return

        ext = '.affecttrack' if self.current_track_type == 'affect' else '.posetrack'
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animation Track",
            "",
            f"Animation Track (*{ext});;All Files (*)"
        )

        if file_path:
            if not file_path.endswith(ext):
                file_path += ext
            self.current_file_path = file_path
            self._on_save()

    def _on_import_fbx(self):
        """Import FBX animation file (e.g., from Mixamo).

        Opens file dialog, imports bone animations, retargets to muscle space,
        and creates a paired empty affect track for PAD+BS annotation.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import FBX Animation",
            "",
            "FBX Files (*.fbx);;All Files (*)"
        )

        if not file_path:
            return

        try:
            from noodlestudio.core.fbx_importer import import_fbx_with_affect_layer

            print(f"[AnimTrackEditor] Importing FBX: {file_path}")

            # Import with paired affect track for annotation
            pose_track, affect_track = import_fbx_with_affect_layer(
                file_path,
                target_fps=30.0,
                include_root_motion=True
            )

            if pose_track is None:
                print("[AnimTrackEditor] FBX import failed - no animation data extracted")
                return

            # Store both tracks
            self.pose_track = pose_track
            self.affect_track = affect_track
            self.current_track = pose_track  # Primary track for editing
            self.current_track_type = 'pose'
            self.current_file_path = None  # Not saved yet

            # Use duration from pose track
            self.duration = pose_track.duration
            self.playback.set_duration(self.duration)

            # Populate the UI with both tracks
            self._populate_from_dual_tracks(pose_track, affect_track)

            # Set markers from affect track (user will add annotation markers)
            self.ruler.markers = [(m.time, m.name) for m in affect_track.markers] if affect_track.markers else []
            self.ruler.update()

            print(f"[AnimTrackEditor] Imported: {pose_track.name}")
            print(f"  Duration: {pose_track.duration:.2f}s")
            print(f"  Muscle channels: {len(pose_track.muscles)}")
            print(f"  Affect channels ready for annotation: {len(affect_track.channels)}")

        except ImportError as e:
            print(f"[AnimTrackEditor] FBX import requires fbx_importer module: {e}")
        except Exception as e:
            import traceback
            print(f"[AnimTrackEditor] FBX import failed: {e}")
            traceback.print_exc()

    def _populate_from_dual_tracks(self, pose_track, affect_track):
        """Populate UI from both pose and affect tracks (FBX import mode).

        Shows muscle channels from pose track and affect channels for annotation.
        """
        self.channel_list.clear()
        self.curve_scene.curve_paths.clear()
        self.curve_scene.keyframe_items.clear()

        # Update scene time range
        max_duration = max(pose_track.duration, affect_track.duration if affect_track else 0)
        self.curve_scene.time_range = (0.0, max(10.0, max_duration + 1))
        self.curve_scene._create_grid()

        # Add Affect category (for PAD+BS annotation - initially mostly empty)
        affect_cat = self.channel_list.add_category("Affect (Annotation)")
        for channel_name, channel in affect_track.channels.items():
            self.channel_list.add_channel(affect_cat, channel_name)
            keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
            self.curve_scene.add_channel(
                channel_name, keyframes, channel.interpolation.value
            )

        # Add Muscles category (imported from FBX)
        if pose_track.muscles:
            muscles_cat = self.channel_list.add_category("Muscles (Imported)")

            # Group by body part for cleaner display
            body_groups = {
                'Spine': [], 'Chest': [], 'Neck': [], 'Head': [],
                'LeftArm': [], 'RightArm': [],
                'LeftLeg': [], 'RightLeg': [],
                'Other': []
            }

            for channel_name in pose_track.muscles.keys():
                grouped = False
                for prefix in ['Spine', 'Chest', 'UpperChest', 'Neck', 'Head',
                               'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                               'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                               'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToes',
                               'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToes']:
                    if channel_name.startswith(prefix):
                        if 'Left' in prefix and 'Arm' in prefix or prefix in ['LeftShoulder', 'LeftHand', 'LeftForeArm']:
                            body_groups['LeftArm'].append(channel_name)
                        elif 'Right' in prefix and 'Arm' in prefix or prefix in ['RightShoulder', 'RightHand', 'RightForeArm']:
                            body_groups['RightArm'].append(channel_name)
                        elif 'Left' in prefix and 'Leg' in prefix or prefix in ['LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToes']:
                            body_groups['LeftLeg'].append(channel_name)
                        elif 'Right' in prefix and 'Leg' in prefix or prefix in ['RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToes']:
                            body_groups['RightLeg'].append(channel_name)
                        elif prefix in ['Spine', 'Chest', 'UpperChest']:
                            body_groups['Spine'].append(channel_name)
                        elif prefix in ['Neck']:
                            body_groups['Neck'].append(channel_name)
                        elif prefix in ['Head']:
                            body_groups['Head'].append(channel_name)
                        else:
                            body_groups['Other'].append(channel_name)
                        grouped = True
                        break
                if not grouped:
                    body_groups['Other'].append(channel_name)

            # Add channels under muscles category
            for channel_name, channel in pose_track.muscles.items():
                self.channel_list.add_channel(muscles_cat, channel_name)
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(
                    channel_name, keyframes, channel.interpolation.value
                )

        # Add BlendShapes if present
        if pose_track.blendshapes:
            bs_cat = self.channel_list.add_category("BlendShapes (Imported)")
            for channel_name, channel in pose_track.blendshapes.items():
                self.channel_list.add_channel(bs_cat, channel_name)
                keyframes = [(kf.time, kf.value) for kf in channel.keyframes]
                self.curve_scene.add_channel(channel_name, keyframes, 'linear')

    def save_affect_annotations(self, file_path: Optional[str] = None):
        """Save the affect annotation track separately.

        After annotating PAD+BS values on an imported animation,
        save just the affect track.
        """
        if not hasattr(self, 'affect_track') or self.affect_track is None:
            print("[AnimTrackEditor] No affect track to save")
            return

        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Affect Annotations",
                "",
                "Affect Track (*.affecttrack);;All Files (*)"
            )

        if file_path:
            if not file_path.endswith('.affecttrack'):
                file_path += '.affecttrack'
            try:
                self.affect_track.save_yaml(file_path)
                print(f"[AnimTrackEditor] Saved affect annotations: {file_path}")
            except Exception as e:
                print(f"[AnimTrackEditor] Failed to save affect annotations: {e}")

    def save_pose_track(self, file_path: Optional[str] = None):
        """Save the pose track (retargeted muscle animation)."""
        if not hasattr(self, 'pose_track') or self.pose_track is None:
            print("[AnimTrackEditor] No pose track to save")
            return

        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Pose Track",
                "",
                "Pose Track (*.posetrack);;All Files (*)"
            )

        if file_path:
            if not file_path.endswith('.posetrack'):
                file_path += '.posetrack'
            try:
                self.pose_track.save_yaml(file_path)
                print(f"[AnimTrackEditor] Saved pose track: {file_path}")
            except Exception as e:
                print(f"[AnimTrackEditor] Failed to save pose track: {e}")

    # -------------------------------------------------------------------------
    # Edit Operations
    # -------------------------------------------------------------------------

    def _on_add_keyframe(self):
        """Add keyframe at current time."""
        if self.curve_scene.visible_channels:
            channel = list(self.curve_scene.visible_channels)[0]
            self._on_keyframe_added(channel, self.current_time, 0.0)

    def _on_delete_keyframe(self):
        """Delete selected keyframes."""
        for item in self.curve_scene.selectedItems():
            if isinstance(item, KeyframeItem):
                channel = item.channel_name
                idx = item.keyframe_index

                # Remove from track
                if self.current_track and self.current_track_type == 'affect':
                    ch = self.current_track.channels.get(channel)
                    if ch and idx < len(ch.keyframes):
                        del ch.keyframes[idx]
                        self._refresh_channel(channel)

    def _on_interp_changed(self, interp: str):
        """Change interpolation for selected channel."""
        # Get selected channel from channel list
        items = self.channel_list.selectedItems()
        if not items:
            return

        channel_name = items[0].text(0)
        interp_type = interp.lower()

        if self.current_track and self.current_track_type == 'affect':
            from noodlestudio.core.affect_track import InterpolationType
            ch = self.current_track.channels.get(channel_name)
            if ch:
                ch.interpolation = InterpolationType(interp_type)
                self._refresh_channel(channel_name)


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Dark theme
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #cccccc;
        }
    """)

    editor = AnimationTrackEditor()
    editor.resize(1000, 600)
    editor.show()

    # Create test data
    from noodlestudio.core.affect_track import AffectTrack, InterpolationType

    track = AffectTrack(name="Test Track", duration=5.0)
    track.add_keyframe('valence', 0.0, 0.0)
    track.add_keyframe('valence', 1.0, 0.8)
    track.add_keyframe('valence', 3.0, -0.4)
    track.add_keyframe('valence', 5.0, 0.2)

    track.add_keyframe('arousal', 0.0, 0.3)
    track.add_keyframe('arousal', 2.0, 0.9)
    track.add_keyframe('arousal', 5.0, 0.4)

    track.channels['valence'].interpolation = InterpolationType.BEZIER

    # Save and load
    track.save_yaml("/tmp/test_editor.affecttrack")
    editor.load_affect_track("/tmp/test_editor.affecttrack")

    sys.exit(app.exec())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
