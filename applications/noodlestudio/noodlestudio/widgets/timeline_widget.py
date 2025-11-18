"""
Multi-Track Timeline Widget - The Krugerrand Profiler

Logic Pro / Audacity style timeline with:
- Collapsible per-Noodling tracks
- 5-D affect vector visualization
- Event markers (clickable, hoverable)
- Zoom/pan controls
- Playhead scrubbing

Worth an ounce of gold!

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsItem, QLabel, QPushButton,
                             QSlider, QScrollArea)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QFont
from typing import Dict, List, Optional
import sys
sys.path.append('../..')
from noodlestudio.data.session_loader import SessionData, TimelineEvent


# Color palette (professional, no woo!)
COLORS = {
    'valence': QColor(102, 187, 106),    # Green
    'arousal': QColor(255, 167, 38),     # Orange
    'fear': QColor(239, 83, 80),         # Red
    'sorrow': QColor(156, 39, 176),      # Purple
    'boredom': QColor(153, 153, 153),    # Gray
    'surprise': QColor(100, 181, 246),   # Blue
    'spike': QColor(255, 235, 59),       # Yellow
    'playhead': QColor(255, 255, 255),   # White
    'background': QColor(10, 14, 26),    # Dark blue
    'track_bg': QColor(15, 20, 25),      # Slightly lighter
    'border': QColor(42, 63, 95),        # Blue-gray
}


class EventMarker(QGraphicsItem):
    """
    Clickable/hoverable event marker node.

    Color-coded:
    - Blue = Speech
    - Gray = Thought
    - Purple = Movement (enter/exit)
    - Green = Expression (FACS/body)
    """

    def __init__(self, event: TimelineEvent, x: float, y: float, parent=None):
        super().__init__(parent)
        self.event = event
        self.x_pos = x
        self.y_pos = y
        self.radius = 6  # Bigger than Swift's 4px!
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.hovered = False

    def boundingRect(self) -> QRectF:
        r = self.radius + 2
        return QRectF(self.x_pos - r, self.y_pos - r, r * 2, r * 2)

    def paint(self, painter: QPainter, option, widget=None):
        # Determine color based on event type
        if self.event.did_speak:
            color = COLORS['surprise']  # Speech = blue
        elif self.event.utterance and not self.event.did_speak:
            color = QColor(150, 150, 150)  # Thought = gray
        elif self.event.event_type in ['enter', 'exit']:
            color = QColor(186, 104, 200)  # Movement = purple
        elif self.event.facs_codes:
            color = COLORS['valence']  # Expression = green
        else:
            color = QColor(100, 100, 100)  # Other = dim gray

        # Brighter if hovered or selected
        if self.hovered or self.isSelected():
            color = color.lighter(150)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(self.x_pos, self.y_pos), self.radius, self.radius)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.scene().eventClicked.emit(self.event)


class AffectTrack(QGraphicsItem):
    """
    Single affect dimension track (e.g., Valence over time).

    Renders as a line graph with proper scaling.
    """

    def __init__(self, title: str, color: QColor, events: List[TimelineEvent],
                 value_accessor, max_time: float, width: float, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.events = events
        self.value_accessor = value_accessor
        self.max_time = max(max_time, 0.1)
        self.width = width
        self.height = 50  # Track height

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.boundingRect(), COLORS['track_bg'])

        # Draw border
        painter.setPen(QPen(self.color.darker(150), 1))
        painter.drawRect(self.boundingRect())

        # Draw label
        painter.setPen(QPen(self.color, 1))
        font = QFont("Monaco", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(5, 15, self.title)

        # Draw graph line
        if len(self.events) < 2:
            return

        path = QPainterPath()
        first = True

        for event in self.events:
            value = self.value_accessor(event)
            # Scale: -1 to 1 → 0 to height
            x = (event.timestamp / self.max_time) * self.width
            y = self.height - ((value + 1.0) / 2.0) * self.height

            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(self.color, 2))
        painter.drawPath(path)

        # Highlight spikes for surprise track
        if self.title == "SURPRISE":
            for event in self.events:
                if event.surprise > 0.3:
                    x = (event.timestamp / self.max_time) * self.width
                    y = self.height - ((event.surprise + 1.0) / 2.0) * self.height
                    painter.setPen(QPen(COLORS['spike'], 2))
                    painter.setBrush(QBrush(COLORS['spike']))
                    painter.drawEllipse(QPointF(x, y), 5, 5)


class TimelineScene(QGraphicsScene):
    """Custom scene that emits event click signals."""
    eventClicked = pyqtSignal(object)  # TimelineEvent


class MultiTrackTimeline(QWidget):
    """
    The Krugerrand Profiler - Multi-track timeline widget.

    Audacity/Logic Pro style with collapsible Noodling tracks.
    """

    eventSelected = pyqtSignal(object)  # TimelineEvent selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_data: Optional[SessionData] = None
        self.playhead_time = 0.0
        self.zoom_level = 1.0
        self.expanded_tracks = set()  # Which Noodling tracks are expanded

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scrub controller (top bar)
        scrub_widget = self.create_scrub_controller()
        layout.addWidget(scrub_widget)

        # Timeline view (main area)
        self.scene = TimelineScene()
        self.scene.eventClicked.connect(self.on_event_clicked)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("background-color: rgb(10, 14, 26); border: none;")
        layout.addWidget(self.view)

    def create_scrub_controller(self) -> QWidget:
        """Create playhead scrub controller (Logic Pro style)."""
        widget = QWidget()
        widget.setStyleSheet("background-color: rgb(19, 24, 36); border-bottom: 1px solid rgb(42, 63, 95);")
        widget.setFixedHeight(60)

        layout = QHBoxLayout(widget)

        # Timecode display
        self.timecode_label = QLabel("00:00.0 / 00:00")
        self.timecode_label.setFont(QFont("Monaco", 18, QFont.Weight.Bold))
        self.timecode_label.setStyleSheet("color: rgb(100, 181, 246);")
        layout.addWidget(self.timecode_label)

        layout.addStretch()

        # Playhead slider
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)
        self.playhead_slider.setMinimum(0)
        self.playhead_slider.setMaximum(1000)
        self.playhead_slider.setValue(0)
        self.playhead_slider.valueChanged.connect(self.on_playhead_moved)
        self.playhead_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgb(42, 63, 95);
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: rgb(100, 181, 246);
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.playhead_slider, stretch=2)

        layout.addStretch()

        # Zoom controls
        zoom_in = QPushButton("+")
        zoom_in.clicked.connect(lambda: self.set_zoom(self.zoom_level * 1.5))
        zoom_in.setFixedSize(30, 30)

        zoom_out = QPushButton("-")
        zoom_out.clicked.connect(lambda: self.set_zoom(self.zoom_level / 1.5))
        zoom_out.setFixedSize(30, 30)

        layout.addWidget(zoom_out)
        layout.addWidget(zoom_in)

        return widget

    def load_session(self, session_data: SessionData):
        """Load session data and render all Noodling tracks."""
        self.session_data = session_data
        self.scene.clear()

        if not session_data or not session_data.timelines:
            return

        # Calculate max time across all Noodlings
        max_time = 0.0
        for events in session_data.timelines.values():
            if events:
                max_time = max(max_time, events[-1].timestamp)

        # Render each Noodling's track
        y_offset = 0
        track_width = 1200  # Will be adjusted by zoom

        for noodling_id in session_data.noodlings:
            events = session_data.timelines.get(noodling_id, [])
            if not events:
                continue

            # Track header (collapsible)
            header_height = 40
            # TODO: Add actual header widget

            y_offset += header_height

            # If expanded, show 5-D tracks + event markers
            if noodling_id in self.expanded_tracks or True:  # Start all expanded
                # 5-D Affect tracks
                tracks = [
                    ("VALENCE", COLORS['valence'], lambda e: e.valence),
                    ("AROUSAL", COLORS['arousal'], lambda e: e.arousal),
                    ("FEAR", COLORS['fear'], lambda e: e.fear),
                    ("SORROW", COLORS['sorrow'], lambda e: e.sorrow),
                    ("SURPRISE", COLORS['surprise'], lambda e: e.surprise),
                ]

                for title, color, accessor in tracks:
                    track = AffectTrack(title, color, events, accessor, max_time, track_width)
                    track.setPos(0, y_offset)
                    self.scene.addItem(track)
                    y_offset += track.height + 2

                # Event traffic timeline
                event_track_height = 40
                for event in events:
                    x = (event.timestamp / max_time) * track_width
                    y = y_offset + event_track_height / 2

                    marker = EventMarker(event, x, y)
                    self.scene.addItem(marker)

                y_offset += event_track_height + 20

        self.scene.setSceneRect(0, 0, track_width, y_offset)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Update playhead slider
        self.max_time = max_time
        self.update_timecode()

    def on_playhead_moved(self, value: int):
        """Handle playhead slider movement."""
        if not self.session_data:
            return

        # Convert slider value (0-1000) to time
        self.playhead_time = (value / 1000.0) * self.max_time
        self.update_timecode()
        self.update()

    def update_timecode(self):
        """Update timecode display."""
        minutes = int(self.playhead_time) // 60
        seconds = int(self.playhead_time) % 60
        tenths = int((self.playhead_time % 1.0) * 10)

        max_min = int(self.max_time) // 60
        max_sec = int(self.max_time) % 60

        self.timecode_label.setText(
            f"{minutes:02d}:{seconds:02d}.{tenths} / {max_min:02d}:{max_sec:02d}"
        )

    def set_zoom(self, new_zoom: float):
        """Set zoom level (1.0 = 100%)."""
        self.zoom_level = max(0.1, min(10.0, new_zoom))
        self.view.resetTransform()
        self.view.scale(self.zoom_level, 1.0)

    def on_event_clicked(self, event: TimelineEvent):
        """Handle event marker click."""
        self.eventSelected.emit(event)
        # Jump playhead to clicked event
        self.playhead_time = event.timestamp
        slider_value = int((event.timestamp / self.max_time) * 1000)
        self.playhead_slider.setValue(slider_value)
