"""
Profiler Panel - The Krugerrand Profiler Main Panel

Complete Logic Pro-style timeline profiler with:
- Multi-track timeline
- Event inspector console
- Live updates from API
- Zoom/pan controls

Worth its weight in gold!

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QSplitter, QDockWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont
import sys
sys.path.append('../..')
from noodlestudio.data.session_loader import SessionLoader, TimelineEvent
from noodlestudio.widgets.timeline_widget import MultiTrackTimeline


class ProfilerPanel(QDockWidget):
    """
    Main profiler panel with timeline + inspector.

    Layout:
    ┌──────────────────────────────────┐
    │ SCRUB CONTROLLER (00:00.0/00:45) │
    ├──────────────────────────────────┤
    │                                  │
    │   MULTI-TRACK TIMELINE           │
    │   (Collapsible Noodling lanes)   │
    │                                  │
    ├──────────────────────────────────┤
    │ EVENT INSPECTOR (click to view)  │
    └──────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__("Timeline Profiler", parent)
        self.loader = SessionLoader()

        # Create central widget for dock
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

        # Auto-refresh timer for live updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_live_session)
        self.refresh_timer.start(2000)  # Refresh every 2 seconds

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for timeline + inspector
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Timeline widget (top)
        self.timeline = MultiTrackTimeline()
        self.timeline.eventSelected.connect(self.on_event_selected)
        splitter.addWidget(self.timeline)

        # Inspector console (bottom)
        self.inspector = self.create_inspector_panel()
        splitter.addWidget(self.inspector)

        splitter.setSizes([600, 250])  # Timeline gets more space
        layout.addWidget(splitter)

    def create_inspector_panel(self) -> QWidget:
        """Create event inspector console (bottom panel)."""
        widget = QWidget()
        widget.setStyleSheet("background-color: rgb(19, 24, 36); border-top: 2px solid rgb(42, 63, 95);")

        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("EVENT INSPECTOR")
        title.setFont(QFont("Monaco", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: rgb(100, 181, 246); padding: 8px;")
        layout.addWidget(title)

        # Console text area
        self.inspector_text = QTextEdit()
        self.inspector_text.setReadOnly(True)
        self.inspector_text.setFont(QFont("Monaco", 11))
        self.inspector_text.setStyleSheet("""
            QTextEdit {
                background-color: rgb(10, 14, 26);
                color: rgb(224, 224, 224);
                border: 1px solid rgb(42, 63, 95);
                padding: 12px;
            }
        """)
        self.inspector_text.setPlainText("Click an event marker to inspect...")
        layout.addWidget(self.inspector_text)

        return widget

    @pyqtSlot(object)
    def on_event_selected(self, event: TimelineEvent):
        """Display full event details in inspector."""
        # Build comprehensive event report
        lines = []
        lines.append(f"═══ EVENT @ {event.timestamp:.2f}s ═══")
        lines.append("")
        lines.append(f"TYPE: {event.event_type.upper()}")
        if event.responding_to:
            lines.append(f"RESPONDING TO: {event.responding_to.replace('agent_', '').replace('user_', '')}")
        lines.append("")

        # FACS codes
        if event.facs_codes:
            lines.append("FACIAL EXPRESSION (FACS):")
            for code, desc in event.facs_codes[:6]:
                lines.append(f"  {code}: {desc}")
            lines.append("")

        # Body codes
        if event.body_codes:
            lines.append("BODY LANGUAGE (LABAN):")
            for code, desc in event.body_codes[:6]:
                lines.append(f"  {code}: {desc}")
            lines.append("")

        # Expression description
        if event.expression_description:
            lines.append(f"DESCRIPTION: {event.expression_description}")
            lines.append("")

        # Speech/Thought
        if event.utterance:
            label = "SPEECH:" if event.did_speak else "PRIVATE THOUGHT:"
            lines.append(label)
            lines.append(f'  "{event.utterance}"')
            lines.append("")

        # 5-D Affect
        lines.append("5-D AFFECT VECTOR:")
        lines.append(f"  Valence: {event.valence:+.3f}  (negative ← → positive)")
        lines.append(f"  Arousal: {event.arousal: .3f}  (calm → excited)")
        lines.append(f"  Fear:    {event.fear: .3f}  (safe → anxious)")
        lines.append(f"  Sorrow:  {event.sorrow: .3f}  (content → sad)")
        lines.append(f"  Boredom: {event.boredom: .3f}  (engaged → bored)")
        lines.append("")

        # Surprise
        lines.append(f"SURPRISE: {event.surprise:.3f}" +
                    (" ⚡ SPIKE!" if event.surprise > 0.3 else ""))
        lines.append(f"Speech Threshold: {event.speech_threshold:.3f}")
        lines.append("")

        # HSI Metrics
        lines.append(f"HSI (Hierarchical Separation Index):")
        lines.append(f"  Slow/Fast:   {event.hsi_slow_fast:.3f}")
        lines.append(f"  Medium/Fast: {event.hsi_medium_fast:.3f}")
        lines.append("")

        # Trigger context
        if event.event_context:
            lines.append("TRIGGER CONTEXT:")
            lines.append(f"  {event.event_context}")

        self.inspector_text.setPlainText("\n".join(lines))

    @pyqtSlot()
    def refresh_live_session(self):
        """Refresh timeline from live API."""
        session_data = self.loader.load_live_session()
        if session_data:
            self.timeline.load_session(session_data)
