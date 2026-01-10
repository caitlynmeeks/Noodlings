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
#   Profiler Panel - The Cognitive Timeline Editor
#
#   Logic Pro-style timeline with: - Affect waveform visualiz...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.profiler_panel
# PURPOSE:  Profiler Panel - The Cognitive Timeline Editor
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetTimelineScene, FacetTimelineView, ProfilerPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QSplitter, QScrollArea,
                             QGraphicsView, QGraphicsScene, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush
import sys
sys.path.append('../..')

# Original affect-based timeline
from noodlestudio.data.session_loader import SessionLoader, TimelineEvent
from noodlestudio.widgets.timeline_widget import MultiTrackTimeline

# New facet-based timeline
from noodlestudio.core.timeline_recorder import (
    get_timeline_recorder, TimelineRecorder,
    FacetRecord, CycleRecord, AffectSample
)
from noodlestudio.widgets.facet_track import (
    FacetSwimlanesWidget, FacetTrack, CycleTrack, FACET_COLORS
)


class FacetTimelineScene(QGraphicsScene):
    """Scene for facet timeline with click signals."""
    facetClicked = pyqtSignal(object)  # FacetRecord


class FacetTimelineView(QWidget):
    """
    Facet swimlanes visualization widget.

    Shows facet executions as colored blocks arranged by facet type.
    """

    facetSelected = pyqtSignal(object)  # FacetRecord

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cycles = []
        self.max_time = 1.0
        self.zoom_level = 1.0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)

        title = QLabel("FACET EXECUTION")
        title.setFont(QFont("Monaco", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #9C27B0;")  # Purple for facets
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Stats label
        self.stats_label = QLabel("0 cycles | 0 facets")
        self.stats_label.setFont(QFont("Monaco", 9))
        self.stats_label.setStyleSheet("color: #666;")
        header_layout.addWidget(self.stats_label)

        layout.addWidget(header)

        # Graphics scene/view for facet tracks
        self.scene = FacetTimelineScene()
        self.scene.facetClicked.connect(self._on_facet_clicked)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("background-color: #0f0f0f; border: none;")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.view)

    def load_cycles(self, cycles: list):
        """Load cycle data and render facet tracks."""
        self.cycles = cycles
        self.scene.clear()

        if not cycles:
            return

        # Calculate max time
        self.max_time = 0.1
        for cycle in cycles:
            if cycle.end_time > self.max_time:
                self.max_time = cycle.end_time

        # Create swimlanes widget
        swimlanes = FacetSwimlanesWidget(
            cycles=cycles,
            max_time=self.max_time,
            width=1200 * self.zoom_level,
            parent=None
        )
        self.scene.addItem(swimlanes)

        # Update scene rect
        self.scene.setSceneRect(swimlanes.boundingRect())

        # Update stats
        total_facets = sum(len(c.facets) for c in cycles)
        self.stats_label.setText(f"{len(cycles)} cycles | {total_facets} facets")

    def add_cycle(self, cycle: CycleRecord):
        """Add a single cycle (for live updates)."""
        self.cycles.append(cycle)
        self.load_cycles(self.cycles)

    def _on_facet_clicked(self, facet: FacetRecord):
        """Handle facet click."""
        self.facetSelected.emit(facet)

    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self.zoom_level = max(0.1, min(10.0, zoom))
        self.load_cycles(self.cycles)


class ProfilerPanel(QWidget):
    """
    Cognitive Timeline Editor - Main profiler panel.

    Layout:
    ┌──────────────────────────────────────────────────────────┐
    │ SCRUB CONTROLLER (00:00.0/00:45)  [REC] [CLEAR]         │
    ├──────────────────────────────────────────────────────────┤
    │ FACET EXECUTION                                          │
    │   CYCLES   ████████░░░░░░░████████░░░░░░░████████       │
    │   INCOMING        █░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░░      │
    │   CharmNetwork    ░█░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░      │
    │   ContextIntel    ░░████░░░░░░░░░░░░░░░████░░░░░░░      │
    ├──────────────────────────────────────────────────────────┤
    │ AFFECT WAVEFORMS                                         │
    │   Valence   ──────▁▂▃▄▅▆▇█▇▆▅▄▃▂▁──────────────────    │
    │   Arousal   ████▇▆▅▄▃▂▁──────▁▂▃▄▅▆▇████░░░░░░░░░░░    │
    ├──────────────────────────────────────────────────────────┤
    │ INSPECTOR (click facet or event to view)                 │
    │   RoastEngine @ Cycle 3, T=00:22.450                     │
    │   Duration: 1.234s | Tokens: 847                         │
    │   INPUTS: {...}                                          │
    │   OUTPUTS: {...}                                         │
    └──────────────────────────────────────────────────────────┘
    """

    # Signals for external connection
    facetSelected = pyqtSignal(object)  # FacetRecord
    recordingToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data loaders
        self.loader = SessionLoader()  # Original affect API
        self.recorder = None  # Timeline recorder (initialized on first use)

        # State
        self.is_recording = False

        # Initialize UI
        self.init_ui()

        # Connect to timeline recorder
        self._connect_recorder()

        # Auto-refresh timer for affect API (legacy)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_live_session)
        self.refresh_timer.start(2000)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top control bar
        control_bar = self._create_control_bar()
        layout.addWidget(control_bar)

        # Main splitter (facets + affect + inspector)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. Facet execution timeline (NEW)
        self.facet_timeline = FacetTimelineView()
        self.facet_timeline.facetSelected.connect(self.on_facet_selected)
        splitter.addWidget(self.facet_timeline)

        # 2. Affect waveforms timeline (original)
        self.affect_timeline = MultiTrackTimeline()
        self.affect_timeline.eventSelected.connect(self.on_event_selected)
        splitter.addWidget(self.affect_timeline)

        # 3. Inspector console (bottom)
        self.inspector = self._create_inspector_panel()
        splitter.addWidget(self.inspector)

        splitter.setSizes([250, 300, 200])  # Facets, Affect, Inspector
        layout.addWidget(splitter)

    def _create_control_bar(self) -> QWidget:
        """Create top control bar with recording controls."""
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 4, 12, 4)

        # Title
        title = QLabel("COGNITIVE TIMELINE")
        title.setFont(QFont("Monaco", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #64B5F6;")
        layout.addWidget(title)

        layout.addStretch()

        # Recording indicator
        self.rec_label = QLabel("LIVE")
        self.rec_label.setFont(QFont("Monaco", 9, QFont.Weight.Bold))
        self.rec_label.setStyleSheet("color: #4CAF50; padding: 2px 8px; background: #1a3a1a; border-radius: 4px;")
        layout.addWidget(self.rec_label)

        # Record button
        self.rec_button = QPushButton("REC")
        self.rec_button.setCheckable(True)
        self.rec_button.setChecked(True)
        self.rec_button.clicked.connect(self._on_rec_toggled)
        self.rec_button.setStyleSheet("""
            QPushButton { background: #333; color: #fff; padding: 4px 12px; border: 1px solid #555; border-radius: 4px; }
            QPushButton:checked { background: #B71C1C; color: #fff; border-color: #F44336; }
        """)
        layout.addWidget(self.rec_button)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        clear_btn.setStyleSheet("""
            QPushButton { background: #333; color: #fff; padding: 4px 12px; border: 1px solid #555; border-radius: 4px; }
            QPushButton:hover { background: #444; }
        """)
        layout.addWidget(clear_btn)

        return bar

    def _create_inspector_panel(self) -> QWidget:
        """Create event/facet inspector console."""
        widget = QWidget()
        widget.setStyleSheet("background-color: #131824; border-top: 2px solid #2A3F5F;")

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(28)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel("INSPECTOR")
        title.setFont(QFont("Monaco", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #64B5F6;")
        title_bar_layout.addWidget(title)

        title_bar_layout.addStretch()

        # Mode indicator
        self.inspector_mode = QLabel("(click facet or event)")
        self.inspector_mode.setFont(QFont("Monaco", 9))
        self.inspector_mode.setStyleSheet("color: #666;")
        title_bar_layout.addWidget(self.inspector_mode)

        layout.addWidget(title_bar)

        # Console text area
        self.inspector_text = QTextEdit()
        self.inspector_text.setReadOnly(True)
        self.inspector_text.setFont(QFont("Monaco", 10))
        self.inspector_text.setStyleSheet("""
            QTextEdit {
                background-color: #0A0E1A;
                color: #E0E0E0;
                border: 1px solid #2A3F5F;
                padding: 8px;
            }
        """)
        self.inspector_text.setPlainText("Click a facet block or event marker to inspect...")
        layout.addWidget(self.inspector_text)

        return widget

    def _connect_recorder(self):
        """Connect to global timeline recorder."""
        try:
            self.recorder = get_timeline_recorder()
            self.recorder.cycleRecorded.connect(self._on_cycle_recorded)
            self.recorder.facetCompleted.connect(self._on_facet_completed)

            # Try to start recording - will defer if no async loop yet
            self.recorder.start_recording()
            self.is_recording = self.recorder.is_recording

            if self.is_recording:
                print("[ProfilerPanel] Connected to TimelineRecorder (LIVE)")
            else:
                print("[ProfilerPanel] Connected to TimelineRecorder (will start when server runs)")

        except Exception as e:
            print(f"[ProfilerPanel] Could not connect to TimelineRecorder: {e}")
            self.recorder = None
            self.is_recording = False

    @pyqtSlot(object)
    def _on_cycle_recorded(self, cycle: CycleRecord):
        """Handle new cycle from recorder (live update)."""
        self.facet_timeline.add_cycle(cycle)

    @pyqtSlot(object)
    def _on_facet_completed(self, facet: FacetRecord):
        """Handle facet completion (could update live indicator)."""
        pass  # Future: real-time block drawing

    def _on_rec_toggled(self, checked: bool):
        """Handle record button toggle."""
        self.is_recording = checked

        if self.recorder:
            if checked:
                self.recorder.start_recording()
                self.rec_label.setText("LIVE")
                self.rec_label.setStyleSheet("color: #4CAF50; padding: 2px 8px; background: #1a3a1a; border-radius: 4px;")
            else:
                self.recorder.stop_recording()
                self.rec_label.setText("PAUSED")
                self.rec_label.setStyleSheet("color: #FF9800; padding: 2px 8px; background: #3a2a1a; border-radius: 4px;")

        self.recordingToggled.emit(checked)

    def _on_clear(self):
        """Clear recorded session."""
        if self.recorder:
            self.recorder.clear_session()

        self.facet_timeline.load_cycles([])
        self.inspector_text.setPlainText("Session cleared. Recording new data...")

    @pyqtSlot(object)
    def on_facet_selected(self, facet: FacetRecord):
        """Display facet details in inspector."""
        self.inspector_mode.setText("FACET")

        lines = []
        lines.append(f"{'='*50}")
        lines.append(f" FACET: {facet.facet_name}")
        lines.append(f"{'='*50}")
        lines.append("")
        lines.append(f"Type:     {facet.facet_type}")
        lines.append(f"Cycle:    {facet.cycle}")
        lines.append(f"Duration: {facet.duration_ms:.1f}ms")
        if facet.token_count > 0:
            lines.append(f"Tokens:   {facet.token_count}")
        lines.append(f"Start:    {facet.start_time:.3f}s")
        lines.append(f"End:      {facet.end_time:.3f}s")
        lines.append("")

        # Inputs
        if facet.inputs:
            lines.append("INPUTS:")
            for key, value in facet.inputs.items():
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                lines.append(f"  {key}: {val_str}")
            lines.append("")

        # Outputs
        if facet.outputs:
            lines.append("OUTPUTS:")
            for key, value in facet.outputs.items():
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                lines.append(f"  {key}: {val_str}")
            lines.append("")

        # Prompt (for LLM facets)
        if facet.prompt:
            lines.append("PROMPT:")
            lines.append(f"  {facet.prompt[:200]}...")

        self.inspector_text.setPlainText("\n".join(lines))
        self.facetSelected.emit(facet)

    @pyqtSlot(object)
    def on_event_selected(self, event: TimelineEvent):
        """Display event details in inspector (original affect events)."""
        self.inspector_mode.setText("AFFECT EVENT")

        lines = []
        lines.append(f"{'='*50}")
        lines.append(f" EVENT @ {event.timestamp:.2f}s")
        lines.append(f"{'='*50}")
        lines.append("")
        lines.append(f"Type: {event.event_type.upper()}")
        if event.responding_to:
            lines.append(f"Responding to: {event.responding_to}")
        lines.append("")

        # 5-D Affect
        lines.append("AFFECT VECTOR:")
        lines.append(f"  Valence: {event.valence:+.3f}")
        lines.append(f"  Arousal: {event.arousal: .3f}")
        lines.append(f"  Fear:    {event.fear: .3f}")
        lines.append(f"  Sorrow:  {event.sorrow: .3f}")
        lines.append(f"  Boredom: {event.boredom: .3f}")
        lines.append("")

        # FACS
        if event.facs_codes:
            lines.append("FACIAL EXPRESSION:")
            for code, desc in event.facs_codes[:4]:
                lines.append(f"  {code}: {desc}")
            lines.append("")

        # Speech
        if event.utterance:
            label = "SPEECH:" if event.did_speak else "THOUGHT:"
            lines.append(label)
            lines.append(f'  "{event.utterance}"')

        self.inspector_text.setPlainText("\n".join(lines))

    @pyqtSlot()
    def refresh_live_session(self):
        """Refresh affect timeline from live API."""
        # Original affect data from profiler API
        session_data = self.loader.load_live_session()
        if session_data:
            self.affect_timeline.load_session(session_data)

        # Update facet timeline from recorder
        if self.recorder:
            # Retry connection if not yet recording
            if not self.is_recording and self.rec_button.isChecked():
                self.recorder.start_recording()
                if self.recorder.is_recording:
                    self.is_recording = True
                    self.rec_label.setText("LIVE")
                    self.rec_label.setStyleSheet("color: #4CAF50; padding: 2px 8px; background: #1a3a1a; border-radius: 4px;")
                    print("[ProfilerPanel] TimelineRecorder connected (delayed)")

            if self.is_recording:
                cycles = self.recorder.get_all_cycles()
                if cycles:
                    self.facet_timeline.load_cycles(cycles)

    def load_session_file(self, filepath: str):
        """Load session from file."""
        from pathlib import Path
        session_data = self.loader.load_session_file(Path(filepath))
        if session_data:
            self.affect_timeline.load_session(session_data)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
