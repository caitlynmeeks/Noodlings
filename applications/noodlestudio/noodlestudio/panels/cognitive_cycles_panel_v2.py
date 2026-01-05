"""
Cognitive Cycles Panel v2 - Hierarchical assembly monitoring

Upgraded from "one cognitive thread per agent" to "multiple assemblies per Thing,
each independently reportable."

The panel is architecture-agnostic - it doesn't know or care what KIND of cognition
is happening (CHARM, custom RNNs, rule systems, quantum chaos). It just provides
a generic dashboard that assemblies can report to.

Key changes from v1:
- Hierarchical layout: Things contain Assemblies
- Collapsible Thing rows with aggregate activity
- Per-assembly status reporting (architecture-agnostic)
- Per-assembly and per-Thing pause/step controls

Author: Caitlyn + Claude
Date: January 2026
"""

import logging
import requests
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QToolButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QRectF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen

# Import local CognitionMonitor for in-process status
try:
    from ..core.cognition_monitor import get_cognition_monitor
    from ..core.cognition_monitor import CyclePhase as MonitorCyclePhase
    HAS_LOCAL_MONITOR = True
except ImportError:
    HAS_LOCAL_MONITOR = False

logger = logging.getLogger(__name__)


# Color scheme - Starbucks green for active cognition
COLORS = {
    'bg': '#383838',
    'row_bg': '#2d2d2d',
    'row_hover': '#3a3a3a',
    'thing_bg': '#252525',
    'thing_border': '#444444',
    'phase_idle': '#2a2a2a',
    'phase_active': '#00704A',       # Starbucks green
    'phase_active_pulse': '#00915E', # Brighter green for pulse
    'phase_complete': '#3a4a3a',     # Muted green for completed
    'text': '#D2D2D2',
    'text_muted': '#888888',
    'text_dim': '#666666',
    'text_active': '#FFFFFF',        # White text on green
    'border': '#555555',
    'button_bg': '#3e3e3e',
    'button_hover': '#4e4e4e',
    'pause_active': '#CC6666',
    'step_active': '#6666CC',
    'badge_bg': '#444444',
    'badge_text': '#AAAAAA',
}


class CyclePhase(Enum):
    """Cognitive cycle phases."""
    IDLE = 0
    INCOMING = 1   # Message received, starting cycle
    PRECOG = 2     # Pre-facet processing (context intelligence)
    FACET = 3      # Facet assembly execution (LLM calls)
    NEURAL = 4     # CharmNetwork / MLX
    POSTCOG = 5    # Post-facet processing (convergence)
    OUTGOING = 6   # Final emission


PHASE_LABELS = ['INCOMING', 'PRECOG', 'FACET', 'NEURAL', 'POSTCOG', 'OUTGOING']


@dataclass
class AssemblyStatus:
    """
    Status report from a running assembly.

    This is architecture-agnostic - the platform doesn't interpret the data,
    just displays it. Assemblies publish their own status strings.
    """
    assembly_id: str
    thing_id: str

    # Phase (required)
    phase: CyclePhase = CyclePhase.IDLE

    # Current facet being executed (optional)
    current_facet: str = ""

    # Free-form status string (optional, assembly-defined)
    # Examples:
    #   "valence: 0.7, arousal: 0.4"  (VKP assembly)
    #   "entropy: 0.83"                (quantum assembly)
    #   "processing: alice.greeting"   (social assembly)
    #   "awaiting LLM response..."     (any LLM assembly)
    status_text: str = ""

    # Activity level 0.0-1.0 (optional, for sparklines)
    activity: float = 0.0

    # Is this assembly paused?
    is_paused: bool = False

    # Custom data blob (optional, for assembly-specific inspectors)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThingStatus:
    """Aggregate status for a Thing with multiple assemblies."""
    thing_id: str
    thing_name: str
    assemblies: Dict[str, AssemblyStatus] = field(default_factory=dict)
    is_collapsed: bool = True  # Compact mode by default

    @property
    def assembly_count(self) -> int:
        return len(self.assemblies)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self.assemblies.values() if a.phase != CyclePhase.IDLE)

    @property
    def aggregate_activity(self) -> float:
        if not self.assemblies:
            return 0.0
        return sum(a.activity for a in self.assemblies.values()) / len(self.assemblies)


class CyclePhaseIndicator(QWidget):
    """
    Custom widget showing 6 cognitive phases as segmented bar.

    Segments: INCOMING | PRECOG | FACET | NEURAL | POSTCOG | OUTGOING
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_phase = CyclePhase.IDLE
        self.pulse_state = False
        self.setMinimumSize(200, 20)
        self.setMaximumHeight(20)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Pulse animation timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._pulse_timer.start(300)

    def _toggle_pulse(self):
        if self.current_phase != CyclePhase.IDLE:
            self.pulse_state = not self.pulse_state
            self.update()

    def set_phase(self, phase: CyclePhase):
        if phase != self.current_phase:
            self.current_phase = phase
            self.pulse_state = True
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        segment_width = width / 6
        gap = 2

        font = QFont("Arial", 5)
        painter.setFont(font)

        for i, label in enumerate(PHASE_LABELS):
            x = i * segment_width
            rect = QRectF(x + gap/2, 1, segment_width - gap, height - 2)

            phase_value = i + 1

            if self.current_phase == CyclePhase.IDLE:
                bg_color = QColor(COLORS['phase_idle'])
                text_color = QColor(COLORS['text_dim'])
            elif phase_value < self.current_phase.value:
                bg_color = QColor(COLORS['phase_complete'])
                text_color = QColor(COLORS['text_muted'])
            elif phase_value == self.current_phase.value:
                if self.pulse_state:
                    bg_color = QColor(COLORS['phase_active_pulse'])
                else:
                    bg_color = QColor(COLORS['phase_active'])
                text_color = QColor(COLORS['text_active'])
            else:
                bg_color = QColor(COLORS['phase_idle'])
                text_color = QColor(COLORS['text_dim'])

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(rect, 2, 2)

            painter.setPen(text_color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()


class AssemblyCycleRow(QFrame):
    """
    Row for a single assembly within a Thing.

    Layout: [Indent][Name] [Phase Indicator] [Status Text] [Pause] [Step]
    """

    pauseClicked = pyqtSignal(str, str, bool)  # thing_id, assembly_id, should_pause
    stepClicked = pyqtSignal(str, str)  # thing_id, assembly_id

    def __init__(self, thing_id: str, assembly_id: str, assembly_name: str, parent=None):
        super().__init__(parent)
        self.thing_id = thing_id
        self.assembly_id = assembly_id
        self.assembly_name = assembly_name
        self.is_paused = False

        self.setStyleSheet(f"""
            AssemblyCycleRow {{
                background: {COLORS['row_bg']};
                border: none;
                border-left: 2px solid {COLORS['border']};
                margin-left: 16px;
            }}
        """)

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        # Assembly name (indented, smaller)
        self.name_label = QLabel(self.assembly_name)
        self.name_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
        self.name_label.setMinimumWidth(100)
        self.name_label.setMaximumWidth(120)
        layout.addWidget(self.name_label)

        # Phase indicator (smaller)
        self.phase_indicator = CyclePhaseIndicator()
        layout.addWidget(self.phase_indicator)

        # Status text (assembly-defined)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic; font-size: 10px;")
        self.status_label.setMinimumWidth(100)
        layout.addWidget(self.status_label, 1)

        # Pause button
        self.pause_btn = QPushButton("||")
        self.pause_btn.setFixedSize(22, 18)
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Pause this assembly")
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
                font-weight: bold;
                font-size: 8px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:checked {{ background: {COLORS['pause_active']}; color: white; }}
        """)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.pause_btn)

        # Step button
        self.step_btn = QPushButton(">|")
        self.step_btn.setFixedSize(22, 18)
        self.step_btn.setToolTip("Step this assembly")
        self.step_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
                font-weight: bold;
                font-size: 8px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:pressed {{ background: {COLORS['step_active']}; }}
        """)
        self.step_btn.clicked.connect(self._on_step_clicked)
        layout.addWidget(self.step_btn)

    def _on_pause_clicked(self):
        self.is_paused = self.pause_btn.isChecked()
        self.pauseClicked.emit(self.thing_id, self.assembly_id, self.is_paused)
        self.pause_btn.setText(">" if self.is_paused else "||")

    def _on_step_clicked(self):
        self.stepClicked.emit(self.thing_id, self.assembly_id)

    def update_status(self, status: AssemblyStatus):
        """Update row display from AssemblyStatus."""
        self.phase_indicator.set_phase(status.phase)

        if status.phase == CyclePhase.IDLE:
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic; font-size: 10px;")
        else:
            # Show assembly-defined status_text, or fall back to facet name
            text = status.status_text or status.current_facet or f"{status.phase.name}..."
            self.status_label.setText(text)
            self.status_label.setStyleSheet(f"color: {COLORS['text']}; font-style: normal; font-size: 10px;")

        # Update pause state
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(status.is_paused)
        self.pause_btn.setText(">" if status.is_paused else "||")
        self.is_paused = status.is_paused
        self.pause_btn.blockSignals(False)


class ThingCycleRow(QFrame):
    """
    Collapsible row for a Thing with multiple assemblies.

    Layout: [Collapse Toggle] [Name] [Badge] [Aggregate Activity] [Pause All] [Step All]
    """

    pauseAllClicked = pyqtSignal(str, bool)  # thing_id, should_pause
    stepAllClicked = pyqtSignal(str)  # thing_id

    def __init__(self, thing_id: str, thing_name: str, parent=None):
        super().__init__(parent)
        self.thing_id = thing_id
        self.thing_name = thing_name
        self.is_collapsed = True
        self.assembly_rows: Dict[str, AssemblyCycleRow] = {}

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            ThingCycleRow {{
                background: {COLORS['thing_bg']};
                border: 1px solid {COLORS['thing_border']};
                border-radius: 3px;
            }}
            ThingCycleRow:hover {{
                border: 1px solid #666666;
            }}
        """)

        self._setup_ui()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header row
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        # Collapse toggle
        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("▷")
        self.collapse_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                color: {COLORS['text_muted']};
                border: none;
                font-size: 10px;
            }}
            QToolButton:hover {{ color: {COLORS['text']}; }}
        """)
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self.collapse_btn)

        # Thing name
        self.name_label = QLabel(self.thing_name)
        self.name_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        self.name_label.setMinimumWidth(100)
        self.name_label.setMaximumWidth(150)
        header_layout.addWidget(self.name_label)

        # Assembly count badge
        self.badge_label = QLabel("0")
        self.badge_label.setStyleSheet(f"""
            QLabel {{
                background: {COLORS['badge_bg']};
                color: {COLORS['badge_text']};
                border-radius: 8px;
                padding: 2px 6px;
                font-size: 9px;
            }}
        """)
        header_layout.addWidget(self.badge_label)

        # Spacer
        header_layout.addStretch()

        # Activity indicator (text-based for now)
        self.activity_label = QLabel("Idle")
        self.activity_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        header_layout.addWidget(self.activity_label)

        # Pause All button
        self.pause_all_btn = QPushButton("||")
        self.pause_all_btn.setFixedSize(28, 22)
        self.pause_all_btn.setCheckable(True)
        self.pause_all_btn.setToolTip("Pause all assemblies for this Thing")
        self.pause_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:checked {{ background: {COLORS['pause_active']}; color: white; }}
        """)
        self.pause_all_btn.clicked.connect(self._on_pause_all)
        header_layout.addWidget(self.pause_all_btn)

        # Step All button
        self.step_all_btn = QPushButton(">|")
        self.step_all_btn.setFixedSize(28, 22)
        self.step_all_btn.setToolTip("Step all assemblies")
        self.step_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:pressed {{ background: {COLORS['step_active']}; }}
        """)
        self.step_all_btn.clicked.connect(self._on_step_all)
        header_layout.addWidget(self.step_all_btn)

        self.main_layout.addWidget(header)

        # Assembly container (hidden when collapsed)
        self.assemblies_container = QWidget()
        self.assemblies_layout = QVBoxLayout(self.assemblies_container)
        self.assemblies_layout.setContentsMargins(0, 0, 0, 4)
        self.assemblies_layout.setSpacing(2)
        self.assemblies_container.setVisible(False)
        self.main_layout.addWidget(self.assemblies_container)

    def _toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        self.collapse_btn.setText("▷" if self.is_collapsed else "▼")
        self.assemblies_container.setVisible(not self.is_collapsed)

    def _on_pause_all(self):
        should_pause = self.pause_all_btn.isChecked()
        self.pauseAllClicked.emit(self.thing_id, should_pause)
        self.pause_all_btn.setText(">" if should_pause else "||")

    def _on_step_all(self):
        self.stepAllClicked.emit(self.thing_id)

    def add_assembly(self, assembly_id: str, assembly_name: str) -> AssemblyCycleRow:
        """Add an assembly row."""
        if assembly_id in self.assembly_rows:
            return self.assembly_rows[assembly_id]

        row = AssemblyCycleRow(self.thing_id, assembly_id, assembly_name)
        self.assembly_rows[assembly_id] = row
        self.assemblies_layout.addWidget(row)
        self._update_badge()
        return row

    def remove_assembly(self, assembly_id: str):
        """Remove an assembly row."""
        if assembly_id not in self.assembly_rows:
            return

        row = self.assembly_rows.pop(assembly_id)
        self.assemblies_layout.removeWidget(row)
        row.deleteLater()
        self._update_badge()

    def _update_badge(self):
        count = len(self.assembly_rows)
        self.badge_label.setText(str(count))

    def update_status(self, status: ThingStatus):
        """Update from ThingStatus."""
        self._update_badge()

        # Update activity indicator
        active = status.active_count
        total = status.assembly_count
        if active == 0:
            self.activity_label.setText("Idle")
            self.activity_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        else:
            self.activity_label.setText(f"{active}/{total} active")
            self.activity_label.setStyleSheet(f"color: {COLORS['phase_active']}; font-size: 10px;")

        # Update assembly rows
        for assembly_id, assembly_status in status.assemblies.items():
            if assembly_id not in self.assembly_rows:
                # Auto-add assembly row
                self.add_assembly(assembly_id, assembly_id)

            self.assembly_rows[assembly_id].update_status(assembly_status)


class CognitiveCyclesPanelV2(QWidget):
    """
    Panel displaying cognitive cycle status for all active Things and their assemblies.

    Features:
    - Hierarchical display: Things contain Assemblies
    - Collapsible Thing rows
    - Per-assembly and per-Thing pause/step controls
    - Architecture-agnostic status reporting
    - Backward compatible with single-agent-per-row format
    """

    _things_changed = pyqtSignal(dict)  # things data from API

    def __init__(self, parent=None):
        super().__init__(parent)

        self.api_base = "http://localhost:8081/api"
        self.thing_states: Dict[str, ThingStatus] = {}
        self.thing_rows: Dict[str, ThingCycleRow] = {}

        self._setup_ui()
        self._connect_signals()
        self._start_polling()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"background: {COLORS['bg']};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("Cognitive Cycles")
        title.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Expand/Collapse All toggle
        self.expand_all_btn = QPushButton("Expand All")
        self.expand_all_btn.setCheckable(True)
        self.expand_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:checked {{ background: {COLORS['phase_active']}; color: white; }}
        """)
        self.expand_all_btn.clicked.connect(self._on_expand_all)
        header_layout.addWidget(self.expand_all_btn)

        # Pause All button
        self.pause_all_btn = QPushButton("Pause All")
        self.pause_all_btn.setCheckable(True)
        self.pause_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: {COLORS['button_hover']}; }}
            QPushButton:checked {{ background: {COLORS['pause_active']}; color: white; }}
        """)
        self.pause_all_btn.clicked.connect(self._on_pause_all)
        header_layout.addWidget(self.pause_all_btn)

        layout.addWidget(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {COLORS['bg']};
                border: none;
            }}
            QScrollBar:vertical {{
                background: #2D2D2D;
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background: #666666;
                min-height: 20px;
                border-radius: 6px;
            }}
        """)

        self.things_container = QWidget()
        self.things_container.setStyleSheet(f"background: {COLORS['bg']};")
        self.things_layout = QVBoxLayout(self.things_container)
        self.things_layout.setContentsMargins(4, 4, 4, 4)
        self.things_layout.setSpacing(4)
        self.things_layout.addStretch()

        scroll.setWidget(self.things_container)
        layout.addWidget(scroll, 1)

        # Status bar
        self.status_label = QLabel("Connecting...")
        self.status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 9px; padding: 4px 8px; background: {COLORS['bg']};"
        )
        layout.addWidget(self.status_label)

    def _connect_signals(self):
        self._things_changed.connect(self._handle_things_changed)

    def _start_polling(self):
        # Agent list polling (slower)
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_things)
        self.poll_timer.start(2000)

        # Phase polling (faster for smooth animation)
        self.phase_poll_timer = QTimer()
        self.phase_poll_timer.timeout.connect(self._poll_cycle_phases)
        self.phase_poll_timer.start(250)

        self._poll_things()

    def _poll_things(self):
        """Fetch thing/agent list from local monitor and API."""
        things_data = {}

        # First, get things from local CognitionMonitor
        local_count = 0
        if HAS_LOCAL_MONITOR:
            try:
                monitor = get_cognition_monitor()
                local_data = monitor.get_hierarchical_data().get('things', {})
                for thing_id, thing_info in local_data.items():
                    assemblies = {}
                    for assembly_id, assembly_info in thing_info.get('assemblies', {}).items():
                        assemblies[assembly_id] = {
                            'name': assembly_id,
                            'phase': assembly_info.get('phase', 'IDLE'),
                            'status_text': assembly_info.get('status_text', '')
                        }
                    things_data[thing_id] = {
                        'name': thing_info.get('name', thing_id),
                        'assemblies': assemblies
                    }
                    local_count += 1
            except Exception as e:
                logger.debug(f"Local monitor error: {e}")

        # Then, get things from HTTP API (cmush agents)
        remote_count = 0
        try:
            resp = requests.get(f"{self.api_base}/agents", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                # Convert to hierarchical format and merge
                remote_things = self._convert_agents_to_things(data.get('agents', []))
                for thing_id, thing_info in remote_things.items():
                    if thing_id not in things_data:
                        things_data[thing_id] = thing_info
                        remote_count += 1
        except requests.exceptions.ConnectionError:
            pass  # Server offline - that's fine, we may have local data
        except Exception as e:
            logger.debug(f"API error: {e}")

        # Emit combined data
        if things_data:
            self._things_changed.emit(things_data)

        # Update status
        total = sum(len(t.get('assemblies', {})) for t in things_data.values())
        if local_count > 0 and remote_count > 0:
            self.status_label.setText(f"{local_count} local + {remote_count} remote, {total} assembly(ies)")
        elif local_count > 0:
            self.status_label.setText(f"{local_count} local thing(s), {total} assembly(ies)")
        elif remote_count > 0:
            self.status_label.setText(f"{remote_count} remote thing(s), {total} assembly(ies)")
        else:
            self.status_label.setText("No active things")

    def _convert_agents_to_things(self, agents: List[dict]) -> Dict[str, dict]:
        """
        Convert flat agent list to hierarchical Things format.

        For backward compatibility, each agent becomes a Thing with a single
        'default' assembly. When the API supports multiple assemblies per Thing,
        this will pass through directly.
        """
        things = {}
        for agent in agents:
            agent_id = agent.get('id', agent.get('agent_id', ''))
            agent_name = agent.get('name', agent_id)

            # Check if API returns hierarchical data
            if 'assemblies' in agent:
                # New hierarchical format
                things[agent_id] = {
                    'name': agent_name,
                    'assemblies': agent['assemblies']
                }
            else:
                # Legacy format - create single 'default' assembly
                things[agent_id] = {
                    'name': agent_name,
                    'assemblies': {
                        'default': {
                            'name': 'cognition',
                            'phase': 'IDLE',
                            'status_text': ''
                        }
                    }
                }
        return things

    def _poll_cycle_phases(self):
        """Poll cycle phases for all things from both local monitor and HTTP API."""
        # First, poll local CognitionMonitor (in-process assemblies)
        self._poll_local_monitor()

        # Then, poll HTTP API (cmush agents)
        try:
            resp = requests.get(f"{self.api_base}/cycle_phases", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()

                # Check for hierarchical format
                if 'things' in data:
                    # New format
                    self._update_from_hierarchical(data['things'])
                elif 'agents' in data:
                    # Legacy format
                    self._update_from_legacy_agents(data['agents'])

        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.debug(f"Phase poll error: {e}")

    def _poll_local_monitor(self):
        """Poll status from local CognitionMonitor singleton."""
        if not HAS_LOCAL_MONITOR:
            return

        try:
            monitor = get_cognition_monitor()
            local_data = monitor.get_hierarchical_data().get('things', {})

            # Process each thing from the local monitor
            for thing_id, thing_data in local_data.items():
                thing_name = thing_data.get('name', thing_id)

                # Auto-add thing if not present
                if thing_id not in self.thing_states:
                    assemblies = {}
                    for assembly_id in thing_data.get('assemblies', {}).keys():
                        assemblies[assembly_id] = {'name': assembly_id}
                    self._add_thing_row(thing_id, thing_name, assemblies)

                if thing_id not in self.thing_states:
                    continue

                status = self.thing_states[thing_id]

                # Update assemblies
                for assembly_id, assembly_data in thing_data.get('assemblies', {}).items():
                    if assembly_id not in status.assemblies:
                        status.assemblies[assembly_id] = AssemblyStatus(
                            assembly_id=assembly_id,
                            thing_id=thing_id
                        )

                    assembly_status = status.assemblies[assembly_id]

                    # Convert MonitorCyclePhase to local CyclePhase
                    phase_str = assembly_data.get('phase', 'IDLE')
                    if isinstance(phase_str, Enum):
                        phase_str = phase_str.name
                    assembly_status.phase = self._parse_phase(phase_str)
                    assembly_status.current_facet = assembly_data.get('current_facet', '')
                    assembly_status.status_text = assembly_data.get('status_text', '')
                    assembly_status.activity = assembly_data.get('activity', 0.0)
                    assembly_status.is_paused = assembly_data.get('is_paused', False)

                # Update UI
                if thing_id in self.thing_rows:
                    self.thing_rows[thing_id].update_status(status)

        except Exception as e:
            logger.debug(f"Local monitor poll error: {e}")

    def _update_from_hierarchical(self, things_data: Dict):
        """Update from hierarchical things data."""
        for thing_id, thing_data in things_data.items():
            if thing_id not in self.thing_states:
                continue

            status = self.thing_states[thing_id]

            for assembly_id, assembly_data in thing_data.get('assemblies', {}).items():
                if assembly_id not in status.assemblies:
                    status.assemblies[assembly_id] = AssemblyStatus(
                        assembly_id=assembly_id,
                        thing_id=thing_id
                    )

                assembly_status = status.assemblies[assembly_id]
                phase_str = assembly_data.get('phase', 'IDLE')
                assembly_status.phase = self._parse_phase(phase_str)
                assembly_status.current_facet = assembly_data.get('current_facet', '')
                assembly_status.status_text = assembly_data.get('status_text', '')
                assembly_status.activity = assembly_data.get('activity', 0.0)
                assembly_status.is_paused = assembly_data.get('is_paused', False)

            # Update UI
            if thing_id in self.thing_rows:
                self.thing_rows[thing_id].update_status(status)

    def _update_from_legacy_agents(self, agents_data: Dict):
        """Update from legacy flat agents data."""
        for agent_id, agent_data in agents_data.items():
            if agent_id not in self.thing_states:
                continue

            status = self.thing_states[agent_id]

            # Create/update 'default' assembly
            if 'default' not in status.assemblies:
                status.assemblies['default'] = AssemblyStatus(
                    assembly_id='default',
                    thing_id=agent_id
                )

            assembly_status = status.assemblies['default']
            phase_str = agent_data.get('phase', 'IDLE')
            assembly_status.phase = self._parse_phase(phase_str)
            assembly_status.current_facet = agent_data.get('current_facet', '')
            assembly_status.status_text = self._format_legacy_status(agent_data)
            assembly_status.is_paused = agent_data.get('is_paused', False)

            # Update UI
            if agent_id in self.thing_rows:
                self.thing_rows[agent_id].update_status(status)

    def _format_legacy_status(self, agent_data: Dict) -> str:
        """Format legacy agent data into status_text."""
        parts = []
        if agent_data.get('current_assembly'):
            parts.append(agent_data['current_assembly'])
        if agent_data.get('current_facet'):
            parts.append(agent_data['current_facet'])
        if agent_data.get('current_model_label'):
            parts.append(agent_data['current_model_label'])

        llm_status = agent_data.get('current_llm_status', '')
        if llm_status:
            parts.append(f"[{llm_status}]")

        return " :: ".join(parts) if parts else ""

    def _parse_phase(self, phase_str: str) -> CyclePhase:
        """Parse phase string to enum."""
        phase_map = {
            'IDLE': CyclePhase.IDLE,
            'INCOMING': CyclePhase.INCOMING,
            'PRECOG': CyclePhase.PRECOG,
            'FACET': CyclePhase.FACET,
            'NEURAL': CyclePhase.NEURAL,
            'POSTCOG': CyclePhase.POSTCOG,
            'OUTGOING': CyclePhase.OUTGOING,
            'OUTPUT': CyclePhase.OUTGOING,  # Legacy
        }
        return phase_map.get(phase_str, CyclePhase.IDLE)

    @pyqtSlot(dict)
    def _handle_things_changed(self, things_data: Dict):
        """Handle thing list update (main thread)."""
        current_ids = set(self.thing_rows.keys())
        new_ids = set(things_data.keys())

        # Add new things
        for thing_id in new_ids - current_ids:
            thing_info = things_data[thing_id]
            self._add_thing_row(thing_id, thing_info['name'], thing_info.get('assemblies', {}))

        # Remove departed things
        for thing_id in current_ids - new_ids:
            self._remove_thing_row(thing_id)

    def _add_thing_row(self, thing_id: str, thing_name: str, assemblies: Dict):
        """Add a row for a new Thing."""
        if thing_id in self.thing_rows:
            return

        # Create state
        status = ThingStatus(thing_id=thing_id, thing_name=thing_name)
        self.thing_states[thing_id] = status

        # Create row
        row = ThingCycleRow(thing_id, thing_name)
        row.pauseAllClicked.connect(self._on_thing_pause_all)
        row.stepAllClicked.connect(self._on_thing_step_all)
        self.thing_rows[thing_id] = row

        # Add assemblies
        for assembly_id, assembly_info in assemblies.items():
            assembly_name = assembly_info.get('name', assembly_id)
            assembly_row = row.add_assembly(assembly_id, assembly_name)
            assembly_row.pauseClicked.connect(self._on_assembly_pause)
            assembly_row.stepClicked.connect(self._on_assembly_step)

            # Create assembly status
            status.assemblies[assembly_id] = AssemblyStatus(
                assembly_id=assembly_id,
                thing_id=thing_id
            )

        # Insert before stretch
        count = self.things_layout.count()
        self.things_layout.insertWidget(count - 1, row)

        logger.info(f"[CognitiveCyclesPanelV2] Added thing: {thing_name} ({thing_id})")

    def _remove_thing_row(self, thing_id: str):
        """Remove row for departed Thing."""
        if thing_id not in self.thing_rows:
            return

        row = self.thing_rows.pop(thing_id)
        self.thing_states.pop(thing_id, None)

        self.things_layout.removeWidget(row)
        row.deleteLater()

        logger.info(f"[CognitiveCyclesPanelV2] Removed thing: {thing_id}")

    def _on_expand_all(self):
        """Toggle expand/collapse all."""
        expand = self.expand_all_btn.isChecked()
        self.expand_all_btn.setText("Collapse All" if expand else "Expand All")

        for row in self.thing_rows.values():
            if expand and row.is_collapsed:
                row._toggle_collapse()
            elif not expand and not row.is_collapsed:
                row._toggle_collapse()

    def _on_pause_all(self):
        """Handle global Pause All."""
        should_pause = self.pause_all_btn.isChecked()

        try:
            resp = requests.post(
                f"{self.api_base}/cognition/pause",
                json={'paused': should_pause},
                timeout=5
            )

            if resp.status_code == 200:
                self.pause_all_btn.setText("Resume All" if should_pause else "Pause All")
                self.status_label.setText(f"{'Paused' if should_pause else 'Resumed'} all")
            else:
                self.status_label.setText(f"Pause failed: {resp.status_code}")
                self.pause_all_btn.setChecked(not should_pause)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")
            self.pause_all_btn.setChecked(not should_pause)

    @pyqtSlot(str, bool)
    def _on_thing_pause_all(self, thing_id: str, should_pause: bool):
        """Handle per-Thing pause all."""
        try:
            resp = requests.post(
                f"{self.api_base}/cognition/pause",
                json={'thing_id': thing_id, 'paused': should_pause},
                timeout=2
            )

            if resp.status_code == 200:
                self.status_label.setText(f"{'Paused' if should_pause else 'Resumed'} {thing_id[:8]}...")
            else:
                self.status_label.setText(f"Pause failed: {resp.status_code}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    @pyqtSlot(str)
    def _on_thing_step_all(self, thing_id: str):
        """Handle per-Thing step all."""
        try:
            resp = requests.post(
                f"{self.api_base}/cognition/step",
                json={'thing_id': thing_id},
                timeout=2
            )

            if resp.status_code == 200:
                self.status_label.setText(f"Stepped {thing_id[:8]}...")
            else:
                self.status_label.setText(f"Step failed: {resp.status_code}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    @pyqtSlot(str, str, bool)
    def _on_assembly_pause(self, thing_id: str, assembly_id: str, should_pause: bool):
        """Handle per-assembly pause."""
        try:
            resp = requests.post(
                f"{self.api_base}/cognition/pause",
                json={
                    'thing_id': thing_id,
                    'assembly_id': assembly_id,
                    'paused': should_pause
                },
                timeout=2
            )

            if resp.status_code == 200:
                self.status_label.setText(
                    f"{'Paused' if should_pause else 'Resumed'} {assembly_id[:12]}..."
                )
            else:
                self.status_label.setText(f"Pause failed: {resp.status_code}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    @pyqtSlot(str, str)
    def _on_assembly_step(self, thing_id: str, assembly_id: str):
        """Handle per-assembly step."""
        try:
            resp = requests.post(
                f"{self.api_base}/cognition/step",
                json={
                    'thing_id': thing_id,
                    'assembly_id': assembly_id
                },
                timeout=2
            )

            if resp.status_code == 200:
                self.status_label.setText(f"Stepped {assembly_id[:12]}...")
            else:
                self.status_label.setText(f"Step failed: {resp.status_code}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    def cleanup(self):
        """Cleanup resources on panel close."""
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        if hasattr(self, 'phase_poll_timer'):
            self.phase_poll_timer.stop()


# Export for backwards compatibility
CognitiveCyclesPanel = CognitiveCyclesPanelV2


__all__ = [
    'CognitiveCyclesPanelV2',
    'CognitiveCyclesPanel',
    'CyclePhase',
    'AssemblyStatus',
    'ThingStatus',
    'ThingCycleRow',
    'AssemblyCycleRow',
]
