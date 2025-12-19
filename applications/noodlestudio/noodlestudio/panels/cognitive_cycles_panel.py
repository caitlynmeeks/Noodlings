"""
Cognitive Cycles Panel - Real-time visualization of agent cognitive processing

Displays all active Noodling agents with their current cognitive cycle phase.
Each agent shows a segmented phase indicator and per-agent pause/step controls.

Phases:
- PRECOG: Pre-facet processing (context parsing, perception)
- FACET: Facet assembly execution (LLM calls, scripted logic)
- NEURAL: Neural processing (CharmNetwork, MLX)
- POSTCOG: Post-facet processing (response convergence, speech gate)
- OUTPUT: Final emission to world

Note: FACET and NEURAL interleave during execution since CharmNetwork
is accessed via facets within the assembly topology.

Author: Commander Spock + Cadet Caity
Date: December 2025
"""

import asyncio
import logging
import requests
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QRectF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen

logger = logging.getLogger(__name__)


# Color scheme - Starbucks green for active cognition
COLORS = {
    'bg': '#383838',
    'row_bg': '#2d2d2d',
    'row_hover': '#3a3a3a',
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
class AgentCycleState:
    """State tracking for a single agent's cognitive cycle."""
    agent_id: str
    agent_name: str
    current_phase: CyclePhase = CyclePhase.IDLE
    current_facet: str = ""
    current_assembly: str = ""
    current_model_label: str = ""
    current_model_name: str = ""
    current_llm_status: str = ""  # QUERYING, AWAITING_RESPONSE, ERROR, or empty
    cycle_uuid: str = ""
    is_paused: bool = False
    step_mode: bool = False
    pending_llm_calls: int = 0
    cycle_in_progress: bool = False


class CyclePhaseIndicator(QWidget):
    """
    Custom widget showing 6 cognitive phases as segmented bar.

    Segments: INCOMING | PRECOG | FACET | NEURAL | POSTCOG | OUTGOING

    Colors indicate state:
    - Idle/pending: dark gray
    - Active: Starbucks green (pulsing)
    - Complete: muted green (phases passed this cycle)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_phase = CyclePhase.IDLE
        self.pulse_state = False  # For animation
        self.setMinimumSize(240, 24)  # Wider for 6 phases
        self.setMaximumHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Pulse animation timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._pulse_timer.start(300)  # Pulse every 300ms

    def _toggle_pulse(self):
        """Toggle pulse state for animation."""
        if self.current_phase != CyclePhase.IDLE:
            self.pulse_state = not self.pulse_state
            self.update()

    def set_phase(self, phase: CyclePhase):
        """Update the current phase and trigger repaint."""
        if phase != self.current_phase:
            self.current_phase = phase
            self.pulse_state = True  # Reset pulse on phase change
            self.update()

    def paintEvent(self, event):
        """Custom paint for the 6-segment phase indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        segment_width = width / 6  # 6 phases now
        gap = 2  # Gap between segments

        # Font for labels
        font = QFont("Arial", 6)  # Slightly smaller for 6 labels
        painter.setFont(font)

        for i, label in enumerate(PHASE_LABELS):
            x = i * segment_width
            rect = QRectF(x + gap/2, 2, segment_width - gap, height - 4)

            # Determine color based on current phase
            phase_value = i + 1  # INCOMING=1, PRECOG=2, FACET=3, etc.

            if self.current_phase == CyclePhase.IDLE:
                # All segments idle
                bg_color = QColor(COLORS['phase_idle'])
                text_color = QColor(COLORS['text_dim'])
            elif phase_value < self.current_phase.value:
                # Completed phases (passed this cycle)
                bg_color = QColor(COLORS['phase_complete'])
                text_color = QColor(COLORS['text_muted'])
            elif phase_value == self.current_phase.value:
                # Active phase - green with pulse
                if self.pulse_state:
                    bg_color = QColor(COLORS['phase_active_pulse'])
                else:
                    bg_color = QColor(COLORS['phase_active'])
                text_color = QColor(COLORS['text_active'])
            else:
                # Pending phases
                bg_color = QColor(COLORS['phase_idle'])
                text_color = QColor(COLORS['text_dim'])

            # Draw segment background
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(rect, 3, 3)

            # Draw phase label
            painter.setPen(text_color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()


class AgentCycleRow(QFrame):
    """
    Single row showing an agent's cognitive cycle status.

    Layout: [Name] [Phase Indicator] [Current Facet] [Pause] [Step]
    """

    pauseClicked = pyqtSignal(str, bool)  # agent_id, should_pause
    stepClicked = pyqtSignal(str)  # agent_id

    def __init__(self, agent_id: str, agent_name: str, parent=None):
        super().__init__(parent)
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.is_paused = False
        self.step_mode = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            AgentCycleRow {{
                background: {COLORS['row_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
            }}
            AgentCycleRow:hover {{
                border: 1px solid #666666;
            }}
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Build the row layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Agent name
        self.name_label = QLabel(self.agent_name)
        self.name_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        self.name_label.setMinimumWidth(120)
        self.name_label.setMaximumWidth(150)
        layout.addWidget(self.name_label)

        # Phase indicator
        self.phase_indicator = CyclePhaseIndicator()
        layout.addWidget(self.phase_indicator)

        # Current facet name
        self.facet_label = QLabel("Idle")
        self.facet_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
        self.facet_label.setMinimumWidth(150)
        layout.addWidget(self.facet_label, 1)  # Stretch

        # Pause button
        self.pause_btn = QPushButton("||")
        self.pause_btn.setFixedSize(28, 24)
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Pause cognition for this agent")
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {COLORS['button_hover']};
            }}
            QPushButton:checked {{
                background: {COLORS['pause_active']};
                color: white;
            }}
        """)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.pause_btn)

        # Step button
        self.step_btn = QPushButton(">|")
        self.step_btn.setFixedSize(28, 24)
        self.step_btn.setToolTip("Step forward one cycle")
        self.step_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['button_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {COLORS['button_hover']};
            }}
            QPushButton:pressed {{
                background: {COLORS['step_active']};
            }}
        """)
        self.step_btn.clicked.connect(self._on_step_clicked)
        layout.addWidget(self.step_btn)

    def _on_pause_clicked(self):
        """Handle pause button click."""
        self.is_paused = self.pause_btn.isChecked()
        self.pauseClicked.emit(self.agent_id, self.is_paused)

        # Update button text
        self.pause_btn.setText(">" if self.is_paused else "||")
        self.pause_btn.setToolTip("Resume cognition" if self.is_paused else "Pause cognition")

    def _on_step_clicked(self):
        """Handle step button click."""
        self.stepClicked.emit(self.agent_id)

    def update_state(self, state: AgentCycleState):
        """Update row display from state."""
        # Update phase indicator
        self.phase_indicator.set_phase(state.current_phase)

        # Update facet label with detailed format:
        # ASSEMBLY::Facet Name, Model Label [STATUS]
        if state.current_phase == CyclePhase.IDLE:
            self.facet_label.setText("Idle")
            self.facet_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic;")
        else:
            # Build detailed status text
            parts = []

            # Assembly::Facet
            if state.current_assembly and state.current_facet:
                parts.append(f"{state.current_assembly}::{state.current_facet}")
            elif state.current_facet:
                parts.append(state.current_facet)
            else:
                parts.append(f"{state.current_phase.name}...")

            # Model label
            if state.current_model_label:
                parts.append(f", {state.current_model_label}")

            # LLM status indicator
            status_text = ""
            if state.current_llm_status == "QUERYING":
                status_text = " [QUERYING]"
            elif state.current_llm_status == "AWAITING_RESPONSE":
                status_text = " [AWAITING RESPONSE]"
            elif state.current_llm_status == "ERROR":
                status_text = " [ERROR]"

            facet_text = "".join(parts) + status_text
            self.facet_label.setText(facet_text)

            # Color based on status - white text, red only for errors
            if state.current_llm_status == "ERROR":
                self.facet_label.setStyleSheet(f"color: #FF6666; font-style: normal; font-weight: bold;")
            else:
                self.facet_label.setStyleSheet(f"color: {COLORS['text']}; font-style: normal;")

        # Update pause button state (without triggering signal)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(state.is_paused)
        self.pause_btn.setText(">" if state.is_paused else "||")
        self.is_paused = state.is_paused
        self.pause_btn.blockSignals(False)


class CognitiveCyclesPanel(QWidget):
    """
    Panel displaying cognitive cycle status for all active agents.

    Features:
    - Real-time phase indicator per agent
    - Per-agent pause/step controls
    - Event bus subscription for live updates
    - Automatic agent discovery via API polling
    """

    # Signals for thread-safe UI updates (event bus is async)
    _cycle_event = pyqtSignal(str, str, str, dict)  # agent_id, subtype, facet_type, data
    _agents_changed = pyqtSignal(list)  # list of agent dicts

    def __init__(self, parent=None):
        super().__init__(parent)

        self.api_base = "http://localhost:8081/api"
        self.agent_states: Dict[str, AgentCycleState] = {}
        self.agent_rows: Dict[str, AgentCycleRow] = {}
        self.event_listener = None

        self._setup_ui()
        self._connect_signals()
        self._start_polling()
        self._subscribe_to_events()

    def _setup_ui(self):
        """Build the panel UI."""
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
            QPushButton:hover {{
                background: {COLORS['button_hover']};
            }}
            QPushButton:checked {{
                background: {COLORS['pause_active']};
                color: white;
            }}
        """)
        self.pause_all_btn.clicked.connect(self._on_pause_all)
        header_layout.addWidget(self.pause_all_btn)

        layout.addWidget(header)

        # Scroll area for agent rows
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

        self.agents_container = QWidget()
        self.agents_container.setStyleSheet(f"background: {COLORS['bg']};")
        self.agents_layout = QVBoxLayout(self.agents_container)
        self.agents_layout.setContentsMargins(4, 4, 4, 4)
        self.agents_layout.setSpacing(4)
        self.agents_layout.addStretch()

        scroll.setWidget(self.agents_container)
        layout.addWidget(scroll, 1)

        # Status bar
        self.status_label = QLabel("Connecting...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 9px; padding: 4px 8px; background: {COLORS['bg']};")
        layout.addWidget(self.status_label)

    def _connect_signals(self):
        """Connect internal signals for thread-safe updates."""
        self._cycle_event.connect(self._handle_cycle_event)
        self._agents_changed.connect(self._handle_agents_changed)

    def _start_polling(self):
        """Start polling for agent list changes."""
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_agents)
        self.poll_timer.start(2000)  # Every 2 seconds

        # Initial poll
        self._poll_agents()

    def _poll_agents(self):
        """Fetch agent list from API."""
        try:
            resp = requests.get(f"{self.api_base}/agents", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                agents = data.get('agents', [])
                self._agents_changed.emit(agents)
                self.status_label.setText(f"{len(agents)} agent(s) active")
        except requests.exceptions.ConnectionError:
            self.status_label.setText("Server offline")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    @pyqtSlot(list)
    def _handle_agents_changed(self, agents: List[dict]):
        """Handle agent list update (main thread)."""
        current_ids = set(self.agent_rows.keys())
        new_ids = set()

        for agent in agents:
            agent_id = agent.get('id', agent.get('agent_id', ''))
            agent_name = agent.get('name', agent_id)
            new_ids.add(agent_id)

            if agent_id not in current_ids:
                # Add new agent
                self._add_agent_row(agent_id, agent_name)

        # Remove departed agents
        for agent_id in current_ids - new_ids:
            self._remove_agent_row(agent_id)

    def _add_agent_row(self, agent_id: str, agent_name: str):
        """Add a row for a new agent."""
        if agent_id in self.agent_rows:
            return

        # Create state
        state = AgentCycleState(agent_id=agent_id, agent_name=agent_name)
        self.agent_states[agent_id] = state

        # Create row widget
        row = AgentCycleRow(agent_id, agent_name)
        row.pauseClicked.connect(self._on_agent_pause)
        row.stepClicked.connect(self._on_agent_step)
        self.agent_rows[agent_id] = row

        # Insert before stretch
        count = self.agents_layout.count()
        self.agents_layout.insertWidget(count - 1, row)

        logger.info(f"[CognitiveCyclesPanel] Added agent: {agent_name} ({agent_id})")

    def _remove_agent_row(self, agent_id: str):
        """Remove row for departed agent."""
        if agent_id not in self.agent_rows:
            return

        row = self.agent_rows.pop(agent_id)
        self.agent_states.pop(agent_id, None)

        self.agents_layout.removeWidget(row)
        row.deleteLater()

        logger.info(f"[CognitiveCyclesPanel] Removed agent: {agent_id}")

    def _subscribe_to_events(self):
        """
        Poll cmush API for real-time cycle phase updates.

        Uses /api/cycle_phases endpoint instead of local ExecutionEventBus
        since cmush runs in a separate process.
        """
        # Start phase polling timer (faster than agent list polling)
        self.phase_poll_timer = QTimer()
        self.phase_poll_timer.timeout.connect(self._poll_cycle_phases)
        self.phase_poll_timer.start(250)  # 4 Hz for smooth animation

        logger.info("[CognitiveCyclesPanel] Started cycle phase polling")

    def _poll_cycle_phases(self):
        """Poll cmush API for current cycle phases."""
        try:
            resp = requests.get(f"{self.api_base}/cycle_phases", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                agents_data = data.get('agents', {})

                for agent_id, agent_data in agents_data.items():
                    if agent_id not in self.agent_states:
                        continue

                    state = self.agent_states[agent_id]

                    # Map phase string to CyclePhase enum
                    phase_str = agent_data.get('phase', 'IDLE')
                    phase_map = {
                        'IDLE': CyclePhase.IDLE,
                        'INCOMING': CyclePhase.INCOMING,
                        'PRECOG': CyclePhase.PRECOG,
                        'FACET': CyclePhase.FACET,
                        'NEURAL': CyclePhase.NEURAL,
                        'POSTCOG': CyclePhase.POSTCOG,
                        'OUTGOING': CyclePhase.OUTGOING,
                        # Legacy support
                        'OUTPUT': CyclePhase.OUTGOING,
                    }
                    new_phase = phase_map.get(phase_str, CyclePhase.IDLE)

                    # Update state
                    state.current_phase = new_phase
                    state.current_facet = agent_data.get('current_facet', '')
                    state.current_assembly = agent_data.get('current_assembly', '')
                    state.current_model_label = agent_data.get('current_model_label', '')
                    state.current_model_name = agent_data.get('current_model_name', '')
                    state.current_llm_status = agent_data.get('current_llm_status', '')
                    state.cycle_in_progress = agent_data.get('cycle_in_progress', False)
                    state.pending_llm_calls = agent_data.get('pending_llm_calls', 0)

                    # Update UI
                    if agent_id in self.agent_rows:
                        self.agent_rows[agent_id].update_state(state)

        except requests.exceptions.ConnectionError:
            pass  # Server offline - silently ignore
        except Exception as e:
            logger.debug(f"[CognitiveCyclesPanel] Phase poll error: {e}")

    @pyqtSlot(str, str, str, dict)
    def _handle_cycle_event(self, agent_id: str, subtype: str, facet_type: str, data: dict):
        """Handle execution event (main thread)."""
        if agent_id not in self.agent_states:
            return

        state = self.agent_states[agent_id]

        # Map event to phase
        new_phase = self._map_event_to_phase(subtype, facet_type, data)

        if new_phase is not None:
            state.current_phase = new_phase

        # Update facet name
        facet_name = data.get('facet_name', '')
        if facet_name:
            state.current_facet = facet_name

        # Update cycle tracking
        if subtype == 'cycle_start':
            state.cycle_in_progress = True
            state.cycle_uuid = data.get('cycle_uuid', '')
        elif subtype == 'cycle_complete':
            state.cycle_in_progress = False
            state.current_facet = ""

        # Update UI
        if agent_id in self.agent_rows:
            self.agent_rows[agent_id].update_state(state)

    def _map_event_to_phase(self, subtype: str, facet_type: str, data: dict) -> Optional[CyclePhase]:
        """Map execution event to cognitive phase."""

        if subtype == 'cycle_start':
            return CyclePhase.PRECOG

        elif subtype == 'cycle_complete':
            return CyclePhase.IDLE

        elif subtype in ('facet_start', 'facet_complete'):
            # Map facet type to phase
            if facet_type == 'ContextIntelligenceFacet':
                return CyclePhase.PRECOG
            elif facet_type == 'CharmNetworkFacet':
                return CyclePhase.NEURAL
            elif facet_type in ('SpeechGateFacet', 'ResponseConvergence'):
                return CyclePhase.POSTCOG
            elif facet_type == 'SpecialNode':
                facet_name = data.get('facet_name', '')
                if 'OUTGOING' in facet_name:
                    return CyclePhase.OUTPUT
                elif 'INCOMING' in facet_name:
                    return CyclePhase.PRECOG
            else:
                # Default LLM facets and others
                return CyclePhase.FACET

        elif subtype == 'data_flow':
            # Check if flowing to OUTGOING
            to_facet = data.get('to_facet', '')
            if 'outgoing' in to_facet.lower():
                return CyclePhase.OUTPUT

        return None

    def _on_pause_all(self):
        """Handle Pause All button."""
        should_pause = self.pause_all_btn.isChecked()

        try:
            resp = requests.post(
                f"{self.api_base}/cognition/pause",
                json={'paused': should_pause},
                timeout=5
            )

            if resp.status_code == 200:
                self.pause_all_btn.setText("Resume All" if should_pause else "Pause All")

                # Update all agent states
                for state in self.agent_states.values():
                    state.is_paused = should_pause

                for row in self.agent_rows.values():
                    row.pause_btn.blockSignals(True)
                    row.pause_btn.setChecked(should_pause)
                    row.pause_btn.setText(">" if should_pause else "||")
                    row.is_paused = should_pause
                    row.pause_btn.blockSignals(False)

                self.status_label.setText(f"{'Paused' if should_pause else 'Resumed'} all agents")
            else:
                self.status_label.setText(f"Pause failed: {resp.status_code}")
                self.pause_all_btn.setChecked(not should_pause)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")
            self.pause_all_btn.setChecked(not should_pause)

    @pyqtSlot(str, bool)
    def _on_agent_pause(self, agent_id: str, should_pause: bool):
        """Handle per-agent pause."""
        try:
            # Use the general pause endpoint with agent_id
            resp = requests.post(
                f"{self.api_base}/cognition/pause",
                json={'agent_id': agent_id, 'paused': should_pause},
                timeout=2
            )

            if resp.status_code == 200:
                if agent_id in self.agent_states:
                    self.agent_states[agent_id].is_paused = should_pause
                self.status_label.setText(f"{'Paused' if should_pause else 'Resumed'} {agent_id[:8]}...")
            else:
                self.status_label.setText(f"Pause failed: {resp.status_code}")
                # Revert button state
                if agent_id in self.agent_rows:
                    row = self.agent_rows[agent_id]
                    row.pause_btn.blockSignals(True)
                    row.pause_btn.setChecked(not should_pause)
                    row.pause_btn.blockSignals(False)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    @pyqtSlot(str)
    def _on_agent_step(self, agent_id: str):
        """Handle per-agent step forward."""
        try:
            resp = requests.post(
                f"{self.api_base}/agents/{agent_id}/step/continue",
                timeout=2
            )

            if resp.status_code == 200:
                self.status_label.setText(f"Stepped {agent_id[:8]}...")
            else:
                self.status_label.setText(f"Step failed: {resp.status_code}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:30]}")

    def cleanup(self):
        """Cleanup resources on panel close."""
        # Stop agent list polling
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()

        # Stop phase polling
        if hasattr(self, 'phase_poll_timer'):
            self.phase_poll_timer.stop()
