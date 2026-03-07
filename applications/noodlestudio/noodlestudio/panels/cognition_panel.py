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
#   Cognition Panel
#
#   Per-character prompt chain debugger. Shows system prompts,
#   formatted prompts, and outputs for each facet in the assembly
#   execution trace. Data arrives via GuidePerformanceManager's
#   _on_turn_trace() callback.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.cognition_panel
# PURPOSE:  Per-facet execution trace viewer
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CollapsibleFacetSection, CognitionPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from datetime import datetime
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QScrollArea, QPlainTextEdit, QFrame
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


# =============================================================================
# Collapsible Facet Section
# =============================================================================

class CollapsibleFacetSection(QWidget):
    """One facet's trace in the Cognition Panel.

    Header: flat button "> FacetName [MODEL] 142ms 47 tok" -- click to expand.
    Body: 3 toggle buttons (System / Prompt / Output) + read-only text area.
    """

    def __init__(self, trace: dict, parent=None):
        super().__init__(parent)
        self._trace = trace
        self._expanded = False
        self._active_view = 'output'

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Header button ---
        facet_name = trace.get('facet_name', trace.get('facet_id', '?'))
        model = trace.get('model_label', '')
        time_ms = int(trace.get('execution_time', 0) * 1000)
        tokens = trace.get('token_count', 0)

        header_parts = [f"> {facet_name}"]
        if model:
            header_parts.append(f"[{model}]")
        header_parts.append(f"{time_ms}ms")
        if tokens:
            header_parts.append(f"{tokens} tok")

        self._header_btn = QPushButton("  ".join(header_parts))
        self._header_btn.setFlat(True)
        self._header_btn.setStyleSheet("""
            QPushButton {
                background-color: #2A2A2A;
                color: #AAAAAA;
                text-align: left;
                padding: 6px 10px;
                border: none;
                border-bottom: 1px solid #333333;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        self._header_btn.clicked.connect(self._toggle)
        layout.addWidget(self._header_btn)

        # --- Body (hidden by default) ---
        self._body = QWidget()
        self._body.setVisible(False)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 8, 4)
        body_layout.setSpacing(4)

        # Toggle buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._btn_system = QPushButton("System")
        self._btn_prompt = QPushButton("Prompt")
        self._btn_output = QPushButton("Output")

        for btn in (self._btn_system, self._btn_prompt, self._btn_output):
            btn.setFixedHeight(22)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2A2A2A;
                    color: #888;
                    border: 1px solid #3A3A3A;
                    border-radius: 3px;
                    padding: 2px 8px;
                    font-size: 10px;
                }
                QPushButton:checked {
                    background-color: #3A3A3A;
                    color: #D2D2D2;
                    border: 1px solid #555;
                }
            """)
            btn_row.addWidget(btn)

        btn_row.addStretch()
        body_layout.addLayout(btn_row)

        self._btn_system.clicked.connect(lambda: self._show_view('system'))
        self._btn_prompt.clicked.connect(lambda: self._show_view('prompt'))
        self._btn_output.clicked.connect(lambda: self._show_view('output'))

        # Text display
        self._text_view = QPlainTextEdit()
        self._text_view.setReadOnly(True)
        self._text_view.setMinimumHeight(80)
        self._text_view.setMaximumHeight(300)
        self._text_view.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1A1A1A;
                color: #888888;
                border: none;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        body_layout.addWidget(self._text_view)

        layout.addWidget(self._body)

        # Default to Output view
        self._btn_output.setChecked(True)
        self._show_view('output')

    def _toggle(self):
        """Toggle body visibility."""
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)

    def _show_view(self, view: str):
        """Switch between system/prompt/output views."""
        self._active_view = view

        # Update button states
        self._btn_system.setChecked(view == 'system')
        self._btn_prompt.setChecked(view == 'prompt')
        self._btn_output.setChecked(view == 'output')

        # Update text
        if view == 'system':
            text = self._trace.get('system_prompt', '')
        elif view == 'prompt':
            text = self._trace.get('formatted_prompt', '')
        else:
            text = str(self._trace.get('output', ''))

        self._text_view.setPlainText(text or '(empty)')


# =============================================================================
# Cognition Panel
# =============================================================================

class CognitionPanel(QWidget):
    """Per-character prompt chain debugger.

    Shows the execution trace for each facet in the assembly.
    Character dropdown selects which noodling's traces to view.
    Auto-follow mode switches to the most recent speaker.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._traces_by_noodling: Dict[str, List] = {}  # noodling_id -> [trace_lists]
        self._auto_follow = True

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Top bar: character dropdown + turn label + auto-follow ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self._character_combo = QComboBox()
        self._character_combo.setMinimumWidth(100)
        self._character_combo.setStyleSheet("""
            QComboBox {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #D2D2D2;
                selection-background-color: #3A3A3A;
            }
        """)
        self._character_combo.currentTextChanged.connect(self._on_character_changed)
        top_bar.addWidget(self._character_combo)

        self._turn_label = QLabel("")
        self._turn_label.setStyleSheet(
            "color: #888; font-size: 10px; font-family: 'SF Mono', monospace;"
        )
        top_bar.addWidget(self._turn_label)

        top_bar.addStretch()

        self._auto_follow_cb = QCheckBox("Auto-follow")
        self._auto_follow_cb.setChecked(True)
        self._auto_follow_cb.setStyleSheet("""
            QCheckBox {
                color: #888;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
            }
        """)
        self._auto_follow_cb.toggled.connect(self._on_auto_follow_toggled)
        top_bar.addWidget(self._auto_follow_cb)

        layout.addLayout(top_bar)

        # --- Scroll area for facet sections ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("""
            QScrollArea { background-color: #1E1E1E; border: none; }
            QScrollBar:vertical {
                background: #1E1E1E;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #3A3A3A;
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self._sections_container = QWidget()
        self._sections_layout = QVBoxLayout(self._sections_container)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(0)
        self._sections_layout.addStretch()

        self._scroll.setWidget(self._sections_container)
        layout.addWidget(self._scroll, stretch=1)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def on_turn_trace(self, noodling_id: str, traces: list,
                      turn_number: int, timestamp: float):
        """Receive execution traces from a noodling's turn.

        Called by GuidePerformanceManager._on_turn_trace().

        Args:
            noodling_id: Which noodling produced these traces
            traces: List of per-facet trace dicts
            turn_number: Sequential turn counter
            timestamp: Unix timestamp when the turn completed
        """
        # Store traces
        if noodling_id not in self._traces_by_noodling:
            self._traces_by_noodling[noodling_id] = []
        self._traces_by_noodling[noodling_id].append({
            'traces': traces,
            'turn': turn_number,
            'timestamp': timestamp,
        })

        # Update character dropdown if new noodling
        if self._character_combo.findText(noodling_id) < 0:
            self._character_combo.addItem(noodling_id)

        # Auto-follow: switch to the speaking noodling
        if self._auto_follow:
            idx = self._character_combo.findText(noodling_id)
            if idx >= 0:
                self._character_combo.setCurrentIndex(idx)

        # If this noodling is currently selected, update display
        if self._character_combo.currentText() == noodling_id:
            self._display_latest(noodling_id, turn_number, timestamp)

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _on_character_changed(self, noodling_id: str):
        """Handle character dropdown selection change."""
        if not noodling_id:
            return
        entries = self._traces_by_noodling.get(noodling_id, [])
        if entries:
            latest = entries[-1]
            self._display_latest(
                noodling_id, latest['turn'], latest['timestamp']
            )

    def _on_auto_follow_toggled(self, checked: bool):
        """Handle auto-follow checkbox toggle."""
        self._auto_follow = checked

    def _display_latest(self, noodling_id: str, turn_number: int,
                        timestamp: float):
        """Display the latest trace for a noodling.

        Args:
            noodling_id: Which noodling to display
            turn_number: Turn number for the label
            timestamp: Unix timestamp for the label
        """
        entries = self._traces_by_noodling.get(noodling_id, [])
        if not entries:
            return

        latest = entries[-1]
        traces = latest['traces']

        # Update turn label
        dt = datetime.fromtimestamp(timestamp)
        self._turn_label.setText(
            f"Turn {turn_number} -- {dt.strftime('%H:%M:%S')}"
        )

        # Clear existing sections
        self._clear_sections()

        # Add new sections
        for trace in traces:
            section = CollapsibleFacetSection(trace)
            # Insert before the stretch
            self._sections_layout.insertWidget(
                self._sections_layout.count() - 1, section
            )

    def _clear_sections(self):
        """Remove all facet sections from the scroll area."""
        while self._sections_layout.count() > 1:  # Keep the stretch
            item = self._sections_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
