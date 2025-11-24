"""
Noodle Tuner Panel - Cognitive Manifold Debug UI

Real-time visualization and control of the cognitive manifold pipeline.
Shows individual transistor outputs, salience weights, and blended results.

"We cannot tune what we cannot see." - Cadet Caity

Phase 1 MVP: Read-only visualization with refresh button
Phase 2: Live editing with salience sliders and recalculate
Phase 3: Advanced features (PV import/export, session recording)

Author: Commander Spock + Cadet Caity
Date: November 23, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QScrollArea,
                             QSlider, QFrame, QGroupBox, QComboBox, QSplitter)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont
import requests
import sys
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock


class TransistorCard(QFrame):
    """
    Individual transistor display widget.

    Shows:
    - Transistor type
    - Output text
    - Salience slider
    """

    def __init__(self, transistor_data, parent=None, font_size=11):
        super().__init__(parent)
        self.transistor_data = transistor_data
        self.font_size = font_size
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { background-color: #2D2D2D; border: 1px solid #3E3E3E; padding: 6px; }")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Header: Type only
        type_label = QLabel(self.transistor_data['type'])
        type_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        type_label.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(type_label)

        # Output text
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(100)
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        self.output_text.setText(self.transistor_data['output'] or "(no output yet)")
        layout.addWidget(self.output_text)

        # Salience slider with value label
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Salience")
        slider_label.setStyleSheet("color: #999999; font-size: 8pt;")
        slider_layout.addWidget(slider_label)

        self.salience_slider = QSlider(Qt.Orientation.Horizontal)
        self.salience_slider.setRange(0, 100)
        self.salience_slider.setValue(int(self.transistor_data['salience'] * 100))
        self.salience_slider.valueChanged.connect(self.on_salience_changed)
        self.salience_slider.sliderPressed.connect(self.on_slider_pressed)
        self.salience_slider.sliderReleased.connect(self.on_slider_released)
        # Disable wheel events to prevent accidental changes during scrolling
        self.salience_slider.wheelEvent = lambda event: event.ignore()
        self.salience_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #3E3E3E;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 10px;
                margin: -3px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #AAAAAA;
            }
        """)
        slider_layout.addWidget(self.salience_slider)

        self.salience_value_label = QLabel(f"{self.transistor_data['salience']:.2f}")
        self.salience_value_label.setStyleSheet("color: #999999; font-size: 8pt;")
        self.salience_value_label.setMinimumWidth(30)
        slider_layout.addWidget(self.salience_value_label)

        layout.addLayout(slider_layout)

    def on_salience_changed(self, value):
        """Update salience label when slider moves."""
        salience = value / 100.0
        self.salience_value_label.setText(f"{salience:.2f}")

    def on_slider_pressed(self):
        """Pause auto-refresh when user starts dragging."""
        if hasattr(self.parent(), 'pause_refresh'):
            self.parent().pause_refresh()

    def on_slider_released(self):
        """Send new salience to API and resume refresh."""
        if hasattr(self.parent(), 'resume_refresh'):
            self.parent().resume_refresh()
        # TODO: Send updated salience to API

    def update_data(self, transistor_data):
        """Update widget with new transistor data."""
        self.transistor_data = transistor_data

        # Only update text if it changed (prevents deselection)
        new_text = transistor_data['output'] or "(no output yet)"
        if self.output_text.toPlainText() != new_text:
            self.output_text.setText(new_text)

        # Only update slider if not being dragged by user
        if not self.salience_slider.isSliderDown():
            new_salience = int(transistor_data['salience'] * 100)
            self.salience_slider.setValue(new_salience)
            self.salience_value_label.setText(f"{transistor_data['salience']:.2f}")

    def set_font_size(self, size):
        """Update font size for output text."""
        self.font_size = size
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)


class NoodleTunerPanel(MaximizableDock):
    """
    Noodle Tuner - Cognitive Manifold Debug UI.

    Phase 1 MVP: Visualize transistor outputs and manifold blend with refresh.
    """

    def __init__(self, parent=None):
        super().__init__("Noodle Tuner", parent)
        self.current_agent_id = None
        self.api_base = "http://localhost:8081/api"
        self.transistor_cards = {}  # type -> TransistorCard
        self.refresh_paused = False  # Track pause state
        self.font_size = 14  # Default font size (larger for readability)

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

        # Auto-refresh timer (1 second interval)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(1000)

    def pause_refresh(self):
        """Pause auto-refresh (when user is editing)."""
        self.refresh_paused = True

    def resume_refresh(self):
        """Resume auto-refresh."""
        self.refresh_paused = False
        self.refresh_data()  # Immediate refresh

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header with font controls
        header_layout = QHBoxLayout()
        self.agent_label = QLabel("No agent selected")
        self.agent_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.agent_label.setStyleSheet("color: #CCCCCC; padding: 6px;")
        header_layout.addWidget(self.agent_label)

        header_layout.addStretch()

        # Pause button for freezing cognitive processing
        self.pause_button = QPushButton("⏸ Pause Cognition")
        self.pause_button.setCheckable(True)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #3E3E3E;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px 8px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #CC6666;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
        """)
        self.pause_button.clicked.connect(self.toggle_pause_cognition)
        header_layout.addWidget(self.pause_button)

        # Font size controls
        font_label = QLabel("Font:")
        font_label.setStyleSheet("color: #888888; font-size: 9pt; margin-left: 10px;")
        header_layout.addWidget(font_label)

        decrease_btn = QPushButton("A-")
        decrease_btn.setMaximumWidth(40)
        decrease_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 2px;")
        decrease_btn.clicked.connect(self.decrease_font_size)
        header_layout.addWidget(decrease_btn)

        self.font_size_label = QLabel(f"{self.font_size}pt")
        self.font_size_label.setStyleSheet("color: #CCCCCC; font-size: 9pt; min-width: 30px;")
        header_layout.addWidget(self.font_size_label)

        increase_btn = QPushButton("A+")
        increase_btn.setMaximumWidth(40)
        increase_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 2px;")
        increase_btn.clicked.connect(self.increase_font_size)
        header_layout.addWidget(increase_btn)

        layout.addLayout(header_layout)

        # Response decision display (what the system decided to do)
        decision_group = QGroupBox("Response Decision (guides transistors)")
        decision_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")
        decision_layout = QVBoxLayout(decision_group)

        self.decision_display = QTextEdit()
        self.decision_display.setReadOnly(True)
        self.decision_display.setMaximumHeight(50)
        self.decision_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A3A;
                color: #AADDFF;
                border: 1px solid #5555AA;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
                font-weight: bold;
            }}
        """)
        decision_layout.addWidget(self.decision_display)

        layout.addWidget(decision_group)

        # Transistor outputs (resizable with splitters)
        transistor_group = QGroupBox("Transistor Outputs (before blend)")
        transistor_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #2D2D2D; }")

        # Use QSplitter for resizable cards
        self.transistor_splitter = QSplitter(Qt.Orientation.Vertical)
        self.transistor_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555555;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #888888;
            }
        """)

        scroll.setWidget(self.transistor_splitter)

        transistor_scroll_layout = QVBoxLayout(transistor_group)
        transistor_scroll_layout.addWidget(scroll)

        layout.addWidget(transistor_group, 1)  # Stretch factor 1

        # Manifold blend output
        blend_group = QGroupBox("Manifold Blend Output")
        blend_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")
        blend_layout = QVBoxLayout(blend_group)

        self.blend_display = QTextEdit()
        self.blend_display.setReadOnly(True)
        self.blend_display.setMaximumHeight(60)
        self.blend_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        blend_layout.addWidget(self.blend_display)

        layout.addWidget(blend_group)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt; padding: 4px;")
        layout.addWidget(self.status_label)

    @pyqtSlot(str)
    def set_agent(self, agent_id: str):
        """Set the agent to monitor."""
        self.current_agent_id = agent_id
        self.agent_label.setText(f"Tuning: {agent_id}")
        self.refresh_data()

    def refresh_data(self):
        """Fetch latest manifold data from API."""
        if not self.current_agent_id or self.refresh_paused:
            return

        try:
            url = f"{self.api_base}/manifold/debug/{self.current_agent_id}"
            response = requests.get(url, timeout=2)

            if response.status_code == 404:
                self.status_label.setText("Agent not found or has no manifold")
                self.status_label.setStyleSheet("color: #999999; font-size: 8pt; padding: 4px;")
                return

            if response.status_code != 200:
                self.status_label.setText(f"API error: {response.status_code}")
                self.status_label.setStyleSheet("color: #999999; font-size: 8pt; padding: 4px;")
                return

            data = response.json()

            # Update response decision display (only if changed)
            response_decision = data.get('response_decision')
            if response_decision:
                decision_text = f"📋 {response_decision['response_type'].upper()}: {response_decision['guidance']}\n\nReasoning: {response_decision.get('reasoning', 'N/A')}"
            else:
                decision_text = "(no decision yet - waiting for event)"

            if self.decision_display.toPlainText() != decision_text:
                self.decision_display.setText(decision_text)

            # Update transistor cards
            transistors = data.get('transistors', [])
            self.update_transistor_cards(transistors)

            # Update blend output (only if changed - prevents deselection)
            new_blend = data.get('blend_result', '(no output yet)')
            if self.blend_display.toPlainText() != new_blend:
                self.blend_display.setText(new_blend)

            # Update status
            self.status_label.setText(f"{len(transistors)} transistors • {data.get('blending_strategy', 'unknown')}")
            self.status_label.setStyleSheet("color: #999999; font-size: 8pt; padding: 4px;")

        except requests.exceptions.Timeout:
            self.status_label.setText("API timeout")
            self.status_label.setStyleSheet("color: #999999; font-size: 8pt; padding: 4px;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #999999; font-size: 8pt; padding: 4px;")

    def update_transistor_cards(self, transistors):
        """Update or create transistor cards."""
        # Remove old cards that no longer exist
        existing_types = {t['type'] for t in transistors}
        for ttype in list(self.transistor_cards.keys()):
            if ttype not in existing_types:
                card = self.transistor_cards.pop(ttype)
                card.deleteLater()

        # Update or create cards
        for transistor_data in transistors:
            ttype = transistor_data['type']
            if ttype in self.transistor_cards:
                # Update existing card
                self.transistor_cards[ttype].update_data(transistor_data)
            else:
                # Create new card with current font size
                card = TransistorCard(transistor_data, self, font_size=self.font_size)
                self.transistor_cards[ttype] = card
                self.transistor_splitter.addWidget(card)

    def increase_font_size(self):
        """Increase font size for all text displays."""
        self.font_size = min(24, self.font_size + 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self._update_all_font_sizes()

    def decrease_font_size(self):
        """Decrease font size for all text displays."""
        self.font_size = max(8, self.font_size - 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self._update_all_font_sizes()

    def _update_all_font_sizes(self):
        """Update font size for all text widgets."""
        # Update decision display
        self.decision_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A3A;
                color: #AADDFF;
                border: 1px solid #5555AA;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
                font-weight: bold;
            }}
        """)

        # Update blend display
        self.blend_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)

        # Update all transistor cards
        for card in self.transistor_cards.values():
            card.set_font_size(self.font_size)

    def toggle_pause_cognition(self, checked):
        """Toggle cognitive processing pause for all agents."""
        try:
            url = f"{self.api_base}/cognition/pause"
            response = requests.post(url, json={'paused': checked}, timeout=2)

            if response.status_code == 200:
                if checked:
                    self.pause_button.setText("▶ Resume Cognition")
                    self.status_label.setText("⏸ Cognition PAUSED - agents frozen")
                    self.status_label.setStyleSheet("color: #CC6666; font-size: 8pt; padding: 4px; font-weight: bold;")
                else:
                    self.pause_button.setText("⏸ Pause Cognition")
                    self.status_label.setText("▶ Cognition RESUMED")
                    self.status_label.setStyleSheet("color: #66CC66; font-size: 8pt; padding: 4px; font-weight: bold;")
            else:
                self.status_label.setText(f"Pause failed: {response.status_code}")
                self.pause_button.setChecked(not checked)  # Revert button state

        except Exception as e:
            self.status_label.setText(f"Pause error: {str(e)}")
            self.pause_button.setChecked(not checked)  # Revert button state
