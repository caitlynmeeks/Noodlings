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
                             QSlider, QFrame, QGroupBox, QComboBox)
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

    def __init__(self, transistor_data, parent=None):
        super().__init__(parent)
        self.transistor_data = transistor_data
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
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: 11pt;
                padding: 4px;
            }
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

        # Header
        self.agent_label = QLabel("No agent selected")
        self.agent_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.agent_label.setStyleSheet("color: #CCCCCC; padding: 6px;")
        layout.addWidget(self.agent_label)

        # Input perception display
        input_group = QGroupBox("Input Perception")
        input_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")
        input_layout = QVBoxLayout(input_group)

        self.input_display = QTextEdit()
        self.input_display.setReadOnly(True)
        self.input_display.setMaximumHeight(50)
        self.input_display.setStyleSheet("""
            QTextEdit {
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                padding: 4px;
            }
        """)
        input_layout.addWidget(self.input_display)

        layout.addWidget(input_group)

        # Transistor outputs (scrollable)
        transistor_group = QGroupBox("Transistor Outputs (before blend)")
        transistor_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #2D2D2D; }")

        self.transistor_container = QWidget()
        self.transistor_layout = QVBoxLayout(self.transistor_container)
        self.transistor_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.transistor_container)

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
        self.blend_display.setStyleSheet("""
            QTextEdit {
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #3E3E3E;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                padding: 4px;
            }
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

            # Update input (only if changed - prevents deselection)
            new_input = data.get('input', '(no input yet)')
            if self.input_display.toPlainText() != new_input:
                self.input_display.setText(new_input)

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
                self.transistor_layout.removeWidget(card)
                card.deleteLater()

        # Update or create cards
        for transistor_data in transistors:
            ttype = transistor_data['type']
            if ttype in self.transistor_cards:
                # Update existing card
                self.transistor_cards[ttype].update_data(transistor_data)
            else:
                # Create new card
                card = TransistorCard(transistor_data, self)
                self.transistor_cards[ttype] = card
                self.transistor_layout.addWidget(card)

        # Add stretch at end
        self.transistor_layout.addStretch()
