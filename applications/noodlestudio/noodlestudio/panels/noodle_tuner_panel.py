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
    - Transistor type and icon
    - Output text
    - Salience slider
    - Enable/disable toggle
    """

    # Icon mapping for transistor types
    ICONS = {
        'SomaticCognitiveTransistor': '🦆',
        'DeceptionTransistor': '🎭',
        'PersonalityTransistor': '🧠',
        'CulturalTransistor': '🌍',
        'IntuitionTransistor': '💫',
        'MoodTransistor': '😊'
    }

    def __init__(self, transistor_data, parent=None):
        super().__init__(parent)
        self.transistor_data = transistor_data
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setStyleSheet("QFrame { background-color: #2D2D2D; border: 2px solid #555; border-radius: 4px; padding: 8px; }")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header: [Icon] Type (salience)
        header_layout = QHBoxLayout()

        icon = self.ICONS.get(self.transistor_data['type'], '⚙️')
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 16))
        header_layout.addWidget(icon_label)

        type_label = QLabel(f"[{self.transistor_data['type']}]")
        type_label.setFont(QFont("Courier", 10, QFont.Weight.Bold))
        type_label.setStyleSheet("color: #4FC3F7;")
        header_layout.addWidget(type_label)

        header_layout.addStretch()

        self.salience_label = QLabel(f"({self.transistor_data['salience']:.2f})")
        self.salience_label.setFont(QFont("Courier", 10))
        self.salience_label.setStyleSheet("color: #FFA726;")
        header_layout.addWidget(self.salience_label)

        layout.addLayout(header_layout)

        # Output text
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(80)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: 1px solid #444;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        self.output_text.setText(self.transistor_data['output'] or "(no output yet)")
        layout.addWidget(self.output_text)

        # Salience slider (Phase 2 - currently read-only display)
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Salience:")
        slider_label.setStyleSheet("color: #D2D2D2;")
        slider_layout.addWidget(slider_label)

        self.salience_slider = QSlider(Qt.Orientation.Horizontal)
        self.salience_slider.setRange(0, 100)
        self.salience_slider.setValue(int(self.transistor_data['salience'] * 100))
        self.salience_slider.setEnabled(False)  # Phase 1: Read-only
        self.salience_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4FC3F7;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        slider_layout.addWidget(self.salience_slider)

        layout.addLayout(slider_layout)

    def update_data(self, transistor_data):
        """Update widget with new transistor data."""
        self.transistor_data = transistor_data
        self.output_text.setText(transistor_data['output'] or "(no output yet)")
        self.salience_label.setText(f"({transistor_data['salience']:.2f})")
        self.salience_slider.setValue(int(transistor_data['salience'] * 100))


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

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

        # Auto-refresh timer (1 second interval)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(1000)

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        self.agent_label = QLabel("No agent selected")
        self.agent_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.agent_label.setStyleSheet("color: #4FC3F7; padding: 8px;")
        layout.addWidget(self.agent_label)

        # Input perception display
        input_group = QGroupBox("Input Perception")
        input_group.setStyleSheet("QGroupBox { color: #D2D2D2; font-weight: bold; border: 1px solid #555; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 3px; }")
        input_layout = QVBoxLayout(input_group)

        self.input_display = QTextEdit()
        self.input_display.setReadOnly(True)
        self.input_display.setMaximumHeight(60)
        self.input_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFF59D;
                border: 1px solid #444;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
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
        self.blend_display.setMaximumHeight(80)
        self.blend_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #81C784;
                border: 1px solid #444;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                font-weight: bold;
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
        if not self.current_agent_id:
            return

        try:
            url = f"{self.api_base}/manifold/debug/{self.current_agent_id}"
            response = requests.get(url, timeout=2)

            if response.status_code == 404:
                self.status_label.setText("Agent not found or has no manifold")
                self.status_label.setStyleSheet("color: #F44336; font-size: 9pt; padding: 4px;")
                return

            if response.status_code != 200:
                self.status_label.setText(f"API error: {response.status_code}")
                self.status_label.setStyleSheet("color: #F44336; font-size: 9pt; padding: 4px;")
                return

            data = response.json()

            # Update input
            self.input_display.setText(data.get('input', '(no input yet)'))

            # Update transistor cards
            transistors = data.get('transistors', [])
            self.update_transistor_cards(transistors)

            # Update blend output
            self.blend_display.setText(data.get('blend_result', '(no output yet)'))

            # Update status
            self.status_label.setText(f"Updated: {len(transistors)} transistors • {data.get('blending_strategy', 'unknown')} strategy")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 9pt; padding: 4px;")

        except requests.exceptions.Timeout:
            self.status_label.setText("API timeout")
            self.status_label.setStyleSheet("color: #FF9800; font-size: 9pt; padding: 4px;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #F44336; font-size: 9pt; padding: 4px;")

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
