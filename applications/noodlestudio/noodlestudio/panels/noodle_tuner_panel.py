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
                             QSlider, QFrame, QGroupBox, QComboBox, QSplitter,
                             QFileDialog, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSettings, QEvent
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
import requests
import sys
import json
import logging
from datetime import datetime
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock

logger = logging.getLogger(__name__)


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

        # Instruction Prompt (always editable)
        instruction_label = QLabel("Instruction Prompt (editable):")
        instruction_label.setStyleSheet("color: #CCCCCC; font-size: 10pt; font-weight: bold;")
        layout.addWidget(instruction_label)

        self.instruction_text = QTextEdit()
        self.instruction_text.setMinimumHeight(200)
        self.instruction_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.instruction_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        self.instruction_text.setText(self.transistor_data.get('instruction_prompt', ''))
        layout.addWidget(self.instruction_text, 1)  # Stretch factor 1

        # Output text (editable when paused)
        output_label = QLabel("Output (edit when paused):")
        output_label.setStyleSheet("color: #CCCCCC; font-size: 10pt; font-weight: bold;")
        layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)  # Read-only by default, editable when paused
        self.output_text.setMinimumHeight(200)
        self.output_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        self.output_text.setText(self.transistor_data['output'] or "(no output yet)")
        layout.addWidget(self.output_text, 1)  # Stretch factor 1 so it expands

        # Salience slider with value label (compact layout)
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(6)
        slider_layout.setContentsMargins(0, 0, 0, 0)

        slider_label = QLabel("Salience")
        slider_label.setStyleSheet("color: #999999;")
        slider_label.setFrameStyle(QFrame.Shape.NoFrame)
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
        self.salience_value_label.setStyleSheet("color: #999999;")
        self.salience_value_label.setFrameStyle(QFrame.Shape.NoFrame)
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
        """Send new salience to API (but don't resume refresh if cognition is paused)."""
        # Send updated salience to API
        if hasattr(self.parent(), 'send_salience_update'):
            uuid_str = self.transistor_data.get('uuid')
            salience = self.salience_slider.value() / 100.0
            self.parent().send_salience_update(uuid_str, salience)

        # Only resume auto-refresh if cognition is not paused
        if hasattr(self.parent(), 'resume_refresh') and hasattr(self.parent(), 'pause_button'):
            if not self.parent().pause_button.isChecked():
                self.parent().resume_refresh()

    def update_data(self, transistor_data):
        """Update widget with new transistor data."""
        self.transistor_data = transistor_data

        # Update instruction prompt if changed
        new_instruction = transistor_data.get('instruction_prompt', '')
        if self.instruction_text.toPlainText() != new_instruction:
            self.instruction_text.setText(new_instruction)

        # Only update output text if it changed (prevents deselection)
        new_text = transistor_data['output'] or "(no output yet)"
        if self.output_text.toPlainText() != new_text:
            self.output_text.setText(new_text)

        # Only update slider if not being dragged by user
        if not self.salience_slider.isSliderDown():
            new_salience = int(transistor_data['salience'] * 100)
            self.salience_slider.setValue(new_salience)
            self.salience_value_label.setText(f"{transistor_data['salience']:.2f}")

    def set_output_editable(self, editable: bool):
        """Enable/disable output editing (for pause/resume cognition)."""
        self.output_text.setReadOnly(not editable)
        if editable:
            self.output_text.setStyleSheet(f"""
                QTextEdit {{
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    border: 2px solid #888888;
                    font-family: 'Courier New', monospace;
                    font-size: {self.font_size}pt;
                    padding: 4px;
                }}
            """)
        else:
            self.output_text.setStyleSheet(f"""
                QTextEdit {{
                    background-color: #222222;
                    color: #CCCCCC;
                    border: 1px solid #555555;
                    font-family: 'Courier New', monospace;
                    font-size: {self.font_size}pt;
                    padding: 4px;
                }}
            """)

    def get_edited_data(self) -> dict:
        """Get current edited values from card."""
        return {
            'uuid': self.transistor_data.get('uuid'),
            'type': self.transistor_data['type'],
            'instruction_prompt': self.instruction_text.toPlainText(),
            'output': self.output_text.toPlainText(),
            'salience': self.salience_slider.value() / 100.0
        }

    def set_font_size(self, size):
        """Update font size for both text fields."""
        self.font_size = size
        self.instruction_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
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

        # Load saved font size or use default
        self.settings = QSettings('NoodleStudio', 'NoodleTuner')
        self.font_size = self.settings.value('font_size', 14, type=int)

        # Store current snapshot data for export
        self.current_snapshot = {}

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

        # Auto-refresh timer (1 second interval)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(1000)

        # Keyboard shortcuts
        self.shortcut_increase = QShortcut(QKeySequence(Qt.Key.Key_Plus), self)
        self.shortcut_increase.activated.connect(self.increase_font_size)

        self.shortcut_increase_alt = QShortcut(QKeySequence(Qt.Key.Key_Equal), self)  # = key (same as + without shift)
        self.shortcut_increase_alt.activated.connect(self.increase_font_size)

        self.shortcut_decrease = QShortcut(QKeySequence(Qt.Key.Key_Minus), self)
        self.shortcut_decrease.activated.connect(self.decrease_font_size)

    def pause_refresh(self):
        """Pause auto-refresh (when user is editing)."""
        self.refresh_paused = True

    def resume_refresh(self):
        """Resume auto-refresh."""
        self.refresh_paused = False
        self.refresh_data()  # Immediate refresh

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with font controls
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 8)
        self.agent_label = QLabel("No agent selected")
        self.agent_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.agent_label.setStyleSheet("color: #CCCCCC;")
        header_layout.addWidget(self.agent_label)

        header_layout.addStretch()

        # Pause button
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

        # Export/Import buttons
        export_btn = QPushButton("↓ Export .tuner")
        export_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 4px 8px; margin-left: 10px;")
        export_btn.clicked.connect(self.export_snapshot)
        header_layout.addWidget(export_btn)

        import_btn = QPushButton("↑ Import .tuner")
        import_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 4px 8px;")
        import_btn.clicked.connect(self.import_snapshot)
        header_layout.addWidget(import_btn)

        copy_btn = QPushButton("Copy All")
        copy_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 4px 8px;")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        header_layout.addWidget(copy_btn)

        layout.addLayout(header_layout)

        # Single scroll area for everything - NO NESTED SCROLLING
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1E1E1E;
                border: none;
            }
            QScrollBar:vertical {
                background: #2D2D2D;
                width: 16px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #666666;
                min-height: 40px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical:hover {
                background: #888888;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Content widget (all sections go here)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(4, 0, 4, 4)
        content_layout.setSpacing(4)

        # Raw Input
        raw_input_group = QGroupBox("Raw Input (perception)")
        raw_input_group.setStyleSheet("""
            QGroupBox {
                color: #D2D2D2;
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 4px;
                background-color: #252525;
            }
        """)
        raw_input_layout = QVBoxLayout(raw_input_group)
        raw_input_layout.setContentsMargins(2, 2, 2, 2)

        self.raw_input_display = QTextEdit()
        self.raw_input_display.setReadOnly(True)
        self.raw_input_display.setMinimumHeight(200)
        self.raw_input_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        raw_input_layout.addWidget(self.raw_input_display)
        content_layout.addWidget(raw_input_group)

        # Response Decision
        decision_group = QGroupBox("Response Decision")
        decision_group.setStyleSheet("""
            QGroupBox {
                color: #D2D2D2;
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 4px;
                background-color: #252525;
            }
        """)
        decision_layout = QVBoxLayout(decision_group)
        decision_layout.setContentsMargins(2, 2, 2, 2)

        self.decision_display = QTextEdit()
        self.decision_display.setReadOnly(True)
        self.decision_display.setMinimumHeight(200)
        self.decision_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        decision_layout.addWidget(self.decision_display)
        content_layout.addWidget(decision_group)

        # Transistor Outputs
        transistor_group = QGroupBox("Transistor Outputs")
        transistor_group.setStyleSheet("""
            QGroupBox {
                color: #D2D2D2;
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 4px;
                background-color: #252525;
            }
        """)
        transistor_layout = QVBoxLayout(transistor_group)
        transistor_layout.setContentsMargins(2, 2, 2, 2)

        # Simple vertical layout for transistor cards - NO SPLITTER, NO NESTED SCROLL
        self.transistor_splitter = QWidget()
        self.transistor_splitter_layout = QVBoxLayout(self.transistor_splitter)
        self.transistor_splitter_layout.setContentsMargins(0, 0, 0, 0)
        self.transistor_splitter_layout.setSpacing(4)

        transistor_layout.addWidget(self.transistor_splitter)
        content_layout.addWidget(transistor_group)

        # Manifold Blend Output
        blend_group = QGroupBox("Manifold Blend Output")
        blend_group.setStyleSheet("""
            QGroupBox {
                color: #D2D2D2;
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 4px;
                background-color: #252525;
            }
        """)
        blend_layout = QVBoxLayout(blend_group)
        blend_layout.setContentsMargins(2, 2, 2, 2)

        # Manifold Instruction Prompt
        manifold_instruction_label = QLabel("Manifold Instruction Prompt:")
        manifold_instruction_label.setStyleSheet("color: #CCCCCC; font-size: 10pt; font-weight: bold;")
        blend_layout.addWidget(manifold_instruction_label)

        self.manifold_instruction_display = QTextEdit()
        self.manifold_instruction_display.setReadOnly(True)
        self.manifold_instruction_display.setMinimumHeight(200)
        self.manifold_instruction_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)
        blend_layout.addWidget(self.manifold_instruction_display)

        # Manifold Output
        manifold_output_label = QLabel("Manifold Output:")
        manifold_output_label.setStyleSheet("color: #CCCCCC; font-size: 10pt; font-weight: bold;")
        blend_layout.addWidget(manifold_output_label)

        self.blend_display = QTextEdit()
        self.blend_display.setReadOnly(True)
        self.blend_display.setMinimumHeight(200)
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
        content_layout.addWidget(blend_group)

        content_layout.addStretch()
        main_scroll.setWidget(content_widget)
        layout.addWidget(main_scroll)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt; padding: 4px; background-color: #1E1E1E;")
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

            # Store snapshot data for export with metadata
            from datetime import datetime
            self.current_snapshot = data
            self.current_snapshot['metadata'] = {
                'agent_name': data.get('agent_name', 'Unknown'),
                'agent_id': data.get('agent_id', 'Unknown'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Update raw input display (only if changed)
            raw_input_text = data.get('input', '(no input yet)')
            if self.raw_input_display.toPlainText() != raw_input_text:
                self.raw_input_display.setText(raw_input_text)

            # Update response decision display (only if changed)
            response_decision = data.get('response_decision')
            if response_decision and isinstance(response_decision, dict):
                resp_type = response_decision.get('response_type', 'UNKNOWN').upper()
                guidance = response_decision.get('guidance', 'N/A')
                reasoning = response_decision.get('reasoning', 'N/A')

                # Show all response types clearly
                if resp_type == 'NONE':
                    decision_text = f"DECIDED NOT TO RESPOND EXTERNALLY\n\nReasoning: {reasoning}"
                elif resp_type == 'THINK':
                    decision_text = f"RUMINATION (THINK)\n\nGuidance: {guidance}\nReasoning: {reasoning}"
                else:
                    decision_text = f"{resp_type}\n\nGuidance: {guidance}\nReasoning: {reasoning}"
            else:
                decision_text = "No response decision available\n\n(Waiting for cognition cycle)"

            if self.decision_display.toPlainText() != decision_text:
                self.decision_display.setText(decision_text)

            # Update transistor cards
            transistors = data.get('transistors', [])
            self.update_transistor_cards(transistors)

            # Update manifold instruction prompt
            manifold_instruction = data.get('manifold_instruction_prompt', '(no instruction prompt available)')
            if self.manifold_instruction_display.toPlainText() != manifold_instruction:
                self.manifold_instruction_display.setText(manifold_instruction)

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
                self.transistor_splitter_layout.addWidget(card)

    def increase_font_size(self):
        """Increase font size for all text displays."""
        self.font_size = min(24, self.font_size + 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self.settings.setValue('font_size', self.font_size)
        self._update_all_font_sizes()

    def decrease_font_size(self):
        """Decrease font size for all text displays."""
        self.font_size = max(8, self.font_size - 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self.settings.setValue('font_size', self.font_size)
        self._update_all_font_sizes()

    def _update_all_font_sizes(self):
        """Update font size for all text widgets."""
        # Update raw input display
        self.raw_input_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A3A2A;
                color: #AAFFAA;
                border: 1px solid #55AA55;
                font-family: 'Courier New', monospace;
                font-size: {self.font_size}pt;
                padding: 4px;
            }}
        """)

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
            if checked:
                # PAUSING: Request pause and wait for cycle completion
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={'paused': True}, timeout=35)

                if response.status_code == 200:
                    # Pause API now waits for cycle completion internally
                    # Enable output editing immediately
                    for card in self.transistor_cards.values():
                        card.set_output_editable(True)

                    self.pause_button.setText("▶ Resume Cognition")
                    self.status_label.setText("⏸ Cycle complete - outputs editable")
                    self.status_label.setStyleSheet("color: #CC6666; font-size: 8pt; padding: 4px; font-weight: bold;")

                    # Refresh data to get final cycle state
                    self.refresh_data()
                else:
                    self.status_label.setText(f"Pause failed: {response.status_code}")
                    self.pause_button.setChecked(False)

            else:
                # RESUMING: Apply edits and resume cognition
                self._apply_edited_values()

                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={'paused': False}, timeout=2)

                if response.status_code == 200:
                    # Disable output editing
                    for card in self.transistor_cards.values():
                        card.set_output_editable(False)

                    self.pause_button.setText("⏸ Pause Cognition")
                    self.status_label.setText("▶ Cognition RESUMED - edits applied")
                    self.status_label.setStyleSheet("color: #66CC66; font-size: 8pt; padding: 4px; font-weight: bold;")
                else:
                    self.status_label.setText(f"Resume failed: {response.status_code}")
                    self.pause_button.setChecked(True)

        except Exception as e:
            self.status_label.setText(f"Pause error: {str(e)}")
            self.pause_button.setChecked(not checked)  # Revert button state

    def send_salience_update(self, uuid_str: str, salience: float):
        """Send single salience update to API immediately."""
        if not self.current_agent_id or not uuid_str:
            return

        try:
            # Send just this one transistor's salience
            url = f"{self.api_base}/manifold/update/{self.current_agent_id}"
            response = requests.post(url, json={
                'transistors': [{
                    'uuid': uuid_str,
                    'salience': salience
                }]
            }, timeout=2)

            if response.status_code == 200:
                logger.info(f"Updated salience for {uuid_str} to {salience:.2f}")
            else:
                logger.warning(f"Failed to update salience: {response.status_code}")

        except Exception as e:
            logger.error(f"Error updating salience: {e}")

    def _apply_edited_values(self):
        """Apply edited transistor values when resuming cognition."""
        if not self.current_agent_id:
            return

        try:
            # Collect edited data from all cards
            edited_transistors = []
            for card in self.transistor_cards.values():
                edited_transistors.append(card.get_edited_data())

            # Send to API
            url = f"{self.api_base}/manifold/update/{self.current_agent_id}"
            response = requests.post(url, json={'transistors': edited_transistors}, timeout=2)

            if response.status_code == 200:
                logger.info(f"Applied edited transistor values for {self.current_agent_id}")
            else:
                logger.warning(f"Failed to apply edits: {response.status_code}")

        except Exception as e:
            logger.error(f"Error applying edited values: {e}")

    def export_snapshot(self):
        """Export current cognitive state snapshot to .tuner file."""
        if not self.current_agent_id:
            QMessageBox.warning(self, "Export Failed", "No agent selected")
            return

        if not self.current_snapshot:
            QMessageBox.warning(self, "Export Failed", "No snapshot data available yet")
            return

        # Get agent name for filename
        agent_name = self.current_agent_id.replace('agent_', '').replace('_', ' ').title()
        default_filename = f"{agent_name}_{int(datetime.now().timestamp())}.tuner"

        # Open save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cognitive Snapshot",
            default_filename,
            "Tuner Files (*.tuner);;JSON Files (*.json)"
        )

        if not filename:
            return

        try:
            # Fetch phenomenal state and predicted affect
            state_url = f"{self.api_base}/agents/{self.current_agent_id}/state"
            state_response = requests.get(state_url, timeout=2)
            state_data = state_response.json() if state_response.status_code == 200 else {}

            # Build snapshot according to spec
            snapshot = {
                "metadata": {
                    "agent_id": self.current_agent_id,
                    "agent_name": agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "noodletuner_version": "1.0"
                },
                "perception": {
                    "input_text": self.current_snapshot.get('last_input_text', ''),
                    "manifold_output": self.current_snapshot.get('blend_result', '')
                },
                "response_decision": self.current_snapshot.get('response_decision', {}),
                "transistors": self.current_snapshot.get('transistors', []),
                "phenomenal_state": state_data.get('phenomenal_state', {}),
                "predicted_affect": state_data.get('predicted_affect', {})
            }

            # Write to file
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2)

            self.status_label.setText(f"Exported to {filename}")
            self.status_label.setStyleSheet("color: #66CC66; font-size: 8pt; padding: 4px;")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting snapshot:\n{str(e)}")
            self.status_label.setText(f"Export failed: {str(e)}")

    def import_snapshot(self):
        """Import cognitive state snapshot from .tuner file."""
        # Open file dialog
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Cognitive Snapshot",
            "",
            "Tuner Files (*.tuner);;JSON Files (*.json)"
        )

        if not filename:
            return

        try:
            # Load snapshot from file
            with open(filename, 'r') as f:
                snapshot = json.load(f)

            # Validate snapshot structure
            if not isinstance(snapshot, dict) or 'metadata' not in snapshot:
                QMessageBox.critical(self, "Import Failed", "Invalid .tuner file format")
                return

            # Display imported data (read-only view)
            self._display_imported_snapshot(snapshot)

            self.status_label.setText(f"Imported from {filename}")
            self.status_label.setStyleSheet("color: #66CC66; font-size: 8pt; padding: 4px;")

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Import Failed", f"Invalid JSON:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error importing snapshot:\n{str(e)}")

    def _display_imported_snapshot(self, snapshot):
        """Display imported snapshot data in the UI (read-only)."""
        # Pause auto-refresh while viewing imported data
        self.pause_refresh()

        # Update agent label
        metadata = snapshot.get('metadata', {})
        agent_name = metadata.get('agent_name', 'Unknown')
        timestamp = metadata.get('timestamp', 'Unknown')
        self.agent_label.setText(f"Viewing Snapshot: {agent_name} @ {timestamp[:19]}")

        # Update response decision
        response_decision = snapshot.get('response_decision', {})
        if response_decision:
            decision_text = f"📋 {response_decision.get('response_type', 'UNKNOWN').upper()}: {response_decision.get('guidance', 'N/A')}\n\nReasoning: {response_decision.get('reasoning', 'N/A')}"
        else:
            decision_text = "(no decision in snapshot)"
        self.decision_display.setText(decision_text)

        # Update transistor cards
        transistors = snapshot.get('transistors', [])
        self.update_transistor_cards(transistors)

        # Update blend output
        perception = snapshot.get('perception', {})
        blend_text = perception.get('manifold_output', '(no output in snapshot)')
        self.blend_display.setText(blend_text)

        # Update status
        self.status_label.setText(f"📂 Viewing imported snapshot (auto-refresh paused)")
        self.status_label.setStyleSheet("color: #FFAA66; font-size: 8pt; padding: 4px; font-weight: bold;")

    def copy_to_clipboard(self):
        """Copy current state to clipboard with input and manifold output."""
        if not self.current_snapshot:
            QMessageBox.warning(self, "No Data", "No cognitive state to copy. Wait for data to load.")
            return

        try:
            # Build comprehensive text output
            lines = []
            lines.append("=" * 80)
            lines.append("COGNITIVE MANIFOLD STATE")
            lines.append("=" * 80)
            lines.append("")

            # Agent info
            metadata = self.current_snapshot.get('metadata', {})
            lines.append(f"Agent: {metadata.get('agent_name', 'Unknown')}")
            lines.append(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
            lines.append("")

            # Raw Input
            lines.append("-" * 80)
            lines.append("RAW INPUT (perception):")
            lines.append("-" * 80)
            lines.append(self.current_snapshot.get('input', '(no input)'))
            lines.append("")

            # Response Decision
            lines.append("-" * 80)
            lines.append("RESPONSE DECISION:")
            lines.append("-" * 80)
            response_decision = self.current_snapshot.get('response_decision')
            if response_decision:
                lines.append(f"Type: {response_decision.get('response_type', 'unknown').upper()}")
                lines.append(f"Guidance: {response_decision.get('guidance', 'N/A')}")
                lines.append(f"Reasoning: {response_decision.get('reasoning', 'N/A')}")
            else:
                lines.append("RUMINATION CYCLE (internal thought, no external response decision)")
            lines.append("")

            # Transistor Outputs
            lines.append("-" * 80)
            lines.append("TRANSISTOR OUTPUTS:")
            lines.append("-" * 80)
            transistors = self.current_snapshot.get('transistors', [])
            for t in transistors:
                lines.append(f"\n[{t['type']}] (salience: {t['salience']:.2f})")
                lines.append(f"Output: {t['output']}")
            lines.append("")

            # Manifold Blend Output
            lines.append("-" * 80)
            lines.append("MANIFOLD BLEND OUTPUT:")
            lines.append("-" * 80)
            lines.append(self.current_snapshot.get('blend_result', '(no output)'))
            lines.append("")

            lines.append("=" * 80)

            # Copy to clipboard
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText('\n'.join(lines))

            self.status_label.setText("Copied to clipboard!")
            self.status_label.setStyleSheet("color: #66CC66; font-size: 8pt; padding: 4px;")

        except Exception as e:
            QMessageBox.critical(self, "Copy Failed", f"Error copying to clipboard:\n{str(e)}")
