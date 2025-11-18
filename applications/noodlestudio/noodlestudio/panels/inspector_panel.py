"""
Inspector Panel - Unity-style property editor

Shows and edits ALL properties of selected entity:
- Users: name, description, location, inventory
- Noodlings: name, species, description, LLM model, personality traits
- Objects: name, description, properties
- Rooms: name, description, exits

Every atom of noodleMUSH exposed and editable!

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QFormLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QScrollArea,
                             QSpinBox, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
import requests


class InspectorPanel(QDockWidget):
    """
    Unity-style Inspector panel.

    Shows editable properties for selected entity.
    Like Unity's Inspector - every field is live-editable!
    """

    def __init__(self, parent=None):
        super().__init__("Inspector", parent)
        self.current_entity = None
        self.api_base = "http://localhost:8081/api"

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        self.entity_header = QLabel("No entity selected")
        self.entity_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.entity_header.setStyleSheet("color: #D2D2D2; padding: 8px;")
        layout.addWidget(self.entity_header)

        # Scrollable properties area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.properties_widget = QWidget()
        self.properties_layout = QVBoxLayout(self.properties_widget)
        self.properties_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.properties_widget)
        layout.addWidget(scroll)

        # Save button
        self.save_button = QPushButton("Apply Changes")
        self.save_button.clicked.connect(self.save_changes)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

    @pyqtSlot(str, dict)
    def load_entity(self, entity_type: str, entity_data: dict):
        """Load entity properties into inspector."""
        self.current_entity = (entity_type, entity_data)

        # Clear existing properties
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Update header
        if entity_type == 'noodling':
            name = entity_data.get('data', {}).get('name', entity_data.get('id'))
            species = entity_data.get('data', {}).get('species', 'unknown')
            self.entity_header.setText(f"Noodling: {name} ({species})")
            self.load_noodling_properties(entity_data)

        elif entity_type == 'user':
            self.entity_header.setText("User: caity")
            self.load_user_properties(entity_data)

        elif entity_type == 'object':
            obj_name = entity_data.get('id', 'Unknown Object')
            self.entity_header.setText(f"Object: {obj_name}")
            self.load_object_properties(entity_data)

        elif entity_type == 'exit':
            direction = entity_data.get('direction', 'unknown')
            self.entity_header.setText(f"Exit: {direction}")
            self.load_exit_properties(entity_data)

        self.save_button.setEnabled(True)

    def load_noodling_properties(self, entity_data):
        """Show Noodling properties (FULL CONTROL!)."""
        agent = entity_data.get('data', {})

        # Transform group
        transform_group = self.create_property_group("Transform")
        self.add_text_field(transform_group, "Room", agent.get('room', 'room_000'))
        self.properties_layout.addWidget(transform_group)

        # Identity group
        identity_group = self.create_property_group("Identity")
        self.add_text_field(identity_group, "Name", agent.get('name', ''))
        self.add_text_field(identity_group, "Species", agent.get('species', ''))
        # TODO: Load description from agent state files
        self.add_text_area(identity_group, "Description", "A consciousness agent...")
        self.properties_layout.addWidget(identity_group)

        # LLM Configuration group
        llm_group = self.create_property_group("LLM Configuration")
        self.add_text_field(llm_group, "Provider", agent.get('llm_provider') or '(global)')
        self.add_text_field(llm_group, "Model", agent.get('llm_model') or '(global default)')
        self.properties_layout.addWidget(llm_group)

        # Personality group (TODO: Load from config)
        personality_group = self.create_property_group("Personality Traits")
        self.add_slider_field(personality_group, "Extraversion", 0.5, 0.0, 1.0)
        self.add_slider_field(personality_group, "Curiosity", 0.65, 0.0, 1.0)
        self.add_slider_field(personality_group, "Spontaneity", 0.75, 0.0, 1.0)
        self.properties_layout.addWidget(personality_group)

        self.properties_layout.addStretch()

    def load_user_properties(self, entity_data):
        """Show user properties."""
        user_group = self.create_property_group("User Info")
        self.add_text_field(user_group, "Username", "caity")
        self.add_text_field(user_group, "Type", "Noodler (human)")
        self.add_text_field(user_group, "Age", "9 years old")
        self.add_text_field(user_group, "Pronouns", "she/her")
        self.properties_layout.addWidget(user_group)

        inventory_group = self.create_property_group("Inventory")
        self.add_text_field(inventory_group, "Item 1", "Wooden sword")
        self.add_text_field(inventory_group, "Item 2", "Atomic fireball candy")
        self.properties_layout.addWidget(inventory_group)

        self.properties_layout.addStretch()

    def load_object_properties(self, entity_data):
        """Show object properties."""
        obj_group = self.create_property_group("Object Properties")
        obj_id = entity_data.get('id', '')
        self.add_text_field(obj_group, "ID", obj_id)
        self.add_text_field(obj_group, "Name", obj_id.replace('obj_', '').replace('_', ' ').title())
        self.add_text_area(obj_group, "Description", "An object in the world...")
        self.properties_layout.addWidget(obj_group)

        self.properties_layout.addStretch()

    def load_exit_properties(self, entity_data):
        """Show exit properties."""
        exit_group = self.create_property_group("Exit Info")
        self.add_text_field(exit_group, "Direction", entity_data.get('direction', ''))
        self.add_text_field(exit_group, "Destination", entity_data.get('destination', ''))
        self.properties_layout.addWidget(exit_group)

        self.properties_layout.addStretch()

    def create_property_group(self, title: str) -> QGroupBox:
        """Create collapsible property group (Unity style)."""
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        group.setStyleSheet("""
            QGroupBox {
                color: #D2D2D2;
                border: 1px solid #1E1E1E;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        group.setLayout(QFormLayout())
        return group

    def add_text_field(self, group: QGroupBox, label: str, value: str):
        """Add editable text field to group."""
        field = QLineEdit(value)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        group.layout().addRow(f"{label}:", field)

    def add_text_area(self, group: QGroupBox, label: str, value: str):
        """Add editable text area to group."""
        field = QTextEdit()
        field.setPlainText(value)
        field.setMaximumHeight(100)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        group.layout().addRow(f"{label}:", field)

    def add_slider_field(self, group: QGroupBox, label: str, value: float, min_val: float, max_val: float):
        """Add slider + numeric field."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2;")
        group.layout().addRow(f"{label}:", spin)

    def save_changes(self):
        """Save edited properties back to noodleMUSH."""
        # TODO: Implement save to API
        print("Saving changes to noodleMUSH...")
        self.save_button.setEnabled(False)
