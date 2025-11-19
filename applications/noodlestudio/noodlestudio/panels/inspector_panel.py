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

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QScrollArea,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QProgressBar, QListWidget,
                             QFileDialog, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon
import requests
import yaml
from pathlib import Path
import sys
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock


class InspectorPanel(MaximizableDock):
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

        # Live update timer for Noodle Component
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_data)
        self.update_timer.start(1000)  # Update every second

        self.live_affect_labels = {}
        self.live_phenomenal_label = None

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
        agent_id = entity_data.get('id', '')

        # Store field references for saving
        self.property_fields = {}

        # Load full recipe data from YAML file
        recipe_data = {}
        try:
            recipe_name = agent_id.replace('agent_', '')
            recipe_path = Path(f"../cmush/recipes/{recipe_name}.yaml")
            if recipe_path.exists():
                with open(recipe_path, 'r') as f:
                    recipe_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading recipe: {e}")

        # Identity group
        identity_group = self.create_property_group("Identity")
        self.property_fields['name'] = self.add_text_field(identity_group, "Name",
                                                           recipe_data.get('name', agent.get('name', '')))
        self.property_fields['species'] = self.add_text_field(identity_group, "Species",
                                                               recipe_data.get('species', agent.get('species', '')))

        # Description from recipe
        description = recipe_data.get('description', 'A kindled Noodling...')
        self.property_fields['description'] = self.add_text_area(identity_group, "Description", description)

        self.properties_layout.addWidget(identity_group)

        # LLM Configuration group
        llm_group = self.create_property_group("LLM Configuration")
        self.property_fields['llm_provider'] = self.add_text_field(llm_group, "Provider", agent.get('llm_provider') or 'local')
        self.property_fields['llm_model'] = self.add_text_field(llm_group, "Model", agent.get('llm_model') or 'qwen/qwen3-4b-2507')
        self.properties_layout.addWidget(llm_group)

        # Personality group (load from recipe)
        personality = recipe_data.get('personality', {})
        personality_group = self.create_property_group("Personality Traits")
        self.property_fields['extraversion'] = self.add_slider_field(personality_group, "Extraversion",
                                                                      personality.get('extraversion', 0.5), 0.0, 1.0)
        self.property_fields['curiosity'] = self.add_slider_field(personality_group, "Curiosity",
                                                                   personality.get('curiosity', 0.65), 0.0, 1.0)
        self.property_fields['impulsivity'] = self.add_slider_field(personality_group, "Impulsivity",
                                                                     personality.get('impulsivity', 0.5), 0.0, 1.0)
        self.property_fields['emotional_volatility'] = self.add_slider_field(personality_group, "Emotional Volatility",
                                                                              personality.get('emotional_volatility', 0.5), 0.0, 1.0)
        self.properties_layout.addWidget(personality_group)

        # ===== NOODLE COMPONENT (Unity-style component!) =====
        noodle_component = self.create_noodle_component(agent_id)
        self.properties_layout.addWidget(noodle_component)

        self.properties_layout.addStretch()

    def load_user_properties(self, entity_data):
        """Show user properties."""
        self.property_fields = {}

        user_group = self.create_property_group("User Info")
        self.property_fields['username'] = self.add_text_field(user_group, "Username", "caity")
        self.property_fields['type'] = self.add_text_field(user_group, "Type", "Noodler (human)")
        self.property_fields['age'] = self.add_text_field(user_group, "Age", "9 years old")
        self.property_fields['pronouns'] = self.add_text_field(user_group, "Pronouns", "she/her")
        self.properties_layout.addWidget(user_group)

        # Description
        desc_group = self.create_property_group("Description")
        desc_text = ("A nine-year-old girl in worn overalls with a shock of wild, curly brown hair "
                    "and sparkling blue eyes. She has a wooden sword hanging from one of her belt loops "
                    "and she's sucking a glowing, heart-shaped red hot atomic fireball candy.")
        self.property_fields['description'] = self.add_text_area(desc_group, "Description", desc_text)
        self.properties_layout.addWidget(desc_group)

        inventory_group = self.create_property_group("Inventory")
        self.property_fields['item1'] = self.add_text_field(inventory_group, "Item 1", "Wooden sword")
        self.property_fields['item2'] = self.add_text_field(inventory_group, "Item 2", "Atomic fireball candy")
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
        field.textChanged.connect(lambda: self.save_button.setEnabled(True))
        group.layout().addRow(f"{label}:", field)
        return field

    def add_text_area(self, group: QGroupBox, label: str, value: str):
        """Add editable text area to group."""
        field = QTextEdit()
        field.setPlainText(value)
        field.setMaximumHeight(100)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        field.textChanged.connect(lambda: self.save_button.setEnabled(True))
        group.layout().addRow(f"{label}:", field)
        return field

    def add_slider_field(self, group: QGroupBox, label: str, value: float, min_val: float, max_val: float):
        """Add slider + numeric field."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2;")
        spin.valueChanged.connect(lambda: self.save_button.setEnabled(True))
        group.layout().addRow(f"{label}:", spin)
        return spin

    def save_changes(self):
        """Save edited properties back to noodleMUSH."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity

        if entity_type == 'noodling':
            # Build update payload
            agent_id = entity_data.get('id', '')
            updates = {}

            # Collect field values
            if 'name' in self.property_fields:
                updates['name'] = self.property_fields['name'].text()
            if 'species' in self.property_fields:
                updates['species'] = self.property_fields['species'].text()
            if 'llm_provider' in self.property_fields:
                updates['llm_provider'] = self.property_fields['llm_provider'].text()
            if 'llm_model' in self.property_fields:
                updates['llm_model'] = self.property_fields['llm_model'].text()

            # Personality traits
            personality = {}
            if 'extraversion' in self.property_fields:
                personality['extraversion'] = self.property_fields['extraversion'].value()
            if 'curiosity' in self.property_fields:
                personality['curiosity'] = self.property_fields['curiosity'].value()
            if 'spontaneity' in self.property_fields:
                personality['spontaneity'] = self.property_fields['spontaneity'].value()
            if 'emotional_sensitivity' in self.property_fields:
                personality['emotional_sensitivity'] = self.property_fields['emotional_sensitivity'].value()

            # Save via API (using config endpoint for now)
            try:
                # Update recipe file
                url = f"{self.api_base}/config/save"
                payload = {
                    'path': f'recipes.{agent_id}',
                    'value': updates
                }
                # Note: This endpoint needs to be created in api_server.py
                # For now, just print what we would save
                print(f"Would save to {agent_id}:")
                print(f"  Updates: {updates}")
                print(f"  Personality: {personality}")

                self.save_button.setEnabled(False)

            except Exception as e:
                print(f"Error saving: {e}")

    def create_noodle_component(self, agent_id: str) -> QGroupBox:
        """
        Create the Noodle Component (Unity-style component).

        Shows LIVE updating:
        - 5-D Affect Vector (current emotion)
        - 40-D Phenomenal State (inner kindling)
        - Surprise metric
        """
        component = QGroupBox("🧠 Noodle Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout()

        # Agent ID reference (for live updates)
        self.current_agent_id = agent_id

        # 5-D Affect Vector (LIVE)
        affect_label = QLabel("5-D Affect Vector (Live)")
        affect_label.setStyleSheet("color: #81C784; font-weight: bold; margin-top: 8px;")
        layout.addWidget(affect_label)

        affect_layout = QFormLayout()

        # Create progress bars for each affect dimension
        self.live_affect_labels['valence'] = self.create_affect_bar("Valence", -1.0, 1.0)
        affect_layout.addRow("Valence:", self.live_affect_labels['valence'])

        self.live_affect_labels['arousal'] = self.create_affect_bar("Arousal", 0.0, 1.0)
        affect_layout.addRow("Arousal:", self.live_affect_labels['arousal'])

        self.live_affect_labels['fear'] = self.create_affect_bar("Fear", 0.0, 1.0)
        affect_layout.addRow("Fear:", self.live_affect_labels['fear'])

        self.live_affect_labels['sorrow'] = self.create_affect_bar("Sorrow", 0.0, 1.0)
        affect_layout.addRow("Sorrow:", self.live_affect_labels['sorrow'])

        self.live_affect_labels['boredom'] = self.create_affect_bar("Boredom", 0.0, 1.0)
        affect_layout.addRow("Boredom:", self.live_affect_labels['boredom'])

        layout.addLayout(affect_layout)

        # Phenomenal State (40-D kindling vector)
        phenomenal_label = QLabel("40-D Phenomenal State (Inner Kindling)")
        phenomenal_label.setStyleSheet("color: #81C784; font-weight: bold; margin-top: 12px;")
        layout.addWidget(phenomenal_label)

        self.live_phenomenal_label = QLabel("Waiting for data...")
        self.live_phenomenal_label.setStyleSheet("color: #B0B0B0; font-family: 'Courier New'; font-size: 9px;")
        self.live_phenomenal_label.setWordWrap(True)
        layout.addWidget(self.live_phenomenal_label)

        # Surprise metric
        surprise_layout = QFormLayout()
        self.live_surprise_label = QLabel("0.000")
        self.live_surprise_label.setStyleSheet("color: #FFA726; font-weight: bold;")
        surprise_layout.addRow("Surprise:", self.live_surprise_label)
        layout.addLayout(surprise_layout)

        component.setLayout(layout)
        return component

    def create_affect_bar(self, name: str, min_val: float, max_val: float) -> QWidget:
        """Create a horizontal bar + value label for affect dimension."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress bar
        bar = QProgressBar()
        bar.setRange(int(min_val * 100), int(max_val * 100))
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setMaximumHeight(12)
        bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 2px;
            }
        """)
        layout.addWidget(bar, stretch=3)

        # Value label
        value_label = QLabel("0.00")
        value_label.setStyleSheet("color: #D2D2D2; font-family: 'Courier New';")
        value_label.setMinimumWidth(45)
        layout.addWidget(value_label)

        container.setLayout(layout)

        # Store references
        container.bar = bar
        container.value_label = value_label

        return container

    def update_live_data(self):
        """Update live Noodle Component data from API."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = self.current_agent_id if hasattr(self, 'current_agent_id') else entity_data.get('id')

        try:
            # Fetch live state from API
            resp = requests.get(f"{self.api_base}/agents/{agent_id}/state", timeout=1)
            if resp.status_code == 200:
                state = resp.json()

                # Update 5-D Affect Vector
                affect = state.get('affect', {})
                for dim, widget in self.live_affect_labels.items():
                    if dim in affect:
                        value = affect[dim]
                        widget.bar.setValue(int(value * 100))
                        widget.value_label.setText(f"{value:+.2f}")

                        # Color code based on value
                        if value > 0.7:
                            color = "#4CAF50"  # Green (positive/high)
                        elif value > 0.3:
                            color = "#FFA726"  # Orange (moderate)
                        else:
                            color = "#EF5350"  # Red (negative/low)

                        widget.bar.setStyleSheet(f"""
                            QProgressBar {{
                                border: 1px solid #555;
                                border-radius: 3px;
                                background: #2a2a2a;
                            }}
                            QProgressBar::chunk {{
                                background: {color};
                                border-radius: 2px;
                            }}
                        """)

                # Update 40-D Phenomenal State
                phenomenal = state.get('phenomenal_state', [])
                if phenomenal:
                    # Format as 3 lines of ~13 values each
                    phenomenal_str = "["
                    for i, val in enumerate(phenomenal):
                        if i % 13 == 0 and i > 0:
                            phenomenal_str += "\n "
                        phenomenal_str += f"{val:+.3f}, "
                    phenomenal_str = phenomenal_str.rstrip(", ") + "]"
                    self.live_phenomenal_label.setText(phenomenal_str)

                # Update Surprise
                surprise = state.get('surprise', 0.0)
                self.live_surprise_label.setText(f"{surprise:.3f}")

                # Color code surprise
                if surprise > 0.5:
                    color = "#EF5350"  # Red (high surprise!)
                elif surprise > 0.3:
                    color = "#FFA726"  # Orange
                else:
                    color = "#4CAF50"  # Green (expected)
                self.live_surprise_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        except requests.exceptions.RequestException:
            # API not available, silently fail
            pass
        except Exception as e:
            print(f"Error updating live data: {e}")

    def add_artbook_component(self):
        """
        Add Artbook component to current entity.

        Shows reference art from assets folder - like ArtStation for your character!
        """
        artbook = self.create_artbook_component()
        self.properties_layout.addWidget(artbook)

    def create_artbook_component(self) -> QGroupBox:
        """
        Create Artbook component (Unity-style component).

        Holds reference art, concept sketches, mood boards for the character.
        """
        component = QGroupBox("🎨 Artbook Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #FF9800;
                border: 2px solid #FF9800;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Reference art and concept images for this character")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Art gallery (thumbnail grid)
        gallery_label = QLabel("Reference Gallery")
        gallery_label.setStyleSheet("color: #FFB74D; font-weight: bold; margin-top: 4px;")
        layout.addWidget(gallery_label)

        # List widget for art thumbnails
        self.art_gallery = QListWidget()
        self.art_gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.art_gallery.setIconSize(QSize(80, 80))
        self.art_gallery.setSpacing(8)
        self.art_gallery.setMaximumHeight(200)
        self.art_gallery.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QListWidget::item {
                background: #1a1a1a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item:hover {
                background: #333;
                border: 1px solid #FF9800;
            }
            QListWidget::item:selected {
                background: #FF9800;
                border: 1px solid #FFB74D;
            }
        """)
        layout.addWidget(self.art_gallery)

        # Buttons
        button_layout = QHBoxLayout()

        add_art_btn = QPushButton("+ Add Art")
        add_art_btn.clicked.connect(self.add_art_to_gallery)
        add_art_btn.setStyleSheet("""
            QPushButton {
                background: #FF9800;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #FFB74D;
            }
        """)
        button_layout.addWidget(add_art_btn)

        remove_art_btn = QPushButton("− Remove")
        remove_art_btn.clicked.connect(self.remove_art_from_gallery)
        remove_art_btn.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        button_layout.addWidget(remove_art_btn)

        layout.addLayout(button_layout)

        # Art source info
        source_label = QLabel("💡 Tip: Keep art in ~/.noodlestudio/assets/[character_name]/")
        source_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        source_label.setWordWrap(True)
        layout.addWidget(source_label)

        component.setLayout(layout)

        # Load existing art if any
        self.load_artbook_gallery()

        return component

    def add_art_to_gallery(self):
        """Add art file to the gallery."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Reference Art",
            str(Path.home() / ".noodlestudio" / "assets"),
            "Images (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*)"
        )

        for filename in filenames:
            if filename:
                # Create thumbnail
                pixmap = QPixmap(filename)
                if not pixmap.isNull():
                    # Scale to thumbnail size
                    scaled = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

                    # Add to gallery
                    item = QListWidgetItem()
                    item.setIcon(QIcon(scaled))
                    item.setToolTip(Path(filename).name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)  # Store full path
                    self.art_gallery.addItem(item)

        # Save gallery state
        self.save_artbook_gallery()

    def remove_art_from_gallery(self):
        """Remove selected art from gallery."""
        current = self.art_gallery.currentItem()
        if current:
            self.art_gallery.takeItem(self.art_gallery.row(current))
            self.save_artbook_gallery()

    def load_artbook_gallery(self):
        """Load artbook gallery from saved state."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = entity_data.get('id', '')

        # Load from .noodlestudio/artbooks/{agent_id}.json
        artbook_dir = Path.home() / ".noodlestudio" / "artbooks"
        artbook_file = artbook_dir / f"{agent_id}.json"

        if artbook_file.exists():
            try:
                import json
                with open(artbook_file, 'r') as f:
                    data = json.load(f)

                for art_path in data.get('art_files', []):
                    if Path(art_path).exists():
                        pixmap = QPixmap(art_path)
                        if not pixmap.isNull():
                            scaled = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)

                            item = QListWidgetItem()
                            item.setIcon(QIcon(scaled))
                            item.setToolTip(Path(art_path).name)
                            item.setData(Qt.ItemDataRole.UserRole, art_path)
                            self.art_gallery.addItem(item)

            except Exception as e:
                print(f"Error loading artbook: {e}")

    def save_artbook_gallery(self):
        """Save artbook gallery state."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = entity_data.get('id', '')

        # Collect all art file paths
        art_files = []
        for i in range(self.art_gallery.count()):
            item = self.art_gallery.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                art_files.append(path)

        # Save to .noodlestudio/artbooks/{agent_id}.json
        artbook_dir = Path.home() / ".noodlestudio" / "artbooks"
        artbook_dir.mkdir(parents=True, exist_ok=True)

        artbook_file = artbook_dir / f"{agent_id}.json"

        try:
            import json
            with open(artbook_file, 'w') as f:
                json.dump({'art_files': art_files}, f, indent=2)
        except Exception as e:
            print(f"Error saving artbook: {e}")

    def add_script_component(self):
        """
        Add Script component with code editor.

        Like Unity's script component!
        """
        script_comp = self.create_script_component()
        self.properties_layout.addWidget(script_comp)

    def create_script_component(self) -> QGroupBox:
        """
        Create Script component (Unity-style).

        Shows code editor with syntax highlighting and compile button.
        """
        component = QGroupBox("📜 Script Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #9C27B0;
                border: 2px solid #9C27B0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Python script for event-driven behavior (Unity-like API)")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Script editor widget
        from ..widgets.script_editor import ScriptEditor
        self.script_editor = ScriptEditor()
        layout.addWidget(self.script_editor)

        component.setLayout(layout)
        return component

