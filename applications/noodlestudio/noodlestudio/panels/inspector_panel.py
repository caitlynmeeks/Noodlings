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
from PyQt6.QtGui import QFont, QPixmap, QIcon, QFontMetrics
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

        elif entity_type == 'prim' or entity_type == 'object':
            obj_name = entity_data.get('id', 'Unknown Object').replace('obj_', '').replace('_', ' ').title()
            self.entity_header.setText(f"Prim: {obj_name}")
            self.load_object_properties(entity_data)

        elif entity_type == 'exit':
            direction = entity_data.get('direction', 'unknown')
            self.entity_header.setText(f"Exit: {direction}")
            self.load_exit_properties(entity_data)

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

        # ===== MMCR COMPONENT (Multimodal Context Reference) =====
        mmcr_component = self.create_mmcr_component(agent_id)
        self.properties_layout.addWidget(mmcr_component)

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
        obj_id = entity_data.get('id', '')

        # Basic properties
        obj_group = self.create_property_group("Object Properties")
        self.add_text_field(obj_group, "ID", obj_id)
        self.add_text_field(obj_group, "Name", obj_id.replace('obj_', '').replace('_', ' ').title())
        self.add_text_area(obj_group, "Description", "An object in the world...")
        self.properties_layout.addWidget(obj_group)

        # Arbitrary metadata editor
        metadata_component = self.create_metadata_component(obj_id)
        self.properties_layout.addWidget(metadata_component)

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
        # Start with expanded triangle
        group = QGroupBox(f"▼ {title}")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        group.setCheckable(True)  # Make collapsible
        group.setChecked(True)  # Expanded by default
        group.setStyleSheet("""
            QGroupBox {
                color: #FFFFFF;
                background-color: #2A2A2A;
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
            QGroupBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)
        group.setLayout(QFormLayout())
        # Store original title for triangle updates
        group.setProperty("original_title", title)
        # Connect toggled signal to hide/show contents and update triangle
        group.toggled.connect(lambda checked: self.on_group_toggled(group, checked))
        return group

    def on_group_toggled(self, group: QGroupBox, checked: bool):
        """Handle group toggle - update triangle and visibility."""
        # Update triangle in title (block signals to prevent re-triggering)
        original_title = group.property("original_title")
        group.blockSignals(True)
        if checked:
            group.setTitle(f"▼ {original_title}")
        else:
            group.setTitle(f"▶ {original_title}")
        group.blockSignals(False)

        # Toggle visibility of contents
        self.toggle_group_contents(group, checked)

    def toggle_group_contents(self, group: QGroupBox, visible: bool):
        """Toggle visibility of group contents (Unity-style collapse)."""
        # Hide/show all child widgets in the group's layout
        layout = group.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    item.widget().setVisible(visible)

    def add_text_field(self, group: QGroupBox, label: str, value: str):
        """Add editable text field to group (Unity-style instant updates)."""
        field = QLineEdit(value)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        # Use editingFinished for instant update when user finishes editing
        field.editingFinished.connect(self.save_changes)
        group.layout().addRow(f"{label}:", field)
        return field

    def add_text_area(self, group: QGroupBox, label: str, value: str):
        """Add editable text area to group (Unity-style instant updates)."""
        field = QTextEdit()
        field.setPlainText(value)
        field.setMaximumHeight(100)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        # Text areas update when focus is lost (avoid spamming API)
        field.focusOutEvent = lambda e: (QTextEdit.focusOutEvent(field, e), self.save_changes())
        group.layout().addRow(f"{label}:", field)
        return field

    def add_slider_field(self, group: QGroupBox, label: str, value: float, min_val: float, max_val: float):
        """Add slider + numeric field (Unity-style instant updates)."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2;")
        # Instant update when value changes (Unity-style)
        spin.valueChanged.connect(lambda: self.save_changes())
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

            # Save via API
            try:
                url = f"{self.api_base}/agents/{agent_id}/update"
                payload = {
                    **updates,
                    'personality': personality
                }

                response = requests.post(url, json=payload, timeout=2)
                if response.status_code == 200:
                    print(f"Saved changes for {agent_id}")
                else:
                    print(f"Error saving: {response.json().get('error', 'Unknown error')}")

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
        # Use standardized component header
        component = QGroupBox("▼ Noodle Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setCheckable(True)  # Make collapsible
        component.setChecked(True)  # Expanded by default
        component.setStyleSheet("""
            QGroupBox {
                color: #FFFFFF;
                background-color: #2A2A2A;
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
            QGroupBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)
        component.setProperty("original_title", "Noodle Component")
        component.toggled.connect(lambda checked: self.on_group_toggled(component, checked))

        layout = QVBoxLayout()

        # Agent ID reference (for live updates)
        self.current_agent_id = agent_id

        # 5-D Affect Vector (LIVE)
        affect_label = QLabel("5-D Affect Vector (Live)")
        affect_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 8px;")
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
        phenomenal_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 12px;")
        layout.addWidget(phenomenal_label)

        self.live_phenomenal_label = QLabel("Waiting for data...")
        self.live_phenomenal_label.setStyleSheet("color: #B0B0B0; font-family: 'Courier New'; font-size: 9px;")
        self.live_phenomenal_label.setWordWrap(True)
        layout.addWidget(self.live_phenomenal_label)

        # Surprise metric
        surprise_layout = QFormLayout()
        self.live_surprise_label = QLabel("0.000")
        self.live_surprise_label.setStyleSheet("color: #D2D2D2; font-weight: bold;")
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

                        # Consistent gray styling
                        widget.bar.setStyleSheet("""
                            QProgressBar {
                                border: 1px solid #555;
                                border-radius: 3px;
                                background: #2a2a2a;
                            }
                            QProgressBar::chunk {
                                background: #666;
                                border-radius: 2px;
                            }
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
        component = QGroupBox("Artbook Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setCheckable(True)  # Make collapsible
        component.setChecked(True)  # Expanded by default
        component.setStyleSheet("""
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
            QGroupBox::indicator {
                width: 0px;
                height: 0px;
                margin-right: 8px;
            }
            QGroupBox::indicator:unchecked {
                border-left: 8px solid #EEEEEE;
                border-top: 5px solid transparent;
                border-bottom: 5px solid transparent;
            }
            QGroupBox::indicator:checked {
                border-top: 8px solid #EEEEEE;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
            }
        """)
        component.toggled.connect(lambda checked: self.toggle_group_contents(component, checked))

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Reference art and concept images for this character")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Art gallery (thumbnail grid)
        gallery_label = QLabel("Reference Gallery")
        gallery_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 4px;")
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
                border: 1px solid #666;
            }
            QListWidget::item:selected {
                background: #444;
                border: 1px solid #888;
            }
        """)
        layout.addWidget(self.art_gallery)

        # Buttons
        button_layout = QHBoxLayout()

        add_art_btn = QPushButton("+ Add Art")
        add_art_btn.clicked.connect(self.add_art_to_gallery)
        add_art_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                color: #D2D2D2;
                padding: 6px 12px;
                border-radius: 3px;
                border: 1px solid #555;
            }
            QPushButton:hover {
                background: #4a4a4a;
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

    def create_mmcr_component(self, agent_id: str) -> QGroupBox:
        """
        Create MMCR (Multimodal Context Reference) component.

        Holds arbitrary media that scripts can access via API:
        - Images (concept art, reference photos, environment maps)
        - Audio (voice clips, sound effects, music)
        - Video (animations, cutscenes)
        - Text (notes, dialogue snippets)

        Unlike Artbook (which is for visual reference), MMCR is for
        runtime-accessible context that affects behavior.
        """
        # Use standardized component header
        component = QGroupBox("▼ Multimodal Context Reference")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setCheckable(True)  # Make collapsible
        component.setChecked(True)  # Expanded by default
        component.setStyleSheet("""
            QGroupBox {
                color: #FFFFFF;
                background-color: #2A2A2A;
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
            QGroupBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)
        component.setProperty("original_title", "Multimodal Context Reference")
        component.toggled.connect(lambda checked: self.on_group_toggled(component, checked))

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Runtime-accessible media for scripts and LLM context")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Images section
        images_label = QLabel("Images")
        images_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 4px;")
        layout.addWidget(images_label)

        self.mmcr_images = QListWidget()
        self.mmcr_images.setMaximumHeight(100)
        self.mmcr_images.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        layout.addWidget(self.mmcr_images)

        # Audio section
        audio_label = QLabel("Audio")
        audio_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 8px;")
        layout.addWidget(audio_label)

        self.mmcr_audio = QListWidget()
        self.mmcr_audio.setMaximumHeight(80)
        self.mmcr_audio.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        layout.addWidget(self.mmcr_audio)

        # Buttons
        button_layout = QHBoxLayout()

        add_media_btn = QPushButton("Add Media")
        add_media_btn.clicked.connect(lambda: self.add_mmcr_media(agent_id))
        add_media_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #64B5F6;
            }
        """)
        button_layout.addWidget(add_media_btn)

        remove_media_btn = QPushButton("Remove")
        remove_media_btn.clicked.connect(lambda: self.remove_mmcr_media())
        remove_media_btn.setStyleSheet("""
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
        button_layout.addWidget(remove_media_btn)

        layout.addLayout(button_layout)

        # API access info
        info_label = QLabel("Scripts access via: noodlings.getComponent('mmcr').images[0]")
        info_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        component.setLayout(layout)

        # Load existing MMCR data if any
        self.load_mmcr_data(agent_id)

        return component

    def add_mmcr_media(self, agent_id: str):
        """Add media files to MMCR component."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Media to MMCR",
            str(Path.home()),
            "Media Files (*.png *.jpg *.jpeg *.gif *.webp *.wav *.mp3 *.mp4 *.mov);;All Files (*)"
        )

        for filename in filenames:
            if filename:
                file_path = Path(filename)
                ext = file_path.suffix.lower()

                # Categorize by file type
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    item = QListWidgetItem(file_path.name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)
                    self.mmcr_images.addItem(item)
                elif ext in ['.wav', '.mp3', '.ogg', '.m4a']:
                    item = QListWidgetItem(file_path.name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)
                    self.mmcr_audio.addItem(item)

        # Save MMCR state
        self.save_mmcr_data(agent_id)

    def remove_mmcr_media(self):
        """Remove selected media from MMCR."""
        # Check images list
        current = self.mmcr_images.currentItem()
        if current:
            self.mmcr_images.takeItem(self.mmcr_images.row(current))
            return

        # Check audio list
        current = self.mmcr_audio.currentItem()
        if current:
            self.mmcr_audio.takeItem(self.mmcr_audio.row(current))

    def load_mmcr_data(self, agent_id: str):
        """Load MMCR data from storage."""
        # TODO: Implement when components dict is added to world structure
        # Would load from: agent_data['components']['mmcr']
        pass

    def save_mmcr_data(self, agent_id: str):
        """Save MMCR data to storage."""
        # Collect all media files
        images = []
        for i in range(self.mmcr_images.count()):
            item = self.mmcr_images.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                images.append(path)

        audio = []
        for i in range(self.mmcr_audio.count()):
            item = self.mmcr_audio.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                audio.append(path)

        # TODO: Save via API to agent's components.mmcr
        # POST /api/agents/{agent_id}/components/mmcr
        print(f"MMCR data for {agent_id}:")
        print(f"  Images: {images}")
        print(f"  Audio: {audio}")

    def create_metadata_component(self, entity_id: str) -> QGroupBox:
        """
        Create Metadata component for arbitrary key-value pairs.

        Like USD custom attributes - author can add any field they want:
        - asteroid.mass_kg = 8500000000
        - asteroid.minerals = ["iron: 45%", "platinum: 0.02%"]
        - asteroid.appearance_far = "A dark speck..."

        Scripts access via: prim.metadata["mass_kg"]
        """
        component = QGroupBox("Custom Metadata")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #9E9E9E;
                border: 2px solid #757575;
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
        desc = QLabel("Arbitrary key-value pairs accessible to scripts and renderers")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Metadata list (shows key: value pairs)
        self.metadata_list = QListWidget()
        self.metadata_list.setMaximumHeight(150)
        self.metadata_list.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
                font-family: 'Courier New';
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:hover {
                background: #333;
            }
            QListWidget::item:selected {
                background: #555;
            }
        """)
        layout.addWidget(self.metadata_list)

        # Buttons
        button_layout = QHBoxLayout()

        add_meta_btn = QPushButton("Add Field")
        add_meta_btn.clicked.connect(lambda: self.add_metadata_field(entity_id))
        add_meta_btn.setStyleSheet("""
            QPushButton {
                background: #757575;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #9E9E9E;
            }
        """)
        button_layout.addWidget(add_meta_btn)

        edit_meta_btn = QPushButton("Edit")
        edit_meta_btn.clicked.connect(lambda: self.edit_metadata_field(entity_id))
        edit_meta_btn.setStyleSheet("""
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
        button_layout.addWidget(edit_meta_btn)

        remove_meta_btn = QPushButton("Remove")
        remove_meta_btn.clicked.connect(lambda: self.remove_metadata_field(entity_id))
        remove_meta_btn.setStyleSheet("""
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
        button_layout.addWidget(remove_meta_btn)

        layout.addLayout(button_layout)

        # Access info
        info_label = QLabel("Example: asteroid.metadata['mass_kg'] or asteroid.metadata['minerals']")
        info_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        component.setLayout(layout)

        # Load existing metadata
        self.load_metadata(entity_id)

        return component

    def add_metadata_field(self, entity_id: str):
        """Add a new metadata field."""
        from PyQt6.QtWidgets import QInputDialog

        # Get field name
        field_name, ok = QInputDialog.getText(
            self,
            "Add Metadata Field",
            "Field name (e.g., 'mass_kg', 'minerals', 'velocity'):"
        )

        if ok and field_name:
            # Get field value
            field_value, ok = QInputDialog.getText(
                self,
                "Add Metadata Field",
                f"Value for '{field_name}':"
            )

            if ok:
                # Add to list
                item = QListWidgetItem(f"{field_name}: {field_value}")
                item.setData(Qt.ItemDataRole.UserRole, {'key': field_name, 'value': field_value})
                self.metadata_list.addItem(item)

                # Save
                self.save_metadata(entity_id)

    def edit_metadata_field(self, entity_id: str):
        """Edit selected metadata field."""
        from PyQt6.QtWidgets import QInputDialog

        current = self.metadata_list.currentItem()
        if not current:
            return

        data = current.data(Qt.ItemDataRole.UserRole)
        field_name = data['key']
        old_value = data['value']

        # Get new value
        new_value, ok = QInputDialog.getText(
            self,
            "Edit Metadata Field",
            f"New value for '{field_name}':",
            text=old_value
        )

        if ok:
            # Update item
            current.setText(f"{field_name}: {new_value}")
            current.setData(Qt.ItemDataRole.UserRole, {'key': field_name, 'value': new_value})

            # Save
            self.save_metadata(entity_id)

    def remove_metadata_field(self, entity_id: str):
        """Remove selected metadata field."""
        current = self.metadata_list.currentItem()
        if current:
            self.metadata_list.takeItem(self.metadata_list.row(current))
            self.save_metadata(entity_id)

    def load_metadata(self, entity_id: str):
        """Load metadata from world state."""
        # TODO: Load from world state when metadata dict is added
        # For now, show example for demonstration
        if entity_id.startswith('obj_'):
            # Example metadata
            examples = {
                'type': 'vending_machine',
                'portable': 'true',
                'takeable': 'true'
            }
            for key, value in examples.items():
                item = QListWidgetItem(f"{key}: {value}")
                item.setData(Qt.ItemDataRole.UserRole, {'key': key, 'value': value})
                self.metadata_list.addItem(item)

    def save_metadata(self, entity_id: str):
        """Save metadata to world state."""
        # Collect all metadata fields
        metadata = {}
        for i in range(self.metadata_list.count()):
            item = self.metadata_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                metadata[data['key']] = data['value']

        # TODO: Save via API
        # POST /api/objects/{entity_id}/metadata
        print(f"Metadata for {entity_id}:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

