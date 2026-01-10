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
#   Main Window Entities Mixin - Entity creation and management
#
#   Contains: - add_noodling, add_object, add_room: Add entit...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_entities_mixin
# PURPOSE:  Main Window Entities Mixin
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowEntitiesMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from pathlib import Path

from PyQt6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox
from PyQt6.QtCore import Qt


class MainWindowEntitiesMixin:
    """Mixin providing entity management for MainWindow."""

    def add_noodling(self):
        """Add a new Noodling to the stage."""
        name, ok = QInputDialog.getText(
            self, "Add Noodling", "Noodling name:", text="NewNoodling"
        )
        if ok and name:
            QMessageBox.information(
                self, "Rez Noodling",
                f"Rezzing Noodling prim: {name}\n\n(API integration not yet implemented)"
            )

    def add_object(self):
        """Add a new object to the stage."""
        name, ok = QInputDialog.getText(
            self, "Add Object", "Object name:", text="NewObject"
        )
        if ok and name:
            QMessageBox.information(
                self, "Add Object",
                f"Adding object prim: {name}\n\n(API integration not yet implemented)"
            )

    def add_room(self):
        """Add a new room to the stage."""
        name, ok = QInputDialog.getText(
            self, "Add Room", "Room name:", text="NewRoom"
        )
        if ok and name:
            QMessageBox.information(
                self, "Add Room",
                f"Adding room prim: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_noodling(self):
        """Create an empty Noodling using the library template."""
        name, ok = QInputDialog.getText(
            self, "Create Empty Noodling", "Noodling name:", text="NewNoodling"
        )
        if ok and name:
            try:
                import requests
                response = requests.post(
                    'http://localhost:8081/api/agents',
                    json={
                        'name': name,
                        'species': 'noodling',
                        'pronouns': 'they/them'
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    QMessageBox.information(
                        self, "Noodling Rezzed",
                        f"Rezzed: {name}\n\n"
                        f"Template: library/empty_noodling (gingerbread foundation)\n"
                        f"Personality: Curious, bewildered, harmless\n"
                        f"Species: noodling"
                    )
                    if hasattr(self, 'scene_hierarchy') and self.scene_hierarchy:
                        self.scene_hierarchy.refresh_scene()
                else:
                    QMessageBox.warning(
                        self, "Rez Failed",
                        f"Could not rez {name}:\n{response.text}"
                    )
            except Exception as e:
                QMessageBox.warning(
                    self, "Rez Failed",
                    f"Could not rez {name}:\n{str(e)}\n\nIs the server running?"
                )

    def create_specialized_noodling(self, species: str):
        """Create a specialized Noodling with species-specific defaults."""
        name, ok = QInputDialog.getText(
            self, f"Create {species.title()} Noodling",
            "Noodling name:", text=f"New{species.title()}"
        )
        if ok and name:
            presets = {
                'kitten': {
                    'extraversion': 0.7, 'curiosity': 0.9,
                    'impulsivity': 0.8, 'emotional_volatility': 0.6
                },
                'robot': {
                    'extraversion': 0.3, 'curiosity': 0.6,
                    'impulsivity': 0.2, 'emotional_volatility': 0.1
                },
                'dragon': {
                    'extraversion': 0.6, 'curiosity': 0.5,
                    'impulsivity': 0.4, 'emotional_volatility': 0.7
                }
            }
            personality = presets.get(species, {})
            QMessageBox.information(
                self, "Create Specialized Noodling",
                f"Creating {species} Noodling: {name}\n\n"
                f"Personality preset:\n"
                f"  Extraversion: {personality.get('extraversion', 0.5)}\n"
                f"  Curiosity: {personality.get('curiosity', 0.5)}\n"
                f"  Impulsivity: {personality.get('impulsivity', 0.5)}\n"
                f"  Volatility: {personality.get('emotional_volatility', 0.5)}\n\n"
                f"(API integration not yet implemented)"
            )

    def create_empty_object(self):
        """Create an empty object prim."""
        name, ok = QInputDialog.getText(
            self, "Create Empty Object", "Object name:", text="NewObject"
        )
        if ok and name:
            QMessageBox.information(
                self, "Create Object",
                f"Creating empty object: {name}\n\n(API integration not yet implemented)"
            )

    def create_specialized_object(self, obj_type: str):
        """Create a specialized object with type-specific properties."""
        name, ok = QInputDialog.getText(
            self, f"Create {obj_type.title()}",
            f"{obj_type.title()} name:", text=f"New{obj_type.title()}"
        )
        if ok and name:
            properties = {
                'prop': 'holdable=true, takeable=true',
                'furniture': 'sittable=true, fixed=true',
                'container': 'openable=true, container=true'
            }
            QMessageBox.information(
                self, "Create Specialized Object",
                f"Creating {obj_type}: {name}\n\n"
                f"Properties: {properties.get(obj_type, 'none')}\n\n"
                f"(API integration not yet implemented)"
            )

    def create_empty_room(self):
        """Create an empty room prim."""
        name, ok = QInputDialog.getText(
            self, "Create Empty Room", "Room name:", text="NewRoom"
        )
        if ok and name:
            QMessageBox.information(
                self, "Create Room",
                f"Creating empty room: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_prim(self):
        """Create a custom empty prim."""
        name, ok = QInputDialog.getText(
            self, "Create Empty Prim", "Prim name:", text="CustomPrim"
        )
        if ok and name:
            QMessageBox.information(
                self, "Create Prim",
                f"Creating empty prim: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_ensemble(self):
        """Create an empty ensemble for organizing Noodlings."""
        name, ok = QInputDialog.getText(
            self, "Create Empty Ensemble", "Ensemble name:", text="New Ensemble"
        )
        if ok and name:
            QMessageBox.information(
                self, "Empty Ensemble Created",
                f"Created empty ensemble: {name}\n\n"
                f"Now drag Noodlings into the ensemble in Scene Hierarchy!\n\n"
                f"When ready:\n"
                f"  1. Right-click ensemble\n"
                f"  2. Choose 'Export Ensemble to .ens'\n"
                f"  3. Share your .ens file!\n\n"
                f"(Full implementation coming soon)"
            )

    def import_ensemble(self):
        """Import an ensemble file into the project."""
        import os

        if not self.project_manager.is_project_open():
            QMessageBox.warning(
                self, "No Project Open",
                "Please create or open a project first."
            )
            return

        default_dir = os.path.join(
            os.path.dirname(__file__), "../../../cmush/ensembles"
        )
        default_dir = os.path.abspath(default_dir)

        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Ensemble", default_dir,
            "Ensemble Files (*.ensemble);;All Files (*)"
        )

        if filename:
            try:
                if self.project_manager.import_ensemble(filename):
                    basename = os.path.basename(filename)
                    self.statusBar().showMessage(f"Imported ensemble: {basename}", 3000)
                    if hasattr(self, 'assets'):
                        self.assets.refresh()
                else:
                    QMessageBox.warning(self, "Import Failed", "Failed to import ensemble.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Import Failed",
                    f"Error importing ensemble:\n{e}"
                )

    def add_component(self, component_type: str):
        """Add a component to the selected entity."""
        component_names = {
            'noodle': 'Noodle Component',
            'memory': 'Memory Bank Component',
            'relationships': 'Relationship Graph Component',
            'artbook': 'Artbook Component',
            'moodboard': 'Mood Board Component',
            'voiceref': 'Voice Reference Component',
            'dialogue': 'Dialogue Tree Component',
            'quests': 'Quest Giver Component',
            'vendor': 'Vendor Component',
            'custom': 'Custom Script'
        }

        component_name = component_names.get(component_type, 'Unknown Component')

        if not hasattr(self.inspector, 'current_entity') or not self.inspector.current_entity:
            QMessageBox.warning(
                self, "No Entity Selected",
                "Please select an entity in the Scene Hierarchy first,\n"
                "then add a component to it."
            )
            return

        entity_type, entity_data = self.inspector.current_entity

        if component_type == 'artbook':
            self.inspector.add_artbook_component()
            self.statusBar().showMessage(f"Added {component_name} to {entity_type}", 3000)
        elif component_type == 'custom':
            self.inspector.add_script_component()
            self.statusBar().showMessage(f"Added Script Component to {entity_type}", 3000)
        elif component_type == 'noodle':
            QMessageBox.information(
                self, "Noodle Component",
                "Noodle Component is automatically added to all Noodlings!\n\n"
                "It shows live affect, phenomenal state, and surprise."
            )
        else:
            QMessageBox.information(
                self, f"Add {component_name}",
                f"Adding {component_name}...\n\n(Implementation coming soon)"
            )

    # ========== USD IMPORT/EXPORT ==========

    def export_stage_to_usd(self):
        """Export current stage to USD format."""
        import requests
        from ..data.usd_exporter import USDExporter

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Stage to USD", "noodlemush_stage.usda",
            "USD ASCII Layer (*.usda)"
        )

        if filename:
            try:
                resp = requests.get("http://localhost:8081/api/agents")
                agents = resp.json().get('agents', [])

                world_data = {
                    'rooms': {},
                    'noodlings': agents,
                    'users': [{'id': 'user_caity', 'username': 'caity',
                              'description': 'A nine-year-old Noodler'}],
                    'objects': {}
                }

                exporter = USDExporter()
                exporter.export_stage(world_data, Path(filename))

                self.statusBar().showMessage(f"Stage exported to {filename}", 5000)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Stage exported to USD layer:\n{filename}\n\n"
                    f"Contains Noodling prims with charm properties.\n"
                    f"Import into Maya/Houdini/Blender to view."
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def import_usd_layer(self):
        """Import USD layer file into noodleMUSH."""
        from ..data.usd_importer import USDImporter

        filename, _ = QFileDialog.getOpenFileName(
            self, "Import USD Layer", "",
            "USD Files (*.usda *.usdc);;All Files (*)"
        )

        if filename:
            try:
                importer = USDImporter()
                imported_data = importer.import_layer(Path(filename))

                noodlings_count = len(imported_data.get('noodlings', []))
                rooms_count = len(imported_data.get('rooms', []))
                objects_count = len(imported_data.get('objects', []))

                QMessageBox.information(
                    self, "Import Complete",
                    f"USD layer imported:\n{filename}\n\n"
                    f"Found:\n"
                    f"- {noodlings_count} Noodling prims\n"
                    f"- {rooms_count} Room prims\n"
                    f"- {objects_count} Object prims\n\n"
                    f"(Rezzing not yet implemented)"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Import Failed",
                    f"Error: {e}\n\nUSD import requires USD Python library."
                )

    # ========== ENSEMBLE STORE ==========

    def show_ensemble_store(self):
        """Show Ensemble Store window."""
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QTextEdit, QPushButton
        )
        from ..data.ensemble_packs import ENSEMBLE_LIBRARY

        dialog = QDialog(self)
        dialog.setWindowTitle("Ensemble Store")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        header = QLabel("<h1>Ensemble Store</h1><p>Ensemble archetypes for your stage</p>")
        header.setStyleSheet("padding: 10px; background: #2a2a2a;")
        layout.addWidget(header)

        list_widget = QListWidget()
        for pack in ENSEMBLE_LIBRARY.list_packs():
            price_str = "FREE" if pack.price == 0.0 else f"${pack.price}"
            list_widget.addItem(
                f"{pack.name} - {price_str} ({len(pack.archetypes)} archetypes)"
            )
        layout.addWidget(list_widget)

        desc_area = QTextEdit()
        desc_area.setReadOnly(True)
        desc_area.setPlainText("Select an ensemble to see details...")
        layout.addWidget(desc_area)

        def on_selection_changed():
            if list_widget.currentRow() >= 0:
                packs = ENSEMBLE_LIBRARY.list_packs()
                pack = packs[list_widget.currentRow()]

                desc = f"**{pack.name}**\n\n{pack.description}\n\n"
                desc += f"**Version:** {pack.version}\n"
                desc += f"**Author:** {pack.author}\n"
                desc += f"**Price:** {'FREE' if pack.price == 0.0 else f'${pack.price}'}\n"
                desc += f"**License:** {pack.license_type}\n\n"
                desc += f"**Archetypes:**\n"
                for arch in pack.archetypes:
                    desc += f"  - {arch.name} ({arch.species})\n"
                desc += f"\n**Setting:** {pack.suggested_setting}\n"
                desc += f"\n**Dynamics:** {pack.relationship_dynamics}\n"
                desc_area.setPlainText(desc)

        list_widget.currentRowChanged.connect(on_selection_changed)

        button_layout = QHBoxLayout()

        export_btn = QPushButton("Export to .ens File")
        export_btn.clicked.connect(
            lambda: self._export_ensemble_to_file(list_widget, ENSEMBLE_LIBRARY)
        )
        button_layout.addWidget(export_btn)

        spawn_btn = QPushButton("Spawn Ensemble Now")
        spawn_btn.clicked.connect(
            lambda: self._spawn_ensemble_from_store(list_widget, ENSEMBLE_LIBRARY, dialog)
        )
        button_layout.addWidget(spawn_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def _export_ensemble_to_file(self, list_widget, library):
        """Export selected ensemble to .ens file."""
        if list_widget.currentRow() >= 0:
            from ..data.ensemble_format import EnsembleFormat

            packs = library.list_packs()
            pack = packs[list_widget.currentRow()]

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Ensemble",
                str(Path.home() / ".noodlestudio" / "ensembles" / f"{pack.id}.ens"),
                "Ensemble Files (*.ens)"
            )

            if filename:
                EnsembleFormat.save_ensemble(pack, Path(filename))
                QMessageBox.information(
                    self, "Export Complete",
                    f"Ensemble exported to:\n{filename}\n\nYou can now share this .ens file!"
                )

    def _spawn_ensemble_from_store(self, list_widget, library, dialog):
        """Spawn selected ensemble into noodleMUSH."""
        if list_widget.currentRow() >= 0:
            from ..data.ensemble_format import EnsembleSpawner

            packs = library.list_packs()
            pack = packs[list_widget.currentRow()]

            room_id, ok = QInputDialog.getText(
                self, "Rez Ensemble",
                f"Rez '{pack.name}' into which room?", text="room_000"
            )

            if ok and room_id:
                rezzed_ids = EnsembleSpawner.rez_ensemble(pack, room_id)
                QMessageBox.information(
                    self, "Ensemble Rezzed",
                    f"Rezzed {len(rezzed_ids)} Noodlings from '{pack.name}'\n\n"
                    f"Room: {room_id}\n\n"
                    f"(API integration not yet implemented)"
                )
                dialog.close()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
