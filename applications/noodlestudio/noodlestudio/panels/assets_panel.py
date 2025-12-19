"""
Assets Panel - Shows all project assets (Noodlings, Ensembles, Prims, Scripts, Generations).

Organizes assets by type with expandable categories.
Right-click context menus for asset management (to be implemented).

Generations:
AI-generated content (images from subconscious, scripted facets, etc.)
is automatically organized in the Generations category with thumbnails.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap
import os
import json


class AssetsPanel(QWidget):
    """
    Assets panel showing all project assets organized by type.

    Categories:
    - Noodlings (individual agents)
    - Ensembles (groups of agents)
    - Generations (AI-generated content)
    - Prims (3D objects/props)
    - Scripts (behavior scripts)
    - Stages (saved scenes)
    """

    assetSelected = pyqtSignal(str, str)  # (asset_type, asset_name)
    agentRezzed = pyqtSignal(str)  # Signal when agent is rezzed (triggers hierarchy refresh)
    generationSelected = pyqtSignal(str, dict)  # (gen_id, metadata)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.project_manager = None  # Will be set by main window
        self._generations_manager = None

        self._setup_ui()
        self._load_assets()

    def set_generations_manager(self, manager):
        """Connect to GenerationsManager for AI-generated content."""
        self._generations_manager = manager
        # Subscribe to new generations
        manager.on('generation_stored', self._on_generation_stored)
        self._load_generations()

    def _setup_ui(self):
        """Build UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Asset tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)

        # Enable drag
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)

        # Style to match Unity
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2b2b2b;
                color: #D2D2D2;
                border: none;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:hover {
                background-color: #3a3a3a;
            }
            QTreeWidget::item:selected {
                background-color: #2d5c8f;
            }
        """)

        layout.addWidget(self.tree)

    def _load_assets(self):
        """Load all assets from the project using PROJECT_SPEC.md structure."""
        self.tree.clear()

        # Check if project is open
        if not self.project_manager or not self.project_manager.is_project_open():
            # No project open - show message
            placeholder = QTreeWidgetItem(self.tree, ["No project open"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)
            placeholder_hint = QTreeWidgetItem(self.tree, ["File > New Project to get started"])
            placeholder_hint.setForeground(0, Qt.GlobalColor.darkGray)
            return

        # Create category nodes
        self.noodlings_node = QTreeWidgetItem(self.tree, ["Noodlings"])
        self.noodlings_node.setExpanded(True)

        self.stages_node = QTreeWidgetItem(self.tree, ["Stages"])
        self.stages_node.setExpanded(True)

        self.prims_node = QTreeWidgetItem(self.tree, ["Prims"])
        self.prims_node.setExpanded(False)

        self.generations_node = QTreeWidgetItem(self.tree, ["Generations"])
        self.generations_node.setExpanded(True)

        # Load Noodlings from project (new format)
        noodlings = self.project_manager.list_noodlings()
        if noodlings:
            for name in sorted(noodlings):
                item = QTreeWidgetItem(self.noodlings_node, [name])
                item.setData(0, Qt.ItemDataRole.UserRole, ("noodling", name, "project"))

                # Load metadata for tooltip
                noodling_path = self.project_manager.get_noodling_path(name)
                try:
                    import yaml
                    manifest_path = os.path.join(noodling_path, "noodling.yaml")
                    if os.path.exists(manifest_path):
                        with open(manifest_path, 'r') as f:
                            manifest = yaml.safe_load(f)
                        desc = manifest.get("description", "")
                        preview = manifest.get("preview", {})
                        species = preview.get("species", "noodling")
                        item.setToolTip(0, f"{species}: {desc}")
                except:
                    pass
        else:
            placeholder = QTreeWidgetItem(self.noodlings_node, ["(No noodlings yet)"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)

        # Also load legacy recipes from cmush/recipes
        cmush_recipes_path = os.path.join(
            os.path.dirname(__file__),
            "../../../cmush/recipes"
        )
        if os.path.exists(cmush_recipes_path):
            for filename in sorted(os.listdir(cmush_recipes_path)):
                if filename.endswith(".yaml"):
                    name = filename.replace(".yaml", "")
                    # Skip if already in project
                    if name not in noodlings:
                        item = QTreeWidgetItem(self.noodlings_node, [f"{name} (legacy)"])
                        item.setData(0, Qt.ItemDataRole.UserRole, ("noodling", name, "recipe"))
                        item.setForeground(0, Qt.GlobalColor.darkCyan)

        # Load Stages from project
        stages = self.project_manager.list_stages()
        if stages:
            for name in sorted(stages):
                item = QTreeWidgetItem(self.stages_node, [name])
                item.setData(0, Qt.ItemDataRole.UserRole, ("stage", name, "project"))

                # Load zone count for tooltip
                stage_path = self.project_manager.get_stage_path(name)
                zones_path = os.path.join(stage_path, "Zones")
                if os.path.exists(zones_path):
                    zone_count = len([f for f in os.listdir(zones_path) if f.endswith(".zone.yaml")])
                    instances_path = os.path.join(stage_path, "Instances")
                    inst_count = len(os.listdir(instances_path)) if os.path.exists(instances_path) else 0
                    item.setToolTip(0, f"{zone_count} zones, {inst_count} instances")
        else:
            placeholder = QTreeWidgetItem(self.stages_node, ["(No stages yet)"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)

        # Load Prims from project
        prims = self.project_manager.list_prims()
        if prims:
            for name in sorted(prims):
                item = QTreeWidgetItem(self.prims_node, [name])
                item.setData(0, Qt.ItemDataRole.UserRole, ("prim", name, "project"))
        else:
            placeholder = QTreeWidgetItem(self.prims_node, ["(No prims yet)"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)

        # Load Generations if manager is available
        self._load_generations()

    def _on_item_clicked(self, item, column):
        """Handle item click."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data:
            if len(data) == 3:
                asset_type, asset_name, source = data
            else:
                asset_type, asset_name = data
                source = "unknown"
            self.assetSelected.emit(asset_type, asset_name)

    def _show_context_menu(self, position):
        """Show right-click context menu."""
        item = self.tree.itemAt(position)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            # Clicked on category header
            return

        # Handle generation items (3-tuple with dict)
        if len(data) == 3 and data[0] == "generation":
            self._show_generation_context_menu(item, position)
            return

        # Handle both 2-tuple and 3-tuple data formats
        if len(data) == 3:
            asset_type, asset_name, source = data
        else:
            asset_type, asset_name = data
            source = "project"

        menu = QMenu(self)

        # Common actions for all assets
        if asset_type == "noodling":
            # PRIMARY ACTION: Add to Hierarchy
            add_action = QAction("Add to Hierarchy", self)
            add_action.triggered.connect(lambda: self._add_to_hierarchy(asset_name, source, fresh=False))
            menu.addAction(add_action)

            # Fresh rez (clears memory, uses recipe defaults)
            fresh_action = QAction("Add to Hierarchy (Fresh)", self)
            fresh_action.triggered.connect(lambda: self._add_to_hierarchy(asset_name, source, fresh=True))
            menu.addAction(fresh_action)

            menu.addSeparator()

            # Alternative: Rez in World (same thing, different wording)
            rez_action = QAction("Rez in World", self)
            rez_action.triggered.connect(lambda: self._add_to_hierarchy(asset_name, source, fresh=False))
            menu.addAction(rez_action)

            # Fresh version
            rez_fresh_action = QAction("Rez in World (Fresh)", self)
            rez_fresh_action.triggered.connect(lambda: self._add_to_hierarchy(asset_name, source, fresh=True))
            menu.addAction(rez_fresh_action)

            menu.addSeparator()

            # Edit recipe
            edit_action = QAction("Edit Recipe...", self)
            edit_action.triggered.connect(lambda: self._edit_noodling(asset_name, source))
            menu.addAction(edit_action)

            # View details
            view_action = QAction("View Details...", self)
            view_action.triggered.connect(lambda: self._view_noodling(asset_name, source))
            menu.addAction(view_action)

            duplicate_action = QAction("Duplicate", self)
            duplicate_action.setEnabled(False)  # TODO
            menu.addAction(duplicate_action)

            menu.addSeparator()

            delete_action = QAction("Delete from Assets", self)
            delete_action.setEnabled(False)  # TODO
            menu.addAction(delete_action)

        elif asset_type == "ensemble":
            load_action = QAction("Load Ensemble to Stage", self)
            load_action.triggered.connect(lambda: self._load_ensemble(asset_name))
            menu.addAction(load_action)

            menu.addSeparator()

            view_action = QAction("View Details...", self)
            view_action.triggered.connect(lambda: self._view_ensemble(asset_name))
            menu.addAction(view_action)

            edit_action = QAction("Edit Ensemble...", self)
            edit_action.setEnabled(False)  # TODO
            menu.addAction(edit_action)

            menu.addSeparator()

            delete_action = QAction("De-Rez", self)
            delete_action.setEnabled(False)  # TODO
            menu.addAction(delete_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _add_to_hierarchy(self, name, source="project", fresh=False):
        """
        Add a noodling to the hierarchy (spawn in world).

        Args:
            name: Recipe name
            source: "project" or "recipe" (determines where to load from)
            fresh: If True, use -f flag (clears memory, fresh state)
        """
        try:
            import json
            import yaml
            from datetime import datetime

            # Load recipe based on source
            if source == "recipe":
                # Load from cmush/recipes (YAML)
                recipes_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/recipes"
                )
                recipe_path = os.path.join(recipes_path, f"{name}.yaml")

                if not os.path.exists(recipe_path):
                    QMessageBox.warning(self, "Recipe Not Found", f"Can't find recipe: {name}.yaml")
                    return

                with open(recipe_path, 'r') as f:
                    recipe = yaml.safe_load(f)

            else:
                # Load from project assets (JSON)
                if not self.project_manager or not self.project_manager.is_project_open():
                    QMessageBox.warning(self, "No Project", "Open a project first.")
                    return

                noodlings_path = self.project_manager.get_assets_path("Noodlings")
                recipe_path = os.path.join(noodlings_path, f"{name}.json")

                if not os.path.exists(recipe_path):
                    QMessageBox.warning(self, "Recipe Not Found", f"Can't find recipe: {name}.json")
                    return

                with open(recipe_path, 'r') as f:
                    recipe = json.load(f)

            # Generate agent ID
            agent_id = f"agent_{name.lower()}"

            # Load current agents from noodleMUSH
            mush_base = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world"
            )
            agents_path = os.path.join(mush_base, "agents.json")

            with open(agents_path, 'r') as f:
                agents = json.load(f)

            # Check if already exists
            if agent_id in agents:
                QMessageBox.warning(self, "Already Rezzed", f"{name} is already in the world.")
                return

            # Create agent entry with all required fields for server
            agent_entry = {
                "name": recipe.get("name", name),
                "species": recipe.get("species", "human"),
                "pronouns": recipe.get("pronouns", "they/them"),
                "description": recipe.get("description", "A Noodling."),
                "personality": recipe.get("personality", ""),
                "voice": recipe.get("voice", ""),
                "perspective": recipe.get("perspective", ""),
                "checkpoint_path": "../../consilience_core/checkpoints_phase4/best_checkpoint.npz",
                "current_room": "room_000",
                "inventory": [],
                "created": datetime.now().isoformat()
            }

            # Add to agents
            agents[agent_id] = agent_entry

            # Save to agents.json (persistence)
            with open(agents_path, 'w') as f:
                json.dump(agents, f, indent=2)

            print(f"Rezzed {name} as {agent_id}")

            # Send @rez command to running server via HTTP API
            try:
                import requests
                api_url = "http://localhost:8081/api/command"  # Command endpoint

                # Build command with -f flag if fresh
                command = f"@rez -f {name}" if fresh else f"@rez {name}"

                payload = {
                    "user_id": "user_caity",  # Default user
                    "command": command
                }
                response = requests.post(api_url, json=payload, timeout=5)

                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✓ Rez command sent to server successfully")
                        print(f"  Server response: {result.get('output', '')[:100]}")
                    else:
                        print(f"⚠ Server rez failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"⚠ Server rez command failed: HTTP {response.status_code}")
            except Exception as cmd_error:
                print(f"⚠ Could not send rez command to server: {cmd_error}")
                print(f"  Agent added to agents.json but may need manual @rez")

            # Emit signal for hierarchy refresh
            self.agentRezzed.emit(agent_id)

            # Success message (no need to mention refresh - auto-refresh handles it)
            fresh_msg = " with fresh state (memory cleared)" if fresh else ""
            QMessageBox.information(
                self,
                "Rezzed!",
                f"{name} has been rezzed into noodleMUSH{fresh_msg}.\n\n{name} should appear in Scene Hierarchy momentarily!"
            )

        except Exception as e:
            QMessageBox.critical(self, "Rez Failed", f"Error: {e}")

    def _edit_noodling(self, name, source="project"):
        """Edit noodling recipe (placeholder)."""
        QMessageBox.information(
            self,
            "Edit Recipe",
            f"Feature in development\n\nWill open recipe editor for {name} (from {source})."
        )

    def _view_noodling(self, name, source="project"):
        """View noodling recipe details."""
        try:
            import json
            import yaml

            # Load recipe based on source
            if source == "recipe":
                recipes_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/recipes"
                )
                recipe_path = os.path.join(recipes_path, f"{name}.yaml")

                if not os.path.exists(recipe_path):
                    QMessageBox.warning(self, "Not Found", f"Recipe file not found: {name}.yaml")
                    return

                with open(recipe_path, 'r') as f:
                    recipe = yaml.safe_load(f)
            else:
                if not self.project_manager or not self.project_manager.is_project_open():
                    QMessageBox.warning(self, "No Project", "Open a project first.")
                    return

                noodlings_path = self.project_manager.get_assets_path("Noodlings")
                recipe_path = os.path.join(noodlings_path, f"{name}.json")

                if not os.path.exists(recipe_path):
                    QMessageBox.warning(self, "Not Found", f"Recipe not found: {name}.json")
                    return

                with open(recipe_path, 'r') as f:
                    recipe = json.load(f)

            # Build details display
            details = f"Name: {recipe.get('name', name)}\n"
            details += f"Species: {recipe.get('species', 'unknown')}\n"
            details += f"Pronouns: {recipe.get('pronouns', 'unknown')}\n\n"
            details += f"{recipe.get('description', 'No description')}\n\n"

            # Show cognitive components if present
            if recipe.get('cognitive_components'):
                details += "Cognitive Components:\n"
                for comp_name, comp_data in recipe['cognitive_components'].items():
                    comp_type = comp_data.get('type', 'unknown')
                    details += f"  - {comp_name}: {comp_type}\n"
                details += "\n"

            # Show personality if present
            personality = recipe.get('personalities', {}).get(f"agent_{name.lower()}")
            if personality:
                details += "Personality:\n"
                for trait, value in personality.items():
                    details += f"  - {trait}: {value}\n"
                details += "\n"

            QMessageBox.information(
                self,
                f"Recipe: {name}",
                details
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load recipe:\n{e}")

    def _load_ensemble(self, filename):
        """Load an ensemble to the stage (rez all agents)."""
        if not self.project_manager or not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        try:
            import json
            from datetime import datetime

            # Load ensemble file
            ensembles_path = self.project_manager.get_assets_path("Ensembles")
            ensemble_path = os.path.join(ensembles_path, filename)

            with open(ensemble_path, 'r') as f:
                ensemble = json.load(f)

            # Load current agents from noodleMUSH
            mush_base = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world"
            )
            agents_path = os.path.join(mush_base, "agents.json")

            with open(agents_path, 'r') as f:
                agents = json.load(f)

            # Rez each agent in the ensemble
            rezzed = []
            skipped = []

            for agent_recipe in ensemble.get("agents", []):
                name = agent_recipe.get("name", "Unknown")
                agent_id = f"agent_{name.lower().replace(' ', '_')}"

                # Skip if already exists
                if agent_id in agents:
                    skipped.append(name)
                    continue

                # Create agent entry with ensemble metadata AND all required fields
                agent_entry = {
                    "name": name,
                    "species": agent_recipe.get("species", "human"),
                    "pronouns": agent_recipe.get("pronouns", "they/them"),
                    "description": agent_recipe.get("description", "A Noodling."),
                    "personality": agent_recipe.get("personality", ""),
                    "voice": agent_recipe.get("voice", ""),
                    "perspective": agent_recipe.get("perspective", ""),
                    "created": datetime.now().isoformat(),
                    "current_room": "room_000",  # Start in Nexus
                    "inventory": [],  # Empty inventory
                    "checkpoint_path": f"world/agents/{agent_id}/checkpoint.npz",
                    "state_path": f"world/agents/{agent_id}/agent_state.json",
                    "ensemble": {
                        "name": ensemble.get("name"),
                        "type": ensemble.get("ensemble_type"),
                        "mission": ensemble.get("shared_mission"),
                        "dynamics": ensemble.get("ensemble_dynamics"),
                        "knowledge": ensemble.get("shared_knowledge"),
                        "role": ensemble.get("ensemble_dynamics", {}).get("role_distribution", {}).get(name, "member")
                    }
                }

                agents[agent_id] = agent_entry
                rezzed.append(name)

            # Save
            with open(agents_path, 'w') as f:
                json.dump(agents, f, indent=2)

            # Show results
            message = f"Rezzed {len(rezzed)} agents from {ensemble.get('name', filename)}\n\n"
            if rezzed:
                message += "Rezzed:\n" + "\n".join(f"  - {n}" for n in rezzed)
            if skipped:
                message += "\n\nAlready in world:\n" + "\n".join(f"  - {n}" for n in skipped)
            message += "\n\nRefresh Scene Hierarchy to see them."

            QMessageBox.information(self, "Ensemble Loaded!", message)

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Error: {e}")

    def _view_ensemble(self, filename):
        """View ensemble details."""
        if not self.project_manager or not self.project_manager.is_project_open():
            return

        ensembles_path = self.project_manager.get_assets_path("Ensembles")
        filepath = os.path.join(ensembles_path, filename)

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            agents = data.get("agents", [])
            agent_names = [a.get("name", "Unknown") for a in agents]

            details = f"{data.get('name', 'Unknown')}\n\n"
            details += f"{data.get('description', 'No description')}\n\n"
            details += f"Agents ({len(agents)}):\n"
            details += "\n".join(f"  - {name}" for name in agent_names)

            QMessageBox.information(
                self,
                "Ensemble Details",
                details
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to load ensemble details:\n{e}"
            )

    def refresh(self):
        """Refresh the asset list."""
        self._load_assets()

    # ========== Generations ==========

    def _load_generations(self):
        """Load AI-generated content into the Generations category."""
        if not hasattr(self, 'generations_node'):
            return

        # Clear existing generation items
        while self.generations_node.childCount() > 0:
            self.generations_node.takeChild(0)

        if not self._generations_manager:
            placeholder = QTreeWidgetItem(self.generations_node, ["(No generations yet)"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)
            return

        # Get recent generations (grouped by source)
        generations = self._generations_manager.get_recent(50)

        if not generations:
            placeholder = QTreeWidgetItem(self.generations_node, ["(No generations yet)"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)
            return

        # Group by source
        by_source = {}
        for gen in generations:
            source = gen.source or 'unknown'
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(gen)

        # Create source folders
        source_display = {
            'subconscious': 'Subconscious Dreams',
            'scripted': 'Scripted Facets',
            'manual': 'Manual Generations',
            'unknown': 'Other'
        }

        for source, gens in by_source.items():
            display_name = source_display.get(source, source.title())
            source_node = QTreeWidgetItem(
                self.generations_node,
                [f"{display_name} ({len(gens)})"]
            )
            source_node.setExpanded(True)

            # Add individual generations
            for gen in gens:
                # Format display name
                if gen.agent:
                    display = f"{gen.agent}: {gen.prompt[:30]}..."
                else:
                    display = f"{gen.prompt[:40]}..."

                item = QTreeWidgetItem(source_node, [display])
                item.setData(0, Qt.ItemDataRole.UserRole, ("generation", gen.id, gen.to_dict()))
                item.setToolTip(0, f"{gen.prompt}\n\nCreated: {gen.created_at}\nStyle: {gen.style}")

                # Add thumbnail if available
                if gen.thumbnail_path and os.path.exists(gen.thumbnail_path):
                    try:
                        pixmap = QPixmap(gen.thumbnail_path)
                        if not pixmap.isNull():
                            item.setIcon(0, QIcon(pixmap.scaled(
                                24, 24,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation
                            )))
                    except Exception:
                        pass

        # Update count in header
        total = len(generations)
        self.generations_node.setText(0, f"Generations ({total})")

    def _on_generation_stored(self, data: dict):
        """Handle new generation stored event."""
        # Refresh the generations list
        self._load_generations()

    def _show_generation_context_menu(self, item, position):
        """Show context menu for generation item."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "generation":
            return

        gen_id = data[1]
        metadata = data[2]

        menu = QMenu(self)

        # View action
        view_action = QAction("View Image", self)
        view_action.triggered.connect(lambda: self._view_generation(gen_id, metadata))
        menu.addAction(view_action)

        # Open in folder
        open_folder_action = QAction("Show in Folder", self)
        open_folder_action.triggered.connect(lambda: self._open_generation_folder(metadata))
        menu.addAction(open_folder_action)

        menu.addSeparator()

        # Copy prompt
        copy_prompt_action = QAction("Copy Prompt", self)
        copy_prompt_action.triggered.connect(lambda: self._copy_generation_prompt(metadata))
        menu.addAction(copy_prompt_action)

        # Details
        details_action = QAction("View Details...", self)
        details_action.triggered.connect(lambda: self._view_generation_details(metadata))
        menu.addAction(details_action)

        menu.addSeparator()

        # Delete
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_generation(gen_id))
        menu.addAction(delete_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _view_generation(self, gen_id: str, metadata: dict):
        """Open generation image in default viewer."""
        filepath = metadata.get('filepath', '')
        if filepath and os.path.exists(filepath):
            import subprocess
            import sys
            if sys.platform == 'darwin':
                subprocess.run(['open', filepath])
            elif sys.platform == 'win32':
                os.startfile(filepath)
            else:
                subprocess.run(['xdg-open', filepath])
        else:
            QMessageBox.warning(self, "Not Found", "Image file not found.")

    def _open_generation_folder(self, metadata: dict):
        """Open containing folder in file manager."""
        filepath = metadata.get('filepath', '')
        if filepath:
            folder = os.path.dirname(filepath)
            if os.path.exists(folder):
                import subprocess
                import sys
                if sys.platform == 'darwin':
                    subprocess.run(['open', folder])
                elif sys.platform == 'win32':
                    os.startfile(folder)
                else:
                    subprocess.run(['xdg-open', folder])

    def _copy_generation_prompt(self, metadata: dict):
        """Copy prompt to clipboard."""
        from PyQt6.QtWidgets import QApplication
        prompt = metadata.get('prompt', '')
        if prompt:
            QApplication.clipboard().setText(prompt)

    def _view_generation_details(self, metadata: dict):
        """Show generation details dialog."""
        details = f"ID: {metadata.get('id', 'unknown')}\n"
        details += f"Source: {metadata.get('source', 'unknown')}\n"
        details += f"Agent: {metadata.get('agent', 'none')}\n"
        details += f"Created: {metadata.get('created_at', 'unknown')}\n\n"
        details += f"Prompt:\n{metadata.get('prompt', 'none')}\n\n"
        details += f"Style: {metadata.get('style', 'none')}\n"
        details += f"Size: {metadata.get('width', 0)}x{metadata.get('height', 0)}\n\n"

        if metadata.get('symbolic_text'):
            details += f"Symbolic Text:\n{metadata.get('symbolic_text')}\n\n"

        if metadata.get('emotional_signature'):
            sig = metadata['emotional_signature']
            details += "Emotional Signature:\n"
            for k, v in sig.items():
                details += f"  {k}: {v:.2f}\n"

        QMessageBox.information(self, "Generation Details", details)

    def _delete_generation(self, gen_id: str):
        """Delete a generation."""
        if not self._generations_manager:
            return

        reply = QMessageBox.question(
            self,
            "Delete Generation",
            "Are you sure you want to delete this generation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._generations_manager.delete_generation(gen_id):
                self._load_generations()
            else:
                QMessageBox.warning(self, "Error", "Failed to delete generation.")
