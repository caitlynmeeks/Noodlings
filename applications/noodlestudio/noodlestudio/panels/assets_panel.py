"""
Assets Panel - Shows all project assets (Noodlings, Ensembles, Prims, Scripts).

Organizes assets by type with expandable categories.
Right-click context menus for asset management (to be implemented).
"""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
import os
import json


class AssetsPanel(QDockWidget):
    """
    Assets panel showing all project assets organized by type.

    Categories:
    - Noodlings (individual agents)
    - Ensembles (groups of agents)
    - Prims (3D objects/props)
    - Scripts (behavior scripts)
    - Stages (saved scenes)
    """

    assetSelected = pyqtSignal(str, str)  # (asset_type, asset_name)

    def __init__(self, parent=None):
        super().__init__("Assets", parent)

        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self.project_manager = None  # Will be set by main window

        self._setup_ui()
        self._load_assets()

    def _setup_ui(self):
        """Build UI components."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Asset tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)

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
        self.setWidget(container)

    def _load_assets(self):
        """Load all assets from the project."""
        self.tree.clear()

        # Check if project is open
        if not self.project_manager or not self.project_manager.is_project_open():
            # No project open - show message
            placeholder = QTreeWidgetItem(self.tree, ["No project open"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)
            placeholder_hint = QTreeWidgetItem(self.tree, ["File > New Project to get started"])
            placeholder_hint.setForeground(0, Qt.GlobalColor.darkGray)
            return

        # Get assets path from project
        assets_path = self.project_manager.get_assets_path()
        if not assets_path or not os.path.exists(assets_path):
            return

        # Create category nodes
        self.noodlings_node = QTreeWidgetItem(self.tree, ["Noodlings"])
        self.noodlings_node.setExpanded(True)

        self.ensembles_node = QTreeWidgetItem(self.tree, ["Ensembles"])
        self.ensembles_node.setExpanded(True)

        self.prims_node = QTreeWidgetItem(self.tree, ["Prims"])
        self.prims_node.setExpanded(False)

        self.scripts_node = QTreeWidgetItem(self.tree, ["Scripts"])
        self.scripts_node.setExpanded(False)

        self.stages_node = QTreeWidgetItem(self.tree, ["Stages"])
        self.stages_node.setExpanded(False)

        # Load Noodlings
        noodlings_path = self.project_manager.get_assets_path("Noodlings")
        if os.path.exists(noodlings_path):
            for filename in sorted(os.listdir(noodlings_path)):
                if filename.endswith(".json"):
                    name = filename.replace(".json", "")
                    item = QTreeWidgetItem(self.noodlings_node, [name])
                    item.setData(0, Qt.ItemDataRole.UserRole, ("noodling", name))

        # Load Ensembles
        ensembles_path = self.project_manager.get_assets_path("Ensembles")
        if os.path.exists(ensembles_path):
            for filename in sorted(os.listdir(ensembles_path)):
                if filename.endswith(".ensemble"):
                    # Load ensemble to get display name
                    filepath = os.path.join(ensembles_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            name = data.get("name", filename.replace(".ensemble", ""))
                            item = QTreeWidgetItem(self.ensembles_node, [name])
                            item.setData(0, Qt.ItemDataRole.UserRole, ("ensemble", filename))

                            # Add agent count as tooltip
                            agent_count = len(data.get("agents", []))
                            item.setToolTip(0, f"{agent_count} agents: {data.get('description', '')}")
                    except Exception as e:
                        print(f"Error loading ensemble {filename}: {e}")

        # Prims (placeholder - will be implemented when USD integration is complete)
        placeholder_prims = QTreeWidgetItem(self.prims_node, ["(Coming soon)"])
        placeholder_prims.setForeground(0, Qt.GlobalColor.gray)

        # Scripts (placeholder)
        placeholder_scripts = QTreeWidgetItem(self.scripts_node, ["(Coming soon)"])
        placeholder_scripts.setForeground(0, Qt.GlobalColor.gray)

        # Stages (placeholder)
        placeholder_stages = QTreeWidgetItem(self.stages_node, ["(Coming soon)"])
        placeholder_stages.setForeground(0, Qt.GlobalColor.gray)

    def _on_item_clicked(self, item, column):
        """Handle item click."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data:
            asset_type, asset_name = data
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

        asset_type, asset_name = data

        menu = QMenu(self)

        # Common actions for all assets
        if asset_type == "noodling":
            spawn_action = QAction("Spawn in World", self)
            spawn_action.triggered.connect(lambda: self._spawn_noodling(asset_name))
            menu.addAction(spawn_action)

            menu.addSeparator()

            edit_action = QAction("Edit Recipe...", self)
            edit_action.triggered.connect(lambda: self._edit_noodling(asset_name))
            menu.addAction(edit_action)

            duplicate_action = QAction("Duplicate", self)
            duplicate_action.setEnabled(False)  # TODO
            menu.addAction(duplicate_action)

            menu.addSeparator()

            delete_action = QAction("Delete", self)
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

            delete_action = QAction("Delete", self)
            delete_action.setEnabled(False)  # TODO
            menu.addAction(delete_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _spawn_noodling(self, name):
        """Spawn a noodling in the world (placeholder)."""
        QMessageBox.information(
            self,
            "Spawn Noodling",
            f"Feature in development\n\nWill spawn {name} into noodleMUSH world."
        )

    def _edit_noodling(self, name):
        """Edit noodling recipe (placeholder)."""
        QMessageBox.information(
            self,
            "Edit Recipe",
            f"Feature in development\n\nWill open recipe editor for {name}."
        )

    def _load_ensemble(self, filename):
        """Load an ensemble to the stage (placeholder)."""
        QMessageBox.information(
            self,
            "Load Ensemble",
            f"Feature in development\n\nWill load {filename} to stage.\n"
            f"All agents will be spawned and ready for interaction."
        )

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
