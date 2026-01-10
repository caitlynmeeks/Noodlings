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
#   Assets Panel - Filesystem browser for project assets (Unity-style).
#
#   Shows the actual project folder structure: - Noodlings/ -...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.assets_panel
# PURPOSE:  assets panel panel UI
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AssetsPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QMessageBox, QAbstractItemView, QLineEdit, QInputDialog,
    QFileDialog, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QFileSystemWatcher
from PyQt6.QtGui import QAction, QIcon, QFont
import os
import shutil
from pathlib import Path


# File type to icon mapping (using text icons for now)
FILE_ICONS = {
    # Folders
    'folder': '\U0001F4C1',  # Folder icon
    'folder_open': '\U0001F4C2',  # Open folder

    # Noodlings assets
    '.yaml': '\U0001F4C4',  # Document
    '.yml': '\U0001F4C4',

    # 3D assets
    '.radiance': '\U0001F30C',  # Galaxy (gaussian splat)
    '.ply': '\U0001F30C',
    '.vrm': '\U0001F464',  # Person (avatar)
    '.glb': '\U0001F4E6',  # Package (3D model)
    '.gltf': '\U0001F4E6',
    '.obj': '\U0001F4E6',
    '.fbx': '\U0001F4E6',

    # Images
    '.png': '\U0001F5BC',  # Image
    '.jpg': '\U0001F5BC',
    '.jpeg': '\U0001F5BC',
    '.webp': '\U0001F5BC',

    # Audio
    '.mp3': '\U0001F3B5',  # Music note
    '.wav': '\U0001F3B5',
    '.ogg': '\U0001F3B5',

    # Scripts
    '.py': '\U0001F40D',  # Snake (Python)
    '.js': '\U0001F4DC',  # Script

    # Neural canvas
    '.nncanvas': '\U0001F9E0',  # Brain

    # Default
    'default': '\U0001F4C4',  # Document
}

# Folders to hide from the tree
HIDDEN_FOLDERS = {'Library', '.git', '__pycache__', '.DS_Store'}


class AssetsPanel(QWidget):
    """
    Assets panel - Unity-style filesystem browser for project assets.

    Shows actual folder/file structure with:
    - Expandable folder tree
    - File icons by type
    - Context menu operations (New Folder, Rename, Delete, Import)
    - Drag-drop file operations
    - File system watching for external changes
    """

    # Signals
    assetSelected = pyqtSignal(str, str)  # (asset_type, asset_path)
    assetDoubleClicked = pyqtSignal(str, str)  # (asset_type, asset_path)
    agentRezzed = pyqtSignal(str)  # For legacy compatibility
    generationSelected = pyqtSignal(str, dict)  # For GenerationsManager
    gaussianSelected = pyqtSignal(str, dict)  # For gaussian assets
    assetRenamed = pyqtSignal(str, str, str)  # (asset_type, old_path, new_path)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.project_manager = None
        self._generations_manager = None
        self._gaussian_manager = None

        # Track expanded folders
        self._expanded_paths: set[str] = set()

        # File system watcher for auto-refresh
        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_directory_changed)

        # Debounce refresh
        self._pending_refresh = False

        self._setup_ui()

    def set_project_manager(self, manager):
        """Connect to ProjectManager."""
        self.project_manager = manager
        self._load_assets()

    def set_generations_manager(self, manager):
        """Connect to GenerationsManager for AI-generated content."""
        self._generations_manager = manager
        if hasattr(manager, 'on'):
            manager.on('generation_stored', self._on_generation_stored)

    def set_gaussian_manager(self, manager):
        """Connect to GaussianAssetManager."""
        self._gaussian_manager = manager

    def _setup_ui(self):
        """Build UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)
        toolbar.setSpacing(4)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(60)
        refresh_btn.clicked.connect(self._load_assets)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                color: #D2D2D2;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()

        # Path label
        self._path_label = QLabel("")
        self._path_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._path_label)

        layout.addLayout(toolbar)

        # Asset tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemExpanded.connect(self._on_item_expanded)
        self.tree.itemCollapsed.connect(self._on_item_collapsed)

        # Enable inline editing
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.itemChanged.connect(self._on_item_changed)

        # Enable drag-drop
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        # Style
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #252526;
                border: none;
                color: #D2D2D2;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 4px 2px;
                border-radius: 3px;
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
        """Load project filesystem into tree."""
        # Save state before clearing
        self._save_expanded_state()
        selected_path = self._get_selected_path()

        self.tree.clear()

        # Clear watcher
        dirs = self._watcher.directories()
        if dirs:
            self._watcher.removePaths(dirs)

        # Check if project is open
        if not self.project_manager or not self.project_manager.is_project_open():
            placeholder = QTreeWidgetItem(self.tree, ["No project open"])
            placeholder.setForeground(0, Qt.GlobalColor.gray)
            hint = QTreeWidgetItem(self.tree, ["File > New Project to get started"])
            hint.setForeground(0, Qt.GlobalColor.darkGray)
            return

        project_path = self.project_manager.current_project_path
        self._path_label.setText(os.path.basename(project_path))

        # Add project root folders
        self._populate_folder(self.tree.invisibleRootItem(), project_path, depth=0)

        # Restore state
        self._restore_expanded_state()
        if selected_path:
            self._restore_selection(selected_path)

    def _populate_folder(self, parent_item, folder_path: str, depth: int = 0):
        """Recursively populate tree with folder contents."""
        if depth > 10:  # Safety limit
            return

        try:
            entries = sorted(os.listdir(folder_path))
        except PermissionError:
            return

        # Separate folders and files
        folders = []
        files = []

        for entry in entries:
            if entry in HIDDEN_FOLDERS or entry.startswith('.'):
                continue

            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                folders.append(entry)
            else:
                files.append(entry)

        # Add folders first
        for folder_name in folders:
            full_path = os.path.join(folder_path, folder_name)
            item = self._create_folder_item(folder_name, full_path)

            if isinstance(parent_item, QTreeWidget):
                parent_item.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

            # Watch this directory
            self._watcher.addPath(full_path)

            # Recursively populate (lazy load would be better for large projects)
            self._populate_folder(item, full_path, depth + 1)

        # Add files
        for file_name in files:
            full_path = os.path.join(folder_path, file_name)
            item = self._create_file_item(file_name, full_path)

            if isinstance(parent_item, QTreeWidget):
                parent_item.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

    def _create_folder_item(self, name: str, path: str) -> QTreeWidgetItem:
        """Create a tree item for a folder."""
        item = QTreeWidgetItem([f"{FILE_ICONS['folder']} {name}"])
        item.setData(0, Qt.ItemDataRole.UserRole, {'path': path, 'is_folder': True})
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

    def _create_file_item(self, name: str, path: str) -> QTreeWidgetItem:
        """Create a tree item for a file."""
        ext = os.path.splitext(name)[1].lower()
        icon = FILE_ICONS.get(ext, FILE_ICONS['default'])

        item = QTreeWidgetItem([f"{icon} {name}"])
        item.setData(0, Qt.ItemDataRole.UserRole, {'path': path, 'is_folder': False})
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

    def _save_expanded_state(self):
        """Save which folders are expanded."""
        self._expanded_paths.clear()

        def save_recursive(item):
            if item is None:
                return
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('is_folder') and item.isExpanded():
                self._expanded_paths.add(data['path'])
            for i in range(item.childCount()):
                save_recursive(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            save_recursive(self.tree.topLevelItem(i))

    def _restore_expanded_state(self):
        """Restore expanded folders."""
        def restore_recursive(item):
            if item is None:
                return
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('path') in self._expanded_paths:
                item.setExpanded(True)
            for i in range(item.childCount()):
                restore_recursive(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            restore_recursive(self.tree.topLevelItem(i))

    def _get_selected_path(self) -> str | None:
        """Get path of currently selected item."""
        item = self.tree.currentItem()
        if item:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                return data.get('path')
        return None

    def _restore_selection(self, path: str):
        """Restore selection by path."""
        def find_recursive(item):
            if item is None:
                return None
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('path') == path:
                return item
            for i in range(item.childCount()):
                found = find_recursive(item.child(i))
                if found:
                    return found
            return None

        for i in range(self.tree.topLevelItemCount()):
            item = find_recursive(self.tree.topLevelItem(i))
            if item:
                self.tree.setCurrentItem(item)
                return

    def _on_directory_changed(self, path: str):
        """Handle external filesystem changes."""
        # Log what triggered the refresh (for debugging)
        print(f"[AssetsPanel] Directory changed: {path}")

        # Debounce multiple rapid changes
        if not self._pending_refresh:
            self._pending_refresh = True
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self._do_refresh)

    def _do_refresh(self):
        """Perform debounced refresh."""
        self._pending_refresh = False
        self._load_assets()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click - emit selection signal."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        path = data['path']
        is_folder = data['is_folder']

        if is_folder:
            # Check if folder is a special asset type (noodling or stage)
            folder_type = self._get_folder_asset_type(path)
            self.assetSelected.emit(folder_type, path)
        else:
            # Determine asset type from path/extension
            asset_type = self._get_asset_type(path)
            self.assetSelected.emit(asset_type, path)

            # Special handling for specific types
            if asset_type == 'radiance' and self._gaussian_manager:
                self.gaussianSelected.emit(os.path.basename(path), {'path': path})
            elif asset_type == 'generation' and self._generations_manager:
                self.generationSelected.emit(os.path.basename(path), {'path': path})

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click - open asset or expand folder."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        path = data['path']
        is_folder = data['is_folder']

        if is_folder:
            # Toggle expansion
            item.setExpanded(not item.isExpanded())
        else:
            # Open/import the asset
            asset_type = self._get_asset_type(path)
            self.assetDoubleClicked.emit(asset_type, path)
            self._open_asset(path, asset_type)

    def _on_item_expanded(self, item: QTreeWidgetItem):
        """Update folder icon when expanded."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get('is_folder'):
            name = os.path.basename(data['path'])
            item.setText(0, f"{FILE_ICONS['folder_open']} {name}")

    def _on_item_collapsed(self, item: QTreeWidgetItem):
        """Update folder icon when collapsed."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get('is_folder'):
            name = os.path.basename(data['path'])
            item.setText(0, f"{FILE_ICONS['folder']} {name}")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle inline rename."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        old_path = data['path']
        is_folder = data['is_folder']

        # Extract new name (remove icon prefix)
        new_text = item.text(0)
        # Find first space after emoji and get the rest
        parts = new_text.split(' ', 1)
        if len(parts) > 1:
            new_name = parts[1].strip()
        else:
            new_name = new_text.strip()

        old_name = os.path.basename(old_path)

        if new_name and new_name != old_name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)

            try:
                os.rename(old_path, new_path)
                data['path'] = new_path
                item.setData(0, Qt.ItemDataRole.UserRole, data)

                # Update icon
                if is_folder:
                    icon = FILE_ICONS['folder_open'] if item.isExpanded() else FILE_ICONS['folder']
                else:
                    ext = os.path.splitext(new_name)[1].lower()
                    icon = FILE_ICONS.get(ext, FILE_ICONS['default'])
                item.setText(0, f"{icon} {new_name}")

                asset_type = 'folder' if is_folder else self._get_asset_type(new_path)
                self.assetRenamed.emit(asset_type, old_path, new_path)

            except OSError as e:
                QMessageBox.warning(self, "Rename Failed", f"Could not rename: {e}")
                # Restore original name
                icon = FILE_ICONS['folder'] if is_folder else FILE_ICONS.get(
                    os.path.splitext(old_name)[1].lower(), FILE_ICONS['default']
                )
                item.setText(0, f"{icon} {old_name}")

    def _get_folder_asset_type(self, path: str) -> str:
        """
        Determine if a folder is a special asset type.

        Checks for:
        - Noodling folder: contains recipe.yaml
        - Stage folder: contains stage.yaml

        Returns:
            'noodling', 'stage', or 'folder'
        """
        # Check if this is a noodling folder (contains recipe.yaml)
        recipe_path = os.path.join(path, 'recipe.yaml')
        if os.path.exists(recipe_path):
            return 'noodling'

        # Check if this is a stage folder (contains stage.yaml)
        stage_path = os.path.join(path, 'stage.yaml')
        if os.path.exists(stage_path):
            return 'stage'

        return 'folder'

    def _get_asset_type(self, path: str) -> str:
        """Determine asset type from path."""
        ext = os.path.splitext(path)[1].lower()

        # Check by extension
        if ext in ('.radiance', '.ply'):
            return 'radiance'
        elif ext == '.vrm':
            return 'vrm'
        elif ext in ('.glb', '.gltf', '.obj', '.fbx'):
            return 'mesh'
        elif ext in ('.png', '.jpg', '.jpeg', '.webp'):
            return 'image'
        elif ext in ('.mp3', '.wav', '.ogg'):
            return 'audio'
        elif ext in ('.yaml', '.yml'):
            # Check path for more specific type
            if '/Noodlings/' in path or '\\Noodlings\\' in path:
                return 'noodling'
            elif '/Stages/' in path or '\\Stages\\' in path:
                return 'stage'
            elif '/Zones/' in path or '\\Zones\\' in path:
                return 'zone'
            return 'yaml'
        elif ext == '.nncanvas':
            return 'neural_canvas'
        elif ext == '.py':
            return 'script'
        elif ext == '.js':
            return 'script'

        # Check by path
        if '/Generations/' in path or '\\Generations\\' in path:
            return 'generation'

        return 'file'

    def _open_asset(self, path: str, asset_type: str):
        """Open or import an asset."""
        if asset_type == 'noodling':
            # Select in inspector
            pass  # Handled by signal
        elif asset_type == 'stage':
            # Open stage
            pass  # Handled by signal
        elif asset_type == 'radiance':
            # Load in gaussian viewer
            pass  # Handled by signal
        elif asset_type in ('image', 'audio'):
            # Open with system default
            import subprocess
            subprocess.run(['open', path])
        elif asset_type == 'neural_canvas':
            # Open in neural canvas editor
            pass  # Handled by signal
        elif asset_type in ('yaml', 'script'):
            # Open in external editor
            import subprocess
            subprocess.run(['open', path])

    def _show_context_menu(self, position):
        """Show context menu for asset operations."""
        item = self.tree.itemAt(position)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #555;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #3a3a3a;
            }
            QMenu::separator {
                height: 1px;
                background: #555;
                margin: 4px 8px;
            }
        """)

        if item:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                path = data['path']
                is_folder = data['is_folder']

                if is_folder:
                    # Folder context menu
                    new_folder_action = menu.addAction("New Folder")
                    new_folder_action.triggered.connect(lambda: self._create_new_folder(path))

                    menu.addSeparator()

                    import_action = menu.addAction("Import Asset...")
                    import_action.triggered.connect(lambda: self._import_asset(path))

                    menu.addSeparator()
                else:
                    # File context menu
                    open_action = menu.addAction("Open")
                    open_action.triggered.connect(lambda: self._open_asset(path, self._get_asset_type(path)))

                    menu.addSeparator()

                # Common actions
                rename_action = menu.addAction("Rename")
                rename_action.triggered.connect(lambda: self._start_rename(item))

                delete_action = menu.addAction("Delete")
                delete_action.triggered.connect(lambda: self._delete_item(item))

                menu.addSeparator()

                reveal_action = menu.addAction("Reveal in Finder")
                reveal_action.triggered.connect(lambda: self._reveal_in_finder(path))
        else:
            # Empty area context menu
            if self.project_manager and self.project_manager.is_project_open():
                project_path = self.project_manager.current_project_path

                new_folder_action = menu.addAction("New Folder")
                new_folder_action.triggered.connect(lambda: self._create_new_folder(project_path))

                menu.addSeparator()

                import_action = menu.addAction("Import Asset...")
                import_action.triggered.connect(lambda: self._import_asset(project_path))

        menu.addSeparator()
        refresh_action = menu.addAction("Refresh")
        refresh_action.triggered.connect(self._load_assets)

        menu.exec(self.tree.mapToGlobal(position))

    def _create_new_folder(self, parent_path: str):
        """Create a new folder."""
        name, ok = QInputDialog.getText(
            self, "New Folder", "Folder name:",
            text="New Folder"
        )

        if ok and name:
            new_path = os.path.join(parent_path, name)
            try:
                os.makedirs(new_path, exist_ok=True)
                self._load_assets()
            except OSError as e:
                QMessageBox.warning(self, "Error", f"Could not create folder: {e}")

    def _import_asset(self, target_folder: str):
        """Import an asset file."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Assets",
            "",
            "All Files (*);;3D Models (*.vrm *.glb *.gltf *.obj *.fbx);;Images (*.png *.jpg *.jpeg);;Audio (*.mp3 *.wav *.ogg)"
        )

        if files:
            for file_path in files:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(target_folder, file_name)

                try:
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    QMessageBox.warning(self, "Import Error", f"Could not import {file_name}: {e}")

            self._load_assets()

    def _start_rename(self, item: QTreeWidgetItem):
        """Start inline rename for an item."""
        self.tree.editItem(item, 0)

    def _delete_item(self, item: QTreeWidgetItem):
        """Delete a file or folder."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        path = data['path']
        is_folder = data['is_folder']
        name = os.path.basename(path)

        type_str = "folder" if is_folder else "file"
        reply = QMessageBox.question(
            self,
            f"Delete {type_str.title()}",
            f"Are you sure you want to delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if is_folder:
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                self._load_assets()
            except OSError as e:
                QMessageBox.warning(self, "Delete Failed", f"Could not delete: {e}")

    def _reveal_in_finder(self, path: str):
        """Reveal file/folder in Finder (macOS)."""
        import subprocess
        subprocess.run(['open', '-R', path])

    def _on_generation_stored(self, gen_id: str, metadata: dict):
        """Handle new AI generation stored."""
        # Refresh to show new generation
        self._load_assets()

    # Legacy compatibility methods
    def refresh(self):
        """Refresh asset tree."""
        self._load_assets()

    def refresh_assets(self):
        """Refresh asset tree (legacy compatibility)."""
        self._load_assets()

    def select_asset(self, asset_type: str, asset_name: str):
        """Select an asset by type and name (legacy compatibility)."""
        # Find and select the item
        def find_recursive(item):
            for i in range(item.childCount()):
                child = item.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data:
                    if asset_name in data['path']:
                        self.tree.setCurrentItem(child)
                        return True
                if find_recursive(child):
                    return True
            return False

        for i in range(self.tree.topLevelItemCount()):
            if find_recursive(self.tree.topLevelItem(i)):
                break

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
