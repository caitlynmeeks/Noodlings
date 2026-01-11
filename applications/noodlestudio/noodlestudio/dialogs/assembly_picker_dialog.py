# ──────────────────────────────────────────────────────────────
#   Assembly Picker Dialog
#
#   Browse and select facet assembly files from the project.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.dialogs.assembly_picker_dialog
# PURPOSE:  Visual assembly file picker
# LAYER:    Studio / Dialogs
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from pathlib import Path
from typing import Optional, List
import yaml

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTreeWidget, QTreeWidgetItem, QWidget,
    QSplitter, QTextEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class AssemblyPickerDialog(QDialog):
    """
    Dialog for browsing and selecting facet assembly files.

    Displays a tree of assembly files found in the project, with
    preview of the selected assembly's metadata.

    Usage:
        dialog = AssemblyPickerDialog(project_path, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_path = dialog.selected_assembly
    """

    def __init__(
        self,
        project_path: Path,
        current_value: str = "",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.selected_assembly: str = current_value
        self._assemblies: List[Path] = []

        self.setWindowTitle("Select Assembly")
        self.setModal(True)
        self.resize(600, 450)

        self._build_ui()
        self._scan_assemblies()
        self._apply_styling()

        # Select current value if provided
        if current_value:
            self._select_path(current_value)

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Search/filter box
        search_layout = QHBoxLayout()
        search_label = QLabel("Filter:")
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Type to filter assemblies...")
        self._search_input.textChanged.connect(self._filter_tree)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self._search_input)
        layout.addLayout(search_layout)

        # Splitter with tree and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Tree view
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Assembly Files"])
        self._tree.setIndentation(20)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        splitter.addWidget(self._tree)

        # Preview panel
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_name = QLabel("")
        self._preview_name.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        preview_layout.addWidget(self._preview_name)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setFont(QFont("Menlo", 10))
        preview_layout.addWidget(self._preview_text)

        splitter.addWidget(preview_group)
        splitter.setSizes([300, 300])

        layout.addWidget(splitter, 1)

        # Selected path display
        path_layout = QHBoxLayout()
        path_label = QLabel("Selected:")
        self._path_display = QLineEdit()
        self._path_display.setReadOnly(True)
        self._path_display.setText(self.selected_assembly)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self._path_display)
        layout.addLayout(path_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        self._select_btn = QPushButton("Select")
        self._select_btn.setDefault(True)
        self._select_btn.clicked.connect(self._on_select)
        self._select_btn.setEnabled(False)
        button_layout.addWidget(self._select_btn)

        layout.addLayout(button_layout)

    def _apply_styling(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit {
                background: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border-color: #666666;
            }
            QLineEdit:read-only {
                background: #252525;
                color: #aaaaaa;
            }
            QTreeWidget {
                background: #252525;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #cccccc;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background: #3a3a3a;
            }
            QTreeWidget::item:hover {
                background: #333333;
            }
            QTextEdit {
                background: #252525;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #aaaaaa;
            }
            QGroupBox {
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 8px 20px;
                background: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #cccccc;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
            QPushButton:pressed {
                background: #333333;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
            }
        """)

    def _scan_assemblies(self):
        """Scan project for assembly files."""
        self._assemblies = []

        # Look for .yaml files that contain assembly definitions
        search_paths = [
            self.project_path / "assemblies",
            self.project_path / "noodlings",
            self.project_path,
        ]

        for search_path in search_paths:
            if search_path.exists():
                for yaml_file in search_path.rglob("*.yaml"):
                    if self._is_assembly_file(yaml_file):
                        self._assemblies.append(yaml_file)

        # Also check parent directories for noodlings
        parent_noodlings = self.project_path.parent / "noodlings"
        if parent_noodlings.exists():
            for yaml_file in parent_noodlings.rglob("assembly.yaml"):
                if self._is_assembly_file(yaml_file):
                    self._assemblies.append(yaml_file)

        # Build tree
        self._build_tree()

    def _is_assembly_file(self, path: Path) -> bool:
        """Check if a YAML file is an assembly definition."""
        try:
            with open(path, 'r') as f:
                content = yaml.safe_load(f)
            if isinstance(content, dict):
                # Check for assembly markers
                return (
                    'facets' in content or
                    'name' in content and 'inputs' in content or
                    'name' in content and 'outputs' in content
                )
        except Exception:
            pass
        return False

    def _build_tree(self):
        """Build the tree widget from found assemblies."""
        self._tree.clear()

        # Group by directory
        dirs = {}
        for path in sorted(self._assemblies):
            rel_path = self._get_relative_path(path)
            parent_dir = str(rel_path.parent) if rel_path.parent != Path('.') else "Project Root"

            if parent_dir not in dirs:
                dirs[parent_dir] = []
            dirs[parent_dir].append((rel_path, path))

        # Create tree items
        for dir_name in sorted(dirs.keys()):
            if len(dirs) > 1:
                # Create folder item
                folder_item = QTreeWidgetItem(self._tree)
                folder_item.setText(0, dir_name)
                folder_item.setExpanded(True)
                parent_item = folder_item
            else:
                parent_item = self._tree

            for rel_path, abs_path in dirs[dir_name]:
                item = QTreeWidgetItem(parent_item if isinstance(parent_item, QTreeWidgetItem) else None)
                item.setText(0, rel_path.name)
                item.setData(0, Qt.ItemDataRole.UserRole, str(rel_path))
                item.setData(0, Qt.ItemDataRole.UserRole + 1, str(abs_path))

                if isinstance(parent_item, QTreeWidget):
                    self._tree.addTopLevelItem(item)

    def _get_relative_path(self, path: Path) -> Path:
        """Get path relative to project."""
        try:
            return path.relative_to(self.project_path)
        except ValueError:
            # Path is outside project - make it relative to parent
            try:
                return Path("..") / path.relative_to(self.project_path.parent)
            except ValueError:
                return path

    def _filter_tree(self, text: str):
        """Filter tree items by text."""
        text = text.lower()

        def set_visible(item: QTreeWidgetItem, visible: bool):
            item.setHidden(not visible)

        def filter_item(item: QTreeWidgetItem) -> bool:
            """Returns True if item or any child matches."""
            # Check children first
            child_match = False
            for i in range(item.childCount()):
                if filter_item(item.child(i)):
                    child_match = True

            # Check this item
            item_match = text in item.text(0).lower()

            visible = item_match or child_match
            set_visible(item, visible)
            return visible

        for i in range(self._tree.topLevelItemCount()):
            filter_item(self._tree.topLevelItem(i))

    def _on_selection_changed(self):
        """Handle tree selection change."""
        items = self._tree.selectedItems()
        if not items:
            self._select_btn.setEnabled(False)
            self._path_display.setText("")
            self._preview_name.setText("")
            self._preview_text.setText("")
            return

        item = items[0]
        rel_path = item.data(0, Qt.ItemDataRole.UserRole)
        abs_path = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if rel_path:
            self.selected_assembly = rel_path
            self._path_display.setText(rel_path)
            self._select_btn.setEnabled(True)
            self._show_preview(Path(abs_path))
        else:
            # Folder selected
            self._select_btn.setEnabled(False)
            self._path_display.setText("")
            self._preview_name.setText("")
            self._preview_text.setText("")

    def _show_preview(self, path: Path):
        """Show preview of selected assembly."""
        try:
            with open(path, 'r') as f:
                content = yaml.safe_load(f)

            name = content.get('name', path.stem)
            self._preview_name.setText(name)

            # Format preview
            preview_lines = []
            if 'description' in content:
                preview_lines.append(f"Description: {content['description']}")
                preview_lines.append("")

            if 'inputs' in content:
                preview_lines.append("Inputs:")
                for inp in content['inputs']:
                    inp_name = inp.get('name', inp) if isinstance(inp, dict) else inp
                    preview_lines.append(f"  - {inp_name}")
                preview_lines.append("")

            if 'outputs' in content:
                preview_lines.append("Outputs:")
                for out in content['outputs']:
                    out_name = out.get('name', out) if isinstance(out, dict) else out
                    preview_lines.append(f"  - {out_name}")
                preview_lines.append("")

            if 'facets' in content:
                preview_lines.append(f"Facets: {len(content['facets'])}")

            self._preview_text.setText("\n".join(preview_lines) if preview_lines else "(No metadata)")

        except Exception as e:
            self._preview_name.setText("Error")
            self._preview_text.setText(f"Failed to load: {e}")

    def _select_path(self, path: str):
        """Select an item by path."""
        def find_item(parent, target: str):
            for i in range(parent.childCount() if hasattr(parent, 'childCount') else parent.topLevelItemCount()):
                item = parent.child(i) if hasattr(parent, 'child') else parent.topLevelItem(i)
                if item.data(0, Qt.ItemDataRole.UserRole) == target:
                    return item
                found = find_item(item, target)
                if found:
                    return found
            return None

        item = find_item(self._tree, path)
        if item:
            self._tree.setCurrentItem(item)

    def _on_double_click(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on item."""
        if item.data(0, Qt.ItemDataRole.UserRole):
            self._on_select()

    def _on_select(self):
        """Handle select button click."""
        if self.selected_assembly:
            self.accept()

    def get_selected_assembly(self) -> str:
        """Get the selected assembly path."""
        return self.selected_assembly


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
