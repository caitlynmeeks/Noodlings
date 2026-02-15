# ------------------------------------------------------------------
#
#   Project Chooser Dialog - Logic Pro-style project selection
#
#   Shown on launch. The chooser handles choosing (template or
#   recent project). A native Save dialog handles naming/location.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.dialogs.project_chooser_dialog
# PURPOSE:  Project Chooser Dialog
# LAYER:    Studio / Dialogs
# ------------------------------------------------------------------
#
# KEY CLASSES:
#   ProjectChooserDialog
#
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import shutil
import yaml

from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QFileDialog, QWidget, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ========================================================================
# Template discovery
# ========================================================================

def _templates_dir() -> Path:
    """Return absolute path to the templates directory."""
    return Path(__file__).resolve().parent.parent.parent / 'library' / 'templates'


def _discover_templates() -> List[dict]:
    """
    Discover available project templates.

    Returns list of dicts with 'name', 'path', and 'description'.
    """
    templates = []
    tdir = _templates_dir()
    if not tdir.is_dir():
        return templates

    for entry in sorted(tdir.iterdir()):
        if not entry.is_dir():
            continue
        proj_file = entry / 'project.noodleproj'
        if not proj_file.exists():
            continue

        # Read description from project.noodleproj
        description = ""
        try:
            with open(proj_file) as f:
                data = yaml.safe_load(f) or {}
            description = data.get('description', '')
        except Exception:
            pass

        templates.append({
            'name': entry.name,
            'path': str(entry),
            'description': description,
        })

    return templates


# ========================================================================
# Project creation
# ========================================================================

def create_project_from_template(
    template_path: str, project_name: str, location: str
) -> Optional[str]:
    """
    Create a new project by deep-copying a template.

    Args:
        template_path: Absolute path to the template directory
        project_name: Name chosen by the user
        location: Parent directory where the project folder will be created

    Returns:
        Absolute path to the new project directory, or None on failure.
    """
    dest = os.path.join(location, project_name)
    if os.path.exists(dest):
        return None  # Already exists

    os.makedirs(location, exist_ok=True)
    shutil.copytree(template_path, dest)

    # Remove any hierarchy.yaml files — they contain absolute paths from
    # the template location and would cause duplicate items in the tree.
    # The hierarchy will be regenerated with correct paths on first open.
    for dirpath, _dirnames, filenames in os.walk(dest):
        for fn in filenames:
            if fn == 'hierarchy.yaml':
                os.remove(os.path.join(dirpath, fn))

    # Update project.noodleproj with the user's chosen name
    proj_file = os.path.join(dest, 'project.noodleproj')
    if os.path.exists(proj_file):
        try:
            with open(proj_file) as f:
                data = yaml.safe_load(f) or {}
            data['name'] = project_name
            with open(proj_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception:
            pass  # Non-fatal: project still works with old name

    return dest


# ========================================================================
# Dialog
# ========================================================================

_DEFAULT_PROJECTS_DIR = os.path.join(
    os.path.expanduser('~'), 'Documents', 'NoodleStudio Projects'
)


class ProjectChooserDialog(QDialog):
    """
    Logic Pro-style project chooser.

    The chooser handles choosing (template or recent project).
    A native macOS Save dialog handles naming and placing new projects.
    """

    projectSelected = pyqtSignal(str)  # Emits absolute path to project

    def __init__(self, recent_projects: List[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose a Project")
        self.setMinimumSize(700, 420)
        self.resize(750, 460)
        self.setModal(True)

        self._recent_projects = recent_projects or []
        self._templates = _discover_templates()
        self._selected_template = None

        self._build_ui()

    # ====================================================================
    # UI Construction
    # ====================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main body: sidebar + content
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # -- Sidebar --
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(140)
        self._sidebar.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: none;
                border-right: 1px solid #444;
                color: #D2D2D2;
                font-size: 12px;
                padding-top: 8px;
            }
            QListWidget::item {
                padding: 8px 12px;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
                color: #FFF;
            }
        """)
        self._sidebar.addItem("Templates")
        self._sidebar.addItem("Recent")
        self._sidebar.currentRowChanged.connect(self._on_category_changed)
        body.addWidget(self._sidebar)

        # -- Content stack --
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background-color: #333;")
        self._stack.addWidget(self._build_templates_page())
        self._stack.addWidget(self._build_recent_page())
        body.addWidget(self._stack, 1)

        root.addLayout(body, 1)

        # -- Bottom bar: [Open existing...] <stretch> [Cancel] [Choose] --
        self._bottom = self._build_bottom_bar()
        root.addWidget(self._bottom)

        # Default to Templates page
        self._sidebar.setCurrentRow(0)

    # ----------------------------------------------------------------
    # Templates page
    # ----------------------------------------------------------------

    def _build_templates_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        label = QLabel("Select a template:")
        label.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(label)

        # Template cards in a grid
        grid = QGridLayout()
        grid.setSpacing(10)

        for i, tmpl in enumerate(self._templates):
            card = self._make_template_card(tmpl)
            grid.addWidget(card, i // 2, i % 2)

        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        layout.addWidget(grid_widget)
        layout.addStretch()

        return page

    def _make_template_card(self, tmpl: dict) -> QFrame:
        card = QFrame()
        card.setFixedSize(240, 80)
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border: none;
                border-radius: 4px;
            }
            QFrame:hover {
                background-color: #444;
            }
        """)
        card.setProperty('template_path', tmpl['path'])
        card.setProperty('template_name', tmpl['name'])

        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(12, 10, 12, 10)

        name_label = QLabel(tmpl['name'])
        name_label.setStyleSheet("color: #D2D2D2; font-size: 13px; font-weight: bold;")
        vbox.addWidget(name_label)

        if tmpl.get('description'):
            desc = QLabel(tmpl['description'])
            desc.setStyleSheet("color: #888; font-size: 10px;")
            desc.setWordWrap(True)
            vbox.addWidget(desc)

        vbox.addStretch()

        # Click handler via mouse press
        card.mousePressEvent = lambda e, t=tmpl: self._on_template_selected(t)

        return card

    def _on_template_selected(self, tmpl: dict):
        self._selected_template = tmpl
        self._action_btn.setText("Choose")
        self._action_btn.setEnabled(True)
        # Visual feedback: highlight selected card
        for card in self._stack.widget(0).findChildren(QFrame):
            if card.property('template_path') == tmpl['path']:
                card.setStyleSheet("""
                    QFrame {
                        background-color: #444;
                        border: none;
                        border-radius: 4px;
                    }
                """)
            elif card.property('template_path'):
                card.setStyleSheet("""
                    QFrame {
                        background-color: #3a3a3a;
                        border: none;
                        border-radius: 4px;
                    }
                    QFrame:hover {
                        background-color: #444;
                    }
                """)

    # ----------------------------------------------------------------
    # Recent page
    # ----------------------------------------------------------------

    def _build_recent_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        label = QLabel("Recent projects:")
        label.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(label)

        self._recent_list = QListWidget()
        self._recent_list.setStyleSheet("""
            QListWidget {
                background-color: #2e2e2e;
                border: 1px solid #444;
                color: #D2D2D2;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
            }
        """)

        for project_path in self._recent_projects:
            if not os.path.exists(project_path):
                continue
            name = os.path.basename(project_path)
            item = QListWidgetItem(f"{name}\n{project_path}")
            item.setData(Qt.ItemDataRole.UserRole, project_path)
            self._recent_list.addItem(item)

        if self._recent_list.count() == 0:
            empty_item = QListWidgetItem("(No recent projects)")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._recent_list.addItem(empty_item)

        self._recent_list.itemDoubleClicked.connect(self._on_recent_double_click)
        self._recent_list.currentItemChanged.connect(self._on_recent_selected)
        layout.addWidget(self._recent_list, 1)

        return page

    def _on_recent_selected(self, current, previous):
        if current and current.data(Qt.ItemDataRole.UserRole):
            self._action_btn.setText("Choose")
            self._action_btn.setEnabled(True)
        else:
            self._action_btn.setEnabled(False)

    def _on_recent_double_click(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            self.projectSelected.emit(path)
            self.accept()

    # ----------------------------------------------------------------
    # Bottom bar: [Open existing...] <stretch> [Cancel] [Choose]
    # ----------------------------------------------------------------

    def _build_bottom_bar(self) -> QWidget:
        bottom = QWidget()
        bottom.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #444;")
        layout = QHBoxLayout(bottom)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Left: Open existing project
        open_btn = QPushButton("Open an existing project...")
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999;
                border: none;
                padding: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #D2D2D2;
            }
        """)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(self._on_open_existing)
        layout.addWidget(open_btn)

        layout.addStretch()

        # Right: Cancel + Choose
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #D2D2D2;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #444; }
        """)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self._action_btn = QPushButton("Choose")
        self._action_btn.setFixedWidth(80)
        self._action_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #666;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { color: #666; }
        """)
        self._action_btn.setEnabled(False)
        self._action_btn.clicked.connect(self._on_action)
        layout.addWidget(self._action_btn)

        return bottom

    # ====================================================================
    # Actions
    # ====================================================================

    def _on_category_changed(self, row):
        self._stack.setCurrentIndex(row)
        # Reset action button based on category
        if row == 0:  # Templates
            if self._selected_template:
                self._action_btn.setText("Choose")
                self._action_btn.setEnabled(True)
            else:
                self._action_btn.setText("Choose")
                self._action_btn.setEnabled(False)
        elif row == 1:  # Recent
            current = self._recent_list.currentItem()
            if current and current.data(Qt.ItemDataRole.UserRole):
                self._action_btn.setText("Choose")
                self._action_btn.setEnabled(True)
            else:
                self._action_btn.setText("Choose")
                self._action_btn.setEnabled(False)

    def _on_open_existing(self):
        """Open an existing project via native directory picker."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Project", _DEFAULT_PROJECTS_DIR
        )
        if path:
            if os.path.exists(os.path.join(path, 'project.noodleproj')):
                self.projectSelected.emit(path)
                self.accept()
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Not a Project",
                    f"No project.noodleproj found in:\n{path}\n\n"
                    "Please select a NoodleStudio project directory."
                )

    def _on_action(self):
        """Handle the Choose button click."""
        category = self._sidebar.currentRow()

        if category == 0:  # Templates -- native Save dialog for name/location
            self._choose_template()
        elif category == 1:  # Recent -- open directly
            current = self._recent_list.currentItem()
            if current:
                path = current.data(Qt.ItemDataRole.UserRole)
                if path and os.path.exists(path):
                    self.projectSelected.emit(path)
                    self.accept()

    def _choose_template(self):
        """Show native Save dialog to name and place the new project."""
        if not self._selected_template:
            return

        # Native Save dialog — user names the project and picks location
        default_path = os.path.join(_DEFAULT_PROJECTS_DIR, "My Project")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save New Project", default_path, ""
        )
        if not path:
            return  # User cancelled — stay in chooser

        location = os.path.dirname(path)
        name = os.path.basename(path)

        dest = create_project_from_template(
            self._selected_template['path'], name, location
        )

        if dest is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Project Exists",
                f"A project named '{name}' already exists at:\n{location}"
            )
            return

        self.projectSelected.emit(dest)
        self.accept()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
