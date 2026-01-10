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
#   Scene Hierarchy UI Setup Mixin - UI initialization
#
#   Contains: - init_ui: Build the scene hierarchy panel UI
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.scene_hierarchy_ui_setup_mixin
# PURPOSE:  Scene Hierarchy UI Setup Mixin - UI initialization
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SceneHierarchyUISetupMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QTreeWidget, QLabel, QComboBox, QHeaderView
)
from PyQt6.QtCore import Qt


class SceneHierarchyUISetupMixin:
    """Mixin providing UI setup for SceneHierarchy."""

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Stage selector dropdown
        stage_layout = QHBoxLayout()
        stage_label = QLabel("Stage:")
        stage_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
        stage_layout.addWidget(stage_label)

        self.stage_selector = QComboBox()
        self.stage_selector.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 3px;
            }
            QComboBox:hover {
                border: 1px solid #888;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #D2D2D2;
                margin-right: 8px;
            }
        """)
        self.stage_selector.currentTextChanged.connect(self.on_stage_changed)
        stage_layout.addWidget(self.stage_selector, stretch=1)

        layout.addLayout(stage_layout)

        # Status label for messages (server offline, no project, etc.)
        # Shown ABOVE the tree, not as tree items
        self.status_label = QLabel()
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                padding: 8px;
                background-color: #252525;
                border: 1px solid #333;
                border-radius: 3px;
            }
        """)
        self.status_label.setWordWrap(True)
        self.status_label.hide()  # Hidden by default
        layout.addWidget(self.status_label)

        # Tree widget with 2 columns: Name (stretch) + Status button (fixed)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)

        # Style the tree widget and inline editor
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: none;
            }
            QTreeWidget::item {
                padding: 4px 2px;
                min-height: 22px;
            }
            QTreeWidget::item:selected {
                background-color: #2D5A88;
            }
            QTreeWidget::item:hover {
                background-color: #2A2A2A;
            }
            /* Inline editor styling */
            QTreeWidget QLineEdit {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #4A90D9;
                border-radius: 2px;
                padding: 2px 4px;
                min-height: 18px;
                selection-background-color: #4A90D9;
            }
        """)

        # Column 0 stretches for name, column 1 fixed width for status button
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 28)  # Fixed width for pause button
        # Right-align column 1 content
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)  # Default for col 0

        # Don't elide text - show full names (users can expand panel if needed)
        header.setStretchLastSection(False)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)

        # Enable horizontal scrollbar for long names
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Prevent auto-collapse: set animation to false
        self.tree.setAnimated(False)

        # Enable inline editing (Unity-style double-click to rename)
        self.tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)  # We control when editing starts
        self.tree.itemChanged.connect(self._on_item_renamed)

        # Enable multi-selection for batch derez
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)

        # Use itemSelectionChanged to avoid interfering with expand/collapse
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)

        # Single-click on text to toggle expansion (consistent with Inspector CollapsibleSection)
        self.tree.itemClicked.connect(self.on_item_clicked_for_expansion)

        # Double-click to unpack ensembles or inspect entities
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Stage View: Accept drops from Assets panel (for rezzing)
        # Also enable internal move for UI component reparenting
        self.tree.setDragEnabled(True)   # Enable drag for UI component reparenting
        self.tree.setAcceptDrops(True)   # Can receive drops from Assets panel
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)
        self.tree.setDefaultDropAction(Qt.DropAction.MoveAction)

        # Context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        layout.addWidget(self.tree)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
