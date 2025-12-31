"""
Scene Hierarchy UI Setup Mixin - UI initialization

Contains:
- init_ui: Build the scene hierarchy panel UI

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QTreeWidget, QLabel, QComboBox
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

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)

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

        # Stage View: Accept drops from Assets panel (for rezzing) but no internal reparenting
        # Folder organization is done in Assets panel, not Stage View
        self.tree.setDragEnabled(False)  # Can't drag scene entities
        self.tree.setAcceptDrops(True)   # Can receive drops from Assets panel
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DropOnly)
        self.tree.setDefaultDropAction(Qt.DropAction.CopyAction)

        # Context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        layout.addWidget(self.tree)
