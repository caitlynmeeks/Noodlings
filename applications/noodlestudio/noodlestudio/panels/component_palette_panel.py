"""
Component Palette Panel - Draggable UI components for the canvas

Left dock tab showing available UI components that can be dragged
onto the UI Canvas Editor.

Organized into groups:
- Standard: Panel, Label, Button, TextInput
- Noodle: ChatHistory, ChatInput, RadianceViewport

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Dict, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, QMimeData, QByteArray
from PyQt6.QtGui import QDrag, QFont


# Component definitions with metadata
COMPONENT_GROUPS = {
    "Standard": [
        ("Panel", "Container for other components"),
        ("Label", "Static text display"),
        ("Button", "Clickable button"),
        ("TextInput", "Single-line text input"),
    ],
    "Noodle": [
        ("ChatHistory", "Scrolling chat message list"),
        ("ChatInput", "Chat input with send button"),
        ("RadianceViewport", "3D Gaussian renderer"),
    ],
}


class DraggableListWidget(QListWidget):
    """
    QListWidget that supports dragging items as component types.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def startDrag(self, supportedActions):
        """Start drag with component type data."""
        item = self.currentItem()
        if not item:
            return

        component_type = item.data(Qt.ItemDataRole.UserRole)
        if not component_type:
            return

        # Create mime data
        mime_data = QMimeData()
        mime_data.setData(
            "application/x-noodlestudio-component",
            QByteArray(component_type.encode())
        )

        # Create drag
        drag = QDrag(self)
        drag.setMimeData(mime_data)

        # Execute drag
        drag.exec(Qt.DropAction.CopyAction)


class ComponentPalettePanel(QWidget):
    """
    Component Palette - shows available UI components for drag-drop.

    Displayed as a tab in the left dock alongside Stage and Assets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the palette UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QLabel("Drag components to canvas")
        header.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(header)

        # Component list
        self.list_widget = DraggableListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: #cccccc;
                padding: 8px 12px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #3d3d3d;
            }
            QListWidget::item:hover {
                background-color: #353535;
            }
        """)

        # Populate list
        self._populate_list()

        layout.addWidget(self.list_widget)

    def _populate_list(self):
        """Populate the list with component groups."""
        for group_name, components in COMPONENT_GROUPS.items():
            # Group header
            header_item = QListWidgetItem(f"-- {group_name} --")
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Not selectable
            header_font = QFont()
            header_font.setBold(True)
            header_item.setFont(header_font)
            header_item.setForeground(Qt.GlobalColor.gray)
            self.list_widget.addItem(header_item)

            # Components
            for component_type, description in components:
                item = QListWidgetItem(f"  {component_type}")
                item.setData(Qt.ItemDataRole.UserRole, component_type)
                item.setToolTip(description)
                self.list_widget.addItem(item)
