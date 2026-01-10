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
#   Node Palette Panel - Draggable node type picker.
#
#   Implements node palette panel functionality.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.node_palette_panel
# PURPOSE:  Node Palette Panel - Draggable node type picker.
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NodePalettePanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ...core.neural_canvas.neural_node import NodeType
from ...core.neural_canvas.node_definitions import get_node_icon, NODE_DEFINITIONS


class NodePalettePanel(QWidget):
    """
    Node palette - shows available node types for adding to canvas.

    Groups:
    - Special (INPUT, OUTPUT)
    - Recurrent (LSTM, GRU, RNN)
    - Feedforward (Linear, Conv1D)
    - Activation (Tanh, ReLU, etc.)
    - Utility (Concat, Split, etc.)
    - Quantum (Microtubule, IBM Quantum, etc.)
    """

    # Signal emitted when node type clicked
    node_type_selected = pyqtSignal(NodeType)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        self.setStyleSheet("""
            QWidget {
                background: #2a2a2a;
                color: #ddd;
            }
            QLabel {
                color: #aaa;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Header
        header = QLabel("Node Palette")
        header.setStyleSheet("font-weight: bold; font-size: 12pt; color: #ddd;")
        layout.addWidget(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #2a2a2a;
            }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        # Node groups
        self._add_group(scroll_layout, "Special", [NodeType.INPUT, NodeType.OUTPUT])
        self._add_group(scroll_layout, "Recurrent", [NodeType.LSTM, NodeType.GRU, NodeType.RNN])
        self._add_group(scroll_layout, "Feedforward", [NodeType.LINEAR, NodeType.CONV1D])
        self._add_group(scroll_layout, "Attention", [NodeType.ATTENTION, NodeType.MULTI_HEAD_ATTENTION])
        self._add_group(scroll_layout, "Activation", [
            NodeType.TANH, NodeType.RELU, NodeType.GELU,
            NodeType.SIGMOID, NodeType.SOFTMAX
        ])
        self._add_group(scroll_layout, "Normalization", [NodeType.LAYER_NORM, NodeType.BATCH_NORM])
        self._add_group(scroll_layout, "Regularization", [NodeType.DROPOUT])
        self._add_group(scroll_layout, "Utility", [
            NodeType.STATE_CONCAT, NodeType.STATE_SPLIT, NodeType.AFFECT_HEAD
        ])
        self._add_group(scroll_layout, "Quantum", [
            NodeType.QUANTUM_MICROTUBULE,
            NodeType.IBM_QUANTUM,
            NodeType.ENTROPY_INJECTION
        ])
        self._add_group(scroll_layout, "Assets", [NodeType.CHECKPOINT])

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def _add_group(self, layout: QVBoxLayout, group_name: str, node_types: list):
        """Add a group of node types."""
        # Group label
        group_label = QLabel(group_name)
        group_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: bold;
            color: #888;
            padding: 8px 4px 4px 4px;
        """)
        layout.addWidget(group_label)

        # Node buttons
        for node_type in node_types:
            btn = self._create_node_button(node_type)
            layout.addWidget(btn)

    def _create_node_button(self, node_type: NodeType) -> QPushButton:
        """Create button for a node type."""
        definition = NODE_DEFINITIONS.get(node_type, {})
        icon = get_node_icon(node_type)
        name = definition.get('name', node_type.value)

        btn = QPushButton(f"{icon} {name}")
        btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px;
                text-align: left;
                font-size: 10pt;
            }
            QPushButton:hover {
                background: #4a4a4a;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background: #2a2a2a;
            }
        """)

        btn.clicked.connect(lambda: self._on_node_button_clicked(node_type))

        return btn

    def _on_node_button_clicked(self, node_type: NodeType):
        """Handle node button clicked."""
        self.node_type_selected.emit(node_type)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
