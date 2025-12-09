"""
Neural Canvas Panel - Main panel for visual neural network editing.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QToolBar, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

from ...core.neural_canvas.neural_graph import NeuralGraph, ValidationResult
from ...core.neural_canvas.mlx_codegen import generate_mlx_code
from .neural_canvas_view import NeuralCanvasView
from .node_palette_panel import NodePalettePanel


class NeuralCanvasPanel(QWidget):
    """
    Main panel for Neural Canvas - visual neural network editor.

    Layout:
    - Toolbar (File, Edit, View, Validate, Export)
    - Horizontal splitter:
      - Node Palette (left)
      - Canvas View (center)
      - Inspector (right, TODO)
    - Status bar (parameters, layers, validation)
    """

    # Signals
    node_selected = pyqtSignal(str)  # node_id
    graph_modified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self.graph = NeuralGraph()
        self.current_filepath: str = None

        # UI setup
        self._init_ui()

        # Setup keyboard shortcuts
        self._init_shortcuts()

        # Load default topology
        self._load_default_topology()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Canvas view (full width - no palette needed, context menu has everything)
        self.canvas_view = NeuralCanvasView(self.graph)
        self.canvas_view.node_selected.connect(self._on_node_selected)
        self.canvas_view.graph_modified.connect(self._on_graph_modified)
        layout.addWidget(self.canvas_view, 1)

        # Status bar
        status_bar = self._create_status_bar()
        layout.addWidget(status_bar)

    def _create_toolbar(self) -> QToolBar:
        """Create toolbar with actions."""
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                background: #2a2a2a;
                border-bottom: 1px solid #555;
                padding: 4px;
            }
            QPushButton {
                background: #3a3a3a;
                color: #ddd;
                border: 1px solid #555;
                padding: 6px 12px;
                margin: 2px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
            QPushButton:pressed {
                background: #2a2a2a;
            }
        """)

        # File menu buttons
        btn_new = QPushButton("New")
        btn_new.clicked.connect(self._on_new)
        toolbar.addWidget(btn_new)

        btn_open = QPushButton("Open...")
        btn_open.clicked.connect(self._on_open)
        toolbar.addWidget(btn_open)

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._on_save)
        toolbar.addWidget(btn_save)

        btn_save_as = QPushButton("Save As...")
        btn_save_as.clicked.connect(self._on_save_as)
        toolbar.addWidget(btn_save_as)

        toolbar.addSeparator()

        # Validate button
        btn_validate = QPushButton("✓ Validate")
        btn_validate.clicked.connect(self._on_validate)
        toolbar.addWidget(btn_validate)

        toolbar.addSeparator()

        # Export buttons
        btn_export_mlx = QPushButton("Export MLX...")
        btn_export_mlx.clicked.connect(self._on_export_mlx)
        toolbar.addWidget(btn_export_mlx)

        toolbar.addSeparator()

        # Layout button
        btn_auto_arrange = QPushButton("Auto-Arrange")
        btn_auto_arrange.clicked.connect(self._on_auto_arrange)
        toolbar.addWidget(btn_auto_arrange)

        return toolbar

    def _create_status_bar(self) -> QWidget:
        """Create status bar showing graph stats."""
        status_widget = QWidget()
        status_widget.setFixedHeight(30)
        status_widget.setStyleSheet("""
            QWidget {
                background: #2a2a2a;
                border-top: 1px solid #555;
                color: #aaa;
            }
            QLabel {
                padding: 4px 8px;
            }
        """)

        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(8, 2, 8, 2)

        self.status_params_label = QLabel("Parameters: 0")
        self.status_layers_label = QLabel("Layers: 0")
        self.status_validation_label = QLabel("● Validation: Unknown")

        layout.addWidget(self.status_params_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.status_layers_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.status_validation_label)
        layout.addStretch()

        return status_widget

    def _init_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Save (Cmd+S / Ctrl+S)
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self._on_save)

        # Delete (Del / Backspace)
        delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        delete_shortcut.activated.connect(self._on_delete_selected)

        # Validate (Cmd+T / Ctrl+T)
        validate_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        validate_shortcut.activated.connect(self._on_validate)

    def _on_delete_selected(self):
        """Delete selected nodes."""
        selected_items = self.canvas_view.scene.selectedItems()

        if not selected_items:
            return

        # Collect selected nodes
        nodes_to_delete = []
        for item in selected_items:
            if isinstance(item, self.canvas_view.NodeGraphicsItem):
                nodes_to_delete.append(item.node.id)

        if not nodes_to_delete:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Nodes",
            f"Delete {len(nodes_to_delete)} node(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for node_id in nodes_to_delete:
                self.graph.remove_node(node_id)

            # Re-render
            self.canvas_view._render_graph()
            self._update_status_bar()
            self.graph_modified.emit()

    def _update_status_bar(self):
        """Update status bar with current graph stats."""
        num_params = self.graph.compute_total_parameters()
        num_nodes = len(self.graph.nodes)

        self.status_params_label.setText(f"Parameters: {num_params:,}")
        self.status_layers_label.setText(f"Layers: {num_nodes}")

        # Validate
        result = self.graph.validate()
        if result.valid:
            self.status_validation_label.setText("✅ Valid")
            self.status_validation_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_validation_label.setText(f"❌ Invalid ({len(result.errors)} errors)")
            self.status_validation_label.setStyleSheet("color: #F44336;")

    def _load_default_topology(self):
        """Load default CharmNetwork topology."""
        # Path: noodlestudio/panels/neural_canvas/neural_canvas_panel.py
        # Need: ../../ (up to noodlestudio) -> ../../ (up to applications) -> ../../ (up to repo root) -> facet_assemblies
        default_path = os.path.join(
            os.path.dirname(__file__),
            '../../../../../facet_assemblies/charm_networks/default.nncanvas'
        )
        default_path = os.path.abspath(default_path)

        print(f"[Neural Canvas] Looking for default topology at: {default_path}")
        print(f"[Neural Canvas] File exists: {os.path.exists(default_path)}")

        if os.path.exists(default_path):
            self._load_from_file(default_path)
            print(f"[Neural Canvas] Loaded {len(self.graph.nodes)} nodes")
            # Frame all nodes so they're visible
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self.canvas_view.frame_all_nodes)
        else:
            print(f"[NeuralCanvas] Default topology not found: {default_path}")
            # Create empty graph
            self.graph = NeuralGraph()
            self.graph.name = "New Network"

        self._update_status_bar()

    def _load_from_file(self, filepath: str):
        """Load graph from .nncanvas file."""
        try:
            self.graph = NeuralGraph.from_json(filepath)
            self.current_filepath = filepath
            self.canvas_view.set_graph(self.graph)
            self._update_status_bar()
            print(f"[NeuralCanvas] Loaded: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")

    def _save_to_file(self, filepath: str):
        """Save graph to .nncanvas file."""
        try:
            self.graph.to_json(filepath)
            self.current_filepath = filepath
            print(f"[NeuralCanvas] Saved: {filepath}")
            QMessageBox.information(self, "Saved", f"Saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")

    # Slots
    def _on_new(self):
        """Create new empty network."""
        reply = QMessageBox.question(
            self, "New Network",
            "Create new network? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.graph = NeuralGraph()
            self.graph.name = "New Network"
            self.current_filepath = None
            self.canvas_view.set_graph(self.graph)
            self._update_status_bar()

    def _on_open(self):
        """Open .nncanvas file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Neural Canvas",
            "",
            "Neural Canvas Files (*.nncanvas);;All Files (*)"
        )
        if filepath:
            self._load_from_file(filepath)

    def _on_save(self):
        """Save to current file."""
        if self.current_filepath:
            self._save_to_file(self.current_filepath)
        else:
            self._on_save_as()

    def _on_save_as(self):
        """Save to new file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neural Canvas As",
            "",
            "Neural Canvas Files (*.nncanvas);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.nncanvas'):
                filepath += '.nncanvas'
            self._save_to_file(filepath)

    def _on_validate(self):
        """Validate graph and show results."""
        result = self.graph.validate()

        if result.valid:
            msg = "✅ Graph is valid!\n\n"
            if result.warnings:
                msg += "Warnings:\n"
                for warning in result.warnings:
                    msg += f"  ⚠️ {warning}\n"
            QMessageBox.information(self, "Validation", msg)
        else:
            msg = "❌ Graph is invalid!\n\nErrors:\n"
            for error in result.errors:
                msg += f"  • {error}\n"

            if result.warnings:
                msg += "\nWarnings:\n"
                for warning in result.warnings:
                    msg += f"  ⚠️ {warning}\n"

            QMessageBox.critical(self, "Validation Failed", msg)

        self._update_status_bar()

    def _on_export_mlx(self):
        """Export to MLX Python code."""
        # Validate first
        result = self.graph.validate()
        if not result.valid:
            QMessageBox.warning(
                self, "Cannot Export",
                "Graph has validation errors. Fix them before exporting."
            )
            return

        # Choose save location
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export MLX Code",
            "",
            "Python Files (*.py);;All Files (*)"
        )

        if not filepath:
            return

        if not filepath.endswith('.py'):
            filepath += '.py'

        try:
            # Generate code
            code = generate_mlx_code(self.graph)

            # Save to file
            with open(filepath, 'w') as f:
                f.write(code)

            QMessageBox.information(
                self, "Export Successful",
                f"MLX code exported to:\n{filepath}\n\n"
                f"Parameters: {self.graph.compute_total_parameters():,}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")


    def _on_node_selected(self, node_id: str):
        """Handle node selected in canvas."""
        self.node_selected.emit(node_id)

    def _on_graph_modified(self):
        """Handle graph modification."""
        self._update_status_bar()
        self.graph_modified.emit()

    def _on_auto_arrange(self):
        """Auto-arrange nodes using topological layering."""
        if not self.graph or len(self.graph.nodes) == 0:
            print("[Neural Canvas] No nodes to arrange")
            return

        print("[Neural Canvas] Starting auto-arrange...")

        try:
            # Get topological order
            node_order = self.graph.topological_sort()

            # Build layer assignment (same layer = same depth in DAG)
            layers = {}  # layer_index -> [node_ids]
            node_layer = {}  # node_id -> layer_index

            # Assign layers based on max dependency depth
            for node_id in node_order:
                # Find max layer of dependencies
                deps = self.graph.get_connections_to_node(node_id)
                if not deps:
                    # No dependencies - layer 0
                    layer = 0
                else:
                    # One layer below max dependency
                    max_dep_layer = max(
                        node_layer.get(conn.from_node, 0)
                        for conn in deps
                    )
                    layer = max_dep_layer + 1

                node_layer[node_id] = layer

                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node_id)

            # Layout parameters (match Facets Editor)
            layer_height = 200
            node_spacing = 280
            start_x = 100
            start_y = 100

            # Position nodes layer by layer
            for layer_idx in sorted(layers.keys()):
                layer_nodes = layers[layer_idx]
                y = start_y + (layer_idx * layer_height)

                for node_idx, node_id in enumerate(sorted(layer_nodes)):
                    x = start_x + (node_idx * node_spacing)

                    # Update node position
                    node = self.graph.nodes[node_id]
                    node.position = (x, y)

                    print(f"[Auto-Arrange] {node.name}: ({x}, {y}) [Layer {layer_idx}]")

            # Re-render graph
            print("[Neural Canvas] Re-rendering graph...")
            self.canvas_view._render_graph()

            print("[Neural Canvas] Updating status...")
            self._update_status_bar()
            self.graph_modified.emit()

            print(f"[Neural Canvas] Auto-arrange complete! {len(layers)} layers")

            # Frame all nodes to show result
            print("[Neural Canvas] Framing all nodes...")
            self.canvas_view.frame_all_nodes()

        except ValueError as e:
            print(f"[Neural Canvas] ValueError during auto-arrange: {e}")
            QMessageBox.warning(
                self, "Auto-Arrange Failed",
                f"Cannot arrange graph with cycles:\n{e}"
            )
        except Exception as e:
            print(f"[Neural Canvas] Unexpected error during auto-arrange: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Auto-Arrange Error",
                f"Unexpected error:\n{e}"
            )
