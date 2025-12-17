"""
Neural Canvas Panel - Main panel for visual neural network editing.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QToolBar, QSplitter, QSpinBox,
    QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

from ...core.neural_canvas.neural_graph import NeuralGraph, ValidationResult
from ...core.neural_canvas.mlx_codegen import generate_mlx_code
from ...core.neural_canvas.test_executor import CanvasTestExecutor, text_to_affect
from .neural_canvas_view import NeuralCanvasView
from .node_palette_panel import NodePalettePanel
from ...dialogs.neural_export_dialog import NeuralExportDialog


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

        # Test executor
        self.test_executor: CanvasTestExecutor = None
        self._test_initialized = False

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

        # Export button
        btn_export = QPushButton("EXPORT")
        btn_export.clicked.connect(self._on_export)
        toolbar.addWidget(btn_export)

        toolbar.addSeparator()

        # Layout button
        btn_auto_arrange = QPushButton("Auto-Arrange")
        btn_auto_arrange.clicked.connect(self._on_auto_arrange)
        toolbar.addWidget(btn_auto_arrange)

        toolbar.addSeparator()

        # Grid snap toggle button (load settings)
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'NeuralCanvas')
        grid_enabled = settings.value('grid/snap_enabled', False, type=bool)
        grid_size = settings.value('grid/size', 20, type=int)

        self.grid_button = QPushButton("⊞")  # Grid icon
        self.grid_button.setFixedWidth(40)
        self.grid_button.setCheckable(True)
        self.grid_button.setChecked(grid_enabled)  # Load from settings
        self.grid_button.setToolTip("Toggle grid snapping")
        self.grid_button.clicked.connect(self._on_toggle_grid)
        toolbar.addWidget(self.grid_button)

        # Grid size input
        self.grid_size_input = QSpinBox()
        self.grid_size_input.setRange(5, 100)
        self.grid_size_input.setValue(grid_size)  # Load from settings
        self.grid_size_input.setSuffix("px")
        self.grid_size_input.setFixedWidth(70)
        self.grid_size_input.setToolTip("Grid size in pixels")
        self.grid_size_input.valueChanged.connect(self._on_grid_size_changed)
        toolbar.addWidget(self.grid_size_input)

        # ===== TEST MODE SECTION =====
        toolbar.addSeparator()

        # Test input field
        test_label = QLabel("Test:")
        test_label.setStyleSheet("color: #aaa; padding: 0 4px;")
        toolbar.addWidget(test_label)

        self.test_input = QLineEdit()
        self.test_input.setPlaceholderText("Enter text or affect values...")
        self.test_input.setFixedWidth(200)
        self.test_input.setStyleSheet("""
            QLineEdit {
                background: #333;
                color: #ddd;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QLineEdit:focus {
                border: 1px solid #666;
            }
        """)
        self.test_input.returnPressed.connect(self._on_test_run)
        toolbar.addWidget(self.test_input)

        # Test Run button
        self.test_button = QPushButton("Run")
        self.test_button.setStyleSheet("""
            QPushButton {
                background: #3a5a3a;
                color: #cfc;
                border: 1px solid #4a6a4a;
                padding: 6px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a6a4a;
            }
            QPushButton:pressed {
                background: #2a4a2a;
            }
        """)
        self.test_button.clicked.connect(self._on_test_run)
        self.test_button.setToolTip("Run test inference (Enter)")
        toolbar.addWidget(self.test_button)

        # Reset states button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background: #4a3a3a;
                color: #fcc;
                border: 1px solid #5a4a4a;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #5a4a4a;
            }
        """)
        self.reset_button.clicked.connect(self._on_reset_states)
        self.reset_button.setToolTip("Reset hidden states to zero")
        toolbar.addWidget(self.reset_button)

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

    def _on_export(self):
        """Export neural network to various formats."""
        # Validate first
        result = self.graph.validate()
        if not result.valid:
            QMessageBox.warning(
                self, "Cannot Export",
                "Graph has validation errors. Fix them before exporting."
            )
            return

        # Show export format dialog
        dialog = NeuralExportDialog(self)
        if dialog.exec() != NeuralExportDialog.DialogCode.Accepted:
            return

        fmt = dialog.get_selected_format()
        if not fmt:
            return

        # Dispatch to appropriate export handler
        handlers = {
            'nncanvas': self._export_nncanvas,
            'mlx': self._export_mlx,
            'onnx': self._export_onnx,
            'pytorch': self._export_pytorch,
            'coreml': self._export_coreml
        }

        handler = handlers.get(fmt['id'])
        if handler:
            handler(fmt)
        else:
            QMessageBox.critical(
                self, "Export Error",
                f"No handler found for format: {fmt['name']}"
            )

    def _export_nncanvas(self, fmt):
        """Export to .nncanvas format."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Neural Canvas",
            "",
            fmt['filter']
        )

        if not filepath:
            return

        if not filepath.endswith(fmt['ext']):
            filepath += fmt['ext']

        try:
            # Save graph as JSON
            self.graph.save(filepath)

            QMessageBox.information(
                self, "Export Successful",
                f"Neural Canvas saved to:\n{filepath}\n\n"
                f"Nodes: {len(self.graph.nodes)}\n"
                f"Connections: {len(self.graph.connections)}\n"
                f"Parameters: {self.graph.compute_total_parameters():,}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _export_mlx(self, fmt):
        """Export to MLX Python code."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export MLX Code",
            "",
            fmt['filter']
        )

        if not filepath:
            return

        if not filepath.endswith(fmt['ext']):
            filepath += fmt['ext']

        try:
            # Generate code
            code = generate_mlx_code(self.graph)

            # Save to file
            with open(filepath, 'w') as f:
                f.write(code)

            QMessageBox.information(
                self, "Export Successful",
                f"MLX code exported to:\n{filepath}\n\n"
                f"Parameters: {self.graph.compute_total_parameters():,}\n\n"
                f"Next steps:\n"
                f"1. Install MLX: pip install mlx\n"
                f"2. Import generated code in your training script\n"
                f"3. Instantiate model and train!"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _export_onnx(self, fmt):
        """Export to ONNX format (planned)."""
        QMessageBox.information(
            self, "Coming Soon",
            "ONNX export is planned for a future update.\n\n"
            "ONNX (Open Neural Network Exchange) is the universal ML interchange format.\n\n"
            "For now, you can:\n"
            "1. Export to MLX Python\n"
            "2. Train the model\n"
            "3. Use mlx-to-onnx converter (community tools)"
        )

    def _export_pytorch(self, fmt):
        """Export to PyTorch format (planned)."""
        QMessageBox.information(
            self, "Coming Soon",
            "PyTorch export is planned for a future update.\n\n"
            "Will generate torch.nn.Module definition compatible with PyTorch training pipelines.\n\n"
            "For now, you can:\n"
            "1. Export to MLX Python\n"
            "2. Manually port to PyTorch (similar API)\n"
            "3. Or export to ONNX → convert to PyTorch"
        )

    def _export_coreml(self, fmt):
        """Export to CoreML format (planned)."""
        QMessageBox.information(
            self, "Coming Soon",
            "CoreML export is planned for a future update.\n\n"
            "CoreML is Apple's native format for iOS/macOS ML deployment.\n\n"
            "For now, you can:\n"
            "1. Export to MLX Python\n"
            "2. Train the model in MLX\n"
            "3. Use coremltools to convert MLX → CoreML"
        )


    def _on_node_selected(self, node_id: str):
        """Handle node selected in canvas."""
        self.node_selected.emit(node_id)

    def _on_graph_modified(self):
        """Handle graph modification - auto-save if we have a file path."""
        self._update_status_bar()
        self.graph_modified.emit()

        # Invalidate test executor (needs re-init on next test)
        self._test_initialized = False

        # Auto-save to current file (like Facets Editor)
        if self.current_filepath:
            try:
                self.graph.to_json(self.current_filepath)
                print(f"[Neural Canvas] Auto-saved to: {os.path.basename(self.current_filepath)}")
            except Exception as e:
                print(f"[Neural Canvas] Auto-save failed: {e}")

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

    def _on_toggle_grid(self):
        """Toggle grid snapping from toolbar button."""
        enabled = self.grid_button.isChecked()
        self.canvas_view.toggle_grid_snap(enabled)
        print(f"[Neural Canvas] Grid snapping: {'ON' if enabled else 'OFF'}")

    def _on_grid_size_changed(self, value: int):
        """Handle grid size spinbox change."""
        self.canvas_view.set_grid_size(value)
        print(f"[Neural Canvas] Grid size: {value}px")

    # ========== TEST MODE ==========

    def _init_test_executor(self):
        """Initialize or re-initialize the test executor."""
        self.test_executor = CanvasTestExecutor(self.graph)
        success, error = self.test_executor.initialize()
        if not success:
            self._test_initialized = False
            return False, error
        self._test_initialized = True
        return True, ""

    def _on_test_run(self):
        """Run test inference with current input."""
        # Initialize executor if needed
        if not self._test_initialized or self.test_executor is None:
            success, error = self._init_test_executor()
            if not success:
                QMessageBox.warning(
                    self, "Test Failed",
                    f"Failed to initialize test executor:\n{error}"
                )
                return

        # Parse input
        input_text = self.test_input.text().strip()

        # Check if input is numeric (direct affect values)
        affect_values = None
        if input_text:
            try:
                # Try parsing as comma-separated numbers
                parts = [p.strip() for p in input_text.replace(' ', ',').split(',') if p.strip()]
                if all(self._is_number(p) for p in parts):
                    affect_values = [float(p) for p in parts[:5]]
            except:
                pass

        # If not numeric, convert text to affect
        if affect_values is None:
            if input_text:
                affect_values = text_to_affect(input_text)
            else:
                # Default neutral
                affect_values = [0.0, 0.5, 0.5, 0.0, 0.0]

        # Run inference
        result = self.test_executor.execute(affect_values)

        if not result.success:
            QMessageBox.warning(
                self, "Test Failed",
                f"Test execution failed:\n{result.error}"
            )
            return

        # Update visual feedback on canvas
        self._display_test_results(result)

        # Update status bar with timing
        self.status_validation_label.setText(
            f"Test: {result.execution_time_ms:.1f}ms"
        )
        self.status_validation_label.setStyleSheet("color: #4CAF50;")

    def _is_number(self, s: str) -> bool:
        """Check if string is a valid number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _on_reset_states(self):
        """Reset hidden states to zero."""
        if self.test_executor:
            self.test_executor.reset_states()
            # Clear visual feedback
            self.canvas_view.clear_test_values()
            self.status_validation_label.setText("States reset")
            self.status_validation_label.setStyleSheet("color: #aaa;")

    def _display_test_results(self, result):
        """Display test results on the canvas nodes."""
        # Pass results to canvas view for visualization
        self.canvas_view.display_test_values(result.node_outputs)

        # Also print summary to console for debugging
        if result.outputs:
            summary = []
            for key, value in result.outputs.items():
                if isinstance(value, list):
                    if len(value) <= 5:
                        formatted = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in value]
                        summary.append(f"{key}: [{', '.join(formatted)}]")
                    else:
                        summary.append(f"{key}: [{len(value)} values]")
                else:
                    summary.append(f"{key}: {value}")
            print(f"[Neural Canvas] Test output: {', '.join(summary)}")
