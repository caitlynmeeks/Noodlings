"""Assembly Editor view for the unified editor panel.

A QGraphicsView subclass that renders facet assemblies using the shared
editor mixins from C.1. Implements DepthViewProtocol for the depth stack
and FacetEditorProtocol for undo command integration.

Execution visualization (C.4): node pulsing, wire data packets, sound
effects, ensemble noodling selector, cognition pause/resume.

This view is the root (level 0) view of UnifiedEditorPanel, handling
facet assembly rendering and interaction.
"""

import os
from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QMenu, QHBoxLayout, QWidget,
    QPushButton, QVBoxLayout, QSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QCursor

from ...core.facet_system import (
    Facet, FacetAssembly, FacetConnection, FacetPad, PadType
)
from .assembly_graphics_items import (
    FacetNodeItem, FacetPortItem, FacetConnectionItem, get_facet_header_color
)
from .shared_input_mixin import SharedInputMixin
from .shared_grid_mixin import SharedGridMixin
from .shared_layout_mixin import SharedLayoutMixin
from .shared_view_ops_mixin import SharedViewOpsMixin
from .shared_wire_mixin import SharedWireMixin
from .shared_clipboard_mixin import SharedClipboardMixin
from .assembly_execution_mixin import AssemblyExecutionMixin
from .assembly_ensemble_mixin import AssemblyEnsembleMixin


class AssemblyEditorView(
    AssemblyExecutionMixin,
    AssemblyEnsembleMixin,
    SharedInputMixin,
    SharedGridMixin,
    SharedLayoutMixin,
    SharedViewOpsMixin,
    SharedWireMixin,
    SharedClipboardMixin,
    QGraphicsView,
):
    """Visual editor for facet assemblies.

    Signals:
        facetSelected(object): Emitted when a facet is selected (Facet or None).
        assemblyModified(): Emitted after any data mutation.
        containerDoubleClicked(str, str, str): (facet_type, data_path, label)
            for depth navigation into container facets.
    """

    facetSelected = pyqtSignal(object)
    assemblyModified = pyqtSignal()
    containerDoubleClicked = pyqtSignal(str, str, str)

    # SharedGridMixin settings
    GRID_SETTINGS_APP = "AssemblyEditor"

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize all mixin states
        self._init_input_state()
        self._init_grid_state()
        self._init_view_ops_state()
        self._init_wire_state()
        self._init_clipboard_state()
        self._init_execution_state()
        self._init_ensemble_state()

        # Scene
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(-2000, -2000, 6000, 6000)
        self.setScene(self._scene)

        # View settings
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(self.renderHints())
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#2A2A2A"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Assembly state
        self._assembly: Optional[FacetAssembly] = None
        self._assembly_path: Optional[str] = None
        self._assembly_name: Optional[str] = None
        self._dirty = False

        # Graphics tracking
        self._node_items: Dict[str, FacetNodeItem] = {}
        self._wire_items: List[FacetConnectionItem] = []

        # Drag undo tracking
        self._drag_start_positions: Dict[str, Tuple[float, float]] = {}

        # Selection signal
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._selection_connected = True

        # Toolbar (sound toggle, pause, ensemble selector)
        self._toolbar_widget = self._create_toolbar()

    def _create_toolbar(self) -> QWidget:
        """Create toolbar widget with sound, pause, and ensemble controls.

        Returns a QWidget so UnifiedEditorPanel can embed/swap it easily.
        """
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet(
            "background-color: #333333; border-bottom: 1px solid #444444;"
        )
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(6)

        # Sound toggle
        self._sound_button = QPushButton("Sound On")
        self._sound_button.setCheckable(True)
        self._sound_button.setChecked(True)
        self._sound_button.setFixedWidth(80)
        self._sound_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A; color: #CCCCCC;
                border: 1px solid #555; border-radius: 3px;
                padding: 2px 6px; font-size: 11px;
            }
            QPushButton:checked { background-color: #4A4A4A; }
        """)
        self._sound_button.toggled.connect(self.toggle_sound)
        toolbar.addWidget(self._sound_button)

        # Pause/resume
        self._pause_button = QPushButton("Pause")
        self._pause_button.setCheckable(True)
        self._pause_button.setFixedWidth(80)
        self._pause_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A; color: #CCCCCC;
                border: 1px solid #555; border-radius: 3px;
                padding: 2px 6px; font-size: 11px;
            }
            QPushButton:checked { background-color: #5A3A3A; }
        """)
        self._pause_button.toggled.connect(self.toggle_pause_cognition)
        toolbar.addWidget(self._pause_button)

        # Ensemble noodling selector (created by ensemble mixin, hidden by default)
        toolbar.addWidget(self._noodling_selector)

        # Grid snap toggle
        self._grid_button = QPushButton("Grid")
        self._grid_button.setFixedWidth(50)
        self._grid_button.setCheckable(True)
        self._grid_button.setChecked(self._snap_to_grid)
        self._grid_button.setToolTip("Toggle grid snapping")
        self._grid_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A; color: #CCCCCC;
                border: 1px solid #555; border-radius: 3px;
                padding: 2px 6px; font-size: 11px;
            }
            QPushButton:checked { background-color: #4A4A4A; border: 1px solid #888; }
        """)
        self._grid_button.toggled.connect(self.toggle_grid)
        toolbar.addWidget(self._grid_button)

        # Grid size spinbox
        self._grid_size_input = QSpinBox()
        self._grid_size_input.setRange(5, 100)
        self._grid_size_input.setValue(self._grid_size)
        self._grid_size_input.setSuffix("px")
        self._grid_size_input.setFixedWidth(70)
        self._grid_size_input.setToolTip("Grid size in pixels")
        self._grid_size_input.setStyleSheet("""
            QSpinBox {
                background-color: #3A3A3A; color: #CCCCCC;
                border: 1px solid #555; border-radius: 3px; padding: 2px;
            }
            QSpinBox:hover { background-color: #4A4A4A; border: 1px solid #777; }
        """)
        self._grid_size_input.valueChanged.connect(self.set_grid_size)
        toolbar.addWidget(self._grid_size_input)

        toolbar.addStretch()

        return toolbar_widget

    def get_toolbar_widget(self) -> Optional[QWidget]:
        """Return the toolbar widget for embedding in a parent container."""
        return self._toolbar_widget

    # ================================================================
    # DepthViewProtocol
    # ================================================================

    def load_data(self, data_path: str, context: dict) -> None:
        """Load a FacetAssembly from YAML and render it."""
        if not os.path.exists(data_path):
            return

        assembly = FacetAssembly.load_yaml(data_path)
        if assembly is None:
            return

        self._assembly = assembly
        self._assembly_path = data_path
        self._assembly_name = assembly.name
        self._dirty = False

        self._render_assembly()

    def save_data(self) -> None:
        """Persist positions and assembly to disk."""
        self._sync_positions_to_data()
        self._save_assembly_to_disk()
        self._dirty = False

    def get_breadcrumb_label(self) -> str:
        return self._assembly_name or "assembly"

    def has_unsaved_changes(self) -> bool:
        return self._dirty

    # ================================================================
    # Assembly loading / rendering
    # ================================================================

    def load_assembly_from_data(self, assembly: FacetAssembly,
                                source_path: Optional[str] = None,
                                force_reload: bool = False):
        """Load an already-parsed assembly object (used by tests and signals).

        Args:
            assembly: Parsed FacetAssembly object.
            source_path: Path to the assembly YAML on disk.
            force_reload: If True, reload even if same assembly is loaded
                          (used by ensemble noodling switching).
        """
        if not force_reload and self._assembly is assembly:
            return
        self._assembly = assembly
        self._assembly_path = source_path
        self._assembly_name = assembly.name
        self._dirty = False
        self._render_assembly()

    def _render_assembly(self):
        """Clear the scene and rebuild all nodes and wires."""
        self.scene_transition_lock = True

        # Stop any existing animation timers on nodes and wires
        for node in self._node_items.values():
            node.stop_animation()
        for wire in self._wire_items:
            wire.stop_animation()

        self._scene.clear()
        self._node_items.clear()
        self._wire_items.clear()

        if self._assembly is None:
            self.scene_transition_lock = False
            return

        # Create node items
        for facet in self._assembly.facets:
            node = FacetNodeItem(facet, editor_view=self)
            self._scene.addItem(node)
            self._node_items[facet.id] = node

        # Create wire items
        for conn in self._assembly.connections:
            from_node = self._node_items.get(conn.from_facet)
            to_node = self._node_items.get(conn.to_facet)
            if from_node is None or to_node is None:
                continue

            from_port = from_node.output_pads.get(conn.from_pad)
            to_port = to_node.input_pads.get(conn.to_pad)
            if from_port is None or to_port is None:
                continue

            wire = FacetConnectionItem(from_port, to_port)
            self._scene.addItem(wire)
            self._wire_items.append(wire)

        # Restore grid if visible
        if self._grid_visible:
            self._draw_grid()

        # Center view
        self.centerOn(QPointF(500, 350))

        self.scene_transition_lock = False

    def _sync_positions_to_data(self):
        """Write current graphics positions back into the data model."""
        for facet_id, node in self._node_items.items():
            facet = node.facet
            facet.position = {'x': node.pos().x(), 'y': node.pos().y()}

    # ================================================================
    # FacetEditorProtocol -- internal mutation methods for undo commands
    # ================================================================

    def _save_assembly_to_disk(self) -> None:
        if self._assembly and self._assembly_path:
            if os.path.exists(os.path.dirname(self._assembly_path) or '.'):
                try:
                    self._sync_positions_to_data()
                    self._assembly.save_yaml(self._assembly_path)
                except Exception as e:
                    print(f"[AssemblyEditorView] Save failed: {e}")

    def _set_facet_position_internal(
        self, facet_id: str, position: Tuple[float, float]
    ) -> None:
        facet = self._find_facet(facet_id)
        if facet is None:
            return

        facet.position = {'x': position[0], 'y': position[1]}

        node = self._node_items.get(facet_id)
        if node:
            # Disable geometry change signals to avoid recursion
            node.setFlag(
                FacetNodeItem.GraphicsItemFlag.ItemSendsGeometryChanges, False
            )
            node.setPos(position[0], position[1])
            node.setFlag(
                FacetNodeItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
            )
            # Update connected wires
            for port in list(node.input_pads.values()) + list(node.output_pads.values()):
                for wire in port.connections:
                    wire.update_path()

        self._save_assembly_to_disk()

    def _create_facet_internal(self, facet_data: Dict[str, Any]) -> None:
        if self._assembly is None:
            return

        facet = Facet.from_dict(facet_data)
        self._assembly.facets.append(facet)

        node = FacetNodeItem(facet, editor_view=self)
        self._scene.addItem(node)
        self._node_items[facet.id] = node

        self._save_assembly_to_disk()
        self._mark_dirty()

    def _delete_facet_internal(self, facet_id: str) -> None:
        if self._assembly is None:
            return

        # Remove from assembly data
        self._assembly.facets = [
            f for f in self._assembly.facets if f.id != facet_id
        ]

        # Remove connections involving this facet
        removed_conns = [
            c for c in self._assembly.connections
            if c.from_facet == facet_id or c.to_facet == facet_id
        ]
        self._assembly.connections = [
            c for c in self._assembly.connections
            if c.from_facet != facet_id and c.to_facet != facet_id
        ]

        # Remove wire graphics for removed connections
        for conn in removed_conns:
            self._remove_wire_for_connection(
                conn.from_facet, conn.from_pad, conn.to_facet, conn.to_pad
            )

        # Remove node graphics
        node = self._node_items.pop(facet_id, None)
        if node and node.scene():
            self._scene.removeItem(node)

        self._save_assembly_to_disk()
        self._mark_dirty()

    def _create_connection_internal(self, conn_data: Dict[str, Any]) -> None:
        if self._assembly is None:
            return

        from_parts = conn_data['from'].split('.')
        to_parts = conn_data['to'].split('.')
        from_facet_id = from_parts[0]
        from_pad_name = '.'.join(from_parts[1:])
        to_facet_id = to_parts[0]
        to_pad_name = '.'.join(to_parts[1:])

        # Create data connection
        conn = FacetConnection(
            from_facet=from_facet_id, from_pad=from_pad_name,
            to_facet=to_facet_id, to_pad=to_pad_name
        )
        self._assembly.connections.append(conn)

        # Create wire graphics
        from_node = self._node_items.get(from_facet_id)
        to_node = self._node_items.get(to_facet_id)
        if from_node and to_node:
            from_port = from_node.output_pads.get(from_pad_name)
            to_port = to_node.input_pads.get(to_pad_name)
            if from_port and to_port:
                wire = FacetConnectionItem(from_port, to_port)
                self._scene.addItem(wire)
                self._wire_items.append(wire)

        self._save_assembly_to_disk()
        self._mark_dirty()

    def _delete_connection_internal(
        self, from_facet: str, from_pad: str, to_facet: str, to_pad: str
    ) -> None:
        if self._assembly is None:
            return

        # Remove from data
        self._assembly.connections = [
            c for c in self._assembly.connections
            if not (c.from_facet == from_facet and c.from_pad == from_pad
                    and c.to_facet == to_facet and c.to_pad == to_pad)
        ]

        # Remove wire graphics
        self._remove_wire_for_connection(from_facet, from_pad, to_facet, to_pad)

        self._save_assembly_to_disk()
        self._mark_dirty()

    def _set_facet_property_internal(
        self, facet_id: str, property_name: str, value: Any
    ) -> None:
        facet = self._find_facet(facet_id)
        if facet is None:
            return

        setattr(facet, property_name, value)

        # Update lock icon visual if needed
        if property_name == 'locked':
            node = self._node_items.get(facet_id)
            if node:
                node.lock_label.setPlainText("[L]" if value else "")
                node.lock_label.setDefaultTextColor(
                    QColor("#CCAA00" if value else "#888888")
                )
                node.setFlag(
                    FacetNodeItem.GraphicsItemFlag.ItemIsMovable, not value
                )

        self._save_assembly_to_disk()
        self._mark_dirty()

    # ================================================================
    # SharedMixin abstract method implementations
    # ================================================================

    def get_node_items(self) -> dict:
        return self._node_items

    def get_graph_edges(self) -> list:
        if self._assembly is None:
            return []
        return [(c.from_facet, c.to_facet) for c in self._assembly.connections]

    def get_selected_node_ids(self) -> list:
        return [
            item.get_node_id()
            for item in self._scene.selectedItems()
            if isinstance(item, FacetNodeItem)
        ]

    def get_existing_connections(self) -> list:
        if self._assembly is None:
            return []
        return [
            (c.from_facet, c.from_pad, c.to_facet, c.to_pad)
            for c in self._assembly.connections
        ]

    def create_connection(self, from_port, to_port):
        """Create connection with undo support (called by SharedWireMixin)."""
        # Normalize direction: output -> input
        if not from_port.is_output:
            from_port, to_port = to_port, from_port

        from ...core.undo_manager import undo_manager
        from ...core.commands import CreateConnectionCommand

        cmd = CreateConnectionCommand(
            editor=self,
            from_facet=from_port.get_parent_node_id(),
            from_pad=from_port.get_port_name(),
            to_facet=to_port.get_parent_node_id(),
            to_pad=to_port.get_port_name(),
        )
        undo_manager.push(cmd)

    def on_layout_complete(self):
        """Called after auto-arrange or alignment changes."""
        for wire in self._wire_items:
            wire.update_path()
        self._sync_positions_to_data()
        self._save_assembly_to_disk()
        self._mark_dirty()

    def serialize_selection(self) -> list:
        """Serialize selected non-special nodes for clipboard."""
        result = []
        for item in self._scene.selectedItems():
            if isinstance(item, FacetNodeItem) and not item.is_special_node:
                result.append(item.facet.to_dict())
        return result

    def deserialize_items(self, items_data: list, connections_data: list):
        """Create pasted facets and connections."""
        from ...core.undo_manager import undo_manager
        from ...core.commands import CreateFacetCommand, CreateConnectionCommand

        if not items_data:
            return

        undo_manager.begin_group("Paste")

        for item_data in items_data:
            cmd = CreateFacetCommand(
                editor=self,
                facet_data=item_data,
                facet_name=item_data.get('name', 'Facet')
            )
            undo_manager.push(cmd)

        for from_id, from_pad, to_id, to_pad in connections_data:
            cmd = CreateConnectionCommand(
                editor=self,
                from_facet=from_id,
                from_pad=from_pad,
                to_facet=to_id,
                to_pad=to_pad,
            )
            undo_manager.push(cmd)

        undo_manager.end_group()

    # ================================================================
    # Mouse events
    # ================================================================

    def mousePressEvent(self, event):
        if self.handle_middle_press(event):
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.record_right_click_time()
            self._show_context_menu(event.pos())
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on a port (wire drawing handled by FacetPortItem)
            # Record drag start positions for selected nodes
            self._drag_start_positions.clear()
            for item in self._scene.selectedItems():
                if isinstance(item, FacetNodeItem):
                    self._drag_start_positions[item.get_node_id()] = (
                        item.pos().x(), item.pos().y()
                    )

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.handle_middle_move(event):
            return

        if self.is_drawing_wire:
            scene_pos = self.mapToScene(event.pos())
            self.update_wire_drawing(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.handle_middle_release(event):
            return

        if self.is_drawing_wire and event.button() == Qt.MouseButton.LeftButton:
            # Check if released on a port
            scene_pos = self.mapToScene(event.pos())
            item_at = self._scene.itemAt(scene_pos, self.transform())
            end_port = item_at if isinstance(item_at, FacetPortItem) else None
            self.finish_wire_drawing(end_port)
            event.accept()
            return

        super().mouseReleaseEvent(event)

        # Check for node drags that need undo commands
        if event.button() == Qt.MouseButton.LeftButton:
            self._push_move_commands_if_needed()

    def wheelEvent(self, event):
        self.wheel_zoom(event)

    # ================================================================
    # Keyboard shortcuts
    # ================================================================

    def keyPressEvent(self, event):
        if self.handle_key_press_input(event):
            return

        key = event.key()
        modifiers = event.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        shift = modifiers & Qt.KeyboardModifier.ShiftModifier

        if key == Qt.Key.Key_F:
            self.focus_selection()
        elif key == Qt.Key.Key_A and not ctrl:
            self.frame_all_nodes()
        elif key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            self._delete_selection()
        elif key == Qt.Key.Key_C and ctrl:
            self.copy_selection()
        elif key == Qt.Key.Key_V and ctrl:
            self.paste_selection()
        elif key == Qt.Key.Key_D and ctrl:
            self.duplicate_selection()
        elif key == Qt.Key.Key_Z and ctrl and shift:
            self._redo()
        elif key == Qt.Key.Key_Z and ctrl:
            self._undo()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self._zoom_view(1.2)
        elif key == Qt.Key.Key_Minus:
            self._zoom_view(1.0 / 1.2)
        elif key == Qt.Key.Key_Home:
            self._reset_view()
        elif key == Qt.Key.Key_E:
            self._toggle_floating_editor()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.handle_key_release_input(event):
            return
        super().keyReleaseEvent(event)

    # ================================================================
    # Selection
    # ================================================================

    def _on_selection_changed(self):
        """Emit facetSelected signal based on current selection."""
        selected = [
            item for item in self._scene.selectedItems()
            if isinstance(item, FacetNodeItem)
        ]
        if len(selected) == 1:
            self.facetSelected.emit(selected[0].facet)
        else:
            self.facetSelected.emit(None)

    # ================================================================
    # Context menu
    # ================================================================

    def _show_context_menu(self, position):
        """Show right-click context menu."""
        # Temporarily disconnect selection signal to prevent crashes
        if self._selection_connected:
            try:
                self._scene.selectionChanged.disconnect(self._on_selection_changed)
                self._selection_connected = False
            except Exception:
                pass

        try:
            menu = QMenu(self)

            # Check if right-click is on a wire
            scene_pos = self.mapToScene(position)
            clicked_item = self._scene.itemAt(scene_pos, self.transform())

            if isinstance(clicked_item, FacetConnectionItem):
                self._build_wire_context_menu(menu, clicked_item)
                menu.exec(self.mapToGlobal(position))
                return

            # Add facet submenu
            self._build_add_facet_menu(menu, position)

            # Layout submenu
            menu.addSeparator()
            layout_menu = menu.addMenu("Layout")
            auto_action = layout_menu.addAction("Auto-Arrange (Topological)")
            auto_action.triggered.connect(self.auto_arrange)

            layout_menu.addSeparator()
            selected = [
                item for item in self._scene.selectedItems()
                if isinstance(item, FacetNodeItem)
            ]

            align_h = layout_menu.addAction(
                f"Align Horizontally ({len(selected)} selected)"
            )
            align_h.setEnabled(len(selected) > 1)
            align_h.triggered.connect(self.align_selected_horizontally)

            align_v = layout_menu.addAction(
                f"Align Vertically ({len(selected)} selected)"
            )
            align_v.setEnabled(len(selected) > 1)
            align_v.triggered.connect(self.align_selected_vertically)

            # Delete
            if selected:
                menu.addSeparator()
                del_action = menu.addAction(f"Delete {len(selected)} facet(s)")
                del_action.triggered.connect(self._delete_selection)

            menu.exec(self.mapToGlobal(position))

        finally:
            if not self._selection_connected:
                try:
                    self._scene.selectionChanged.connect(
                        self._on_selection_changed
                    )
                    self._selection_connected = True
                except Exception:
                    pass

    def _build_wire_context_menu(self, menu: QMenu, wire: FacetConnectionItem):
        """Build context menu for clicking on a wire."""
        from_facet = wire.from_port.facet_node.facet
        to_facet = wire.to_port.facet_node.facet
        from_pad_name = wire.from_port.pad.name
        to_pad_name = wire.to_port.pad.name

        info = menu.addAction(
            f"Connection: {from_facet.name}.{from_pad_name} -> "
            f"{to_facet.name}.{to_pad_name}"
        )
        info.setEnabled(False)

        menu.addSeparator()
        del_action = menu.addAction("Delete Connection")
        del_action.triggered.connect(
            lambda: self._delete_connection_with_undo(wire)
        )

    def _build_add_facet_menu(self, menu: QMenu, position):
        """Build the Add Facet submenu."""
        add_menu = menu.addMenu("Add Facet")

        facet_types = [
            ("Intuition Facet", "IntuitionFacet"),
            ("Emotion Facet", "EmotionFacet"),
            ("Social Context Facet", "SocialFacet"),
            ("Memory Recall Facet", "MemoryFacet"),
            ("Response Planning Facet", "PlanningFacet"),
            ("Convergence Facet", "ConvergenceFacet"),
            ("Scripted Facet (JavaScript)", "ScriptedFacet"),
            ("MCP Tool Facet", "MCPFacet"),
        ]

        # Math submenu
        math_menu = add_menu.addMenu("Math")
        for name, ft in [
            ("Add (a + b)", "MathAddFacet"),
            ("Subtract (a - b)", "MathSubtractFacet"),
            ("Multiply (a * b)", "MathMultiplyFacet"),
            ("Divide (a / b)", "MathDivideFacet"),
            ("Min", "MathMinFacet"), ("Max", "MathMaxFacet"),
            ("Clamp", "MathClampFacet"), ("Absolute Value", "MathAbsFacet"),
        ]:
            action = math_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # Logic submenu
        logic_menu = add_menu.addMenu("Logic")
        for name, ft in [
            ("AND", "LogicAndFacet"), ("OR", "LogicOrFacet"),
            ("NOT", "LogicNotFacet"), ("Compare", "LogicCompareFacet"),
            ("Switch (If/Else)", "LogicSwitchFacet"),
        ]:
            action = logic_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # String submenu
        string_menu = add_menu.addMenu("String")
        for name, ft in [
            ("Concat", "StringConcatFacet"), ("Split", "StringSplitFacet"),
            ("Replace", "StringReplaceFacet"),
            ("Format (Template)", "StringFormatFacet"),
            ("Length", "StringLengthFacet"),
            ("Contains", "StringContainsFacet"),
            ("Regex Match", "StringRegexFacet"),
        ]:
            action = string_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # Array submenu
        array_menu = add_menu.addMenu("Array")
        for name, ft in [
            ("Get Element", "ArrayGetFacet"), ("First", "ArrayFirstFacet"),
            ("Last", "ArrayLastFacet"), ("Join", "ArrayJoinFacet"),
            ("Length", "ArrayLengthFacet"),
        ]:
            action = array_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # Data submenu
        data_menu = add_menu.addMenu("Data")
        for name, ft in [
            ("Pass Through", "PassThroughFacet"), ("Gate", "GateFacet"),
            ("Counter", "CounterFacet"),
            ("JSON Parse", "JSONParseFacet"),
            ("JSON Stringify", "JSONStringifyFacet"),
            ("Get Property", "GetPropertyFacet"),
            ("Set Property", "SetPropertyFacet"),
        ]:
            action = data_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # Neural submenu
        neural_menu = add_menu.addMenu("Neural")
        for name, ft in [
            ("Neural Canvas (NNCanvas)", "NeuralCanvasFacet"),
            ("Charm Network (.npz)", "CharmNetworkFacet"),
            ("Transformer", "TransformerFacet"),
        ]:
            action = neural_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        # Top-level cognitive facets
        for name, ft in facet_types:
            action = add_menu.addAction(name)
            action.triggered.connect(
                lambda checked, _ft=ft, _dn=name: self._add_facet(_ft, _dn, position)
            )

        add_menu.addSeparator()
        custom = add_menu.addAction("Create empty facet")
        custom.triggered.connect(
            lambda: self._add_facet("CustomFacet", "Custom Facet", position)
        )

    # ================================================================
    # Actions
    # ================================================================

    def _add_facet(self, facet_type: str, display_name: str, position):
        """Add a new facet (with undo)."""
        if self._assembly is None:
            return

        scene_pos = self.mapToScene(position)
        facet_id = Facet.generate_uuid()
        facet = Facet(
            id=facet_id,
            name=display_name,
            facet_type=facet_type,
            prompt=f"TODO: Define prompt for {display_name}",
            position={'x': scene_pos.x(), 'y': scene_pos.y()}
        )

        # Auto-create .nncanvas file for NeuralCanvasFacet (ported from old FE)
        if facet_type == "NeuralCanvasFacet":
            nncanvas_path = self._create_blank_nncanvas(facet_id, display_name)
            if nncanvas_path:
                facet.nncanvas_path = nncanvas_path

        if facet_type == "ConvergenceFacet":
            facet.add_input_pad("input1", "First input")
            facet.add_input_pad("input2", "Second input")
            facet.add_output_pad("output", "Merged output")
        else:
            facet.add_input_pad("in", "Input")
            facet.add_output_pad("out", "Output")

        from ...core.undo_manager import undo_manager
        from ...core.commands import CreateFacetCommand

        cmd = CreateFacetCommand(
            editor=self,
            facet_data=facet.to_dict(),
            facet_name=display_name
        )
        undo_manager.push(cmd)

    def _delete_selection(self):
        """Delete selected nodes with undo support."""
        selected = [
            item for item in self._scene.selectedItems()
            if isinstance(item, FacetNodeItem) and not item.is_special_node
        ]
        if not selected:
            return

        from ...core.undo_manager import undo_manager
        from ...core.commands import DeleteFacetCommand

        if len(selected) > 1:
            undo_manager.begin_group(f"Delete {len(selected)} facets")

        for node in selected:
            # Collect connections involving this facet
            conns_data = [
                c.to_dict()
                for c in (self._assembly.connections if self._assembly else [])
                if c.from_facet == node.facet.id or c.to_facet == node.facet.id
            ]
            cmd = DeleteFacetCommand(
                editor=self,
                facet_data=node.facet.to_dict(),
                connections_data=conns_data,
                facet_name=node.facet.name,
            )
            undo_manager.push(cmd)

        if len(selected) > 1:
            undo_manager.end_group()

    def _delete_connection_with_undo(self, wire: FacetConnectionItem):
        """Delete a wire connection with undo support."""
        from ...core.undo_manager import undo_manager
        from ...core.commands import DeleteConnectionCommand

        cmd = DeleteConnectionCommand(
            editor=self,
            from_facet=wire.from_port.get_parent_node_id(),
            from_pad=wire.from_port.get_port_name(),
            to_facet=wire.to_port.get_parent_node_id(),
            to_pad=wire.to_port.get_port_name(),
        )
        undo_manager.push(cmd)

    def _undo(self):
        from ...core.undo_manager import undo_manager
        undo_manager.undo()

    def _redo(self):
        from ...core.undo_manager import undo_manager
        undo_manager.redo()

    def _reset_view(self):
        """Reset view transform to identity."""
        self.resetTransform()
        self.centerOn(QPointF(500, 350))

    def _toggle_floating_editor(self):
        """Open floating editor for selected facet (E key)."""
        selected = [
            item for item in self._scene.selectedItems()
            if isinstance(item, FacetNodeItem)
        ]
        if len(selected) != 1:
            return

        node = selected[0]
        facet = node.facet

        # Find the 'prompt' field
        fields = facet.get_editable_fields()
        prompt_field = None
        for f in fields:
            if f.get('key') == 'prompt':
                prompt_field = f
                break

        if prompt_field:
            self._show_floating_editor(facet, prompt_field)

    def _show_floating_editor(self, facet: Facet, field_data: dict):
        """Show floating text editor for a facet field."""
        try:
            from ..floating_text_editor import FloatingTextEditor
        except ImportError:
            return

        editor = FloatingTextEditor(
            field_name=field_data['name'],
            field_key=field_data['key'],
            initial_value=field_data['value'],
            read_only=field_data.get('read_only', False),
            parent=self,
        )

        def on_applied(key, value):
            if key == 'prompt':
                facet.prompt = value
            self._save_assembly_to_disk()

        editor.textApplied.connect(on_applied)
        editor.move(
            self.mapToGlobal(self.rect().center()).x() - 250,
            self.mapToGlobal(self.rect().center()).y() - 200,
        )
        editor.exec()

    # ================================================================
    # Drag undo
    # ================================================================

    def _on_node_drag_finished(self):
        """Called by FacetNodeItem when a drag completes with movement."""
        self._push_move_commands_if_needed()

    def _push_move_commands_if_needed(self):
        """Compare current positions to drag starts and push undo commands."""
        if not self._drag_start_positions:
            return

        from ...core.undo_manager import undo_manager
        from ...core.commands import MoveFacetCommand

        moves = []
        for facet_id, old_pos in self._drag_start_positions.items():
            node = self._node_items.get(facet_id)
            if node is None:
                continue
            new_pos = (node.pos().x(), node.pos().y())
            dx = abs(new_pos[0] - old_pos[0])
            dy = abs(new_pos[1] - old_pos[1])
            if dx > 1 or dy > 1:
                moves.append((facet_id, old_pos, new_pos, node.facet.name))

        self._drag_start_positions.clear()

        if not moves:
            return

        if len(moves) > 1:
            undo_manager.begin_group(f"Move {len(moves)} facets")

        for facet_id, old_pos, new_pos, name in moves:
            cmd = MoveFacetCommand(
                editor=self,
                facet_id=facet_id,
                old_pos=old_pos,
                new_pos=new_pos,
                facet_name=name,
            )
            undo_manager.push(cmd)

        if len(moves) > 1:
            undo_manager.end_group()

    # ================================================================
    # Helpers
    # ================================================================

    def _find_facet(self, facet_id: str) -> Optional[Facet]:
        """Find a facet by ID in the current assembly."""
        if self._assembly is None:
            return None
        for f in self._assembly.facets:
            if f.id == facet_id:
                return f
        return None

    def _remove_wire_for_connection(
        self, from_facet: str, from_pad: str, to_facet: str, to_pad: str
    ):
        """Remove wire graphics matching the given connection."""
        to_remove = []
        for wire in self._wire_items:
            if (wire.from_port.get_parent_node_id() == from_facet
                    and wire.from_port.get_port_name() == from_pad
                    and wire.to_port.get_parent_node_id() == to_facet
                    and wire.to_port.get_port_name() == to_pad):
                to_remove.append(wire)

        for wire in to_remove:
            # Unregister from ports
            if wire in wire.from_port.connections:
                wire.from_port.connections.remove(wire)
            if wire in wire.to_port.connections:
                wire.to_port.connections.remove(wire)
            wire.to_port.update_color_from_connection()

            # Remove from scene
            if wire.scene():
                self._scene.removeItem(wire)
            self._wire_items.remove(wire)

    def _mark_dirty(self):
        """Mark the view as having unsaved changes."""
        self._dirty = True
        self.assemblyModified.emit()

    def refresh_node_for_facet(self, facet_id: str):
        """Refresh a specific node's visual after external property change."""
        node = self._node_items.get(facet_id)
        if node:
            node.update()

    def _create_blank_nncanvas(self, facet_id: str, display_name: str):
        """Create a blank .nncanvas file for a new NeuralCanvasFacet.

        Returns a project-relative path so depth navigation can resolve it,
        or None if the assembly has no known disk location.

        Creates a blank .nncanvas file alongside the assembly YAML.
        """
        import json
        import re
        from datetime import datetime

        if not self._assembly_path:
            return None

        assembly_dir = os.path.dirname(self._assembly_path)
        safe_name = re.sub(r'[^a-z0-9_]', '', display_name.lower().replace(' ', '_'))
        if not safe_name:
            safe_name = 'neural_canvas'
        nncanvas_filename = f"{safe_name}_{facet_id[:8]}.nncanvas"
        nncanvas_abs_path = os.path.join(assembly_dir, nncanvas_filename)

        blank_canvas = {
            "version": "1.0",
            "name": display_name,
            "description": "",
            "metadata": {
                "created": datetime.now().isoformat(),
                "modified": None,
                "author": "",
                "total_parameters": 0
            },
            "nodes": [],
            "connections": [],
            "hidden_states": {},
            "export_targets": {"mlx": False, "pytorch": False, "onnx": False}
        }

        try:
            with open(nncanvas_abs_path, 'w') as f:
                json.dump(blank_canvas, f, indent=2)
            print(f"[AssemblyEditor] Created .nncanvas: {nncanvas_abs_path}")
        except Exception as e:
            print(f"[AssemblyEditor] Failed to create .nncanvas: {e}")
            return None

        # Return project-relative path if possible
        main_window = self.window()
        pm = getattr(main_window, 'project_manager', None)
        if pm and getattr(pm, 'current_project_path', None):
            return os.path.relpath(nncanvas_abs_path, pm.current_project_path)
        return nncanvas_filename
