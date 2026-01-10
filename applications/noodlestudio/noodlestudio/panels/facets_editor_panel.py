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
#   Facets Editor Panel - Node-based cognitive architecture editor
#
#   Visual node graph editor for designing facet assemblies. ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.facets_editor_panel
# PURPOSE:  facets editor panel facet implementation
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetsEditorPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsLineItem, QPushButton, QLabel, QSpinBox,
    QGraphicsTextItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont
from typing import Optional, List, Dict
import asyncio

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import facet system
from ..core.facet_system import FacetAssembly

# Import graphics classes from separate module
from .facets_editor_graphics import (
    FacetNodeGraphics, FacetPadGraphics, ConnectionWire, ClickableTextItem,
    get_facet_header_color
)

# Import mixins
from .facets_editor_assembly_mixin import FacetsEditorAssemblyMixin
from .facets_editor_view_mixin import FacetsEditorViewMixin
from .facets_editor_selection_mixin import FacetsEditorSelectionMixin
from .facets_editor_wire_mixin import FacetsEditorWireMixin
from .facets_editor_events_mixin import FacetsEditorEventsMixin


class FacetsEditorPanel(
    QWidget,
    FacetsEditorAssemblyMixin,
    FacetsEditorViewMixin,
    FacetsEditorSelectionMixin,
    FacetsEditorWireMixin,
    FacetsEditorEventsMixin
):
    """
    Main facets editor panel with node graph.

    Provides visual editing of facet assemblies with drag-and-drop,
    connection wires, and right-click menus.

    Functionality is composed from mixins:
    - Assembly I/O (FacetsEditorAssemblyMixin)
    - View navigation (FacetsEditorViewMixin)
    - Selection/undo (FacetsEditorSelectionMixin)
    - Wire drawing (FacetsEditorWireMixin)
    - Events/WebSocket (FacetsEditorEventsMixin)
    """

    # Signal emitted when assembly is modified
    assemblyModified = pyqtSignal()

    # Signal emitted when a facet is selected (for Inspector)
    facetSelected = pyqtSignal(object)  # Emits Facet object

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_assembly: Optional[FacetAssembly] = None
        self.current_assembly_name: Optional[str] = None  # Track loaded assembly
        self.current_assembly_path: Optional[str] = None  # Track loaded assembly filepath
        self.node_graphics: Dict[str, FacetNodeGraphics] = {}
        self.wire_graphics: List[ConnectionWire] = []

        # CRITICAL: Lock to prevent event processing during scene transitions
        self.scene_transition_lock = False

        # Clipboard for copy/paste
        self.clipboard: List = []

        # Track drag start positions for undo (facet_id -> (x, y))
        self.drag_start_positions: Dict[str, tuple] = {}

        # Space-drag navigation
        self.space_pressed = False

        # Wire drawing state
        self.wire_being_drawn: Optional[QGraphicsLineItem] = None
        self.wire_start_pad: Optional[FacetPadGraphics] = None

        # Grid snapping settings (load from persistent settings)
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'FacetsEditor')
        self.snap_to_grid = settings.value('grid/snap_enabled', False, type=bool)
        self.grid_size = settings.value('grid/size', 20, type=int)
        self.grid_visible = self.snap_to_grid  # Grid visible when snapping enabled
        self.grid_lines: List[QGraphicsLineItem] = []  # Track grid lines for removal

        # Cognition pause state
        self.current_agent_id: Optional[str] = None
        self.cognition_paused: bool = False

        # Right-click timestamp guard (prevents trackpad zoom quirk)
        self._last_right_click_time: float = 0.0
        self._in_right_click: bool = False  # Prevents selection changes during right-click
        self._selection_signal_connected: bool = True  # Track signal connection state
        self.api_base = "http://localhost:8081/api"

        # Focus state tracking (for F key toggle)
        self.is_focused = False
        self.pre_focus_transform = None
        self.focused_node_id = None

        # Empty state message
        self.empty_state_label: Optional[QGraphicsTextItem] = None

        # WebSocket connection for execution event streaming (AUTOBAHN!)
        self.ws_connection = None
        self.ws_task = None
        self.ws_connected = False
        self.event_queue = asyncio.Queue() if WEBSOCKETS_AVAILABLE else None

        # Cycle color tracking (for async cycle visualization)
        # Maps execution_id -> QColor for consistent coloring
        self.cycle_colors: Dict[str, QColor] = {}
        self.cycle_color_palette = [
            QColor("#00BFFF"),  # Deep sky blue
            QColor("#32CD32"),  # Lime green
            QColor("#FFD700"),  # Gold
            QColor("#FF69B4"),  # Hot pink
            QColor("#00CED1"),  # Dark turquoise
            QColor("#FFA500"),  # Orange
            QColor("#9370DB"),  # Medium purple
            QColor("#20B2AA"),  # Light sea green
        ]
        self.next_cycle_color_index = 0

        self.init_ui()

        # Start WebSocket connection if available
        if WEBSOCKETS_AVAILABLE:
            self._start_websocket_connection()

    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        # Assembly info (hidden - takes up space without value)
        self.assembly_label = QLabel("No assembly loaded")
        self.assembly_label.setStyleSheet("color: #CCCCCC; font-size: 11pt; font-weight: bold;")
        self.assembly_label.hide()  # Not shown - assembly name is redundant
        toolbar.addWidget(self.assembly_label)

        toolbar.addStretch()

        # Pause/Resume cognition button
        self.pause_button = QPushButton("Pause Cognition")
        self.pause_button.setFixedWidth(140)
        self.pause_button.setCheckable(True)
        self.pause_button.setEnabled(False)  # Disabled until agent loaded
        self.pause_button.clicked.connect(self.toggle_pause_cognition)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
        """)
        toolbar.addWidget(self.pause_button)

        # Sound toggle button
        self.sound_enabled = True  # Sound on by default
        self.sound_button = QPushButton("Sound")
        self.sound_button.setFixedWidth(60)
        self.sound_button.setCheckable(True)
        self.sound_button.setChecked(True)  # On by default
        self.sound_button.setToolTip("Toggle execution sounds")
        self.sound_button.clicked.connect(self.toggle_sound)
        self.sound_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
        """)
        toolbar.addWidget(self.sound_button)

        # Grid snap toggle button
        self.grid_button = QPushButton("Grid")
        self.grid_button.setFixedWidth(50)
        self.grid_button.setCheckable(True)
        self.grid_button.setChecked(self.snap_to_grid)  # Load from settings
        self.grid_button.setToolTip("Toggle grid snapping")
        self.grid_button.clicked.connect(self.toggle_grid_snap_button)
        self.grid_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
        """)
        toolbar.addWidget(self.grid_button)

        # Grid size input
        self.grid_size_input = QSpinBox()
        self.grid_size_input.setRange(5, 100)  # 5px to 100px
        self.grid_size_input.setValue(self.grid_size)  # Load from settings
        self.grid_size_input.setSuffix("px")
        self.grid_size_input.setFixedWidth(70)
        self.grid_size_input.setToolTip("Grid size in pixels")
        self.grid_size_input.valueChanged.connect(self.on_grid_size_changed)
        self.grid_size_input.setStyleSheet("""
            QSpinBox {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
            QSpinBox:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
        """)
        toolbar.addWidget(self.grid_size_input)

        # Save/Load buttons
        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(60)
        save_btn.clicked.connect(self.save_assembly)
        toolbar.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(60)
        load_btn.clicked.connect(self.load_assembly)
        toolbar.addWidget(load_btn)

        validate_btn = QPushButton("Validate")
        validate_btn.setFixedWidth(80)
        validate_btn.clicked.connect(self.validate_assembly)
        toolbar.addWidget(validate_btn)

        layout.addLayout(toolbar)

        # Graphics view for node graph
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)

        # Draw grid background
        self.scene.setBackgroundBrush(QBrush(QColor("#2A2A2A")))
        self._draw_grid_background()

        # Connect selection changes to inspector
        self.scene.selectionChanged.connect(self.on_selection_changed)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Enable rubber band selection
        self.view.setStyleSheet("border: none;")

        # Enable mouse wheel zoom
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Override wheelEvent for zoom
        self.view.wheelEvent = self.zoom_wheel_event

        # Enable rubber band selection with modifier key support
        self.view.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)

        # Enable multi-selection (Ctrl/Cmd click, Shift-drag)
        self.scene.setSelectionArea = self.scene.setSelectionArea  # Allow additive selection

        # Enable context menu
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)

        # Install event filter for wire drawing
        self.view.viewport().installEventFilter(self)

        layout.addWidget(self.view)

        # Bottom-right floating control panel (overlay on view)
        self.control_panel = QWidget(self.view)
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(6)

        # Pause button (duplicate, more accessible)
        self.bottom_pause_btn = QPushButton("||")
        self.bottom_pause_btn.setFixedSize(40, 40)
        self.bottom_pause_btn.setCheckable(True)
        self.bottom_pause_btn.setEnabled(False)
        self.bottom_pause_btn.clicked.connect(self.toggle_pause_cognition)
        self.bottom_pause_btn.setToolTip("Pause/Resume Cognition")
        self.bottom_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 20px;
                font-size: 16pt;
            }
            QPushButton:hover:enabled {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.bottom_pause_btn)

        self.control_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(42, 42, 42, 200);
                border-radius: 6px;
                border: 1px solid #555555;
            }
        """)
        self.control_panel.setFixedSize(56, 56)  # Just one button now

        # Position control panel bottom-right (will be repositioned on resize)
        self.position_control_panel()

        # Enable key event handling
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Keyboard shortcuts
        self.setup_shortcuts()

        # Restore grid if it was enabled
        if self.grid_visible:
            self._draw_grid_background()

        # Show empty state initially
        self.show_empty_state()

    def show_empty_state(self):
        """Show 'select a noodling' message when no assembly loaded."""
        # CRITICAL: Lock scene during transition to prevent event processing
        self.scene_transition_lock = True

        # CRITICAL: Stop all animations before clearing scene to prevent segfault
        for node_gfx in self.node_graphics.values():
            if hasattr(node_gfx, 'animation_timer') and node_gfx.animation_timer:
                node_gfx.animation_timer.stop()
                node_gfx.animation_timer = None

        # Clear any existing assembly
        self.scene.clear()
        self.node_graphics.clear()
        self.wire_graphics.clear()

        # Add centered text message
        self.empty_state_label = QGraphicsTextItem("Select a noodling to edit its facets")
        self.empty_state_label.setDefaultTextColor(QColor("#CCCCCC"))
        self.empty_state_label.setFont(QFont("Arial", 14))

        # Center the text
        text_rect = self.empty_state_label.boundingRect()
        self.empty_state_label.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
        self.scene.addItem(self.empty_state_label)

        # Update label
        self.assembly_label.setText("No assembly loaded")

        # Clear agent reference
        self.current_assembly = None
        self.current_agent_id = None

        # Unlock scene - safe to process events now
        self.scene_transition_lock = False

    def clear_editor(self):
        """Clear editor when nothing is selected (alias for show_empty_state)."""
        self.show_empty_state()
        self.current_assembly_name = None

    def hide_empty_state(self):
        """Remove empty state message."""
        if self.empty_state_label and self.empty_state_label.scene():
            self.scene.removeItem(self.empty_state_label)
        self.empty_state_label = None

    def position_control_panel(self):
        """Position the floating control panel in bottom-right corner."""
        view_width = self.view.width()
        view_height = self.view.height()
        panel_width = self.control_panel.width()
        panel_height = self.control_panel.height()

        # Position 20px from bottom-right corner
        x = view_width - panel_width - 20
        y = view_height - panel_height - 20
        self.control_panel.move(x, y)
        self.control_panel.raise_()  # Ensure it's on top

    def resizeEvent(self, event):
        """Reposition control panel on window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'control_panel'):
            self.position_control_panel()

    def eventFilter(self, obj, event):
        """Handle viewport events for wire drawing, selection, and undo tracking."""
        if obj == self.view.viewport():
            # Handle LEFT BUTTON clicks only for drag/selection
            if event.type() == event.Type.MouseButtonPress:
                # Skip right-click - let context menu handle it
                if event.button() == Qt.MouseButton.RightButton:
                    try:
                        import time
                        self._last_right_click_time = time.time()  # Record for wheel guard
                        # Set flag to prevent selection changes during right-click
                        self._in_right_click = True
                        # CRITICAL: Disconnect selection signal BEFORE Qt processes right-click
                        # This prevents crashes from selection changes during context menu
                        if self._selection_signal_connected:
                            try:
                                self.scene.selectionChanged.disconnect(self.on_selection_changed)
                                self._selection_signal_connected = False
                            except:
                                pass
                        # Reconnect after a delay (context menu uses exec which blocks)
                        QTimer.singleShot(500, self._reconnect_selection_signal)
                    except Exception:
                        pass  # Silent right-click errors
                    return False  # Pass through to context menu system

                # Only handle left button for dragging/selection
                if event.button() != Qt.MouseButton.LeftButton:
                    return False

                scene_pos = self.view.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                # Filter to only facet nodes
                clicked_nodes = [
                    item for item in items
                    if isinstance(item, FacetNodeGraphics)
                ]

                if clicked_nodes:
                    # Clicked on a node - record start positions for undo
                    # Record positions of all selected nodes (they may all move together)
                    self.drag_start_positions = {}
                    for item in self.scene.selectedItems():
                        if isinstance(item, FacetNodeGraphics):
                            self.drag_start_positions[item.facet.id] = (
                                item.pos().x(), item.pos().y()
                            )
                    # Also record clicked node if not already selected
                    for node in clicked_nodes:
                        if node.facet.id not in self.drag_start_positions:
                            self.drag_start_positions[node.facet.id] = (
                                node.pos().x(), node.pos().y()
                            )
                else:
                    # Clicked empty background
                    try:
                        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            # Cmd-click - invert selection
                            self.invert_selection()
                            return True
                        else:
                            # Regular click - collapse any expanded nodes
                            self.collapse_all_nodes()
                            return False  # Allow default behavior (clear selection)
                    except Exception:
                        return False

            # Handle mouse release to push move commands
            elif event.type() == event.Type.MouseButtonRelease and not self.wire_being_drawn:
                if self.drag_start_positions:
                    self._push_move_commands_if_needed()
                    self.drag_start_positions = {}

            if event.type() == event.Type.MouseMove and self.wire_being_drawn:
                # Update wire endpoint to follow mouse
                scene_pos = self.view.mapToScene(event.pos())
                line = self.wire_being_drawn.line()
                self.wire_being_drawn.setLine(
                    line.x1(), line.y1(),
                    scene_pos.x(), scene_pos.y()
                )
                return True

            elif event.type() == event.Type.MouseButtonRelease and self.wire_being_drawn:
                # Check if released over a pad
                scene_pos = self.view.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                end_pad = None
                for item in items:
                    if isinstance(item, FacetPadGraphics):
                        end_pad = item
                        break

                if end_pad and self.wire_start_pad:
                    # Validate connection
                    if self.can_connect(self.wire_start_pad, end_pad):
                        self.create_connection(self.wire_start_pad, end_pad)

                # Clean up temporary wire
                self.scene.removeItem(self.wire_being_drawn)
                self.wire_being_drawn = None
                self.wire_start_pad = None
                return True

        return super().eventFilter(obj, event)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
