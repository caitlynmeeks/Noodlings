"""
Facets Editor Panel - Node-based cognitive architecture editor

Visual node graph editor for designing facet assemblies.
Unity-style node editor with drag-and-drop, connection wires, and right-click menus.

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QPushButton, QLabel, QMenu, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath, QCursor
)
from typing import Optional, List, Dict, Tuple
import sys
import os

# Import facet system
from ..core.facet_system import (
    Facet, FacetAssembly, FacetConnection, FacetPad, PadType
)


class FacetPadGraphics(QGraphicsEllipseItem):
    """Visual representation of a facet pad (connection point)."""

    PAD_RADIUS = 8

    def __init__(self, pad: FacetPad, facet_node: 'FacetNodeGraphics', parent=None):
        super().__init__(-self.PAD_RADIUS, -self.PAD_RADIUS,
                         self.PAD_RADIUS * 2, self.PAD_RADIUS * 2, parent)
        self.pad = pad
        self.facet_node = facet_node

        # Visual styling
        if pad.pad_type == PadType.INPUT:
            self.setBrush(QBrush(QColor("#64B5F6")))  # Blue for inputs
        else:
            self.setBrush(QBrush(QColor("#76AF6A")))  # Green for outputs

        self.setPen(QPen(QColor("#FFFFFF"), 2))
        self.setAcceptHoverEvents(True)

        # Connection tracking
        self.connections: List['ConnectionWire'] = []

    def hoverEnterEvent(self, event):
        """Highlight pad on hover."""
        self.setBrush(QBrush(QColor("#FFFFFF")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Restore pad color on hover exit."""
        if self.pad.pad_type == PadType.INPUT:
            self.setBrush(QBrush(QColor("#64B5F6")))
        else:
            self.setBrush(QBrush(QColor("#76AF6A")))
        super().hoverLeaveEvent(event)

    def get_scene_position(self) -> QPointF:
        """Get pad position in scene coordinates."""
        return self.scenePos()


class FacetNodeGraphics(QGraphicsRectItem):
    """Visual representation of a facet node."""

    NODE_WIDTH = 200
    NODE_HEIGHT = 120
    PAD_SPACING = 25

    def __init__(self, facet: Facet, parent=None):
        super().__init__(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT, parent)
        self.facet = facet

        # Visual styling based on type
        if facet.id == "INCOMING":
            self.setBrush(QBrush(QColor("#2E7D32")))  # Dark green
        elif facet.id == "OUTGOING":
            self.setBrush(QBrush(QColor("#D84315")))  # Dark orange
        elif "Convergence" in facet.facet_type:
            self.setBrush(QBrush(QColor("#6A1B9A")))  # Purple
        else:
            self.setBrush(QBrush(QColor("#424242")))  # Dark gray

        self.setPen(QPen(QColor("#888888"), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Title text
        self.title = QGraphicsTextItem(facet.name, self)
        self.title.setPos(10, 5)
        self.title.setDefaultTextColor(QColor("#FFFFFF"))
        font = QFont("Arial", 11, QFont.Weight.Bold)
        self.title.setFont(font)

        # Type label
        self.type_label = QGraphicsTextItem(facet.facet_type, self)
        self.type_label.setPos(10, 25)
        self.type_label.setDefaultTextColor(QColor("#AAAAAA"))
        type_font = QFont("Arial", 9)
        self.type_label.setFont(type_font)

        # Create pad graphics
        self.input_pads: Dict[str, FacetPadGraphics] = {}
        self.output_pads: Dict[str, FacetPadGraphics] = {}

        self._create_pads()

        # Set initial position from facet metadata
        self.setPos(facet.position['x'], facet.position['y'])

    def _create_pads(self):
        """Create visual representations of input/output pads."""
        # Input pads on left side
        for i, pad in enumerate(self.facet.input_pads):
            pad_graphics = FacetPadGraphics(pad, self, self)
            y_pos = 50 + (i * self.PAD_SPACING)
            pad_graphics.setPos(0, y_pos)
            self.input_pads[pad.name] = pad_graphics

            # Pad label
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(15, y_pos - 8)
            label.setDefaultTextColor(QColor("#CCCCCC"))
            label.setFont(QFont("Arial", 8))

        # Output pads on right side
        for i, pad in enumerate(self.facet.output_pads):
            pad_graphics = FacetPadGraphics(pad, self, self)
            y_pos = 50 + (i * self.PAD_SPACING)
            pad_graphics.setPos(self.NODE_WIDTH, y_pos)
            self.output_pads[pad.name] = pad_graphics

            # Pad label (right-aligned)
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(self.NODE_WIDTH - 60, y_pos - 8)
            label.setDefaultTextColor(QColor("#CCCCCC"))
            label.setFont(QFont("Arial", 8))

    def itemChange(self, change, value):
        """Handle item changes (e.g., position updates)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update facet metadata
            pos = self.pos()
            self.facet.position['x'] = pos.x()
            self.facet.position['y'] = pos.y()

            # Update connected wires
            for pad_dict in [self.input_pads, self.output_pads]:
                for pad_graphics in pad_dict.values():
                    for wire in pad_graphics.connections:
                        wire.update_path()

        return super().itemChange(change, value)


class ConnectionWire(QGraphicsItem):
    """Visual representation of a connection between facet pads."""

    def __init__(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics, parent=None):
        super().__init__(parent)
        self.from_pad = from_pad
        self.to_pad = to_pad

        # Register with pads
        self.from_pad.connections.append(self)
        self.to_pad.connections.append(self)

        # Visual styling
        self.pen = QPen(QColor("#888888"), 3)
        self.setZValue(-1)  # Draw behind nodes

    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for drawing."""
        start = self.from_pad.get_scene_position()
        end = self.to_pad.get_scene_position()

        # Add padding for bezier curve
        return QRectF(start, end).normalized().adjusted(-50, -50, 50, 50)

    def paint(self, painter: QPainter, option, widget=None):
        """Draw the connection wire as a bezier curve."""
        start = self.from_pad.get_scene_position()
        end = self.to_pad.get_scene_position()

        # Create bezier curve path
        path = QPainterPath()
        path.moveTo(start)

        # Control points for smooth curve
        dx = end.x() - start.x()
        ctrl1 = QPointF(start.x() + dx * 0.5, start.y())
        ctrl2 = QPointF(start.x() + dx * 0.5, end.y())

        path.cubicTo(ctrl1, ctrl2, end)

        # Draw
        painter.setPen(self.pen)
        painter.drawPath(path)

    def update_path(self):
        """Update path when nodes move."""
        self.prepareGeometryChange()
        self.update()


class FacetsEditorPanel(QWidget):
    """
    Main facets editor panel with node graph.

    Provides visual editing of facet assemblies with drag-and-drop,
    connection wires, and right-click menus.
    """

    # Signal emitted when assembly is modified
    assemblyModified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_assembly: Optional[FacetAssembly] = None
        self.current_assembly_name: Optional[str] = None  # Track loaded assembly
        self.node_graphics: Dict[str, FacetNodeGraphics] = {}
        self.wire_graphics: List[ConnectionWire] = []

        self.init_ui()

    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        # Assembly info
        self.assembly_label = QLabel("No assembly loaded")
        self.assembly_label.setStyleSheet("color: #CCCCCC; font-size: 11pt; font-weight: bold;")
        toolbar.addWidget(self.assembly_label)

        toolbar.addStretch()

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
        self.scene.setBackgroundBrush(QBrush(QColor("#2A2A2A")))

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setStyleSheet("border: none;")

        # Enable context menu
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)

        layout.addWidget(self.view)

    def load_assembly_from_data(self, assembly: FacetAssembly, force_reload: bool = False):
        """
        Load a facet assembly into the editor.

        Args:
            assembly: FacetAssembly to load
            force_reload: If True, reload even if same assembly already loaded
        """
        # Check if this assembly is already loaded
        if not force_reload and self.current_assembly_name == assembly.name:
            print(f"[Facets Editor] Assembly '{assembly.name}' already loaded, skipping reload")
            return

        print(f"[Facets Editor] Loading assembly: {assembly.name}")
        self.current_assembly = assembly
        self.current_assembly_name = assembly.name
        self.assembly_label.setText(f"{assembly.name} [REF]")

        # Clear existing graphics
        self.scene.clear()
        self.node_graphics.clear()
        self.wire_graphics.clear()

        # Create node graphics for each facet
        for facet in assembly.facets:
            node = FacetNodeGraphics(facet)
            self.scene.addItem(node)
            self.node_graphics[facet.id] = node

        # Create connection wires
        for conn in assembly.connections:
            from_node = self.node_graphics.get(conn.from_facet)
            to_node = self.node_graphics.get(conn.to_facet)

            if from_node and to_node:
                from_pad = from_node.output_pads.get(conn.from_pad)
                to_pad = to_node.input_pads.get(conn.to_pad)

                if from_pad and to_pad:
                    wire = ConnectionWire(from_pad, to_pad)
                    self.scene.addItem(wire)
                    self.wire_graphics.append(wire)

        # Center view on content
        self.view.centerOn(500, 350)
        print(f"[Facets Editor] Assembly loaded successfully with {len(assembly.facets)} facets")

    def show_context_menu(self, position):
        """Show right-click context menu for adding facets."""
        menu = QMenu(self)

        # Add facet submenu
        add_menu = menu.addMenu("Add Facet")

        facet_types = [
            ("Intuition Facet", "IntuitionFacet"),
            ("Emotion Facet", "EmotionFacet"),
            ("Social Context Facet", "SocialFacet"),
            ("Memory Recall Facet", "MemoryFacet"),
            ("Response Planning Facet", "PlanningFacet"),
            ("Convergence Facet", "ConvergenceFacet"),
            ("Custom Facet", "CustomFacet")
        ]

        for display_name, facet_type in facet_types:
            action = add_menu.addAction(display_name)
            action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                    self.add_facet(ft, dn, position))

        menu.exec(self.view.mapToGlobal(position))

    def add_facet(self, facet_type: str, display_name: str, position):
        """Add a new facet to the assembly."""
        print(f"[Facets Editor] add_facet called: type={facet_type}, name={display_name}")
        print(f"[Facets Editor] current_assembly exists: {self.current_assembly is not None}")

        if not self.current_assembly:
            print("[Facets Editor] ERROR: No assembly loaded - cannot add facet")
            return

        # Convert view position to scene position
        scene_pos = self.view.mapToScene(position)
        print(f"[Facets Editor] Position - view: {position}, scene: ({scene_pos.x()}, {scene_pos.y()})")

        # Create new facet with UUID
        facet_id = Facet.generate_uuid()
        facet = Facet(
            id=facet_id,
            name=display_name,
            facet_type=facet_type,
            prompt=f"TODO: Define prompt for {display_name}",
            position={'x': scene_pos.x(), 'y': scene_pos.y()}
        )
        print(f"[Facets Editor] Created facet: {facet_id}")

        # Add default pads based on type
        if facet_type == "ConvergenceFacet":
            facet.add_input_pad("input1", "First input")
            facet.add_input_pad("input2", "Second input")
            facet.add_output_pad("output", "Merged output")
        else:
            facet.add_input_pad("in", "Input")
            facet.add_output_pad("out", "Output")
        print(f"[Facets Editor] Added {len(facet.input_pads)} inputs, {len(facet.output_pads)} outputs")

        # Add to assembly
        self.current_assembly.facets.append(facet)
        print(f"[Facets Editor] Assembly now has {len(self.current_assembly.facets)} facets")

        # Create graphics
        node = FacetNodeGraphics(facet)
        self.scene.addItem(node)
        self.node_graphics[facet.id] = node
        print(f"[Facets Editor] Added node graphics to scene at ({node.pos().x()}, {node.pos().y()})")

        self.assemblyModified.emit()

    def save_assembly(self):
        """Save current assembly to YAML file."""
        if not self.current_assembly:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Facet Assembly",
            f"../facet_assemblies/{self.current_assembly.name}.yaml",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                self.current_assembly.save_yaml(filepath)
                QMessageBox.information(self, "Success", f"Assembly saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save assembly: {e}")

    def load_assembly(self):
        """Load assembly from YAML file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Facet Assembly",
            "../facet_assemblies/",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                assembly = FacetAssembly.load_yaml(filepath)
                self.load_assembly_from_data(assembly)
                QMessageBox.information(self, "Success", f"Loaded assembly: {assembly.name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load assembly: {e}")

    def validate_assembly(self):
        """Validate current assembly and show errors."""
        if not self.current_assembly:
            return

        errors = self.current_assembly.validate()

        if errors:
            error_text = "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(self, "Validation Errors", f"Assembly has errors:\n\n{error_text}")
        else:
            QMessageBox.information(self, "Validation Success", "Assembly is valid!")


if __name__ == "__main__":
    """Test the facets editor panel."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Load test assembly
    test_assembly_path = "../facet_assemblies/anklebiter_default.yaml"
    if os.path.exists(test_assembly_path):
        assembly = FacetAssembly.load_yaml(test_assembly_path)
    else:
        # Create simple test assembly
        from ..core.facet_system import create_default_assembly
        assembly = create_default_assembly()

    # Create and show editor
    editor = FacetsEditorPanel()
    editor.load_assembly_from_data(assembly)
    editor.setWindowTitle("Facets Editor - Test")
    editor.resize(1200, 800)
    editor.show()

    sys.exit(app.exec())
