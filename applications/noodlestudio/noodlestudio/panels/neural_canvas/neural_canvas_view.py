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
#   Neural Canvas View - Visual rendering of neural network graph.
#
#   Handles node rendering, wire routing, pan/zoom, and inter...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.neural_canvas_view
# PURPOSE:  Neural Canvas View
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NodeHelpDialog, PortGraphicsItem, NodeGraphicsItem, ConnectionGraphicsItem, TemporaryWireItem
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QMenu, QInputDialog,
    QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QWheelEvent, QMouseEvent, QPainterPath, QAction
)

from ...core.neural_canvas.neural_graph import NeuralGraph
from ...core.neural_canvas.neural_node import NeuralNode, NodeType, Connection
from ...core.neural_canvas.node_definitions import create_node_from_type, get_node_color, NODE_DEFINITIONS
from ..floating_text_editor import FloatingTextEditor, markdown_to_html

# Import all mixins
from .neural_canvas_input_mixin import NeuralCanvasInputMixin
from .neural_canvas_context_menu_mixin import NeuralCanvasContextMenuMixin
from .neural_canvas_node_ops_mixin import NeuralCanvasNodeOpsMixin
from .neural_canvas_view_ops_mixin import NeuralCanvasViewOpsMixin
from .neural_canvas_layout_mixin import NeuralCanvasLayoutMixin
from .neural_canvas_grid_mixin import NeuralCanvasGridMixin
from .neural_canvas_internal_mixin import NeuralCanvasInternalMixin
from .neural_canvas_test_mixin import NeuralCanvasTestMixin


class NodeHelpDialog(QDialog):
    """Floating HTML help viewer for node documentation."""

    # Class-level reference to keep one dialog open at a time
    _instance = None

    def __init__(self, node_type: NodeType, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Help: {node_type.value}")
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumSize(550, 450)
        self.resize(620, 550)

        # Get content from node definitions
        node_def = NODE_DEFINITIONS.get(node_type, {})
        how_it_works = node_def.get('how_it_works', '')
        description = node_def.get('description', 'No description available')
        name = node_def.get('name', node_type.value)

        # Convert plain text to HTML
        html_content = self._format_as_html(name, node_type.value, how_it_works or description)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # HTML browser
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html_content)
        self.browser.setStyleSheet("""
            QTextBrowser {
                background-color: #2a2a2a;
                color: #e8e8e0;
                border: none;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 16px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.browser)

        # Close button bar
        button_bar = QHBoxLayout()
        button_bar.setContentsMargins(8, 8, 8, 8)
        button_bar.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #e8e8e0;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        button_bar.addWidget(close_btn)
        layout.addLayout(button_bar)

        self.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
            }
        """)

    def _format_as_html(self, name: str, node_type: str, content: str) -> str:
        """Convert plain text content to styled HTML."""
        # Escape HTML entities in content
        import html
        content = html.escape(content)

        # Convert patterns to HTML
        lines = content.split('\n')
        html_lines = []
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Headers (lines ending with : or all caps)
            if stripped and stripped.endswith(':') and len(stripped) < 40:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h3 style="color: #8ab4f8; margin: 20px 0 10px 0; font-size: 18px; font-weight: 600;">{stripped}</h3>')

            # Bullet points
            elif stripped.startswith('- '):
                if not in_list:
                    html_lines.append('<ul style="margin: 8px 0; padding-left: 24px;">')
                    in_list = True
                html_lines.append(f'<li style="margin: 6px 0; line-height: 1.4;">{stripped[2:]}</li>')

            # Code/formula lines (contain math symbols or indented)
            elif stripped and (stripped.startswith('  ') or '=' in stripped or '->' in stripped):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<code style="display: block; background: #1e1e1e; padding: 8px 12px; margin: 8px 0; font-family: \'SF Mono\', Menlo, monospace; font-size: 15px; color: #98c379; border-radius: 4px;">{stripped}</code>')

            # Empty lines
            elif not stripped:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('<br/>')

            # Regular text
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<p style="margin: 8px 0; line-height: 1.5;">{stripped}</p>')

        if in_list:
            html_lines.append('</ul>')

        body = '\n'.join(html_lines)

        return f'''
        <html>
        <body style="margin: 0; padding: 0;">
            <h2 style="color: #e8e8e0; margin: 0 0 6px 0; font-size: 24px; font-weight: 600;">{name}</h2>
            <p style="color: #888; margin: 0 0 20px 0; font-size: 14px;">{node_type}</p>
            <hr style="border: none; border-top: 1px solid #444; margin: 0 0 20px 0;"/>
            {body}
        </body>
        </html>
        '''

    @classmethod
    def show_for_node(cls, node_type: NodeType, parent=None):
        """Show help dialog for a node type, closing any existing one."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = cls(node_type, parent)
        cls._instance.show()

    def closeEvent(self, event):
        """Clear instance reference on close."""
        NodeHelpDialog._instance = None
        super().closeEvent(event)


class PortGraphicsItem(QGraphicsItem):
    """Graphics item representing an input or output port."""

    def __init__(self, port_name: str, port, is_output: bool, index: int, is_connected: bool = False, parent=None):
        super().__init__(parent)
        self.port_name = port_name
        self.port = port  # Port object with data_type, shape, etc.
        self.is_output = is_output
        self.is_connected = is_connected
        self.index = index
        self.radius = 5

        # Vertical spacing for ports
        port_spacing = 20
        # Y position depends on whether we're showing params + whether ports come after params
        # The parent NodeGraphicsItem will need to calculate this properly
        port_start_y = 35  # Will be adjusted by parent

        # Position relative to parent node
        if is_output:
            self.setPos(180, port_start_y + index * port_spacing)  # Right edge (updated for wider nodes)
        else:
            self.setPos(0, port_start_y + index * port_spacing)  # Left edge

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(10)  # Draw on top
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        # Expand bounding rect to include label
        if self.is_output:
            # Output: label to the left of circle
            return QRectF(-80, -self.radius, 80 + self.radius, self.radius * 2)
        else:
            # Input: label to the right of circle
            return QRectF(-self.radius, -self.radius, 80 + self.radius, self.radius * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the port as a circle with label."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Port color based on data type (coffee shop palette - darker, saturated)
        data_type_colors = {
            'AFFECT': '#8A7A4A',       # Muted gold (affective warmth)
            'HIDDEN_STATE': '#4A5A6A', # Slate blue (recurrent state)
            'CELL_STATE': '#5A6A7A',   # Steel blue (LSTM cell)
            'PHENOMENAL_STATE': '#6A4A6A', # Deep mauve (consciousness)
            'TENSOR': '#5A5A5A',       # Medium gray
            'SCALAR': '#6A6A6A'        # Light gray
        }

        color = QColor(data_type_colors.get(self.port.data_type.value, '#555'))

        # Draw circle
        painter.setPen(QPen(QColor("#333"), 1.5))
        painter.setBrush(QBrush(color))
        circle_rect = QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        painter.drawEllipse(circle_rect)

        # Draw label (warm white, uniform brightness)
        painter.setPen(QColor("#e8e8e0"))  # Warm white, matches node text
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Use display label (human-readable)
        display_label = self.port.get_display_label()

        if self.is_output:
            # Output: label on left of circle
            label_rect = QRectF(-75, -self.radius, 70, self.radius * 2)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, display_label)
        else:
            # Input: label on right of circle
            label_rect = QRectF(self.radius + 5, -self.radius, 70, self.radius * 2)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, display_label)

    def hoverEnterEvent(self, event):
        """Show tooltip on hover."""
        # Format tooltip like Blender
        shape_str = f"{self.port.shape}" if self.port.shape else "dynamic"
        tooltip = f"{self.port.get_display_label()}\nTechnical: {self.port_name}\nType: {self.port.data_type.value}\nShape: {shape_str}"
        self.setToolTip(tooltip)
        super().hoverEnterEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click to rename port label."""
        new_label, ok = QInputDialog.getText(
            None,
            "Rename Port",
            f"Enter label for '{self.port_name}':",
            text=self.port.get_display_label()
        )

        if ok and new_label:
            self.port.label = new_label
            self.update()  # Redraw
            print(f"[Neural Canvas] Renamed port '{self.port_name}' to '{new_label}'")

            # Emit modification signal
            if self.scene():
                for view in self.scene().views():
                    if isinstance(view, NeuralCanvasView):
                        view.graph_modified.emit()
                        break

        super().mouseDoubleClickEvent(event)

    def get_connection_point(self) -> QPointF:
        """Get the point where wires should connect (in scene coordinates)."""
        return self.scenePos()


class NodeGraphicsItem(QGraphicsItem):
    """Graphics item representing a neural network node."""

    def __init__(self, node: NeuralNode, parent=None):
        super().__init__(parent)
        self.node = node
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Visual properties (wider to fit labels, taller for ports + params)
        self.width = 180

        # Calculate height: header + params + separator + ports
        header_height = 24
        # Estimate params count (will calculate properly in _get_display_params)
        num_params = len(node.params) if node.params else 0
        params_height = num_params * 14 + (8 if num_params > 0 else 0)  # 14px per param + separator
        max_ports = max(len(node.inputs), len(node.outputs), 1)
        ports_height = max_ports * 20 + 20  # 20px per port + margin

        # Extra height for special interactive nodes
        if node.type == NodeType.NUMBER_INPUT:
            extra_height = 60  # Space for slider + value text
        elif node.type == NodeType.THRESHOLD_OUTPUT:
            extra_height = 90  # Space for indicator + info text
        elif node.type == NodeType.COMMENT:
            # COMMENT nodes have special sizing based on text content
            comment_text = node.params.get('text', '')
            comment_width = node.params.get('width', 320)
            comment_height = node.params.get('height', None)

            self.width = comment_width
            if comment_height:
                # Use explicit height if provided
                self.height = comment_height
            else:
                # Calculate height based on text (rough estimate: 16px per line)
                lines = comment_text.split('\n')
                # Wrap long lines (estimate 7 chars per 10px at font size 11)
                chars_per_line = int((comment_width - 24) / 7)
                wrapped_lines = 0
                for line in lines:
                    if len(line) == 0:
                        wrapped_lines += 1
                    else:
                        wrapped_lines += max(1, (len(line) + chars_per_line - 1) // chars_per_line)
                text_height = wrapped_lines * 16 + 16  # line height + padding
                self.height = 28 + text_height  # header + text
            extra_height = 0  # Already calculated
            # Skip normal height calculation
            header_height = 0
            params_height = 0
            ports_height = 0
        else:
            extra_height = 0

        self.height = header_height + params_height + ports_height + extra_height if node.type != NodeType.COMMENT else self.height
        self.setPos(node.position[0], node.position[1])

        # Calculate where ports start (after header + params)
        display_params = self._get_display_params()
        self.params_height = len(display_params) * 14 + (8 if display_params else 0)
        self.port_start_y = header_height + self.params_height + 8

        # Create port items
        self.input_ports: dict[str, PortGraphicsItem] = {}
        self.output_ports: dict[str, PortGraphicsItem] = {}

        for i, (port_name, port) in enumerate(node.inputs.items()):
            port_item = PortGraphicsItem(port_name, port, is_output=False, index=i, is_connected=False, parent=self)
            # Adjust position to account for params
            port_item.setPos(0, self.port_start_y + i * 20)
            self.input_ports[port_name] = port_item

        for i, (port_name, port) in enumerate(node.outputs.items()):
            port_item = PortGraphicsItem(port_name, port, is_output=True, index=i, is_connected=False, parent=self)
            # Adjust position to account for params
            port_item.setPos(self.width, self.port_start_y + i * 20)
            self.output_ports[port_name] = port_item

        # Drag tracking for undo
        self.drag_start_pos = None
        self.is_being_dragged = False

        # Test mode values (displayed during test inference)
        self.test_values: dict = {}  # port_name -> value

        # Slider interaction state (for NUMBER_INPUT nodes)
        self._slider_rect = None
        self._slider_x = 0
        self._slider_width = 0
        self._slider_dragging = False
        self._slider_value_before_drag = None  # For undo

        # Help icon rect (for click detection)
        self._help_icon_rect = QRectF(self.width - 20, 4, 16, 16)

        # COMMENT node resize state
        self._comment_resizing = False
        self._resize_start_pos = None
        self._resize_start_size = None
        self._comment_width_before_resize = None  # For undo

    def mousePressEvent(self, event):
        """Record position when drag starts (for undo)."""
        from PyQt6.QtCore import Qt
        if event.button() == Qt.MouseButton.LeftButton:
            local_pos = event.pos()
            modifiers = event.modifiers()
            is_cmd_click = modifiers & Qt.KeyboardModifier.MetaModifier or modifiers & Qt.KeyboardModifier.ControlModifier

            # Cmd-click on COMMENT node opens in floating editor
            if self.node.type == NodeType.COMMENT and is_cmd_click:
                comment_text = self.node.params.get('text', '')
                self._show_floating_editor(
                    title=f"Comment: {self.node.name}",
                    content=comment_text,
                    render_markdown=True
                )
                event.accept()
                return

            # Check if clicking on help icon (not for COMMENT nodes)
            if self.node.type != NodeType.COMMENT and self._help_icon_rect.contains(local_pos):
                # Show help in FloatingTextEditor with markdown
                self._show_node_help()
                event.accept()
                return

            # Check if clicking on COMMENT node resize handle
            if self.node.type == NodeType.COMMENT:
                resize_rect = QRectF(self.width - 12, self.height - 12, 12, 12)
                if resize_rect.contains(local_pos):
                    self._comment_resizing = True
                    self._resize_start_pos = local_pos
                    self._resize_start_size = (self.width, self.height)
                    self._comment_width_before_resize = self.node.params.get('width', self.width)  # Store for undo
                    event.accept()
                    return

            # Check if clicking on slider (for NUMBER_INPUT nodes)
            if self.node.type == NodeType.NUMBER_INPUT and self._slider_rect:
                # Expand hit area vertically for easier interaction
                slider_hit_rect = QRectF(
                    self._slider_rect.x() - 5,
                    self._slider_rect.y() - 10,
                    self._slider_rect.width() + 10,
                    self._slider_rect.height() + 20
                )
                if slider_hit_rect.contains(local_pos):
                    self._slider_dragging = True
                    self._slider_value_before_drag = self.node.params.get('value', 0)  # Store for undo
                    self._update_slider_value(local_pos.x())
                    event.accept()
                    return

            self.drag_start_pos = (int(self.pos().x()), int(self.pos().y()))
            self.is_being_dragged = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for slider dragging and COMMENT resizing."""
        if self._slider_dragging:
            self._update_slider_value(event.pos().x())
            event.accept()
            return

        # Handle COMMENT node resizing
        if self._comment_resizing and self._resize_start_pos:
            delta = event.pos() - self._resize_start_pos
            new_width = max(150, self._resize_start_size[0] + delta.x())
            new_height = max(60, self._resize_start_size[1] + delta.y())

            self.prepareGeometryChange()
            self.width = new_width
            self.height = new_height
            self.node.params['width'] = int(new_width)
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Push move command when drag ends (if position changed)."""
        from PyQt6.QtCore import Qt

        # Handle slider release
        if event.button() == Qt.MouseButton.LeftButton and self._slider_dragging:
            self._slider_dragging = False
            # Push undo command if value changed
            new_value = self.node.params.get('value', 0)
            if self._slider_value_before_drag is not None and new_value != self._slider_value_before_drag:
                if self.scene():
                    for view in self.scene().views():
                        if isinstance(view, NeuralCanvasView):
                            from ...core.undo_manager import undo_manager
                            from ...core.commands import EditNeuralNodeParamCommand

                            cmd = EditNeuralNodeParamCommand(
                                view=view,
                                node_id=self.node.id,
                                param_name='value',
                                old_value=self._slider_value_before_drag,
                                new_value=new_value,
                                node_name=self.node.name
                            )
                            undo_manager.push(cmd)
                            break
            self._slider_value_before_drag = None
            event.accept()
            return

        # Handle COMMENT resize release
        if event.button() == Qt.MouseButton.LeftButton and self._comment_resizing:
            self._comment_resizing = False
            self._resize_start_pos = None
            self._resize_start_size = None
            # Push undo command if width changed
            new_width = self.node.params.get('width', self.width)
            if self._comment_width_before_resize is not None and new_width != self._comment_width_before_resize:
                if self.scene():
                    for view in self.scene().views():
                        if isinstance(view, NeuralCanvasView):
                            from ...core.undo_manager import undo_manager
                            from ...core.commands import EditNeuralNodeParamCommand

                            cmd = EditNeuralNodeParamCommand(
                                view=view,
                                node_id=self.node.id,
                                param_name='width',
                                old_value=self._comment_width_before_resize,
                                new_value=new_width,
                                node_name=self.node.name
                            )
                            undo_manager.push(cmd)
                            break
            self._comment_width_before_resize = None
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.is_being_dragged:
            self.is_being_dragged = False

            if self.drag_start_pos:
                new_pos = (int(self.pos().x()), int(self.pos().y()))
                old_pos = self.drag_start_pos

                # Only push command if position changed significantly
                if abs(new_pos[0] - old_pos[0]) > 1 or abs(new_pos[1] - old_pos[1]) > 1:
                    # Push move command via UndoManager
                    if self.scene():
                        for view in self.scene().views():
                            if isinstance(view, NeuralCanvasView):
                                from ...core.undo_manager import undo_manager
                                from ...core.commands import MoveNeuralNodeCommand

                                cmd = MoveNeuralNodeCommand(
                                    view=view,
                                    node_id=self.node.id,
                                    old_pos=old_pos,
                                    new_pos=new_pos,
                                    node_name=self.node.name
                                )
                                undo_manager.push(cmd)
                                break

            self.drag_start_pos = None

        super().mouseReleaseEvent(event)

    def _update_slider_value(self, x: float):
        """Update NUMBER_INPUT slider value from mouse x position."""
        if not self._slider_width or self._slider_width <= 0:
            return

        # Calculate normalized position (0-1)
        normalized = (x - self._slider_x) / self._slider_width
        normalized = max(0.0, min(1.0, normalized))

        # Convert to actual value range
        min_val = self.node.params.get('min_value', 0.0)
        max_val = self.node.params.get('max_value', 1.0)
        step = self.node.params.get('step', 0.1)

        new_value = min_val + normalized * (max_val - min_val)

        # Snap to step
        if step > 0:
            new_value = round(new_value / step) * step
            new_value = max(min_val, min(max_val, new_value))

        # Update node parameter
        self.node.params['value'] = new_value

        # Trigger repaint
        self.update()

        # Emit param changed signal via the view
        scene = self.scene()
        if scene:
            views = scene.views()
            if views and hasattr(views[0], 'node_param_changed'):
                views[0].node_param_changed.emit(self.node.id, 'value', new_value)

    def _get_display_params(self) -> dict:
        """Get key parameters to display inline on the node (moved up for init access)."""
        display = {}

        # Special nodes have custom UI, no inline params
        if self.node.type in (NodeType.NUMBER_INPUT, NodeType.THRESHOLD_OUTPUT):
            return display

        if self.node.type in (NodeType.LSTM, NodeType.GRU, NodeType.RNN):
            if 'hidden_dim' in self.node.params:
                display['hidden'] = self.node.params['hidden_dim']
            if 'dropout' in self.node.params and self.node.params['dropout'] > 0:
                display['dropout'] = self.node.params['dropout']

        elif self.node.type == NodeType.LINEAR:
            if 'out_features' in self.node.params:
                display['out'] = self.node.params['out_features']

        elif self.node.type == NodeType.DROPOUT:
            if 'p' in self.node.params:
                display['p'] = self.node.params['p']

        elif self.node.type == NodeType.AFFECT_HEAD:
            if 'hidden_dim' in self.node.params:
                display['hidden'] = self.node.params['hidden_dim']

        elif self.node.type == NodeType.QUANTUM_MICROTUBULE:
            if 'collapse_threshold' in self.node.params:
                display['collapse'] = self.node.params['collapse_threshold']
            if 'noise_scale' in self.node.params:
                display['noise'] = self.node.params['noise_scale']

        elif self.node.type == NodeType.IBM_QUANTUM:
            if 'num_qubits' in self.node.params:
                display['qubits'] = self.node.params['num_qubits']
            if 'shots' in self.node.params:
                display['shots'] = self.node.params['shots']

        # Show total parameters if node has weights
        total_params = self.node.compute_num_parameters()
        if total_params > 0:
            display['params'] = f"{total_params:,}"

        return display

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the node (Blender-style: colored header, gray body)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # COMMENT nodes have completely different rendering
        if self.node.type == NodeType.COMMENT:
            self._paint_comment_node(painter)
            return

        rect = self.boundingRect()

        # Main body background (uniform dark gray) - rounded corners
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.drawRoundedRect(rect, 4, 4)

        # Header bar with taxonomic color (sharp edges, clipped by node outline)
        header_height = 24
        header_rect = QRectF(0, 0, self.width, header_height)

        # Get node type color (rich, earthy tones)
        header_color = QColor(get_node_color(self.node.type))
        painter.setBrush(QBrush(header_color))
        painter.drawRect(header_rect)  # Sharp rectangle, not rounded

        # Selection highlight (white outline with padding, drawn LAST on top)
        if self.isSelected():
            padding = 3  # Pixels between node and selection box
            selection_rect = rect.adjusted(-padding, -padding, padding, padding)
            painter.setPen(QPen(QColor("#FFFFFF"), 3))  # 3px border (matches Facets)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(selection_rect, 4, 4)

        # Node name in header (warm white, uniform brightness)
        painter.setPen(QColor("#e8e8e0"))  # Warm white
        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            header_rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.node.name
        )

        # Help icon (question mark circle) - right side of header
        help_x = self.width - 20
        help_y = 4
        help_size = 16

        # Draw subtle circle background
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.setBrush(QBrush(QColor("#4a4a4a")))
        painter.drawEllipse(QRectF(help_x, help_y, help_size, help_size))

        # Draw question mark
        painter.setPen(QColor("#aaaaaa"))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(help_x, help_y, help_size, help_size),
            Qt.AlignmentFlag.AlignCenter,
            "?"
        )

        # Update help icon rect for click detection
        self._help_icon_rect = QRectF(help_x, help_y, help_size, help_size)

        # Draw inline parameters (between header and ports)
        self._paint_parameters(painter, header_height)

        # Draw special interactive elements for tutorial nodes
        if self.node.type == NodeType.NUMBER_INPUT:
            self._paint_number_input_slider(painter)
        elif self.node.type == NodeType.THRESHOLD_OUTPUT:
            self._paint_threshold_output(painter)
        # Draw test values if present (for non-special nodes)
        elif self.test_values:
            self._paint_test_values(painter)

    def _paint_number_input_slider(self, painter: QPainter):
        """Paint interactive slider for NUMBER_INPUT node."""
        # Slider track area
        slider_y = self.port_start_y + 25
        slider_x = 15
        slider_width = self.width - 30
        slider_height = 8

        # Get current value (normalized 0-1)
        value = self.node.params.get('value', 0.5)
        min_val = self.node.params.get('min_value', 0.0)
        max_val = self.node.params.get('max_value', 1.0)
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        # Draw track background (dark)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#252525")))
        track_rect = QRectF(slider_x, slider_y, slider_width, slider_height)
        painter.drawRoundedRect(track_rect, 3, 3)

        # Draw filled portion (green gradient)
        filled_width = slider_width * normalized
        painter.setBrush(QBrush(QColor("#4A8A4A")))
        filled_rect = QRectF(slider_x, slider_y, filled_width, slider_height)
        painter.drawRoundedRect(filled_rect, 3, 3)

        # Draw thumb
        thumb_x = slider_x + filled_width - 6
        thumb_y = slider_y - 2
        thumb_size = 12
        painter.setBrush(QBrush(QColor("#CCCCCC")))
        painter.setPen(QPen(QColor("#888888"), 1))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_size, thumb_size))

        # Draw value text
        painter.setPen(QColor("#e8e8e0"))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        value_text = f"{value:.2f}"
        value_rect = QRectF(slider_x, slider_y + slider_height + 5, slider_width, 16)
        painter.drawText(value_rect, Qt.AlignmentFlag.AlignCenter, value_text)

        # Store slider geometry for mouse interaction
        self._slider_rect = track_rect
        self._slider_x = slider_x
        self._slider_width = slider_width

    def _paint_threshold_output(self, painter: QPainter):
        """Paint ON/OFF indicator for THRESHOLD_OUTPUT node."""
        # Indicator area
        indicator_y = self.port_start_y + 10
        indicator_x = self.width / 2 - 25
        indicator_size = 50

        # Get state from test values or params
        is_on = self.test_values.get('is_on', False)
        value = self.test_values.get('value', 0.0)
        threshold = self.node.params.get('threshold', 0.5)

        # Draw indicator light
        painter.setPen(QPen(QColor("#333333"), 2))

        if is_on:
            # ON - bright green with glow effect
            painter.setBrush(QBrush(QColor("#44FF44")))
            # Draw glow (outer ring)
            glow_rect = QRectF(indicator_x - 5, indicator_y - 5, indicator_size + 10, indicator_size + 10)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(68, 255, 68, 80)))
            painter.drawEllipse(glow_rect)
        else:
            # OFF - dim gray/red
            painter.setBrush(QBrush(QColor("#553333")))

        # Main indicator circle
        painter.setPen(QPen(QColor("#222222"), 2))
        indicator_rect = QRectF(indicator_x, indicator_y, indicator_size, indicator_size)
        painter.drawEllipse(indicator_rect)

        # Draw ON/OFF text
        painter.setPen(QColor("#FFFFFF" if is_on else "#666666"))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        status_text = "ON" if is_on else "OFF"
        painter.drawText(indicator_rect, Qt.AlignmentFlag.AlignCenter, status_text)

        # Draw value and threshold info below
        painter.setPen(QColor("#e8e8e0"))
        font = QFont("Arial", 8)
        painter.setFont(font)
        info_y = indicator_y + indicator_size + 8
        info_rect = QRectF(10, info_y, self.width - 20, 12)
        if isinstance(value, float):
            info_text = f"Value: {value:.3f} (threshold: {threshold:.2f})"
        else:
            info_text = f"threshold: {threshold:.2f}"
        painter.drawText(info_rect, Qt.AlignmentFlag.AlignCenter, info_text)

    def _paint_test_values(self, painter: QPainter):
        """
        Paint test inference values on the node.

        Shows output values from test mode as a floating badge.
        """
        if not self.test_values:
            return

        # Draw a semi-transparent overlay at bottom of node
        font = QFont("Monospace", 7)
        painter.setFont(font)

        # Format values
        lines = []
        for port_name, value in self.test_values.items():
            if isinstance(value, list):
                if len(value) <= 3:
                    formatted = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in value]
                    lines.append(f"{port_name}: [{', '.join(formatted)}]")
                elif len(value) <= 5:
                    # Format as affect-like output
                    formatted = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in value]
                    lines.append(f"[{', '.join(formatted)}]")
                else:
                    lines.append(f"{port_name}: [{len(value)} vals]")
            elif isinstance(value, float):
                lines.append(f"{port_name}: {value:.3f}")
            else:
                lines.append(f"{port_name}: {value}")

        if not lines:
            return

        # Draw badge background (green tinted for active values)
        badge_height = len(lines) * 12 + 8
        badge_rect = QRectF(4, self.height - badge_height - 4, self.width - 8, badge_height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(60, 90, 60, 200)))  # Dark green, semi-transparent
        painter.drawRoundedRect(badge_rect, 3, 3)

        # Draw border
        painter.setPen(QPen(QColor(80, 140, 80), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(badge_rect, 3, 3)

        # Draw text
        painter.setPen(QColor("#aaffaa"))  # Light green text
        y = self.height - badge_height
        for line in lines:
            painter.drawText(QRectF(8, y, self.width - 16, 12),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           line)
            y += 12

    def set_test_values(self, values: dict):
        """Set test inference values and trigger repaint."""
        self.test_values = values
        self.update()

    def clear_test_values(self):
        """Clear test values and trigger repaint."""
        self.test_values = {}
        self.update()

    def _show_floating_editor(self, title: str, content: str, render_markdown: bool = False):
        """Show content in a floating text editor."""
        # Get parent view to position relative to
        views = self.scene().views() if self.scene() else []
        view = views[0] if views else None

        editor = FloatingTextEditor(
            field_name=title,
            field_key="help_content",
            initial_value=content,
            read_only=True,
            render_markdown=render_markdown,
            parent=view
        )

        # Position near the node but offset
        if view:
            scene_pos = self.mapToScene(QPointF(self.width + 20, 0))
            view_pos = view.mapFromScene(scene_pos)
            global_pos = view.mapToGlobal(view_pos)
            editor.move(global_pos)

        editor.resize(550, 450)
        editor.show()

    def _show_node_help(self):
        """Show help documentation for this node type in floating editor."""
        node_def = NODE_DEFINITIONS.get(self.node.type, {})
        how_it_works = node_def.get('how_it_works', '')
        description = node_def.get('description', 'No description available')
        name = node_def.get('name', self.node.type.value)

        # Format as markdown
        content = how_it_works or description

        # Add a header with the node type
        markdown_content = f"# {name}\n\n`{self.node.type.value}`\n\n---\n\n{content}"

        self._show_floating_editor(
            title=f"Help: {name}",
            content=markdown_content,
            render_markdown=True
        )

    def _paint_comment_node(self, painter: QPainter):
        """
        Paint COMMENT node as a floating text box (tutorial explainer).

        These are purely decorative nodes for adding explanations to canvases.
        Resizable via bottom-right handle. Coffee-colored header.
        """
        rect = self.boundingRect()
        comment_text = self.node.params.get('text', '')

        # Semi-transparent background with subtle border
        painter.setPen(QPen(QColor("#4a4540"), 1))
        painter.setBrush(QBrush(QColor(42, 40, 38, 235)))  # Warm dark, slightly transparent
        painter.drawRoundedRect(rect, 6, 6)

        # Header bar (coffee/tobacco brown)
        header_height = 28
        header_rect = QRectF(0, 0, self.width, header_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#5c4a3d")))  # Coffee brown
        painter.drawRoundedRect(header_rect.adjusted(1, 1, -1, 0), 5, 5)
        # Square off bottom of header
        painter.drawRect(QRectF(1, header_height - 6, self.width - 2, 6))

        # Title text
        painter.setPen(QColor("#e8e0d8"))  # Warm cream
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        title_rect = QRectF(10, 0, self.width - 20, header_height)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                        self.node.name)

        # Draw text content with word wrapping
        painter.setPen(QColor("#c8c0b8"))  # Warm gray
        font = QFont("Arial", 11)
        painter.setFont(font)

        text_rect = QRectF(12, header_height + 8, self.width - 24, self.height - header_height - 20)

        # Qt's drawText with TextWordWrap handles wrapping
        painter.drawText(text_rect,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                        comment_text)

        # Resize handle (bottom-right corner) - diagonal lines
        handle_size = 12
        handle_x = self.width - handle_size
        handle_y = self.height - handle_size

        painter.setPen(QPen(QColor("#6a5a4a"), 1))
        # Draw three diagonal lines (standard resize grip)
        for i in range(3):
            offset = i * 4
            painter.drawLine(
                int(handle_x + offset + 4), int(handle_y + handle_size - 2),
                int(handle_x + handle_size - 2), int(handle_y + offset + 4)
            )

        # Selection highlight
        if self.isSelected():
            padding = 3
            selection_rect = rect.adjusted(-padding, -padding, padding, padding)
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(selection_rect, 6, 6)

    def _paint_parameters(self, painter: QPainter, y_offset: float):
        """
        Paint inline parameter values (Blender-style).

        Shows key parameters like hidden_dim, dropout, etc. directly on node.
        """
        font = QFont("Arial", 8)
        painter.setFont(font)

        current_y = y_offset + 6

        # Key parameters to display inline
        display_params = self._get_display_params()

        # Uniform warm white for all text
        painter.setPen(QColor("#e8e8e0"))

        for param_name, param_value in display_params.items():
            # Parameter name
            param_rect = QRectF(10, current_y, 70, 12)
            painter.drawText(param_rect, Qt.AlignmentFlag.AlignLeft, param_name + ":")

            # Parameter value (same color, uniform brightness)
            value_rect = QRectF(85, current_y, 85, 12)

            # Format value
            if isinstance(param_value, float):
                value_str = f"{param_value:.3f}"
            elif isinstance(param_value, bool):
                value_str = "yes" if param_value else "no"
            else:
                value_str = str(param_value)

            painter.drawText(value_rect, Qt.AlignmentFlag.AlignRight, value_str)

            current_y += 14

        # Draw separator line before ports
        if display_params:
            painter.setPen(QPen(QColor("#555"), 1))
            painter.drawLine(8, int(current_y + 2), int(self.width - 8), int(current_y + 2))

    def itemChange(self, change, value):
        """Handle item changes (e.g., position)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Apply grid snapping if enabled
            new_pos = value
            views = self.scene().views() if self.scene() else []
            if views and hasattr(views[0], 'snap_to_grid'):
                view = views[0]
                if view.snap_to_grid:
                    grid = view.grid_size
                    snapped_x = round(new_pos.x() / grid) * grid
                    snapped_y = round(new_pos.y() / grid) * grid
                    return QPointF(snapped_x, snapped_y)
            return new_pos

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update node position
            new_pos = self.pos()
            self.node.position = (int(new_pos.x()), int(new_pos.y()))

            # Force redraw of all wires connected to this node
            if self.scene():
                self.scene().update()

        return super().itemChange(change, value)

    def get_port_item(self, port_name: str, is_output: bool) -> PortGraphicsItem:
        """Get port graphics item by name."""
        if is_output:
            return self.output_ports.get(port_name)
        else:
            return self.input_ports.get(port_name)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to rename node (with undo support)."""
        # Check if double-clicking on header
        header_height = 24
        local_pos = event.pos()

        if local_pos.y() <= header_height:
            # Double-clicked header - rename node
            old_name = self.node.name
            new_name, ok = QInputDialog.getText(
                None,
                "Rename Node",
                "Enter new name:",
                text=old_name
            )

            if ok and new_name and new_name != old_name:
                # Push rename command via UndoManager
                if self.scene():
                    for view in self.scene().views():
                        if isinstance(view, NeuralCanvasView):
                            from ...core.undo_manager import undo_manager
                            from ...core.commands import RenameNeuralNodeCommand

                            cmd = RenameNeuralNodeCommand(
                                view=view,
                                node_id=self.node.id,
                                old_name=old_name,
                                new_name=new_name
                            )
                            undo_manager.push(cmd)
                            print(f"[Neural Canvas] Renamed node (undoable): {old_name} -> {new_name}")
                            break

        super().mouseDoubleClickEvent(event)


class ConnectionGraphicsItem(QGraphicsItem):
    """Graphics item representing a connection between nodes."""

    def __init__(self, connection: Connection, from_item: NodeGraphicsItem, to_item: NodeGraphicsItem, parent=None):
        super().__init__(parent)
        self.connection = connection
        self.from_item = from_item
        self.to_item = to_item
        self.setZValue(-1)  # Draw behind nodes

    def boundingRect(self) -> QRectF:
        """Bounding rectangle for the connection."""
        from_port = self.from_item.get_port_item(self.connection.from_port, is_output=True)
        to_port = self.to_item.get_port_item(self.connection.to_port, is_output=False)

        if from_port and to_port:
            from_pos = from_port.get_connection_point()
            to_pos = to_port.get_connection_point()
        else:
            # Fallback to node centers
            from_pos = self.from_item.scenePos() + QPointF(self.from_item.width, self.from_item.height / 2)
            to_pos = self.to_item.scenePos() + QPointF(0, self.to_item.height / 2)

        return QRectF(from_pos, to_pos).normalized().adjusted(-10, -10, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the connection wire (orthogonal routing, color-coded by type)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # Sharp lines

        # Get port positions
        from_port = self.from_item.get_port_item(self.connection.from_port, is_output=True)
        to_port = self.to_item.get_port_item(self.connection.to_port, is_output=False)

        if from_port and to_port:
            from_pos = from_port.get_connection_point()
            to_pos = to_port.get_connection_point()
            # Wire color matches output port data type
            wire_color = self._get_wire_color(from_port.port.data_type.value)
        else:
            # Fallback
            from_pos = self.from_item.scenePos() + QPointF(self.from_item.width, self.from_item.height / 2)
            to_pos = self.to_item.scenePos() + QPointF(0, self.to_item.height / 2)
            wire_color = "#888"

        # Wire color based on data type (matches port colors)
        pen = QPen(QColor(wire_color), 2.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)  # Smooth curves

        # Bezier curve routing (Blender-style)
        # Control points extend horizontally from ports
        distance = abs(to_pos.x() - from_pos.x())
        handle_distance = min(distance * 0.5, 100)  # Adaptive handle length

        control1 = QPointF(from_pos.x() + handle_distance, from_pos.y())  # Horizontal right
        control2 = QPointF(to_pos.x() - handle_distance, to_pos.y())      # Horizontal left

        path = QPainterPath()
        path.moveTo(from_pos)
        path.cubicTo(control1, control2, to_pos)  # Smooth Bezier curve

        painter.drawPath(path)

        # Arrowhead at target (no fill, just outline to match Blender)
        arrow_size = 6
        painter.setBrush(QBrush(QColor(wire_color)))
        arrow = QPainterPath()
        arrow.moveTo(to_pos)
        arrow.lineTo(to_pos.x() - arrow_size, to_pos.y() - arrow_size / 2)
        arrow.lineTo(to_pos.x() - arrow_size, to_pos.y() + arrow_size / 2)
        arrow.closeSubpath()
        painter.drawPath(arrow)

    def _get_wire_color(self, data_type: str) -> str:
        """Get wire color based on data type (coffee shop palette - matches ports)."""
        data_type_colors = {
            'AFFECT': '#8A7A4A',       # Muted gold
            'HIDDEN_STATE': '#4A5A6A', # Slate blue
            'CELL_STATE': '#5A6A7A',   # Steel blue
            'PHENOMENAL_STATE': '#6A4A6A', # Deep mauve
            'TENSOR': '#5A5A5A',       # Medium gray
            'SCALAR': '#6A6A6A'        # Light gray
        }
        return data_type_colors.get(data_type, '#555')


class TemporaryWireItem(QGraphicsItem):
    """Temporary wire shown while dragging to create a connection."""

    def __init__(self, start_pos: QPointF, parent=None):
        super().__init__(parent)
        self.start_pos = start_pos
        self.end_pos = start_pos
        self.setZValue(5)  # Above connections, below ports

    def set_end_pos(self, pos: QPointF):
        """Update the end position of the wire."""
        self.prepareGeometryChange()
        self.end_pos = pos

    def boundingRect(self) -> QRectF:
        return QRectF(self.start_pos, self.end_pos).normalized().adjusted(-10, -10, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        """Render temporary wire."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Dashed line
        pen = QPen(QColor("#FFA500"), 2, Qt.PenStyle.DashLine)  # Orange dashed
        painter.setPen(pen)

        # Simple straight line for now
        painter.drawLine(self.start_pos, self.end_pos)


class NeuralCanvasView(
    NeuralCanvasInputMixin,
    NeuralCanvasContextMenuMixin,
    NeuralCanvasNodeOpsMixin,
    NeuralCanvasViewOpsMixin,
    NeuralCanvasLayoutMixin,
    NeuralCanvasGridMixin,
    NeuralCanvasInternalMixin,
    NeuralCanvasTestMixin,
    QGraphicsView
):
    """
    Graphics view for rendering and interacting with neural network graph.

    Features:
    - Pan (middle mouse drag)
    - Zoom (scroll wheel)
    - Node dragging
    - Node selection
    - Grid (optional)
    """

    # Signals
    node_selected = pyqtSignal(str)  # node_id
    node_param_changed = pyqtSignal(str, str, object)  # node_id, param_name, new_value
    graph_modified = pyqtSignal()

    def __init__(self, graph: NeuralGraph, parent=None):
        super().__init__(parent)

        self.graph = graph
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Set large scene rect to allow infinite panning (Blender-style)
        # Without this, Qt limits scrolling to item bounds
        self.scene.setSceneRect(-10000, -10000, 20000, 20000)

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Enable rectangle selection
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Style (lighter background for better visibility)
        self.setStyleSheet("""
            QGraphicsView {
                background: #2a2a2a;
                border: none;
            }
        """)

        # Interaction state
        self.panning = False
        self.last_pan_pos = QPointF()
        self.space_pressed = False

        # Add node mode
        self.add_node_mode = False
        self.add_node_type = None

        # Right-click timestamp guard (prevents trackpad zoom quirk)
        self._last_right_click_time: float = 0.0

        # Wire dragging mode
        self.wire_drag_mode = False
        self.wire_drag_start_port: PortGraphicsItem = None
        self.wire_drag_start_node: NodeGraphicsItem = None
        self.temp_wire: TemporaryWireItem = None

        # Focus state (F key toggle)
        self.is_focused = False
        self.focused_node_id: str = None
        self.pre_focus_transform = None

        # Graphics items cache
        self.node_items: dict[str, NodeGraphicsItem] = {}
        self.connection_items: list[ConnectionGraphicsItem] = []

        # Grid snapping (load from persistent settings)
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'NeuralCanvas')
        self.snap_to_grid = settings.value('grid/snap_enabled', False, type=bool)
        self.grid_size = settings.value('grid/size', 20, type=int)
        self.grid_visible = self.snap_to_grid
        self.grid_lines: list = []

        # Initial render
        self._render_graph()

        # Restore grid if it was enabled
        if self.grid_visible:
            self._draw_grid()

    def set_graph(self, graph: NeuralGraph):
        """Set a new graph to display."""
        self.graph = graph
        self._render_graph()

    def _render_graph(self):
        """Render the graph as graphics items."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()

        # Create node items
        for node_id, node in self.graph.nodes.items():
            item = NodeGraphicsItem(node)
            self.scene.addItem(item)
            self.node_items[node_id] = item

        # Create connection items
        for connection in self.graph.connections:
            from_item = self.node_items.get(connection.from_node)
            to_item = self.node_items.get(connection.to_node)

            if from_item and to_item:
                conn_item = ConnectionGraphicsItem(connection, from_item, to_item)
                self.scene.addItem(conn_item)
                self.connection_items.append(conn_item)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
