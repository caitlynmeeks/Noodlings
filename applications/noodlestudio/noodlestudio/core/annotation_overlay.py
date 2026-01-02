"""
Annotation Overlay - Visual debugging tool for screenshot communication

A transparent overlay for placing colorful annotations on top of the UI
to communicate issues, sequences, and regions of interest in screenshots.

Design: Colorful and curvy - visually distinct from monochrome UI.

Primitives:
- Dot: Simple colored marker, optionally numbered
- Arrow: Curved arrow pointing at something, with optional label
- Circle: Highlight a region loosely
- Box: Highlight a rectangular area
- Line: Show alignment or distance
- Text: Floating note

Shortcuts:
- Shift+Tab: Toggle annotations visible/hidden/edit
- 1-9: Quick place numbered dot at cursor
- D: Dot mode
- A: Arrow mode
- C: Circle mode
- B: Box mode
- T: Text mode
- R/G: Red/Green color
- Delete: Remove selected annotation
- Escape: Cancel/deselect
- Cmd+Z: Undo
- Cmd+Shift+Z: Redo
- Option+Drag: Rotate selected annotation
- Option+Click: Set custom pivot point
- Double-click: Reset pivot to center

Author: Noodlings Project
Date: December 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Any
from copy import deepcopy
import math

from PyQt6.QtWidgets import (
    QWidget, QMenu, QInputDialog, QApplication
)
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QFontMetrics, QCursor, QLinearGradient, QRadialGradient,
    QTransform
)


class AnnotationType(Enum):
    DOT = "dot"
    ARROW = "arrow"
    CIRCLE = "circle"
    BOX = "box"
    LINE = "line"
    TEXT = "text"


class AnnotationColor(Enum):
    """Vibrant colors that pop against dark UI."""
    RED = "#FF4466"      # Problems, errors
    GREEN = "#44FF88"    # Good, expected, target
    BLUE = "#4488FF"     # Info, reference
    YELLOW = "#FFEE44"   # Warning, attention
    MAGENTA = "#FF44FF"  # Secondary highlight
    CYAN = "#44FFEE"     # Tertiary highlight
    ORANGE = "#FF8844"   # Sequence markers
    WHITE = "#FFFFFF"    # Neutral


# Color display names for menus
COLOR_NAMES = {
    AnnotationColor.RED: "Red (Problem)",
    AnnotationColor.GREEN: "Green (Expected)",
    AnnotationColor.BLUE: "Blue (Info)",
    AnnotationColor.YELLOW: "Yellow (Attention)",
    AnnotationColor.MAGENTA: "Magenta",
    AnnotationColor.CYAN: "Cyan",
    AnnotationColor.ORANGE: "Orange (Sequence)",
    AnnotationColor.WHITE: "White",
}


@dataclass
class Annotation:
    """Base annotation data."""
    type: AnnotationType
    color: AnnotationColor
    pos: QPointF  # Primary position
    pos2: Optional[QPointF] = None  # Secondary position (for arrows, lines, boxes)
    text: str = ""  # Label text
    number: Optional[int] = None  # For numbered markers
    selected: bool = False
    rotation: float = 0.0  # Rotation in degrees
    pivot: Optional[QPointF] = None  # Custom pivot point (None = use center)

    def get_center(self) -> QPointF:
        """Get the center point of this annotation."""
        if self.type == AnnotationType.DOT:
            return QPointF(self.pos)
        elif self.type == AnnotationType.TEXT:
            return QPointF(self.pos)
        elif self.pos2:
            return QPointF(
                (self.pos.x() + self.pos2.x()) / 2,
                (self.pos.y() + self.pos2.y()) / 2
            )
        return QPointF(self.pos)

    def get_pivot(self) -> QPointF:
        """Get the pivot point for rotation."""
        if self.pivot:
            return self.pivot
        return self.get_center()

    def set_pivot_to_center(self):
        """Reset pivot to center."""
        self.pivot = None

    def rotate_around_pivot(self, angle_delta: float):
        """Rotate annotation by angle_delta degrees around its pivot."""
        self.rotation += angle_delta
        # Normalize to -180 to 180
        while self.rotation > 180:
            self.rotation -= 360
        while self.rotation < -180:
            self.rotation += 360

    def contains(self, point: QPointF, threshold: float = 20.0) -> bool:
        """Check if point is within selection threshold of this annotation."""
        # For rotated annotations, we need to un-rotate the test point
        if self.rotation != 0:
            pivot = self.get_pivot()
            point = self._rotate_point(point, pivot, -self.rotation)

        if self.type == AnnotationType.DOT:
            return self._distance(point, self.pos) < threshold
        elif self.type == AnnotationType.TEXT:
            return self._distance(point, self.pos) < threshold + 30
        elif self.type in (AnnotationType.ARROW, AnnotationType.LINE):
            if self.pos2:
                return self._point_to_line_distance(point, self.pos, self.pos2) < threshold
            return self._distance(point, self.pos) < threshold
        elif self.type == AnnotationType.CIRCLE:
            if self.pos2:
                radius = self._distance(self.pos, self.pos2)
                dist = self._distance(point, self.pos)
                return abs(dist - radius) < threshold
            return self._distance(point, self.pos) < threshold
        elif self.type == AnnotationType.BOX:
            if self.pos2:
                rect = QRectF(self.pos, self.pos2).normalized()
                expanded = rect.adjusted(-threshold, -threshold, threshold, threshold)
                inner = rect.adjusted(threshold, threshold, -threshold, -threshold)
                return expanded.contains(point) and not inner.contains(point)
            return self._distance(point, self.pos) < threshold
        return False

    def _rotate_point(self, point: QPointF, pivot: QPointF, angle_deg: float) -> QPointF:
        """Rotate a point around a pivot by angle in degrees."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dx = point.x() - pivot.x()
        dy = point.y() - pivot.y()
        return QPointF(
            pivot.x() + dx * cos_a - dy * sin_a,
            pivot.y() + dx * sin_a + dy * cos_a
        )

    def _distance(self, p1: QPointF, p2: QPointF) -> float:
        return math.sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)

    def _point_to_line_distance(self, point: QPointF, line_start: QPointF, line_end: QPointF) -> float:
        """Distance from point to line segment."""
        px, py = point.x(), point.y()
        x1, y1 = line_start.x(), line_start.y()
        x2, y2 = line_end.x(), line_end.y()

        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return self._distance(point, line_start)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


# =============================================================================
# Undo/Redo Command System
# =============================================================================

class AnnotationCommand:
    """Base class for undoable annotation commands."""

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError


class AddAnnotationCommand(AnnotationCommand):
    """Command for adding an annotation."""

    def __init__(self, overlay: 'AnnotationOverlay', annotation: Annotation):
        self.overlay = overlay
        self.annotation = annotation

    def undo(self):
        if self.annotation in self.overlay.annotations:
            self.overlay.annotations.remove(self.annotation)
            if self.overlay.selected_annotation == self.annotation:
                self.overlay.selected_annotation = None
        self.overlay.update()

    def redo(self):
        if self.annotation not in self.overlay.annotations:
            self.overlay.annotations.append(self.annotation)
        self.overlay.update()


class DeleteAnnotationCommand(AnnotationCommand):
    """Command for deleting an annotation."""

    def __init__(self, overlay: 'AnnotationOverlay', annotation: Annotation, index: int):
        self.overlay = overlay
        self.annotation = annotation
        self.index = index

    def undo(self):
        self.overlay.annotations.insert(self.index, self.annotation)
        self.overlay.update()

    def redo(self):
        if self.annotation in self.overlay.annotations:
            self.overlay.annotations.remove(self.annotation)
            if self.overlay.selected_annotation == self.annotation:
                self.overlay.selected_annotation = None
        self.overlay.update()


class MoveAnnotationCommand(AnnotationCommand):
    """Command for moving an annotation."""

    def __init__(self, overlay: 'AnnotationOverlay', annotation: Annotation,
                 old_pos: QPointF, new_pos: QPointF,
                 old_pos2: Optional[QPointF], new_pos2: Optional[QPointF]):
        self.overlay = overlay
        self.annotation = annotation
        self.old_pos = QPointF(old_pos)
        self.new_pos = QPointF(new_pos)
        self.old_pos2 = QPointF(old_pos2) if old_pos2 else None
        self.new_pos2 = QPointF(new_pos2) if new_pos2 else None

    def undo(self):
        self.annotation.pos = QPointF(self.old_pos)
        if self.old_pos2:
            self.annotation.pos2 = QPointF(self.old_pos2)
        self.overlay.update()

    def redo(self):
        self.annotation.pos = QPointF(self.new_pos)
        if self.new_pos2:
            self.annotation.pos2 = QPointF(self.new_pos2)
        self.overlay.update()


class RotateAnnotationCommand(AnnotationCommand):
    """Command for rotating an annotation."""

    def __init__(self, overlay: 'AnnotationOverlay', annotation: Annotation,
                 old_rotation: float, new_rotation: float):
        self.overlay = overlay
        self.annotation = annotation
        self.old_rotation = old_rotation
        self.new_rotation = new_rotation

    def undo(self):
        self.annotation.rotation = self.old_rotation
        self.overlay.update()

    def redo(self):
        self.annotation.rotation = self.new_rotation
        self.overlay.update()


class SetPivotCommand(AnnotationCommand):
    """Command for setting pivot point."""

    def __init__(self, overlay: 'AnnotationOverlay', annotation: Annotation,
                 old_pivot: Optional[QPointF], new_pivot: Optional[QPointF]):
        self.overlay = overlay
        self.annotation = annotation
        self.old_pivot = QPointF(old_pivot) if old_pivot else None
        self.new_pivot = QPointF(new_pivot) if new_pivot else None

    def undo(self):
        self.annotation.pivot = QPointF(self.old_pivot) if self.old_pivot else None
        self.overlay.update()

    def redo(self):
        self.annotation.pivot = QPointF(self.new_pivot) if self.new_pivot else None
        self.overlay.update()


class ClearAllCommand(AnnotationCommand):
    """Command for clearing all annotations."""

    def __init__(self, overlay: 'AnnotationOverlay', annotations: List[Annotation]):
        self.overlay = overlay
        # Deep copy the annotations
        self.annotations = [self._copy_annotation(a) for a in annotations]

    def _copy_annotation(self, ann: Annotation) -> Annotation:
        return Annotation(
            type=ann.type,
            color=ann.color,
            pos=QPointF(ann.pos),
            pos2=QPointF(ann.pos2) if ann.pos2 else None,
            text=ann.text,
            number=ann.number,
            selected=False,
            rotation=ann.rotation,
            pivot=QPointF(ann.pivot) if ann.pivot else None
        )

    def undo(self):
        self.overlay.annotations = [self._copy_annotation(a) for a in self.annotations]
        self.overlay.update()

    def redo(self):
        self.overlay.annotations.clear()
        self.overlay.selected_annotation = None
        self.overlay.update()


# =============================================================================
# Main Overlay Widget
# =============================================================================

class AnnotationOverlay(QWidget):
    """
    Transparent overlay widget for visual annotations.

    Place on top of MainWindow to annotate UI elements for screenshots.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Make transparent and overlay
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMouseTracking(True)

        # Annotation state
        self.annotations: List[Annotation] = []
        self.visible_annotations = True
        self.edit_mode = True  # When True, can interact with annotations

        # Undo/Redo stacks
        self.undo_stack: List[AnnotationCommand] = []
        self.redo_stack: List[AnnotationCommand] = []
        self.max_undo = 50

        # Current operation
        self.current_tool: Optional[AnnotationType] = None
        self.current_color = AnnotationColor.RED
        self.drawing = False
        self.draw_start: Optional[QPointF] = None
        self.draw_current: Optional[QPointF] = None

        # Selection and manipulation
        self.selected_annotation: Optional[Annotation] = None
        self.dragging = False
        self.drag_offset = QPointF(0, 0)
        self.drag_start_pos: Optional[QPointF] = None
        self.drag_start_pos2: Optional[QPointF] = None

        # Rotation state
        self.rotating = False
        self.rotate_start_angle: float = 0.0
        self.rotate_initial_rotation: float = 0.0

        # Numbered marker counter
        self.next_number = 1

        # Cursor position for number shortcuts
        self.last_cursor_pos = QPointF(0, 0)

        # Glow animation
        self.glow_phase = 0.0
        self.glow_timer = QTimer(self)
        self.glow_timer.timeout.connect(self._update_glow)
        self.glow_timer.start(50)  # 20 FPS glow animation

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def _push_command(self, cmd: AnnotationCommand):
        """Push a command to the undo stack and execute it."""
        cmd.redo()
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        """Undo the last command."""
        if self.undo_stack:
            cmd = self.undo_stack.pop()
            cmd.undo()
            self.redo_stack.append(cmd)

    def redo(self):
        """Redo the last undone command."""
        if self.redo_stack:
            cmd = self.redo_stack.pop()
            cmd.redo()
            self.undo_stack.append(cmd)

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    # =========================================================================
    # State management
    # =========================================================================

    def _update_glow(self):
        """Animate the glow effect."""
        self.glow_phase = (self.glow_phase + 0.1) % (2 * math.pi)
        if self.visible_annotations and self.annotations:
            self.update()

    def toggle_visibility(self):
        """Toggle annotation visibility (Shift+Tab)."""
        self.visible_annotations = not self.visible_annotations
        self.update()
        return self.visible_annotations

    def toggle_edit_mode(self):
        """Toggle between edit mode and passthrough mode."""
        self.edit_mode = not self.edit_mode
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self.edit_mode)
        self.update()
        return self.edit_mode

    def set_tool(self, tool: Optional[AnnotationType]):
        """Set current drawing tool."""
        self.current_tool = tool
        self.drawing = False
        self.draw_start = None

    def set_color(self, color: AnnotationColor):
        """Set current drawing color."""
        self.current_color = color

    def clear_all(self):
        """Remove all annotations (undoable)."""
        if self.annotations:
            cmd = ClearAllCommand(self, self.annotations)
            self._push_command(cmd)
        self.selected_annotation = None
        self.next_number = 1

    def delete_selected(self):
        """Delete currently selected annotation (undoable)."""
        if self.selected_annotation and self.selected_annotation in self.annotations:
            index = self.annotations.index(self.selected_annotation)
            cmd = DeleteAnnotationCommand(self, self.selected_annotation, index)
            self._push_command(cmd)
            self.selected_annotation = None

    def add_numbered_dot(self, pos: QPointF, number: Optional[int] = None):
        """Add a numbered dot at position (undoable)."""
        if number is None:
            number = self.next_number
            self.next_number += 1

        ann = Annotation(
            type=AnnotationType.DOT,
            color=self.current_color,
            pos=pos,
            number=number
        )
        cmd = AddAnnotationCommand(self, ann)
        self._push_command(cmd)

    def add_text(self, pos: QPointF, text: str):
        """Add a text annotation (undoable)."""
        ann = Annotation(
            type=AnnotationType.TEXT,
            color=self.current_color,
            pos=pos,
            text=text
        )
        cmd = AddAnnotationCommand(self, ann)
        self._push_command(cmd)

    # =========================================================================
    # Mouse handling
    # =========================================================================

    def mousePressEvent(self, event):
        if not self.edit_mode:
            event.ignore()
            return

        pos = QPointF(event.position())
        modifiers = event.modifiers()

        if event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Option+Click on selected annotation = set pivot
            if (modifiers & Qt.KeyboardModifier.AltModifier) and self.selected_annotation:
                old_pivot = QPointF(self.selected_annotation.pivot) if self.selected_annotation.pivot else None
                cmd = SetPivotCommand(self, self.selected_annotation, old_pivot, pos)
                self._push_command(cmd)
                return

            # Check for selection first
            clicked_ann = None
            for ann in reversed(self.annotations):  # Top to bottom
                if ann.contains(pos):
                    clicked_ann = ann
                    break

            if clicked_ann and self.current_tool is None:
                # Select annotation
                if self.selected_annotation:
                    self.selected_annotation.selected = False
                self.selected_annotation = clicked_ann
                clicked_ann.selected = True

                # Option+Drag = rotate
                if modifiers & Qt.KeyboardModifier.AltModifier:
                    self.rotating = True
                    pivot = clicked_ann.get_pivot()
                    self.rotate_start_angle = math.degrees(
                        math.atan2(pos.y() - pivot.y(), pos.x() - pivot.x())
                    )
                    self.rotate_initial_rotation = clicked_ann.rotation
                else:
                    # Normal drag = move
                    self.dragging = True
                    self.drag_offset = clicked_ann.pos - pos
                    self.drag_start_pos = QPointF(clicked_ann.pos)
                    self.drag_start_pos2 = QPointF(clicked_ann.pos2) if clicked_ann.pos2 else None

                self.update()

            elif self.current_tool:
                # Start drawing
                self.drawing = True
                self.draw_start = pos
                self.draw_current = pos

                # Single-click tools
                if self.current_tool == AnnotationType.DOT:
                    self.add_numbered_dot(pos)
                    self.drawing = False
                elif self.current_tool == AnnotationType.TEXT:
                    text, ok = QInputDialog.getText(self, "Add Note", "Text:")
                    if ok and text:
                        self.add_text(pos, text)
                    self.drawing = False
            else:
                # Deselect
                if self.selected_annotation:
                    self.selected_annotation.selected = False
                    self.selected_annotation = None
                self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click resets pivot to center."""
        if not self.edit_mode:
            event.ignore()
            return

        if self.selected_annotation and self.selected_annotation.pivot:
            old_pivot = QPointF(self.selected_annotation.pivot)
            cmd = SetPivotCommand(self, self.selected_annotation, old_pivot, None)
            self._push_command(cmd)

    def mouseMoveEvent(self, event):
        pos = QPointF(event.position())
        self.last_cursor_pos = pos

        if not self.edit_mode:
            event.ignore()
            return

        if self.rotating and self.selected_annotation:
            # Rotate around pivot
            pivot = self.selected_annotation.get_pivot()
            current_angle = math.degrees(
                math.atan2(pos.y() - pivot.y(), pos.x() - pivot.x())
            )
            delta = current_angle - self.rotate_start_angle
            self.selected_annotation.rotation = self.rotate_initial_rotation + delta
            # Normalize
            while self.selected_annotation.rotation > 180:
                self.selected_annotation.rotation -= 360
            while self.selected_annotation.rotation < -180:
                self.selected_annotation.rotation += 360
            self.update()

        elif self.dragging and self.selected_annotation:
            # Move annotation
            delta = pos + self.drag_offset - self.selected_annotation.pos
            self.selected_annotation.pos = pos + self.drag_offset
            if self.selected_annotation.pos2:
                self.selected_annotation.pos2 = self.selected_annotation.pos2 + delta
            # Also move pivot if custom
            if self.selected_annotation.pivot:
                self.selected_annotation.pivot = self.selected_annotation.pivot + delta
            self.update()

        elif self.drawing and self.draw_start:
            self.draw_current = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.edit_mode:
            event.ignore()
            return

        pos = QPointF(event.position())

        if self.rotating and self.selected_annotation:
            # Create undo command for rotation
            cmd = RotateAnnotationCommand(
                self, self.selected_annotation,
                self.rotate_initial_rotation,
                self.selected_annotation.rotation
            )
            # Don't use _push_command since we already applied the change
            self.undo_stack.append(cmd)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self.rotating = False

        if self.dragging and self.selected_annotation:
            # Create undo command for move
            if self.drag_start_pos:
                cmd = MoveAnnotationCommand(
                    self, self.selected_annotation,
                    self.drag_start_pos, self.selected_annotation.pos,
                    self.drag_start_pos2, self.selected_annotation.pos2
                )
                self.undo_stack.append(cmd)
                if len(self.undo_stack) > self.max_undo:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
            self.dragging = False
            self.drag_start_pos = None
            self.drag_start_pos2 = None

        if self.drawing and self.draw_start and self.current_tool:
            # Finish multi-point drawing
            if self.current_tool in (AnnotationType.ARROW, AnnotationType.LINE,
                                      AnnotationType.CIRCLE, AnnotationType.BOX):
                ann = Annotation(
                    type=self.current_tool,
                    color=self.current_color,
                    pos=self.draw_start,
                    pos2=pos
                )
                cmd = AddAnnotationCommand(self, ann)
                self._push_command(cmd)

            self.drawing = False
            self.draw_start = None
            self.draw_current = None
            self.update()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # Cmd+Z = Undo, Cmd+Shift+Z = Redo
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                return

        # Number keys 1-9 for quick numbered dots
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            number = key - Qt.Key.Key_1 + 1
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            self.add_numbered_dot(QPointF(cursor_pos), number)
            return

        # Tool shortcuts
        if key == Qt.Key.Key_D:
            self.set_tool(AnnotationType.DOT)
        elif key == Qt.Key.Key_A:
            self.set_tool(AnnotationType.ARROW)
        elif key == Qt.Key.Key_C:
            self.set_tool(AnnotationType.CIRCLE)
        elif key == Qt.Key.Key_B:
            self.set_tool(AnnotationType.BOX)
        elif key == Qt.Key.Key_L:
            self.set_tool(AnnotationType.LINE)
        elif key == Qt.Key.Key_T:
            self.set_tool(AnnotationType.TEXT)
        elif key == Qt.Key.Key_Escape:
            self.set_tool(None)
            if self.selected_annotation:
                self.selected_annotation.selected = False
                self.selected_annotation = None
            self.update()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        # Color shortcuts
        elif key == Qt.Key.Key_R:
            self.set_color(AnnotationColor.RED)
        elif key == Qt.Key.Key_G:
            self.set_color(AnnotationColor.GREEN)

    def _show_context_menu(self, global_pos):
        """Show right-click context menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2D2D2D;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #3E3E3E;
            }
            QMenu::separator {
                height: 1px;
                background: #555;
                margin: 4px 8px;
            }
        """)

        # Add submenu
        add_menu = menu.addMenu("Add Marker")
        for color in AnnotationColor:
            action = add_menu.addAction(COLOR_NAMES[color])
            action.triggered.connect(lambda checked, c=color: self._add_dot_at_cursor(c))

        menu.addSeparator()

        # Tool actions
        menu.addAction("Arrow (A)", lambda: self.set_tool(AnnotationType.ARROW))
        menu.addAction("Circle (C)", lambda: self.set_tool(AnnotationType.CIRCLE))
        menu.addAction("Box (B)", lambda: self.set_tool(AnnotationType.BOX))
        menu.addAction("Line (L)", lambda: self.set_tool(AnnotationType.LINE))
        menu.addAction("Text (T)", lambda: self.set_tool(AnnotationType.TEXT))

        menu.addSeparator()

        # Color submenu
        color_menu = menu.addMenu("Set Color")
        for color in AnnotationColor:
            action = color_menu.addAction(COLOR_NAMES[color])
            action.triggered.connect(lambda checked, c=color: self.set_color(c))

        menu.addSeparator()

        # Edit actions
        undo_action = menu.addAction("Undo (Cmd+Z)", self.undo)
        undo_action.setEnabled(self.can_undo())
        redo_action = menu.addAction("Redo (Cmd+Shift+Z)", self.redo)
        redo_action.setEnabled(self.can_redo())

        menu.addSeparator()

        if self.selected_annotation:
            menu.addAction("Delete Selected (Del)", self.delete_selected)
            if self.selected_annotation.pivot:
                menu.addAction("Reset Pivot (Double-click)", lambda: self._reset_selected_pivot())
        menu.addAction("Clear All", self.clear_all)

        menu.addSeparator()

        toggle_action = menu.addAction(
            "Hide Annotations (Shift+Tab)" if self.visible_annotations else "Show Annotations (Shift+Tab)"
        )
        toggle_action.triggered.connect(self.toggle_visibility)

        menu.exec(global_pos)

    def _add_dot_at_cursor(self, color: AnnotationColor):
        """Add a dot at current cursor position."""
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        old_color = self.current_color
        self.current_color = color
        self.add_numbered_dot(QPointF(cursor_pos))
        self.current_color = old_color

    def _reset_selected_pivot(self):
        """Reset selected annotation's pivot to center."""
        if self.selected_annotation and self.selected_annotation.pivot:
            old_pivot = QPointF(self.selected_annotation.pivot)
            cmd = SetPivotCommand(self, self.selected_annotation, old_pivot, None)
            self._push_command(cmd)

    # =========================================================================
    # Painting
    # =========================================================================

    def paintEvent(self, event):
        if not self.visible_annotations:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw existing annotations
        for ann in self.annotations:
            self._draw_annotation(painter, ann)

        # Draw in-progress annotation
        if self.drawing and self.draw_start and self.draw_current and self.current_tool:
            temp_ann = Annotation(
                type=self.current_tool,
                color=self.current_color,
                pos=self.draw_start,
                pos2=self.draw_current
            )
            self._draw_annotation(painter, temp_ann, preview=True)

        # Draw mode indicator
        if self.edit_mode:
            self._draw_mode_indicator(painter)

    def _draw_annotation(self, painter: QPainter, ann: Annotation, preview: bool = False):
        """Draw a single annotation."""
        color = QColor(ann.color.value)

        # Glow amount (pulsing)
        glow = 0.3 + 0.2 * math.sin(self.glow_phase) if ann.selected else 0.2

        # Apply rotation transform
        if ann.rotation != 0:
            painter.save()
            pivot = ann.get_pivot()
            painter.translate(pivot)
            painter.rotate(ann.rotation)
            painter.translate(-pivot)

        if ann.type == AnnotationType.DOT:
            self._draw_dot(painter, ann, color, glow)
        elif ann.type == AnnotationType.ARROW:
            self._draw_arrow(painter, ann, color, glow)
        elif ann.type == AnnotationType.CIRCLE:
            self._draw_circle(painter, ann, color, glow)
        elif ann.type == AnnotationType.BOX:
            self._draw_box(painter, ann, color, glow)
        elif ann.type == AnnotationType.LINE:
            self._draw_line(painter, ann, color, glow)
        elif ann.type == AnnotationType.TEXT:
            self._draw_text(painter, ann, color, glow)

        # Restore transform
        if ann.rotation != 0:
            painter.restore()

        # Draw pivot point if selected and has custom pivot
        if ann.selected and ann.pivot:
            self._draw_pivot(painter, ann.pivot)

    def _draw_pivot(self, painter: QPainter, pivot: QPointF):
        """Draw the pivot point indicator."""
        # Crosshair style pivot indicator
        size = 10
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawLine(
            QPointF(pivot.x() - size, pivot.y()),
            QPointF(pivot.x() + size, pivot.y())
        )
        painter.drawLine(
            QPointF(pivot.x(), pivot.y() - size),
            QPointF(pivot.x(), pivot.y() + size)
        )
        # Circle around it
        painter.setPen(QPen(QColor("#FFAA00"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(pivot, 6, 6)

    def _draw_dot(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a dot marker, optionally with number."""
        center = ann.pos
        radius = 16 if ann.selected else 12

        # Glow
        glow_color = QColor(color)
        glow_color.setAlphaF(glow)
        glow_gradient = QRadialGradient(center, radius * 2)
        glow_gradient.setColorAt(0, glow_color)
        glow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius * 2, radius * 2)

        # Main dot
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(center, radius, radius)

        # Number
        if ann.number is not None:
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#000000")))

            text = str(ann.number)
            fm = QFontMetrics(font)
            text_rect = fm.boundingRect(text)
            painter.drawText(
                int(center.x() - text_rect.width() / 2),
                int(center.y() + text_rect.height() / 4),
                text
            )

    def _draw_arrow(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a curved arrow."""
        if not ann.pos2:
            return

        start = ann.pos
        end = ann.pos2

        # Calculate control point for curve (offset perpendicular to line)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1:
            return

        # Perpendicular offset for curve
        curve_amount = min(length * 0.2, 40)
        ctrl = QPointF(
            (start.x() + end.x()) / 2 - dy / length * curve_amount,
            (start.y() + end.y()) / 2 + dx / length * curve_amount
        )

        # Draw glow (ensure no fill from previous operations)
        glow_color = QColor(color)
        glow_color.setAlphaF(glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(glow_color, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        path = QPainterPath()
        path.moveTo(start)
        path.quadTo(ctrl, end)
        painter.drawPath(path)

        # Draw main line
        painter.setPen(QPen(color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)

        # Arrowhead
        arrow_size = 15
        angle = math.atan2(end.y() - ctrl.y(), end.x() - ctrl.x())

        p1 = QPointF(
            end.x() - arrow_size * math.cos(angle - math.pi/6),
            end.y() - arrow_size * math.sin(angle - math.pi/6)
        )
        p2 = QPointF(
            end.x() - arrow_size * math.cos(angle + math.pi/6),
            end.y() - arrow_size * math.sin(angle + math.pi/6)
        )

        arrow_path = QPainterPath()
        arrow_path.moveTo(end)
        arrow_path.lineTo(p1)
        arrow_path.lineTo(p2)
        arrow_path.closeSubpath()

        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(arrow_path)

    def _draw_circle(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a circle highlight."""
        if not ann.pos2:
            return

        center = ann.pos
        radius = math.sqrt((ann.pos2.x() - center.x())**2 + (ann.pos2.y() - center.y())**2)

        # Glow
        glow_color = QColor(color)
        glow_color.setAlphaF(glow * 0.5)
        painter.setPen(QPen(glow_color, 12))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Main circle
        painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine))
        painter.drawEllipse(center, radius, radius)

    def _draw_box(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a box highlight."""
        if not ann.pos2:
            return

        rect = QRectF(ann.pos, ann.pos2).normalized()

        # Glow
        glow_color = QColor(color)
        glow_color.setAlphaF(glow * 0.5)
        painter.setPen(QPen(glow_color, 12))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 8, 8)

        # Main box
        painter.setPen(QPen(color, 3))
        painter.drawRoundedRect(rect, 8, 8)

    def _draw_line(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a line (for showing alignment/distance)."""
        if not ann.pos2:
            return

        # Glow
        glow_color = QColor(color)
        glow_color.setAlphaF(glow)
        painter.setPen(QPen(glow_color, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(ann.pos, ann.pos2)

        # Main line
        painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(ann.pos, ann.pos2)

        # End caps
        painter.setBrush(QBrush(color))
        painter.drawEllipse(ann.pos, 5, 5)
        painter.drawEllipse(ann.pos2, 5, 5)

        # Distance label
        dist = math.sqrt((ann.pos2.x() - ann.pos.x())**2 + (ann.pos2.y() - ann.pos.y())**2)
        mid = QPointF((ann.pos.x() + ann.pos2.x()) / 2, (ann.pos.y() + ann.pos2.y()) / 2)

        font = QFont("Arial", 9)
        painter.setFont(font)
        text = f"{dist:.0f}px"
        fm = QFontMetrics(font)
        text_rect = fm.boundingRect(text)

        # Background pill
        pill_rect = QRectF(
            mid.x() - text_rect.width() / 2 - 6,
            mid.y() - text_rect.height() / 2 - 2,
            text_rect.width() + 12,
            text_rect.height() + 4
        )
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pill_rect, 8, 8)

        painter.setPen(QPen(color))
        painter.drawText(
            int(mid.x() - text_rect.width() / 2),
            int(mid.y() + text_rect.height() / 4),
            text
        )

    def _draw_text(self, painter: QPainter, ann: Annotation, color: QColor, glow: float):
        """Draw a text note."""
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_rect = fm.boundingRect(ann.text)

        # Background pill with glow
        padding = 10
        pill_rect = QRectF(
            ann.pos.x() - padding,
            ann.pos.y() - text_rect.height() - padding / 2,
            text_rect.width() + padding * 2,
            text_rect.height() + padding
        )

        # Glow
        glow_color = QColor(color)
        glow_color.setAlphaF(glow)
        painter.setPen(QPen(glow_color, 8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(pill_rect, 12, 12)

        # Background
        bg_color = QColor(color)
        bg_color.setAlphaF(0.9)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pill_rect, 12, 12)

        # Text
        painter.setPen(QPen(QColor("#000000")))
        painter.drawText(int(ann.pos.x()), int(ann.pos.y()), ann.text)

    def _draw_mode_indicator(self, painter: QPainter):
        """Draw current tool/state indicator in corner - monochromatic, matches status bar."""
        tool_names = {
            AnnotationType.DOT: "DOT",
            AnnotationType.ARROW: "ARROW",
            AnnotationType.CIRCLE: "CIRCLE",
            AnnotationType.BOX: "BOX",
            AnnotationType.LINE: "LINE",
            AnnotationType.TEXT: "TEXT",
        }

        # Match status bar font size (12px)
        font = QFont("Arial", 12)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Build full status text
        parts = ["EDIT mode"]
        if self.current_tool:
            parts.append(f"Tool: {tool_names.get(self.current_tool, '?')}")
        parts.append(f"Color: {self.current_color.name}")

        undo_count = len(self.undo_stack)
        redo_count = len(self.redo_stack)
        if undo_count > 0 or redo_count > 0:
            parts.append(f"Undo: {undo_count} | Redo: {redo_count}")

        status_text = " | ".join(parts)
        text_rect = fm.boundingRect(status_text)

        # Position to the right of server status, vertically aligned with status bar
        x = 180
        y = self.height() - 10  # Adjusted to align with status bar text baseline

        # Single pill background - match server status pill styling (3px vertical, 8px horizontal padding)
        pill_height = text_rect.height() + 6  # ~3px padding top/bottom
        pill_rect = QRectF(x - 8, y - text_rect.height() - 3, text_rect.width() + 16, pill_height)
        painter.setBrush(QBrush(QColor(40, 40, 40, 220)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pill_rect, 4, 4)

        # Monochromatic text (light gray to match status bar style)
        painter.setPen(QPen(QColor("#AAAAAA")))
        painter.drawText(x, y, status_text)
