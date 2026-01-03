"""
Qt Widget Renderer

Renders UIComponent trees using Qt Widgets.

This is the v1 desktop renderer. The component abstraction layer allows
future renderers (WebGL, etc.) to read the same ui.yaml files without
changing user projects.
"""

from typing import Any, Callable, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette

from .component import UIComponent, Anchors


def hex_to_qcolor(hex_color: str) -> QColor:
    """Convert hex color string to QColor."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return QColor(r, g, b)
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16)
        return QColor(r, g, b, a)
    return QColor(hex_color)


class QtWidgetRenderer:
    """
    Renders UIComponent trees to Qt Widgets.

    Usage:
        renderer = QtWidgetRenderer()
        root_widget = renderer.render(component_tree)
        root_widget.show()
    """

    def __init__(self):
        # Map component names to their widgets (for event wiring)
        self._widget_map: Dict[str, QWidget] = {}

        # Map component names to their UIComponents (for value access)
        self._component_map: Dict[str, UIComponent] = {}

        # Event dispatcher callback
        self._event_dispatcher: Optional[Callable] = None

    def set_event_dispatcher(self, dispatcher: Callable) -> None:
        """
        Set the callback for handling UI events.

        Args:
            dispatcher: Callable(event_name, component, binding)
        """
        self._event_dispatcher = dispatcher

    def render(self, component: UIComponent, parent: Optional[QWidget] = None) -> QWidget:
        """
        Render a component tree to Qt widgets.

        Args:
            component: Root UIComponent
            parent: Optional parent Qt widget

        Returns:
            Root QWidget
        """
        widget = self._render_component(component, parent)
        return widget

    def _render_component(self, component: UIComponent, parent: Optional[QWidget]) -> QWidget:
        """Render a single component based on its type."""
        from .components.panel import Panel
        from .components.label import Label
        from .components.button import Button
        from .components.text_input import TextInput
        from .components.radiance_viewport import RadianceViewport

        if isinstance(component, Panel):
            widget = self._render_panel(component, parent)
        elif isinstance(component, Label):
            widget = self._render_label(component, parent)
        elif isinstance(component, Button):
            widget = self._render_button(component, parent)
        elif isinstance(component, TextInput):
            widget = self._render_text_input(component, parent)
        elif isinstance(component, RadianceViewport):
            widget = self._render_radiance_viewport(component, parent)
        else:
            # Fallback: render as simple frame
            widget = self._render_frame(component, parent)

        # Store references
        if component.name:
            self._widget_map[component.name] = widget
            self._component_map[component.name] = component
        component._widget = widget

        # Apply common properties
        self._apply_geometry(widget, component, parent)
        widget.setVisible(component.visible)
        widget.setEnabled(component.enabled)

        # Render children
        for child in component.children:
            self._render_component(child, widget)

        return widget

    def _render_frame(self, component: UIComponent, parent: Optional[QWidget]) -> QFrame:
        """Render a generic frame."""
        frame = QFrame(parent)
        frame.setObjectName(component.name or "frame")
        return frame

    def _render_panel(self, component: 'Panel', parent: Optional[QWidget]) -> QFrame:
        """Render a Panel component."""
        from .components.panel import Panel

        frame = QFrame(parent)
        frame.setObjectName(component.name or "panel")

        # Build stylesheet
        style_parts = [f"background-color: {component.background};"]

        if component.border_color and component.border_width > 0:
            style_parts.append(f"border: {component.border_width}px solid {component.border_color};")

        if component.border_radius > 0:
            style_parts.append(f"border-radius: {component.border_radius}px;")

        frame.setStyleSheet(f"QFrame#{frame.objectName()} {{ {' '.join(style_parts)} }}")

        return frame

    def _render_label(self, component: 'Label', parent: Optional[QWidget]) -> QLabel:
        """Render a Label component."""
        from .components.label import Label, TextAlign, TextVAlign

        label = QLabel(component.text, parent)
        label.setObjectName(component.name or "label")

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        if component.font_weight == "bold":
            font.setBold(True)
        if component.font_family:
            font.setFamily(component.font_family)
        label.setFont(font)

        # Alignment
        h_align = {
            TextAlign.LEFT: Qt.AlignmentFlag.AlignLeft,
            TextAlign.CENTER: Qt.AlignmentFlag.AlignHCenter,
            TextAlign.RIGHT: Qt.AlignmentFlag.AlignRight,
        }.get(component.align, Qt.AlignmentFlag.AlignLeft)

        v_align = {
            TextVAlign.TOP: Qt.AlignmentFlag.AlignTop,
            TextVAlign.MIDDLE: Qt.AlignmentFlag.AlignVCenter,
            TextVAlign.BOTTOM: Qt.AlignmentFlag.AlignBottom,
        }.get(component.valign, Qt.AlignmentFlag.AlignVCenter)

        label.setAlignment(h_align | v_align)

        # Word wrap
        label.setWordWrap(component.word_wrap)

        # Style
        label.setStyleSheet(f"QLabel {{ color: {component.text_color}; background: transparent; }}")

        return label

    def _render_button(self, component: 'Button', parent: Optional[QWidget]) -> QPushButton:
        """Render a Button component."""
        from .components.button import Button

        button = QPushButton(component.text, parent)
        button.setObjectName(component.name or "button")

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        button.setFont(font)

        # Calculate hover/pressed colors if not specified
        bg = component.background
        hover_bg = component.hover_background or self._lighten_color(bg, 10)
        pressed_bg = component.pressed_background or self._darken_color(bg, 10)

        # Style
        style = f"""
            QPushButton {{
                color: {component.text_color};
                background-color: {bg};
                border: none;
                border-radius: {component.border_radius}px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            QPushButton:pressed {{
                background-color: {pressed_bg};
            }}
            QPushButton:disabled {{
                background-color: #3a3a3a;
                color: #666666;
            }}
        """
        button.setStyleSheet(style)

        # Connect click event
        if "onClick" in component.events and self._event_dispatcher:
            binding = component.events["onClick"]
            button.clicked.connect(
                lambda: self._event_dispatcher("onClick", component, binding)
            )

        return button

    def _render_text_input(self, component: 'TextInput', parent: Optional[QWidget]) -> QLineEdit:
        """Render a TextInput component."""
        from .components.text_input import TextInput

        line_edit = QLineEdit(parent)
        line_edit.setObjectName(component.name or "text_input")
        line_edit.setText(component.value)
        line_edit.setPlaceholderText(component.placeholder)
        line_edit.setReadOnly(component.read_only)

        if component.max_length > 0:
            line_edit.setMaxLength(component.max_length)

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        line_edit.setFont(font)

        # Style
        style = f"""
            QLineEdit {{
                color: {component.text_color};
                background-color: {component.background};
                border: {component.border_width}px solid {component.border_color};
                border-radius: {component.border_radius}px;
                padding: 4px 8px;
            }}
            QLineEdit:focus {{
                border-color: #3b82f6;
            }}
            QLineEdit::placeholder {{
                color: {component.placeholder_color};
            }}
        """
        line_edit.setStyleSheet(style)

        # Connect events
        if "onChange" in component.events and self._event_dispatcher:
            binding = component.events["onChange"]
            line_edit.textChanged.connect(
                lambda text: self._handle_text_change(component, text, binding)
            )

        if "onSubmit" in component.events and self._event_dispatcher:
            binding = component.events["onSubmit"]
            line_edit.returnPressed.connect(
                lambda: self._event_dispatcher("onSubmit", component, binding)
            )

        return line_edit

    def _handle_text_change(self, component: 'TextInput', text: str, binding) -> None:
        """Handle text input changes."""
        component.value = text  # Update component state
        if self._event_dispatcher:
            self._event_dispatcher("onChange", component, binding)

    def _render_radiance_viewport(self, component: 'RadianceViewport', parent: Optional[QWidget]) -> QWidget:
        """Render a RadianceViewport component."""
        from .components.radiance_viewport import RadianceViewport, RadianceViewportWidget

        # Create the custom viewport widget
        viewport_widget = RadianceViewportWidget(component, parent)

        return viewport_widget

    def _apply_geometry(self, widget: QWidget, component: UIComponent, parent: Optional[QWidget]) -> None:
        """Apply geometry and anchor constraints."""
        geom = component.geometry

        # Set initial geometry
        widget.setGeometry(geom.x, geom.y, geom.width, geom.height)

        # Mark margins as uninitialized - they'll be calculated on first resize
        # when we have actual parent sizes
        geom._margin_right = None
        geom._margin_bottom = None

    def get_widget(self, name: str) -> Optional[QWidget]:
        """Get a rendered widget by component name."""
        return self._widget_map.get(name)

    def get_component(self, name: str) -> Optional[UIComponent]:
        """Get a component by name."""
        return self._component_map.get(name)

    def get_value(self, name: str) -> Any:
        """Get the current value of a component by name."""
        component = self._component_map.get(name)
        if component:
            from .components.text_input import TextInput
            if isinstance(component, TextInput):
                return component.value
        return None

    def _lighten_color(self, hex_color: str, percent: int) -> str:
        """Lighten a hex color by percent."""
        color = hex_to_qcolor(hex_color)
        h, s, l, a = color.getHsl()
        l = min(255, l + int(255 * percent / 100))
        color.setHsl(h, s, l, a)
        return color.name()

    def _darken_color(self, hex_color: str, percent: int) -> str:
        """Darken a hex color by percent."""
        color = hex_to_qcolor(hex_color)
        h, s, l, a = color.getHsl()
        l = max(0, l - int(255 * percent / 100))
        color.setHsl(h, s, l, a)
        return color.name()


class AnchoredWidget(QWidget):
    """
    A widget container that handles anchor-based resizing.

    This wraps the root widget and manages resize events to
    update child positions based on their anchor settings.
    """

    def __init__(self, root_component: UIComponent, renderer: QtWidgetRenderer):
        super().__init__()
        self.root_component = root_component
        self.renderer = renderer

        # Render the component tree (hidden initially to prevent artifacts)
        self.root_widget = renderer.render(root_component, self)
        self.root_widget.hide()

    def showEvent(self, event):
        """Initialize geometry when first shown."""
        super().showEvent(event)
        # Make root widget fill this container
        self.root_widget.setGeometry(0, 0, self.width(), self.height())
        # Update all anchored children
        self._update_all_geometry()
        # Now show the properly laid-out widget tree
        self.root_widget.show()

    def resizeEvent(self, event):
        """Handle resize by updating anchored children."""
        super().resizeEvent(event)
        new_size = event.size()
        # Root widget fills the container
        self.root_widget.setGeometry(0, 0, new_size.width(), new_size.height())
        # Update children
        self._update_anchored_children(self.root_component, new_size.width(), new_size.height())

    def _update_all_geometry(self):
        """Update all geometry from current sizes."""
        self._update_anchored_children(self.root_component, self.width(), self.height())

    def _update_anchored_children(self, component: UIComponent, parent_width: int, parent_height: int):
        """Recursively update children based on anchors."""
        for child in component.children:
            widget = child._widget
            if not widget:
                continue

            geom = child.geometry
            anchors = child.anchors

            # Calculate new geometry based on anchors
            x = geom.x
            y = geom.y
            width = geom.width
            height = geom.height

            # Handle horizontal anchoring
            if anchors.left and anchors.right:
                # Stretch to fill width (respect x position, fill to right edge)
                # If component didn't specify explicit width (default 100), fill completely
                if geom._margin_right is None:
                    # First time: if width is default (100), assume user wants full stretch
                    if geom.width == 100:
                        geom._margin_right = 0
                    else:
                        geom._margin_right = max(0, parent_width - (geom.x + geom.width))
                width = max(1, parent_width - geom.x - geom._margin_right)
            elif anchors.right and not anchors.left:
                # Stick to right edge - component should be at right with small margin
                if geom._margin_right is None:
                    # If x is default (0), user wants it flush to right
                    if geom.x == 0:
                        geom._margin_right = 0
                    else:
                        # x was explicitly set, use it as margin from right
                        geom._margin_right = geom.x
                x = parent_width - geom._margin_right - width
            else:
                # Just left anchor (default) - use fixed position
                if geom._margin_right is None:
                    geom._margin_right = 0

            # Handle vertical anchoring
            if anchors.top and anchors.bottom:
                # Stretch to fill height
                if geom._margin_bottom is None:
                    if geom.height == 32:  # default height
                        geom._margin_bottom = 0
                    else:
                        geom._margin_bottom = max(0, parent_height - (geom.y + geom.height))
                height = max(1, parent_height - geom.y - geom._margin_bottom)
            elif anchors.bottom and not anchors.top:
                # Stick to bottom edge - component should be at bottom with small margin
                if geom._margin_bottom is None:
                    # If y is default (0), user wants it flush to bottom
                    if geom.y == 0:
                        geom._margin_bottom = 0
                    else:
                        # y was explicitly set, use it as margin from bottom
                        geom._margin_bottom = geom.y
                y = parent_height - geom._margin_bottom - height
            else:
                # Just top anchor (default) - use fixed position
                if geom._margin_bottom is None:
                    geom._margin_bottom = 0

            widget.setGeometry(int(x), int(y), int(width), int(height))

            # Recurse for nested children using the child's actual rendered size
            if child.children:
                self._update_anchored_children(child, int(width), int(height))
