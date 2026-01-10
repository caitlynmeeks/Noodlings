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
#   Qt Widget Renderer
#
#   Renders UIComponent trees using Qt Widgets. This is the v...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.renderer
# PURPOSE:  Qt Widget Renderer
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   QtWidgetRenderer, AnchoredWidget, ChatHistoryWidget, ChatInputWidget, WebViewWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Any, Callable, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QSizePolicy, QScrollArea,
    QSpacerItem, QCheckBox, QComboBox, QSlider, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

from .component import UIComponent, Anchors
from .event_data import UIEventData, EVENT_CLICK, EVENT_CHANGE, EVENT_SUBMIT
from .event_widgets import (
    EventEmittingFrame,
    EventEmittingButton,
    EventEmittingLineEdit,
)

logger = logging.getLogger(__name__)


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

        # Binding manager (lazy-initialized)
        self._binding_manager: Optional['BindingManager'] = None

        # Project path for resolving relative asset paths
        self._project_path: Optional[str] = None

    def _get_binding_manager(self) -> 'BindingManager':
        """Get or create the binding manager."""
        if self._binding_manager is None:
            from .bindings import BindingManager
            self._binding_manager = BindingManager(self)
        return self._binding_manager

    def set_event_dispatcher(self, dispatcher: Callable) -> None:
        """
        Set the callback for handling UI events.

        Args:
            dispatcher: Callable(event_name, component, binding)
        """
        self._event_dispatcher = dispatcher

    def set_project_path(self, path: str) -> None:
        """
        Set the project path for resolving relative asset paths.

        Args:
            path: Absolute path to project directory
        """
        self._project_path = path

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

        # Set up bindings after all components are rendered
        self._setup_bindings(component)

        # Evaluate all bindings to set initial values
        if self._binding_manager:
            self._binding_manager.evaluate_all()

        return widget

    def _setup_bindings(self, component: UIComponent) -> None:
        """Set up bindings for a component and its children."""
        # Add component's bindings to the manager
        if component.bindings and component.name:
            manager = self._get_binding_manager()
            for target_property, source_expression in component.bindings.items():
                manager.add_binding(
                    target_component=component.name,
                    target_property=target_property,
                    source_expression=source_expression
                )

        # Recurse to children
        for child in component.children:
            self._setup_bindings(child)

    def notify_binding_change(self, component_name: str, property_name: str = 'value') -> None:
        """
        Notify the binding manager of a value change.

        Call this when a component's value changes to update any bound targets.
        """
        if self._binding_manager:
            self._binding_manager.notify_change(component_name, property_name)

    def _render_component(self, component: UIComponent, parent: Optional[QWidget]) -> QWidget:
        """Render a single component based on its type."""
        from .components.panel import Panel
        from .components.label import Label
        from .components.button import Button
        from .components.text_input import TextInput
        from .components.radiance_viewport import RadianceViewport
        from .components.vrm_viewport import VRMViewport
        from .components.chat_history import ChatHistory
        from .components.chat_input import ChatInput
        from .components.checkbox import Checkbox
        from .components.dropdown import Dropdown
        from .components.slider import Slider
        from .components.radio import RadioButton, RadioGroup
        from .components.webview import WebView

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
        elif isinstance(component, VRMViewport):
            widget = self._render_vrm_viewport(component, parent)
        elif isinstance(component, ChatHistory):
            widget = self._render_chat_history(component, parent)
        elif isinstance(component, ChatInput):
            widget = self._render_chat_input(component, parent)
        elif isinstance(component, Checkbox):
            widget = self._render_checkbox(component, parent)
        elif isinstance(component, Dropdown):
            widget = self._render_dropdown(component, parent)
        elif isinstance(component, Slider):
            widget = self._render_slider(component, parent)
        elif isinstance(component, RadioButton):
            widget = self._render_radio_button(component, parent)
        elif isinstance(component, RadioGroup):
            widget = self._render_radio_group(component, parent)
        elif isinstance(component, WebView):
            widget = self._render_webview(component, parent)
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

        # Use event-emitting frame for full event support
        frame = EventEmittingFrame(component, self, parent)
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

        # Use event-emitting button for full event support
        button = EventEmittingButton(component, self, component.text, parent)
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

        # Connect click event via Qt signal (provides UIEventData through mixin)
        if EVENT_CLICK in component.events and self._event_dispatcher:
            binding = component.events[EVENT_CLICK]
            button.clicked.connect(
                lambda: self._event_dispatcher(
                    EVENT_CLICK,
                    component,
                    binding,
                    UIEventData.click(component.name)
                )
            )

        return button

    def _render_text_input(self, component: 'TextInput', parent: Optional[QWidget]) -> QLineEdit:
        """Render a TextInput component."""
        from .components.text_input import TextInput

        # Use event-emitting line edit for full event support
        line_edit = EventEmittingLineEdit(component, self, parent)
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

        # Track previous value for change events
        self._previous_values = getattr(self, '_previous_values', {})
        self._previous_values[component.name] = component.value

        # Connect events - always track text changes for bindings
        line_edit.textChanged.connect(
            lambda text: self._handle_text_input_change(component, text)
        )

        if EVENT_SUBMIT in component.events and self._event_dispatcher:
            binding = component.events[EVENT_SUBMIT]
            line_edit.returnPressed.connect(
                lambda: self._event_dispatcher(
                    EVENT_SUBMIT,
                    component,
                    binding,
                    UIEventData.submit(component.name, component.value)
                )
            )

        return line_edit

    def _handle_text_input_change(self, component: 'TextInput', text: str) -> None:
        """Handle text input changes (for bindings and events)."""
        # Get previous value for event data
        previous_values = getattr(self, '_previous_values', {})
        previous_value = previous_values.get(component.name)

        component.value = text  # Update component state

        # Update tracked previous value
        previous_values[component.name] = text

        # Notify binding manager
        if component.name:
            self.notify_binding_change(component.name, 'value')

        # Fire onChange event if configured
        if EVENT_CHANGE in component.events and self._event_dispatcher:
            binding = component.events[EVENT_CHANGE]
            self._event_dispatcher(
                EVENT_CHANGE,
                component,
                binding,
                UIEventData.value_change(component.name, text, previous_value)
            )

    def _render_radiance_viewport(self, component: 'RadianceViewport', parent: Optional[QWidget]) -> QWidget:
        """Render a RadianceViewport component."""
        from .components.radiance_viewport import RadianceViewport, RadianceViewportWidget
        from pathlib import Path
        import sys

        print(f"[DEBUG] _render_radiance_viewport called for: {component.name}", file=sys.stderr)
        print(f"[DEBUG] radiance_path: {component.radiance_path!r}", file=sys.stderr)
        print(f"[DEBUG] project_path: {self._project_path!r}", file=sys.stderr)

        # Create the custom viewport widget
        viewport_widget = RadianceViewportWidget(component, parent)

        # Load radiance file if specified
        if component.radiance_path:
            radiance_path = component.radiance_path

            # Resolve relative path against project path if available
            if not Path(radiance_path).is_absolute() and self._project_path:
                radiance_path = str(Path(self._project_path) / radiance_path)

            if Path(radiance_path).exists():
                viewport_widget.load_file(radiance_path, component.name or "viewport")
                logger.debug(f"Loaded radiance: {radiance_path}")
            else:
                logger.warning(f"Radiance file not found: {radiance_path}")

        return viewport_widget

    def _render_vrm_viewport(self, component: 'VRMViewport', parent: Optional[QWidget]) -> QWidget:
        """Render a VRMViewport component."""
        from .components.vrm_viewport import VRMViewport, VRMViewportWidget
        from pathlib import Path

        logger.debug(f"Creating VRMViewport: {component.name}")

        # Create the OpenGL viewport widget
        viewport_widget = VRMViewportWidget(component, parent)

        # Load VRM file if specified
        if component.vrm_path:
            vrm_path = component.vrm_path

            # Resolve relative path against project path if available
            if not Path(vrm_path).is_absolute() and self._project_path:
                vrm_path = str(Path(self._project_path) / vrm_path)

            if Path(vrm_path).exists():
                viewport_widget.load_vrm(vrm_path)
                logger.debug(f"Loaded VRM: {vrm_path}")
            else:
                logger.warning(f"VRM file not found: {vrm_path}")

        return viewport_widget

    def _render_chat_history(self, component: 'ChatHistory', parent: Optional[QWidget]) -> QWidget:
        """Render a ChatHistory component."""
        widget = ChatHistoryWidget(component, self, parent)
        return widget

    def _render_chat_input(self, component: 'ChatInput', parent: Optional[QWidget]) -> QWidget:
        """Render a ChatInput component."""
        widget = ChatInputWidget(component, self, parent)
        return widget

    def _render_checkbox(self, component: 'Checkbox', parent: Optional[QWidget]) -> QCheckBox:
        """Render a Checkbox component."""
        from .components.checkbox import Checkbox

        checkbox = QCheckBox(component.text, parent)
        checkbox.setObjectName(component.name or "checkbox")
        checkbox.setChecked(component.checked)

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        checkbox.setFont(font)

        # Style
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {component.text_color};
                spacing: {component.spacing}px;
            }}
            QCheckBox::indicator {{
                width: {component.box_size}px;
                height: {component.box_size}px;
                border: 2px solid {component.box_color};
                border-radius: 3px;
                background-color: transparent;
            }}
            QCheckBox::indicator:checked {{
                background-color: {component.check_color};
                border-color: {component.check_color};
            }}
            QCheckBox::indicator:hover {{
                border-color: {component.check_color};
            }}
        """)

        # Connect events
        def on_state_changed(state):
            component.checked = (state == Qt.CheckState.Checked.value)
            if component.name:
                self.notify_binding_change(component.name, 'checked')
                self.notify_binding_change(component.name, 'value')
            if EVENT_CHANGE in component.events and self._event_dispatcher:
                binding = component.events[EVENT_CHANGE]
                self._event_dispatcher(
                    EVENT_CHANGE,
                    component,
                    binding,
                    UIEventData.value_change(component.name, component.checked, not component.checked)
                )

        checkbox.stateChanged.connect(on_state_changed)

        return checkbox

    def _render_dropdown(self, component: 'Dropdown', parent: Optional[QWidget]) -> QComboBox:
        """Render a Dropdown component."""
        from .components.dropdown import Dropdown

        combo = QComboBox(parent)
        combo.setObjectName(component.name or "dropdown")
        combo.setEditable(component.editable)

        # Add placeholder as first item if set
        if component.placeholder:
            combo.addItem(component.placeholder)
            combo.setItemData(0, False, Qt.ItemDataRole.UserRole)  # Mark as placeholder

        # Add options
        for opt in component.options:
            combo.addItem(opt)

        # Set selection
        if component.selected_index >= 0:
            # Account for placeholder offset
            idx = component.selected_index + (1 if component.placeholder else 0)
            combo.setCurrentIndex(idx)
        elif component.placeholder:
            combo.setCurrentIndex(0)

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        combo.setFont(font)

        # Style
        combo.setStyleSheet(f"""
            QComboBox {{
                color: {component.text_color};
                background-color: {component.background};
                border: 1px solid {component.border_color};
                border-radius: {component.border_radius}px;
                padding: 4px 8px;
                padding-right: 24px;
            }}
            QComboBox:hover {{
                background-color: {component.hover_background};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {component.text_color};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {component.dropdown_background};
                color: {component.text_color};
                selection-background-color: {component.item_hover_background};
                border: 1px solid {component.border_color};
            }}
        """)

        # Connect events
        def on_index_changed(index):
            # Calculate actual index (accounting for placeholder)
            offset = 1 if component.placeholder else 0
            actual_index = index - offset
            if actual_index >= 0:
                old_value = component.value
                component.selected_index = actual_index
                if component.name:
                    self.notify_binding_change(component.name, 'value')
                    self.notify_binding_change(component.name, 'selected_index')
                if EVENT_CHANGE in component.events and self._event_dispatcher:
                    binding = component.events[EVENT_CHANGE]
                    self._event_dispatcher(
                        EVENT_CHANGE,
                        component,
                        binding,
                        UIEventData.value_change(component.name, component.value, old_value)
                    )

        combo.currentIndexChanged.connect(on_index_changed)

        return combo

    def _render_slider(self, component: 'Slider', parent: Optional[QWidget]) -> QWidget:
        """Render a Slider component."""
        from .components.slider import Slider

        # Create container for slider + optional value label
        container = QWidget(parent)
        container.setObjectName(component.name or "slider")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create slider
        slider = QSlider(Qt.Orientation.Horizontal)

        # Qt sliders use integers, so we scale to get float precision
        scale = 1000  # 3 decimal places
        slider.setMinimum(int(component.min_value * scale))
        slider.setMaximum(int(component.max_value * scale))
        slider.setValue(int(component.value * scale))

        if component.step > 0:
            slider.setSingleStep(int(component.step * scale))
            slider.setPageStep(int(component.step * scale * 10))

        # Style
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: {component.track_height}px;
                background: {component.track_color};
                border-radius: {component.track_height // 2}px;
            }}
            QSlider::sub-page:horizontal {{
                background: {component.track_fill_color};
                border-radius: {component.track_height // 2}px;
            }}
            QSlider::handle:horizontal {{
                width: {component.handle_size}px;
                height: {component.handle_size}px;
                margin: -{(component.handle_size - component.track_height) // 2}px 0;
                background: {component.handle_color};
                border-radius: {component.handle_size // 2}px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {component.handle_hover_color};
            }}
        """)

        layout.addWidget(slider, 1)

        # Value label (optional)
        value_label = None
        if component.show_value:
            value_label = QLabel(component.formatted_value)
            value_label.setStyleSheet("color: #cccccc; min-width: 40px;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(value_label)

        # Store slider reference for value updates
        container._slider = slider
        container._value_label = value_label
        container._scale = scale

        # Connect events
        def on_value_changed(scaled_value):
            old_value = component.value
            component.value = scaled_value / scale
            if value_label:
                value_label.setText(component.formatted_value)
            if component.name:
                self.notify_binding_change(component.name, 'value')
            if EVENT_CHANGE in component.events and self._event_dispatcher:
                binding = component.events[EVENT_CHANGE]
                self._event_dispatcher(
                    EVENT_CHANGE,
                    component,
                    binding,
                    UIEventData.value_change(component.name, component.value, old_value)
                )

        slider.valueChanged.connect(on_value_changed)

        return container

    def _render_radio_button(self, component: 'RadioButton', parent: Optional[QWidget]) -> QRadioButton:
        """Render a standalone RadioButton component."""
        from .components.radio import RadioButton

        radio = QRadioButton(component.text, parent)
        radio.setObjectName(component.name or "radio")
        radio.setChecked(component.checked)

        # Font
        font = QFont()
        font.setPointSize(component.font_size)
        radio.setFont(font)

        # Style
        radio.setStyleSheet(f"""
            QRadioButton {{
                color: {component.text_color};
                spacing: {component.spacing}px;
            }}
            QRadioButton::indicator {{
                width: {component.radio_size}px;
                height: {component.radio_size}px;
                border: 2px solid {component.radio_color};
                border-radius: {component.radio_size // 2}px;
                background-color: transparent;
            }}
            QRadioButton::indicator:checked {{
                background-color: {component.checked_color};
                border-color: {component.checked_color};
            }}
            QRadioButton::indicator:hover {{
                border-color: {component.checked_color};
            }}
        """)

        # Connect events
        def on_toggled(checked):
            component.checked = checked
            if component.name:
                self.notify_binding_change(component.name, 'checked')
            if checked and EVENT_CHANGE in component.events and self._event_dispatcher:
                binding = component.events[EVENT_CHANGE]
                self._event_dispatcher(
                    EVENT_CHANGE,
                    component,
                    binding,
                    UIEventData.value_change(component.name, component.option_value, None)
                )

        radio.toggled.connect(on_toggled)

        return radio

    def _render_radio_group(self, component: 'RadioGroup', parent: Optional[QWidget]) -> QWidget:
        """Render a RadioGroup component."""
        from .components.radio import RadioGroup

        container = QWidget(parent)
        container.setObjectName(component.name or "radio_group")

        # Layout based on orientation
        if component.orientation == "horizontal":
            layout = QHBoxLayout(container)
        else:
            layout = QVBoxLayout(container)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(component.spacing)

        # Button group for mutual exclusion
        button_group = QButtonGroup(container)

        # Create radio buttons for each option
        for i, option in enumerate(component.options):
            radio = QRadioButton(option)
            radio.setChecked(i == component.selected_index)

            # Font
            font = QFont()
            font.setPointSize(component.font_size)
            radio.setFont(font)

            # Style
            radio.setStyleSheet(f"""
                QRadioButton {{
                    color: {component.text_color};
                    spacing: 8px;
                }}
                QRadioButton::indicator {{
                    width: {component.radio_size}px;
                    height: {component.radio_size}px;
                    border: 2px solid {component.radio_color};
                    border-radius: {component.radio_size // 2}px;
                    background-color: transparent;
                }}
                QRadioButton::indicator:checked {{
                    background-color: {component.checked_color};
                    border-color: {component.checked_color};
                }}
                QRadioButton::indicator:hover {{
                    border-color: {component.checked_color};
                }}
            """)

            button_group.addButton(radio, i)
            layout.addWidget(radio)

        # Store reference
        container._button_group = button_group

        # Connect events
        def on_button_clicked(button_id):
            old_value = component.value
            component.selected_index = button_id
            if component.name:
                self.notify_binding_change(component.name, 'value')
                self.notify_binding_change(component.name, 'selected_index')
            if EVENT_CHANGE in component.events and self._event_dispatcher:
                binding = component.events[EVENT_CHANGE]
                self._event_dispatcher(
                    EVENT_CHANGE,
                    component,
                    binding,
                    UIEventData.value_change(component.name, component.value, old_value)
                )

        button_group.idClicked.connect(on_button_clicked)

        layout.addStretch()
        return container

    def _render_webview(self, component: 'WebView', parent: Optional[QWidget]) -> QWidget:
        """Render a WebView component."""
        widget = WebViewWidget(component, self, parent)
        widget.setObjectName(component.name or "webview")
        return widget

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


class ChatHistoryWidget(QFrame):
    """
    Qt widget for rendering ChatHistory component.

    Displays messages in a scrollable container with different
    styling for user messages (right-aligned) and noodling
    messages (left-aligned).
    """

    def __init__(self, component: 'ChatHistory', renderer: QtWidgetRenderer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from .components.chat_history import ChatHistory, MessageRole

        self.component = component
        self.renderer = renderer

        self.setObjectName(component.name or "chat_history")

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {component.background};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {component.background};
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #3a3a3a;
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        main_layout.addWidget(self.scroll_area)

        # Container for messages
        self.messages_container = QWidget()
        self.messages_container.setStyleSheet(f"background-color: {component.background};")
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(8, 8, 8, 8)
        self.messages_layout.setSpacing(component.message_spacing)
        self.messages_layout.addStretch()  # Push messages to top

        self.scroll_area.setWidget(self.messages_container)

        # Background style
        self.setStyleSheet(f"QFrame#{self.objectName()} {{ background-color: {component.background}; }}")

        # Render existing messages
        for msg in component.messages:
            self._add_message_widget(msg)

    def add_message(self, message: 'ChatMessage') -> None:
        """Add a new message widget."""
        self._add_message_widget(message)

        # Auto-scroll to bottom
        if self.component.auto_scroll:
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )

    def _add_message_widget(self, message: 'ChatMessage') -> None:
        """Create and add a message bubble widget."""
        from .components.chat_history import MessageRole

        # Insert before the stretch spacer
        insert_index = self.messages_layout.count() - 1

        # Create message bubble
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(
            self.component.bubble_padding,
            self.component.bubble_padding,
            self.component.bubble_padding,
            self.component.bubble_padding
        )
        bubble_layout.setSpacing(4)

        # Determine styling based on role
        if message.role == MessageRole.USER:
            bg_color = self.component.user_bubble_color
            text_color = self.component.user_text_color
            align = Qt.AlignmentFlag.AlignRight
        elif message.role == MessageRole.NOODLING:
            bg_color = self.component.noodling_bubble_color
            text_color = self.component.noodling_text_color
            align = Qt.AlignmentFlag.AlignLeft
        else:  # SYSTEM
            bg_color = "transparent"
            text_color = self.component.system_color
            align = Qt.AlignmentFlag.AlignCenter

        # Sender name (if shown)
        if self.component.show_sender_names and message.sender_name and message.role != MessageRole.SYSTEM:
            name_label = QLabel(message.sender_name)
            name_font = QFont()
            name_font.setPointSize(self.component.font_size - 2)
            name_font.setBold(True)
            name_label.setFont(name_font)
            name_label.setStyleSheet(f"color: {text_color}; background: transparent;")
            bubble_layout.addWidget(name_label)

        # Message content
        content_label = QLabel(message.content)
        content_label.setWordWrap(True)
        content_font = QFont()
        content_font.setPointSize(self.component.font_size)
        content_label.setFont(content_font)
        content_label.setStyleSheet(f"color: {text_color}; background: transparent;")
        bubble_layout.addWidget(content_label)

        # Timestamp (if shown)
        if self.component.show_timestamps and message.timestamp:
            time_str = message.timestamp.strftime("%H:%M")
            time_label = QLabel(time_str)
            time_font = QFont()
            time_font.setPointSize(self.component.font_size - 3)
            time_label.setFont(time_font)
            time_label.setStyleSheet(f"color: {text_color}; opacity: 0.7; background: transparent;")
            time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            bubble_layout.addWidget(time_label)

        # Style the bubble
        bubble.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: {self.component.bubble_radius}px;
            }}
        """)

        # Set maximum width for bubbles (70% of container)
        bubble.setMaximumWidth(400)

        # Create container for alignment
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        if align == Qt.AlignmentFlag.AlignRight:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        elif align == Qt.AlignmentFlag.AlignLeft:
            container_layout.addWidget(bubble)
            container_layout.addStretch()
        else:  # Center
            container_layout.addStretch()
            container_layout.addWidget(bubble)
            container_layout.addStretch()

        container.setStyleSheet("background: transparent;")

        self.messages_layout.insertWidget(insert_index, container)

    def clear(self) -> None:
        """Clear all messages."""
        # Remove all message widgets (keep the stretch spacer)
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class ChatInputWidget(QFrame):
    """
    Qt widget for rendering ChatInput component.

    A compound widget containing a text input and send button.
    """

    submitted = pyqtSignal(str)  # Emits the message text

    def __init__(self, component: 'ChatInput', renderer: QtWidgetRenderer, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.component = component
        self.renderer = renderer

        self.setObjectName(component.name or "chat_input")

        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            component.padding,
            component.padding,
            component.padding,
            component.padding
        )
        layout.setSpacing(component.spacing)

        # Text input
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(component.placeholder)
        self.input_field.setText(component.value)
        if component.max_length > 0:
            self.input_field.setMaxLength(component.max_length)

        # Input font
        input_font = QFont()
        input_font.setPointSize(component.font_size)
        self.input_field.setFont(input_font)

        # Input style
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                color: {component.text_color};
                background-color: {component.input_background};
                border: 1px solid {component.border_color};
                border-radius: {component.input_border_radius}px;
                padding: 8px 12px;
            }}
            QLineEdit:focus {{
                border-color: #3b82f6;
            }}
            QLineEdit::placeholder {{
                color: {component.placeholder_color};
            }}
        """)

        layout.addWidget(self.input_field, 1)  # Stretch to fill

        # Send button
        self.send_button = QPushButton(component.send_button_text)

        # Button font
        btn_font = QFont()
        btn_font.setPointSize(component.font_size)
        self.send_button.setFont(btn_font)

        # Button style
        hover_bg = hex_to_qcolor(component.button_background)
        h, s, l, a = hover_bg.getHsl()
        l = min(255, l + 25)
        hover_bg.setHsl(h, s, l, a)

        pressed_bg = hex_to_qcolor(component.button_background)
        h, s, l, a = pressed_bg.getHsl()
        l = max(0, l - 15)
        pressed_bg.setHsl(h, s, l, a)

        self.send_button.setStyleSheet(f"""
            QPushButton {{
                color: {component.button_text_color};
                background-color: {component.button_background};
                border: none;
                border-radius: {component.button_border_radius}px;
                padding: 8px 16px;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg.name()};
            }}
            QPushButton:pressed {{
                background-color: {pressed_bg.name()};
            }}
            QPushButton:disabled {{
                background-color: #3a3a3a;
                color: #666666;
            }}
        """)

        layout.addWidget(self.send_button)

        # Container style
        self.setStyleSheet(f"""
            QFrame#{self.objectName()} {{
                background-color: {component.background};
                border-radius: {component.border_radius}px;
            }}
        """)

        # Connect events
        self.input_field.returnPressed.connect(self._on_submit)
        self.send_button.clicked.connect(self._on_submit)
        self.input_field.textChanged.connect(self._on_text_change)

    def _on_submit(self) -> None:
        """Handle submit (Enter key or button click)."""
        text = self.input_field.text().strip()
        if not text:
            return

        # Update component value
        self.component.value = text

        # Clear input if configured
        if self.component.clear_on_submit:
            self.input_field.clear()
            self.component.value = ""

        # Emit signal
        self.submitted.emit(text)

        # Fire event through dispatcher
        if "onSubmit" in self.component.events and self.renderer._event_dispatcher:
            binding = self.component.events["onSubmit"]
            self.renderer._event_dispatcher("onSubmit", self.component, binding)

    def _on_text_change(self, text: str) -> None:
        """Handle text changes."""
        self.component.value = text

        # Notify binding manager
        if self.component.name:
            self.renderer.notify_binding_change(self.component.name, 'value')

        # Fire event through dispatcher
        if "onChange" in self.component.events and self.renderer._event_dispatcher:
            binding = self.component.events["onChange"]
            self.renderer._event_dispatcher("onChange", self.component, binding)


class WebViewWidget(QWidget):
    """Qt widget wrapper for WebView component using QWebEngineView."""

    def __init__(self, component: 'WebView', renderer: 'QtWidgetRenderer', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.component = component
        self.renderer = renderer

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Try to import QWebEngineView
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            from PyQt6.QtCore import QUrl

            self.web_view = QWebEngineView()
            self.web_view.setStyleSheet(f"background-color: {component.background};")

            # Set zoom
            self.web_view.setZoomFactor(component.zoom_factor)

            # Load content
            if component.html:
                base_url = QUrl(component.url) if component.url else QUrl()
                self.web_view.setHtml(component.html, base_url)
            elif component.url:
                self.web_view.setUrl(QUrl(component.url))

            # Connect signals
            self.web_view.loadFinished.connect(self._on_load_finished)
            self.web_view.urlChanged.connect(self._on_url_changed)

            layout.addWidget(self.web_view)

        except ImportError:
            # QWebEngineView not available - show placeholder
            placeholder = QLabel("WebView requires PyQt6-WebEngine\n\npip install PyQt6-WebEngine")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"""
                QLabel {{
                    background-color: {component.background};
                    color: #888888;
                    border: 1px dashed #555555;
                    padding: 20px;
                }}
            """)
            layout.addWidget(placeholder)
            self.web_view = None

    def _on_load_finished(self, ok: bool):
        """Handle page load completion."""
        if ok:
            if "onLoad" in self.component.events and self.renderer._event_dispatcher:
                binding = self.component.events["onLoad"]
                self.renderer._event_dispatcher("onLoad", self.component, binding)
        else:
            if "onError" in self.component.events and self.renderer._event_dispatcher:
                binding = self.component.events["onError"]
                self.renderer._event_dispatcher("onError", self.component, binding)

    def _on_url_changed(self, url):
        """Handle URL changes."""
        self.component.url = url.toString()
        if "onUrlChanged" in self.component.events and self.renderer._event_dispatcher:
            binding = self.component.events["onUrlChanged"]
            self.renderer._event_dispatcher("onUrlChanged", self.component, binding)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
