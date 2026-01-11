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
        from .components.facet_assembly import FacetAssembly
        from .components.led import LED
        from .components.gauge import Gauge
        from .components.qml_widget import QMLWidget
        from .components.seven_segment import SevenSegment
        from .components.level_meter import LevelMeter

        if isinstance(component, LevelMeter):
            widget = self._render_level_meter(component, parent)
        elif isinstance(component, SevenSegment):
            widget = self._render_seven_segment(component, parent)
        elif isinstance(component, QMLWidget):
            widget = self._render_qml_widget(component, parent)
        elif isinstance(component, Gauge):
            widget = self._render_gauge(component, parent)
        elif isinstance(component, LED):
            widget = self._render_led(component, parent)
        elif isinstance(component, FacetAssembly):
            # FacetAssembly is invisible at runtime - just a placeholder widget
            widget = self._render_facet_assembly(component, parent)
        elif isinstance(component, Panel):
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

    def _render_led(self, component: 'LED', parent: Optional[QWidget]) -> QWidget:
        """Render an LED indicator component."""
        widget = LEDWidget(component, self, parent)
        widget.setObjectName(component.name or "led")
        return widget

    def _render_gauge(self, component: 'Gauge', parent: Optional[QWidget]) -> QWidget:
        """Render a Gauge component."""
        widget = GaugeWidget(component, self, parent)
        widget.setObjectName(component.name or "gauge")
        return widget

    def _render_qml_widget(self, component: 'QMLWidget', parent: Optional[QWidget]) -> QWidget:
        """Render a QMLWidget component."""
        widget = QMLWidgetWidget(component, self, parent)
        widget.setObjectName(component.name or "qml_widget")
        return widget

    def _render_seven_segment(self, component: 'SevenSegment', parent: Optional[QWidget]) -> QWidget:
        """Render a SevenSegment display component."""
        widget = SevenSegmentWidget(component, self, parent)
        widget.setObjectName(component.name or "seven_segment")
        return widget

    def _render_level_meter(self, component: 'LevelMeter', parent: Optional[QWidget]) -> QWidget:
        """Render a LevelMeter component."""
        widget = LevelMeterWidget(component, self, parent)
        widget.setObjectName(component.name or "level_meter")
        return widget

    def _render_facet_assembly(self, component: 'FacetAssembly', parent: Optional[QWidget]) -> QWidget:
        """
        Render a FacetAssembly component.

        At runtime, FacetAssembly is invisible - it's pure logic with no visual
        representation. We create a zero-size widget as a placeholder for the
        component system to reference.
        """
        from PyQt6.QtWidgets import QWidget as QW

        widget = QW(parent)
        widget.setObjectName(component.name or "facet_assembly")
        widget.setFixedSize(0, 0)  # Invisible
        widget.hide()

        # Store assembly path for event dispatcher to use
        widget.setProperty("assembly_path", component.assembly_path)
        widget.setProperty("input_bindings", component.get_input_sources())
        widget.setProperty("output_bindings", component.get_output_targets())

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


class LEDWidget(QWidget):
    """
    Qt widget for rendering LED indicator component.

    Renders a physical-style LED with glow effects, supporting
    round or square shapes, optional labels, and blinking animation.
    """

    def __init__(
        self,
        component: 'LED',
        renderer: 'QtWidgetRenderer',
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        from PyQt6.QtGui import QPainter, QBrush, QPen, QRadialGradient
        from PyQt6.QtCore import QTimer, QRectF

        self.component = component
        self.renderer = renderer

        # For blinking
        self._blink_state = True
        self._blink_timer: Optional[QTimer] = None

        # Calculate widget size including label
        self._calculate_size()

        # Set up blinking if configured
        if component.blink_rate > 0:
            self._blink_timer = QTimer(self)
            self._blink_timer.timeout.connect(self._on_blink)
            self._blink_timer.start(int(component.blink_rate * 1000))

        # Enable mouse tracking for click events
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _calculate_size(self):
        """Calculate widget size based on LED size and label."""
        size = self.component.size
        width = size
        height = size

        if self.component.label:
            # Estimate label width (rough calculation)
            label_width = len(self.component.label) * self.component.font_size * 0.6
            label_height = self.component.font_size + 4

            if self.component.label_position in ("left", "right"):
                width = size + self.component.label_spacing + int(label_width)
                height = max(size, int(label_height))
            else:  # top, bottom
                width = max(size, int(label_width))
                height = size + self.component.label_spacing + int(label_height)

        self.setMinimumSize(int(width), int(height))

    def _on_blink(self):
        """Toggle blink state."""
        self._blink_state = not self._blink_state
        self.update()

    def paintEvent(self, event):
        """Custom paint to render LED with glow."""
        from PyQt6.QtGui import QPainter, QBrush, QPen, QRadialGradient, QLinearGradient
        from PyQt6.QtCore import QRectF, QPointF
        from .components.led import LEDShape

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = self.component.size
        half = size / 2

        # Calculate LED position (accounting for label)
        led_x = 0
        led_y = 0

        if self.component.label:
            if self.component.label_position == "left":
                led_x = self.width() - size
            elif self.component.label_position == "top":
                led_y = self.height() - size
            elif self.component.label_position == "bottom":
                led_y = 0
            # right is default: led_x = 0

        # Determine if LED should appear lit (considering blink)
        is_lit = self.component.on and self._blink_state

        # Get colors
        if is_lit:
            led_color = hex_to_qcolor(self.component.color)
        else:
            led_color = hex_to_qcolor(self.component.get_effective_off_color())

        border_color = hex_to_qcolor(self.component.border_color)

        # Draw glow if lit and glow > 0
        if is_lit and self.component.glow > 0:
            glow_size = size * (1 + self.component.glow * 0.5)
            glow_offset = (glow_size - size) / 2

            # Create radial gradient for glow
            center = QPointF(led_x + half, led_y + half)
            glow_gradient = QRadialGradient(center, glow_size / 2)

            glow_color = QColor(led_color)
            glow_color.setAlpha(int(100 * self.component.glow))
            glow_gradient.setColorAt(0.0, glow_color)
            glow_color.setAlpha(0)
            glow_gradient.setColorAt(1.0, glow_color)

            painter.setBrush(QBrush(glow_gradient))
            painter.setPen(Qt.PenStyle.NoPen)

            if self.component.shape == LEDShape.ROUND:
                painter.drawEllipse(
                    QRectF(led_x - glow_offset, led_y - glow_offset, glow_size, glow_size)
                )
            else:
                painter.drawRect(
                    QRectF(led_x - glow_offset, led_y - glow_offset, glow_size, glow_size)
                )

        # Draw LED body with gradient for 3D effect
        body_gradient = QRadialGradient(
            QPointF(led_x + half * 0.7, led_y + half * 0.7),  # Light from top-left
            size * 0.8
        )

        # Lighter highlight version
        highlight = QColor(led_color)
        highlight = highlight.lighter(150 if is_lit else 120)

        body_gradient.setColorAt(0.0, highlight)
        body_gradient.setColorAt(0.5, led_color)
        body_gradient.setColorAt(1.0, led_color.darker(130))

        painter.setBrush(QBrush(body_gradient))
        painter.setPen(QPen(border_color, self.component.border_width))

        led_rect = QRectF(led_x, led_y, size, size)

        if self.component.shape == LEDShape.ROUND:
            painter.drawEllipse(led_rect)
        else:
            corner_radius = size * 0.15  # Slight rounding for square
            painter.drawRoundedRect(led_rect, corner_radius, corner_radius)

        # Draw specular highlight (small bright spot)
        if is_lit:
            highlight_size = size * 0.25
            highlight_offset = size * 0.2
            specular = QRadialGradient(
                QPointF(led_x + highlight_offset + highlight_size / 2,
                        led_y + highlight_offset + highlight_size / 2),
                highlight_size
            )
            specular.setColorAt(0.0, QColor(255, 255, 255, 180))
            specular.setColorAt(1.0, QColor(255, 255, 255, 0))

            painter.setBrush(QBrush(specular))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QRectF(led_x + highlight_offset, led_y + highlight_offset,
                       highlight_size, highlight_size)
            )

        # Draw label if configured
        if self.component.label:
            font = QFont()
            font.setPointSize(self.component.font_size)
            painter.setFont(font)
            painter.setPen(QPen(hex_to_qcolor(self.component.label_color)))

            label_x = 0
            label_y = 0
            align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

            if self.component.label_position == "right":
                label_x = size + self.component.label_spacing
                label_y = 0
                label_rect = QRectF(label_x, label_y, self.width() - label_x, size)
            elif self.component.label_position == "left":
                label_rect = QRectF(0, 0, led_x - self.component.label_spacing, size)
                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            elif self.component.label_position == "top":
                label_rect = QRectF(0, 0, self.width(), led_y - self.component.label_spacing)
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
            else:  # bottom
                label_y = size + self.component.label_spacing
                label_rect = QRectF(0, label_y, self.width(), self.height() - label_y)
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop

            painter.drawText(label_rect, align, self.component.label)

    def mousePressEvent(self, event):
        """Handle mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Fire onClick event
            if EVENT_CLICK in self.component.events and self.renderer._event_dispatcher:
                binding = self.component.events[EVENT_CLICK]
                self.renderer._event_dispatcher(
                    EVENT_CLICK,
                    self.component,
                    binding,
                    UIEventData.click(self.component.name)
                )
        super().mousePressEvent(event)

    def set_on(self, on: bool):
        """Programmatically set LED state and update display."""
        old_value = self.component.on
        self.component.on = on
        self.update()

        # Notify bindings
        if self.component.name:
            self.renderer.notify_binding_change(self.component.name, 'on')
            self.renderer.notify_binding_change(self.component.name, 'value')

        # Fire onChange if state changed
        if old_value != on and EVENT_CHANGE in self.component.events:
            if self.renderer._event_dispatcher:
                binding = self.component.events[EVENT_CHANGE]
                self.renderer._event_dispatcher(
                    EVENT_CHANGE,
                    self.component,
                    binding,
                    UIEventData.value_change(self.component.name, on, old_value)
                )


class GaugeWidget(QWidget):
    """
    Qt widget for rendering Gauge component.

    Renders an analog gauge with arc, needle, tick marks, and value display.
    Mercedes dashboard aesthetic.
    """

    def __init__(
        self,
        component: 'Gauge',
        renderer: 'QtWidgetRenderer',
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.component = component
        self.renderer = renderer

        self.setMinimumSize(component.size, component.size)

    def paintEvent(self, event):
        """Custom paint to render gauge."""
        from PyQt6.QtGui import QPainter, QBrush, QPen, QConicalGradient
        from PyQt6.QtCore import QRectF, QPointF
        import math

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        size = min(self.width(), self.height())
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = (size - self.component.arc_width) / 2 - 4  # Padding

        # Draw background circle
        painter.setBrush(QBrush(hex_to_qcolor(self.component.background_color)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            QRectF(center_x - size/2, center_y - size/2, size, size)
        )

        # Draw arc background
        arc_rect = QRectF(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2
        )

        pen = QPen(hex_to_qcolor(self.component.arc_color), self.component.arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Qt angles are in 1/16th of a degree
        start_16th = int(self.component.start_angle * 16)
        sweep_16th = int(self.component.sweep_angle * 16)
        painter.drawArc(arc_rect, start_16th, sweep_16th)

        # Draw color zones
        for zone in self.component.zones:
            self._draw_zone(painter, arc_rect, zone, radius)

        # Draw tick marks
        self._draw_ticks(painter, center_x, center_y, radius)

        # Draw needle
        self._draw_needle(painter, center_x, center_y, radius)

        # Draw center cap
        cap_radius = radius * self.component.center_radius
        painter.setBrush(QBrush(hex_to_qcolor(self.component.center_color)))
        painter.setPen(QPen(hex_to_qcolor("#444444"), 1))
        painter.drawEllipse(
            QPointF(center_x, center_y),
            cap_radius, cap_radius
        )

        # Draw value and label
        if self.component.show_value or self.component.label:
            self._draw_labels(painter, center_x, center_y, radius)

    def _draw_zone(self, painter, arc_rect, zone, radius):
        """Draw a colored zone on the arc."""
        from PyQt6.QtGui import QPen
        from PyQt6.QtCore import QRectF

        # Calculate zone angles
        range_size = self.component.max_value - self.component.min_value
        if range_size == 0:
            return

        start_norm = (zone.start_value - self.component.min_value) / range_size
        end_norm = (zone.end_value - self.component.min_value) / range_size

        zone_start = self.component.start_angle + start_norm * self.component.sweep_angle
        zone_end = self.component.start_angle + end_norm * self.component.sweep_angle
        zone_sweep = zone_end - zone_start

        pen = QPen(hex_to_qcolor(zone.color), self.component.arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        start_16th = int(zone_start * 16)
        sweep_16th = int(zone_sweep * 16)
        painter.drawArc(arc_rect, start_16th, sweep_16th)

    def _draw_ticks(self, painter, cx, cy, radius):
        """Draw major and minor tick marks."""
        from PyQt6.QtGui import QPen
        from PyQt6.QtCore import QPointF, QLineF
        import math

        tick_color = hex_to_qcolor(self.component.tick_color)
        text_color = hex_to_qcolor(self.component.text_color)

        total_ticks = self.component.major_ticks * (self.component.minor_ticks + 1)
        angle_step = self.component.sweep_angle / total_ticks if total_ticks > 0 else 0

        outer_radius = radius - self.component.arc_width / 2 - 2
        major_inner = outer_radius * (1 - self.component.major_tick_length)
        minor_inner = outer_radius * (1 - self.component.minor_tick_length)

        for i in range(total_ticks + 1):
            angle_deg = self.component.start_angle + i * angle_step
            angle_rad = math.radians(angle_deg)

            is_major = (i % (self.component.minor_ticks + 1)) == 0

            if is_major:
                inner_r = major_inner
                pen = QPen(tick_color, 2)
            else:
                inner_r = minor_inner
                pen = QPen(tick_color, 1)

            painter.setPen(pen)

            # Calculate tick line endpoints
            outer_x = cx + outer_radius * math.cos(angle_rad)
            outer_y = cy - outer_radius * math.sin(angle_rad)
            inner_x = cx + inner_r * math.cos(angle_rad)
            inner_y = cy - inner_r * math.sin(angle_rad)

            painter.drawLine(QPointF(inner_x, inner_y), QPointF(outer_x, outer_y))

            # Draw tick labels for major ticks
            if is_major and self.component.show_tick_labels:
                tick_value = self.component.min_value + (i / total_ticks) * (
                    self.component.max_value - self.component.min_value
                )
                label_text = self.component.value_format.format(tick_value)

                font = QFont()
                font.setPointSize(self.component.tick_label_font_size)
                painter.setFont(font)
                painter.setPen(QPen(text_color))

                # Position label inside the arc
                label_radius = inner_r - 12
                label_x = cx + label_radius * math.cos(angle_rad)
                label_y = cy - label_radius * math.sin(angle_rad)

                # Draw centered on point
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(label_text)
                text_height = fm.height()
                painter.drawText(
                    int(label_x - text_width / 2),
                    int(label_y + text_height / 4),
                    label_text
                )

    def _draw_needle(self, painter, cx, cy, radius):
        """Draw the gauge needle."""
        from PyQt6.QtGui import QPen, QBrush, QPolygonF
        from PyQt6.QtCore import QPointF
        import math

        angle_deg = self.component.get_needle_angle()
        angle_rad = math.radians(angle_deg)

        needle_length = radius * 0.75
        needle_width = self.component.needle_width

        # Calculate needle tip
        tip_x = cx + needle_length * math.cos(angle_rad)
        tip_y = cy - needle_length * math.sin(angle_rad)

        # Calculate needle base (perpendicular to needle direction)
        perp_angle = angle_rad + math.pi / 2
        base_offset = needle_width

        base_x1 = cx + base_offset * math.cos(perp_angle)
        base_y1 = cy - base_offset * math.sin(perp_angle)
        base_x2 = cx - base_offset * math.cos(perp_angle)
        base_y2 = cy + base_offset * math.sin(perp_angle)

        # Draw needle as triangle
        needle_polygon = QPolygonF([
            QPointF(tip_x, tip_y),
            QPointF(base_x1, base_y1),
            QPointF(base_x2, base_y2)
        ])

        painter.setBrush(QBrush(hex_to_qcolor(self.component.needle_color)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(needle_polygon)

    def _draw_labels(self, painter, cx, cy, radius):
        """Draw value display and label."""
        from PyQt6.QtGui import QPen
        from PyQt6.QtCore import QRectF

        text_color = hex_to_qcolor(self.component.text_color)
        painter.setPen(QPen(text_color))

        # Value display (centered below center)
        if self.component.show_value:
            font = QFont()
            font.setPointSize(self.component.font_size)
            font.setBold(True)
            painter.setFont(font)

            value_text = self.component.get_formatted_value()
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(value_text)

            value_y = cy + radius * 0.35
            painter.drawText(
                int(cx - text_width / 2),
                int(value_y),
                value_text
            )

            # Label below value
            if self.component.label:
                font.setPointSize(self.component.label_font_size)
                font.setBold(False)
                painter.setFont(font)

                fm = painter.fontMetrics()
                label_width = fm.horizontalAdvance(self.component.label)
                label_y = value_y + fm.height() + 2

                painter.drawText(
                    int(cx - label_width / 2),
                    int(label_y),
                    self.component.label
                )
        elif self.component.label:
            # Just label, no value
            font = QFont()
            font.setPointSize(self.component.label_font_size)
            painter.setFont(font)

            fm = painter.fontMetrics()
            label_width = fm.horizontalAdvance(self.component.label)
            label_y = cy + radius * 0.35

            painter.drawText(
                int(cx - label_width / 2),
                int(label_y),
                self.component.label
            )

    def set_value(self, value: float):
        """Programmatically set gauge value and update display."""
        old_value = self.component.value
        self.component.set_value(value)
        self.update()

        # Notify bindings
        if self.component.name:
            self.renderer.notify_binding_change(self.component.name, 'value')

        # Fire onChange if value changed
        if old_value != self.component.value and EVENT_CHANGE in self.component.events:
            if self.renderer._event_dispatcher:
                binding = self.component.events[EVENT_CHANGE]
                self.renderer._event_dispatcher(
                    EVENT_CHANGE,
                    self.component,
                    binding,
                    UIEventData.value_change(
                        self.component.name,
                        self.component.value,
                        old_value
                    )
                )


# ─────────────────────────────────────────────────────────────
# Seven-Segment Display
# ─────────────────────────────────────────────────────────────

class SevenSegmentWidget(QWidget):
    """
    Qt widget that renders a seven-segment display.

    Uses QPainter to draw each segment with optional glow effects.
    """

    def __init__(self, component: 'SevenSegment', renderer: 'QtWidgetRenderer', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.component = component
        self.renderer = renderer

        # Reference the component back to this widget
        component._widget = self

        # Enable mouse tracking for potential interaction
        self.setMouseTracking(True)

    def paintEvent(self, event) -> None:
        """Paint the seven-segment display."""
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QLinearGradient
        from PyQt6.QtCore import QRectF, QPointF
        import math

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get component properties
        comp = self.component
        width = self.width()
        height = self.height()

        # Draw background
        bg_color = QColor(comp.background_color)
        painter.fillRect(self.rect(), bg_color)

        # Calculate digit dimensions
        digit_height = comp.digit_height
        digit_width = comp.get_effective_digit_width()
        segment_thickness = comp.get_effective_segment_thickness()
        digit_spacing = comp.digit_spacing

        # Colors
        on_color = QColor(comp.on_color)
        off_color = QColor(comp.get_effective_off_color())

        # Get display string
        display_str = comp.get_display_string()

        # Center the display
        total_width = (digit_width * comp.digit_count) + (digit_spacing * (comp.digit_count - 1))
        start_x = (width - total_width) / 2
        start_y = (height - digit_height) / 2

        # Draw each character
        x_pos = start_x
        for i, char in enumerate(display_str):
            if char == '.':
                # Draw decimal point
                self._draw_decimal_point(
                    painter, x_pos - digit_spacing / 2, start_y + digit_height,
                    segment_thickness, on_color, comp.glow
                )
            else:
                # Draw digit
                pattern = comp.get_segment_pattern(char)
                self._draw_digit(
                    painter, x_pos, start_y, digit_width, digit_height,
                    segment_thickness, pattern, on_color, off_color, comp.glow
                )
                x_pos += digit_width + digit_spacing

        painter.end()

    def _draw_digit(
        self, painter: 'QPainter', x: float, y: float,
        width: float, height: float, thickness: int,
        pattern: int, on_color: 'QColor', off_color: 'QColor', glow: bool
    ) -> None:
        """Draw a single digit with segments."""
        from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath, QRadialGradient
        from PyQt6.QtCore import QRectF, QPointF

        # Segment positions relative to digit bounds
        # Pattern bits: abcdefg (MSB to LSB)
        # a = top horizontal
        # b = top right vertical
        # c = bottom right vertical
        # d = bottom horizontal
        # e = bottom left vertical
        # f = top left vertical
        # g = middle horizontal

        gap = thickness * 0.3  # Gap between segments
        half_h = height / 2

        segments = [
            # (bit_position, is_horizontal, x, y, length)
            (6, True, x + gap, y, width - 2 * gap),                    # a - top
            (5, False, x + width - thickness, y + gap, half_h - gap),   # b - top right
            (4, False, x + width - thickness, y + half_h + gap, half_h - gap - thickness),  # c - bottom right
            (3, True, x + gap, y + height - thickness, width - 2 * gap),  # d - bottom
            (2, False, x, y + half_h + gap, half_h - gap - thickness),  # e - bottom left
            (1, False, x, y + gap, half_h - gap),                       # f - top left
            (0, True, x + gap, y + half_h - thickness / 2, width - 2 * gap),  # g - middle
        ]

        for bit_pos, is_horizontal, sx, sy, length in segments:
            is_on = (pattern >> bit_pos) & 1
            color = on_color if is_on else off_color

            if is_horizontal:
                self._draw_horizontal_segment(painter, sx, sy, length, thickness, color, glow and is_on)
            else:
                self._draw_vertical_segment(painter, sx, sy, thickness, length, color, glow and is_on)

    def _draw_horizontal_segment(
        self, painter: 'QPainter', x: float, y: float,
        width: float, thickness: int, color: 'QColor', glow: bool
    ) -> None:
        """Draw a horizontal segment with pointed ends."""
        from PyQt6.QtGui import QPainterPath, QBrush, QRadialGradient
        from PyQt6.QtCore import QPointF

        half_t = thickness / 2

        # Create hexagonal shape for segment
        path = QPainterPath()
        path.moveTo(x + half_t, y)
        path.lineTo(x + width - half_t, y)
        path.lineTo(x + width, y + half_t)
        path.lineTo(x + width - half_t, y + thickness)
        path.lineTo(x + half_t, y + thickness)
        path.lineTo(x, y + half_t)
        path.closeSubpath()

        # Draw glow first if enabled
        if glow:
            glow_color = QColor(color)
            glow_color.setAlpha(60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_color))

            # Expand path slightly for glow
            glow_path = QPainterPath()
            glow_path.moveTo(x + half_t - 2, y - 2)
            glow_path.lineTo(x + width - half_t + 2, y - 2)
            glow_path.lineTo(x + width + 2, y + half_t)
            glow_path.lineTo(x + width - half_t + 2, y + thickness + 2)
            glow_path.lineTo(x + half_t - 2, y + thickness + 2)
            glow_path.lineTo(x - 2, y + half_t)
            glow_path.closeSubpath()
            painter.drawPath(glow_path)

        # Draw segment
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _draw_vertical_segment(
        self, painter: 'QPainter', x: float, y: float,
        thickness: int, height: float, color: 'QColor', glow: bool
    ) -> None:
        """Draw a vertical segment with pointed ends."""
        from PyQt6.QtGui import QPainterPath, QBrush
        from PyQt6.QtCore import QPointF

        half_t = thickness / 2

        # Create hexagonal shape for segment
        path = QPainterPath()
        path.moveTo(x + half_t, y)
        path.lineTo(x + thickness, y + half_t)
        path.lineTo(x + thickness, y + height - half_t)
        path.lineTo(x + half_t, y + height)
        path.lineTo(x, y + height - half_t)
        path.lineTo(x, y + half_t)
        path.closeSubpath()

        # Draw glow first if enabled
        if glow:
            glow_color = QColor(color)
            glow_color.setAlpha(60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_color))

            glow_path = QPainterPath()
            glow_path.moveTo(x + half_t, y - 2)
            glow_path.lineTo(x + thickness + 2, y + half_t)
            glow_path.lineTo(x + thickness + 2, y + height - half_t)
            glow_path.lineTo(x + half_t, y + height + 2)
            glow_path.lineTo(x - 2, y + height - half_t)
            glow_path.lineTo(x - 2, y + half_t)
            glow_path.closeSubpath()
            painter.drawPath(glow_path)

        # Draw segment
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)

    def _draw_decimal_point(
        self, painter: 'QPainter', x: float, y: float,
        size: int, color: 'QColor', glow: bool
    ) -> None:
        """Draw a decimal point."""
        from PyQt6.QtGui import QBrush, QRadialGradient
        from PyQt6.QtCore import QRectF, QPointF

        radius = size / 2

        # Draw glow
        if glow:
            glow_gradient = QRadialGradient(QPointF(x, y - radius), radius * 2)
            glow_color = QColor(color)
            glow_color.setAlpha(80)
            glow_gradient.setColorAt(0, glow_color)
            glow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_gradient))
            painter.drawEllipse(QRectF(x - radius * 2, y - radius * 3, radius * 4, radius * 4))

        # Draw point
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QRectF(x - radius, y - size, size, size))

    def update_from_component(self) -> None:
        """Update widget from component state."""
        self.update()

    def sizeHint(self):
        """Return preferred size."""
        from PyQt6.QtCore import QSize
        return QSize(self.component.geometry.width, self.component.geometry.height)


# ─────────────────────────────────────────────────────────────
# Level Meter
# ─────────────────────────────────────────────────────────────

class LevelMeterWidget(QWidget):
    """
    Qt widget that renders a VU-meter style level indicator.

    Uses QPainter to draw segmented or continuous bars with
    color zones and optional peak hold indicator.
    """

    def __init__(self, component: 'LevelMeter', renderer: 'QtWidgetRenderer', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.component = component
        self.renderer = renderer

        # Reference the component back to this widget
        component._widget = self

        # Peak decay timer
        self._peak_timer = None
        self._last_peak_time = 0.0

        # Enable mouse tracking
        self.setMouseTracking(True)

        # Set fixed size based on component
        self.setMinimumSize(component.width, component.height)

    def paintEvent(self, event) -> None:
        """Paint the level meter."""
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient
        from PyQt6.QtCore import QRectF, QPointF
        import time

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        comp = self.component
        width = self.width()
        height = self.height()

        # Draw background
        bg_color = QColor(comp.background_color)
        painter.fillRect(self.rect(), bg_color)

        # Draw border
        if comp.border_width > 0:
            border_color = QColor(comp.border_color)
            painter.setPen(QPen(border_color, comp.border_width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(0, 0, width - 1, height - 1)

        # Calculate drawable area (inside border)
        border = comp.border_width
        draw_x = border + 2
        draw_y = border + 2
        draw_width = width - 2 * border - 4
        draw_height = height - 2 * border - 4

        is_vertical = comp.orientation == "vertical"

        if comp.segments > 0:
            # Segmented mode
            self._draw_segmented(
                painter, draw_x, draw_y, draw_width, draw_height,
                is_vertical, comp
            )
        else:
            # Continuous/smooth mode
            self._draw_continuous(
                painter, draw_x, draw_y, draw_width, draw_height,
                is_vertical, comp
            )

        painter.end()

    def _draw_segmented(
        self, painter: 'QPainter', x: float, y: float,
        width: float, height: float, is_vertical: bool, comp: 'LevelMeter'
    ) -> None:
        """Draw segmented bar display."""
        from PyQt6.QtGui import QColor, QBrush, QRadialGradient
        from PyQt6.QtCore import QRectF, QPointF

        segments = comp.segments
        gap = comp.segment_gap
        radius = comp.corner_radius
        inactive_color = QColor(comp.inactive_color)

        # Calculate segment dimensions
        if is_vertical:
            seg_height = (height - gap * (segments - 1)) / segments
            seg_width = width
        else:
            seg_width = (width - gap * (segments - 1)) / segments
            seg_height = height

        # How many segments are lit
        lit_count = comp.get_segment_count_lit()
        peak_segment = comp.get_peak_segment()

        for i in range(segments):
            # Calculate segment position
            if is_vertical:
                # Bottom to top for vertical
                seg_y = y + height - (i + 1) * (seg_height + gap) + gap
                seg_x = x
            else:
                # Left to right for horizontal
                seg_x = x + i * (seg_width + gap)
                seg_y = y

            rect = QRectF(seg_x, seg_y, seg_width, seg_height)

            # Determine if this segment is lit
            is_lit = i < lit_count
            is_peak = (i == peak_segment) and comp.peak_hold

            if is_lit or is_peak:
                # Get color for this level
                segment_value = (i + 0.5) / segments
                color = QColor(comp.get_color_at_value(segment_value))

                # Draw glow if enabled
                if comp.glow > 0:
                    self._draw_segment_glow(painter, rect, color, comp.glow, radius)

                # Draw segment
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(rect, radius, radius)
            else:
                # Draw inactive segment
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(inactive_color))
                painter.drawRoundedRect(rect, radius, radius)

    def _draw_continuous(
        self, painter: 'QPainter', x: float, y: float,
        width: float, height: float, is_vertical: bool, comp: 'LevelMeter'
    ) -> None:
        """Draw continuous/smooth bar display."""
        from PyQt6.QtGui import QColor, QBrush, QLinearGradient
        from PyQt6.QtCore import QRectF, QPointF

        inactive_color = QColor(comp.inactive_color)
        radius = comp.corner_radius
        value = comp.value

        # Draw inactive background bar
        full_rect = QRectF(x, y, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(inactive_color))
        painter.drawRoundedRect(full_rect, radius, radius)

        if value <= 0:
            return

        # Calculate filled area
        if is_vertical:
            fill_height = height * value
            fill_rect = QRectF(x, y + height - fill_height, width, fill_height)
        else:
            fill_width = width * value
            fill_rect = QRectF(x, y, fill_width, height)

        # Create gradient based on zones
        if is_vertical:
            gradient = QLinearGradient(QPointF(x, y + height), QPointF(x, y))
        else:
            gradient = QLinearGradient(QPointF(x, y), QPointF(x + width, y))

        # Add zone colors to gradient
        for zone in comp.zones:
            color = QColor(zone.color)
            gradient.setColorAt(zone.start_value, color)
            gradient.setColorAt(zone.end_value, color)

        # Draw filled portion
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(fill_rect, radius, radius)

        # Draw glow if enabled
        if comp.glow > 0:
            # Use the color at current value for glow
            glow_color = QColor(comp.get_color_at_value(value))
            self._draw_bar_glow(painter, fill_rect, glow_color, comp.glow, is_vertical)

        # Draw peak indicator if enabled
        if comp.peak_hold and comp.peak_value > value:
            peak_color = QColor(comp.get_color_at_value(comp.peak_value))
            self._draw_peak_indicator(
                painter, x, y, width, height, is_vertical,
                comp.peak_value, peak_color, radius
            )

    def _draw_segment_glow(
        self, painter: 'QPainter', rect: 'QRectF', color: 'QColor',
        intensity: float, radius: int
    ) -> None:
        """Draw glow around a segment."""
        from PyQt6.QtGui import QColor, QBrush
        from PyQt6.QtCore import QRectF

        glow_color = QColor(color)
        glow_color.setAlpha(int(60 * intensity))

        # Expand rect for glow
        glow_rect = QRectF(
            rect.x() - 2, rect.y() - 2,
            rect.width() + 4, rect.height() + 4
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawRoundedRect(glow_rect, radius + 1, radius + 1)

    def _draw_bar_glow(
        self, painter: 'QPainter', rect: 'QRectF', color: 'QColor',
        intensity: float, is_vertical: bool
    ) -> None:
        """Draw glow effect on the filled bar."""
        from PyQt6.QtGui import QColor, QBrush, QLinearGradient
        from PyQt6.QtCore import QPointF

        glow_color = QColor(color)
        glow_color.setAlpha(int(80 * intensity))

        # Draw a subtle gradient overlay
        if is_vertical:
            gradient = QLinearGradient(
                QPointF(rect.x(), rect.y()),
                QPointF(rect.x() + rect.width(), rect.y())
            )
        else:
            gradient = QLinearGradient(
                QPointF(rect.x(), rect.y()),
                QPointF(rect.x(), rect.y() + rect.height())
            )

        gradient.setColorAt(0.0, QColor(255, 255, 255, int(40 * intensity)))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 0))
        gradient.setColorAt(1.0, QColor(0, 0, 0, int(30 * intensity)))

        painter.setBrush(QBrush(gradient))
        painter.drawRect(rect)

    def _draw_peak_indicator(
        self, painter: 'QPainter', x: float, y: float,
        width: float, height: float, is_vertical: bool,
        peak_value: float, color: 'QColor', radius: int
    ) -> None:
        """Draw peak hold indicator line."""
        from PyQt6.QtGui import QPen
        from PyQt6.QtCore import QRectF

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        indicator_thickness = 3

        if is_vertical:
            peak_y = y + height * (1 - peak_value)
            rect = QRectF(x, peak_y - indicator_thickness / 2, width, indicator_thickness)
        else:
            peak_x = x + width * peak_value
            rect = QRectF(peak_x - indicator_thickness / 2, y, indicator_thickness, height)

        painter.drawRoundedRect(rect, 1, 1)

    def update_from_component(self) -> None:
        """Update widget from component state."""
        self.update()

    def sizeHint(self):
        """Return preferred size."""
        from PyQt6.QtCore import QSize
        return QSize(self.component.width, self.component.height)


# ─────────────────────────────────────────────────────────────
# QML Widget
# ─────────────────────────────────────────────────────────────

# Check for QML availability
QML_AVAILABLE = False
QQuickWidget = None

try:
    from PyQt6.QtQuickWidgets import QQuickWidget as _QQuickWidget
    from PyQt6.QtCore import QUrl
    QQuickWidget = _QQuickWidget
    QML_AVAILABLE = True
except ImportError:
    pass


class QMLWidgetWidget(QWidget):
    """
    Qt widget that wraps QML content.

    Uses QQuickWidget to render QML files. Falls back to a placeholder
    if QML support is not available.
    """

    def __init__(self, component: 'QMLWidget', renderer: 'QtWidgetRenderer', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.component = component
        self.renderer = renderer

        # Reference the component back to this widget
        component._widget = self

        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if QML_AVAILABLE and component.qml_source:
            self._setup_qml_widget(layout)
        else:
            self._setup_fallback_widget(layout)

    def _setup_qml_widget(self, layout: QVBoxLayout) -> None:
        """Set up the QQuickWidget for QML rendering."""
        from .qml_engine_manager import QMLEngineManager
        from pathlib import Path

        # Resolve QML path
        qml_path = self.component.resolve_qml_path()

        if not qml_path:
            self.component.set_error(f"QML file not found: {self.component.qml_source}")
            self._setup_fallback_widget(layout)
            return

        # Create QQuickWidget
        self._quick_widget = QQuickWidget()
        self._quick_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)

        # Set source from file
        self._quick_widget.setSource(QUrl.fromLocalFile(str(qml_path)))

        # Check for errors
        if self._quick_widget.status() == QQuickWidget.Status.Error:
            errors = self._quick_widget.errors()
            error_messages = [e.toString() for e in errors]
            self.component.set_error("\n".join(error_messages))
            self._setup_fallback_widget(layout)
            return

        # Get root object for property access
        self._root = self._quick_widget.rootObject()

        if self._root:
            # Discover properties
            self._discover_properties()

            # Apply initial properties
            self._apply_properties()

            # Set up property bindings
            self._setup_bindings()

        self.component.clear_error()
        layout.addWidget(self._quick_widget)

    def _discover_properties(self) -> None:
        """Discover QML properties from the root object's metaObject."""
        from .components.qml_widget import qml_type_to_python

        if not self._root:
            return

        meta = self._root.metaObject()
        discovered = {}

        # Skip internal Qt properties
        skip_props = {'objectName', 'parent', 'data', 'resources', 'children',
                      'anchors', 'x', 'y', 'z', 'width', 'height', 'opacity',
                      'enabled', 'visible', 'state', 'states', 'transitions',
                      'focus', 'clip', 'scale', 'rotation', 'transformOrigin',
                      'transform', 'layer', 'smooth', 'antialiasing'}

        for i in range(meta.propertyCount()):
            prop = meta.property(i)
            name = prop.name()

            if name.startswith('_') or name in skip_props:
                continue

            # Get type and default value
            prop_type = qml_type_to_python(prop.typeName())
            default_val = self._root.property(name)

            discovered[name] = {
                'type': prop_type,
                'default': default_val,
                'bindable': prop.isWritable()
            }

        self.component._discovered_properties = discovered

    def _apply_properties(self) -> None:
        """Apply configured properties to the QML root object."""
        if not self._root:
            return

        for name, value in self.component.qml_properties.items():
            try:
                self._root.setProperty(name, value)
            except Exception as e:
                logger.warning(f"Failed to set QML property {name}: {e}")

    def _setup_bindings(self) -> None:
        """Set up channel bindings for QML properties."""
        # Property bindings would connect to the channel system
        # For now, we just track them for the renderer to use
        pass

    def _setup_fallback_widget(self, layout: QVBoxLayout) -> None:
        """Set up fallback display when QML is not available."""
        self._quick_widget = None
        self._root = None

        fallback = QFrame(self)
        fallback.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        fallback.setStyleSheet(f"""
            QFrame {{
                background-color: {self.component.fallback_color};
                border: 1px dashed #666666;
                border-radius: 4px;
            }}
        """)

        fb_layout = QVBoxLayout(fallback)
        fb_layout.setContentsMargins(8, 8, 8, 8)

        # Error or placeholder text
        if self.component.has_error:
            text = f"QML Error:\n{self.component.error_message}"
        elif not QML_AVAILABLE:
            text = "QML not available\n(install PyQt6-QtQuickWidgets)"
        else:
            text = self.component.fallback_text

        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
            }
        """)
        fb_layout.addWidget(label)

        layout.addWidget(fallback)

    def set_qml_property(self, name: str, value: Any) -> None:
        """Set a QML property value at runtime."""
        if self._root:
            try:
                self._root.setProperty(name, value)
                self.component.qml_properties[name] = value
            except Exception as e:
                logger.warning(f"Failed to set QML property {name}: {e}")

    def get_qml_property(self, name: str) -> Any:
        """Get a QML property value."""
        if self._root:
            try:
                return self._root.property(name)
            except Exception:
                pass
        return self.component.qml_properties.get(name)

    def refresh_qml(self) -> None:
        """Reload the QML source (for hot reload)."""
        if self._quick_widget and self.component.qml_source:
            from .qml_engine_manager import QMLEngineManager
            from pathlib import Path

            # Clear cache for this file
            qml_path = self.component.resolve_qml_path()
            if qml_path:
                QMLEngineManager.instance().clear_cache_for(qml_path)

            # Reload
            self._quick_widget.setSource(QUrl.fromLocalFile(str(qml_path)))


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
