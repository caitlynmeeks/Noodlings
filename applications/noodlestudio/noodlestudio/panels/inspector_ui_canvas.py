"""
Inspector UI Canvas Mixin - Property editor for UI components

Provides property editing for components selected in the UI Canvas Editor.
Displays:
- Geometry (x, y, width, height)
- Anchors (left, top, right, bottom)
- Appearance (component-specific: text, color, etc.)
- Events (onClick, onChange, etc.)

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QCheckBox, QComboBox,
    QTextEdit, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal

from noodlestudio.widgets.collapsible_section import CollapsibleSection


class UICanvasInspectorMixin:
    """
    Mixin for InspectorPanel to handle UI Canvas component editing.

    Call load_ui_component() when a component is selected in the canvas.
    """

    # Signal emitted when a UI property changes
    ui_property_changed = pyqtSignal(str, str, object)  # component_name, property, value

    def init_ui_canvas_inspector(self):
        """Initialize UI canvas inspector state. Call from __init__."""
        self._current_ui_component = None
        self._ui_widgets: Dict[str, QWidget] = {}
        self._ui_updating = False

    def load_ui_component(self, component):
        """
        Load a UIComponent for editing.

        Args:
            component: UIComponent instance or None to clear
        """
        self._current_ui_component = component
        self._clear_inspector()

        if component is None:
            self._show_no_selection()
            return

        self._build_ui_component_inspector(component)

    def _clear_inspector(self):
        """Clear current inspector content."""
        # Clear the properties layout
        if hasattr(self, 'properties_layout'):
            while self.properties_layout.count():
                item = self.properties_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        self._ui_widgets.clear()

    def _show_no_selection(self):
        """Show message when nothing is selected."""
        if hasattr(self, 'properties_layout'):
            label = QLabel("Select a component to edit")
            label.setStyleSheet("color: #888888; padding: 20px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.properties_layout.addWidget(label)

    def _build_ui_component_inspector(self, component):
        """Build inspector UI for a component."""
        # Identity section
        identity = self._create_ui_section("Identity")
        self._add_ui_text_field(identity, "Name", "name", component.name)
        self._add_ui_label(identity, "Type", component.component_type)
        self.properties_layout.addWidget(identity)

        # Geometry section
        geometry = self._create_ui_section("Geometry")
        self._add_ui_spin_field(geometry, "X", "geometry.x", component.geometry.x, -9999, 9999)
        self._add_ui_spin_field(geometry, "Y", "geometry.y", component.geometry.y, -9999, 9999)
        self._add_ui_spin_field(geometry, "Width", "geometry.width", component.geometry.width, 1, 9999)
        self._add_ui_spin_field(geometry, "Height", "geometry.height", component.geometry.height, 1, 9999)
        self.properties_layout.addWidget(geometry)

        # Anchors section
        anchors = self._create_ui_section("Anchors")
        self._add_ui_anchor_checkboxes(anchors, component)
        self.properties_layout.addWidget(anchors)

        # Appearance section (component-specific)
        appearance = self._create_appearance_section(component)
        if appearance:
            self.properties_layout.addWidget(appearance)

        # Events section
        events = self._create_events_section(component)
        if events:
            self.properties_layout.addWidget(events)

        # Visibility/Enabled
        state = self._create_ui_section("State")
        self._add_ui_checkbox(state, "Visible", "visible", component.visible)
        self._add_ui_checkbox(state, "Enabled", "enabled", component.enabled)
        self.properties_layout.addWidget(state)

        # Add stretch to push everything up
        self.properties_layout.addStretch()

    def _create_ui_section(self, title: str) -> CollapsibleSection:
        """Create a collapsible section for UI properties."""
        section = CollapsibleSection(title)
        section.setStyleSheet("""
            CollapsibleSection {
                background-color: transparent;
                border: none;
            }
        """)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(12, 8, 12, 8)
        form_layout.setSpacing(6)
        section.set_content_layout(form_layout)

        # Restore saved state if available
        if hasattr(self, '_restore_collapsible_state'):
            self._restore_collapsible_state(section)

        return section

    def _add_ui_label(self, section: CollapsibleSection, label: str, value: str):
        """Add a read-only label field."""
        layout = section.content_layout
        field = QLabel(value)
        field.setStyleSheet("color: #aaaaaa;")
        layout.addRow(label + ":", field)

    def _add_ui_text_field(self, section: CollapsibleSection, label: str, prop: str, value: str):
        """Add a text input field."""
        layout = section.content_layout
        field = QLineEdit(value)
        field.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px;
                color: #cccccc;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
        """)

        def on_changed():
            if not self._ui_updating:
                self._set_ui_property(prop, field.text())

        field.editingFinished.connect(on_changed)
        layout.addRow(label + ":", field)
        self._ui_widgets[prop] = field

    def _add_ui_spin_field(self, section: CollapsibleSection, label: str, prop: str,
                           value: int, min_val: int, max_val: int):
        """Add a spin box field."""
        layout = section.content_layout
        field = QSpinBox()
        field.setRange(min_val, max_val)
        field.setValue(value)
        field.setStyleSheet("""
            QSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px;
                color: #cccccc;
            }
            QSpinBox:focus {
                border-color: #4a9eff;
            }
        """)

        def on_changed(new_value):
            if not self._ui_updating:
                self._set_ui_property(prop, new_value)

        field.valueChanged.connect(on_changed)
        layout.addRow(label + ":", field)
        self._ui_widgets[prop] = field

    def _add_ui_checkbox(self, section: CollapsibleSection, label: str, prop: str, value: bool):
        """Add a checkbox field."""
        layout = section.content_layout
        field = QCheckBox()
        field.setChecked(value)
        field.setStyleSheet("color: #cccccc;")

        def on_changed(state):
            if not self._ui_updating:
                self._set_ui_property(prop, state == Qt.CheckState.Checked.value)

        field.stateChanged.connect(on_changed)
        layout.addRow(label + ":", field)
        self._ui_widgets[prop] = field

    def _add_ui_anchor_checkboxes(self, section: CollapsibleSection, component):
        """Add anchor checkboxes in a 2x2 grid-like layout."""
        layout = section.content_layout

        # Top row: Left - Top
        row1 = QHBoxLayout()
        left_cb = QCheckBox("Left")
        left_cb.setChecked(component.anchors.left)
        left_cb.setStyleSheet("color: #cccccc;")
        left_cb.stateChanged.connect(lambda s: self._set_anchor("left", s == Qt.CheckState.Checked.value))

        top_cb = QCheckBox("Top")
        top_cb.setChecked(component.anchors.top)
        top_cb.setStyleSheet("color: #cccccc;")
        top_cb.stateChanged.connect(lambda s: self._set_anchor("top", s == Qt.CheckState.Checked.value))

        row1.addWidget(left_cb)
        row1.addWidget(top_cb)
        row1.addStretch()

        # Bottom row: Right - Bottom
        row2 = QHBoxLayout()
        right_cb = QCheckBox("Right")
        right_cb.setChecked(component.anchors.right)
        right_cb.setStyleSheet("color: #cccccc;")
        right_cb.stateChanged.connect(lambda s: self._set_anchor("right", s == Qt.CheckState.Checked.value))

        bottom_cb = QCheckBox("Bottom")
        bottom_cb.setChecked(component.anchors.bottom)
        bottom_cb.setStyleSheet("color: #cccccc;")
        bottom_cb.stateChanged.connect(lambda s: self._set_anchor("bottom", s == Qt.CheckState.Checked.value))

        row2.addWidget(right_cb)
        row2.addWidget(bottom_cb)
        row2.addStretch()

        # Add to layout
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addLayout(row1)
        container_layout.addLayout(row2)
        layout.addRow("Anchors:", container)

        self._ui_widgets["anchors.left"] = left_cb
        self._ui_widgets["anchors.top"] = top_cb
        self._ui_widgets["anchors.right"] = right_cb
        self._ui_widgets["anchors.bottom"] = bottom_cb

    def _set_anchor(self, edge: str, value: bool):
        """Set an anchor edge value."""
        if not self._current_ui_component or self._ui_updating:
            return

        setattr(self._current_ui_component.anchors, edge, value)
        self._emit_change("anchors." + edge, value)

    def _create_appearance_section(self, component) -> Optional[CollapsibleSection]:
        """Create appearance section based on component type."""
        section = self._create_ui_section("Appearance")
        layout = section.content_layout
        has_fields = False

        # Common appearance properties
        if hasattr(component, 'text'):
            self._add_ui_text_field(section, "Text", "text", getattr(component, 'text', ''))
            has_fields = True

        if hasattr(component, 'text_color'):
            self._add_ui_text_field(section, "Text Color", "text_color",
                                    getattr(component, 'text_color', '#ffffff'))
            has_fields = True

        if hasattr(component, 'background'):
            self._add_ui_text_field(section, "Background", "background",
                                    getattr(component, 'background', '#2a2a2a'))
            has_fields = True

        if hasattr(component, 'font_size'):
            self._add_ui_spin_field(section, "Font Size", "font_size",
                                    getattr(component, 'font_size', 14), 8, 72)
            has_fields = True

        if hasattr(component, 'placeholder'):
            self._add_ui_text_field(section, "Placeholder", "placeholder",
                                    getattr(component, 'placeholder', ''))
            has_fields = True

        return section if has_fields else None

    def _create_events_section(self, component) -> Optional[CollapsibleSection]:
        """Create events section showing bound events."""
        if not component.events:
            return None

        section = self._create_ui_section("Events")
        layout = section.content_layout

        for event_name, binding in component.events.items():
            # Show event name and action
            event_label = QLabel(f"{event_name}: {binding.action}")
            event_label.setStyleSheet("color: #aaaaaa; font-family: monospace;")
            layout.addRow("", event_label)

            # Show target if present
            if binding.target:
                target_label = QLabel(f"  target: {binding.target}")
                target_label.setStyleSheet("color: #888888; font-family: monospace; font-size: 11px;")
                layout.addRow("", target_label)

        return section

    def _set_ui_property(self, prop: str, value: Any):
        """Set a property on the current UI component."""
        if not self._current_ui_component or self._ui_updating:
            return

        # Handle nested properties (e.g., "geometry.x")
        parts = prop.split(".")
        obj = self._current_ui_component

        for part in parts[:-1]:
            obj = getattr(obj, part)

        setattr(obj, parts[-1], value)
        self._emit_change(prop, value)

    def _emit_change(self, prop: str, value: Any):
        """Emit property change signal."""
        if self._current_ui_component:
            # Emit signal if the mixin has it
            if hasattr(self, 'ui_property_changed'):
                self.ui_property_changed.emit(
                    self._current_ui_component.name,
                    prop,
                    value
                )

            # Notify canvas of change
            main_window = self.window() if hasattr(self, 'window') else None
            if main_window and hasattr(main_window, 'ui_canvas_editor'):
                main_window.ui_canvas_editor.view.canvas_modified.emit()

    def update_ui_component_field(self, prop: str, value: Any):
        """
        Update a field in the inspector without triggering change signals.

        Used when the canvas updates component position/size via drag.
        """
        if prop not in self._ui_widgets:
            return

        self._ui_updating = True
        try:
            widget = self._ui_widgets[prop]
            if isinstance(widget, QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
        finally:
            self._ui_updating = False
