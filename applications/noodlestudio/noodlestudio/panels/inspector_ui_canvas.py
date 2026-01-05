"""
Inspector UI Canvas Mixin - Property editor for UI components

Provides property editing for components selected in the UI Canvas Editor.
Displays:
- Geometry (x, y, width, height)
- Anchors (left, top, right, bottom)
- Appearance (component-specific: text, color, etc.)
- Events (onClick, onChange, etc.) - Interactive Delphi-style Events tab

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QCheckBox, QComboBox,
    QTextEdit, QPushButton, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal

from noodlestudio.widgets.collapsible_section import CollapsibleSection
from noodlestudio.widgets.event_binding_widget import EventBindingWidget, COMMON_EVENTS
from noodlestudio.widgets.color_picker_widget import ColorFieldWidget
from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog
from noodlestudio.runtime.ui.component import EventBinding


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

        # Events section (always shown - users can add events)
        events = self._create_events_section(component)
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
        form_layout.setContentsMargins(8, 4, 8, 4)
        form_layout.setSpacing(4)
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
                border-radius: 2px;
                padding: 2px 4px;
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
                border-radius: 2px;
                padding: 2px 4px;
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

    def _add_ui_color_field(self, section: CollapsibleSection, label: str, prop: str, value: str):
        """Add a color picker field."""
        layout = section.content_layout
        field = ColorFieldWidget()
        field.setColor(value)

        def on_changed(color):
            if not self._ui_updating:
                self._set_ui_property(prop, color.name())

        field.colorChanged.connect(on_changed)
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
            self._add_ui_color_field(section, "Text Color", "text_color",
                                     getattr(component, 'text_color', '#ffffff'))
            has_fields = True

        if hasattr(component, 'background'):
            self._add_ui_color_field(section, "Background", "background",
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

    def _create_events_section(self, component) -> CollapsibleSection:
        """
        Create interactive events section with Delphi-style event wiring.

        Shows existing event bindings and allows adding new ones.
        """
        section = CollapsibleSection("Events")
        section.setStyleSheet("""
            CollapsibleSection {
                background-color: transparent;
                border: none;
            }
        """)

        # Create VBoxLayout for event widgets
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 4)
        content_layout.setSpacing(4)

        # Store event binding widgets
        self._event_binding_widgets: Dict[str, EventBindingWidget] = {}

        # Get available components and noodlings for dropdowns
        available_components = self._get_available_components()
        available_noodlings = self._get_available_noodlings()

        # Create widget for each existing event
        for event_name, binding in component.events.items():
            binding_data = {
                "action": binding.action,
                "target": binding.target,
                "message_source": binding.message_source,
                "chat_history": binding.chat_history,
                "script": binding.script,
                "script_file": binding.script_file,
            }

            widget = EventBindingWidget(
                event_name=event_name,
                binding_data=binding_data,
                available_components=available_components,
                available_noodlings=available_noodlings,
            )

            # Connect signals
            widget.changed.connect(lambda en=event_name, w=widget: self._on_event_binding_changed(en, w))
            widget.delete_requested.connect(lambda en=event_name: self._on_delete_event(en))
            widget.edit_script_requested.connect(lambda script, en=event_name: self._on_edit_script(en, script))

            content_layout.addWidget(widget)
            self._event_binding_widgets[event_name] = widget

        # Add Event button
        add_btn = QPushButton("+ Add Event")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        add_btn.clicked.connect(lambda: self._show_add_event_menu(add_btn))
        content_layout.addWidget(add_btn)

        content_layout.addStretch()
        section.set_content_layout(content_layout)

        # Restore saved state if available
        if hasattr(self, '_restore_collapsible_state'):
            self._restore_collapsible_state(section)

        return section

    def _get_available_components(self) -> List[str]:
        """Get list of component names in current UI canvas."""
        components = []
        main_window = self.window() if hasattr(self, 'window') else None
        if main_window and hasattr(main_window, 'ui_canvas_editor'):
            canvas = main_window.ui_canvas_editor
            if hasattr(canvas, 'view') and hasattr(canvas.view, '_root_component'):
                root = canvas.view._root_component
                if root:
                    self._collect_component_names(root, components)
        return components

    def _collect_component_names(self, component, names: List[str]):
        """Recursively collect component names."""
        if component.name:
            names.append(component.name)
        for child in component.children:
            self._collect_component_names(child, names)

    def _get_available_noodlings(self) -> List[str]:
        """Get list of available noodling names."""
        noodlings = []
        main_window = self.window() if hasattr(self, 'window') else None
        if main_window and hasattr(main_window, 'project_manager'):
            pm = main_window.project_manager
            if hasattr(pm, 'list_noodlings'):
                noodlings = pm.list_noodlings()
        # Fallback defaults
        if not noodlings:
            noodlings = ["red", "blue", "green"]
        return noodlings

    def _show_add_event_menu(self, button: QPushButton):
        """Show menu to select event type to add."""
        if not self._current_ui_component:
            return

        menu = QMenu(button)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
        """)

        # Get already-bound events
        existing_events = set(self._current_ui_component.events.keys())

        # Add common events (not already bound)
        for event_name in COMMON_EVENTS:
            if event_name not in existing_events:
                action = menu.addAction(event_name)
                action.triggered.connect(lambda checked, en=event_name: self._add_event(en))

        # Separator and "More..." option
        if len(COMMON_EVENTS) > 0:
            menu.addSeparator()

        more_menu = menu.addMenu("More Events...")
        more_events = [
            "onMouseDown", "onMouseUp", "onMouseMove", "onMouseWheel",
            "onContextMenu", "onKeyPress",
            "onDragStart", "onDrag", "onDrop",
            "onCreate", "onDestroy", "onShow", "onHide", "onResize",
        ]
        for event_name in more_events:
            if event_name not in existing_events and event_name not in COMMON_EVENTS:
                action = more_menu.addAction(event_name)
                action.triggered.connect(lambda checked, en=event_name: self._add_event(en))

        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _add_event(self, event_name: str):
        """Add a new event binding."""
        if not self._current_ui_component:
            return

        # Create default binding
        binding = EventBinding(action="send_to_noodling")
        self._current_ui_component.events[event_name] = binding

        # Reload inspector to show new event
        self.load_ui_component(self._current_ui_component)

        # Notify canvas
        self._emit_change(f"events.{event_name}", binding)

    def _on_event_binding_changed(self, event_name: str, widget: EventBindingWidget):
        """Handle event binding widget changes."""
        if not self._current_ui_component or self._ui_updating:
            return

        binding_data = widget.get_binding_data()

        # Update the component's event binding
        binding = EventBinding(
            action=binding_data.get("action", "send_to_noodling"),
            target=binding_data.get("target"),
            message_source=binding_data.get("message_source"),
            chat_history=binding_data.get("chat_history"),
            script=binding_data.get("script"),
            script_file=binding_data.get("script_file"),
        )
        self._current_ui_component.events[event_name] = binding

        # Notify canvas of change
        self._emit_change(f"events.{event_name}", binding_data)

    def _on_delete_event(self, event_name: str):
        """Delete an event binding."""
        if not self._current_ui_component:
            return

        if event_name in self._current_ui_component.events:
            del self._current_ui_component.events[event_name]

        # Reload inspector
        self.load_ui_component(self._current_ui_component)

        # Notify canvas
        self._emit_change(f"events.{event_name}", None)

    def _on_edit_script(self, event_name: str, current_script: str):
        """Open script editor dialog."""
        if not self._current_ui_component:
            return

        # Show script editor dialog
        new_script = ScriptEditorDialog.edit_script(
            script_content=current_script,
            event_name=event_name,
            parent=self if isinstance(self, QWidget) else None
        )

        if new_script is not None:
            # Update the widget
            if event_name in self._event_binding_widgets:
                self._event_binding_widgets[event_name].set_script_content(new_script)

            # Update the component
            if event_name in self._current_ui_component.events:
                self._current_ui_component.events[event_name].script = new_script
                self._emit_change(f"events.{event_name}.script", new_script)

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
