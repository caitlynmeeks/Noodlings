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
#   Inspector Base - Common UI utilities and property binding
#
#   Contains: - UI helper methods (create_property_group, add...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_base
# PURPOSE:  Inspector Base
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ClickableTextEdit, InspectorBaseMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QScrollArea,
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, QComboBox,
    QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize, QEvent
from PyQt6.QtGui import QFont, QFontMetrics
import os

from noodlestudio.widgets.collapsible_section import CollapsibleSection
from ..panels.floating_text_editor import FloatingTextEditor
from ..core.property_binding import PropertyBindingManager, PropertyMeta, property_registry
from ..core.undo_manager import UndoManager


class ClickableTextEdit(QTextEdit):
    """QTextEdit that opens floating editor on Cmd+Click."""

    def __init__(self, field_name: str, on_apply_callback, parent=None):
        super().__init__(parent)
        self.field_name = field_name
        self.on_apply_callback = on_apply_callback
        self.floating_editor = None

    def mousePressEvent(self, event):
        """Detect Cmd+Click to open floating editor."""
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.MetaModifier or modifiers & Qt.KeyboardModifier.ControlModifier:
                self.open_floating_editor()
                return
        super().mousePressEvent(event)

    def open_floating_editor(self):
        """Open floating text editor for this field."""
        if self.floating_editor and self.floating_editor.isVisible():
            self.floating_editor.raise_()
            self.floating_editor.activateWindow()
            return

        self.floating_editor = FloatingTextEditor(
            field_name=self.field_name,
            field_key=self.field_name,
            initial_value=self.toPlainText(),
            read_only=self.isReadOnly(),
            parent=self.window()
        )

        def on_text_applied(key, value):
            self.setPlainText(value)
            if self.on_apply_callback:
                self.on_apply_callback(value)

        self.floating_editor.textApplied.connect(on_text_applied)
        self.floating_editor.show()


class InspectorBaseMixin:
    """
    Base mixin providing common inspector utilities.

    Provides:
    - Property group creation
    - Field creation (text, dropdown, slider, etc.)
    - Collapsible section management
    - Property binding support
    """

    def init_base_inspector(self):
        """Initialize base inspector state. Call from __init__."""
        self.property_fields = {}
        self.component_widgets = {}
        self.collapsible_states = {}
        self.is_loading = False
        self._bound_widgets = {}

    # ========== PROPERTY GROUP CREATION ==========

    def create_property_group(self, title: str) -> CollapsibleSection:
        """Create a collapsible property group with QFormLayout."""
        section = CollapsibleSection(title)
        section.setStyleSheet("""
            CollapsibleSection {
                background-color: transparent;
                border: none;
            }
        """)

        # Set up QFormLayout for property rows
        form_layout = QFormLayout()
        form_layout.setContentsMargins(12, 8, 12, 8)
        form_layout.setSpacing(6)
        section.set_content_layout(form_layout)

        # Restore saved state
        self._restore_collapsible_state(section)
        section.toggled.connect(lambda exp, t=title: self._on_collapsible_toggled(t, exp))

        return section

    def _save_collapsible_states(self):
        """Save collapsible section states."""
        try:
            main_window = self.window()
            if hasattr(main_window, 'settings_manager'):
                for title, expanded in self.collapsible_states.items():
                    key = f"inspector_section_{title}"
                    main_window.settings_manager.set_setting(key, expanded)
        except Exception as e:
            print(f"[Inspector] Error saving collapsible states: {e}")

    def _restore_collapsible_state(self, section: CollapsibleSection):
        """Restore collapsible section state from settings."""
        try:
            title = section.title_text  # CollapsibleSection uses title_text
            main_window = self.window()
            if hasattr(main_window, 'settings_manager'):
                key = f"inspector_section_{title}"
                saved = main_window.settings_manager.get_setting(key, None)
                if saved is not None:
                    section.set_expanded(saved)  # Use set_expanded not setExpanded
                    self.collapsible_states[title] = saved
        except Exception as e:
            pass  # Silently ignore - window may not be ready

    def _on_collapsible_toggled(self, title: str, expanded: bool):
        """Handle collapsible section toggle."""
        self.collapsible_states[title] = expanded
        self._save_collapsible_states()

    # ========== FIELD CREATION HELPERS ==========

    def add_text_field(self, group: QGroupBox, label: str, value: str, read_only: bool = False):
        """Add a text input field to a property group."""
        field = QLineEdit(value)
        field.setReadOnly(read_only)
        field.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;"
            if not read_only else
            "background-color: #2A2A2A; color: #888888; padding: 4px;"
        )

        # Auto-save on editing finished (Enter key or focus out) for editable fields
        if not read_only:
            def on_editing_finished():
                if hasattr(self, 'save_changes') and not getattr(self, 'is_loading', False):
                    self.save_changes()
            field.editingFinished.connect(on_editing_finished)

        group.content.layout().addRow(f"{label}:", field)
        return field

    def add_text_area(self, group: QGroupBox, label: str, value: str):
        """Add a multi-line text area to a property group."""
        text_edit = QTextEdit(value)
        text_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        text_edit.setMaximumHeight(100)
        text_edit.setTabChangesFocus(True)

        # Auto-save on focus out
        def safe_focus_out(event):
            if hasattr(self, 'save_changes') and not self.is_loading:
                QTimer.singleShot(100, self.save_changes)
            QTextEdit.focusOutEvent(text_edit, event)
        text_edit.focusOutEvent = safe_focus_out

        group.content.layout().addRow(f"{label}:", text_edit)
        return text_edit

    def add_vector3_field(self, group: QGroupBox, label: str, values: list, read_only: bool = True):
        """Add a 3-component vector field (x, y, z)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        fields = []
        for i, (axis, val) in enumerate(zip(['X', 'Y', 'Z'], values)):
            field = QDoubleSpinBox()
            field.setRange(-99999, 99999)
            field.setDecimals(2)
            field.setValue(float(val) if val is not None else 0.0)
            field.setReadOnly(read_only)
            field.setPrefix(f"{axis}: ")
            field.setStyleSheet(
                "background-color: #1E1E1E; color: #D2D2D2; padding: 2px;"
                if not read_only else
                "background-color: #2A2A2A; color: #888888; padding: 2px;"
            )
            field.setFixedWidth(80)
            layout.addWidget(field)
            fields.append(field)

        layout.addStretch()
        group.content.layout().addRow(f"{label}:", container)
        return fields

    def add_dropdown_field(self, group: QGroupBox, label: str, value: str, options: list, on_change=None):
        """Add a dropdown/combo box field."""
        combo = QComboBox()
        combo.addItems(options)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                padding: 4px;
                border: 1px solid #3A3A3A;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                color: #D2D2D2;
                selection-background-color: #3A3A3A;
            }
        """)

        # Set current value
        if value in options:
            combo.setCurrentText(value)

        if on_change:
            combo.currentTextChanged.connect(on_change)

        group.content.layout().addRow(f"{label}:", combo)
        return combo

    def add_checkbox_field(self, group: QGroupBox, label: str, checked: bool, on_change=None):
        """Add a checkbox field to a property group.

        Args:
            group: CollapsibleSection to add to
            label: Label text (e.g. "Ensemble Active")
            checked: Initial checked state
            on_change: Optional callback(bool) on state change

        Returns:
            QCheckBox widget
        """
        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        checkbox.setStyleSheet("QCheckBox { color: #D2D2D2; }")

        if on_change:
            checkbox.stateChanged.connect(
                lambda state: on_change(state == Qt.CheckState.Checked.value)
            )

        group.content.layout().addRow(f"{label}:", checkbox)
        return checkbox

    def add_slider_field(self, group: QGroupBox, label: str, value: float, min_val: float, max_val: float):
        """Add a slider with value display."""
        from PyQt6.QtWidgets import QSlider

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(value * 100))

        value_label = QLabel(f"{value:.2f}")
        value_label.setFixedWidth(40)
        value_label.setStyleSheet("color: #888;")

        slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100:.2f}"))

        layout.addWidget(slider)
        layout.addWidget(value_label)

        group.content.layout().addRow(f"{label}:", container)
        return slider

    # ========== PROPERTY BINDING SUPPORT ==========

    def create_bound_spinbox(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QSpinBox:
        """Create a spinbox bound to an object property."""
        spin = QSpinBox()
        spin.setRange(
            int(meta.minimum) if meta.minimum is not None else -999999,
            int(meta.maximum) if meta.maximum is not None else 999999
        )
        spin.setValue(int(getattr(obj, meta.name, 0)))
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")

        def on_change(val):
            if self.is_loading:
                return
            old_val = getattr(obj, meta.name, 0)
            if old_val != val:
                setattr(obj, meta.name, val)
                self._push_generic_property_command(obj, meta.name, old_val, val)

        spin.valueChanged.connect(on_change)
        layout.addRow(f"{meta.display_name}:", spin)
        self._bound_widgets[meta.name] = spin
        return spin

    def create_bound_double_spinbox(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QDoubleSpinBox:
        """Create a double spinbox bound to an object property."""
        spin = QDoubleSpinBox()
        spin.setRange(
            meta.minimum if meta.minimum is not None else -999999.0,
            meta.maximum if meta.maximum is not None else 999999.0
        )
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setValue(float(getattr(obj, meta.name, 0.0)))
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")

        def on_change(val):
            if self.is_loading:
                return
            old_val = getattr(obj, meta.name, 0.0)
            if abs(old_val - val) > 0.0001:
                setattr(obj, meta.name, val)
                self._push_generic_property_command(obj, meta.name, old_val, val)

        spin.valueChanged.connect(on_change)
        layout.addRow(f"{meta.display_name}:", spin)
        self._bound_widgets[meta.name] = spin
        return spin

    def create_bound_combobox(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QComboBox:
        """Create a combobox bound to an object property."""
        combo = QComboBox()
        if meta.choices:
            combo.addItems(meta.choices)
        combo.setCurrentText(str(getattr(obj, meta.name, '')))
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                padding: 4px;
            }
        """)

        def on_change(text):
            if self.is_loading:
                return
            old_val = getattr(obj, meta.name, '')
            if old_val != text:
                setattr(obj, meta.name, text)
                self._push_generic_property_command(obj, meta.name, old_val, text)

        combo.currentTextChanged.connect(on_change)
        layout.addRow(f"{meta.display_name}:", combo)
        self._bound_widgets[meta.name] = combo
        return combo

    def create_bound_textedit(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QTextEdit:
        """Create a text edit bound to an object property."""
        text_edit = QTextEdit()
        text_edit.setPlainText(str(getattr(obj, meta.name, '')))
        text_edit.setMaximumHeight(100)
        text_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")

        def on_focus_out():
            if self.is_loading:
                return
            new_val = text_edit.toPlainText()
            old_val = getattr(obj, meta.name, '')
            if old_val != new_val:
                setattr(obj, meta.name, new_val)
                self._push_generic_property_command(obj, meta.name, old_val, new_val)

        text_edit.focusOutEvent = lambda e: (on_focus_out(), QTextEdit.focusOutEvent(text_edit, e))
        layout.addRow(f"{meta.display_name}:", text_edit)
        self._bound_widgets[meta.name] = text_edit
        return text_edit

    def create_bound_checkbox(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QCheckBox:
        """Create a checkbox bound to an object property."""
        checkbox = QCheckBox()
        checkbox.setChecked(bool(getattr(obj, meta.name, False)))
        checkbox.setStyleSheet("color: #D2D2D2;")

        def on_change(state):
            if self.is_loading:
                return
            new_val = state == Qt.CheckState.Checked.value
            old_val = getattr(obj, meta.name, False)
            if old_val != new_val:
                setattr(obj, meta.name, new_val)
                self._push_generic_property_command(obj, meta.name, old_val, new_val)

        checkbox.stateChanged.connect(on_change)
        layout.addRow(f"{meta.display_name}:", checkbox)
        self._bound_widgets[meta.name] = checkbox
        return checkbox

    def create_bound_lineedit(
        self, obj, meta: PropertyMeta, layout: QFormLayout
    ) -> QLineEdit:
        """Create a line edit bound to an object property."""
        line_edit = QLineEdit()
        line_edit.setText(str(getattr(obj, meta.name, '')))
        line_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")

        def on_editing_finished():
            if self.is_loading:
                return
            new_val = line_edit.text()
            old_val = getattr(obj, meta.name, '')
            if old_val != new_val:
                setattr(obj, meta.name, new_val)
                self._push_generic_property_command(obj, meta.name, old_val, new_val)

        line_edit.editingFinished.connect(on_editing_finished)
        layout.addRow(f"{meta.display_name}:", line_edit)
        self._bound_widgets[meta.name] = line_edit
        return line_edit

    def create_widget_for_property(self, obj, meta: PropertyMeta):
        """Create appropriate widget based on property metadata."""
        # Returns widget - caller adds to layout
        if meta.prop_type is int:
            return self.create_bound_spinbox(obj, meta, QFormLayout())
        elif meta.prop_type is float:
            return self.create_bound_double_spinbox(obj, meta, QFormLayout())
        elif meta.prop_type is bool:
            return self.create_bound_checkbox(obj, meta, QFormLayout())
        elif meta.prop_type is str and meta.choices:
            return self.create_bound_combobox(obj, meta, QFormLayout())
        elif meta.prop_type is str:
            return self.create_bound_lineedit(obj, meta, QFormLayout())
        elif meta.multiline:
            return self.create_bound_textedit(obj, meta, QFormLayout())
        return None

    def build_inspector_for_object(self, obj, layout, include_base: bool = False):
        """Build inspector UI from object's registered properties."""
        class_name = obj.__class__.__name__
        properties = property_registry.get(class_name, [])

        if not properties:
            return

        for meta in properties:
            if meta.hidden:
                continue
            widget = self.create_widget_for_property(obj, meta)
            if widget:
                layout.addRow(f"{meta.display_name}:", widget)

    def _push_generic_property_command(
        self, obj, property_name: str, old_value, new_value
    ):
        """Push a generic property change command to undo stack."""
        from ..core.commands.base_command import SetPropertyCommand

        cmd = SetPropertyCommand(
            target=obj,
            property_name=property_name,
            old_value=old_value,
            new_value=new_value
        )
        UndoManager.instance().push(cmd)

    # ========== EVENT HANDLING ==========

    def eventFilter(self, obj, event):
        """Handle events for child widgets."""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                if isinstance(obj, QTextEdit):
                    # Single Enter in text edit - save and move focus
                    if hasattr(self, 'save_changes'):
                        self.save_changes()
                    self.setFocus()
                    return True
        return super().eventFilter(obj, event)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
