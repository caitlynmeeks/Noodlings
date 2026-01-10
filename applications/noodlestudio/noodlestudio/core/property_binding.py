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
#   Property Binding System - Automatic undo for Inspector properties
#
#   This module provides a binding system that connects Qt wi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.property_binding
# PURPOSE:  Property Binding
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PropertyBinding, PropertyBindingManager, PropertyMeta, PropertyRegistry, editable_properties()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Any, Optional, Callable, Dict, Type, TYPE_CHECKING
from PyQt6.QtWidgets import (
    QWidget, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QTextEdit, QPlainTextEdit, QCheckBox, QSlider
)
from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from ..panels.inspector_panel import InspectorPanel


class PropertyBinding(QObject):
    """
    Binds a Qt widget to an object property with automatic undo support.

    Automatically detects widget type and connects the appropriate signal.
    When the widget value changes, creates and pushes an undo command.

    Supports command merging for continuous operations (typing, slider drag).
    """

    # Signal emitted when property value changes (for external listeners)
    value_changed = pyqtSignal(object, str, object)  # (obj, prop_name, new_value)

    def __init__(
        self,
        inspector: 'InspectorPanel',
        widget: QWidget,
        obj: Any,
        property_name: str,
        display_name: Optional[str] = None,
        transform_to_model: Optional[Callable] = None,
        transform_from_model: Optional[Callable] = None
    ):
        """
        Create a property binding.

        Args:
            inspector: Reference to InspectorPanel (for pushing commands)
            widget: Qt widget to bind
            obj: Object containing the property
            property_name: Name of property on obj
            display_name: Human-readable name for undo text (defaults to property_name)
            transform_to_model: Optional function to transform widget value before setting
            transform_from_model: Optional function to transform model value for widget
        """
        super().__init__()

        self.inspector = inspector
        self.widget = widget
        self.obj = obj
        self.property_name = property_name
        self.display_name = display_name or property_name.replace('_', ' ').title()
        self.transform_to_model = transform_to_model or (lambda x: x)
        self.transform_from_model = transform_from_model or (lambda x: x)

        # Track last value for undo
        self._last_value = self._get_model_value()

        # For text widgets, track baseline for merge grouping
        self._baseline_value = self._last_value

        # Connect appropriate signal based on widget type
        self._connect_widget_signal()

        # Store binding reference on widget for debugging
        widget.setProperty('_property_binding', self)

    def _get_model_value(self) -> Any:
        """Get current value from model object."""
        return getattr(self.obj, self.property_name, None)

    def _set_model_value(self, value: Any):
        """Set value on model object."""
        setattr(self.obj, self.property_name, value)

    def _get_widget_value(self) -> Any:
        """Get current value from widget."""
        widget = self.widget

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
            return widget.toPlainText()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSlider):
            return widget.value()
        else:
            raise ValueError(f"Unsupported widget type: {type(widget)}")

    def _set_widget_value(self, value: Any):
        """Set value on widget without triggering signals."""
        widget = self.widget

        # Block signals to prevent feedback loop
        widget.blockSignals(True)
        try:
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value if value is not None else 0)
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value) if value else "")
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentText(str(value) if value else "")
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
                widget.setPlainText(str(value) if value is not None else "")
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSlider):
                widget.setValue(int(value) if value is not None else 0)
        finally:
            widget.blockSignals(False)

    def _connect_widget_signal(self):
        """Connect appropriate signal based on widget type."""
        widget = self.widget

        if isinstance(widget, QSpinBox):
            widget.valueChanged.connect(self._on_spinbox_changed)
        elif isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(self._on_double_spinbox_changed)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self._on_combo_changed)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._on_line_edit_changed)
        elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
            widget.textChanged.connect(self._on_text_edit_changed)
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._on_checkbox_changed)
        elif isinstance(widget, QSlider):
            widget.valueChanged.connect(self._on_slider_changed)
        else:
            print(f"[PropertyBinding] Warning: No signal handler for {type(widget)}")

    def _push_command(self, old_value: Any, new_value: Any):
        """Push an undo command for this property change."""
        if old_value == new_value:
            return

        # Get object name for undo text
        obj_name = getattr(self.obj, 'name', None) or str(self.obj)

        self.inspector._push_generic_property_command(
            obj=self.obj,
            property_name=self.property_name,
            old_value=old_value,
            new_value=new_value,
            display_name=self.display_name,
            obj_name=obj_name
        )

        # Emit signal for external listeners
        self.value_changed.emit(self.obj, self.property_name, new_value)

    # === Signal Handlers ===

    def _on_spinbox_changed(self, value: int):
        """Handle QSpinBox value change."""
        old_value = self._last_value
        new_value = self.transform_to_model(value)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        self._last_value = new_value

    def _on_double_spinbox_changed(self, value: float):
        """Handle QDoubleSpinBox value change."""
        old_value = self._last_value
        new_value = self.transform_to_model(value)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        self._last_value = new_value

    def _on_combo_changed(self, text: str):
        """Handle QComboBox selection change."""
        old_value = self._last_value
        new_value = self.transform_to_model(text)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        self._last_value = new_value

    def _on_line_edit_changed(self, text: str):
        """Handle QLineEdit text change (uses baseline for merge)."""
        # For text, use baseline value so typing merges into one command
        old_value = self._baseline_value
        new_value = self.transform_to_model(text)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        # Don't update baseline - let commands merge
        self._last_value = new_value

    def _on_text_edit_changed(self):
        """Handle QTextEdit/QPlainTextEdit text change (uses baseline for merge)."""
        text = self.widget.toPlainText()
        old_value = self._baseline_value
        new_value = self.transform_to_model(text)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        # Don't update baseline - let commands merge
        self._last_value = new_value

    def _on_checkbox_changed(self, state: int):
        """Handle QCheckBox state change."""
        old_value = self._last_value
        new_value = self.transform_to_model(state != 0)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        self._last_value = new_value

    def _on_slider_changed(self, value: int):
        """Handle QSlider value change."""
        old_value = self._last_value
        new_value = self.transform_to_model(value)
        self._set_model_value(new_value)
        self._push_command(old_value, new_value)
        self._last_value = new_value

    # === External API ===

    def refresh_from_model(self):
        """
        Refresh widget to show current model value.

        Call this after undo/redo to sync widget display.
        """
        value = self._get_model_value()
        display_value = self.transform_from_model(value)
        self._set_widget_value(display_value)
        self._last_value = value
        self._baseline_value = value

    def reset_baseline(self):
        """
        Reset the baseline value for text merge grouping.

        Call this on focus lost to start a new undo group.
        """
        self._baseline_value = self._get_model_value()


class PropertyBindingManager:
    """
    Manages property bindings for an Inspector panel.

    Provides factory methods to create bound widgets and tracks
    all active bindings for cleanup and refresh operations.
    """

    def __init__(self, inspector: 'InspectorPanel'):
        self.inspector = inspector
        self._bindings: list[PropertyBinding] = []

    def create_binding(
        self,
        widget: QWidget,
        obj: Any,
        property_name: str,
        **kwargs
    ) -> PropertyBinding:
        """
        Create a property binding and track it.

        Args:
            widget: Pre-created widget to bind
            obj: Object containing the property
            property_name: Name of property on obj
            **kwargs: Additional args for PropertyBinding

        Returns:
            The created PropertyBinding
        """
        binding = PropertyBinding(
            self.inspector,
            widget,
            obj,
            property_name,
            **kwargs
        )
        self._bindings.append(binding)
        return binding

    def clear_bindings(self):
        """Clear all tracked bindings (call when Inspector clears)."""
        self._bindings.clear()

    def refresh_all(self):
        """Refresh all bound widgets from model (call after undo/redo)."""
        for binding in self._bindings:
            binding.refresh_from_model()

    def get_binding_for_widget(self, widget: QWidget) -> Optional[PropertyBinding]:
        """Find binding for a specific widget."""
        for binding in self._bindings:
            if binding.widget is widget:
                return binding
        return None

    def get_bindings_for_object(self, obj: Any) -> list[PropertyBinding]:
        """Get all bindings for a specific object."""
        return [b for b in self._bindings if b.obj is obj]


# === Property Metadata System ===

class PropertyMeta:
    """
    Metadata describing an editable property.

    Used by facets to declare which properties should appear
    in the Inspector and how they should be edited.
    """

    def __init__(
        self,
        name: str,
        prop_type: Type,
        display_name: Optional[str] = None,
        default: Any = None,
        description: str = "",
        # Type-specific options
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        step: Optional[float] = None,
        choices: Optional[list] = None,
        multiline: bool = False,
        code_language: Optional[str] = None,
    ):
        """
        Define property metadata.

        Args:
            name: Property name on object
            prop_type: Python type (int, float, str, bool)
            display_name: Label for Inspector (defaults to name)
            default: Default value
            description: Tooltip/help text
            minimum: Min value for numeric types
            maximum: Max value for numeric types
            step: Step size for numeric types
            choices: List of valid values for enum/choice types
            multiline: True for multiline text
            code_language: Language for code editor ('javascript', 'python')
        """
        self.name = name
        self.prop_type = prop_type
        self.display_name = display_name or name.replace('_', ' ').title()
        self.default = default
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.choices = choices
        self.multiline = multiline
        self.code_language = code_language

    def get_widget_type(self) -> Type[QWidget]:
        """Determine appropriate widget type for this property."""
        if self.choices:
            return QComboBox
        elif self.prop_type == bool:
            return QCheckBox
        elif self.prop_type == int:
            return QSpinBox
        elif self.prop_type == float:
            return QDoubleSpinBox
        elif self.prop_type == str:
            if self.code_language or self.multiline:
                return QPlainTextEdit
            else:
                return QLineEdit
        else:
            return QLineEdit  # Fallback


def editable_properties(*properties: PropertyMeta):
    """
    Class decorator to declare editable properties on a facet.

    Usage:
        @editable_properties(
            PropertyMeta('temperature', float, minimum=0.0, maximum=2.0, default=0.7),
            PropertyMeta('prompt', str, multiline=True),
        )
        class MyFacet(Facet):
            ...
    """
    def decorator(cls):
        # Store property metadata on class
        cls._editable_properties = {p.name: p for p in properties}
        return cls
    return decorator


class PropertyRegistry:
    """
    Central registry for editable property metadata.

    Components (facets, neural nodes, etc.) register their editable properties
    here. The Inspector queries this registry to auto-generate UI with undo.

    Supports:
    - Built-in facet types (LLMFacet, ScriptedFacet, etc.)
    - Dynamically registered types from scripts
    - Neural Canvas node types
    - Any custom component type

    Usage:
        # Register a type's properties
        property_registry.register('LLMFacet', [
            PropertyMeta('temperature', float, minimum=0.0, maximum=2.0),
            PropertyMeta('prompt', str, multiline=True),
        ])

        # Query properties for an object
        props = property_registry.get_properties_for(facet)

        # In scripts:
        context.noodle.register_component_type('MyCustomFacet', [
            {'name': 'threshold', 'type': 'float', 'min': 0, 'max': 1},
            {'name': 'mode', 'type': 'choice', 'choices': ['fast', 'slow']},
        ])
    """

    _instance: Optional['PropertyRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._register_builtin_types()

    def _register_builtin_types(self):
        """Register built-in facet types and their properties."""

        # LLMFacet - Language model based facets
        self.register('LLMFacet', [
            PropertyMeta('model', str, display_name='Model',
                        choices=['Small', 'Medium', 'Large'],
                        description='LLM model size'),
            PropertyMeta('temperature', float, display_name='Temperature',
                        minimum=0.0, maximum=2.0, step=0.1, default=0.7,
                        description='Sampling temperature'),
            PropertyMeta('max_tokens', int, display_name='Max Tokens',
                        minimum=1, maximum=4096, default=150,
                        description='Maximum output tokens'),
            PropertyMeta('prompt', str, display_name='Prompt',
                        multiline=True,
                        description='LLM prompt template'),
        ])

        # ScriptedFacet - JavaScript controlled facets
        self.register('ScriptedFacet', [
            PropertyMeta('salience_script', str, display_name='Salience Script',
                        multiline=True, code_language='javascript',
                        description='JavaScript code for dynamic salience'),
        ])

        # CharmNetworkFacet - Neural affect model
        self.register('CharmNetworkFacet', [
            # CharmNetwork has no editable properties in Inspector
            # (configured via Neural Canvas)
        ])

        # ContextIntelligenceFacet
        self.register('ContextIntelligenceFacet', [
            PropertyMeta('model', str, display_name='Model',
                        choices=['Small', 'Medium', 'Large']),
            PropertyMeta('temperature', float, display_name='Temperature',
                        minimum=0.0, maximum=2.0, step=0.1, default=0.3),
        ])

        # InsightEmergenceFacet
        self.register('InsightEmergenceFacet', [
            PropertyMeta('model', str, display_name='Model',
                        choices=['Small', 'Medium', 'Large']),
            PropertyMeta('temperature', float, display_name='Temperature',
                        minimum=0.0, maximum=2.0, step=0.1, default=0.8),
            PropertyMeta('prompt', str, display_name='Prompt', multiline=True),
        ])

        # SubconsciousFacet
        self.register('SubconsciousFacet', [
            PropertyMeta('model', str, display_name='Model',
                        choices=['Small', 'Medium', 'Large']),
            PropertyMeta('prompt', str, display_name='Prompt', multiline=True),
        ])

        # ConvergenceFacet
        self.register('ConvergenceFacet', [
            PropertyMeta('model', str, display_name='Model',
                        choices=['Small', 'Medium', 'Large']),
            PropertyMeta('temperature', float, display_name='Temperature',
                        minimum=0.0, maximum=2.0, step=0.1, default=0.7),
            PropertyMeta('prompt', str, display_name='Prompt', multiline=True),
        ])

        # Generic base facet properties (applied to all types)
        self.register('_base_facet', [
            PropertyMeta('name', str, display_name='Name'),
            PropertyMeta('enabled', bool, display_name='Enabled', default=True),
            PropertyMeta('locked', bool, display_name='Locked', default=False),
        ])

    def register(self, type_name: str, properties: list):
        """
        Register editable properties for a component type.

        Args:
            type_name: Type identifier (e.g., 'LLMFacet', 'MyCustomNode')
            properties: List of PropertyMeta objects
        """
        self._registry[type_name] = {p.name: p for p in properties}
        print(f"[PropertyRegistry] Registered {len(properties)} properties for '{type_name}'")

    def register_from_dict(self, type_name: str, property_dicts: list):
        """
        Register properties from dictionary format (for scripting API).

        Args:
            type_name: Type identifier
            property_dicts: List of dicts with keys:
                - name: Property name (required)
                - type: 'int', 'float', 'str', 'bool', 'choice' (required)
                - display_name: Human label
                - min/max: For numeric types
                - step: For float types
                - choices: For choice type
                - multiline: For str type
                - code_language: For code editors
                - default: Default value
                - description: Help text

        Example:
            register_from_dict('MyFacet', [
                {'name': 'threshold', 'type': 'float', 'min': 0, 'max': 1},
                {'name': 'mode', 'type': 'choice', 'choices': ['fast', 'slow']},
            ])
        """
        properties = []
        type_map = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'choice': str,  # Choice is string with choices list
        }

        for d in property_dicts:
            prop_type = type_map.get(d.get('type', 'str'), str)
            meta = PropertyMeta(
                name=d['name'],
                prop_type=prop_type,
                display_name=d.get('display_name'),
                default=d.get('default'),
                description=d.get('description', ''),
                minimum=d.get('min'),
                maximum=d.get('max'),
                step=d.get('step'),
                choices=d.get('choices'),
                multiline=d.get('multiline', False),
                code_language=d.get('code_language'),
            )
            properties.append(meta)

        self.register(type_name, properties)

    def unregister(self, type_name: str):
        """Remove a registered type (for cleanup or hot-reload)."""
        if type_name in self._registry:
            del self._registry[type_name]
            print(f"[PropertyRegistry] Unregistered '{type_name}'")

    def get_properties(self, type_name: str, include_base: bool = True) -> Dict[str, PropertyMeta]:
        """
        Get all editable properties for a type.

        Args:
            type_name: Type identifier
            include_base: If True, include base facet properties

        Returns:
            Dict mapping property name to PropertyMeta
        """
        result = {}

        # Include base properties if requested
        if include_base and '_base_facet' in self._registry:
            result.update(self._registry['_base_facet'])

        # Add type-specific properties
        if type_name in self._registry:
            result.update(self._registry[type_name])

        return result

    def get_properties_for(self, obj) -> Dict[str, PropertyMeta]:
        """
        Get editable properties for an object instance.

        Automatically determines type from:
        - obj.facet_type (for Facets)
        - obj.node_type (for Neural Canvas nodes)
        - obj.__class__.__name__ (fallback)

        Args:
            obj: Object to get properties for

        Returns:
            Dict mapping property name to PropertyMeta
        """
        # Try facet_type first
        type_name = getattr(obj, 'facet_type', None)

        # Try node_type for Neural Canvas
        if not type_name:
            type_name = getattr(obj, 'node_type', None)

        # Fallback to class name
        if not type_name:
            type_name = obj.__class__.__name__

        return self.get_properties(type_name)

    def is_registered(self, type_name: str) -> bool:
        """Check if a type is registered."""
        return type_name in self._registry

    def get_registered_types(self) -> list:
        """Get list of all registered type names."""
        return list(self._registry.keys())


# Global singleton instance
property_registry = PropertyRegistry()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
