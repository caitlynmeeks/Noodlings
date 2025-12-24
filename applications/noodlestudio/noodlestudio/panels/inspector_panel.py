"""
Inspector Panel - Component-based property editor

Shows and edits ALL properties of selected entity:
- Users: name, description, location, inventory
- Noodlings: name, species, description, LLM model, personality traits
- Objects: name, description, properties
- Rooms: name, description, exits

Every atom of noodleMUSH exposed and editable!

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QScrollArea,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QProgressBar, QListWidget,
                             QFileDialog, QListWidgetItem, QApplication)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon, QFontMetrics
import requests
import yaml
from pathlib import Path
import sys
sys.path.append('..')

from noodlestudio.widgets.collapsible_section import CollapsibleSection
from ..panels.floating_text_editor import FloatingTextEditor
from ..core.property_binding import PropertyBindingManager, PropertyMeta, property_registry
from ..core.undo_manager import UndoManager
from ..core.commands.neural_commands import EditNeuralNodeParamCommand, RenameNeuralNodeCommand


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
            # Check for Cmd (macOS) or Ctrl (other platforms)
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.MetaModifier or modifiers & Qt.KeyboardModifier.ControlModifier:
                # Cmd/Ctrl + Click - open floating editor
                self.open_floating_editor()
                return

        # Normal click behavior
        super().mousePressEvent(event)

    def open_floating_editor(self):
        """Open floating text editor for this field."""
        if self.floating_editor and self.floating_editor.isVisible():
            self.floating_editor.raise_()
            self.floating_editor.activateWindow()
            return

        # Create floating editor
        self.floating_editor = FloatingTextEditor(
            field_name=self.field_name,
            field_key=self.field_name,
            initial_value=self.toPlainText(),
            read_only=self.isReadOnly(),
            parent=self.window()
        )

        # Connect apply signal
        def on_text_applied(key, value):
            self.setPlainText(value)
            if self.on_apply_callback:
                self.on_apply_callback(value)

        self.floating_editor.textApplied.connect(on_text_applied)
        self.floating_editor.show()


class InspectorPanel(QWidget):
    """
    Component-based Inspector panel.

    Shows editable properties for selected entity.
    Every field is live-editable with instant save!
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_entity = None
        self.current_agent_id = None  # Initialize to None explicitly
        self.current_facet = None  # Track if currently showing a facet
        self.last_entity_type = None  # Remember last entity for restore
        self.last_entity_data = None
        self.api_base = "http://localhost:8081/api"

        # Allow panel to shrink to very small sizes for tight panel layouts
        self.setMinimumWidth(200)

        # Flag to prevent double-triggering during toggle operations
        self.toggling = False

        # Flag to prevent refresh during save operations
        self.is_saving = False

        # Flag to prevent re-entrant loading (e.g., double-tap)
        self.is_loading = False

        # Track component widgets for save operations
        # Structure: {agent_id: {component_id: {field_name: widget}}}
        self.component_widgets = {}

        # Track CollapsibleSection expanded state (like SceneHierarchy does)
        # Structure: {section_title: bool}
        self.collapsible_expanded_state = {}

        # Initialize facet dropdown and container (set to None until agent loaded)
        self.facet_dropdown = None
        self.facet_properties_container = None
        self.facet_properties_layout = None
        self.current_assembly = None

        # Property binding manager for automatic undo support
        self._binding_manager = PropertyBindingManager(self)

        # Initialize UI directly on this widget
        self.init_ui(self)

        # Ensure Inspector starts clear (no phantom selections)
        self.clear_inspector()

        # Live update timer for Noodle Component
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_data)
        self.update_timer.start(1000)  # Update every second

        self.live_affect_labels = {}
        self.live_surprise_label = None

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        self.entity_header = QLabel("No entity selected")
        self.entity_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.entity_header.setStyleSheet("color: #D2D2D2; padding: 8px;")
        layout.addWidget(self.entity_header)

        # Scrollable properties area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.properties_widget = QWidget()
        self.properties_layout = QVBoxLayout(self.properties_widget)
        self.properties_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.properties_widget)
        layout.addWidget(scroll)

    def clear_inspector(self):
        """Clear inspector when nothing is selected."""
        print("[Inspector] clear_inspector() called")
        import traceback
        print(''.join(traceback.format_stack()[-5:]))

        self.current_entity = None
        self.current_agent_id = None
        self.current_facet = None
        self.entity_header.setText("Select a noodling or prim")

        # Clear property bindings (important for undo system)
        self._binding_manager.clear_bindings()

        # Clear existing properties
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.component_widgets.clear()

    def load_facet(self, facet):
        """
        Load and display facet properties for editing.

        NEW BEHAVIOR (Dropdown Model):
        - If agent loaded → Sync dropdown to selected facet
        - If facet is None → Reset dropdown to "(none)"
        - Agent basics always stay visible
        - NEVER rebuild inspector if agent already loaded

        Args:
            facet: Facet object from facet_system.py (None to deselect)
        """
        if facet is None:
            # Facet deselected - reset dropdown
            print("[Inspector] Facet deselected - resetting dropdown")
            try:
                if hasattr(self, 'facet_dropdown') and self.facet_dropdown:
                    self.facet_dropdown.setCurrentIndex(0)  # (none)
                self.current_facet = None
            except Exception as e:
                print(f"[Inspector] Error resetting dropdown: {e}")
            return

        try:
            self.current_facet = facet
            print(f"[Inspector] Facet selected from graph: {facet.name} (ID: {facet.id})")

            # Check if agent is already loaded (has dropdown)
            if hasattr(self, 'facet_dropdown') and self.facet_dropdown:
                print(f"[Inspector] Agent already loaded - syncing dropdown")
                # Sync dropdown to match selected facet
                for i in range(self.facet_dropdown.count()):
                    if self.facet_dropdown.itemData(i) == facet.id:
                        self.facet_dropdown.setCurrentIndex(i)
                        print(f"[Inspector] Synced dropdown to: {facet.name}")
                        return

                # Facet not found in dropdown - might be from different agent
                print(f"[Inspector] Warning: Facet '{facet.name}' not in current agent's dropdown")
                return

            # No agent loaded - show standalone facet view
            print(f"[Inspector] No agent loaded - using standalone facet view")
            self._load_facet_standalone(facet)

        except Exception as e:
            print(f"[Inspector] Error loading facet: {e}")
            import traceback
            traceback.print_exc()

    def _load_facet_standalone(self, facet):
        """
        Load facet in standalone mode using PropertyBinding system.

        Uses the PropertyRegistry to auto-generate UI with undo support.

        Args:
            facet: Facet object from facet_system.py
        """
        try:
            self.clear_inspector()
            self.current_facet = facet
            self.entity_header.setText(f"Facet: {facet.name}")
        except Exception as e:
            print(f"[Inspector] Error in standalone facet load: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # Basic Properties section (read-only info + basic editable)
            basic_section = CollapsibleSection("Basic Properties")
            basic_form = QFormLayout()

            # ID (read-only)
            id_field = QLineEdit(facet.id)
            id_field.setReadOnly(True)
            id_field.setStyleSheet("color: #888;")
            basic_form.addRow("ID:", id_field)

            # Name (with undo)
            name_field = self.create_bound_lineedit(facet, 'name', display_name='Name')
            basic_form.addRow("Name:", name_field)

            # Type (read-only)
            type_field = QLineEdit(facet.facet_type)
            type_field.setReadOnly(True)
            type_field.setStyleSheet("color: #888;")
            basic_form.addRow("Type:", type_field)

            # Enabled (with undo)
            enabled_checkbox = self.create_bound_checkbox(facet, 'enabled', display_name='Enabled')
            basic_form.addRow("Enabled:", enabled_checkbox)

            basic_section.set_content_layout(basic_form)
            self.properties_layout.addWidget(basic_section)

            # Type-specific Configuration section
            # Uses PropertyRegistry to auto-generate widgets with undo
            props = property_registry.get_properties(facet.facet_type, include_base=False)
            if props:
                config_section = CollapsibleSection(f"{facet.facet_type} Configuration")
                config_form = QFormLayout()

                for prop_name, meta in props.items():
                    widget = self.create_widget_for_property(facet, meta)
                    config_form.addRow(f"{meta.display_name}:", widget)

                # Add template variables hint for prompt fields
                if 'prompt' in props:
                    variables_hint = QLabel(
                        "Variables: {incoming_data}, {observations}, "
                        "{affect_valence:.2f}, {affect_arousal:.2f}, {affect_dominance:.2f}, "
                        "{affect_boredom:.2f}, {affect_sorrow:.2f}"
                    )
                    variables_hint.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
                    variables_hint.setWordWrap(True)
                    config_form.addRow(variables_hint)

                config_section.set_content_layout(config_form)
                self.properties_layout.addWidget(config_section)

            # Inputs/Outputs (informational, read-only)
            io_section = CollapsibleSection("Inputs & Outputs")
            io_form = QFormLayout()

            inputs_list = "\n".join(f"  {inp.name} ({'required' if inp.required else 'optional'})"
                                    for inp in (facet.input_pads or []))
            outputs_list = "\n".join(f"  {out.name}" for out in (facet.output_pads or []))

            io_label = QLabel(f"Inputs:\n{inputs_list or '  None'}\n\nOutputs:\n{outputs_list or '  None'}")
            io_label.setStyleSheet("color: #AAA; font-size: 10pt;")
            io_label.setWordWrap(True)
            io_form.addRow(io_label)

            io_section.set_content_layout(io_form)
            self.properties_layout.addWidget(io_section)

            self.properties_layout.addStretch()

        except Exception as e:
            print(f"[Inspector] Error building facet properties UI: {e}")
            import traceback
            traceback.print_exc()

            error_label = QLabel(f"Error loading facet properties:\n{str(e)}")
            error_label.setStyleSheet("color: #FF6B6B;")
            error_label.setWordWrap(True)
            self.properties_layout.addWidget(error_label)

    def _auto_save_facet_assembly(self):
        """Auto-save the current facet assembly to its YAML file."""
        try:
            # Get the facets editor panel to access current assembly
            main_window = self.window()
            if hasattr(main_window, 'facets_editor'):
                assembly = main_window.facets_editor.current_assembly
                if assembly and hasattr(assembly, 'filepath') and assembly.filepath:
                    assembly.save_yaml(assembly.filepath)
                    print(f"[Inspector] Auto-saved facet assembly to {assembly.filepath}")
        except Exception as e:
            print(f"[Inspector] Error auto-saving facet assembly: {e}")

    # ========== UNDO SYSTEM INTERNAL METHODS ==========

    def _set_facet_property_internal(self, facet_id: str, property_name: str, value):
        """
        Set facet property without pushing undo command.

        Called by InspectorPropertyCommand during undo/redo.
        Updates both the data model and any visible widgets.
        """
        # Get the facet from the current assembly
        main_window = self.window()
        if not hasattr(main_window, 'facets_editor'):
            return

        assembly = main_window.facets_editor.current_assembly
        if not assembly:
            return

        facet = assembly.get_facet(facet_id)
        if not facet:
            return

        # Set the property
        setattr(facet, property_name, value)

        # Update widget if visible (find widget by facet_id and property_name)
        # This is tricky because widgets are created dynamically
        # For now, just trigger a refresh if the facet is currently displayed
        if self.current_facet and self.current_facet.id == facet_id:
            # Refresh the facet display to show new value
            self._refresh_facet_widget(facet_id, property_name, value)

        # Save to disk
        self._save_facet_property_to_disk()

    def _save_facet_property_to_disk(self):
        """Save facet assembly to disk (called by undo commands)."""
        self._auto_save_facet_assembly()

    def _refresh_facet_widget(self, facet_id: str, property_name: str, value):
        """
        Update widget display after undo/redo changes property.

        This finds and updates the specific widget showing this property.
        """
        # For now, we don't have widget tracking - the widget will show
        # stale data until user re-selects the facet. This is acceptable
        # for initial implementation.
        #
        # TODO: Track widgets by (facet_id, property_name) for live updates
        pass

    def _push_facet_property_command(self, facet, property_name: str, old_value, new_value):
        """
        Push an undo command for a facet property change.

        Call this AFTER setting the property on the facet.

        Args:
            facet: The Facet object being modified
            property_name: Name of the property (e.g., 'prompt', 'temperature')
            old_value: Value before change
            new_value: Value after change
        """
        if old_value == new_value:
            return  # No change

        from ..core.undo_manager import undo_manager
        from ..core.commands import InspectorPropertyCommand

        cmd = InspectorPropertyCommand(
            inspector=self,
            facet_id=facet.id,
            property_name=property_name,
            old_value=old_value,
            new_value=new_value,
            facet_name=facet.name
        )
        undo_manager.push(cmd)

    def _push_generic_property_command(
        self,
        obj,
        property_name: str,
        old_value,
        new_value,
        display_name: str = "",
        obj_name: str = ""
    ):
        """
        Push a generic undo command for any property change.

        This is called by PropertyBinding to create undo commands.
        Works with any object type, not just facets.

        Args:
            obj: Object containing the property
            property_name: Name of property being changed
            old_value: Value before change
            new_value: Value after change
            display_name: Human-readable property name
            obj_name: Name of object for undo text
        """
        if old_value == new_value:
            return

        from ..core.undo_manager import undo_manager
        from ..core.commands import GenericPropertyCommand

        cmd = GenericPropertyCommand(
            inspector=self,
            obj=obj,
            property_name=property_name,
            old_value=old_value,
            new_value=new_value,
            display_name=display_name,
            obj_name=obj_name
        )
        undo_manager.push(cmd)

    # === Widget Factory Methods ===

    def create_bound_spinbox(
        self,
        obj,
        property_name: str,
        minimum: int = 0,
        maximum: int = 100,
        display_name: str = None,
        **kwargs
    ) -> QSpinBox:
        """
        Create a QSpinBox bound to a property with automatic undo.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            minimum: Minimum value
            maximum: Maximum value
            display_name: Label for undo text
            **kwargs: Additional spinbox config

        Returns:
            Bound QSpinBox widget
        """
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        for key, val in kwargs.items():
            if hasattr(widget, f'set{key.title()}'):
                getattr(widget, f'set{key.title()}')(val)

        # Set initial value
        value = getattr(obj, property_name, 0) or 0
        widget.setValue(value)

        # Create binding
        self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name
        )

        return widget

    def create_bound_double_spinbox(
        self,
        obj,
        property_name: str,
        minimum: float = 0.0,
        maximum: float = 1.0,
        step: float = 0.1,
        decimals: int = 2,
        display_name: str = None,
        **kwargs
    ) -> QDoubleSpinBox:
        """
        Create a QDoubleSpinBox bound to a property with automatic undo.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            minimum: Minimum value
            maximum: Maximum value
            step: Single step value
            decimals: Number of decimal places
            display_name: Label for undo text

        Returns:
            Bound QDoubleSpinBox widget
        """
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)

        # Set initial value
        value = getattr(obj, property_name, 0.0) or 0.0
        widget.setValue(value)

        # Create binding
        self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name
        )

        return widget

    def create_bound_combobox(
        self,
        obj,
        property_name: str,
        choices: list,
        display_name: str = None,
        transform_to_model=None,
        transform_from_model=None,
        **kwargs
    ):
        """
        Create a QComboBox bound to a property with automatic undo.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            choices: List of valid choices
            display_name: Label for undo text
            transform_to_model: Transform combo text to model value
            transform_from_model: Transform model value to combo text

        Returns:
            Bound QComboBox widget
        """
        from PyQt6.QtWidgets import QComboBox

        widget = QComboBox()
        widget.addItems(choices)

        # Set initial value
        value = getattr(obj, property_name, None)
        if value:
            if transform_from_model:
                value = transform_from_model(value)
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)

        # Create binding
        self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name,
            transform_to_model=transform_to_model,
            transform_from_model=transform_from_model
        )

        return widget

    def create_bound_textedit(
        self,
        obj,
        property_name: str,
        max_height: int = 150,
        display_name: str = None,
        code_language: str = None,
        **kwargs
    ):
        """
        Create a text editor bound to a property with automatic undo.

        Supports Cmd+Click floating editor for prompts and code.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            max_height: Maximum widget height
            display_name: Label for undo text
            code_language: If set, style as code editor

        Returns:
            Bound ClickableTextEdit widget
        """
        obj_name = getattr(obj, 'name', '') or ''
        field_name = f"{obj_name} - {display_name or property_name}"

        # Create with floating editor callback that uses binding
        widget = ClickableTextEdit(
            field_name=field_name,
            on_apply_callback=None  # Will be set after binding
        )

        # Set initial value
        value = getattr(obj, property_name, '') or ''
        widget.setPlainText(value)
        widget.setMaximumHeight(max_height)

        # Style based on code language
        if code_language:
            widget.setStyleSheet(
                "background-color: #1E1E1E; color: #D2D2D2; "
                "padding: 4px; font-family: 'Courier New';"
            )
        else:
            widget.setStyleSheet(
                "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;"
            )

        # Create binding
        binding = self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name
        )

        # Connect floating editor to binding (applies through widget → binding)
        widget.on_apply_callback = lambda text: widget.setPlainText(text)

        return widget

    def create_bound_checkbox(
        self,
        obj,
        property_name: str,
        display_name: str = None,
        **kwargs
    ):
        """
        Create a QCheckBox bound to a property with automatic undo.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            display_name: Label for undo text

        Returns:
            Bound QCheckBox widget
        """
        from PyQt6.QtWidgets import QCheckBox

        widget = QCheckBox()

        # Set initial value
        value = getattr(obj, property_name, False)
        widget.setChecked(bool(value))

        # Create binding
        self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name
        )

        return widget

    def create_bound_lineedit(
        self,
        obj,
        property_name: str,
        display_name: str = None,
        placeholder: str = "",
        **kwargs
    ):
        """
        Create a QLineEdit bound to a property with automatic undo.

        Args:
            obj: Object containing the property
            property_name: Name of property to bind
            display_name: Label for undo text
            placeholder: Placeholder text

        Returns:
            Bound QLineEdit widget
        """
        widget = QLineEdit()
        if placeholder:
            widget.setPlaceholderText(placeholder)

        # Set initial value
        value = getattr(obj, property_name, '') or ''
        widget.setText(str(value))

        # Create binding
        self._binding_manager.create_binding(
            widget, obj, property_name,
            display_name=display_name
        )

        return widget

    def create_widget_for_property(self, obj, meta: PropertyMeta):
        """
        Create appropriate widget for a property based on its metadata.

        This is the main factory method for auto-generating Inspector UI
        from property metadata. Scripted components can use this to get
        automatic undo support.

        Args:
            obj: Object containing the property
            meta: PropertyMeta describing the property

        Returns:
            Bound widget appropriate for the property type
        """
        if meta.choices:
            return self.create_bound_combobox(
                obj, meta.name,
                choices=meta.choices,
                display_name=meta.display_name
            )
        elif meta.prop_type == bool:
            return self.create_bound_checkbox(
                obj, meta.name,
                display_name=meta.display_name
            )
        elif meta.prop_type == int:
            return self.create_bound_spinbox(
                obj, meta.name,
                minimum=int(meta.minimum or 0),
                maximum=int(meta.maximum or 9999),
                display_name=meta.display_name
            )
        elif meta.prop_type == float:
            return self.create_bound_double_spinbox(
                obj, meta.name,
                minimum=meta.minimum or 0.0,
                maximum=meta.maximum or 1.0,
                step=meta.step or 0.1,
                display_name=meta.display_name
            )
        elif meta.prop_type == str:
            if meta.multiline or meta.code_language:
                return self.create_bound_textedit(
                    obj, meta.name,
                    display_name=meta.display_name,
                    code_language=meta.code_language
                )
            else:
                return self.create_bound_lineedit(
                    obj, meta.name,
                    display_name=meta.display_name
                )
        else:
            # Default to line edit for unknown types
            return self.create_bound_lineedit(
                obj, meta.name,
                display_name=meta.display_name
            )

    def build_inspector_for_object(self, obj, layout, include_base: bool = False):
        """
        Auto-generate Inspector widgets for an object with editable properties.

        Uses the PropertyRegistry to look up editable properties for the object's
        type. Works with:
        - Built-in facet types (LLMFacet, ScriptedFacet, etc.)
        - Dynamically registered types from scripts
        - Any class with _editable_properties attribute

        Args:
            obj: Object to inspect
            layout: QFormLayout to add widgets to
            include_base: If True, include base properties (name, enabled, locked)

        Returns:
            True if properties were added, False if none found
        """
        # First try the PropertyRegistry
        editable = property_registry.get_properties_for(obj)

        # Fall back to class attribute if registry has nothing
        if not editable:
            editable = getattr(obj.__class__, '_editable_properties', None)

        if not editable:
            return False

        # Filter out base properties unless requested
        if not include_base:
            base_props = {'name', 'enabled', 'locked'}
            editable = {k: v for k, v in editable.items() if k not in base_props}

        if not editable:
            return False

        for prop_name, meta in editable.items():
            widget = self.create_widget_for_property(obj, meta)
            layout.addRow(f"{meta.display_name}:", widget)

        return True

    def _get_agent_assembly(self, agent_id: str, agent_data: dict):
        """
        Load agent's facet assembly from YAML file.

        Args:
            agent_id: Agent ID (UUID like "agent_a56e0ac2...")
            agent_data: Agent data dict (or full entity_data with 'data' nested)

        Returns:
            FacetAssembly or None if not found
        """
        try:
            from ..core.facet_system import FacetAssembly

            # Get the actual agent dict (might be nested in 'data')
            agent = agent_data.get('data', agent_data)

            # Get facet_assembly reference from config (like Facets Editor does)
            config = agent.get('config', {})
            facet_assembly_ref = config.get('facet_assembly')

            # Handle both string and dict formats
            if isinstance(facet_assembly_ref, dict):
                facet_assembly_ref = facet_assembly_ref.get('ref')

            if not facet_assembly_ref:
                print(f"[Inspector] No facet_assembly in agent config for: {agent_id}")
                return None

            print(f"[Inspector] _get_agent_assembly: facet_assembly_ref='{facet_assembly_ref}' from config")

            # Use __file__ based path resolution (CWD-independent)
            import os
            # Try facet_assemblies directory first (new location)
            assembly_path = os.path.join(
                os.path.dirname(__file__),
                '../../facet_assemblies',
                f'{facet_assembly_ref}.yaml'
            )
            print(f"[Inspector] PRIMARY path: {assembly_path}")
            print(f"[Inspector] PRIMARY exists? {os.path.exists(assembly_path)}")

            if not os.path.exists(assembly_path):
                # Fallback: check cmush/recipes for embedded assemblies
                assembly_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../../cmush/recipes',
                    f'{facet_assembly_ref}.yaml'
                )
                print(f"[Inspector] FALLBACK path: {assembly_path}")
                print(f"[Inspector] FALLBACK exists? {os.path.exists(assembly_path)}")

            if not os.path.exists(assembly_path):
                print(f"[Inspector] ❌ RETURNING NONE - No facet assembly found for '{facet_assembly_ref}'")
                return None

            print(f"[Inspector] ✅ Loading assembly from: {assembly_path}")
            assembly = FacetAssembly.load_yaml(str(assembly_path))
            print(f"[Inspector] ✅ SUCCESS - Loaded assembly '{assembly.name}' with {len(assembly.facets)} facets")
            return assembly

        except Exception as e:
            print(f"[Inspector] Error loading agent assembly: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_facet_dropdown_selector(self, agent_id: str, agent_data: dict):
        """
        Add facet dropdown selector + properties editor.

        Replaces collapsible sections with a single dropdown to select which facet to edit.
        When facet selected in graph, dropdown syncs to show that facet.

        Args:
            agent_id: Agent UUID
            agent_data: Agent data dict
        """
        from PyQt6.QtWidgets import QFrame, QComboBox

        print(f"[Inspector] _add_facet_dropdown_selector called for agent: {agent_id}")
        print(f"[Inspector] agent_data keys: {agent_data.keys() if isinstance(agent_data, dict) else 'NOT A DICT'}")

        # Load agent's facet assembly
        assembly = self._get_agent_assembly(agent_id, agent_data)
        if not assembly:
            print(f"[Inspector] ERROR: No assembly loaded, cannot create facet dropdown")
            print(f"[Inspector] Returning early - facet dropdown will NOT be created")
            return

        print(f"[Inspector] ✅ Assembly loaded: {assembly.name} with {len(assembly.facets)} facets")
        print(f"[Inspector] About to add separator and dropdown widgets...")

        # Store assembly reference for dropdown updates
        self.current_assembly = assembly

        # Add separator
        print(f"[Inspector] Creating separator...")
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555555; max-height: 2px;")
        self.properties_layout.addWidget(separator)
        print(f"[Inspector] ✅ Separator added to layout")

        # Facet selector section
        print(f"[Inspector] Creating facet selector group...")
        facet_selector_group = self.create_property_group("Facet")
        print(f"[Inspector] ✅ Facet selector group created")

        # Dropdown with facet names
        print(f"[Inspector] Creating dropdown with {len(assembly.facets)} facets...")
        self.facet_dropdown = QComboBox()
        self.facet_dropdown.addItem("(none)", None)  # Default empty selection

        for facet in assembly.facets:
            self.facet_dropdown.addItem(facet.name, facet.id)
            print(f"[Inspector]   - Added facet: {facet.name}")

        print(f"[Inspector] ✅ Dropdown created with {self.facet_dropdown.count()} items")

        self.facet_dropdown.setStyleSheet("""
            QComboBox {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 4px 8px;
                padding-right: 25px;
                border-radius: 3px;
                min-width: 150px;
            }
            QComboBox:hover {
                border: 1px solid #666666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #D2D2D2;
                width: 0px;
                height: 0px;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: #3e3e3e;
                color: #D2D2D2;
                selection-background-color: #555555;
                border: 1px solid #555555;
            }
        """)

        def on_facet_dropdown_changed(index):
            facet_id = self.facet_dropdown.itemData(index)
            if facet_id:
                # Find facet object
                facet = next((f for f in assembly.facets if f.id == facet_id), None)
                if facet:
                    # Defer UI rebuild to allow selection highlight to paint first
                    # This prevents the 0.5-1s lag when clicking facets in the graph
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._load_facet_properties_inline(facet))
            else:
                # Clear facet properties
                self._clear_facet_properties_inline()

        print(f"[Inspector] Connecting dropdown signal...")
        self.facet_dropdown.currentIndexChanged.connect(on_facet_dropdown_changed)
        print(f"[Inspector] Adding dropdown to facet_selector_group...")
        facet_selector_group.content.layout().addRow("Select:", self.facet_dropdown)
        print(f"[Inspector] ✅ Dropdown added to group")

        print(f"[Inspector] Adding facet_selector_group to properties_layout...")
        self.properties_layout.addWidget(facet_selector_group)
        print(f"[Inspector] ✅ facet_selector_group added to layout")

        # Container for facet properties (populated when dropdown changes)
        print(f"[Inspector] Creating facet_properties_container...")
        self.facet_properties_container = QWidget()
        self.facet_properties_layout = QVBoxLayout(self.facet_properties_container)
        self.facet_properties_layout.setContentsMargins(0, 0, 0, 0)
        print(f"[Inspector] Adding facet_properties_container to layout...")
        self.properties_layout.addWidget(self.facet_properties_container)
        print(f"[Inspector] ✅ facet_properties_container added to layout")

        print(f"[Inspector] ✅✅✅ FACET DROPDOWN SETUP COMPLETE ✅✅✅")
        print(f"[Inspector] Total items in dropdown: {self.facet_dropdown.count()}")
        print(f"[Inspector] facet_dropdown exists: {hasattr(self, 'facet_dropdown')}")
        print(f"[Inspector] Properties layout widget count: {self.properties_layout.count()}")

    def _load_facet_properties_inline(self, facet):
        """Load facet properties into the inline container (below dropdown)."""
        from PyQt6.QtWidgets import QComboBox, QCheckBox

        # Clear existing properties
        self._clear_facet_properties_inline()

        print(f"[Inspector] Loading facet properties inline for: {facet.name}")

        # Create properties widget directly (no CollapsibleSection wrapper)
        props_widget = QWidget()
        props_layout = QFormLayout(props_widget)
        props_layout.setContentsMargins(8, 8, 8, 8)

        # LLM Configuration (for LLMFacet types)
        if facet.facet_type == "LLMFacet":
            # Model dropdown
            model_combo = QComboBox()
            model_combo.addItems(["Small", "Medium", "Large"])
            model_combo.setStyleSheet("""
                QComboBox {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 4px 8px;
                    padding-right: 25px;
                    border-radius: 3px;
                    min-width: 100px;
                }
                QComboBox:hover {
                    border: 1px solid #666666;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 20px;
                    border-left: 1px solid #555555;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 6px solid #D2D2D2;
                    width: 0px;
                    height: 0px;
                    margin-right: 5px;
                }
                QComboBox QAbstractItemView {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    selection-background-color: #555555;
                    border: 1px solid #555555;
                }
            """)

            # Set current value
            current_model = facet.model or "Medium"
            index = model_combo.findText(current_model.title() if isinstance(current_model, str) else current_model, Qt.MatchFlag.MatchFixedString)
            if index < 0:
                index = model_combo.findText(current_model.upper() if isinstance(current_model, str) else current_model, Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                model_combo.setCurrentIndex(index)

            model_combo._last_value = facet.model or "Medium"  # Track for undo

            def on_model_changed(text, f=facet, combo=model_combo):
                old_val = getattr(combo, '_last_value', f.model)
                new_val = text.upper()
                setattr(f, 'model', new_val)
                self._push_facet_property_command(f, 'model', old_val, new_val)
                combo._last_value = new_val

            model_combo.currentTextChanged.connect(on_model_changed)
            props_layout.addRow("Model:", model_combo)

            # Temperature (with undo support)
            temp_spin = QDoubleSpinBox()
            temp_spin.setRange(0.0, 2.0)
            temp_spin.setSingleStep(0.1)
            initial_temp = facet.temperature or 0.7
            temp_spin.setValue(initial_temp)
            temp_spin._last_value = initial_temp  # Track for undo

            def on_temp_changed(val, f=facet, spin=temp_spin):
                old_val = getattr(spin, '_last_value', f.temperature)
                setattr(f, 'temperature', val)
                self._push_facet_property_command(f, 'temperature', old_val, val)
                spin._last_value = val

            temp_spin.valueChanged.connect(on_temp_changed)
            props_layout.addRow("Temperature:", temp_spin)

            # Max tokens (with undo support)
            tokens_spin = QSpinBox()
            tokens_spin.setRange(1, 4096)
            initial_tokens = facet.max_tokens or 150
            tokens_spin.setValue(initial_tokens)
            tokens_spin._last_value = initial_tokens  # Track for undo

            def on_tokens_changed(val, f=facet, spin=tokens_spin):
                old_val = getattr(spin, '_last_value', f.max_tokens)
                setattr(f, 'max_tokens', val)
                self._push_facet_property_command(f, 'max_tokens', old_val, val)
                spin._last_value = val

            tokens_spin.valueChanged.connect(on_tokens_changed)
            props_layout.addRow("Max Tokens:", tokens_spin)

            # Prompt (with undo support - uses command merging for typing)
            prompt_edit = ClickableTextEdit(
                field_name=f"{facet.name} - Prompt",
                on_apply_callback=lambda text, f=facet: (
                    setattr(f, 'prompt', text),
                    self._push_facet_property_command(f, 'prompt', f.prompt, text)
                )
            )
            prompt_edit.setPlainText(facet.prompt or "")
            prompt_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
            prompt_edit.setMaximumHeight(150)
            prompt_edit._baseline_value = facet.prompt or ""  # Track baseline for undo

            def on_prompt_changed(edit=prompt_edit, f=facet):
                new_text = edit.toPlainText()
                old_text = getattr(edit, '_baseline_value', '')
                setattr(f, 'prompt', new_text)
                self._push_facet_property_command(f, 'prompt', old_text, new_text)
                # Don't update baseline - let commands merge until focus lost

            prompt_edit.textChanged.connect(on_prompt_changed)
            props_layout.addRow("Prompt:", prompt_edit)

        # CharmNetworkFacet
        elif facet.facet_type == "CharmNetworkFacet":
            info_label = QLabel("Neural affect model (PAD + boredom + sorrow)")
            info_label.setStyleSheet("color: #888888; font-style: italic;")
            props_layout.addRow(info_label)

        # ScriptedFacet (with undo support)
        elif facet.facet_type == "ScriptedFacet":
            script_edit = ClickableTextEdit(
                field_name=f"{facet.name} - Salience Script",
                on_apply_callback=lambda text, f=facet: (
                    setattr(f, 'salience_script', text),
                    self._push_facet_property_command(f, 'salience_script', f.salience_script, text)
                )
            )
            script_edit.setPlainText(facet.salience_script or "")
            script_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px; font-family: 'Courier New';")
            script_edit.setMaximumHeight(150)
            script_edit._baseline_value = facet.salience_script or ""  # Track baseline for undo

            def on_script_changed(edit=script_edit, f=facet):
                new_text = edit.toPlainText()
                old_text = getattr(edit, '_baseline_value', '')
                setattr(f, 'salience_script', new_text)
                self._push_facet_property_command(f, 'salience_script', old_text, new_text)

            script_edit.textChanged.connect(on_script_changed)
            props_layout.addRow("Salience Script:", script_edit)

        # Generic
        else:
            type_label = QLabel(f"Type: {facet.facet_type}")
            type_label.setStyleSheet("color: #888888;")
            props_layout.addRow(type_label)

        self.facet_properties_layout.addWidget(props_widget)

        # Add collapsible section for Last Execution Data (debugging feature)
        self._add_execution_data_section(facet)

        print(f"[Inspector] Facet properties loaded for: {facet.name}")

    def _add_execution_data_section(self, facet):
        """
        Add collapsible section showing captured inputs/outputs from last execution.

        Gets data from FacetNodeGraphics in facets_editor panel.
        """
        from PyQt6.QtWidgets import QPlainTextEdit
        import json

        try:
            # Find facets editor to get node graphics
            main_window = self.window()
            if not main_window or not hasattr(main_window, 'facets_editor'):
                return

            facets_editor = main_window.facets_editor
            if not facets_editor:
                return

            # Check if node_graphics exists and has this facet
            if not hasattr(facets_editor, 'node_graphics') or not facets_editor.node_graphics:
                return

            if facet.id not in facets_editor.node_graphics:
                return

            node = facets_editor.node_graphics[facet.id]
            if not node:
                return

            # Only show if there's data to display
            has_data = (
                (hasattr(node, 'last_inputs') and node.last_inputs) or
                (hasattr(node, 'last_outputs') and node.last_outputs) or
                (hasattr(node, 'active_cycles') and node.active_cycles)
            )
            if not has_data:
                return
        except Exception as e:
            print(f"[Inspector] Error checking execution data: {e}")
            return

        try:
            # Create collapsible section
            exec_section = CollapsibleSection("Last Execution Data")
            exec_section.setCollapsed(True)  # Start collapsed

            def format_value(val):
                if val is None:
                    return "(none)"
                if isinstance(val, str):
                    # Truncate long strings for display
                    if len(val) > 500:
                        return val[:500] + "..."
                    return val
                try:
                    return json.dumps(val, indent=2, default=str)
                except:
                    return str(val)

            # Show inputs
            if hasattr(node, 'last_inputs') and node.last_inputs:
                inputs_label = QLabel("Inputs:")
                inputs_label.setStyleSheet("color: #888888; font-weight: bold; margin-top: 4px;")
                exec_section.content.layout().addRow(inputs_label)

                for key, value in node.last_inputs.items():
                    value_edit = QPlainTextEdit()
                    value_edit.setReadOnly(True)
                    value_edit.setPlainText(format_value(value))
                    value_edit.setStyleSheet("""
                        QPlainTextEdit {
                            background-color: #1A1A1A;
                            color: #AAAAAA;
                            border: 1px solid #333333;
                            font-family: 'Monaco', 'Consolas', monospace;
                            font-size: 10px;
                        }
                    """)
                    value_edit.setMaximumHeight(80)
                    exec_section.content.layout().addRow(f"  {key}:", value_edit)

            # Show outputs
            if hasattr(node, 'last_outputs') and node.last_outputs:
                outputs_label = QLabel("Outputs:")
                outputs_label.setStyleSheet("color: #888888; font-weight: bold; margin-top: 8px;")
                exec_section.content.layout().addRow(outputs_label)

                for key, value in node.last_outputs.items():
                    value_edit = QPlainTextEdit()
                    value_edit.setReadOnly(True)
                    value_edit.setPlainText(format_value(value))
                    value_edit.setStyleSheet("""
                        QPlainTextEdit {
                            background-color: #1A1A1A;
                            color: #AAAAAA;
                            border: 1px solid #333333;
                            font-family: 'Monaco', 'Consolas', monospace;
                            font-size: 10px;
                        }
                    """)
                    value_edit.setMaximumHeight(80)
                    exec_section.content.layout().addRow(f"  {key}:", value_edit)

            # Show active cycles count
            if hasattr(node, 'active_cycles') and node.active_cycles:
                active_label = QLabel(f"Active Cycles: {len(node.active_cycles)}")
                active_label.setStyleSheet("color: #FFD700; margin-top: 8px;")
                exec_section.content.layout().addRow(active_label)

            self.facet_properties_layout.addWidget(exec_section)

        except Exception as e:
            print(f"[Inspector] Error building execution data section: {e}")
            import traceback
            traceback.print_exc()

    def _clear_facet_properties_inline(self):
        """Clear facet properties from the inline container."""
        try:
            if not hasattr(self, 'facet_properties_layout') or not self.facet_properties_layout:
                return
            while self.facet_properties_layout.count():
                item = self.facet_properties_layout.takeAt(0)
                if item and item.widget():
                    try:
                        item.widget().deleteLater()
                    except RuntimeError:
                        pass  # Widget already deleted
        except Exception as e:
            print(f"[Inspector] Error clearing facet properties: {e}")

    def _add_facets_section(self, agent_id: str, agent_data: dict):
        """
        Add FACETS section showing all facets as collapsible sections.

        This creates the Unity component-style list where each facet
        is a collapsible section that can be expanded to edit properties.

        Args:
            agent_id: Agent ID (UUID)
            agent_data: Agent data dict with 'name' field
        """
        from PyQt6.QtWidgets import QFrame

        # Load the agent's facet assembly
        assembly = self._get_agent_assembly(agent_id, agent_data)
        if not assembly:
            return

        # Add horizontal separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555555; max-height: 2px;")
        self.properties_layout.addWidget(separator)

        # Add "FACETS" header label
        facets_header = QLabel("FACETS")
        facets_header.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        facets_header.setStyleSheet("color: #888888; padding: 8px 4px 4px 4px;")
        self.properties_layout.addWidget(facets_header)

        # Create collapsible section for each facet
        for facet in assembly.facets:
            facet_section = self._create_facet_section(facet, agent_id)
            self.properties_layout.addWidget(facet_section)

    def _create_facet_section(self, facet, agent_id: str):
        """
        Create a CollapsibleSection for a single facet using PropertyBinding.

        Uses the PropertyRegistry to auto-generate widgets with undo support.
        Works with any facet type, including dynamically registered ones.

        Args:
            facet: Facet object from facet_system
            agent_id: Agent ID for saving changes

        Returns:
            CollapsibleSection widget
        """
        # Create collapsible section with facet name
        section = CollapsibleSection(facet.name)
        section_form = QFormLayout()

        # Store facet reference for later expansion
        section.setProperty("facet_id", facet.id)

        # Get properties for this facet type from registry
        props = property_registry.get_properties(facet.facet_type, include_base=False)

        if props:
            # Auto-generate widgets with undo support
            for prop_name, meta in props.items():
                widget = self.create_widget_for_property(facet, meta)
                section_form.addRow(f"{meta.display_name}:", widget)

            # Add template variables hint for prompt fields
            if 'prompt' in props:
                vars_label = QLabel(
                    "Variables: {incoming_data}, {observations}, "
                    "{affect_valence:.2f}, {affect_arousal:.2f}, {affect_dominance:.2f}"
                )
                vars_label.setStyleSheet("color: #666666; font-size: 9px; padding: 2px;")
                vars_label.setWordWrap(True)
                section_form.addRow("", vars_label)
        else:
            # No properties registered - show type info
            type_label = QLabel(f"Type: {facet.facet_type}")
            type_label.setStyleSheet("color: #888888;")
            section_form.addRow(type_label)

        section.set_content_layout(section_form)

        # Restore collapsed state (all start collapsed by default)
        self._restore_collapsible_state(section)

        return section

    @pyqtSlot(str, dict)
    def load_entity(self, entity_type: str, entity_data: dict):
        """Load entity properties into inspector."""
        # Save for later restore when facet is deselected
        if entity_type and entity_data:
            self.last_entity_type = entity_type
            self.last_entity_data = entity_data

        # NEW BEHAVIOR (Unity Component Model):
        # Always honor hierarchy selections - load agent with facets
        # Facet selection just expands/collapses sections, doesn't block hierarchy
        # (Old behavior blocked hierarchy when facet was selected)

        # CRITICAL: Prevent re-entrant loading (e.g., double-tap events)
        if self.is_loading:
            print(f"[DIAGNOSTIC] BLOCKING re-entrant load_entity call (is_loading=True)")
            return

        # Handle deselection (nothing selected)
        # Empty string or empty dict means nothing selected
        if not entity_type or not entity_data:
            self.clear_inspector()
            return

        # DIAGNOSTIC: Track ALL load_entity calls
        import traceback
        print(f"\n{'#'*80}")
        print(f"[DIAGNOSTIC] load_entity() called")
        print(f"[DIAGNOSTIC] entity_type={entity_type}, entity_id={entity_data.get('id', 'unknown')}")
        print(f"[DIAGNOSTIC] is_saving={self.is_saving}, is_loading={self.is_loading}")
        focused_widget = QApplication.focusWidget()
        print(f"[DIAGNOSTIC] focused_widget={focused_widget} (type: {type(focused_widget).__name__ if focused_widget else 'None'})")
        print(f"[DIAGNOSTIC] Call stack:")
        print(''.join(traceback.format_stack()[-8:-1]))
        print(f"{'#'*80}\n")

        # CRITICAL: Check if same entity - don't reload if it hasn't changed
        if self.current_entity:
            old_type, old_data = self.current_entity
            old_id = old_data.get('id') if old_data else None
            new_id = entity_data.get('id')
            if old_type == entity_type and old_id == new_id:
                print(f"[DIAGNOSTIC] SKIPPING load_entity - same entity already loaded (no flash!)")
                return

        self.current_entity = (entity_type, entity_data)

        # CRITICAL: Don't reload if a text widget has focus (user is editing)
        if focused_widget and (isinstance(focused_widget, QLineEdit) or isinstance(focused_widget, QTextEdit)):
            # User is actively editing - skip reload to preserve their changes
            print(f"[DIAGNOSTIC] SKIPPING load_entity - text widget has focus")
            return

        # CRITICAL: Don't reload if save is in progress
        if self.is_saving:
            print(f"[DIAGNOSTIC] SKIPPING load_entity - save in progress")
            return

        print(f"[DIAGNOSTIC] PROCEEDING with load_entity - will destroy all widgets")

        # Set loading flag to prevent re-entrance
        self.is_loading = True

        try:
            # CRITICAL: Save CollapsibleSection expanded state before destroying widgets
            self._save_collapsible_states()

            # Clear existing properties
            while self.properties_layout.count():
                child = self.properties_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Clear component widget tracking for clean slate
            self.component_widgets.clear()

            # Update header
            if entity_type == 'noodling':
                name = entity_data.get('data', {}).get('name', entity_data.get('id'))
                species = entity_data.get('data', {}).get('species', 'unknown')
                self.entity_header.setText(f"Noodling: {name} ({species})")
                self.load_noodling_properties(entity_data)

            elif entity_type == 'user':
                self.entity_header.setText("User: caity")
                self.load_user_properties(entity_data)

            elif entity_type in ('prim', 'object', 'prop'):
                obj_name = entity_data.get('name') or entity_data.get('id', 'Unknown Object').replace('obj_', '').replace('_', ' ').title()
                label = "Prop" if entity_type == 'prop' else "Prim"
                self.entity_header.setText(f"{label}: {obj_name}")
                self.load_object_properties(entity_data)

            elif entity_type == 'exit':
                direction = entity_data.get('direction', 'unknown')
                self.entity_header.setText(f"Exit: {direction}")
                self.load_exit_properties(entity_data)

            elif entity_type == 'stage':
                stage_name = entity_data.get('data', {}).get('name', 'Unknown Stage')
                self.entity_header.setText(f"Stage: {stage_name}")
                self.load_stage_properties(entity_data)

            elif entity_type == 'zone':
                zone_name = entity_data.get('name', 'Unknown Zone')
                self.entity_header.setText(f"Zone: {zone_name}")
                self.load_zone_properties(entity_data)

            elif entity_type == 'neural_node':
                node_name = entity_data.get('name', 'Unknown Node')
                node_type = entity_data.get('type', 'UNKNOWN')
                self.entity_header.setText(f"Node: {node_name}")
                self.load_neural_node_properties(entity_data)

            elif entity_type == 'radiance':
                asset_name = entity_data.get('name', 'Unknown Asset')
                self.entity_header.setText(f"Radiance: {asset_name}")
                self.load_radiance_properties(entity_data)

        finally:
            # ALWAYS clear loading flag, even on error
            self.is_loading = False
            print(f"[DIAGNOSTIC] load_entity completed, is_loading cleared")

    def load_stage_properties(self, entity_data):
        """Show Stage properties (room metadata)."""
        stage = entity_data.get('data', {})
        stage_id = entity_data.get('id', '')

        # Basic Info Component
        basic_group = self.create_property_group("Basic Info")
        self.add_text_field(basic_group, "Name", stage.get('name', ''))
        self.add_text_field(basic_group, "Stage ID", stage_id)
        self.properties_layout.addWidget(basic_group)

        # Description Component
        desc_group = self.create_property_group("Description")
        desc_text = QTextEdit(stage.get('description', ''))
        desc_text.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        desc_text.setMaximumHeight(100)
        desc_text.setTabChangesFocus(True)  # TAB moves focus instead of inserting tab
        # Store reference to stage_id for auto-save
        desc_text.setProperty("stage_id", stage_id)
        # Auto-save on text change
        desc_text.textChanged.connect(lambda: self.save_stage_description(desc_text))
        # Install event filter for RETURN key handling
        desc_text.installEventFilter(self)
        desc_group.content.layout().addRow("Description:", desc_text)
        self.properties_layout.addWidget(desc_group)

        # Exits Component
        exits_group = self.create_property_group("Exits")
        exits = stage.get('exits', {})
        if exits:
            for direction, dest_id in exits.items():
                exit_label = QLabel(f"{direction} → {dest_id}")
                exit_label.setStyleSheet("color: #D2D2D2; padding: 4px;")
                exits_group.content.layout().addRow(exit_label)
        else:
            no_exits = QLabel("No exits defined")
            no_exits.setStyleSheet("color: #888; padding: 4px;")
            exits_group.content.layout().addRow(no_exits)
        self.properties_layout.addWidget(exits_group)

        # Occupants Component (read-only)
        occupants_group = self.create_property_group("Occupants")
        occupants = stage.get('occupants', [])
        if occupants:
            for occ_id in occupants:
                occ_label = QLabel(occ_id)
                occ_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
                occupants_group.content.layout().addRow(occ_label)
        else:
            no_occ = QLabel("No occupants")
            no_occ.setStyleSheet("color: #888; padding: 4px;")
            occupants_group.content.layout().addRow(no_occ)
        self.properties_layout.addWidget(occupants_group)

        self.properties_layout.addStretch()

    def load_zone_properties(self, zone_data):
        """Show Zone properties from Spatial View."""
        zone_id = zone_data.get('id', '')
        zone_name = zone_data.get('name', zone_id)
        file_path = zone_data.get('file_path', '')

        # Spatial Properties - compact layout
        spatial_group = self.create_property_group("Spatial")
        center = zone_data.get('center', [0, 0, 0])
        self.add_vector3_field(spatial_group, "Center", center)

        # Radius/Falloff on one row
        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(8)

        radius_field = QDoubleSpinBox()
        radius_field.setRange(0.1, 9999)
        radius_field.setDecimals(1)
        radius_field.setValue(float(zone_data.get('radius', 10)))
        radius_field.setReadOnly(True)
        radius_field.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        radius_field.setFixedWidth(60)
        radius_field.setStyleSheet("background-color: #1E1E1E; color: #888; border: 1px solid #3A3A3A; padding: 2px;")

        falloff_field = QDoubleSpinBox()
        falloff_field.setRange(0, 9999)
        falloff_field.setDecimals(1)
        falloff_field.setValue(float(zone_data.get('falloff', 5)))
        falloff_field.setReadOnly(True)
        falloff_field.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        falloff_field.setFixedWidth(60)
        falloff_field.setStyleSheet("background-color: #1E1E1E; color: #888; border: 1px solid #3A3A3A; padding: 2px;")

        size_layout.addWidget(QLabel("R:"))
        size_layout.addWidget(radius_field)
        size_layout.addWidget(QLabel("Fall:"))
        size_layout.addWidget(falloff_field)
        size_layout.addStretch()
        spatial_group.content.layout().addRow(size_row)

        shape_label = QLabel(zone_data.get('shape', 'sphere'))
        shape_label.setStyleSheet("color: #888; padding: 2px;")
        spatial_group.content.layout().addRow("Shape:", shape_label)
        self.properties_layout.addWidget(spatial_group)

        # Description
        desc_group = self.create_property_group("Description")
        description = zone_data.get('description', '')
        desc_text = QTextEdit(description)
        desc_text.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        desc_text.setMaximumHeight(120)
        desc_text.setReadOnly(True)  # Read-only for now
        desc_group.content.layout().addRow(desc_text)
        self.properties_layout.addWidget(desc_group)

        # Exits/Connections
        exits_group = self.create_property_group("Connections")
        exits = zone_data.get('exits', {})
        if exits:
            for direction, dest_id in exits.items():
                exit_label = QLabel(f"{direction} -> {dest_id}")
                exit_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
                exits_group.content.layout().addRow(exit_label)
        else:
            no_exits = QLabel("No connections")
            no_exits.setStyleSheet("color: #888; padding: 4px;")
            exits_group.content.layout().addRow(no_exits)
        self.properties_layout.addWidget(exits_group)

        # Perception
        perception = zone_data.get('perception', {})
        if perception:
            perc_group = self.create_property_group("Perception")
            self.add_text_field(perc_group, "Visibility", str(perception.get('visibility', 20)))
            self.add_text_field(perc_group, "Audibility", str(perception.get('audibility', 20)))
            self.add_text_field(perc_group, "Lighting", str(perception.get('lighting', 'natural')))
            self.properties_layout.addWidget(perc_group)

        # Ambient
        ambient = zone_data.get('ambient', {})
        if ambient:
            amb_group = self.create_property_group("Ambient")
            sounds = ambient.get('sounds', [])
            self.add_text_field(amb_group, "Sounds", ', '.join(sounds) if sounds else '(none)')
            self.add_text_field(amb_group, "Mood", str(ambient.get('mood', 'neutral')))
            self.add_text_field(amb_group, "Temperature", str(ambient.get('temperature', 'pleasant')))
            self.properties_layout.addWidget(amb_group)

        # File Info
        if file_path:
            file_group = self.create_property_group("File")
            file_label = QLabel(file_path)
            file_label.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
            file_label.setWordWrap(True)
            file_group.content.layout().addRow(file_label)
            self.properties_layout.addWidget(file_group)

        self.properties_layout.addStretch()

    def load_neural_node_properties(self, entity_data):
        """
        Show Neural Canvas node properties for editing.

        Allows editing node params including comment text.
        Changes are synced back to the canvas graph.
        """
        node_id = entity_data.get('id', '')
        node_type = entity_data.get('type', 'UNKNOWN')
        node_name = entity_data.get('name', 'Unknown')
        params = entity_data.get('params', {})
        weights = entity_data.get('weights', {})
        inputs = entity_data.get('inputs', {})
        outputs = entity_data.get('outputs', {})
        position = entity_data.get('position', (0, 0))
        description = entity_data.get('description', '')

        # Store node_id for save operations
        self._current_neural_node_id = node_id

        # Track param widgets for live updates from canvas
        self._neural_node_param_widgets = {}

        # Basic Info Component
        basic_group = self.create_property_group("Basic Info")

        # Name (editable)
        name_field = QLineEdit(node_name)
        name_field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        name_field.editingFinished.connect(lambda: self._save_neural_node_field('name', name_field.text()))
        basic_group.content.layout().addRow("Name:", name_field)

        # Type (read-only)
        type_label = QLabel(node_type)
        type_label.setStyleSheet("color: #888; padding: 4px;")
        basic_group.content.layout().addRow("Type:", type_label)

        # ID (read-only)
        id_label = QLabel(node_id[:16] + "..." if len(node_id) > 16 else node_id)
        id_label.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        id_label.setToolTip(node_id)
        basic_group.content.layout().addRow("ID:", id_label)

        # Position (read-only)
        pos_label = QLabel(f"({position[0]}, {position[1]})")
        pos_label.setStyleSheet("color: #888; padding: 4px;")
        basic_group.content.layout().addRow("Position:", pos_label)

        self.properties_layout.addWidget(basic_group)

        # Parameters Component (editable)
        if params:
            params_group = self.create_property_group("Parameters")

            for param_name, param_value in params.items():
                if param_name == 'text':
                    # Multi-line text field for COMMENT nodes
                    text_edit = ClickableTextEdit(
                        field_name=param_name,
                        on_apply_callback=lambda val, pn=param_name: self._save_neural_node_param(pn, val)
                    )
                    text_edit.setPlainText(str(param_value))
                    text_edit.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
                    text_edit.setMinimumHeight(150)
                    text_edit.setMaximumHeight(300)
                    text_edit.textChanged.connect(
                        lambda te=text_edit, pn=param_name: self._save_neural_node_param(pn, te.toPlainText())
                    )
                    params_group.content.layout().addRow("Text:", text_edit)
                    self._neural_node_param_widgets[param_name] = text_edit

                elif isinstance(param_value, bool):
                    # Boolean as checkbox
                    from PyQt6.QtWidgets import QCheckBox
                    checkbox = QCheckBox()
                    checkbox.setChecked(param_value)
                    checkbox.stateChanged.connect(
                        lambda state, pn=param_name: self._save_neural_node_param(pn, state == 2)
                    )
                    # Prettier labels for known params
                    label_map = {
                        'show_on_start': 'Show on Start',
                    }
                    label = label_map.get(param_name, param_name)
                    params_group.content.layout().addRow(f"{label}:", checkbox)
                    self._neural_node_param_widgets[param_name] = checkbox

                elif isinstance(param_value, (int, float)):
                    # Numeric field
                    if isinstance(param_value, int):
                        spin = QSpinBox()
                        spin.setRange(-99999, 99999)
                        spin.setValue(param_value)
                        spin.valueChanged.connect(
                            lambda val, pn=param_name: self._save_neural_node_param(pn, val)
                        )
                    else:
                        spin = QDoubleSpinBox()
                        spin.setRange(-99999.0, 99999.0)
                        spin.setDecimals(3)
                        spin.setValue(param_value)
                        spin.valueChanged.connect(
                            lambda val, pn=param_name: self._save_neural_node_param(pn, val)
                        )
                    spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 2px;")
                    params_group.content.layout().addRow(f"{param_name}:", spin)
                    self._neural_node_param_widgets[param_name] = spin

                else:
                    # String field
                    field = QLineEdit(str(param_value))
                    field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
                    field.editingFinished.connect(
                        lambda f=field, pn=param_name: self._save_neural_node_param(pn, f.text())
                    )
                    params_group.content.layout().addRow(f"{param_name}:", field)
                    self._neural_node_param_widgets[param_name] = field

            self.properties_layout.addWidget(params_group)

        # Weights Component (read-only)
        if weights:
            weights_group = self.create_property_group("Weights")
            total_params = 0
            for weight_name, weight_info in weights.items():
                shape = weight_info.get('shape', [])
                trainable = weight_info.get('trainable', True)
                num_params = weight_info.get('num_params', 0)
                total_params += num_params

                shape_str = "x".join(str(d) for d in shape)
                trainable_str = " (trainable)" if trainable else " (frozen)"
                label = QLabel(f"{weight_name}: [{shape_str}] = {num_params} params{trainable_str}")
                label.setStyleSheet("color: #888; padding: 2px;")
                weights_group.content.layout().addRow(label)

            total_label = QLabel(f"Total: {total_params} parameters")
            total_label.setStyleSheet("color: #D2D2D2; font-weight: bold; padding: 4px;")
            weights_group.content.layout().addRow(total_label)
            self.properties_layout.addWidget(weights_group)

        # Inputs/Outputs Component (read-only)
        if inputs or outputs:
            ports_group = self.create_property_group("Ports")

            if inputs:
                inputs_label = QLabel("Inputs:")
                inputs_label.setStyleSheet("color: #D2D2D2; font-weight: bold; padding: 2px;")
                ports_group.content.layout().addRow(inputs_label)
                for port_name, port_info in inputs.items():
                    port_label = QLabel(f"  {port_name}: {port_info}")
                    port_label.setStyleSheet("color: #888; padding: 2px;")
                    ports_group.content.layout().addRow(port_label)

            if outputs:
                outputs_label = QLabel("Outputs:")
                outputs_label.setStyleSheet("color: #D2D2D2; font-weight: bold; padding: 2px;")
                ports_group.content.layout().addRow(outputs_label)
                for port_name, port_info in outputs.items():
                    port_label = QLabel(f"  {port_name}: {port_info}")
                    port_label.setStyleSheet("color: #888; padding: 2px;")
                    ports_group.content.layout().addRow(port_label)

            self.properties_layout.addWidget(ports_group)

        # Description (if present)
        if description:
            desc_group = self.create_property_group("Description")
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888; padding: 4px;")
            desc_label.setWordWrap(True)
            desc_group.content.layout().addRow(desc_label)
            self.properties_layout.addWidget(desc_group)

        self.properties_layout.addStretch()

    def _save_neural_node_field(self, field_name: str, value):
        """Save a basic field (name) to the neural node with undo support."""
        if not hasattr(self, '_current_neural_node_id'):
            return

        node_id = self._current_neural_node_id

        # Get main window reference
        main_window = self.window()
        if not hasattr(main_window, 'neural_canvas'):
            return

        canvas_view = main_window.neural_canvas.canvas_view
        node = main_window.neural_canvas.graph.get_node_by_id(node_id)
        if not node:
            return

        if field_name == 'name':
            old_name = node.name
            # Skip if value hasn't changed
            if old_name == value:
                return

            # Push undo command
            cmd = RenameNeuralNodeCommand(
                view=canvas_view,
                node_id=node_id,
                old_name=old_name,
                new_name=value
            )
            UndoManager.instance().push(cmd)
            print(f"[Inspector] Neural node name updated: {old_name} -> {value}")

    def _save_neural_node_param(self, param_name: str, value):
        """Save a parameter value to the neural node with undo support."""
        # Guard against re-entrant calls (e.g., signal feedback loops)
        if getattr(self, '_saving_neural_param', False):
            return
        self._saving_neural_param = True

        try:
            if not hasattr(self, '_current_neural_node_id'):
                return  # finally will reset _saving_neural_param

            node_id = self._current_neural_node_id

            # Get main window reference
            main_window = self.window()
            if not hasattr(main_window, 'neural_canvas'):
                return  # finally will reset _saving_neural_param

            canvas_view = main_window.neural_canvas.canvas_view
            node = main_window.neural_canvas.graph.get_node_by_id(node_id)
            if not node:
                return  # finally will reset _saving_neural_param

            # Get old value
            old_value = node.params.get(param_name)

            # Skip if value hasn't changed
            if old_value == value:
                return  # finally will reset _saving_neural_param

            # Special handling for show_on_start - only one COMMENT can have it
            # Note: This is a side effect that won't be undone, but it's minor
            if param_name == 'show_on_start' and value is True:
                from ..core.neural_canvas.neural_node import NodeType
                # Uncheck all other COMMENT nodes
                for other_id, other_node in main_window.neural_canvas.graph.nodes.items():
                    if other_id != node_id and other_node.type == NodeType.COMMENT:
                        if other_node.params.get('show_on_start', False):
                            other_node.params['show_on_start'] = False

            # Push undo command
            cmd = EditNeuralNodeParamCommand(
                view=canvas_view,
                node_id=node_id,
                param_name=param_name,
                old_value=old_value,
                new_value=value,
                node_name=node.name
            )
            UndoManager.instance().push(cmd)
            print(f"[Inspector] Neural node param '{param_name}' updated")
        except Exception as e:
            import traceback
            print(f"[Inspector] ERROR in _save_neural_node_param: {e}")
            traceback.print_exc()
        finally:
            self._saving_neural_param = False

    def update_neural_node_param(self, param_name: str, new_value):
        """
        Update a displayed param value in the Inspector (called externally).

        Used when canvas slider changes to sync the Inspector's widget.
        """
        if not hasattr(self, '_neural_node_param_widgets'):
            return

        widget = self._neural_node_param_widgets.get(param_name)
        if not widget:
            return

        # Block signals to prevent feedback loop
        widget.blockSignals(True)
        try:
            from PyQt6.QtWidgets import QCheckBox
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(new_value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(new_value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(new_value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(new_value))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(new_value))
        finally:
            widget.blockSignals(False)

    def _save_collapsible_states(self):
        """
        Save expanded/collapsed state of all CollapsibleSections before widget rebuild.

        Pattern copied from SceneHierarchy.save_expanded_state() to prevent
        bounce-back when timer refreshes Inspector.
        """
        # Find all CollapsibleSection widgets in the properties layout
        for i in range(self.properties_layout.count()):
            widget = self.properties_layout.itemAt(i).widget()
            if isinstance(widget, CollapsibleSection):
                self.collapsible_expanded_state[widget.title_text] = widget.is_expanded
                print(f"[STATE] Saved '{widget.title_text}': expanded={widget.is_expanded}")

    def _restore_collapsible_state(self, section: CollapsibleSection):
        """
        Restore expanded state for a newly-created CollapsibleSection.

        Called immediately after creating CollapsibleSection to restore user's
        previous expanded/collapsed preference.

        Args:
            section: The CollapsibleSection widget to restore state for
        """
        if section.title_text in self.collapsible_expanded_state:
            saved_state = self.collapsible_expanded_state[section.title_text]
            section.set_expanded(saved_state)
            print(f"[STATE] Restored '{section.title_text}': expanded={saved_state}")

    def _on_collapsible_toggled(self, title: str, expanded: bool):
        """
        Handle CollapsibleSection toggle - update state tracking.

        Called whenever user expands/collapses a section. Ensures state
        is preserved across Inspector refreshes.

        Args:
            title: Section title (identifier)
            expanded: New expanded state
        """
        self.collapsible_expanded_state[title] = expanded
        print(f"[STATE] User toggled '{title}': expanded={expanded}")

    def eventFilter(self, obj, event):
        """Handle keyboard events for text fields."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if event.type() == QEvent.Type.KeyPress:
            if isinstance(obj, QTextEdit):
                # TAB = save and clear focus
                if event.key() == Qt.Key.Key_Tab:
                    obj.clearFocus()  # Move focus away (triggers save)
                    return True  # Event handled, don't insert tab
                # RETURN (without shift) = save and clear focus
                if event.key() == Qt.Key.Key_Return and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    obj.clearFocus()  # Move focus away (triggers save)
                    return True  # Event handled, don't insert newline
                # SHIFT+RETURN = insert newline (default behavior)

        return super().eventFilter(obj, event)

    def save_stage_description(self, text_widget: QTextEdit):
        """Auto-save stage description via API (updates both file and in-memory state)."""
        stage_id = text_widget.property("stage_id")
        new_description = text_widget.toPlainText()

        try:
            # Use API to update both file and in-memory dict
            url = f"{self.api_base}/rooms/{stage_id}/update"
            payload = {'description': new_description}

            response = requests.post(url, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"Stage description saved for {stage_id}")
            else:
                print(f"Error saving stage description: {response.json().get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error saving stage description: {e}")

    def load_noodling_properties(self, entity_data):
        """Show Noodling properties - unified inspector view."""
        if not entity_data:
            print("[Inspector] ERROR: entity_data is None or empty")
            return

        agent = entity_data.get('data', {})
        agent_id = entity_data.get('id', '')

        if not agent_id:
            print(f"[Inspector] ERROR: No agent_id in entity_data: {entity_data}")
            return

        # Store for facet dropdown updates
        self.current_agent_id = agent_id
        self.property_fields = {}

        # Load full recipe data from YAML file
        recipe_data = {}
        try:
            import os
            recipe_name = agent.get('name', agent_id.replace('agent_', ''))
            recipe_path = os.path.join(
                os.path.dirname(__file__),
                '../../../cmush/recipes',
                f'{recipe_name}.yaml'
            )
            if os.path.exists(recipe_path):
                with open(recipe_path, 'r') as f:
                    recipe_data = yaml.safe_load(f)
                    print(f"[Inspector] Loaded recipe from: {recipe_path}")
            else:
                print(f"[Inspector] Recipe not found: {recipe_path}")
        except Exception as e:
            print(f"[Inspector] Error loading recipe: {e}")
            import traceback
            traceback.print_exc()

        # ===== AGENT BASICS (always visible) =====
        basics_group = self.create_property_group("Noodling")

        # Name (editable)
        self.property_fields['name'] = self.add_text_field(
            basics_group, "Name",
            recipe_data.get('name', agent.get('name', ''))
        )

        # UUID (read-only) - simple label on same row
        # Use short display format but copy full ID
        display_id = agent_id.replace('agent_', '')
        if len(display_id) > 20:
            display_id = display_id[:8] + "..." + display_id[-4:]

        uuid_widget = QLabel(f'<span style="color:#888888">{display_id}</span> '
                            f'<a href="copy" style="color:#666666;text-decoration:none">[copy]</a>')
        uuid_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        uuid_widget.linkActivated.connect(lambda: QApplication.clipboard().setText(agent_id))
        uuid_widget.setToolTip(f"Full ID: {agent_id}\nClick [copy] to copy")
        basics_group.content.layout().addRow("UUID:", uuid_widget)

        # Description (editable text area)
        description = recipe_data.get('description', agent.get('description', 'An empty noodling...'))
        print(f"[Inspector] Description loaded: {description[:50]}..." if description else "[Inspector] No description found")
        self.property_fields['description'] = self.add_text_area(basics_group, "Description", description)

        self.properties_layout.addWidget(basics_group)

        # ===== FACET DROPDOWN SELECTOR =====
        print(f"[Inspector] load_noodling_properties: About to call _add_facet_dropdown_selector...")
        try:
            # Pass entity_data which has 'name' at top level (from hierarchy)
            # NOT agent which is entity_data['data']
            self._add_facet_dropdown_selector(agent_id, entity_data)
            print(f"[Inspector] load_noodling_properties: _add_facet_dropdown_selector returned successfully")
        except Exception as e:
            print(f"[Inspector] ERROR creating facet dropdown: {e}")
            import traceback
            traceback.print_exc()

        print(f"[Inspector] load_noodling_properties: Adding stretch...")
        self.properties_layout.addStretch()

        print(f"[Inspector] ========================================")
        print(f"[Inspector] load_noodling_properties COMPLETE")
        print(f"[Inspector] Final layout widget count: {self.properties_layout.count()}")
        print(f"[Inspector] facet_dropdown exists: {hasattr(self, 'facet_dropdown')}")
        print(f"[Inspector] ========================================")

    def load_user_properties(self, entity_data):
        """Show user properties."""
        self.property_fields = {}

        user_group = self.create_property_group("User Info")
        self.property_fields['username'] = self.add_text_field(user_group, "Username", "caity")
        self.property_fields['type'] = self.add_text_field(user_group, "Type", "Noodler (human)")
        self.property_fields['age'] = self.add_text_field(user_group, "Age", "9 years old")
        self.property_fields['pronouns'] = self.add_text_field(user_group, "Pronouns", "she/her")
        self.properties_layout.addWidget(user_group)

        # Description
        desc_group = self.create_property_group("Description")
        desc_text = ("A nine-year-old girl in worn overalls with a shock of wild, curly brown hair "
                    "and sparkling blue eyes. She has a wooden sword hanging from one of her belt loops "
                    "and she's sucking a glowing, heart-shaped red hot atomic fireball candy.")
        self.property_fields['description'] = self.add_text_area(desc_group, "Description", desc_text)
        self.properties_layout.addWidget(desc_group)

        inventory_group = self.create_property_group("Inventory")
        self.property_fields['item1'] = self.add_text_field(inventory_group, "Item 1", "Wooden sword")
        self.property_fields['item2'] = self.add_text_field(inventory_group, "Item 2", "Atomic fireball candy")
        self.properties_layout.addWidget(inventory_group)

        self.properties_layout.addStretch()

    def load_object_properties(self, entity_data):
        """Show object properties including physics settings."""
        # Clear previous entity's fields
        self.property_fields = {}

        obj_id = entity_data.get('id', '')
        obj_data = entity_data.get('data', {})

        # Basic properties
        obj_group = self.create_property_group("Object Properties")
        self.add_text_field(obj_group, "ID", obj_id, read_only=True)  # UUID is immutable
        self.property_fields['name'] = self.add_text_field(obj_group, "Name", obj_data.get('name', 'Unnamed'))
        self.property_fields['description'] = self.add_text_area(obj_group, "Description", obj_data.get('description', 'An object in the world.'))
        self.properties_layout.addWidget(obj_group)

        # Physics Properties section
        physics_group = self.create_property_group("Physics (SPE)")
        self._create_physics_properties(physics_group, obj_data)
        self.properties_layout.addWidget(physics_group)

        # Arbitrary metadata editor
        metadata_component = self.create_metadata_component(obj_id)
        self.properties_layout.addWidget(metadata_component)

        self.properties_layout.addStretch()

    def _create_physics_properties(self, group, obj_data):
        """Create physics property dropdowns for a prim."""
        # Import material presets
        try:
            from noodlestudio.core.semantic_world import MATERIAL_PRESETS
        except ImportError:
            MATERIAL_PRESETS = {}

        # Current values (with defaults)
        current_material = obj_data.get('material', 'unknown')
        current_mass = obj_data.get('mass', 'medium')
        current_friction = obj_data.get('friction', 'medium')
        current_elasticity = obj_data.get('elasticity', 'normal')
        current_softness = obj_data.get('softness', 'normal')

        # Material preset dropdown - sorted alphabetically with "(custom)" at top
        material_options = ["(custom)"] + sorted(MATERIAL_PRESETS.keys())

        def on_material_preset_change(material):
            """Apply material preset to all physics fields."""
            if material == "(custom)":
                return  # Don't change anything for custom

            preset = MATERIAL_PRESETS.get(material, {})
            if not preset:
                return

            # Update the dropdowns to reflect preset values
            if 'mass' in self.property_fields and 'mass' in preset:
                self.property_fields['mass'].setCurrentText(preset['mass'])
            if 'friction' in self.property_fields and 'friction' in preset:
                self.property_fields['friction'].setCurrentText(preset['friction'])
            if 'elasticity' in self.property_fields and 'elasticity' in preset:
                self.property_fields['elasticity'].setCurrentText(preset['elasticity'])
            if 'softness' in self.property_fields and 'softness' in preset:
                self.property_fields['softness'].setCurrentText(preset['softness'])

            # Trigger save
            self.save_changes()

        # Material dropdown with preset application
        self.property_fields['material'] = self.add_dropdown_field(
            group, "Material Preset", current_material, material_options,
            on_change=on_material_preset_change
        )

        # Mass options (from light to immovable)
        mass_options = ["negligible", "very_light", "light", "medium", "heavy", "very_heavy", "immovable"]
        self.property_fields['mass'] = self.add_dropdown_field(
            group, "Mass", current_mass, mass_options
        )

        # Friction options
        friction_options = ["slippery", "low", "medium", "high", "sticky"]
        self.property_fields['friction'] = self.add_dropdown_field(
            group, "Friction", current_friction, friction_options
        )

        # Elasticity options
        elasticity_options = ["none", "low", "normal", "high", "bouncy"]
        self.property_fields['elasticity'] = self.add_dropdown_field(
            group, "Elasticity", current_elasticity, elasticity_options
        )

        # Softness options
        softness_options = ["rigid", "hard", "normal", "soft", "squishy"]
        self.property_fields['softness'] = self.add_dropdown_field(
            group, "Softness", current_softness, softness_options
        )

        # Help text
        from PyQt6.QtWidgets import QLabel
        help_label = QLabel("Select a material preset to auto-fill physics properties,\nor set each property individually.")
        help_label.setStyleSheet("color: #808080; font-size: 10px; padding: 4px;")
        help_label.setWordWrap(True)
        group.content.layout().addRow("", help_label)

    def load_exit_properties(self, entity_data):
        """Show exit properties."""
        exit_group = self.create_property_group("Exit Info")
        self.add_text_field(exit_group, "Direction", entity_data.get('direction', ''))
        self.add_text_field(exit_group, "Destination", entity_data.get('destination', ''))
        self.properties_layout.addWidget(exit_group)

        self.properties_layout.addStretch()

    def load_radiance_properties(self, entity_data):
        """Show RadianceComponent properties using RadianceInspector widget."""
        from noodlestudio.panels.radiance_inspector import RadianceInspector

        # Store reference to prevent garbage collection
        if not hasattr(self, '_radiance_inspector'):
            self._radiance_inspector = RadianceInspector()

        # Get component and path from entity_data
        component = entity_data.get('component')
        path = entity_data.get('path', '')
        on_change = entity_data.get('on_change')

        # Configure inspector
        self._radiance_inspector.set_component(component, path)
        if on_change:
            self._radiance_inspector.set_on_change_callback(on_change)

        # Add to properties layout
        self.properties_layout.addWidget(self._radiance_inspector)

    def create_property_group(self, title: str) -> CollapsibleSection:
        """
        Create collapsible property group using CollapsibleSection (no bounce-back!).

        Returns CollapsibleSection configured with QFormLayout.
        To add fields, use: group.content.layout().addRow(label, widget)
        """
        section = CollapsibleSection(title)

        # Replace default VBoxLayout with QFormLayout
        form_layout = QFormLayout()
        form_layout.setContentsMargins(12, 8, 12, 8)
        form_layout.setSpacing(6)
        section.set_content_layout(form_layout)

        # Connect toggled signal to track state changes
        section.toggled.connect(lambda expanded: self._on_collapsible_toggled(section.title_text, expanded))

        # Restore previous expanded state (if any)
        self._restore_collapsible_state(section)

        return section

    def on_group_toggled(self, group: QGroupBox, checked: bool):
        """Handle group toggle - update triangle and visibility."""
        # Use blockSignals to prevent signal loops
        group.blockSignals(True)
        try:
            # Update triangle in title
            original_title = group.property("original_title")
            if checked:
                group.setTitle(f"▼ {original_title}")
            else:
                group.setTitle(f"▶ {original_title}")

            # Toggle visibility of contents
            self.toggle_group_contents(group, checked)
        finally:
            # Delay unblocking to ensure Qt event queue clears
            from PyQt6.QtCore import QTimer
            # Safety: check widget still exists before accessing
            def safely_unblock():
                try:
                    if group and not group.isHidden():  # Widget still valid
                        group.blockSignals(False)
                except RuntimeError:
                    pass  # Widget was deleted, ignore
            QTimer.singleShot(100, safely_unblock)

    def toggle_group_contents(self, group: QGroupBox, visible: bool):
        """Toggle visibility of group contents (collapsible sections)."""
        # Hide/show all child widgets in the group's layout
        layout = group.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    item.widget().setVisible(visible)

    def add_text_field(self, group: QGroupBox, label: str, value: str, read_only: bool = False):
        """Add text field to group (instant updates on change unless read_only)."""
        field = QLineEdit(value)
        if read_only:
            field.setReadOnly(True)
            field.setStyleSheet("background-color: #1A1A1A; color: #808080; padding: 4px;")
        else:
            field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
            # Use editingFinished for instant update when user finishes editing
            field.editingFinished.connect(self.save_changes)
        group.content.layout().addRow(f"{label}:", field)
        return field

    def add_vector3_field(self, group: QGroupBox, label: str, values: list, read_only: bool = True):
        """Add compact XYZ vector field on single line."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        fields = []
        for i, (axis, val) in enumerate(zip(['X', 'Y', 'Z'], values)):
            field = QDoubleSpinBox()
            field.setRange(-99999, 99999)
            field.setDecimals(2)
            field.setValue(float(val) if val else 0)
            field.setReadOnly(read_only)
            field.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            field.setFixedWidth(65)
            field.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #1E1E1E;
                    color: #D2D2D2;
                    border: 1px solid #3A3A3A;
                    padding: 2px 4px;
                }
                QDoubleSpinBox:read-only {
                    color: #888888;
                }
            """)
            field.setToolTip(axis)
            row_layout.addWidget(field)
            fields.append(field)

        row_layout.addStretch()
        group.content.layout().addRow(f"{label}:", row)
        return fields

    def add_dropdown_field(self, group: QGroupBox, label: str, value: str, options: list, on_change=None):
        """Add dropdown (combo box) field to group."""
        from PyQt6.QtWidgets import QComboBox

        combo = QComboBox()
        combo.addItems(options)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                padding: 4px;
                border: 1px solid #3A3A3A;
                border-radius: 2px;
            }
            QComboBox:hover {
                border-color: #5A5A5A;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #808080;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #252525;
                color: #D2D2D2;
                selection-background-color: #3D5A80;
            }
        """)

        # Set current value
        if value:
            index = combo.findText(str(value))
            if index >= 0:
                combo.setCurrentIndex(index)

        # Connect change handler
        if on_change:
            combo.currentTextChanged.connect(on_change)
        else:
            combo.currentTextChanged.connect(lambda _: self.save_changes())

        group.content.layout().addRow(f"{label}:", combo)
        return combo

    def add_text_area(self, group: QGroupBox, label: str, value: str):
        """Add editable text area to group (instant updates on change)."""
        field = QTextEdit()
        field.setPlainText(value)
        field.setMaximumHeight(100)
        field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")

        # Enable context menu for external editor
        field.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        field.customContextMenuRequested.connect(lambda pos: self._show_text_editor_context_menu(field, pos))

        # Text areas update when focus is lost (avoid spamming API)
        # Use proper method instead of lambda to handle exceptions
        original_focus_out = field.focusOutEvent
        def safe_focus_out(event):
            try:
                original_focus_out(event)
                self.save_changes()
            except Exception as e:
                print(f"Error in focusOutEvent: {e}")
        field.focusOutEvent = safe_focus_out
        group.content.layout().addRow(f"{label}:", field)
        return field

    def add_slider_field(self, group: QGroupBox, label: str, value: float, min_val: float, max_val: float):
        """Add slider + numeric field (instant updates on change)."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2;")
        # Instant update when value changes
        spin.valueChanged.connect(lambda: self.save_changes())
        group.content.layout().addRow(f"{label}:", spin)
        return spin

    def save_changes(self):
        """Save edited properties back to noodleMUSH."""
        if not self.current_entity:
            return

        # Set flag to prevent refresh during save
        self.is_saving = True

        try:
            entity_type, entity_data = self.current_entity

            if entity_type == 'noodling':
                # Build update payload
                agent_id = entity_data.get('id', '')
                updates = {}

                # Collect field values
                if 'name' in self.property_fields:
                    updates['name'] = self.property_fields['name'].text()
                if 'species' in self.property_fields:
                    updates['species'] = self.property_fields['species'].text()
                if 'description' in self.property_fields:
                    # Description is a QTextEdit, use toPlainText()
                    updates['description'] = self.property_fields['description'].toPlainText()
                if 'llm_provider' in self.property_fields:
                    updates['llm_provider'] = self.property_fields['llm_provider'].text()
                if 'llm_model' in self.property_fields:
                    updates['llm_model'] = self.property_fields['llm_model'].text()

                # Save via API
                try:
                    url = f"{self.api_base}/agents/{agent_id}/update"
                    payload = updates

                    response = requests.post(url, json=payload, timeout=2)
                    if response.status_code == 200:
                        print(f"Saved changes for {agent_id}")
                    else:
                        print(f"Error saving: {response.json().get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"Error saving: {e}")

            elif entity_type in ('prim', 'prop'):
                # Build update payload for prim/prop
                updates = {}

                # Collect basic field values
                if 'name' in self.property_fields:
                    updates['name'] = self.property_fields['name'].text()
                if 'description' in self.property_fields:
                    updates['description'] = self.property_fields['description'].toPlainText()

                # Collect physics properties (from dropdowns)
                physics_fields = ['material', 'mass', 'friction', 'elasticity', 'softness']
                for field_name in physics_fields:
                    if field_name in self.property_fields:
                        widget = self.property_fields[field_name]
                        # QComboBox - use currentText()
                        if hasattr(widget, 'currentText'):
                            value = widget.currentText()
                            # Don't save "(custom)" as material - it's just a UI indicator
                            if field_name == 'material' and value == "(custom)":
                                continue
                            updates[field_name] = value

                # Save based on type
                if entity_type == 'prop':
                    # Project mode - save to prop.yaml
                    self._save_prop_to_file(entity_data, updates)
                else:
                    # Legacy mode - save via API
                    object_id = entity_data.get('id', '')
                    try:
                        url = f"{self.api_base}/objects/{object_id}/update"
                        response = requests.post(url, json=updates, timeout=2)
                        if response.status_code == 200:
                            print(f"Saved prim {object_id}: {list(updates.keys())}")
                        else:
                            print(f"Error saving: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"Error saving prim: {e}")

        finally:
            # Clear flag after save completes (wait longer than refresh interval)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2500, lambda: setattr(self, 'is_saving', False))

    def _save_prop_to_file(self, entity_data: dict, updates: dict):
        """Save prop changes to prop.yaml file."""
        import yaml
        import os

        prop_path = entity_data.get('path', '')
        if not prop_path:
            print("No path for prop - cannot save")
            return

        prop_yaml = os.path.join(prop_path, "prop.yaml")
        if not os.path.exists(prop_yaml):
            print(f"prop.yaml not found at {prop_yaml}")
            return

        try:
            # Load existing data
            with open(prop_yaml, 'r') as f:
                prop_data = yaml.safe_load(f) or {}

            # Update with new values
            prop_data.update(updates)

            # Save back
            with open(prop_yaml, 'w') as f:
                yaml.dump(prop_data, f, default_flow_style=False)

            prop_name = updates.get('name', entity_data.get('name', 'prop'))
            print(f"Saved prop {prop_name}: {list(updates.keys())}")

        except Exception as e:
            print(f"Error saving prop to file: {e}")

    def save_component_changes(self, agent_id: str, component_id: str):
        """
        Save component changes to backend API.

        Args:
            agent_id: Agent identifier
            component_id: Component identifier
        """
        try:
            self.is_saving = True

            # Verify component widgets exist
            if agent_id not in self.component_widgets:
                return
            if component_id not in self.component_widgets[agent_id]:
                return

            comp_widgets = self.component_widgets[agent_id][component_id]

            # Collect parameters
            parameters = {}

            # Enabled state
            if 'enabled' in comp_widgets:
                parameters['enabled'] = comp_widgets['enabled'].isChecked()

            # Parameter values
            for key, widget in comp_widgets.items():
                if key.startswith('param_'):
                    param_name = key[6:]  # Remove 'param_' prefix

                    # Extract value based on widget type
                    if hasattr(widget, 'isChecked'):
                        parameters[param_name] = widget.isChecked()
                    elif hasattr(widget, 'value'):
                        parameters[param_name] = widget.value()
                    elif hasattr(widget, 'text'):
                        parameters[param_name] = widget.text()

            # Build update payload
            update_data = {'parameters': parameters}

            # POST to API
            response = requests.post(
                f"{self.api_base}/agents/{agent_id}/components/{component_id}/update",
                json=update_data,
                timeout=5
            )

            if response.status_code == 200:
                print(f"Component {component_id} saved for {agent_id}")
            else:
                print(f"Error saving component: {response.text}")

        except Exception as e:
            print(f"Error saving component: {e}")

        finally:
            # Clear saving flag
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2500, lambda: setattr(self, 'is_saving', False))

    def create_components_section(self, agent_id: str) -> QWidget:
        """
        Create Cognitive Components section.

        Displays all cognitive processing components with editable prompts
        and parameters. Uses custom CollapsibleSection to avoid QGroupBox
        double-trigger bug.

        Args:
            agent_id: Agent identifier

        Returns:
            Widget containing all components, or None if API fails
        """
        try:
            # Fetch components from API
            response = requests.get(f"{self.api_base}/agents/{agent_id}/components", timeout=2)
            if response.status_code != 200:
                return None

            components_data = response.json()
            components = components_data.get('components', [])

            if not components:
                return None

            # SKIP if using Facet Assembly system (handled in Noodle Component now!)
            if components and components[0].get('component_id') == 'facet_assembly':
                # Skip the first item (Facet Assembly) as it's in Noodle Component
                components = components[1:]

            if not components:
                return None  # No components to show

            # Main container widget
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)

            # Header label
            header = QLabel("Cognitive Components")
            header_font = QFont()
            header_font.setBold(True)
            header_font.setPointSize(11)
            header.setFont(header_font)
            header.setStyleSheet("color: #00FF00; padding: 4px;")
            container_layout.addWidget(header)

            # Create collapsible section for each component
            for comp_data in components:
                comp_section = self.create_single_component(agent_id, comp_data)
                if comp_section:
                    container_layout.addWidget(comp_section)

            return container

        except Exception as e:
            print(f"Error loading components: {e}")
            return None

    def create_single_component(self, agent_id: str, comp_data: dict) -> CollapsibleSection:
        """
        Create UI for a single cognitive component.

        Args:
            agent_id: Agent identifier
            comp_data: Component data from API

        Returns:
            CollapsibleSection widget with component details
        """
        component_id = comp_data.get('component_id', '')
        component_type = comp_data.get('component_type', 'Unknown')
        description = comp_data.get('description', '')
        enabled = comp_data.get('enabled', True)
        prompt_template = comp_data.get('prompt_template', '')
        parameters = comp_data.get('parameters', {})

        # Initialize widget tracking structure for this agent/component
        if agent_id not in self.component_widgets:
            self.component_widgets[agent_id] = {}
        if component_id not in self.component_widgets[agent_id]:
            self.component_widgets[agent_id][component_id] = {}

        comp_widgets = self.component_widgets[agent_id][component_id]

        # Create collapsible section (no double-trigger!)
        section = CollapsibleSection(f"{component_type}")

        # Connect toggled signal to track state changes
        section.toggled.connect(lambda expanded: self._on_collapsible_toggled(section.title_text, expanded))

        # Restore previous expanded state (if any)
        self._restore_collapsible_state(section)

        # Component description (read-only)
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #AAAAAA; font-size: 9pt; font-style: italic; padding: 4px 0;")
        section.add_widget(desc_label)

        # If facet assembly, add "Open Editor" button
        if parameters.get('clickable'):
            open_button = QPushButton("Open Facet Editor")
            open_button.setStyleSheet("""
                QPushButton {
                    background-color: #4A4A4A;
                    color: #FFFFFF;
                    border: 1px solid #5A5A5A;
                    border-radius: 3px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5A5A5A;
                }
                QPushButton:pressed {
                    background-color: #3A3A3A;
                }
            """)
            open_button.clicked.connect(lambda: self.open_facet_editor(agent_id))
            section.add_widget(open_button)

        # Enabled checkbox
        from PyQt6.QtWidgets import QCheckBox
        enabled_checkbox = QCheckBox("Enabled")
        enabled_checkbox.setChecked(enabled)
        enabled_checkbox.setStyleSheet("color: #FFFFFF;")

        # Track widget and wire state change to save
        comp_widgets['enabled'] = enabled_checkbox

        section.add_widget(enabled_checkbox)

        # Prompt template (editable)
        prompt_label = QLabel("Prompt Template:")
        prompt_label.setStyleSheet("color: #FFFFFF; font-weight: bold; margin-top: 8px;")
        section.add_widget(prompt_label)

        prompt_edit = QTextEdit()
        prompt_edit.setPlainText(prompt_template)
        prompt_edit.setMaximumHeight(150)
        prompt_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #00FF00;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 4px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }
        """)

        # Track widget
        comp_widgets['prompt_template'] = prompt_edit

        section.add_widget(prompt_edit)

        # Parameters (editable)
        if parameters:
            params_label = QLabel("Parameters:")
            params_label.setStyleSheet("color: #FFFFFF; font-weight: bold; margin-top: 8px;")
            section.add_widget(params_label)

            # Create form layout for parameters
            from PyQt6.QtWidgets import QFormLayout
            params_form = QFormLayout()
            params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            params_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

            for param_name, param_value in parameters.items():
                if param_name == 'enabled':
                    continue  # Already shown as checkbox

                param_label = QLabel(param_name.replace('_', ' ').title() + ":")
                param_label.setStyleSheet("color: #CCCCCC;")

                # Create appropriate input widget based on type
                if isinstance(param_value, bool):
                    param_widget = QCheckBox()
                    param_widget.setChecked(param_value)
                elif isinstance(param_value, float):
                    param_widget = QDoubleSpinBox()
                    param_widget.setValue(param_value)
                    param_widget.setRange(0.0, 10.0)
                    param_widget.setSingleStep(0.1)
                    param_widget.setDecimals(2)
                elif isinstance(param_value, int):
                    param_widget = QSpinBox()
                    param_widget.setValue(param_value)
                    param_widget.setRange(0, 1000)
                else:
                    param_widget = QLineEdit(str(param_value))

                param_widget.setStyleSheet("""
                    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox {
                        background-color: #1E1E1E;
                        color: #FFFFFF;
                        border: 1px solid #3A3A3A;
                        border-radius: 3px;
                        padding: 3px;
                    }
                """)

                # Track widget for save operations
                comp_widgets[f'param_{param_name}'] = param_widget

                params_form.addRow(param_label, param_widget)

            params_widget = QWidget()
            params_widget.setLayout(params_form)
            section.add_widget(params_widget)

        return section

    def create_noodle_component(self, agent_id: str) -> CollapsibleSection:
        """
        Create the Noodle Component (charm component).

        Shows LIVE updating:
        - Affect Vector (continuous affect state)
        - Surprise metric
        - Cognitive Architecture (facet assembly)
        """
        # Create CollapsibleSection (no bounce-back!)
        component = CollapsibleSection("Noodle Component")

        # Connect toggled signal for state tracking
        component.toggled.connect(lambda expanded: self._on_collapsible_toggled(component.title_text, expanded))

        # Restore previous expanded state
        self._restore_collapsible_state(component)

        # Content uses VBoxLayout (not FormLayout) for this special component
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        component.set_content_layout(layout)

        # Agent ID reference (for live updates)
        self.current_agent_id = agent_id

        # Affect Vector (LIVE)
        affect_label = QLabel("Affect Vector")
        affect_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 8px;")
        layout.addWidget(affect_label)

        affect_layout = QFormLayout()

        # Create progress bars for each affect dimension
        self.live_affect_labels['valence'] = self.create_affect_bar("Valence", -1.0, 1.0)
        affect_layout.addRow("Valence:", self.live_affect_labels['valence'])

        self.live_affect_labels['arousal'] = self.create_affect_bar("Arousal", 0.0, 1.0)
        affect_layout.addRow("Arousal:", self.live_affect_labels['arousal'])

        self.live_affect_labels['dominance'] = self.create_affect_bar("Dominance", 0.0, 1.0)
        affect_layout.addRow("Dominance:", self.live_affect_labels['dominance'])

        self.live_affect_labels['sorrow'] = self.create_affect_bar("Sorrow", 0.0, 1.0)
        affect_layout.addRow("Sorrow:", self.live_affect_labels['sorrow'])

        self.live_affect_labels['boredom'] = self.create_affect_bar("Boredom", 0.0, 1.0)
        affect_layout.addRow("Boredom:", self.live_affect_labels['boredom'])

        layout.addLayout(affect_layout)

        # Surprise metric
        surprise_layout = QFormLayout()
        self.live_surprise_label = QLabel("0.000")
        self.live_surprise_label.setStyleSheet("color: #D2D2D2; font-weight: bold;")
        surprise_layout.addRow("Surprise:", self.live_surprise_label)
        layout.addLayout(surprise_layout)

        # Facet Assembly section (if agent uses facet system)
        try:
            resp = requests.get(f"{self.api_base}/agents/{agent_id}/components", timeout=1)
            if resp.status_code == 200:
                components_data = resp.json()
                components = components_data.get('components', [])

                # Check if first component is Facet Assembly
                if components and components[0].get('component_id') == 'facet_assembly':
                    facet_assembly_section = QLabel("Cognitive Architecture")
                    facet_assembly_section.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 12px;")
                    layout.addWidget(facet_assembly_section)

                    facet_data = components[0]
                    assembly_name = facet_data.get('parameters', {}).get('assembly_name', 'unknown')
                    facet_count = facet_data.get('parameters', {}).get('facet_count', 0)

                    assembly_info = QLabel(f"Assembly: {assembly_name}\nFacets: {facet_count}")
                    assembly_info.setStyleSheet("color: #B0B0B0; font-size: 10pt;")
                    layout.addWidget(assembly_info)

                    # Open Editor button
                    open_button = QPushButton("Open Facets Editor")
                    open_button.setStyleSheet("""
                        QPushButton {
                            background-color: #4A4A4A;
                            color: #FFFFFF;
                            border: 1px solid #5A5A5A;
                            border-radius: 3px;
                            padding: 6px 12px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #5A5A5A;
                        }
                        QPushButton:pressed {
                            background-color: #3A3A3A;
                        }
                    """)
                    open_button.clicked.connect(lambda: self.open_facet_editor(agent_id))
                    layout.addWidget(open_button)
        except:
            pass  # Silent fail if API unavailable

        component.setLayout(layout)
        return component

    def create_affect_bar(self, name: str, min_val: float, max_val: float) -> QWidget:
        """Create a horizontal bar + value label for affect dimension."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress bar
        bar = QProgressBar()
        bar.setRange(int(min_val * 100), int(max_val * 100))
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setMaximumHeight(12)
        bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 2px;
            }
        """)
        layout.addWidget(bar, stretch=3)

        # Value label
        value_label = QLabel("0.00")
        value_label.setStyleSheet("color: #D2D2D2; font-family: 'Courier New';")
        value_label.setMinimumWidth(45)
        layout.addWidget(value_label)

        container.setLayout(layout)

        # Store references
        container.bar = bar
        container.value_label = value_label

        return container

    def update_live_data(self):
        """Update live Noodle Component data from API."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = self.current_agent_id if hasattr(self, 'current_agent_id') else entity_data.get('id')

        try:
            # Fetch live state from API
            resp = requests.get(f"{self.api_base}/agents/{agent_id}/state", timeout=1)
            if resp.status_code == 200:
                state = resp.json()

                # Update 5-D Affect Vector
                affect = state.get('affect', {})

                # Monochromatic color mapping (Ordnung muss sein!)
                # Brighter readable gray for affect bars
                affect_color = '#BBBBBB'  # Light gray - more visible

                for dim, widget in self.live_affect_labels.items():
                    if dim in affect and widget and widget.bar and widget.value_label:
                        value = affect[dim]
                        # Update bar and label atomically to prevent flash
                        widget.bar.setStyleSheet(f"""
                            QProgressBar {{
                                border: 1px solid #555;
                                border-radius: 3px;
                                background: #2a2a2a;
                            }}
                            QProgressBar::chunk {{
                                background: {affect_color};
                                border-radius: 2px;
                            }}
                        """)
                        widget.bar.setValue(int(value * 100))
                        widget.value_label.setText(f"{value:.2f}")

                # Update Surprise (only if widget exists)
                if hasattr(self, 'live_surprise_label') and self.live_surprise_label:
                    surprise = state.get('surprise', 0.0)
                    self.live_surprise_label.setText(f"{surprise:.3f}")

        except requests.exceptions.RequestException:
            # API not available, silently fail
            pass
        except Exception as e:
            print(f"Error updating live data: {e}")

    def add_artbook_component(self):
        """
        Add Artbook component to current entity.

        Shows reference art from assets folder - like ArtStation for your character!
        """
        artbook = self.create_artbook_component()
        self.properties_layout.addWidget(artbook)

    def create_artbook_component(self) -> CollapsibleSection:
        """
        Create Artbook component (modular component).

        Holds reference art, concept sketches, mood boards for the character.
        """
        # Create CollapsibleSection (no bounce-back!)
        component = CollapsibleSection("Artbook Component")
        component.toggled.connect(lambda expanded: self._on_collapsible_toggled(component.title_text, expanded))
        self._restore_collapsible_state(component)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        component.set_content_layout(layout)

        # Description
        desc = QLabel("Reference art and concept images for this character")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Art gallery (thumbnail grid)
        gallery_label = QLabel("Reference Gallery")
        gallery_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 4px;")
        layout.addWidget(gallery_label)

        # List widget for art thumbnails
        self.art_gallery = QListWidget()
        self.art_gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.art_gallery.setIconSize(QSize(80, 80))
        self.art_gallery.setSpacing(8)
        self.art_gallery.setMaximumHeight(200)
        self.art_gallery.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.art_gallery.customContextMenuRequested.connect(self._show_image_context_menu)
        self.art_gallery.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QListWidget::item {
                background: #1a1a1a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item:hover {
                background: #333;
                border: 1px solid #666;
            }
            QListWidget::item:selected {
                background: #444;
                border: 1px solid #888;
            }
        """)
        layout.addWidget(self.art_gallery)

        # Buttons
        button_layout = QHBoxLayout()

        add_art_btn = QPushButton("+ Add Art")
        add_art_btn.clicked.connect(self.add_art_to_gallery)
        add_art_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                color: #D2D2D2;
                padding: 6px 12px;
                border-radius: 3px;
                border: 1px solid #555;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        button_layout.addWidget(add_art_btn)

        remove_art_btn = QPushButton("− Remove")
        remove_art_btn.clicked.connect(self.remove_art_from_gallery)
        remove_art_btn.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        button_layout.addWidget(remove_art_btn)

        layout.addLayout(button_layout)

        # Art source info
        source_label = QLabel("💡 Tip: Keep art in ~/.noodlestudio/assets/[character_name]/")
        source_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        source_label.setWordWrap(True)
        layout.addWidget(source_label)

        component.setLayout(layout)

        # Load existing art if any
        self.load_artbook_gallery()

        return component

    def add_art_to_gallery(self):
        """Add art file to the gallery."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Reference Art",
            str(Path.home() / ".noodlestudio" / "assets"),
            "Images (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*)"
        )

        for filename in filenames:
            if filename:
                # Create thumbnail
                pixmap = QPixmap(filename)
                if not pixmap.isNull():
                    # Scale to thumbnail size
                    scaled = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

                    # Add to gallery
                    item = QListWidgetItem()
                    item.setIcon(QIcon(scaled))
                    item.setToolTip(Path(filename).name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)  # Store full path
                    self.art_gallery.addItem(item)

        # Save gallery state
        self.save_artbook_gallery()

    def remove_art_from_gallery(self):
        """Remove selected art from gallery."""
        current = self.art_gallery.currentItem()
        if current:
            self.art_gallery.takeItem(self.art_gallery.row(current))
            self.save_artbook_gallery()

    def load_artbook_gallery(self):
        """Load artbook gallery from saved state."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = entity_data.get('id', '')

        # Load from .noodlestudio/artbooks/{agent_id}.json
        artbook_dir = Path.home() / ".noodlestudio" / "artbooks"
        artbook_file = artbook_dir / f"{agent_id}.json"

        if artbook_file.exists():
            try:
                import json
                with open(artbook_file, 'r') as f:
                    data = json.load(f)

                for art_path in data.get('art_files', []):
                    if Path(art_path).exists():
                        pixmap = QPixmap(art_path)
                        if not pixmap.isNull():
                            scaled = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)

                            item = QListWidgetItem()
                            item.setIcon(QIcon(scaled))
                            item.setToolTip(Path(art_path).name)
                            item.setData(Qt.ItemDataRole.UserRole, art_path)
                            self.art_gallery.addItem(item)

            except Exception as e:
                print(f"Error loading artbook: {e}")

    def save_artbook_gallery(self):
        """Save artbook gallery state."""
        if not self.current_entity:
            return

        entity_type, entity_data = self.current_entity
        if entity_type != 'noodling':
            return

        agent_id = entity_data.get('id', '')

        # Collect all art file paths
        art_files = []
        for i in range(self.art_gallery.count()):
            item = self.art_gallery.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                art_files.append(path)

        # Save to .noodlestudio/artbooks/{agent_id}.json
        artbook_dir = Path.home() / ".noodlestudio" / "artbooks"
        artbook_dir.mkdir(parents=True, exist_ok=True)

        artbook_file = artbook_dir / f"{agent_id}.json"

        try:
            import json
            with open(artbook_file, 'w') as f:
                json.dump({'art_files': art_files}, f, indent=2)
        except Exception as e:
            print(f"Error saving artbook: {e}")

    def add_script_component(self):
        """
        Add Script component with code editor.

        Event-driven scripting component!
        """
        script_comp = self.create_script_component()
        self.properties_layout.addWidget(script_comp)

    def create_script_component(self) -> QGroupBox:
        """
        Create Script component (modular scripting).

        Shows code editor with syntax highlighting and compile button.
        """
        component = QGroupBox("📜 Script Component")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #9C27B0;
                border: 2px solid #9C27B0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Python script for event-driven behavior (component-based API)")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Script editor widget
        from ..widgets.script_editor import ScriptEditor
        self.script_editor = ScriptEditor()
        layout.addWidget(self.script_editor)

        component.setLayout(layout)
        return component

    def create_mmcr_component(self, agent_id: str) -> CollapsibleSection:
        """
        Create MMCR (Multimodal Context Reference) component.

        Holds arbitrary media that scripts can access via API:
        - Images (concept art, reference photos, environment maps)
        - Audio (voice clips, sound effects, music)
        - Video (animations, cutscenes)
        - Text (notes, dialogue snippets)

        Unlike Artbook (which is for visual reference), MMCR is for
        runtime-accessible context that affects behavior.
        """
        # Create CollapsibleSection (no bounce-back!)
        component = CollapsibleSection("Multimodal Context Reference")
        component.toggled.connect(lambda expanded: self._on_collapsible_toggled(component.title_text, expanded))
        self._restore_collapsible_state(component)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        component.set_content_layout(layout)

        # Description
        desc = QLabel("Runtime-accessible media for scripts and LLM context")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Images section
        images_label = QLabel("Images")
        images_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 4px;")
        layout.addWidget(images_label)

        self.mmcr_images = QListWidget()
        self.mmcr_images.setMaximumHeight(100)
        self.mmcr_images.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        layout.addWidget(self.mmcr_images)

        # Audio section
        audio_label = QLabel("Audio")
        audio_label.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 8px;")
        layout.addWidget(audio_label)

        self.mmcr_audio = QListWidget()
        self.mmcr_audio.setMaximumHeight(80)
        self.mmcr_audio.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mmcr_audio.customContextMenuRequested.connect(self._show_audio_context_menu)
        self.mmcr_audio.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        layout.addWidget(self.mmcr_audio)

        # Buttons
        button_layout = QHBoxLayout()

        add_media_btn = QPushButton("Add Media")
        add_media_btn.clicked.connect(lambda: self.add_mmcr_media(agent_id))
        add_media_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #64B5F6;
            }
        """)
        button_layout.addWidget(add_media_btn)

        remove_media_btn = QPushButton("Remove")
        remove_media_btn.clicked.connect(lambda: self.remove_mmcr_media())
        remove_media_btn.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        button_layout.addWidget(remove_media_btn)

        layout.addLayout(button_layout)

        # API access info
        info_label = QLabel("Scripts access via: noodlings.getComponent('mmcr').images[0]")
        info_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        component.setLayout(layout)

        # Load existing MMCR data if any
        self.load_mmcr_data(agent_id)

        return component

    def add_mmcr_media(self, agent_id: str):
        """Add media files to MMCR component."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Media to MMCR",
            str(Path.home()),
            "Media Files (*.png *.jpg *.jpeg *.gif *.webp *.wav *.mp3 *.mp4 *.mov);;All Files (*)"
        )

        for filename in filenames:
            if filename:
                file_path = Path(filename)
                ext = file_path.suffix.lower()

                # Categorize by file type
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    item = QListWidgetItem(file_path.name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)
                    self.mmcr_images.addItem(item)
                elif ext in ['.wav', '.mp3', '.ogg', '.m4a']:
                    item = QListWidgetItem(file_path.name)
                    item.setData(Qt.ItemDataRole.UserRole, filename)
                    self.mmcr_audio.addItem(item)

        # Save MMCR state
        self.save_mmcr_data(agent_id)

    def remove_mmcr_media(self):
        """Remove selected media from MMCR."""
        # Check images list
        current = self.mmcr_images.currentItem()
        if current:
            self.mmcr_images.takeItem(self.mmcr_images.row(current))
            return

        # Check audio list
        current = self.mmcr_audio.currentItem()
        if current:
            self.mmcr_audio.takeItem(self.mmcr_audio.row(current))

    def load_mmcr_data(self, agent_id: str):
        """Load MMCR data from storage."""
        # TODO: Implement when components dict is added to world structure
        # Would load from: agent_data['components']['mmcr']
        pass

    def save_mmcr_data(self, agent_id: str):
        """Save MMCR data to storage."""
        # Collect all media files
        images = []
        for i in range(self.mmcr_images.count()):
            item = self.mmcr_images.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                images.append(path)

        audio = []
        for i in range(self.mmcr_audio.count()):
            item = self.mmcr_audio.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                audio.append(path)

        # TODO: Save via API to agent's components.mmcr
        # POST /api/agents/{agent_id}/components/mmcr
        print(f"MMCR data for {agent_id}:")
        print(f"  Images: {images}")
        print(f"  Audio: {audio}")

    def create_metadata_component(self, entity_id: str) -> QGroupBox:
        """
        Create Metadata component for arbitrary key-value pairs.

        Like USD custom attributes - author can add any field they want:
        - asteroid.mass_kg = 8500000000
        - asteroid.minerals = ["iron: 45%", "platinum: 0.02%"]
        - asteroid.appearance_far = "A dark speck..."

        Scripts access via: prim.metadata["mass_kg"]
        """
        component = QGroupBox("Custom Metadata")
        component.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        component.setStyleSheet("""
            QGroupBox {
                color: #9E9E9E;
                border: 2px solid #757575;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout()

        # Description
        desc = QLabel("Arbitrary key-value pairs accessible to scripts and renderers")
        desc.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Metadata list (shows key: value pairs)
        self.metadata_list = QListWidget()
        self.metadata_list.setMaximumHeight(150)
        self.metadata_list.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #D2D2D2;
                font-size: 10px;
                font-family: 'Courier New';
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:hover {
                background: #333;
            }
            QListWidget::item:selected {
                background: #555;
            }
        """)
        layout.addWidget(self.metadata_list)

        # Buttons
        button_layout = QHBoxLayout()

        add_meta_btn = QPushButton("Add Field")
        add_meta_btn.clicked.connect(lambda: self.add_metadata_field(entity_id))
        add_meta_btn.setStyleSheet("""
            QPushButton {
                background: #757575;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #9E9E9E;
            }
        """)
        button_layout.addWidget(add_meta_btn)

        edit_meta_btn = QPushButton("Edit")
        edit_meta_btn.clicked.connect(lambda: self.edit_metadata_field(entity_id))
        edit_meta_btn.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        button_layout.addWidget(edit_meta_btn)

        remove_meta_btn = QPushButton("Remove")
        remove_meta_btn.clicked.connect(lambda: self.remove_metadata_field(entity_id))
        remove_meta_btn.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        button_layout.addWidget(remove_meta_btn)

        layout.addLayout(button_layout)

        # Access info
        info_label = QLabel("Example: asteroid.metadata['mass_kg'] or asteroid.metadata['minerals']")
        info_label.setStyleSheet("color: #888; font-size: 9px; margin-top: 8px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        component.setLayout(layout)

        # Load existing metadata
        self.load_metadata(entity_id)

        return component

    def add_metadata_field(self, entity_id: str):
        """Add a new metadata field."""
        from PyQt6.QtWidgets import QInputDialog

        # Get field name
        field_name, ok = QInputDialog.getText(
            self,
            "Add Metadata Field",
            "Field name (e.g., 'mass_kg', 'minerals', 'velocity'):"
        )

        if ok and field_name:
            # Get field value
            field_value, ok = QInputDialog.getText(
                self,
                "Add Metadata Field",
                f"Value for '{field_name}':"
            )

            if ok:
                # Add to list
                item = QListWidgetItem(f"{field_name}: {field_value}")
                item.setData(Qt.ItemDataRole.UserRole, {'key': field_name, 'value': field_value})
                self.metadata_list.addItem(item)

                # Save
                self.save_metadata(entity_id)

    def edit_metadata_field(self, entity_id: str):
        """Edit selected metadata field."""
        from PyQt6.QtWidgets import QInputDialog

        current = self.metadata_list.currentItem()
        if not current:
            return

        data = current.data(Qt.ItemDataRole.UserRole)
        field_name = data['key']
        old_value = data['value']

        # Get new value
        new_value, ok = QInputDialog.getText(
            self,
            "Edit Metadata Field",
            f"New value for '{field_name}':",
            text=old_value
        )

        if ok:
            # Update item
            current.setText(f"{field_name}: {new_value}")
            current.setData(Qt.ItemDataRole.UserRole, {'key': field_name, 'value': new_value})

            # Save
            self.save_metadata(entity_id)

    def remove_metadata_field(self, entity_id: str):
        """Remove selected metadata field."""
        current = self.metadata_list.currentItem()
        if current:
            self.metadata_list.takeItem(self.metadata_list.row(current))
            self.save_metadata(entity_id)

    def load_metadata(self, entity_id: str):
        """Load metadata from world state."""
        # TODO: Load from world state when metadata dict is added
        # For now, show example for demonstration
        if entity_id.startswith('obj_'):
            # Example metadata
            examples = {
                'type': 'vending_machine',
                'portable': 'true',
                'takeable': 'true'
            }
            for key, value in examples.items():
                item = QListWidgetItem(f"{key}: {value}")
                item.setData(Qt.ItemDataRole.UserRole, {'key': key, 'value': value})
                self.metadata_list.addItem(item)

    def save_metadata(self, entity_id: str):
        """Save metadata to world state."""
        # Collect all metadata fields
        metadata = {}
        for i in range(self.metadata_list.count()):
            item = self.metadata_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                metadata[data['key']] = data['value']

        # TODO: Save via API
        # POST /api/objects/{entity_id}/metadata
        print(f"Metadata for {entity_id}:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    def _show_text_editor_context_menu(self, text_widget, pos):
        """Show context menu with 'Open in External Editor' option."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        menu = QMenu(text_widget)

        # Add standard edit actions safely
        try:
            standard_menu = text_widget.createStandardContextMenu()
            if standard_menu:
                for action in standard_menu.actions():
                    if action.text():
                        menu.addAction(action)
                menu.addSeparator()
        except Exception as e:
            print(f"Error creating standard context menu: {e}")

        # External editor action
        external_action = QAction("View in External Editor", menu)
        external_action.triggered.connect(lambda: self._view_in_external_text_editor(text_widget))
        menu.addAction(external_action)

        menu.exec(text_widget.mapToGlobal(pos))

    def _view_in_external_text_editor(self, text_widget):
        """View text in external editor (read-only snapshot)."""
        import tempfile
        import subprocess
        import json
        from pathlib import Path

        # Get external editor path from settings
        settings_file = Path.home() / ".noodlestudio" / "settings.json"
        editor_path = None

        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    editor_path = settings.get('external_apps', {}).get('text_editor')
            except:
                pass

        if not editor_path or not Path(editor_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Text Editor Configured",
                "Please configure a text editor in:\nSettings → External Applications"
            )
            return

        # Create temp file with current text
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='noodlestudio_view_')
        with open(temp_path, 'w') as f:
            f.write(text_widget.toPlainText())

        print(f"[ExternalEditor] Viewing in editor: {temp_path}")

        # Open in external editor
        try:
            subprocess.Popen(['open', '-a', editor_path, temp_path])
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Failed to Open Editor", f"Error: {e}")

    def _show_image_context_menu(self, pos):
        """Show context menu for images in gallery."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        current_item = self.art_gallery.currentItem()
        if not current_item:
            return

        menu = QMenu(self.art_gallery)

        # Open in image editor
        edit_action = QAction("Open in Image Editor", menu)
        edit_action.triggered.connect(lambda: self._open_image_in_editor(current_item))
        menu.addAction(edit_action)

        menu.addSeparator()

        # Remove from gallery
        remove_action = QAction("Remove from Gallery", menu)
        remove_action.triggered.connect(lambda: self.remove_art_from_gallery())
        menu.addAction(remove_action)

        menu.exec(self.art_gallery.mapToGlobal(pos))

    def _open_image_in_editor(self, list_item):
        """Open image in external editor, watch for changes."""
        import subprocess
        import json
        from pathlib import Path
        from PyQt6.QtCore import QFileSystemWatcher

        # Get image path from item data
        image_path = list_item.data(Qt.ItemDataRole.UserRole)
        if not image_path or not Path(image_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Image Not Found", f"Image file not found: {image_path}")
            return

        # Get external image editor from settings
        settings_file = Path.home() / ".noodlestudio" / "settings.json"
        editor_path = None

        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    editor_path = settings.get('external_apps', {}).get('image_editor')
            except:
                pass

        if not editor_path or not Path(editor_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Image Editor Configured",
                "Please configure an image editor in:\nSettings → External Applications"
            )
            return

        # Open in external editor
        try:
            subprocess.Popen(['open', '-a', editor_path, image_path])
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Failed to Open Editor", f"Error: {e}")
            return

        # Watch image file for changes
        watcher = QFileSystemWatcher([image_path])

        def on_image_changed(path):
            """Reload thumbnail when image changes."""
            try:
                # Reload the pixmap
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    icon = QIcon(pixmap)
                    list_item.setIcon(icon)
                    print(f"Image reloaded: {path}")
            except Exception as e:
                print(f"Error reloading image: {e}")

        watcher.fileChanged.connect(on_image_changed)

        # Keep watcher alive
        if not hasattr(self, '_file_watchers'):
            self._file_watchers = []
        self._file_watchers.append((watcher, image_path))

    def _show_audio_context_menu(self, pos):
        """Show context menu for audio files."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        current_item = self.mmcr_audio.currentItem()
        if not current_item:
            return

        menu = QMenu(self.mmcr_audio)

        # Open in audio editor
        edit_action = QAction("Open in Audio Editor", menu)
        edit_action.triggered.connect(lambda: self._open_audio_in_editor(current_item))
        menu.addAction(edit_action)

        menu.addSeparator()

        # Remove from list
        remove_action = QAction("Remove from List", menu)
        remove_action.triggered.connect(lambda: self.remove_media_from_mmcr())
        menu.addAction(remove_action)

        menu.exec(self.mmcr_audio.mapToGlobal(pos))

    def _open_audio_in_editor(self, list_item):
        """Open audio file in external editor."""
        import subprocess
        import json
        from pathlib import Path

        # Get audio path from item text (for now - might need UserRole data)
        audio_path = list_item.text()
        if not Path(audio_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Audio File Not Found", f"File not found: {audio_path}")
            return

        # Get external audio editor from settings
        settings_file = Path.home() / ".noodlestudio" / "settings.json"
        editor_path = None

        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    editor_path = settings.get('external_apps', {}).get('audio_editor')
            except:
                pass

        if not editor_path or not Path(editor_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Audio Editor Configured",
                "Please configure an audio editor in:\nSettings → External Applications"
            )
            return

        # Open in external editor
        try:
            subprocess.Popen(['open', '-a', editor_path, audio_path])
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Failed to Open Editor", f"Error: {e}")
            return

        # Note: Audio files don't need live reload like images do
        # User will manually re-add if they want updated version

    def open_facet_editor(self, agent_id: str):
        """
        Open Facets Editor for agent's facet assembly.

        Switches to Facets Editor tab and loads the agent's assembly.
        """
        # Emit signal to main window to switch tabs
        # The main window will catch this and switch to Facets Editor
        from PyQt6.QtCore import pyqtSignal

        # Get parent main window
        main_window = self.window()
        if hasattr(main_window, 'right_tabs'):
            # Switch to Facets Editor tab (index 2: Inspector=0, Noodle Tuner=1, Facets Editor=2)
            main_window.right_tabs.setCurrentIndex(2)

            # Signal Facets Editor to load this agent
            if hasattr(main_window, 'facets_editor_panel'):
                main_window.facets_editor_panel.set_current_agent(agent_id)

