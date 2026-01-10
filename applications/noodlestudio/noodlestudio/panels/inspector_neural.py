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
#   Neural Inspector Mixin - Neural Canvas node inspection
#
#   Handles inspection of Neural Canvas nodes: - Node paramet...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_neural
# PURPOSE:  Inspector Neural
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralInspectorMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

from ..core.undo_manager import UndoManager
from ..core.commands.neural_commands import EditNeuralNodeParamCommand, RenameNeuralNodeCommand


class NeuralInspectorMixin:
    """
    Mixin providing Neural Canvas node inspection.

    Requires host class to have:
    - self.properties_layout (QVBoxLayout)
    - self.create_property_group(title)
    - ClickableTextEdit class available
    """

    def load_neural_node_properties(self, entity_data):
        """Show Neural Canvas node properties for editing."""
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

        # Basic Info
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

        # Parameters (editable)
        if params:
            params_group = self.create_property_group("Parameters")

            for param_name, param_value in params.items():
                widget = self._create_param_widget(param_name, param_value)
                if widget:
                    label = self._format_param_label(param_name)
                    params_group.content.layout().addRow(f"{label}:", widget)
                    self._neural_node_param_widgets[param_name] = widget

            self.properties_layout.addWidget(params_group)

        # Weights (read-only)
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

        # Inputs/Outputs (read-only)
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

        # Description
        if description:
            desc_group = self.create_property_group("Description")
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888; padding: 4px;")
            desc_label.setWordWrap(True)
            desc_group.content.layout().addRow(desc_label)
            self.properties_layout.addWidget(desc_group)

        self.properties_layout.addStretch()

    def _create_param_widget(self, param_name: str, param_value):
        """Create appropriate widget for parameter type."""
        from .inspector_base import ClickableTextEdit

        if param_name == 'text':
            # Multi-line text for COMMENT nodes
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
            return text_edit

        elif isinstance(param_value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(param_value)
            checkbox.stateChanged.connect(
                lambda state, pn=param_name: self._save_neural_node_param(pn, state == 2)
            )
            return checkbox

        elif isinstance(param_value, int):
            spin = QSpinBox()
            spin.setRange(-99999, 99999)
            spin.setValue(param_value)
            spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 2px;")
            spin.valueChanged.connect(
                lambda val, pn=param_name: self._save_neural_node_param(pn, val)
            )
            return spin

        elif isinstance(param_value, float):
            spin = QDoubleSpinBox()
            spin.setRange(-99999.0, 99999.0)
            spin.setDecimals(3)
            spin.setValue(param_value)
            spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 2px;")
            spin.valueChanged.connect(
                lambda val, pn=param_name: self._save_neural_node_param(pn, val)
            )
            return spin

        else:
            # String field
            field = QLineEdit(str(param_value))
            field.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
            field.editingFinished.connect(
                lambda f=field, pn=param_name: self._save_neural_node_param(pn, f.text())
            )
            return field

    def _format_param_label(self, param_name: str) -> str:
        """Format parameter name for display."""
        label_map = {
            'show_on_start': 'Show on Start',
            'text': 'Text',
        }
        return label_map.get(param_name, param_name)

    def _save_neural_node_field(self, field_name: str, value):
        """Save a basic field (name) to the neural node with undo support."""
        if not hasattr(self, '_current_neural_node_id'):
            return

        node_id = self._current_neural_node_id

        main_window = self.window()
        if not hasattr(main_window, 'neural_canvas'):
            return

        canvas_view = main_window.neural_canvas.canvas_view
        node = main_window.neural_canvas.graph.get_node_by_id(node_id)
        if not node:
            return

        if field_name == 'name':
            old_name = node.name
            if old_name == value:
                return

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
        # Guard against re-entrant calls
        if getattr(self, '_saving_neural_param', False):
            return
        self._saving_neural_param = True

        try:
            if not hasattr(self, '_current_neural_node_id'):
                return

            node_id = self._current_neural_node_id

            main_window = self.window()
            if not hasattr(main_window, 'neural_canvas'):
                return

            canvas_view = main_window.neural_canvas.canvas_view
            node = main_window.neural_canvas.graph.get_node_by_id(node_id)
            if not node:
                return

            old_value = node.params.get(param_name)

            if old_value == value:
                return

            # Special handling for show_on_start - only one COMMENT can have it
            if param_name == 'show_on_start' and value is True:
                from ..core.neural_canvas.neural_node import NodeType
                for other_id, other_node in main_window.neural_canvas.graph.nodes.items():
                    if other_id != node_id and other_node.type == NodeType.COMMENT:
                        if other_node.params.get('show_on_start', False):
                            other_node.params['show_on_start'] = False

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
        """Update a displayed param value in the Inspector (called externally)."""
        if not hasattr(self, '_neural_node_param_widgets'):
            return

        widget = self._neural_node_param_widgets.get(param_name)
        if not widget:
            return

        widget.blockSignals(True)
        try:
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
