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
#   Inspector Panel - Component-based property editor
#
#   Shows and edits ALL properties of selected entity: - User...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_panel
# PURPOSE:  Inspector Panel - Component-based property editor
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   InspectorPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QScrollArea,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QProgressBar, QListWidget,
                             QFileDialog, QListWidgetItem, QApplication)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon, QFontMetrics
import os
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

# Import mixins
from .inspector_base import InspectorBaseMixin, ClickableTextEdit
from .inspector_entity import EntityInspectorMixin
from .inspector_asset import AssetInspectorMixin
from .inspector_neural import NeuralInspectorMixin
from .inspector_components import ComponentInspectorMixin
from .inspector_ui_canvas import UICanvasInspectorMixin


class InspectorPanel(
    InspectorBaseMixin,
    EntityInspectorMixin,
    AssetInspectorMixin,
    NeuralInspectorMixin,
    ComponentInspectorMixin,
    UICanvasInspectorMixin,
    QWidget
):
    """
    Component-based Inspector panel.

    Shows editable properties for selected entity.
    Every field is live-editable with instant save!

    Inherits from mixins for modular code organization:
    - InspectorBaseMixin: UI utilities
    - EntityInspectorMixin: Entity loading
    - AssetInspectorMixin: Asset loading
    - NeuralInspectorMixin: Neural canvas node loading
    - ComponentInspectorMixin: Component system display
    - UICanvasInspectorMixin: UI canvas component editing
    """

    # Signal emitted when entity name is changed (entity_type, entity_id, new_name)
    nameChanged = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize base inspector state (property_fields, component_widgets,
        # collapsible_states, is_loading, _bound_widgets)
        self.init_base_inspector()

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

        # Track CollapsibleSection expanded state (like SceneHierarchy does)
        # Structure: {section_title: bool}
        self.collapsible_expanded_state = {}

        # Current entity's component collection
        self._current_components = None

        # Initialize facet dropdown and container (set to None until agent loaded)
        self.facet_dropdown = None
        self.facet_properties_container = None
        self.facet_properties_layout = None
        self.current_assembly = None

        # Property binding manager for automatic undo support
        self._binding_manager = PropertyBindingManager(self)

        # Initialize UI Canvas inspector mixin
        self.init_ui_canvas_inspector()

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
        layout.setContentsMargins(8, 8, 16, 8)  # Extra right margin

        # Header (hidden - redundant with Name field in properties)
        self.entity_header = QLabel("No entity selected")
        self.entity_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.entity_header.setStyleSheet("color: #D2D2D2; padding: 8px;")
        self.entity_header.hide()  # Name is shown in properties section
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

        self.current_entity = None
        self.current_agent_id = None
        self.current_facet = None
        self.entity_header.setText("Select a thing to inspect its properties")

        # Clear property bindings (important for undo system)
        self._binding_manager.clear_bindings()

        # Clear existing properties
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.component_widgets.clear()

    # ========== FACET LOADING (kept inline - complex interdependencies) ==========

    def load_facet(self, facet):
        """
        Load dedicated facet inspector view.

        Unity pattern: click a component in the scene -> inspector shows
        that component's properties. Replaces full inspector content.

        Args:
            facet: Facet object from facet_system.py (None to deselect)
        """
        if facet is None:
            self.current_facet = None
            # Restore noodling view if available
            if self.last_entity_type and self.last_entity_data:
                self.current_entity = None  # Force past same-entity guard
                self.load_entity(self.last_entity_type, self.last_entity_data)
            return

        self._load_facet_standalone(facet)

    def _load_facet_standalone(self, facet):
        """
        Build dedicated facet inspector view with full type-specific properties.

        Replaces all inspector content. When a noodling is re-selected in
        the hierarchy, load_entity() restores the noodling view.
        """
        from PyQt6.QtWidgets import QComboBox

        self.clear_inspector()
        self.current_facet = facet
        self.current_entity = None  # Allow noodling reload on re-selection
        self.entity_header.setText(f"Facet: {facet.name}")

        try:
            # --- Basic Properties ---
            basic_section = CollapsibleSection("Basic Properties")
            basic_form = QFormLayout()

            # Name (editable with undo)
            name_meta = PropertyMeta(name='name', prop_type=str, display_name='Name')
            self.create_bound_lineedit(facet, name_meta, basic_form)

            # ID (read-only)
            id_field = QLineEdit(facet.id)
            id_field.setReadOnly(True)
            id_field.setStyleSheet("color: #888;")
            basic_form.addRow("ID:", id_field)

            # Type (read-only)
            type_field = QLineEdit(facet.facet_type)
            type_field.setReadOnly(True)
            type_field.setStyleSheet("color: #888;")
            basic_form.addRow("Type:", type_field)

            # Enabled (with undo)
            enabled_meta = PropertyMeta(name='enabled', prop_type=bool, display_name='Enabled')
            self.create_bound_checkbox(facet, enabled_meta, basic_form)

            basic_section.set_content_layout(basic_form)
            self.properties_layout.addWidget(basic_section)

            # --- Type-specific Properties ---
            # Non-LLM types get their own sections; everything else
            # (LLM, LLMFacet, IntuitionFacet, EmotionFacet, etc.)
            # defaults to LLM configuration since all cognitive facets
            # have prompt, model, temperature, max_tokens.
            if facet.facet_type == "NeuralCanvasFacet":
                self._build_neural_canvas_facet_section(facet)
            elif facet.facet_type == "CharmNetworkFacet":
                charm_section = CollapsibleSection("Charm Network")
                charm_form = QFormLayout()
                info = QLabel("Neural affect model (PAD + boredom + sorrow)")
                info.setStyleSheet("color: #888888; font-style: italic;")
                charm_form.addRow(info)
                charm_section.set_content_layout(charm_form)
                self.properties_layout.addWidget(charm_section)
            elif facet.facet_type == "ScriptedFacet":
                self._build_scripted_facet_section(facet)
            elif facet.facet_type in ("INCOMING", "OUTGOING"):
                pass  # Terminal nodes have no extra configuration
            else:
                # LLM Configuration for all cognitive/LLM facets
                self._build_llm_facet_section(facet)

            # --- Inputs & Outputs ---
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

            # Execution data if available
            self._add_execution_data_section(facet)

        except Exception as e:
            print(f"[Inspector] Error building facet properties UI: {e}")
            import traceback
            traceback.print_exc()

            error_label = QLabel(f"Error loading facet properties:\n{str(e)}")
            error_label.setStyleSheet("color: #FF6B6B;")
            error_label.setWordWrap(True)
            self.properties_layout.addWidget(error_label)

    # ========== TYPE-SPECIFIC FACET SECTION BUILDERS ==========

    def _build_llm_facet_section(self, facet):
        """Build LLM Configuration section: Model, Temperature, Max Tokens, Prompt."""
        from PyQt6.QtWidgets import QComboBox

        llm_section = CollapsibleSection("LLM Configuration")
        llm_form = QFormLayout()

        # Model dropdown
        model_combo = QComboBox()
        model_combo.addItems(["Small", "Medium", "Large"])
        model_combo.setStyleSheet("""
            QComboBox {
                background: #3e3e3e; color: #D2D2D2;
                border: 1px solid #555555; padding: 4px 8px;
                padding-right: 25px; border-radius: 3px; min-width: 100px;
            }
            QComboBox:hover { border: 1px solid #666666; }
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 20px; border-left: 1px solid #555555;
            }
            QComboBox::down-arrow {
                image: none; border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #D2D2D2;
                width: 0px; height: 0px; margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: #3e3e3e; color: #D2D2D2;
                selection-background-color: #555555; border: 1px solid #555555;
            }
        """)

        current_model = facet.model or "Medium"
        index = model_combo.findText(
            current_model.title() if isinstance(current_model, str) else str(current_model),
            Qt.MatchFlag.MatchFixedString
        )
        if index < 0:
            index = model_combo.findText(
                str(current_model).upper(), Qt.MatchFlag.MatchFixedString
            )
        if index >= 0:
            model_combo.setCurrentIndex(index)

        model_combo._last_value = facet.model or "Medium"

        def on_model_changed(text, f=facet, combo=model_combo):
            old_val = getattr(combo, '_last_value', f.model)
            new_val = text.upper()
            setattr(f, 'model', new_val)
            self._push_facet_property_command(f, 'model', old_val, new_val)
            combo._last_value = new_val

        model_combo.currentTextChanged.connect(on_model_changed)
        llm_form.addRow("Model:", model_combo)

        # Temperature
        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.1)
        initial_temp = facet.temperature or 0.7
        temp_spin.setValue(initial_temp)
        temp_spin._last_value = initial_temp

        def on_temp_changed(val, f=facet, spin=temp_spin):
            old_val = getattr(spin, '_last_value', f.temperature)
            setattr(f, 'temperature', val)
            self._push_facet_property_command(f, 'temperature', old_val, val)
            spin._last_value = val

        temp_spin.valueChanged.connect(on_temp_changed)
        llm_form.addRow("Temperature:", temp_spin)

        # Max Tokens
        tokens_spin = QSpinBox()
        tokens_spin.setRange(1, 4096)
        initial_tokens = facet.max_tokens or 150
        tokens_spin.setValue(initial_tokens)
        tokens_spin._last_value = initial_tokens

        def on_tokens_changed(val, f=facet, spin=tokens_spin):
            old_val = getattr(spin, '_last_value', f.max_tokens)
            setattr(f, 'max_tokens', val)
            self._push_facet_property_command(f, 'max_tokens', old_val, val)
            spin._last_value = val

        tokens_spin.valueChanged.connect(on_tokens_changed)
        llm_form.addRow("Max Tokens:", tokens_spin)

        # Prompt (with floating editor on Cmd+Click)
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
        prompt_edit._baseline_value = facet.prompt or ""

        def on_prompt_changed(edit=prompt_edit, f=facet):
            new_text = edit.toPlainText()
            old_text = getattr(edit, '_baseline_value', '')
            setattr(f, 'prompt', new_text)
            self._push_facet_property_command(f, 'prompt', old_text, new_text)

        prompt_edit.textChanged.connect(on_prompt_changed)
        llm_form.addRow("Prompt:", prompt_edit)

        llm_section.set_content_layout(llm_form)
        self.properties_layout.addWidget(llm_section)

    def _build_neural_canvas_facet_section(self, facet):
        """Build NeuralCanvasFacet section: NNCanvas path with browse button."""
        nc_section = CollapsibleSection("Neural Canvas")
        nc_form = QFormLayout()

        path_layout = QHBoxLayout()
        path_edit = QLineEdit(facet.nncanvas_path or "")
        path_edit.setPlaceholderText("Path to .nncanvas file...")
        path_edit.setStyleSheet("background-color: #2D2D2D; color: #D2D2D2; padding: 4px;")
        path_edit._baseline_value = facet.nncanvas_path or ""

        def on_path_changed(text, edit=path_edit, f=facet):
            old_val = getattr(edit, '_baseline_value', '')
            if text != old_val:
                f.nncanvas_path = text
                self._push_facet_property_command(f, 'nncanvas_path', old_val, text)
                edit._baseline_value = text

        path_edit.editingFinished.connect(lambda: on_path_changed(path_edit.text()))
        path_layout.addWidget(path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040; color: #D2D2D2;
                border: 1px solid #555555; padding: 4px 8px;
            }
            QPushButton:hover { background-color: #505050; }
        """)

        def browse_nncanvas():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select NNCanvas File", "",
                "NNCanvas Files (*.nncanvas);;All Files (*)"
            )
            if file_path:
                path_edit.setText(file_path)
                on_path_changed(file_path)

        browse_btn.clicked.connect(browse_nncanvas)
        path_layout.addWidget(browse_btn)

        path_widget = QWidget()
        path_widget.setLayout(path_layout)
        nc_form.addRow("NNCanvas Path:", path_widget)

        info_label = QLabel("Visual neural network from NeuralCanvas editor")
        info_label.setStyleSheet("color: #9C27B0; font-style: italic;")
        nc_form.addRow(info_label)

        nc_section.set_content_layout(nc_form)
        self.properties_layout.addWidget(nc_section)

    # ------------------------------------------------------------------
    # Performance facet helpers (ScriptedFacet with typing timing)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_performance_facet(facet) -> bool:
        """Detect whether a ScriptedFacet is a Performance (typing timer)."""
        return (facet.facet_type == "ScriptedFacet"
                and 'performance_script' in (facet.prompt or ''))

    @staticmethod
    def _parse_performance_params(prompt: str) -> dict:
        """Extract tunable timing values from Performance facet JS code."""
        import re
        params = {}
        m = re.search(r'var\s+base_delay\s*=\s*(\d+)', prompt or '')
        if m:
            params['base_delay'] = int(m.group(1))
        m = re.search(r'var\s+speaking_intensity\s*=\s*([\d.]+)', prompt or '')
        if m:
            params['speaking_intensity'] = float(m.group(1))
        return params

    @staticmethod
    def _update_performance_param(prompt: str, param: str, value) -> str:
        """Replace a timing parameter value in Performance facet JS code."""
        import re
        if param == 'base_delay':
            return re.sub(
                r'(var\s+base_delay\s*=\s*)\d+',
                f'\\g<1>{int(value)}', prompt)
        elif param == 'speaking_intensity':
            return re.sub(
                r'(var\s+speaking_intensity\s*=\s*)[\d.]+',
                f'\\g<1>{value:.1f}', prompt)
        return prompt

    def _build_scripted_facet_section(self, facet):
        """Build ScriptedFacet section -- friendly controls for Performance facets."""
        if self._is_performance_facet(facet):
            self._build_performance_facet_section(facet)
            return

        # Generic ScriptedFacet: salience script editor
        script_section = CollapsibleSection("Script")
        script_form = QFormLayout()

        script_edit = ClickableTextEdit(
            field_name=f"{facet.name} - Salience Script",
            on_apply_callback=lambda text, f=facet: (
                setattr(f, 'salience_script', text),
                self._push_facet_property_command(f, 'salience_script', f.salience_script, text)
            )
        )
        script_edit.setPlainText(facet.salience_script or "")
        script_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px; font-family: 'Courier New';"
        )
        script_edit.setMaximumHeight(150)
        script_edit._baseline_value = facet.salience_script or ""

        def on_script_changed(edit=script_edit, f=facet):
            new_text = edit.toPlainText()
            old_text = getattr(edit, '_baseline_value', '')
            setattr(f, 'salience_script', new_text)
            self._push_facet_property_command(f, 'salience_script', old_text, new_text)

        script_edit.textChanged.connect(on_script_changed)
        script_form.addRow("Salience Script:", script_edit)

        script_section.set_content_layout(script_form)
        self.properties_layout.addWidget(script_section)

    def _build_performance_facet_section(self, facet):
        """Build friendly inspector for Performance ScriptedFacet.

        Shows typing speed and speaking intensity as first-class controls
        instead of raw JavaScript code.
        """
        from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox

        params = self._parse_performance_params(facet.prompt)

        perf_section = CollapsibleSection("Performance Timing")
        perf_form = QFormLayout()

        # -- Typing Speed (base_delay in ms) --
        speed_spin = QSpinBox()
        speed_spin.setRange(20, 80)
        speed_spin.setSuffix(" ms")
        speed_spin.setValue(params.get('base_delay', 35))
        speed_spin.setToolTip("Milliseconds per character (lower = faster typing)")
        speed_spin.setStyleSheet(
            "QSpinBox { background: #2D2D2D; color: #D2D2D2; "
            "border: 1px solid #555; padding: 4px; }"
        )
        speed_spin._last_value = speed_spin.value()

        def on_speed_changed(val, f=facet, spin=speed_spin):
            old_val = spin._last_value
            old_prompt = f.prompt
            f.prompt = self._update_performance_param(f.prompt, 'base_delay', val)
            self._push_facet_property_command(f, 'prompt', old_prompt, f.prompt)
            spin._last_value = val

        speed_spin.valueChanged.connect(on_speed_changed)
        perf_form.addRow("Typing Speed:", speed_spin)

        # -- Speaking Intensity (0.0 - 1.0) --
        intensity_spin = QDoubleSpinBox()
        intensity_spin.setRange(0.0, 1.0)
        intensity_spin.setSingleStep(0.1)
        intensity_spin.setDecimals(1)
        intensity_spin.setValue(params.get('speaking_intensity', 0.7))
        intensity_spin.setToolTip("VRM mouth animation intensity while speaking")
        intensity_spin.setStyleSheet(
            "QDoubleSpinBox { background: #2D2D2D; color: #D2D2D2; "
            "border: 1px solid #555; padding: 4px; }"
        )
        intensity_spin._last_value = intensity_spin.value()

        def on_intensity_changed(val, f=facet, spin=intensity_spin):
            old_val = spin._last_value
            old_prompt = f.prompt
            f.prompt = self._update_performance_param(
                f.prompt, 'speaking_intensity', val)
            self._push_facet_property_command(f, 'prompt', old_prompt, f.prompt)
            spin._last_value = val

        intensity_spin.valueChanged.connect(on_intensity_changed)
        perf_form.addRow("Speaking Intensity:", intensity_spin)

        # -- Punctuation pause info (read-only) --
        pause_label = QLabel(
            "Punctuation pauses: .  !  ?  ,  :  ;\n"
            "Edit the script below for custom pause values."
        )
        pause_label.setStyleSheet("color: #888; font-size: 9pt;")
        pause_label.setWordWrap(True)
        perf_form.addRow(pause_label)

        perf_section.set_content_layout(perf_form)
        self.properties_layout.addWidget(perf_section)

        # -- Advanced: raw script (collapsed by default) --
        advanced_section = CollapsibleSection("Advanced Script")
        advanced_section.toggle_button.setChecked(False)  # Start collapsed
        advanced_form = QFormLayout()

        script_edit = ClickableTextEdit(
            field_name=f"{facet.name} - Script",
            on_apply_callback=lambda text, f=facet: (
                setattr(f, 'prompt', text),
                self._push_facet_property_command(f, 'prompt', f.prompt, text)
            )
        )
        script_edit.setPlainText(facet.prompt or "")
        script_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; "
            "padding: 4px; font-family: 'Courier New';"
        )
        script_edit.setMaximumHeight(150)
        script_edit._baseline_value = facet.prompt or ""

        def on_script_changed(edit=script_edit, f=facet):
            new_text = edit.toPlainText()
            old_text = getattr(edit, '_baseline_value', '')
            setattr(f, 'prompt', new_text)
            self._push_facet_property_command(f, 'prompt', old_text, new_text)

        script_edit.textChanged.connect(on_script_changed)
        advanced_form.addRow("Script:", script_edit)

        advanced_section.set_content_layout(advanced_form)
        self.properties_layout.addWidget(advanced_section)

    def _auto_save_facet_assembly(self):
        """Auto-save the current facet assembly to its YAML file."""
        try:
            main_window = self.window()
            editor = getattr(main_window, 'unified_editor', None)
            if editor and hasattr(editor, '_save_assembly_to_disk'):
                editor._save_assembly_to_disk()
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
        editor = getattr(main_window, 'unified_editor', None)
        if not editor:
            return

        assembly = editor.current_assembly
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

        Rebuilds the facet inspector view from current data. The data model
        is already updated (setattr was called before this), so rebuilding
        the widgets shows the correct values with fresh baseline trackers.
        """
        if self.current_facet and self.current_facet.id == facet_id:
            self._load_facet_standalone(self.current_facet)

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

    # ========== FACET DROPDOWN SYSTEM ==========

    def _get_agent_assembly(self, agent_id: str, agent_data: dict):
        """
        Load agent's facet assembly from the noodling template directory.

        Resolves the assembly path from entity_data's instance path + noodling_ref,
        matching the same pattern used by the facets editor
        (_load_facet_assembly_for_noodling in main_window_signals_mixin).

        Args:
            agent_id: Agent ID (UUID like "agent_a56e0ac2...")
            agent_data: Entity data dict with 'path' and 'noodling_ref'

        Returns:
            FacetAssembly or None if not found
        """
        try:
            from ..core.facet_system import FacetAssembly
            import os

            instance_path = agent_data.get('path', '')
            noodling_ref = agent_data.get('noodling_ref', '')

            assembly_path = None

            # Primary: resolve from instance path + noodling_ref (relative path)
            if noodling_ref and instance_path:
                noodling_dir = os.path.normpath(os.path.join(instance_path, noodling_ref))
                candidate = os.path.join(noodling_dir, 'assembly.yaml')
                if os.path.exists(candidate):
                    assembly_path = candidate

            # Fallback: bundled library (for templates with simple names)
            if not assembly_path and noodling_ref:
                simple_name = os.path.basename(noodling_ref)
                noodlestudio_dir = os.path.join(os.path.dirname(__file__), '../..')
                candidate = os.path.join(
                    noodlestudio_dir, 'library', 'noodlings', simple_name, 'assembly.yaml'
                )
                if os.path.exists(candidate):
                    assembly_path = candidate

            if not assembly_path:
                print(f"[Inspector] No assembly found for {agent_id}")
                return None

            assembly = FacetAssembly.load_yaml(str(assembly_path))
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

        # Load agent's facet assembly
        assembly = self._get_agent_assembly(agent_id, agent_data)
        if not assembly:
            print(f"[Inspector] ERROR: No assembly loaded, cannot create facet dropdown")
            return

        print(f"[Inspector] Assembly loaded: {assembly.name} with {len(assembly.facets)} facets")

        # Store assembly reference for dropdown updates
        self.current_assembly = assembly

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555555; max-height: 2px;")
        self.properties_layout.addWidget(separator)

        # Facet selector section
        facet_selector_group = self.create_property_group("Facet")

        # Dropdown with facet names
        self.facet_dropdown = QComboBox()
        self.facet_dropdown.addItem("(none)", None)  # Default empty selection

        for facet in assembly.facets:
            self.facet_dropdown.addItem(facet.name, facet.id)

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
                    QTimer.singleShot(0, lambda: self._load_facet_properties_inline(facet))
            else:
                # Clear facet properties
                self._clear_facet_properties_inline()

        self.facet_dropdown.currentIndexChanged.connect(on_facet_dropdown_changed)
        facet_selector_group.content.layout().addRow("Select:", self.facet_dropdown)

        self.properties_layout.addWidget(facet_selector_group)

        # Container for facet properties (populated when dropdown changes)
        self.facet_properties_container = QWidget()
        self.facet_properties_layout = QVBoxLayout(self.facet_properties_container)
        self.facet_properties_layout.setContentsMargins(0, 0, 0, 0)
        self.properties_layout.addWidget(self.facet_properties_container)

        print(f"[Inspector] FACET DROPDOWN SETUP COMPLETE")

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

        # NeuralCanvasFacet
        elif facet.facet_type == "NeuralCanvasFacet":
            # NNCanvas path field with browse button
            from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QFileDialog

            path_layout = QHBoxLayout()
            path_edit = QLineEdit(facet.nncanvas_path or "")
            path_edit.setPlaceholderText("Path to .nncanvas file...")
            path_edit.setStyleSheet("background-color: #2D2D2D; color: #D2D2D2; padding: 4px;")
            path_edit._baseline_value = facet.nncanvas_path or ""

            def on_path_changed(text, edit=path_edit, f=facet):
                old_val = getattr(edit, '_baseline_value', '')
                if text != old_val:
                    f.nncanvas_path = text
                    self._push_facet_property_command(f, 'nncanvas_path', old_val, text)
                    edit._baseline_value = text

            path_edit.editingFinished.connect(lambda: on_path_changed(path_edit.text()))
            path_layout.addWidget(path_edit)

            browse_btn = QPushButton("Browse...")
            browse_btn.setMaximumWidth(80)
            browse_btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
            """)

            def browse_nncanvas():
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select NNCanvas File",
                    "",
                    "NNCanvas Files (*.nncanvas);;All Files (*)"
                )
                if file_path:
                    path_edit.setText(file_path)
                    on_path_changed(file_path)

            browse_btn.clicked.connect(browse_nncanvas)
            path_layout.addWidget(browse_btn)

            path_widget = QWidget()
            path_widget.setLayout(path_layout)
            props_layout.addRow("NNCanvas Path:", path_widget)

            info_label = QLabel("Visual neural network from NeuralCanvas editor")
            info_label.setStyleSheet("color: #9C27B0; font-style: italic;")  # Purple for neural
            props_layout.addRow(info_label)

        # ScriptedFacet (with undo support)
        elif facet.facet_type == "ScriptedFacet":
            if self._is_performance_facet(facet):
                # Performance facet: friendly timing controls
                from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox
                params = self._parse_performance_params(facet.prompt)

                speed_spin = QSpinBox()
                speed_spin.setRange(20, 80)
                speed_spin.setSuffix(" ms")
                speed_spin.setValue(params.get('base_delay', 35))
                speed_spin.setToolTip("Milliseconds per character")
                speed_spin.setStyleSheet(
                    "QSpinBox { background: #2D2D2D; color: #D2D2D2; "
                    "border: 1px solid #555; padding: 4px; }")
                speed_spin._last_value = speed_spin.value()

                def on_speed_changed(val, f=facet, spin=speed_spin):
                    old_prompt = f.prompt
                    f.prompt = self._update_performance_param(
                        f.prompt, 'base_delay', val)
                    self._push_facet_property_command(
                        f, 'prompt', old_prompt, f.prompt)
                    spin._last_value = val

                speed_spin.valueChanged.connect(on_speed_changed)
                props_layout.addRow("Typing Speed:", speed_spin)

                intensity_spin = QDoubleSpinBox()
                intensity_spin.setRange(0.0, 1.0)
                intensity_spin.setSingleStep(0.1)
                intensity_spin.setDecimals(1)
                intensity_spin.setValue(
                    params.get('speaking_intensity', 0.7))
                intensity_spin.setToolTip("VRM mouth animation intensity")
                intensity_spin.setStyleSheet(
                    "QDoubleSpinBox { background: #2D2D2D; color: #D2D2D2; "
                    "border: 1px solid #555; padding: 4px; }")
                intensity_spin._last_value = intensity_spin.value()

                def on_intensity_changed(val, f=facet, spin=intensity_spin):
                    old_prompt = f.prompt
                    f.prompt = self._update_performance_param(
                        f.prompt, 'speaking_intensity', val)
                    self._push_facet_property_command(
                        f, 'prompt', old_prompt, f.prompt)
                    spin._last_value = val

                intensity_spin.valueChanged.connect(on_intensity_changed)
                props_layout.addRow("Speaking Intensity:", intensity_spin)
            else:
                # Generic ScriptedFacet: salience script editor
                script_edit = ClickableTextEdit(
                    field_name=f"{facet.name} - Salience Script",
                    on_apply_callback=lambda text, f=facet: (
                        setattr(f, 'salience_script', text),
                        self._push_facet_property_command(
                            f, 'salience_script', f.salience_script, text)
                    )
                )
                script_edit.setPlainText(facet.salience_script or "")
                script_edit.setStyleSheet(
                    "background-color: #1E1E1E; color: #D2D2D2; "
                    "padding: 4px; font-family: 'Courier New';")
                script_edit.setMaximumHeight(150)
                script_edit._baseline_value = facet.salience_script or ""

                def on_script_changed(edit=script_edit, f=facet):
                    new_text = edit.toPlainText()
                    old_text = getattr(edit, '_baseline_value', '')
                    setattr(f, 'salience_script', new_text)
                    self._push_facet_property_command(
                        f, 'salience_script', old_text, new_text)

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
            # Find assembly editor to get node graphics
            main_window = self.window()
            editor = getattr(main_window, 'unified_editor', None)
            if not editor:
                return

            # Check if node_graphics exists and has this facet
            if not hasattr(editor, 'node_graphics') or not editor.node_graphics:
                return

            if facet.id not in editor.node_graphics:
                return

            node = editor.node_graphics[facet.id]
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

            self.properties_layout.addWidget(exec_section)

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

    # ========== ENTITY LOADING DISPATCHER ==========

    @pyqtSlot(str, dict)
    def load_entity(self, entity_type: str, entity_data: dict):
        """Load entity properties into inspector."""
        # Save for later restore when facet is deselected
        # CRITICAL: Store a COPY to avoid reference mutation issues
        if entity_type and entity_data:
            self.last_entity_type = entity_type
            self.last_entity_data = entity_data.copy()

        # CRITICAL: Prevent re-entrant loading (e.g., double-tap events)
        if self.is_loading:
            print(f"[DIAGNOSTIC] BLOCKING re-entrant load_entity call (is_loading=True)")
            return

        # Handle deselection (nothing selected)
        if not entity_type or not entity_data:
            self.clear_inspector()
            return

        # CRITICAL: Check if same entity - don't reload if it hasn't changed
        # But DO reload if the name changed (from inline rename)
        if self.current_entity:
            old_type, old_data = self.current_entity
            old_id = old_data.get('id') if old_data else None
            new_id = entity_data.get('id')
            old_name = old_data.get('name') if old_data else None
            new_name = entity_data.get('name')
            if old_type == entity_type and old_id == new_id and old_name == new_name:
                print(f"[DIAGNOSTIC] SKIPPING load_entity - same entity already loaded")
                return

        # CRITICAL: Don't reload if save is in progress (guard BEFORE assignment
        # to prevent current_entity from being set without a full reload)
        if self.is_saving:
            print(f"[DIAGNOSTIC] SKIPPING load_entity - save in progress")
            return

        # CRITICAL: Don't reload if a text widget has focus (user is editing)
        focused_widget = QApplication.focusWidget()
        if focused_widget and (isinstance(focused_widget, QLineEdit) or isinstance(focused_widget, QTextEdit)):
            print(f"[DIAGNOSTIC] SKIPPING load_entity - text widget has focus")
            return

        # CRITICAL: Store a COPY of entity_data to avoid reference mutation
        # (Stage View modifies entity_data dict before emitting, which would
        # corrupt our comparison on next load if we held a reference)
        # NOTE: This MUST come after all guards -- if a guard returns early,
        # current_entity must stay at its previous value so the skip guard
        # at lines 1030-1038 doesn't incorrectly match on next selection.
        self.current_entity = (entity_type, entity_data.copy())

        # Set loading flag to prevent re-entrance
        self.is_loading = True

        try:
            # CRITICAL: Save CollapsibleSection expanded state before destroying widgets
            self._save_collapsible_states()

            # Clear facet mode when loading an entity
            self.current_facet = None

            # Clear existing properties
            while self.properties_layout.count():
                child = self.properties_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Clear component widget tracking for clean slate
            self.component_widgets.clear()

            # Update header - just show the name (no type prefix)
            if entity_type == 'noodling':
                name = entity_data.get('name') or entity_data.get('data', {}).get('name', entity_data.get('id', 'Noodling'))
                self.entity_header.setText(name)
                self.load_noodling_properties(entity_data)

            elif entity_type == 'user':
                self.entity_header.setText("caity")
                self.load_user_properties(entity_data)

            elif entity_type in ('prim', 'object', 'prop'):
                obj_name = entity_data.get('name') or entity_data.get('id', 'Unknown').replace('obj_', '').replace('_', ' ').title()
                self.entity_header.setText(obj_name)
                self.load_object_properties(entity_data)

            elif entity_type == 'exit':
                direction = entity_data.get('direction', 'unknown')
                self.entity_header.setText(direction)
                self.load_exit_properties(entity_data)

            elif entity_type == 'stage':
                stage_name = entity_data.get('data', {}).get('name', 'Stage')
                self.entity_header.setText(stage_name)
                self.load_stage_properties(entity_data)

            elif entity_type == 'zone':
                zone_name = entity_data.get('name') or entity_data.get('data', {}).get('name', 'Zone')
                self.entity_header.setText(zone_name)
                self.load_zone_properties(entity_data)

            elif entity_type == 'neural_node':
                node_name = entity_data.get('name', 'Node')
                self.entity_header.setText(node_name)
                self.load_neural_node_properties(entity_data)

            elif entity_type == 'radiance':
                asset_name = entity_data.get('name', 'Radiance')
                self.entity_header.setText(asset_name)
                self.load_radiance_properties(entity_data)

            elif entity_type == 'asset':
                # Asset from Assets panel - dispatch to sub-type handler
                self.load_asset_properties(entity_data)

            elif entity_type in ('ui', 'ui_component'):
                # UI Canvas or UI Component - use UICanvasInspectorMixin
                component = entity_data.get('component')
                if component:
                    comp_name = entity_data.get('name', 'UI')
                    self.entity_header.setText(comp_name)
                    self.load_ui_component(component)

        finally:
            # ALWAYS clear loading flag, even on error
            self.is_loading = False

    # ========== COLLAPSIBLE STATE MANAGEMENT ==========

    def _save_collapsible_states(self):
        """
        Save expanded/collapsed state of all CollapsibleSections before widget rebuild.

        Pattern copied from SceneHierarchy.save_expanded_state() to prevent
        bounce-back when timer refreshes Inspector.
        """
        # Find all CollapsibleSection widgets in the properties layout
        for i in range(self.properties_layout.count()):
            item = self.properties_layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, CollapsibleSection):
                    self.collapsible_expanded_state[widget.title_text] = widget.is_expanded

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

    # ========== SAVE METHODS ==========

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
            QTimer.singleShot(2500, lambda: setattr(self, 'is_saving', False))

    # ========== COMPONENTS (legacy system) ==========

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
        from PyQt6.QtWidgets import QCheckBox

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

        # Create live-updating labels for 5D affect
        affect_dims = [
            ('valence', 'Valence'),
            ('arousal', 'Arousal'),
            ('dominance', 'Dominance'),
            ('boredom', 'Boredom'),
            ('sorrow', 'Sorrow')
        ]

        for dim_key, dim_label in affect_dims:
            value_label = QLabel("0.00")
            value_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
            self.live_affect_labels[dim_key] = value_label
            affect_layout.addRow(f"{dim_label}:", value_label)

        affect_widget = QWidget()
        affect_widget.setLayout(affect_layout)
        layout.addWidget(affect_widget)

        # Surprise (LIVE)
        surprise_section = QLabel("Surprise Metric")
        surprise_section.setStyleSheet("color: #D2D2D2; font-weight: bold; margin-top: 8px;")
        layout.addWidget(surprise_section)

        self.live_surprise_label = QLabel("0.00")
        self.live_surprise_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
        layout.addWidget(self.live_surprise_label)

        return component

    def update_live_data(self):
        """Update live data (affect, surprise) from agent state."""
        if not self.current_agent_id:
            return

        try:
            # Fetch current agent state
            response = requests.get(
                f"{self.api_base}/agents/{self.current_agent_id}",
                timeout=1
            )

            if response.status_code != 200:
                return

            agent_data = response.json()
            state = agent_data.get('state', {})

            # Update affect labels
            affect = state.get('affect', {})
            for dim_key, label in self.live_affect_labels.items():
                value = affect.get(dim_key, 0.0)
                label.setText(f"{value:.2f}")

            # Update surprise
            surprise = state.get('surprise', 0.0)
            if self.live_surprise_label:
                self.live_surprise_label.setText(f"{surprise:.2f}")

        except Exception:
            pass  # Silently ignore update failures

    def open_facet_editor(self, agent_id: str):
        """Open Assembly tab with agent's assembly loaded."""
        main_window = self.window()
        editor = getattr(main_window, 'unified_editor', None)
        if editor:
            # Switch to Assembly tab
            if hasattr(main_window, 'center_tabs'):
                for i in range(main_window.center_tabs.count()):
                    if main_window.center_tabs.tabText(i) == "Assembly":
                        main_window.center_tabs.setCurrentIndex(i)
                        break
            # Load agent's assembly
            if hasattr(editor, 'load_agent_assembly'):
                editor.load_agent_assembly(agent_id)

    def create_metadata_component(self, obj_id: str) -> CollapsibleSection:
        """Create arbitrary metadata editor for prims."""
        section = CollapsibleSection("Custom Properties")

        # Connect toggled signal
        section.toggled.connect(lambda expanded: self._on_collapsible_toggled(section.title_text, expanded))

        # Restore previous state
        self._restore_collapsible_state(section)

        info_label = QLabel("Add custom key-value properties")
        info_label.setStyleSheet("color: #808080; font-style: italic; padding: 4px;")
        section.add_widget(info_label)

        # TODO: Add key-value editor UI

        return section

    def _show_text_editor_context_menu(self, field: QTextEdit, pos):
        """Show context menu for text fields with external editor option."""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(field)

        # Standard edit actions
        menu.addAction("Cut", field.cut)
        menu.addAction("Copy", field.copy)
        menu.addAction("Paste", field.paste)
        menu.addSeparator()

        # External editor action
        edit_action = menu.addAction("Edit in External Editor...")
        edit_action.triggered.connect(lambda: self._open_in_external_editor(field))

        menu.exec(field.mapToGlobal(pos))

    def _open_in_external_editor(self, field: QTextEdit):
        """Open text field content in system's default text editor."""
        import tempfile
        import subprocess

        content = field.toPlainText()

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        # Open with system default
        subprocess.run(['open', temp_path])

        # TODO: Watch for file changes and reload

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
