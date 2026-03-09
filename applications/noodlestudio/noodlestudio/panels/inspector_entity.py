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
#   Entity Inspector Mixin - Scene entity loading (noodling, zone, prop, etc)
#
#   Handles inspection of scene entities selected from Stage ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_entity
# PURPOSE:  Inspector Entity
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EntityInspectorMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QTextEdit, QDoubleSpinBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer
import os
import yaml
import requests


class EntityInspectorMixin:
    """
    Mixin providing entity inspection methods.

    Requires host class to have:
    - self.properties_layout (QVBoxLayout)
    - self.property_fields (dict)
    - self.current_entity (tuple)
    - self.api_base (str)
    - self.create_property_group(title)
    - self.add_text_field(group, label, value, read_only)
    - self.add_text_area(group, label, value)
    - self.add_vector3_field(group, label, values, read_only)
    - self.add_dropdown_field(group, label, value, options, on_change)
    - self.nameChanged signal
    """

    # ========== NOODLING PROPERTIES ==========

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

        # Initialize component collection for this entity
        self._init_component_collection(agent_id, entity_data)

        # Load full recipe data from YAML file
        recipe_data = self._load_noodling_recipe(entity_data, agent)

        # ===== NOODLING PROPERTIES (always visible) =====
        basics_group = self.create_property_group("Noodling")

        # Name (editable) - prefer entity_data['name'] (set by Stage View rename)
        name = entity_data.get('name') or recipe_data.get('name', agent.get('name', ''))
        self.property_fields['name'] = self.add_text_field(
            basics_group, "Name", name
        )

        # UUID (read-only) with copy button
        self._add_uuid_field(basics_group, agent_id, prefix='agent_')

        # Description (editable text area)
        description = recipe_data.get('description', agent.get('description', 'An empty noodling...'))
        self.property_fields['description'] = self.add_text_area(basics_group, "Description", description)

        # ===== ROLE (dropdown) =====
        role_options = ['(None)', 'Director', 'Performer']
        current_role = self._get_instance_override(entity_data, 'role', '')
        current_role_display = current_role.capitalize() if current_role else '(None)'
        self.property_fields['role'] = self.add_dropdown_field(
            basics_group, "Role", current_role_display, role_options,
            on_change=lambda name: self._on_role_changed(name, entity_data)
        )

        # ===== VRM MODEL (dropdown) =====
        vrm_items = self._discover_vrm_for_dropdown()
        vrm_options = ['(None)'] + [item['name'] for item in vrm_items]
        current_vrm = self._get_current_vrm_name(entity_data, vrm_items)

        self.property_fields['vrm_model'] = self.add_dropdown_field(
            basics_group, "VRM Model", current_vrm, vrm_options,
            on_change=lambda name: self._on_vrm_changed(name, entity_data, vrm_items)
        )

        # ===== MARK (dropdown) =====
        mark_items = self._discover_marks_for_dropdown(entity_data)
        mark_options = ['(None)'] + [item['name'] for item in mark_items]
        current_mark = self._get_current_mark_name(entity_data, mark_items)
        self.property_fields['mark'] = self.add_dropdown_field(
            basics_group, "Mark", current_mark, mark_options,
            on_change=lambda name: self._on_mark_changed(name, entity_data, mark_items)
        )

        # ===== ENSEMBLE ACTIVE (checkbox) =====
        ensemble_active = self._get_instance_override(entity_data, 'ensemble_active', False)
        self.property_fields['ensemble_active'] = self.add_checkbox_field(
            basics_group, "Ensemble Active", ensemble_active,
            on_change=lambda checked: self._on_ensemble_active_changed(checked, entity_data)
        )

        # ===== VISIBLE ON STAGE (checkbox) =====
        visible = self._get_instance_override(entity_data, 'visible', True)
        self.property_fields['visible'] = self.add_checkbox_field(
            basics_group, "Visible on Stage", visible,
            on_change=lambda checked: self._on_visible_changed(checked, entity_data)
        )

        self.properties_layout.addWidget(basics_group)

        # NOTE: Affect Baseline removed from inspector -- affect is computed
        # dynamically by charm networks, not set by static spinners.
        # Data remains in recipe.yaml for cmush/exporter compatibility.

        # NOTE: Facet dropdown removed -- facets get a dedicated inspector
        # view when clicked in the facets editor (B.10.5 _load_facet_standalone).

        # ===== COMPONENTS SECTION =====
        if hasattr(self, 'create_components_section_new') and hasattr(self, '_current_components'):
            try:
                components_widget = self.create_components_section_new(self._current_components)
                if components_widget:
                    self.properties_layout.addWidget(components_widget)
            except Exception as e:
                print(f"[Inspector] ERROR creating components section: {e}")

        self.properties_layout.addStretch()

    def _load_noodling_recipe(self, entity_data, agent) -> dict:
        """Load recipe data from YAML file.

        Uses _resolve_recipe_path() -- the same path resolution used by save --
        so that load and save always agree on which file to use.
        """
        recipe_data = {}
        try:
            recipe_path = self._resolve_recipe_path(entity_data)
            if recipe_path and os.path.exists(recipe_path):
                with open(recipe_path, 'r') as f:
                    recipe_data = yaml.safe_load(f) or {}

            # Apply overrides from instance
            overrides = agent.get('overrides', {})
            if overrides:
                for key, value in overrides.items():
                    if value:
                        recipe_data[key] = value

        except Exception as e:
            print(f"[Inspector] Error loading recipe: {e}")

        return recipe_data

    # ========== VRM / ENSEMBLE / VISIBILITY HELPERS ==========

    def _discover_vrm_for_dropdown(self) -> list:
        """Get VRM file list for the dropdown.

        Uses vrm_discovery.discover_vrm_files() with the current project root.
        """
        from noodlestudio.core.vrm_discovery import discover_vrm_files
        project_root = self._get_project_root()
        return discover_vrm_files(project_root)

    def _get_project_root(self) -> str:
        """Get current project root path from main window's project manager."""
        try:
            main_window = self.window()
            if hasattr(main_window, 'project_manager'):
                pm = main_window.project_manager
                if pm and hasattr(pm, 'current_project_path'):
                    return pm.current_project_path
        except Exception:
            pass
        return None

    def _get_current_vrm_name(self, entity_data: dict, vrm_items: list) -> str:
        """Resolve current VRM display name from instance or template.

        Checks instance override 'vrm_path' first (absolute or relative),
        then falls back to the template noodling.yaml vrm_path.
        Matches against vrm_items to find the display name.
        """
        instance_path = entity_data.get('path', '')

        # Check instance override first
        vrm_override = self._get_instance_override(entity_data, 'vrm_path', '')
        if vrm_override and instance_path:
            abs_vrm = os.path.normpath(os.path.join(instance_path, vrm_override))
            for item in vrm_items:
                if os.path.normpath(item['path']) == abs_vrm:
                    return item['name']

        # Fall back to noodling template's vrm_path
        noodling_ref = entity_data.get('noodling_ref', '')
        if noodling_ref and instance_path:
            noodling_dir = os.path.normpath(os.path.join(instance_path, noodling_ref))
            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            if os.path.exists(noodling_yaml):
                try:
                    with open(noodling_yaml) as f:
                        data = yaml.safe_load(f) or {}
                    vrm_ref = data.get('vrm_path', '')
                    if vrm_ref:
                        abs_vrm = os.path.normpath(os.path.join(noodling_dir, vrm_ref))
                        for item in vrm_items:
                            if os.path.normpath(item['path']) == abs_vrm:
                                return item['name']
                except Exception:
                    pass

        return '(None)'

    def _get_instance_override(self, entity_data: dict, key: str, default):
        """Read a value from instance.yaml overrides."""
        data = entity_data.get('data', {})
        overrides = data.get('overrides', {})
        return overrides.get(key, default)

    def _on_vrm_changed(self, vrm_name: str, entity_data: dict, vrm_items: list):
        """Handle VRM dropdown change."""
        if self.is_loading:
            return
        instance_path = entity_data.get('path', '')
        if not instance_path:
            return

        if vrm_name == '(None)':
            vrm_rel_path = ''
            abs_vrm_path = ''
        else:
            # Find the selected VRM item
            selected = next((item for item in vrm_items if item['name'] == vrm_name), None)
            if not selected:
                return
            # Store relative path from instance directory to VRM
            vrm_rel_path = os.path.relpath(selected['path'], instance_path)
            abs_vrm_path = selected['path']

        self._save_instance_override(instance_path, 'vrm_path', vrm_rel_path)
        # Signal for runtime wiring
        agent_id = entity_data.get('id', '')
        if hasattr(self, 'noodlingPropertyChanged'):
            self.noodlingPropertyChanged.emit(agent_id, 'vrm_path', abs_vrm_path)

    def _on_ensemble_active_changed(self, checked: bool, entity_data: dict):
        """Handle ensemble active checkbox change."""
        if self.is_loading:
            return
        instance_path = entity_data.get('path', '')
        if instance_path:
            self._save_instance_override(instance_path, 'ensemble_active', checked)
            agent_id = entity_data.get('id', '')
            if hasattr(self, 'noodlingPropertyChanged'):
                self.noodlingPropertyChanged.emit(agent_id, 'ensemble_active', checked)

    def _on_visible_changed(self, checked: bool, entity_data: dict):
        """Handle visible on stage checkbox change."""
        if self.is_loading:
            return
        instance_path = entity_data.get('path', '')
        if instance_path:
            self._save_instance_override(instance_path, 'visible', checked)
            agent_id = entity_data.get('id', '')
            if hasattr(self, 'noodlingPropertyChanged'):
                self.noodlingPropertyChanged.emit(agent_id, 'visible', checked)

    def _on_role_changed(self, role_name: str, entity_data: dict):
        """Handle role dropdown change."""
        if self.is_loading:
            return
        instance_path = entity_data.get('path', '')
        if not instance_path:
            return

        role_value = role_name.lower() if role_name != '(None)' else ''
        self._save_instance_override(instance_path, 'role', role_value)
        agent_id = entity_data.get('id', '')
        if hasattr(self, 'noodlingPropertyChanged'):
            self.noodlingPropertyChanged.emit(agent_id, 'role', role_value)

    # ========== MARK HELPERS ==========

    def _discover_marks_for_dropdown(self, entity_data: dict) -> list:
        """Find available blocking marks for the current stage.

        Returns list of dicts: [{'id': ..., 'name': ..., 'path': ...}]
        """
        instance_path = entity_data.get('path', '')
        if not instance_path:
            return []

        # Walk up from instance dir to find Marks/ sibling of Instances/
        # Instance path: .../Stages/the_nexus/Instances/ajo
        # Stage path:    .../Stages/the_nexus
        stage_path = os.path.normpath(os.path.join(instance_path, '..', '..'))
        marks_dir = os.path.join(stage_path, 'Marks')
        if not os.path.isdir(marks_dir):
            return []

        results = []
        for filename in sorted(os.listdir(marks_dir)):
            if not filename.endswith('.mark.yaml'):
                continue
            try:
                with open(os.path.join(marks_dir, filename), 'r') as f:
                    data = yaml.safe_load(f) or {}
                results.append({
                    'id': data.get('id', filename.replace('.mark.yaml', '')),
                    'name': data.get('name', filename.replace('.mark.yaml', '')),
                    'path': os.path.join(marks_dir, filename),
                })
            except Exception:
                pass

        return results

    def _get_current_mark_name(self, entity_data: dict, mark_items: list) -> str:
        """Resolve current mark display name from instance override."""
        mark_id = self._get_instance_override(entity_data, 'mark', '')
        if not mark_id:
            return '(None)'
        for item in mark_items:
            if item['id'] == mark_id:
                return item['name']
        return '(None)'

    def _on_mark_changed(self, mark_name: str, entity_data: dict, mark_items: list):
        """Handle mark dropdown change."""
        if self.is_loading:
            return
        instance_path = entity_data.get('path', '')
        if not instance_path:
            return

        if mark_name == '(None)':
            mark_id = ''
        else:
            selected = next((item for item in mark_items if item['name'] == mark_name), None)
            if not selected:
                return
            mark_id = selected['id']

        self._save_instance_override(instance_path, 'mark', mark_id)
        agent_id = entity_data.get('id', '')
        if hasattr(self, 'noodlingPropertyChanged'):
            self.noodlingPropertyChanged.emit(agent_id, 'mark', mark_id)

    def _save_instance_override(self, instance_path: str, key: str, value):
        """Write a single override to instance.yaml (read-modify-write)."""
        inst_yaml = os.path.join(instance_path, 'instance.yaml')
        if not os.path.exists(inst_yaml):
            return

        try:
            with open(inst_yaml, 'r') as f:
                inst_data = yaml.safe_load(f) or {}

            if 'overrides' not in inst_data:
                inst_data['overrides'] = {}
            inst_data['overrides'][key] = value

            with open(inst_yaml, 'w') as f:
                yaml.dump(inst_data, f, default_flow_style=False)

            print(f"[Inspector] Saved override {key} to instance: {inst_yaml}")

        except Exception as e:
            print(f"[Inspector] Error saving override to instance: {e}")

    # ========== ZONE PROPERTIES ==========

    def load_zone_properties(self, zone_data):
        """Show Zone properties from Stage View."""
        zone_id = zone_data.get('id', '')
        zone_name = zone_data.get('name', zone_id)
        file_path = zone_data.get('file_path', zone_data.get('path', ''))

        self.property_fields = {}

        # Zone Basics
        basics_group = self.create_property_group("Zone")
        self.property_fields['name'] = self.add_text_field(basics_group, "Name", zone_name)
        self._add_uuid_field(basics_group, zone_id)
        self.properties_layout.addWidget(basics_group)

        # Spatial Properties
        spatial_group = self.create_property_group("Spatial")
        center = zone_data.get('center', [0, 0, 0])
        self.add_vector3_field(spatial_group, "Center", center)

        # Radius/Falloff
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
        description = zone_data.get('description', '')
        if description:
            desc_group = self.create_property_group("Description")
            desc_text = QTextEdit(description)
            desc_text.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
            desc_text.setMaximumHeight(120)
            desc_text.setReadOnly(True)
            desc_group.content.layout().addRow(desc_text)
            self.properties_layout.addWidget(desc_group)

        # Exits/Connections
        exits = zone_data.get('exits', {})
        if exits:
            exits_group = self.create_property_group("Connections")
            for direction, dest_id in exits.items():
                exit_label = QLabel(f"{direction} -> {dest_id}")
                exit_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
                exits_group.content.layout().addRow(exit_label)
            self.properties_layout.addWidget(exits_group)

        # Perception
        perception = zone_data.get('perception', {})
        if perception:
            perc_group = self.create_property_group("Perception")
            self.add_text_field(perc_group, "Visibility", str(perception.get('visibility', 20)), read_only=True)
            self.add_text_field(perc_group, "Audibility", str(perception.get('audibility', 20)), read_only=True)
            self.add_text_field(perc_group, "Lighting", str(perception.get('lighting', 'natural')), read_only=True)
            self.properties_layout.addWidget(perc_group)

        # Ambient
        ambient = zone_data.get('ambient', {})
        if ambient:
            amb_group = self.create_property_group("Ambient")
            sounds = ambient.get('sounds', [])
            self.add_text_field(amb_group, "Sounds", ', '.join(sounds) if sounds else '(none)', read_only=True)
            self.add_text_field(amb_group, "Mood", str(ambient.get('mood', 'neutral')), read_only=True)
            self.add_text_field(amb_group, "Temperature", str(ambient.get('temperature', 'pleasant')), read_only=True)
            self.properties_layout.addWidget(amb_group)

        self.properties_layout.addStretch()

    # ========== OBJECT/PROP PROPERTIES ==========

    def load_object_properties(self, entity_data):
        """Show object properties including physics settings."""
        self.property_fields = {}

        obj_id = entity_data.get('id', '')
        obj_data = entity_data.get('data', {})

        # Basic properties
        obj_group = self.create_property_group("Object Properties")
        self.property_fields['name'] = self.add_text_field(obj_group, "Name", obj_data.get('name', 'Unnamed'))
        self._add_uuid_field(obj_group, obj_id)
        self.property_fields['description'] = self.add_text_area(obj_group, "Description", obj_data.get('description', 'An object in the world.'))
        self.properties_layout.addWidget(obj_group)

        # Physics Properties
        physics_group = self.create_property_group("Physics (SPE)")
        self._create_physics_properties(physics_group, obj_data)
        self.properties_layout.addWidget(physics_group)

        # Metadata
        if hasattr(self, 'create_metadata_component'):
            metadata_component = self.create_metadata_component(obj_id)
            self.properties_layout.addWidget(metadata_component)

        self.properties_layout.addStretch()

    def _create_physics_properties(self, group, obj_data):
        """Create physics property dropdowns for a prim."""
        try:
            from noodlestudio.core.semantic_world import MATERIAL_PRESETS
        except ImportError:
            MATERIAL_PRESETS = {}

        current_material = obj_data.get('material', 'unknown')
        current_mass = obj_data.get('mass', 'medium')
        current_friction = obj_data.get('friction', 'medium')
        current_elasticity = obj_data.get('elasticity', 'normal')
        current_softness = obj_data.get('softness', 'normal')

        material_options = ["(custom)"] + sorted(MATERIAL_PRESETS.keys())

        def on_material_preset_change(material):
            if material == "(custom)":
                return
            preset = MATERIAL_PRESETS.get(material, {})
            if not preset:
                return
            if 'mass' in self.property_fields and 'mass' in preset:
                self.property_fields['mass'].setCurrentText(preset['mass'])
            if 'friction' in self.property_fields and 'friction' in preset:
                self.property_fields['friction'].setCurrentText(preset['friction'])
            if 'elasticity' in self.property_fields and 'elasticity' in preset:
                self.property_fields['elasticity'].setCurrentText(preset['elasticity'])
            if 'softness' in self.property_fields and 'softness' in preset:
                self.property_fields['softness'].setCurrentText(preset['softness'])
            self.save_changes()

        self.property_fields['material'] = self.add_dropdown_field(
            group, "Material Preset", current_material, material_options,
            on_change=on_material_preset_change
        )

        mass_options = ["negligible", "very_light", "light", "medium", "heavy", "very_heavy", "immovable"]
        self.property_fields['mass'] = self.add_dropdown_field(group, "Mass", current_mass, mass_options)

        friction_options = ["slippery", "low", "medium", "high", "sticky"]
        self.property_fields['friction'] = self.add_dropdown_field(group, "Friction", current_friction, friction_options)

        elasticity_options = ["none", "low", "normal", "high", "bouncy"]
        self.property_fields['elasticity'] = self.add_dropdown_field(group, "Elasticity", current_elasticity, elasticity_options)

        softness_options = ["rigid", "hard", "normal", "soft", "squishy"]
        self.property_fields['softness'] = self.add_dropdown_field(group, "Softness", current_softness, softness_options)

        help_label = QLabel("Select a material preset to auto-fill physics properties,\nor set each property individually.")
        help_label.setStyleSheet("color: #808080; font-size: 10px; padding: 4px;")
        help_label.setWordWrap(True)
        group.content.layout().addRow("", help_label)

    # ========== STAGE PROPERTIES ==========

    def load_stage_properties(self, entity_data):
        """Show Stage properties (room metadata)."""
        stage = entity_data.get('data', {})
        stage_id = entity_data.get('id', '')

        basic_group = self.create_property_group("Basic Info")
        self.add_text_field(basic_group, "Name", stage.get('name', ''))
        self.add_text_field(basic_group, "Stage ID", stage_id)
        self.properties_layout.addWidget(basic_group)

        # Description
        desc_group = self.create_property_group("Description")
        desc_text = QTextEdit(stage.get('description', ''))
        desc_text.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        desc_text.setMaximumHeight(100)
        desc_text.setTabChangesFocus(True)
        desc_text.setProperty("stage_id", stage_id)
        desc_text.textChanged.connect(lambda: self.save_stage_description(desc_text))
        desc_text.installEventFilter(self)
        desc_group.content.layout().addRow("Description:", desc_text)
        self.properties_layout.addWidget(desc_group)

        # Exits
        exits_group = self.create_property_group("Exits")
        exits = stage.get('exits', {})
        if exits:
            for direction, dest_id in exits.items():
                exit_label = QLabel(f"{direction} -> {dest_id}")
                exit_label.setStyleSheet("color: #D2D2D2; padding: 4px;")
                exits_group.content.layout().addRow(exit_label)
        else:
            no_exits = QLabel("No exits defined")
            no_exits.setStyleSheet("color: #888; padding: 4px;")
            exits_group.content.layout().addRow(no_exits)
        self.properties_layout.addWidget(exits_group)

        # Occupants
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

    def save_stage_description(self, text_widget: QTextEdit):
        """Save stage description to backend."""
        if self.is_loading:
            return

        stage_id = text_widget.property("stage_id")
        if not stage_id:
            return

        try:
            url = f"{self.api_base}/rooms/{stage_id}/update"
            payload = {"description": text_widget.toPlainText()}

            response = requests.post(url, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"Stage description saved for {stage_id}")
            else:
                print(f"Error saving stage description: {response.json().get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error saving stage description: {e}")

    # ========== USER PROPERTIES ==========

    def load_user_properties(self, entity_data):
        """Show user properties."""
        self.property_fields = {}

        user_group = self.create_property_group("User Info")
        self.property_fields['username'] = self.add_text_field(user_group, "Username", "caity")
        self.property_fields['type'] = self.add_text_field(user_group, "Type", "Noodler (human)")
        self.property_fields['age'] = self.add_text_field(user_group, "Age", "9 years old")
        self.property_fields['pronouns'] = self.add_text_field(user_group, "Pronouns", "she/her")
        self.properties_layout.addWidget(user_group)

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

    # ========== EXIT PROPERTIES ==========

    def load_exit_properties(self, entity_data):
        """Show exit properties."""
        exit_group = self.create_property_group("Exit Info")
        self.add_text_field(exit_group, "Direction", entity_data.get('direction', ''))
        self.add_text_field(exit_group, "Destination", entity_data.get('destination', ''))
        self.properties_layout.addWidget(exit_group)
        self.properties_layout.addStretch()

    # ========== RADIANCE PROPERTIES ==========

    def load_radiance_properties(self, entity_data):
        """Show RadianceComponent properties using RadianceInspector widget."""
        from noodlestudio.panels.radiance_inspector import RadianceInspector

        if not hasattr(self, '_radiance_inspector'):
            self._radiance_inspector = RadianceInspector()

        component = entity_data.get('component')
        path = entity_data.get('path', '')
        on_change = entity_data.get('on_change')

        self._radiance_inspector.set_component(component, path)
        if on_change:
            self._radiance_inspector.set_on_change_callback(on_change)

        self.properties_layout.addWidget(self._radiance_inspector)

    # ========== SAVE METHODS ==========

    def save_changes(self):
        """Save edited properties to disk."""
        if not self.current_entity:
            return

        self.is_saving = True

        try:
            entity_type, entity_data = self.current_entity

            if entity_type == 'noodling':
                self._save_noodling_changes(entity_data)

            elif entity_type in ('prim', 'prop'):
                self._save_prop_changes(entity_type, entity_data)

            elif entity_type == 'zone':
                self._save_zone_changes(entity_data)

        finally:
            # Invalidate the same-entity skip guard so re-selection
            # reloads fresh from disk (description edits don't change
            # name/id, so the guard would otherwise block reload)
            self.current_entity = None
            QTimer.singleShot(2500, lambda: setattr(self, 'is_saving', False))

    def _save_noodling_changes(self, entity_data):
        """Save noodling property changes to YAML files on disk.

        Name -> instance.yaml overrides (instance-level display name)
        Description -> recipe.yaml (template-level character data)
        """
        agent_id = entity_data.get('id', '')
        updates = {}

        if 'name' in self.property_fields:
            updates['name'] = self.property_fields['name'].text()
        if 'species' in self.property_fields:
            updates['species'] = self.property_fields['species'].text()
        if 'description' in self.property_fields:
            updates['description'] = self.property_fields['description'].toPlainText()

        # Emit signal immediately for UI sync
        if 'name' in updates:
            self.nameChanged.emit('noodling', agent_id, updates['name'])

        # Persist name to instance.yaml overrides
        instance_path = entity_data.get('path', '')
        if instance_path and 'name' in updates:
            self._save_name_to_instance(instance_path, updates['name'])

        # Persist description to recipe.yaml
        if 'description' in updates:
            recipe_path = self._resolve_recipe_path(entity_data)
            if recipe_path:
                self._save_description_to_recipe(recipe_path, updates['description'])

    def _save_name_to_instance(self, instance_path, new_name):
        """Write name override to instance.yaml (read-modify-write)."""
        inst_yaml = os.path.join(instance_path, 'instance.yaml')
        if not os.path.exists(inst_yaml):
            return

        try:
            with open(inst_yaml, 'r') as f:
                inst_data = yaml.safe_load(f) or {}

            if 'overrides' not in inst_data:
                inst_data['overrides'] = {}
            inst_data['overrides']['name'] = new_name

            with open(inst_yaml, 'w') as f:
                yaml.dump(inst_data, f, default_flow_style=False)

            print(f"[Inspector] Saved name to instance: {inst_yaml}")

        except Exception as e:
            print(f"[Inspector] Error saving name to instance: {e}")

    def _save_description_to_recipe(self, recipe_path, new_description):
        """Write description to recipe.yaml (read-modify-write)."""
        try:
            with open(recipe_path, 'r') as f:
                recipe_data = yaml.safe_load(f) or {}

            recipe_data['description'] = new_description

            with open(recipe_path, 'w') as f:
                yaml.dump(recipe_data, f, default_flow_style=False)

            print(f"[Inspector] Saved description to recipe: {recipe_path}")

        except Exception as e:
            print(f"[Inspector] Error saving description to recipe: {e}")

    def _resolve_recipe_path(self, entity_data):
        """Resolve the recipe.yaml path for a noodling entity.

        noodling_ref is a relative path from the instance directory
        (e.g. '../../../../Noodlings/ajo_majo'). Resolve it to find
        the noodling template directory, then look for recipe.yaml.
        """
        noodling_ref = entity_data.get('noodling_ref', '')
        instance_path = entity_data.get('path', '')

        if not noodling_ref or not instance_path:
            return None

        # Primary: resolve relative path from instance directory
        noodling_dir = os.path.normpath(os.path.join(instance_path, noodling_ref))
        recipe = os.path.join(noodling_dir, 'recipe.yaml')
        if os.path.exists(recipe):
            return recipe

        # Fallback: bundled library (simple name, not relative path)
        simple_name = os.path.basename(noodling_ref)
        app_library = os.path.join(
            os.path.dirname(__file__), '..', '..', 'library', 'noodlings',
            simple_name, 'recipe.yaml'
        )
        if os.path.exists(app_library):
            return app_library

        return None

    def _save_prop_changes(self, entity_type, entity_data):
        """Save prop/prim property changes."""
        updates = {}

        if 'name' in self.property_fields:
            updates['name'] = self.property_fields['name'].text()
        if 'description' in self.property_fields:
            updates['description'] = self.property_fields['description'].toPlainText()

        physics_fields = ['material', 'mass', 'friction', 'elasticity', 'softness']
        for field_name in physics_fields:
            if field_name in self.property_fields:
                widget = self.property_fields[field_name]
                if hasattr(widget, 'currentText'):
                    value = widget.currentText()
                    if field_name == 'material' and value == "(custom)":
                        continue
                    updates[field_name] = value

        # Emit signal immediately for UI sync
        if 'name' in updates:
            prop_id = entity_data.get('id', '')
            self.nameChanged.emit('prop', prop_id, updates['name'])

        if entity_type == 'prop':
            self._save_prop_to_file(entity_data, updates)
        else:
            object_id = entity_data.get('id', '')
            try:
                url = f"{self.api_base}/objects/{object_id}/update"
                response = requests.post(url, json=updates, timeout=2)
                if response.status_code == 200:
                    print(f"Saved prim {object_id}: {list(updates.keys())}")
            except Exception as e:
                print(f"Error saving prim: {e}")

    def _save_zone_changes(self, entity_data):
        """Save zone property changes."""
        updates = {}
        if 'name' in self.property_fields:
            updates['name'] = self.property_fields['name'].text()

        # Emit signal immediately for UI sync
        if 'name' in updates:
            zone_id = entity_data.get('id', '')
            self.nameChanged.emit('zone', zone_id, updates['name'])

        self._save_zone_to_file(entity_data, updates)

    def _save_prop_to_file(self, entity_data: dict, updates: dict):
        """Save prop changes to prop.yaml file."""
        prop_path = entity_data.get('path', '')
        if not prop_path:
            return

        prop_yaml = os.path.join(prop_path, "prop.yaml")
        if not os.path.exists(prop_yaml):
            return

        try:
            with open(prop_yaml, 'r') as f:
                prop_data = yaml.safe_load(f) or {}

            prop_data.update(updates)

            with open(prop_yaml, 'w') as f:
                yaml.dump(prop_data, f, default_flow_style=False)

            print(f"Saved prop to file: {prop_yaml}")

        except Exception as e:
            print(f"Error saving prop to file: {e}")

    def _save_zone_to_file(self, entity_data: dict, updates: dict):
        """Save zone changes to zone.yaml file."""
        zone_path = entity_data.get('path', '')
        if not zone_path or not os.path.exists(zone_path):
            return

        try:
            with open(zone_path, 'r') as f:
                zone_data = yaml.safe_load(f) or {}

            zone_data.update(updates)

            with open(zone_path, 'w') as f:
                yaml.dump(zone_data, f, default_flow_style=False)

            print(f"Saved zone to file: {zone_path}")

        except Exception as e:
            print(f"Error saving zone to file: {e}")

    # ========== HELPER METHODS ==========

    def _add_uuid_field(self, group, entity_id: str, prefix: str = ''):
        """Add a UUID field with copy button - properly aligned."""
        from PyQt6.QtWidgets import QSizePolicy

        display_id = entity_id.replace(prefix, '') if prefix else entity_id
        if len(display_id) > 20:
            display_id = display_id[:8] + "..." + display_id[-4:]

        # Use HBox container with fixed height to prevent QFormLayout alignment issues
        uuid_container = QWidget()
        uuid_container.setFixedHeight(20)  # Match typical label height
        uuid_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        uuid_layout = QHBoxLayout(uuid_container)
        uuid_layout.setContentsMargins(0, 0, 0, 0)
        uuid_layout.setSpacing(4)
        uuid_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        uuid_text = QLabel(display_id)
        uuid_text.setStyleSheet("color: #888;")
        uuid_layout.addWidget(uuid_text)

        # Plain text copy button (no HTML to avoid baseline issues)
        copy_btn = QLabel("[copy]")
        copy_btn.setStyleSheet("color: #666;")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setToolTip(f"Copy full ID: {entity_id}")
        copy_btn.mousePressEvent = lambda e: QApplication.clipboard().setText(entity_id)
        uuid_layout.addWidget(copy_btn)

        uuid_layout.addStretch()
        group.content.layout().addRow("UUID:", uuid_container)

    # ========== COMPONENT COLLECTION ==========

    def _init_component_collection(self, entity_id: str, entity_data: dict):
        """
        Initialize ComponentCollection for an entity.

        Loads components from entity's YAML if present, otherwise creates
        an empty collection ready for components to be added.

        Args:
            entity_id: Entity identifier
            entity_data: Entity data dict (may contain 'path' to YAML location)
        """
        from noodlestudio.core.component_collection import ComponentCollection
        from noodlestudio.core.component_base import component_registry

        # Ensure components package is loaded (triggers @register_component decorators)
        try:
            import noodlestudio.core.components  # noqa: F401
        except ImportError:
            pass

        # Create collection for this entity
        collection = ComponentCollection(entity_id=entity_id)

        # Try to load from entity's components.yaml if it exists
        entity_path = entity_data.get('path', '')
        if entity_path:
            import os
            components_file = os.path.join(os.path.dirname(entity_path), 'components.yaml')
            if os.path.exists(components_file):
                try:
                    with open(components_file, 'r') as f:
                        components_data = yaml.safe_load(f) or {}
                    collection.from_dict(components_data)
                    print(f"[Inspector] Loaded {len(collection)} components from {components_file}")
                except Exception as e:
                    print(f"[Inspector] Error loading components: {e}")

        # Store on the inspector
        self._current_components = collection

    def _save_component_changes(self, component):
        """
        Save component changes to disk.

        Called when a component property is modified via Inspector.
        """
        if not self._current_components:
            return

        entity_data = self.current_entity[1] if self.current_entity else {}
        entity_path = entity_data.get('path', '')

        if entity_path:
            import os
            components_file = os.path.join(os.path.dirname(entity_path), 'components.yaml')
            try:
                data = self._current_components.to_dict()
                with open(components_file, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
                print(f"[Inspector] Saved components to {components_file}")
            except Exception as e:
                print(f"[Inspector] Error saving components: {e}")

    def _refresh_components_display(self):
        """Refresh the Inspector to show updated components."""
        # Re-load current entity to refresh display
        if self.current_entity:
            entity_type, entity_data = self.current_entity
            if entity_type == 'noodling':
                # Clear and reload
                self.clear_inspector()
                self.load_noodling_properties(entity_data)

    def _remove_component_from_entity(self, component):
        """Remove a component from the current entity."""
        if self._current_components:
            self._current_components.remove(component.component_type)
            self._save_component_changes(component)
            self._refresh_components_display()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
