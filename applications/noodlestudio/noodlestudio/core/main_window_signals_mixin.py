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
#   Main Window Signals Mixin - Panel signal handlers
#
#   Contains signal handler methods for connecting panels: - ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_signals_mixin
# PURPOSE:  Main Window Signals Mixin - Panel signal handlers
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowSignalsMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

class MainWindowSignalsMixin:
    """Mixin providing signal handlers for MainWindow."""

    def _on_entity_selected_for_performance(self, entity_type: str, entity_data: dict):
        """Sync hierarchy noodling selection to performance window.

        When a noodling instance is selected during an active ensemble
        performance, highlights it in the window and switches the facets
        editor to its assembly.
        """
        manager = getattr(self, 'guide_performance_manager', None)
        if not manager or not manager.is_active:
            return

        if entity_type != 'noodling' or not entity_data:
            return

        # Extract noodling_id from instance path (directory name)
        import os
        path = entity_data.get('path', '')
        if path:
            noodling_id = os.path.basename(path)
        else:
            # Fallback: strip agent_ prefix from id
            noodling_id = entity_data.get('id', '').replace('agent_', '')

        if noodling_id:
            manager.on_hierarchy_noodling_selected(noodling_id)

    def on_entity_selected_for_console(self, entity_type: str, entity_data: dict):
        """Update Console filter when entity is selected in hierarchy."""
        if not hasattr(self, 'console'):
            return

        if entity_type is None or entity_data is None:
            self.console.set_selected_entities([])
            return

        entity_id = entity_data.get('id', '')
        if entity_id:
            self.console.set_selected_entities([entity_id])

    def on_entity_selected_for_facets_editor(self, entity_type: str, entity_data: dict):
        """Update Facets Editor when a noodling is selected in hierarchy."""
        if not hasattr(self, 'facets_editor'):
            return

        if entity_type is None or entity_data is None:
            self.facets_editor.clear_editor()
            return

        if entity_type == 'noodling':
            self._load_facet_assembly_for_noodling(entity_data)

    def _load_facet_assembly_for_noodling(self, entity_data: dict):
        """Load the facet assembly for a noodling entity."""
        import os
        import yaml
        from ..core.facet_system import FacetAssembly

        agent_id = entity_data.get('id', '')
        if not agent_id:
            return

        agent_full_data = entity_data.get('data', {})

        # Try multiple locations where facet_assembly might be
        facet_assembly_config = (
            entity_data.get('facet_assembly') or
            agent_full_data.get('facet_assembly') or
            agent_full_data.get('config', {}).get('facet_assembly')
        )

        ref = None
        if facet_assembly_config:
            if isinstance(facet_assembly_config, str):
                ref = facet_assembly_config
            elif isinstance(facet_assembly_config, dict):
                ref = facet_assembly_config.get('ref')

        # If not found in instance data, load from recipe.yaml
        if not ref:
            noodling_ref = entity_data.get('noodling_ref') or agent_full_data.get('noodling', '')
            instance_path = entity_data.get('path', '')

            if noodling_ref and instance_path:
                project_root = instance_path
                for _ in range(4):
                    project_root = os.path.dirname(project_root)

                library_recipe = os.path.join(
                    project_root, 'Library', 'Noodlings', noodling_ref, 'recipe.yaml'
                )

                recipe_path = None
                if os.path.exists(library_recipe):
                    recipe_path = library_recipe
                else:
                    app_library = os.path.join(
                        os.path.dirname(__file__), '..', '..', 'library', 'noodlings',
                        noodling_ref, 'recipe.yaml'
                    )
                    if os.path.exists(app_library):
                        recipe_path = app_library

                if recipe_path:
                    try:
                        with open(recipe_path, 'r') as f:
                            recipe_data = yaml.safe_load(f) or {}
                        ref = recipe_data.get('facet_assembly')
                    except Exception as e:
                        print(f"[Facets Editor] Error loading recipe: {e}")

        # Fallback to default
        if not ref:
            ref = "library/empty_noodling"

        # Resolve assembly path
        noodlestudio_dir = os.path.join(os.path.dirname(__file__), '../..')

        if ref.startswith('library/'):
            template_name = ref.replace('library/', '')
            assembly_path = os.path.join(
                noodlestudio_dir, 'library/noodlings', template_name, 'assembly.yaml'
            )
        else:
            assembly_path = os.path.join(noodlestudio_dir, 'facet_assemblies', f'{ref}.yaml')

        if os.path.exists(assembly_path):
            try:
                assembly = FacetAssembly.load_yaml(assembly_path)
                self.facets_editor.load_assembly_from_data(
                    assembly, force_reload=True, source_path=assembly_path
                )
                self.facets_editor.set_current_agent(agent_id)
            except Exception as e:
                import traceback
                print(f"[Facets Editor] Error loading facet assembly: {e}")
                traceback.print_exc()

    def _on_neural_canvas_node_selected(self, node_id: str):
        """Handle node selection in Neural Canvas - show in Inspector."""
        if not node_id:
            self.inspector.clear_inspector()
            return

        node = self.neural_canvas.graph.get_node_by_id(node_id)
        if not node:
            return

        from ..core.neural_canvas.node_definitions import NODE_DEFINITIONS
        default_params = NODE_DEFINITIONS.get(node.type, {}).get('params', {})
        merged_params = {**default_params, **node.params}

        entity_data = {
            'id': node.id,
            'name': node.name,
            'type': node.type.value,
            'params': merged_params,
            'weights': {
                name: {
                    'shape': list(weight.shape),
                    'path': weight.path,
                    'trainable': weight.trainable,
                    'num_params': weight.num_parameters()
                }
                for name, weight in node.weights.items()
            },
            'inputs': {name: str(port) for name, port in node.inputs.items()},
            'outputs': {name: str(port) for name, port in node.outputs.items()},
            'position': node.position,
            'description': node.description,
            'tags': node.tags
        }

        self.inspector.load_entity('neural_node', entity_data)

    def _on_neural_canvas_param_changed(self, node_id: str, param_name: str, new_value):
        """Handle param change from Neural Canvas - update Inspector live."""
        if not hasattr(self.inspector, '_current_neural_node_id'):
            return
        if self.inspector._current_neural_node_id != node_id:
            return
        self.inspector.update_neural_node_param(param_name, new_value)

    def _on_neural_canvas_graph_loaded(self):
        """Handle new graph loaded in Neural Canvas - clear Inspector."""
        if hasattr(self.inspector, '_current_neural_node_id') and self.inspector._current_neural_node_id:
            self.inspector._current_neural_node_id = None
            self.inspector._neural_node_param_widgets = {}
            self.inspector.clear_inspector()

    def _on_radiance_loaded(self, path: str, component):
        """Handle radiance loaded in Gaussian Viewer - show in Inspector."""
        from pathlib import Path

        entity_data = {
            'name': Path(path).stem,
            'path': path,
            'component': component,
            'on_change': lambda: self.gaussian_viewer._request_full_render()
        }

        self.inspector.load_entity('radiance', entity_data)

        # Connect inspector's bone signals to viewer
        if hasattr(self.inspector, '_radiance_inspector') and self.inspector._radiance_inspector:
            ri = self.inspector._radiance_inspector
            try:
                ri.focusBoneRequested.disconnect()
            except TypeError:
                pass
            try:
                ri.boneSelected.disconnect()
            except TypeError:
                pass
            try:
                ri.requestViewerFocus.disconnect()
            except TypeError:
                pass
            try:
                self.gaussian_viewer.boneSelectionChanged.disconnect()
            except TypeError:
                pass
            ri.focusBoneRequested.connect(self.gaussian_viewer.focus_on_bone)
            self.gaussian_viewer.boneSelectionChanged.connect(ri.set_selected_bone)
            ri.boneSelected.connect(self.gaussian_viewer.set_bone_selection)
            ri.requestViewerFocus.connect(self.gaussian_viewer.setFocus)

        # Add to Assets panel
        if hasattr(self, 'assets'):
            self.assets.add_loaded_radiance(path, component)

    def _on_mesh_imported(self, source_path: str, mesh_type: str, output_radiance_path: str):
        """Handle mesh imported in Gaussian Viewer - add to Assets panel."""
        if hasattr(self, 'assets'):
            metadata = {'radiance_path': output_radiance_path}
            self.assets.add_loaded_mesh(source_path, mesh_type, metadata)

    def _on_inspector_name_changed(self, entity_type: str, entity_id: str, new_name: str):
        """Handle name change in Inspector - update Stage View tree item."""
        if hasattr(self, 'hierarchy') and self.hierarchy:
            self.hierarchy.update_entity_name(entity_type, entity_id, new_name)

    def _on_asset_renamed(self, asset_type: str, asset_id: str, new_name: str):
        """Handle name change in Assets Panel."""
        print(f"[MainWindow] Asset renamed: {asset_type}/{asset_id} -> {new_name}")

    def _on_asset_selected(self, asset_type: str, path: str):
        """Handle asset selection in Assets Panel - show properties in Inspector."""
        import os

        entity_data = {
            'asset_type': asset_type,
            'path': path,
            'name': os.path.basename(path)
        }

        if hasattr(self, 'inspector') and self.inspector:
            self.inspector.load_entity('asset', entity_data)

    def _on_zone_selected(self, zone_id: str, zone_data: dict):
        """Handle zone selection from Spatial View panel."""
        if not zone_id:
            return

        zone_name = zone_data.get('name', zone_id)
        self.statusBar().showMessage(f"Selected zone: {zone_name}", 3000)

        if hasattr(self, 'inspector'):
            self.inspector.load_entity('zone', zone_data)

    def _on_ui_component_selected(self, component):
        """Handle UI component selection from UI Canvas Editor."""
        # Update Inspector
        if hasattr(self, 'inspector') and self.inspector:
            if hasattr(self.inspector, 'load_ui_component'):
                self.inspector.load_ui_component(component)

        # Sync selection to Stage hierarchy (bidirectional)
        if hasattr(self, 'scene_hierarchy') and self.scene_hierarchy:
            if component:
                # Select the matching item in Stage hierarchy
                if hasattr(self.scene_hierarchy, 'select_ui_component_by_name'):
                    self.scene_hierarchy.select_ui_component_by_name(component.name)
            else:
                # Clear selection when nothing selected in Canvas
                if hasattr(self.scene_hierarchy, 'clear_ui_selection'):
                    self.scene_hierarchy.clear_ui_selection()

    def _on_ui_entity_selected_for_canvas_editor(self, entity_type, entity_data):
        """Handle UI entity selection from Stage hierarchy - load into canvas editor."""
        from pathlib import Path

        if entity_type == 'ui':
            # Load the UI canvas file into the editor
            ui_path = entity_data.get('path')
            if ui_path and hasattr(self, 'ui_canvas_editor'):
                self.ui_canvas_editor.load_ui_file(Path(ui_path))
                # Switch to UI Canvas tab
                if hasattr(self, 'center_tabs'):
                    for i in range(self.center_tabs.count()):
                        if self.center_tabs.tabText(i) == "UI Canvas":
                            self.center_tabs.setCurrentIndex(i)
                            break
        elif entity_type == 'ui_component':
            # Load parent canvas if not already loaded, then select component
            ui_path = entity_data.get('path')
            component = entity_data.get('component')
            if ui_path and hasattr(self, 'ui_canvas_editor'):
                # Load the canvas if not already showing this file
                current_path = getattr(self.ui_canvas_editor.view, 'ui_file_path', None)
                if current_path is None or str(current_path) != str(ui_path):
                    self.ui_canvas_editor.load_ui_file(Path(ui_path))

                # Select the component in the canvas view
                if component and hasattr(self.ui_canvas_editor.view, 'component_items'):
                    comp_name = component.name
                    if comp_name in self.ui_canvas_editor.view.component_items:
                        self.ui_canvas_editor.view.canvas_scene.clearSelection()
                        item = self.ui_canvas_editor.view.component_items[comp_name]
                        item.setSelected(True)

                # Switch to UI Canvas tab
                if hasattr(self, 'center_tabs'):
                    for i in range(self.center_tabs.count()):
                        if self.center_tabs.tabText(i) == "UI Canvas":
                            self.center_tabs.setCurrentIndex(i)
                            break

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
