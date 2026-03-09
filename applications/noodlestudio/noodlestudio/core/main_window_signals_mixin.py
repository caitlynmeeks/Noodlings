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
        """Update assembly editor when a noodling is selected in hierarchy."""
        editor = getattr(self, 'unified_editor', None)
        if not editor:
            return

        if entity_type is None or entity_data is None:
            editor.clear_editor()
            return

        if entity_type == 'noodling':
            self._load_facet_assembly_for_noodling(entity_data)
        else:
            editor.clear_editor()

    def _load_facet_assembly_for_noodling(self, entity_data: dict):
        """Load the facet assembly for a noodling entity.

        Resolves the assembly path from the noodling template directory,
        using the instance path + noodling_ref relative path from instance.yaml.
        """
        import os
        from ..core.facet_system import FacetAssembly

        agent_id = entity_data.get('id', '')
        if not agent_id:
            return

        instance_path = entity_data.get('path', '')
        noodling_ref = entity_data.get('noodling_ref', '')

        assembly_path = None

        # Primary: resolve from instance path + noodling_ref (relative path)
        # noodling_ref is e.g. "../../../../Noodlings/ajo_majo"
        if noodling_ref and instance_path:
            noodling_dir = os.path.normpath(os.path.join(instance_path, noodling_ref))
            candidate = os.path.join(noodling_dir, 'assembly.yaml')
            if os.path.exists(candidate):
                assembly_path = candidate

        # Fallback: bundled library (for templates that use simple names)
        if not assembly_path and noodling_ref:
            simple_name = os.path.basename(noodling_ref)
            noodlestudio_dir = os.path.join(os.path.dirname(__file__), '../..')
            candidate = os.path.join(
                noodlestudio_dir, 'library', 'noodlings', simple_name, 'assembly.yaml'
            )
            if os.path.exists(candidate):
                assembly_path = candidate

        # Last resort: empty noodling
        if not assembly_path:
            noodlestudio_dir = os.path.join(os.path.dirname(__file__), '../..')
            assembly_path = os.path.join(
                noodlestudio_dir, 'library', 'noodlings', 'empty_noodling', 'assembly.yaml'
            )

        if os.path.exists(assembly_path):
            try:
                assembly = FacetAssembly.load_yaml(assembly_path)
                self.unified_editor.load_assembly_from_data(
                    assembly, force_reload=True, source_path=assembly_path
                )
                self.unified_editor.set_current_agent(agent_id)
            except Exception as e:
                import traceback
                print(f"[Assembly Editor] Error loading assembly: {e}")
                traceback.print_exc()
        else:
            print(f"[Assembly Editor] Assembly not found for {agent_id}: {assembly_path}")

    @staticmethod
    def _build_neural_node_entity_data(node) -> dict:
        """Build inspector entity_data dict from a neural canvas node.

        Shared by both the standalone NC handler and the unified editor
        NC depth view handler.
        """
        from ..core.neural_canvas.node_definitions import NODE_DEFINITIONS
        default_params = NODE_DEFINITIONS.get(node.type, {}).get('params', {})
        merged_params = {**default_params, **node.params}

        return {
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

    def _on_neural_canvas_node_selected(self, node_id: str):
        """Handle node selection in standalone Neural Canvas - show in Inspector."""
        if not node_id:
            self.inspector.clear_inspector()
            return

        node = self.neural_canvas.graph.get_node_by_id(node_id)
        if not node:
            return

        entity_data = self._build_neural_node_entity_data(node)
        self.inspector.load_entity('neural_node', entity_data)

    def _on_nc_depth_node_selected(self, node_id: str):
        """Handle node selection from NC depth view inside unified editor."""
        if not node_id:
            self.inspector.clear_inspector()
            return

        editor = getattr(self, 'unified_editor', None)
        if not editor:
            return

        graph = editor.get_current_nc_graph()
        if not graph:
            return

        node = graph.get_node_by_id(node_id)
        if not node:
            return

        entity_data = self._build_neural_node_entity_data(node)
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

    def _on_inspector_name_changed(self, entity_type: str, entity_id: str, new_name: str):
        """Handle name change in Inspector - update Stage View tree item."""
        if hasattr(self, 'hierarchy') and self.hierarchy:
            self.hierarchy.update_entity_name(entity_type, entity_id, new_name)

    def _on_noodling_property_changed(self, agent_id: str, prop_name: str, value):
        """Handle noodling property change from Inspector.

        Routes to GuidePerformanceManager when a performance is active.
        Properties are already persisted to instance.yaml by the inspector;
        this method handles the runtime (live performance) side.
        """
        manager = getattr(self, 'guide_performance_manager', None)
        if not manager or not manager.is_active:
            return

        # Strip agent_ prefix to get noodling_id (instance dir name)
        noodling_id = agent_id.replace('agent_', '')

        if prop_name == 'vrm_path':
            manager.update_vrm(noodling_id, value)
        elif prop_name == 'ensemble_active':
            manager.set_ensemble_active(noodling_id, value)
        elif prop_name == 'visible':
            manager.set_visible(noodling_id, value)
        elif prop_name == 'mark':
            manager.update_mark(noodling_id, value)
        elif prop_name == 'role':
            manager.update_role(noodling_id, value)

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


#♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
