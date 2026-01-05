"""
Scene Hierarchy Context Menu Mixin - Right-click context menu

Contains:
- show_context_menu: Entry point for context menu
- _show_context_menu_impl: Implementation with error handling
- _prompt_open_project: Helper for no-project message
- inspect_entity: Emit selection signal

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtWidgets import QMenu


def _safe_callback(func):
    """Wrap a callback function to catch and log exceptions.

    Qt slots that raise exceptions can crash the app fatally.
    This wrapper ensures exceptions are logged but don't crash Qt.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[SceneHierarchy] Callback error: {e}")
            import traceback
            traceback.print_exc()
    return wrapper


class SceneHierarchyContextMenuMixin:
    """Mixin providing context menu for SceneHierarchy."""

    def show_context_menu(self, position):
        """Show right-click context menu (Unity-style)."""
        try:
            self._show_context_menu_impl(position)
        except Exception as e:
            print(f"[SceneHierarchy] CONTEXT MENU ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _show_context_menu_impl(self, position):
        """Implementation of context menu (separated for error handling)."""
        item = self.tree.itemAt(position)
        selected_items = self.tree.selectedItems()

        menu = QMenu()

        if item:
            # Capture data immediately (item may be deleted after menu closes)
            entity_data = item.data(0, self._get_user_role())

            # Check if it's an ensemble (tuple from Assets)
            if isinstance(entity_data, tuple):
                asset_type, asset_name = entity_data
                if asset_type == "ensemble":
                    # Ensemble context menu
                    menu.addAction("Unpack Ensemble", lambda: self.unpack_ensemble(asset_name))
                    menu.addAction("View Ensemble Info", lambda: self.view_ensemble_info(asset_name))
                    menu.addSeparator()
                    menu.addAction("Remove from Hierarchy", lambda: self.remove_item_from_tree(item))
                    menu.exec(self.tree.viewport().mapToGlobal(position))
                    return

            entity_type = entity_data.get('type', '') if entity_data and isinstance(entity_data, dict) else None

            # Check if multiple items of different types are selected
            selected_types = set()
            for sel_item in selected_items:
                sel_data = sel_item.data(0, self._get_user_role())
                if sel_data and isinstance(sel_data, dict):
                    selected_types.add(sel_data.get('type', ''))

            is_multi_type_selection = len(selected_types) > 1
            is_multi_selection = len(selected_items) > 1

            if is_multi_type_selection:
                # Multiple types selected - only show common actions
                count = len(selected_items)
                menu.addAction(f"De-Rez {count} Selected Items", _safe_callback(lambda: self.delete_selected_items()))
            elif is_multi_selection:
                # Multiple items of same type - show count-aware actions
                count = len(selected_items)
                menu.addAction(f"De-Rez {count} Selected", _safe_callback(lambda: self.delete_selected_items()))
            else:
                # Single selection - show type-specific actions
                # Context-specific actions (capture data, not item reference)
                # All callbacks wrapped in _safe_callback to prevent Qt slot crashes
                if entity_type == 'noodling':
                    menu.addAction("Toggle Enlightenment", _safe_callback(lambda d=entity_data: self.toggle_enlightenment_data(d)))

                    # Check if cognition is paused for this agent
                    agent_id = entity_data.get('id')
                    is_paused = self.get_agent_pause_state(agent_id)
                    pause_text = "Resume Cognition" if is_paused else "Pause Cognition"
                    menu.addAction(pause_text, _safe_callback(lambda d=entity_data: self.toggle_cognition_pause_data(d)))

                    menu.addSeparator()
                    menu.addAction("Export Noodling", _safe_callback(lambda d=entity_data: self.export_noodling_data(d)))
                    menu.addSeparator()
                    menu.addAction("Duplicate Noodling", _safe_callback(lambda d=entity_data: self.duplicate_prim_data(d)))
                    menu.addAction("Reset State", _safe_callback(lambda d=entity_data: self.reset_prim_state_data(d)))
                    menu.addSeparator()
                    menu.addAction("De-Rez Noodling", _safe_callback(lambda d=entity_data: self.delete_selected_items()))

                elif entity_type == 'prim':
                    menu.addAction("Edit Description", _safe_callback(lambda d=entity_data: self.edit_description_data(d)))
                    menu.addSeparator()
                    menu.addAction("Export Prim", _safe_callback(lambda d=entity_data: self.export_prim_data(d)))
                    menu.addSeparator()
                    menu.addAction("Duplicate Prim", _safe_callback(lambda d=entity_data: self.duplicate_prim_data(d)))
                    menu.addAction("De-Rez Prim", _safe_callback(lambda d=entity_data: self.delete_selected_items()))

                elif entity_type == 'prop':
                    # Project-mode prop
                    menu.addAction("Duplicate", _safe_callback(lambda d=entity_data: self.duplicate_prop(d)))
                    menu.addAction("De-Rez", _safe_callback(lambda d=entity_data: self.delete_selected_items()))

                # Note: noodling instances handled in 'noodling' case above

                elif entity_type == 'zone':
                    # Project-mode zone
                    menu.addAction("De-Rez", _safe_callback(lambda d=entity_data: self.delete_selected_items()))

                elif entity_type == 'user':
                    menu.addAction("View Profile", _safe_callback(lambda d=entity_data: self.view_user_profile_data(d)))

                elif entity_type == 'exit':
                    menu.addAction("Edit Exit", _safe_callback(lambda d=entity_data: self.edit_exit_data(d)))
                    menu.addAction("De-Rez Exit", _safe_callback(lambda d=entity_data: self.delete_prim_data(d)))

                elif entity_type == 'ui':
                    # UI Canvas root - can add child components
                    add_ui_menu = menu.addMenu("Add")
                    add_ui_menu.addAction("Panel", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Panel')))
                    add_ui_menu.addAction("Label", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Label')))
                    add_ui_menu.addAction("Button", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Button')))
                    add_ui_menu.addAction("TextInput", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'TextInput')))
                    add_ui_menu.addSeparator()
                    add_ui_menu.addAction("Checkbox", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Checkbox')))
                    add_ui_menu.addAction("Dropdown", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Dropdown')))
                    add_ui_menu.addAction("Slider", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Slider')))
                    add_ui_menu.addAction("RadioGroup", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'RadioGroup')))
                    add_ui_menu.addSeparator()
                    add_ui_menu.addAction("ChatHistory", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'ChatHistory')))
                    add_ui_menu.addAction("ChatInput", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'ChatInput')))
                    add_ui_menu.addAction("RadianceViewport", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'RadianceViewport')))
                    add_ui_menu.addAction("WebView", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'WebView')))

                elif entity_type == 'ui_component':
                    # UI Component - can add children (if it's a Panel) and delete
                    component = entity_data.get('component')
                    if component and component.component_type == 'Panel':
                        add_ui_menu = menu.addMenu("Add Child")
                        add_ui_menu.addAction("Panel", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Panel')))
                        add_ui_menu.addAction("Label", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Label')))
                        add_ui_menu.addAction("Button", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Button')))
                        add_ui_menu.addAction("TextInput", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'TextInput')))
                        add_ui_menu.addSeparator()
                        add_ui_menu.addAction("Checkbox", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Checkbox')))
                        add_ui_menu.addAction("Dropdown", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Dropdown')))
                        add_ui_menu.addAction("Slider", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'Slider')))
                        add_ui_menu.addAction("RadioGroup", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'RadioGroup')))
                        add_ui_menu.addSeparator()
                        add_ui_menu.addAction("ChatHistory", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'ChatHistory')))
                        add_ui_menu.addAction("ChatInput", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'ChatInput')))
                        add_ui_menu.addAction("RadianceViewport", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'RadianceViewport')))
                        add_ui_menu.addAction("WebView", _safe_callback(lambda d=entity_data: self._add_ui_component(d, 'WebView')))
                        menu.addSeparator()
                    menu.addAction("Delete", _safe_callback(lambda d=entity_data: self._delete_ui_component(d)))

                # Note: Folders are for Assets panel, not Stage View
                # Stage View shows scene entities only (zones, noodlings, props, ui)
        else:
            # Empty space - show rez options only if project is open AND server is running
            if self.project_manager and self.project_manager.is_project_open():
                if self._server_running and self.current_stage:
                    create_menu = menu.addMenu("Rez")
                    create_menu.addAction("New Noodling", _safe_callback(lambda: self.create_empty_noodling()))
                    create_menu.addAction("New Prim", _safe_callback(lambda: self.create_empty_prim()))
                    create_menu.addAction("New Zone", _safe_callback(lambda: self.create_empty_zone()))
                    create_menu.addSeparator()
                    create_menu.addAction("New UI Canvas", _safe_callback(lambda: self._create_ui_canvas()))

                    menu.addSeparator()
                    menu.addAction("Import Prim...", _safe_callback(lambda: self.import_prim()))
                elif not self._server_running:
                    # Server offline
                    info_action = menu.addAction("Start server to create items")
                    info_action.setEnabled(False)
                else:
                    # Server running but no stage selected
                    info_action = menu.addAction("Create a stage first (File > New Stage)")
                    info_action.setEnabled(False)
            else:
                menu.addAction("Open Project...", _safe_callback(lambda: self._prompt_open_project()))

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _get_user_role(self):
        """Get Qt.ItemDataRole.UserRole constant."""
        from PyQt6.QtCore import Qt
        return Qt.ItemDataRole.UserRole

    def _prompt_open_project(self):
        """Prompt user to open a project."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "No Project Open",
            "Please open a project first.\n\nFile > Open Project..."
        )

    def inspect_entity(self, entity_data):
        """Inspect entity (safe - uses data not item)."""
        entity_type = entity_data.get('type', 'unknown')
        self.entitySelected.emit(entity_type, entity_data)
