"""
Scene Hierarchy UI Mixin - UI Canvas component operations

Contains:
- _add_ui_component: Add new UI component to canvas
- _delete_ui_component: Delete UI component from canvas
- _save_ui_yaml: Save UI changes back to ui.yaml

Author: Caitlyn + Claude
Date: January 2026
"""

from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtCore import Qt


class SceneHierarchyUIMixin:
    """Mixin providing UI Canvas operations for SceneHierarchy."""

    def _create_ui_canvas(self, name: str = None):
        """
        Create a new UI Canvas (ui.yaml) in the current stage.

        Args:
            name: Optional name for the canvas file (default: "ui")
        """
        from ..runtime.ui.loader import UILoader, create_default_ui
        from pathlib import Path

        print(f"[SceneHierarchy] _create_ui_canvas called, stage={self.current_stage}", flush=True)

        if not self.project_manager or not self.current_stage:
            print("[SceneHierarchy] Cannot create UI Canvas: no project or stage", flush=True)
            return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        print(f"[SceneHierarchy] stage_path={stage_path}", flush=True)
        if not stage_path:
            print("[SceneHierarchy] Cannot create UI Canvas: stage path not found", flush=True)
            return

        # Determine filename
        if name:
            filename = f"{name}.ui.yaml" if not name.endswith('.yaml') else name
        else:
            # Check if ui.yaml already exists
            ui_path = Path(stage_path) / "ui.yaml"
            if ui_path.exists():
                # Find unique name
                counter = 2
                while (Path(stage_path) / f"ui{counter}.ui.yaml").exists():
                    counter += 1
                filename = f"ui{counter}.ui.yaml"
            else:
                filename = "ui.yaml"

        ui_path = Path(stage_path) / filename

        # Create default UI structure
        root = create_default_ui()

        # Save to disk
        loader = UILoader()
        loader.save_file(root, ui_path)

        print(f"[SceneHierarchy] Created UI Canvas: {ui_path}", flush=True)

        # Refresh to show new canvas
        self.refresh_scene()

        # Notify UI Canvas Editor if present
        main_window = self.window() if hasattr(self, 'window') else None
        if main_window and hasattr(main_window, 'ui_canvas_editor'):
            main_window.ui_canvas_editor.reload_ui()

    def _add_ui_component(self, parent_data: dict, component_type: str):
        """
        Add a new UI component as a child of the selected component.

        Args:
            parent_data: Entity data of the parent (ui or ui_component)
            component_type: Type of component to create (Panel, Button, etc.)
        """
        from ..runtime.ui.component import get_component_class
        from ..runtime.ui.loader import UILoader
        from ..core.scene_node import SceneNodeType
        from pathlib import Path

        # Get parent component
        parent_component = parent_data.get('component')
        ui_path = parent_data.get('path')

        if not parent_component or not ui_path:
            print(f"[SceneHierarchy] Cannot add UI component: missing parent or path")
            return

        # Get component class
        component_class = get_component_class(component_type)
        if not component_class:
            print(f"[SceneHierarchy] Unknown component type: {component_type}")
            return

        # Generate unique name
        base_name = component_type.lower()
        existing_names = self._get_all_ui_component_names(parent_component)
        counter = 1
        while f"{base_name}{counter}" in existing_names:
            counter += 1
        name = f"{base_name}{counter}"

        # Create component with default geometry
        new_component = component_class(name=name)
        default_sizes = {
            "Panel": (200, 150),
            "Label": (100, 24),
            "Button": (80, 32),
            "TextInput": (200, 32),
            "ChatHistory": (300, 200),
            "ChatInput": (300, 48),
            "RadianceViewport": (400, 300),
        }
        w, h = default_sizes.get(component_type, (100, 32))
        new_component.geometry.width = w
        new_component.geometry.height = h

        # Add to parent
        parent_component.add_child(new_component)

        # Save to disk
        loader = UILoader()
        # Find root component
        root = parent_component
        while root.parent:
            root = root.parent
        loader.save_file(root, Path(ui_path))

        # Refresh the tree to show new component
        self.refresh_scene()

        # Notify UI Canvas Editor if present
        main_window = self.window() if hasattr(self, 'window') else None
        if main_window and hasattr(main_window, 'ui_canvas_editor'):
            main_window.ui_canvas_editor.reload_ui()

        print(f"[SceneHierarchy] Added UI component: {name} ({component_type})")

    def _delete_ui_component(self, entity_data: dict):
        """
        Delete a UI component from the canvas.

        Args:
            entity_data: Entity data of the component to delete
        """
        from ..runtime.ui.loader import UILoader
        from pathlib import Path

        component = entity_data.get('component')
        ui_path = entity_data.get('path')

        if not component or not ui_path:
            print(f"[SceneHierarchy] Cannot delete UI component: missing component or path")
            return

        # Don't allow deleting root
        if not component.parent:
            print(f"[SceneHierarchy] Cannot delete root UI component")
            return

        # Remove from parent
        component.parent.remove_child(component)

        # Save to disk
        loader = UILoader()
        # Find root component
        root = component.parent
        while root.parent:
            root = root.parent
        loader.save_file(root, Path(ui_path))

        # Refresh the tree
        self.refresh_scene()

        # Notify UI Canvas Editor if present
        main_window = self.window() if hasattr(self, 'window') else None
        if main_window and hasattr(main_window, 'ui_canvas_editor'):
            main_window.ui_canvas_editor.reload_ui()

        print(f"[SceneHierarchy] Deleted UI component: {entity_data.get('name')}")

    def _get_all_ui_component_names(self, root_component) -> set:
        """Get all component names in the UI tree (for unique name generation)."""
        names = set()

        def collect_names(component):
            names.add(component.name)
            for child in component.children:
                collect_names(child)

        # Find true root
        root = root_component
        while root.parent:
            root = root.parent

        collect_names(root)
        return names
