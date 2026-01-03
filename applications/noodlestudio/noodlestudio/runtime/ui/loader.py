"""
UI YAML Loader

Loads ui.yaml files and constructs component trees.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .component import UIComponent, get_component_class


class UILoader:
    """
    Loads UI definitions from YAML files.

    The ui.yaml format is the stable contract between the designer
    and renderer. Changes to renderer implementation don't affect
    user projects.
    """

    def __init__(self):
        # Ensure components are registered by importing them
        from . import components  # noqa: F401

    def load_file(self, path: str | Path) -> UIComponent:
        """
        Load a UI definition from a YAML file.

        Args:
            path: Path to ui.yaml file

        Returns:
            Root UIComponent tree
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"UI file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return self.load_dict(data)

    def load_dict(self, data: Dict[str, Any]) -> UIComponent:
        """
        Load a UI definition from a dictionary.

        Args:
            data: Parsed YAML data

        Returns:
            Root UIComponent tree
        """
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported ui.yaml version: {version}")

        root_data = data.get("root")
        if not root_data:
            raise ValueError("ui.yaml must have a 'root' component")

        return self._load_component(root_data)

    def _load_component(self, data: Dict[str, Any]) -> UIComponent:
        """
        Recursively load a component and its children.

        Args:
            data: Component data dictionary

        Returns:
            UIComponent instance
        """
        component_type = data.get("type", "Component")
        component_class = get_component_class(component_type)

        if not component_class:
            raise ValueError(f"Unknown component type: {component_type}")

        # Create component using its from_dict method
        component = component_class.from_dict(data)

        # Load children recursively
        children_data = data.get("children", [])
        for child_data in children_data:
            child = self._load_component(child_data)
            component.add_child(child)

        return component

    def save_file(self, component: UIComponent, path: str | Path) -> None:
        """
        Save a UI definition to a YAML file.

        Args:
            component: Root component to save
            path: Output path
        """
        path = Path(path)
        data = {
            "version": 1,
            "root": component.to_dict()
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_ui(path: str | Path) -> UIComponent:
    """
    Convenience function to load a UI file.

    Args:
        path: Path to ui.yaml

    Returns:
        Root UIComponent tree
    """
    loader = UILoader()
    return loader.load_file(path)


def create_default_ui() -> UIComponent:
    """
    Create the default UI for new projects.

    Returns a fullscreen Panel with a welcome message.
    """
    from .components.panel import Panel
    from .components.label import Label

    root = Panel(name="root")
    root.background = "#1a1a1a"
    root.set_anchors(left=True, top=True, right=True, bottom=True)

    # Welcome message
    title = Label(name="title", text="NoodleStudio Runtime")
    title.text_color = "#ffffff"
    title.font_size = 24
    title.set_geometry(20, 20, 400, 40)
    root.add_child(title)

    subtitle = Label(name="subtitle", text="No ui.yaml loaded - showing default canvas")
    subtitle.text_color = "#888888"
    subtitle.font_size = 14
    subtitle.set_geometry(20, 70, 500, 24)
    root.add_child(subtitle)

    return root


def create_default_ui_yaml() -> str:
    """
    Get the default ui.yaml content for new projects.

    Returns:
        YAML string
    """
    return """# NoodleStudio UI Definition
# Design your application interface here
version: 1
root:
  type: Panel
  name: "root"
  background: "#1a1a1a"
  anchors: [left, right, top, bottom]
  children:
    # Add a RadianceViewport for 3D content:
    # - type: RadianceViewport
    #   name: "viewport"
    #   anchors: [left, right, top, bottom]
    #   stage: "main_stage"

    # Or add UI components:
    - type: Label
      name: "title"
      text: "Welcome to NoodleStudio"
      x: 20
      y: 20
      width: 300
      height: 32
      font_size: 24
"""
