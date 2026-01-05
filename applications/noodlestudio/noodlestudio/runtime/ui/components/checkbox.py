"""
Checkbox Component

Boolean toggle with label. Equivalent to Delphi's TCheckBox.
"""

from typing import Any, Dict, Optional

from ..component import UIComponent, register_component


@register_component
class Checkbox(UIComponent):
    """
    Checkbox component with label.

    Properties:
        checked: Boolean state
        text: Label text displayed next to checkbox
        text_color: Label color (hex string)
        check_color: Checkmark color when checked
        box_color: Checkbox box border/background color
        box_size: Size of checkbox box in pixels
        font_size: Label font size in pixels
        spacing: Gap between checkbox and label

    Events:
        onChange: Triggered when checked state changes
        onCheck: Triggered when checked becomes True
        onUncheck: Triggered when checked becomes False
    """

    component_type = "Checkbox"

    def __init__(self, name: str = "", text: str = "Checkbox", checked: bool = False):
        super().__init__(name)
        self.checked: bool = checked
        self.text: str = text
        self.text_color: str = "#cccccc"
        self.check_color: str = "#76AF6A"  # Noodle green
        self.box_color: str = "#3d3d3d"
        self.box_size: int = 18
        self.font_size: int = 14
        self.spacing: int = 8

        # Default size
        self.geometry.width = 150
        self.geometry.height = 24

    @property
    def value(self) -> bool:
        """Alias for checked state (for consistency with other input components)."""
        return self.checked

    @value.setter
    def value(self, val: bool) -> None:
        self.checked = val

    def toggle(self) -> bool:
        """Toggle checked state and return new state."""
        self.checked = not self.checked
        return self.checked

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Checkbox-specific properties to serialization."""
        data["text"] = self.text
        data["checked"] = self.checked

        if self.text_color != "#cccccc":
            data["text_color"] = self.text_color
        if self.check_color != "#76AF6A":
            data["check_color"] = self.check_color
        if self.box_color != "#3d3d3d":
            data["box_color"] = self.box_color
        if self.box_size != 18:
            data["box_size"] = self.box_size
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.spacing != 8:
            data["spacing"] = self.spacing

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkbox':
        """Deserialize from dictionary."""
        checkbox = cls(
            name=data.get("name", ""),
            text=data.get("text", "Checkbox"),
            checked=data.get("checked", False)
        )

        checkbox._apply_base_properties(data)

        checkbox.geometry.width = data.get("width", 150)
        checkbox.geometry.height = data.get("height", 24)

        checkbox.text_color = data.get("text_color", "#cccccc")
        checkbox.check_color = data.get("check_color", "#76AF6A")
        checkbox.box_color = data.get("box_color", "#3d3d3d")
        checkbox.box_size = data.get("box_size", 18)
        checkbox.font_size = data.get("font_size", 14)
        checkbox.spacing = data.get("spacing", 8)

        return checkbox
