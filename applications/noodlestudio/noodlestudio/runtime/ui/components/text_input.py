"""
TextInput Component

Single-line text input. Equivalent to Delphi's TEdit.
"""

from typing import Any, Dict, Optional

from ..component import UIComponent, register_component


@register_component
class TextInput(UIComponent):
    """
    Single-line text input component.

    Properties:
        value: Current text value
        placeholder: Placeholder text when empty
        text_color: Text color (hex string)
        background: Background color (hex string)
        placeholder_color: Placeholder text color
        font_size: Font size in pixels
        border_color: Border color
        border_width: Border width in pixels
        border_radius: Corner radius in pixels
        max_length: Maximum character count (0 = unlimited)
        read_only: If true, text cannot be edited

    Events:
        onChange: Triggered when value changes
        onSubmit: Triggered when Enter is pressed
        onFocus: Triggered when input gains focus
        onBlur: Triggered when input loses focus
    """

    component_type = "TextInput"

    def __init__(self, name: str = "", placeholder: str = ""):
        super().__init__(name)
        self.value: str = ""
        self.placeholder: str = placeholder
        self.text_color: str = "#ffffff"
        self.background: str = "#1a1a1a"
        self.placeholder_color: str = "#666666"
        self.font_size: int = 14
        self.border_color: str = "#3a3a3a"
        self.border_width: int = 1
        self.border_radius: int = 4
        self.max_length: int = 0  # 0 = unlimited
        self.read_only: bool = False

        # Default size for text inputs
        self.geometry.width = 200
        self.geometry.height = 32

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add TextInput-specific properties to serialization."""
        if self.value:
            data["value"] = self.value
        if self.placeholder:
            data["placeholder"] = self.placeholder

        if self.text_color != "#ffffff":
            data["text_color"] = self.text_color
        if self.background != "#1a1a1a":
            data["background"] = self.background
        if self.placeholder_color != "#666666":
            data["placeholder_color"] = self.placeholder_color
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.border_color != "#3a3a3a":
            data["border_color"] = self.border_color
        if self.border_width != 1:
            data["border_width"] = self.border_width
        if self.border_radius != 4:
            data["border_radius"] = self.border_radius
        if self.max_length > 0:
            data["max_length"] = self.max_length
        if self.read_only:
            data["read_only"] = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextInput':
        """Deserialize from dictionary."""
        text_input = cls(
            name=data.get("name", ""),
            placeholder=data.get("placeholder", "")
        )

        # Apply base properties (geometry, anchors, events, bindings)
        text_input._apply_base_properties(data)

        # Override geometry defaults for text inputs
        text_input.geometry.width = data.get("width", 200)
        text_input.geometry.height = data.get("height", 32)

        # TextInput-specific properties
        text_input.value = data.get("value", "")
        text_input.text_color = data.get("text_color", "#ffffff")
        text_input.background = data.get("background", "#1a1a1a")
        text_input.placeholder_color = data.get("placeholder_color", "#666666")
        text_input.font_size = data.get("font_size", 14)
        text_input.border_color = data.get("border_color", "#3a3a3a")
        text_input.border_width = data.get("border_width", 1)
        text_input.border_radius = data.get("border_radius", 4)
        text_input.max_length = data.get("max_length", 0)
        text_input.read_only = data.get("read_only", False)

        return text_input
