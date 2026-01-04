"""
Label Component

Static text display. Equivalent to Delphi's TLabel.
"""

from typing import Any, Dict, Optional
from enum import Enum

from ..component import UIComponent, register_component


class TextAlign(Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class TextVAlign(Enum):
    """Vertical text alignment options."""
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


@register_component
class Label(UIComponent):
    """
    Static text display component.

    Properties:
        text: The displayed text
        text_color: Text color (hex string)
        font_size: Font size in pixels
        font_weight: Font weight ("normal", "bold")
        font_family: Font family name
        align: Horizontal text alignment
        valign: Vertical text alignment
        word_wrap: Enable word wrapping
    """

    component_type = "Label"

    def __init__(self, name: str = "", text: str = ""):
        super().__init__(name)
        self.text: str = text
        self.text_color: str = "#ffffff"
        self.font_size: int = 14
        self.font_weight: str = "normal"
        self.font_family: Optional[str] = None
        self.align: TextAlign = TextAlign.LEFT
        self.valign: TextVAlign = TextVAlign.MIDDLE
        self.word_wrap: bool = False

        # Default size for labels
        self.geometry.width = 100
        self.geometry.height = 24

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Label-specific properties to serialization."""
        data["text"] = self.text

        if self.text_color != "#ffffff":
            data["text_color"] = self.text_color
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.font_weight != "normal":
            data["font_weight"] = self.font_weight
        if self.font_family:
            data["font_family"] = self.font_family
        if self.align != TextAlign.LEFT:
            data["align"] = self.align.value
        if self.valign != TextVAlign.MIDDLE:
            data["valign"] = self.valign.value
        if self.word_wrap:
            data["word_wrap"] = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Label':
        """Deserialize from dictionary."""
        label = cls(
            name=data.get("name", ""),
            text=data.get("text", "")
        )

        # Apply base properties (geometry, anchors, events, bindings)
        label._apply_base_properties(data)

        # Override geometry defaults for labels
        label.geometry.width = data.get("width", 100)
        label.geometry.height = data.get("height", 24)

        # Label-specific properties
        label.text_color = data.get("text_color", "#ffffff")
        label.font_size = data.get("font_size", 14)
        label.font_weight = data.get("font_weight", "normal")
        label.font_family = data.get("font_family")
        label.word_wrap = data.get("word_wrap", False)

        if "align" in data:
            label.align = TextAlign(data["align"])
        if "valign" in data:
            label.valign = TextVAlign(data["valign"])

        return label
