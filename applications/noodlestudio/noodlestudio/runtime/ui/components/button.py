"""
Button Component

Clickable button. Equivalent to Delphi's TButton.
"""

from typing import Any, Dict, Optional

from ..component import UIComponent, register_component


@register_component
class Button(UIComponent):
    """
    Clickable button component.

    Properties:
        text: Button label text
        text_color: Text color (hex string)
        background: Background color (hex string)
        hover_background: Background color on hover
        pressed_background: Background color when pressed
        font_size: Font size in pixels
        border_radius: Corner radius in pixels

    Events:
        onClick: Triggered when button is clicked
        onHover: Triggered when mouse enters
        onLeave: Triggered when mouse leaves
    """

    component_type = "Button"

    def __init__(self, name: str = "", text: str = "Button"):
        super().__init__(name)
        self.text: str = text
        self.text_color: str = "#ffffff"
        self.background: str = "#3b82f6"  # Blue
        self.hover_background: Optional[str] = None  # Slightly lighter if not set
        self.pressed_background: Optional[str] = None  # Slightly darker if not set
        self.font_size: int = 14
        self.border_radius: int = 4

        # Default size for buttons
        self.geometry.width = 80
        self.geometry.height = 32

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Button-specific properties to serialization."""
        data["text"] = self.text

        if self.text_color != "#ffffff":
            data["text_color"] = self.text_color
        if self.background != "#3b82f6":
            data["background"] = self.background
        if self.hover_background:
            data["hover_background"] = self.hover_background
        if self.pressed_background:
            data["pressed_background"] = self.pressed_background
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.border_radius != 4:
            data["border_radius"] = self.border_radius

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Button':
        """Deserialize from dictionary."""
        button = cls(
            name=data.get("name", ""),
            text=data.get("text", "Button")
        )

        # Base properties
        button.geometry.x = data.get("x", 0)
        button.geometry.y = data.get("y", 0)
        button.geometry.width = data.get("width", 80)
        button.geometry.height = data.get("height", 32)

        if "anchors" in data:
            from ..component import Anchors
            button.anchors = Anchors.from_list(data["anchors"])

        button.visible = data.get("visible", True)
        button.enabled = data.get("enabled", True)

        # Button-specific
        button.text_color = data.get("text_color", "#ffffff")
        button.background = data.get("background", "#3b82f6")
        button.hover_background = data.get("hover_background")
        button.pressed_background = data.get("pressed_background")
        button.font_size = data.get("font_size", 14)
        button.border_radius = data.get("border_radius", 4)

        # Events are handled by base class
        if "events" in data:
            from ..component import EventBinding
            for event_name, event_data in data["events"].items():
                button.events[event_name] = EventBinding(
                    action=event_data.get("action", ""),
                    target=event_data.get("target"),
                    message_source=event_data.get("message_source"),
                    params=event_data.get("params", {}),
                )

        return button
