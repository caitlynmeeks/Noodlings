"""
ChatInput Component

A text input field with send button for composing messages.
"""

from ..component import UIComponent, EventBinding, register_component


@register_component
class ChatInput(UIComponent):
    """
    Chat message input with send button.

    A compound component containing a text input and send button.
    Typically placed at the bottom of a chat interface.

    Properties:
        placeholder: Placeholder text for empty input
        value: Current input value
        send_button_text: Text on send button
        background: Background color of the input area
        input_background: Background of the text field
        button_background: Background of the send button
        text_color: Color of input text
        button_text_color: Color of button text
        border_color: Border color for input
        border_radius: Border radius for elements
        font_size: Font size for text
        max_length: Maximum input length (0 = unlimited)

    Events:
        onSubmit: Fired when user presses Enter or clicks Send
    """

    component_type: str = "ChatInput"

    def __init__(self, name: str = "", placeholder: str = "Type a message..."):
        super().__init__(name)

        # Input state
        self.placeholder: str = placeholder
        self.value: str = ""
        self.max_length: int = 0

        # Button text
        self.send_button_text: str = "Send"

        # Styling - container
        self.background: str = "#1f1f1f"
        self.border_radius: int = 0

        # Styling - input field
        self.input_background: str = "#2a2a2a"
        self.text_color: str = "#ffffff"
        self.placeholder_color: str = "#6b7280"
        self.border_color: str = "#3a3a3a"
        self.input_border_radius: int = 8

        # Styling - button
        self.button_background: str = "#3b82f6"
        self.button_text_color: str = "#ffffff"
        self.button_border_radius: int = 8

        # Font
        self.font_size: int = 14

        # Layout
        self.padding: int = 12
        self.spacing: int = 8  # Space between input and button

        # Clear input after send
        self.clear_on_submit: bool = True

    def to_dict(self) -> dict:
        """Serialize to dict."""
        data = super().to_dict()
        data.update({
            "placeholder": self.placeholder,
            "value": self.value,
            "max_length": self.max_length,
            "send_button_text": self.send_button_text,
            "background": self.background,
            "border_radius": self.border_radius,
            "input_background": self.input_background,
            "text_color": self.text_color,
            "placeholder_color": self.placeholder_color,
            "border_color": self.border_color,
            "input_border_radius": self.input_border_radius,
            "button_background": self.button_background,
            "button_text_color": self.button_text_color,
            "button_border_radius": self.button_border_radius,
            "font_size": self.font_size,
            "padding": self.padding,
            "spacing": self.spacing,
            "clear_on_submit": self.clear_on_submit
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatInput':
        """Deserialize from dict."""
        component = cls(
            name=data.get("name", ""),
            placeholder=data.get("placeholder", "Type a message...")
        )
        component._apply_base_properties(data)

        # Load properties
        component.value = data.get("value", "")
        component.max_length = data.get("max_length", 0)
        component.send_button_text = data.get("send_button_text", "Send")
        component.background = data.get("background", component.background)
        component.border_radius = data.get("border_radius", component.border_radius)
        component.input_background = data.get("input_background", component.input_background)
        component.text_color = data.get("text_color", component.text_color)
        component.placeholder_color = data.get("placeholder_color", component.placeholder_color)
        component.border_color = data.get("border_color", component.border_color)
        component.input_border_radius = data.get("input_border_radius", component.input_border_radius)
        component.button_background = data.get("button_background", component.button_background)
        component.button_text_color = data.get("button_text_color", component.button_text_color)
        component.button_border_radius = data.get("button_border_radius", component.button_border_radius)
        component.font_size = data.get("font_size", component.font_size)
        component.padding = data.get("padding", component.padding)
        component.spacing = data.get("spacing", component.spacing)
        component.clear_on_submit = data.get("clear_on_submit", component.clear_on_submit)

        return component
