"""
RadioButton and RadioGroup Components

Mutually exclusive selection. Equivalent to Delphi's TRadioButton/TRadioGroup.
"""

from typing import Any, Dict, List, Optional

from ..component import UIComponent, register_component


@register_component
class RadioButton(UIComponent):
    """
    Single radio button. Usually used inside a RadioGroup.

    Properties:
        checked: Whether this button is selected
        text: Label text
        value: Value associated with this option (for form data)
        group_name: Name of the group this belongs to (for standalone use)
        text_color: Label color
        radio_color: Radio button circle color
        checked_color: Color when checked
        radio_size: Size of radio circle in pixels
        font_size: Label font size
        spacing: Gap between radio and label

    Events:
        onChange: Triggered when selection state changes
        onSelect: Triggered when this button becomes selected
    """

    component_type = "RadioButton"

    def __init__(
        self,
        name: str = "",
        text: str = "Option",
        value: str = "",
        checked: bool = False
    ):
        super().__init__(name)
        self.checked: bool = checked
        self.text: str = text
        self.option_value: str = value or text  # Value for form data
        self.group_name: str = ""  # Group association for standalone buttons
        self.text_color: str = "#cccccc"
        self.radio_color: str = "#3d3d3d"
        self.checked_color: str = "#76AF6A"
        self.radio_size: int = 18
        self.font_size: int = 14
        self.spacing: int = 8

        # Default size
        self.geometry.width = 150
        self.geometry.height = 24

    @property
    def value(self) -> str:
        """Get the option value."""
        return self.option_value

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add RadioButton-specific properties to serialization."""
        data["text"] = self.text
        data["checked"] = self.checked

        if self.option_value != self.text:
            data["value"] = self.option_value
        if self.group_name:
            data["group_name"] = self.group_name
        if self.text_color != "#cccccc":
            data["text_color"] = self.text_color
        if self.radio_color != "#3d3d3d":
            data["radio_color"] = self.radio_color
        if self.checked_color != "#76AF6A":
            data["checked_color"] = self.checked_color
        if self.radio_size != 18:
            data["radio_size"] = self.radio_size
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.spacing != 8:
            data["spacing"] = self.spacing

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadioButton':
        """Deserialize from dictionary."""
        button = cls(
            name=data.get("name", ""),
            text=data.get("text", "Option"),
            value=data.get("value", ""),
            checked=data.get("checked", False)
        )

        button._apply_base_properties(data)

        button.geometry.width = data.get("width", 150)
        button.geometry.height = data.get("height", 24)

        button.group_name = data.get("group_name", "")
        button.text_color = data.get("text_color", "#cccccc")
        button.radio_color = data.get("radio_color", "#3d3d3d")
        button.checked_color = data.get("checked_color", "#76AF6A")
        button.radio_size = data.get("radio_size", 18)
        button.font_size = data.get("font_size", 14)
        button.spacing = data.get("spacing", 8)

        return button


@register_component
class RadioGroup(UIComponent):
    """
    Container for mutually exclusive RadioButtons.

    Properties:
        options: List of option strings (convenience for creating buttons)
        selected_index: Currently selected option index (-1 for none)
        selected_value: Value of currently selected option
        orientation: 'vertical' or 'horizontal'
        spacing: Gap between options
        text_color: Label color for all buttons
        radio_color: Radio circle color
        checked_color: Color when checked
        radio_size: Size of radio circles
        font_size: Label font size

    Events:
        onChange: Triggered when selection changes
    """

    component_type = "RadioGroup"

    def __init__(
        self,
        name: str = "",
        options: Optional[List[str]] = None
    ):
        super().__init__(name)
        self._options: List[str] = options or []
        self.selected_index: int = -1
        self.orientation: str = "vertical"  # or "horizontal"
        self.spacing: int = 8
        self.text_color: str = "#cccccc"
        self.radio_color: str = "#3d3d3d"
        self.checked_color: str = "#76AF6A"
        self.radio_size: int = 18
        self.font_size: int = 14

        # Default size
        self.geometry.width = 200
        self.geometry.height = 100

    @property
    def options(self) -> List[str]:
        """Get options list."""
        return self._options

    @options.setter
    def options(self, opts: List[str]) -> None:
        """Set options, adjusting selection if needed."""
        self._options = opts
        if self.selected_index >= len(opts):
            self.selected_index = -1

    @property
    def value(self) -> Optional[str]:
        """Get currently selected value."""
        if 0 <= self.selected_index < len(self._options):
            return self._options[self.selected_index]
        return None

    @value.setter
    def value(self, val: str) -> None:
        """Set selection by value."""
        try:
            self.selected_index = self._options.index(val)
        except ValueError:
            self.selected_index = -1

    @property
    def selected_value(self) -> Optional[str]:
        """Alias for value."""
        return self.value

    def select(self, index: int) -> bool:
        """Select option by index. Returns True if changed."""
        if 0 <= index < len(self._options):
            if self.selected_index != index:
                self.selected_index = index
                return True
        return False

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add RadioGroup-specific properties to serialization."""
        if self._options:
            data["options"] = self._options.copy()
        if self.selected_index >= 0:
            data["selected_index"] = self.selected_index

        if self.orientation != "vertical":
            data["orientation"] = self.orientation
        if self.spacing != 8:
            data["spacing"] = self.spacing
        if self.text_color != "#cccccc":
            data["text_color"] = self.text_color
        if self.radio_color != "#3d3d3d":
            data["radio_color"] = self.radio_color
        if self.checked_color != "#76AF6A":
            data["checked_color"] = self.checked_color
        if self.radio_size != 18:
            data["radio_size"] = self.radio_size
        if self.font_size != 14:
            data["font_size"] = self.font_size

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadioGroup':
        """Deserialize from dictionary."""
        group = cls(
            name=data.get("name", ""),
            options=data.get("options", [])
        )

        group._apply_base_properties(data)

        group.geometry.width = data.get("width", 200)
        group.geometry.height = data.get("height", 100)

        group.selected_index = data.get("selected_index", -1)
        group.orientation = data.get("orientation", "vertical")
        group.spacing = data.get("spacing", 8)
        group.text_color = data.get("text_color", "#cccccc")
        group.radio_color = data.get("radio_color", "#3d3d3d")
        group.checked_color = data.get("checked_color", "#76AF6A")
        group.radio_size = data.get("radio_size", 18)
        group.font_size = data.get("font_size", 14)

        return group
