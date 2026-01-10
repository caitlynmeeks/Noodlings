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
#   Dropdown Component
#
#   Select/ComboBox component. Equivalent to Delphi's TComboBox.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.dropdown
# PURPOSE:  Dropdown Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Dropdown
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Any, Dict, List, Optional

from ..component import UIComponent, register_component


@register_component
class Dropdown(UIComponent):
    """
    Dropdown/ComboBox component for selecting from options.

    Properties:
        options: List of option strings
        selected_index: Currently selected index (-1 for none)
        placeholder: Text shown when nothing selected
        text_color: Text color (hex string)
        background: Background color (hex string)
        border_color: Border color (hex string)
        hover_background: Background on hover
        dropdown_background: Background of dropdown list
        item_hover_background: Background when hovering over item
        font_size: Font size in pixels
        border_radius: Corner radius in pixels
        editable: Whether user can type custom values

    Events:
        onChange: Triggered when selection changes
        onOpen: Triggered when dropdown opens
        onClose: Triggered when dropdown closes
    """

    component_type = "Dropdown"

    def __init__(
        self,
        name: str = "",
        options: Optional[List[str]] = None,
        selected_index: int = -1
    ):
        super().__init__(name)
        self.options: List[str] = options or []
        self.selected_index: int = selected_index
        self.placeholder: str = "Select..."
        self.text_color: str = "#cccccc"
        self.background: str = "#2d2d2d"
        self.border_color: str = "#3d3d3d"
        self.hover_background: str = "#363636"
        self.dropdown_background: str = "#2d2d2d"
        self.item_hover_background: str = "#3d3d3d"
        self.font_size: int = 14
        self.border_radius: int = 4
        self.editable: bool = False

        # Default size
        self.geometry.width = 200
        self.geometry.height = 32

    @property
    def value(self) -> Optional[str]:
        """Get currently selected value."""
        if 0 <= self.selected_index < len(self.options):
            return self.options[self.selected_index]
        return None

    @value.setter
    def value(self, val: str) -> None:
        """Set selection by value string."""
        try:
            self.selected_index = self.options.index(val)
        except ValueError:
            self.selected_index = -1

    @property
    def selected_text(self) -> str:
        """Get selected text or placeholder."""
        if self.value is not None:
            return self.value
        return self.placeholder

    def add_option(self, option: str) -> int:
        """Add an option and return its index."""
        self.options.append(option)
        return len(self.options) - 1

    def remove_option(self, option: str) -> bool:
        """Remove an option by value. Returns True if removed."""
        try:
            idx = self.options.index(option)
            self.options.pop(idx)
            # Adjust selected_index if needed
            if self.selected_index == idx:
                self.selected_index = -1
            elif self.selected_index > idx:
                self.selected_index -= 1
            return True
        except ValueError:
            return False

    def clear_options(self) -> None:
        """Remove all options."""
        self.options.clear()
        self.selected_index = -1

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add Dropdown-specific properties to serialization."""
        if self.options:
            data["options"] = self.options.copy()
        if self.selected_index >= 0:
            data["selected_index"] = self.selected_index

        if self.placeholder != "Select...":
            data["placeholder"] = self.placeholder
        if self.text_color != "#cccccc":
            data["text_color"] = self.text_color
        if self.background != "#2d2d2d":
            data["background"] = self.background
        if self.border_color != "#3d3d3d":
            data["border_color"] = self.border_color
        if self.hover_background != "#363636":
            data["hover_background"] = self.hover_background
        if self.dropdown_background != "#2d2d2d":
            data["dropdown_background"] = self.dropdown_background
        if self.item_hover_background != "#3d3d3d":
            data["item_hover_background"] = self.item_hover_background
        if self.font_size != 14:
            data["font_size"] = self.font_size
        if self.border_radius != 4:
            data["border_radius"] = self.border_radius
        if self.editable:
            data["editable"] = self.editable

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dropdown':
        """Deserialize from dictionary."""
        dropdown = cls(
            name=data.get("name", ""),
            options=data.get("options", []),
            selected_index=data.get("selected_index", -1)
        )

        dropdown._apply_base_properties(data)

        dropdown.geometry.width = data.get("width", 200)
        dropdown.geometry.height = data.get("height", 32)

        dropdown.placeholder = data.get("placeholder", "Select...")
        dropdown.text_color = data.get("text_color", "#cccccc")
        dropdown.background = data.get("background", "#2d2d2d")
        dropdown.border_color = data.get("border_color", "#3d3d3d")
        dropdown.hover_background = data.get("hover_background", "#363636")
        dropdown.dropdown_background = data.get("dropdown_background", "#2d2d2d")
        dropdown.item_hover_background = data.get("item_hover_background", "#3d3d3d")
        dropdown.font_size = data.get("font_size", 14)
        dropdown.border_radius = data.get("border_radius", 4)
        dropdown.editable = data.get("editable", False)

        return dropdown

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
