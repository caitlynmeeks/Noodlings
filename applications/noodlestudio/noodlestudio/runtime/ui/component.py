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
#   UI Component System - Base Classes
#
#   The foundation of the Delphi-style UI canvas. Users inter...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.component
# PURPOSE:  UI Component System - Base Classes
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AnchorEdge, Anchors, Geometry, EventBinding, UIComponent
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class AnchorEdge(Enum):
    """Which edges a component is anchored to."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class Anchors:
    """
    Delphi-style anchor system.

    Components resize intelligently based on which edges are anchored:
    - [left, top]: Fixed position (default)
    - [right, bottom]: Sticks to bottom-right corner
    - [left, right, top]: Stretches horizontally with parent
    - [left, right, top, bottom]: Fills parent
    """
    left: bool = True
    top: bool = True
    right: bool = False
    bottom: bool = False

    @classmethod
    def from_list(cls, edges: List[str]) -> 'Anchors':
        """Create Anchors from a list of edge names."""
        return cls(
            left="left" in edges,
            top="top" in edges,
            right="right" in edges,
            bottom="bottom" in edges
        )

    def to_list(self) -> List[str]:
        """Convert to list of edge names."""
        edges = []
        if self.left:
            edges.append("left")
        if self.top:
            edges.append("top")
        if self.right:
            edges.append("right")
        if self.bottom:
            edges.append("bottom")
        return edges


@dataclass
class Geometry:
    """Component position and size."""
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 32

    # Cached distances from parent edges (for anchor calculations)
    _margin_right: int = 0
    _margin_bottom: int = 0


@dataclass
class EventBinding:
    """
    Binding from a UI event to an action.

    Actions:
    - send_to_noodling: Send message to a noodling
    - call_script: Execute inline JavaScript or script file
    - set_value: Set another component's value
    - show/hide/toggle_visible: Control component visibility
    - run_assembly: Execute a facet assembly one-shot

    Attributes:
        action: The action type to perform
        target: Target name (noodling, script, or component)
        message_source: Component to get message text from
        chat_history: Component name for chat history display (for send_to_noodling)
        script: Inline JavaScript code (for call_script action)
        script_file: Path to JavaScript file (for call_script action)
        assembly: Path to assembly YAML file (for run_assembly action)
        inputs: Input bindings for assembly (for run_assembly action)
        outputs: Output bindings for assembly (for run_assembly action)
        params: Additional parameters
    """
    action: str
    target: Optional[str] = None
    message_source: Optional[str] = None
    chat_history: Optional[str] = None
    script: Optional[str] = None
    script_file: Optional[str] = None
    assembly: Optional[str] = None  # For run_assembly
    inputs: Dict[str, Any] = field(default_factory=dict)  # For run_assembly
    outputs: Dict[str, str] = field(default_factory=dict)  # For run_assembly
    thinking_target: Optional[str] = None  # Show "Thinking..." in this component
    clear_input: bool = False  # Clear the source input after submit
    params: Dict[str, Any] = field(default_factory=dict)


class UIComponent:
    """
    Base class for all UI components.

    This is what users see - Panel, Button, Label, etc. are all subclasses.
    The Qt implementation details are handled by the renderer.
    """

    # Component type name (overridden by subclasses)
    component_type: str = "Component"

    def __init__(self, name: str = ""):
        self.name = name
        self.geometry = Geometry()
        self.anchors = Anchors()
        self.visible: bool = True
        self.enabled: bool = True

        # Parent/children hierarchy
        self.parent: Optional['UIComponent'] = None
        self.children: List['UIComponent'] = []

        # Event bindings: event_name -> EventBinding
        self.events: Dict[str, EventBinding] = {}

        # Property bindings: target_property -> source_expression
        # e.g., {"text": "input.value", "visible": "checkbox.checked"}
        self.bindings: Dict[str, str] = {}

        # Runtime callbacks (set by renderer)
        self._event_callbacks: Dict[str, List[Callable]] = {}

        # Reference to rendered widget (set by renderer)
        self._widget: Any = None

    # --- Hierarchy ---

    def add_child(self, child: 'UIComponent') -> None:
        """Add a child component."""
        if child.parent:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: 'UIComponent') -> None:
        """Remove a child component."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def find_by_name(self, name: str) -> Optional['UIComponent']:
        """Find a descendant by name."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_by_name(name)
            if found:
                return found
        return None

    # --- Events ---

    def bind_event(self, event_name: str, binding: EventBinding) -> None:
        """Bind an event to an action."""
        self.events[event_name] = binding

    def on(self, event_name: str, callback: Callable) -> None:
        """Register a runtime callback for an event."""
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def emit(self, event_name: str, **kwargs) -> None:
        """Emit an event, triggering all registered callbacks."""
        for callback in self._event_callbacks.get(event_name, []):
            callback(**kwargs)

    # --- Geometry helpers ---

    def set_geometry(self, x: int, y: int, width: int, height: int) -> None:
        """Set position and size."""
        self.geometry.x = x
        self.geometry.y = y
        self.geometry.width = width
        self.geometry.height = height

    def set_anchors(self, left: bool = True, top: bool = True,
                    right: bool = False, bottom: bool = False) -> None:
        """Set anchor edges."""
        self.anchors = Anchors(left=left, top=top, right=right, bottom=bottom)

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for YAML)."""
        data: Dict[str, Any] = {
            "type": self.component_type,
            "name": self.name,
            "x": self.geometry.x,
            "y": self.geometry.y,
            "width": self.geometry.width,
            "height": self.geometry.height,
        }

        # Only include non-default anchors
        anchor_list = self.anchors.to_list()
        if anchor_list != ["left", "top"]:
            data["anchors"] = anchor_list

        # Include visibility/enabled only if non-default
        if not self.visible:
            data["visible"] = False
        if not self.enabled:
            data["enabled"] = False

        # Events
        if self.events:
            data["events"] = {
                name: {
                    "action": binding.action,
                    **({"target": binding.target} if binding.target else {}),
                    **({"message_source": binding.message_source} if binding.message_source else {}),
                    **({"chat_history": binding.chat_history} if binding.chat_history else {}),
                    **({"script": binding.script} if binding.script else {}),
                    **({"script_file": binding.script_file} if binding.script_file else {}),
                    **({"assembly": binding.assembly} if binding.assembly else {}),
                    **({"inputs": binding.inputs} if binding.inputs else {}),
                    **({"outputs": binding.outputs} if binding.outputs else {}),
                    **({"params": binding.params} if binding.params else {}),
                }
                for name, binding in self.events.items()
            }

        # Property bindings
        if self.bindings:
            data["bindings"] = self.bindings.copy()

        # Children
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]

        # Subclasses add their own properties
        self._serialize_properties(data)

        return data

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Override in subclasses to add component-specific properties."""
        pass

    def _apply_base_properties(self, data: Dict[str, Any]) -> None:
        """
        Apply base properties from a dictionary.

        Helper method for subclasses to apply base UIComponent properties
        during deserialization.
        """
        self.geometry.x = data.get("x", 0)
        self.geometry.y = data.get("y", 0)
        self.geometry.width = data.get("width", self.geometry.width)
        self.geometry.height = data.get("height", self.geometry.height)

        if "anchors" in data:
            self.anchors = Anchors.from_list(data["anchors"])

        self.visible = data.get("visible", True)
        self.enabled = data.get("enabled", True)

        # Events
        if "events" in data:
            for event_name, event_data in data["events"].items():
                self.events[event_name] = EventBinding(
                    action=event_data.get("action", ""),
                    target=event_data.get("target"),
                    message_source=event_data.get("message_source"),
                    chat_history=event_data.get("chat_history"),
                    script=event_data.get("script"),
                    script_file=event_data.get("script_file"),
                    assembly=event_data.get("assembly"),
                    inputs=event_data.get("inputs", {}),
                    outputs=event_data.get("outputs", {}),
                    thinking_target=event_data.get("thinking_target"),
                    clear_input=event_data.get("clear_input", False),
                    params=event_data.get("params", {}),
                )

        # Property bindings
        if "bindings" in data:
            self.bindings = data["bindings"].copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIComponent':
        """
        Deserialize from dictionary.

        Note: This creates a base UIComponent. The loader uses the component
        registry to create the correct subclass.
        """
        component = cls(name=data.get("name", ""))
        component._apply_base_properties(data)
        return component

    def __repr__(self) -> str:
        return f"<{self.component_type} name='{self.name}' at ({self.geometry.x}, {self.geometry.y})>"


# --- Component Registry ---

_component_registry: Dict[str, type] = {}


def register_component(component_class: type) -> type:
    """Decorator to register a component type."""
    _component_registry[component_class.component_type] = component_class
    return component_class


def get_component_class(type_name: str) -> Optional[type]:
    """Get component class by type name."""
    return _component_registry.get(type_name)


def list_component_types() -> List[str]:
    """List all registered component types."""
    return list(_component_registry.keys())


# Register base component
register_component(UIComponent)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
