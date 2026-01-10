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
#   Component Base - Foundation for NoodleStudio's component system.
#
#   Components are modular, attachable behaviors/data contain...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.component_base
# PURPOSE:  Component Base
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ComponentCategory, PropertySpec, ComponentBase, ComponentRegistry, register_component()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Set, Callable
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class ComponentCategory(Enum):
    """Visual categorization for Inspector display."""
    CHARM = "charm"              # Core consciousness (green border)
    ART_REFERENCE = "art"        # Visual references (orange border)
    BEHAVIOR = "behavior"        # Game mechanics (blue border)
    RENDERING = "rendering"      # Visual presentation (cyan border)
    AUDIO = "audio"              # Sound/voice (purple border)
    CUSTOM = "custom"            # User scripts (gray border)


# Border colors for each category (matching COMPONENT_SYSTEM.md spec)
CATEGORY_COLORS = {
    ComponentCategory.CHARM: "#4CAF50",       # Green
    ComponentCategory.ART_REFERENCE: "#FF9800", # Orange
    ComponentCategory.BEHAVIOR: "#2196F3",    # Blue
    ComponentCategory.RENDERING: "#00BCD4",   # Cyan
    ComponentCategory.AUDIO: "#9C27B0",       # Purple
    ComponentCategory.CUSTOM: "#757575",      # Gray
}


@dataclass
class PropertySpec:
    """
    Specification for an editable property.

    Used by Inspector to generate appropriate UI controls.
    """
    name: str                      # Internal property name
    display_name: str              # Human-readable label
    property_type: str             # 'string', 'int', 'float', 'bool', 'text', 'file', 'color'
    default: Any = None            # Default value
    readonly: bool = False         # Can it be edited?
    description: str = ""          # Tooltip text
    min_value: Optional[float] = None   # For numeric types
    max_value: Optional[float] = None   # For numeric types
    options: Optional[List[str]] = None # For dropdown/enum types
    file_filter: str = ""          # For file picker (e.g., "Images (*.png *.jpg)")


class ComponentBase(ABC):
    """
    Base class for all NoodleStudio components.

    Subclasses must implement:
    - component_type: Unique type identifier (e.g., "artbook", "facet_assembly")
    - display_name: Human-readable name for Inspector
    - category: ComponentCategory for visual grouping
    - property_specs: List of PropertySpec for editable properties
    - to_dict() / from_dict(): Serialization

    Optional overrides:
    - dependencies: List of required component types
    - singleton: Whether only one instance is allowed per entity
    - on_added(): Called when component is added to entity
    - on_removed(): Called when component is removed
    - validate(): Check if component state is valid
    """

    def __init__(self, entity_id: str = ""):
        self._id: str = str(uuid.uuid4())
        self._entity_id: str = entity_id
        self._enabled: bool = True
        self._dirty: bool = False

    # ==========================================================================
    # Abstract properties - subclasses MUST implement
    # ==========================================================================

    @property
    @abstractmethod
    def component_type(self) -> str:
        """Unique type identifier (e.g., 'artbook', 'facet_assembly')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in Inspector header."""
        pass

    @property
    @abstractmethod
    def category(self) -> ComponentCategory:
        """Category for visual grouping and border color."""
        pass

    @property
    @abstractmethod
    def property_specs(self) -> List[PropertySpec]:
        """List of editable properties for Inspector."""
        pass

    # ==========================================================================
    # Optional properties - subclasses MAY override
    # ==========================================================================

    @property
    def dependencies(self) -> List[str]:
        """Component types that must be present before this one can be added."""
        return []

    @property
    def singleton(self) -> bool:
        """If True, only one instance allowed per entity. Default: True."""
        return True

    @property
    def description(self) -> str:
        """Brief description shown in Inspector."""
        return ""

    @property
    def icon(self) -> Optional[str]:
        """Optional icon path or emoji for display."""
        return None

    # ==========================================================================
    # Core properties
    # ==========================================================================

    @property
    def id(self) -> str:
        """Unique instance ID."""
        return self._id

    @property
    def entity_id(self) -> str:
        """ID of the entity this component is attached to."""
        return self._entity_id

    @entity_id.setter
    def entity_id(self, value: str):
        self._entity_id = value

    @property
    def enabled(self) -> bool:
        """Is this component active?"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        self._mark_dirty()

    @property
    def is_dirty(self) -> bool:
        """Has this component been modified since last save?"""
        return self._dirty

    @property
    def border_color(self) -> str:
        """Get the border color for this component's category."""
        return CATEGORY_COLORS.get(self.category, "#757575")

    # ==========================================================================
    # Lifecycle hooks
    # ==========================================================================

    def on_added(self, entity: Any) -> None:
        """
        Called when this component is added to an entity.

        Override to perform initialization that requires the entity context.
        """
        pass

    def on_removed(self) -> None:
        """
        Called when this component is removed from its entity.

        Override to perform cleanup.
        """
        pass

    def on_property_changed(self, property_name: str, old_value: Any, new_value: Any) -> None:
        """
        Called when a property is modified through the Inspector.

        Override to react to property changes.
        """
        pass

    # ==========================================================================
    # Validation
    # ==========================================================================

    def validate(self) -> List[str]:
        """
        Validate component state.

        Returns:
            List of error messages (empty if valid)
        """
        return []

    # ==========================================================================
    # Property access
    # ==========================================================================

    def get_property(self, name: str) -> Any:
        """Get a property value by name."""
        return getattr(self, name, None)

    def set_property(self, name: str, value: Any) -> bool:
        """
        Set a property value by name.

        Returns:
            True if successful, False if property doesn't exist or is readonly
        """
        # Check if property exists in specs
        spec = next((s for s in self.property_specs if s.name == name), None)
        if spec is None:
            return False
        if spec.readonly:
            return False

        old_value = getattr(self, name, None)
        setattr(self, name, value)
        self._mark_dirty()
        self.on_property_changed(name, old_value, value)
        return True

    def _mark_dirty(self):
        """Mark component as modified."""
        self._dirty = True

    def clear_dirty(self):
        """Clear dirty flag (called after save)."""
        self._dirty = False

    # ==========================================================================
    # Serialization
    # ==========================================================================

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to dictionary for YAML storage.

        Must include 'type' key with component_type value.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], entity_id: str = "") -> 'ComponentBase':
        """
        Deserialize component from dictionary.

        Args:
            data: Dictionary from YAML
            entity_id: ID of the owning entity
        """
        pass

    def _base_to_dict(self) -> Dict[str, Any]:
        """Helper to serialize base properties."""
        return {
            'type': self.component_type,
            'id': self._id,
            'enabled': self._enabled,
        }

    def _base_from_dict(self, data: Dict[str, Any]):
        """Helper to deserialize base properties."""
        self._id = data.get('id', str(uuid.uuid4()))
        self._enabled = data.get('enabled', True)


class ComponentRegistry:
    """
    Registry of all available component types.

    Handles:
    - Component type registration
    - Dependency resolution
    - Component instantiation
    - Type lookup
    """

    _instance: Optional['ComponentRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components: Dict[str, Type[ComponentBase]] = {}
            cls._instance._by_category: Dict[ComponentCategory, List[str]] = {
                cat: [] for cat in ComponentCategory
            }
        return cls._instance

    def register(self, component_class: Type[ComponentBase]) -> None:
        """
        Register a component type.

        Args:
            component_class: The component class to register
        """
        # Create temporary instance to get type info
        temp = component_class.__new__(component_class)
        temp.__init__("")

        type_name = temp.component_type
        category = temp.category

        if type_name in self._components:
            logger.warning(f"Component type '{type_name}' already registered, overwriting")

        self._components[type_name] = component_class

        if type_name not in self._by_category[category]:
            self._by_category[category].append(type_name)

        logger.debug(f"Registered component: {type_name} ({category.value})")

    def get_class(self, type_name: str) -> Optional[Type[ComponentBase]]:
        """Get component class by type name."""
        return self._components.get(type_name)

    def create(self, type_name: str, entity_id: str = "", **kwargs) -> Optional[ComponentBase]:
        """
        Create a new component instance.

        Args:
            type_name: Component type to create
            entity_id: ID of owning entity
            **kwargs: Additional constructor arguments

        Returns:
            New component instance, or None if type not found
        """
        component_class = self._components.get(type_name)
        if component_class is None:
            logger.error(f"Unknown component type: {type_name}")
            return None

        try:
            component = component_class(entity_id=entity_id, **kwargs)
            return component
        except Exception as e:
            logger.error(f"Failed to create component '{type_name}': {e}")
            return None

    def deserialize(self, data: Dict[str, Any], entity_id: str = "") -> Optional[ComponentBase]:
        """
        Deserialize a component from dictionary data.

        Args:
            data: Dictionary with 'type' key
            entity_id: ID of owning entity

        Returns:
            Deserialized component, or None if type not found
        """
        type_name = data.get('type')
        if not type_name:
            logger.error("Component data missing 'type' key")
            return None

        component_class = self._components.get(type_name)
        if component_class is None:
            logger.warning(f"Unknown component type in data: {type_name}")
            return None

        try:
            return component_class.from_dict(data, entity_id)
        except Exception as e:
            logger.error(f"Failed to deserialize component '{type_name}': {e}")
            return None

    def get_all_types(self) -> List[str]:
        """Get all registered component type names."""
        return list(self._components.keys())

    def get_by_category(self, category: ComponentCategory) -> List[str]:
        """Get component types in a category."""
        return self._by_category.get(category, [])

    def get_dependencies(self, type_name: str) -> List[str]:
        """Get dependency list for a component type."""
        component_class = self._components.get(type_name)
        if component_class is None:
            return []

        # Create temp instance to get dependencies
        temp = component_class.__new__(component_class)
        temp.__init__("")
        return temp.dependencies

    def check_dependencies(self, type_name: str, existing_types: Set[str]) -> List[str]:
        """
        Check which dependencies are missing.

        Args:
            type_name: Component type to check
            existing_types: Set of already-present component types

        Returns:
            List of missing dependency type names
        """
        deps = self.get_dependencies(type_name)
        return [d for d in deps if d not in existing_types]

    def get_display_info(self, type_name: str) -> Dict[str, Any]:
        """
        Get display info for Inspector menus.

        Returns dict with: display_name, category, description, icon
        """
        component_class = self._components.get(type_name)
        if component_class is None:
            return {}

        temp = component_class.__new__(component_class)
        temp.__init__("")

        return {
            'type': type_name,
            'display_name': temp.display_name,
            'category': temp.category.value,
            'description': temp.description,
            'icon': temp.icon,
            'singleton': temp.singleton,
            'dependencies': temp.dependencies,
            'border_color': temp.border_color,
        }


# Global registry instance
component_registry = ComponentRegistry()


def register_component(cls: Type[ComponentBase]) -> Type[ComponentBase]:
    """
    Decorator to register a component class.

    Usage:
        @register_component
        class MyComponent(ComponentBase):
            ...
    """
    component_registry.register(cls)
    return cls


__all__ = [
    'ComponentBase',
    'ComponentCategory',
    'ComponentRegistry',
    'PropertySpec',
    'CATEGORY_COLORS',
    'component_registry',
    'register_component',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
