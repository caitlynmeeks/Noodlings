"""
Component Collection - Manages components attached to an entity.

Like Unity's GameObject.GetComponent<T>() / AddComponent<T>() pattern.

Each entity (Noodling, Prop, Zone, etc.) has a ComponentCollection that:
- Stores all attached components
- Handles add/remove with dependency resolution
- Provides type-safe component lookup
- Serializes/deserializes to YAML

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Set, Iterator
import logging

from .component_base import (
    ComponentBase,
    ComponentRegistry,
    component_registry,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=ComponentBase)


class ComponentCollection:
    """
    Collection of components attached to an entity.

    Usage:
        collection = ComponentCollection(entity_id="red_fire_anklebiter")

        # Add components
        collection.add("artbook")
        collection.add("radiance")

        # Get components
        artbook = collection.get("artbook")
        radiance = collection.get_typed(RadianceComponent)

        # Iterate
        for component in collection:
            print(component.display_name)

        # Serialize
        data = collection.to_dict()
        collection.from_dict(data)
    """

    def __init__(self, entity_id: str = "", entity: Any = None):
        """
        Create a new component collection.

        Args:
            entity_id: ID of the owning entity
            entity: Optional reference to the entity object
        """
        self._entity_id = entity_id
        self._entity = entity
        self._components: Dict[str, ComponentBase] = {}  # type -> component
        self._order: List[str] = []  # Preserve add order
        self._dirty = False

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @entity_id.setter
    def entity_id(self, value: str):
        self._entity_id = value
        # Update all components
        for comp in self._components.values():
            comp.entity_id = value

    @property
    def entity(self) -> Any:
        return self._entity

    @entity.setter
    def entity(self, value: Any):
        self._entity = value

    @property
    def is_dirty(self) -> bool:
        """Check if collection or any component has unsaved changes."""
        if self._dirty:
            return True
        return any(c.is_dirty for c in self._components.values())

    def clear_dirty(self):
        """Clear dirty flags on collection and all components."""
        self._dirty = False
        for comp in self._components.values():
            comp.clear_dirty()

    # ==========================================================================
    # Component access
    # ==========================================================================

    def __contains__(self, type_name: str) -> bool:
        """Check if component type is present."""
        return type_name in self._components

    def __iter__(self) -> Iterator[ComponentBase]:
        """Iterate over components in add order."""
        for type_name in self._order:
            if type_name in self._components:
                yield self._components[type_name]

    def __len__(self) -> int:
        return len(self._components)

    def get(self, type_name: str) -> Optional[ComponentBase]:
        """
        Get component by type name.

        Args:
            type_name: Component type (e.g., "artbook", "radiance")

        Returns:
            Component instance or None
        """
        return self._components.get(type_name)

    def get_typed(self, component_class: Type[T]) -> Optional[T]:
        """
        Get component with type checking.

        Args:
            component_class: Component class to retrieve

        Returns:
            Typed component instance or None
        """
        # Create temp instance to get type
        temp = component_class.__new__(component_class)
        temp.__init__("")
        type_name = temp.component_type

        component = self._components.get(type_name)
        if component is not None and isinstance(component, component_class):
            return component
        return None

    def get_all(self) -> List[ComponentBase]:
        """Get all components in add order."""
        return list(self)

    def get_types(self) -> Set[str]:
        """Get set of all component type names."""
        return set(self._components.keys())

    # ==========================================================================
    # Add / Remove
    # ==========================================================================

    def add(
        self,
        type_name: str,
        auto_add_dependencies: bool = True,
        **kwargs
    ) -> Optional[ComponentBase]:
        """
        Add a component by type name.

        Args:
            type_name: Component type to add
            auto_add_dependencies: If True, automatically add missing dependencies
            **kwargs: Additional arguments passed to component constructor

        Returns:
            New component instance, or existing if singleton and already present

        Raises:
            ValueError: If dependencies are missing and auto_add=False
        """
        # Check if already present (for singletons)
        if type_name in self._components:
            existing = self._components[type_name]
            if existing.singleton:
                logger.debug(f"Singleton component '{type_name}' already present")
                return existing

        # Check dependencies
        missing = component_registry.check_dependencies(
            type_name,
            self.get_types()
        )

        if missing:
            if auto_add_dependencies:
                # Recursively add dependencies first
                for dep_type in missing:
                    logger.info(f"Auto-adding dependency: {dep_type}")
                    dep_result = self.add(dep_type, auto_add_dependencies=True)
                    if dep_result is None:
                        logger.error(f"Failed to add dependency: {dep_type}")
                        return None
            else:
                raise ValueError(
                    f"Component '{type_name}' requires: {', '.join(missing)}"
                )

        # Create the component
        component = component_registry.create(type_name, self._entity_id, **kwargs)
        if component is None:
            return None

        # Add to collection
        self._components[type_name] = component
        if type_name not in self._order:
            self._order.append(type_name)

        # Notify component
        component.on_added(self._entity)

        self._dirty = True
        logger.info(f"Added component '{type_name}' to entity '{self._entity_id}'")

        return component

    def add_instance(self, component: ComponentBase) -> bool:
        """
        Add an existing component instance.

        Useful for components created externally (e.g., deserialized).

        Args:
            component: Component instance to add

        Returns:
            True if added, False if singleton collision
        """
        type_name = component.component_type

        # Check singleton constraint
        if type_name in self._components:
            existing = self._components[type_name]
            if existing.singleton:
                logger.warning(f"Cannot add duplicate singleton: {type_name}")
                return False

        # Update entity reference
        component.entity_id = self._entity_id

        # Add to collection
        self._components[type_name] = component
        if type_name not in self._order:
            self._order.append(type_name)

        component.on_added(self._entity)

        self._dirty = True
        return True

    def remove(self, type_name: str) -> bool:
        """
        Remove a component by type name.

        Args:
            type_name: Component type to remove

        Returns:
            True if removed, False if not present
        """
        if type_name not in self._components:
            return False

        # Check if other components depend on this one
        dependents = self._find_dependents(type_name)
        if dependents:
            logger.warning(
                f"Cannot remove '{type_name}': required by {dependents}"
            )
            # TODO: Could cascade remove, but that's dangerous
            return False

        component = self._components.pop(type_name)
        if type_name in self._order:
            self._order.remove(type_name)

        component.on_removed()

        self._dirty = True
        logger.info(f"Removed component '{type_name}' from entity '{self._entity_id}'")

        return True

    def _find_dependents(self, type_name: str) -> List[str]:
        """Find components that depend on the given type."""
        dependents = []
        for other_type in self._components.keys():
            if other_type == type_name:
                continue
            deps = component_registry.get_dependencies(other_type)
            if type_name in deps:
                dependents.append(other_type)
        return dependents

    def clear(self):
        """Remove all components."""
        for type_name in list(self._components.keys()):
            self.remove(type_name)
        self._order.clear()
        self._dirty = True

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize collection to dictionary.

        Returns:
            Dict with 'components' list for YAML storage
        """
        components_list = []
        for type_name in self._order:
            if type_name in self._components:
                component = self._components[type_name]
                try:
                    data = component.to_dict()
                    components_list.append(data)
                except Exception as e:
                    logger.error(f"Failed to serialize component '{type_name}': {e}")

        return {
            'components': components_list
        }

    def from_dict(self, data: Dict[str, Any]) -> int:
        """
        Deserialize collection from dictionary.

        Args:
            data: Dict with 'components' list

        Returns:
            Number of components loaded
        """
        self.clear()

        components_list = data.get('components', [])
        count = 0

        for comp_data in components_list:
            component = component_registry.deserialize(comp_data, self._entity_id)
            if component:
                self.add_instance(component)
                count += 1
            else:
                logger.warning(f"Failed to load component: {comp_data.get('type', 'unknown')}")

        self._dirty = False
        return count

    # ==========================================================================
    # Utility
    # ==========================================================================

    def validate(self) -> List[str]:
        """
        Validate all components.

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        for component in self:
            comp_errors = component.validate()
            for err in comp_errors:
                errors.append(f"{component.display_name}: {err}")
        return errors

    def get_display_list(self) -> List[Dict[str, Any]]:
        """
        Get list of component info for Inspector display.

        Returns:
            List of dicts with display_name, type, category, border_color, enabled
        """
        result = []
        for component in self:
            result.append({
                'type': component.component_type,
                'display_name': component.display_name,
                'category': component.category.value,
                'border_color': component.border_color,
                'enabled': component.enabled,
                'description': component.description,
            })
        return result


__all__ = [
    'ComponentCollection',
]
