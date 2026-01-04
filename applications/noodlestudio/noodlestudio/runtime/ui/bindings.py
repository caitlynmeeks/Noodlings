"""
Component Value Binding System

Enables automatic synchronization between component properties.
When a source value changes, bound targets update automatically.

Like Delphi's TDataSource or WPF's data binding, but simpler:
- One-way binding (source -> target)
- Expression support (simple property paths)
- No complex MVVM patterns

Usage in YAML:
    Label:
      name: "status"
      bindings:
        text: "input.value"           # Bind to another component's value
        visible: "checkbox.checked"   # Bind visibility to a checkbox

Author: Caitlyn + Claude
Date: January 3, 2026
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .component import UIComponent
    from .renderer import QtWidgetRenderer

logger = logging.getLogger(__name__)


@dataclass
class Binding:
    """
    A single property binding.

    Attributes:
        target_component: Component name to update
        target_property: Property to set (e.g., "text", "visible")
        source_expression: Source expression (e.g., "input.value")
        transform: Optional transform function
    """
    target_component: str
    target_property: str
    source_expression: str
    transform: Optional[Callable[[Any], Any]] = None


class BindingManager:
    """
    Manages component property bindings.

    Tracks bindings and updates targets when sources change.
    Integrates with QtWidgetRenderer for widget updates.
    """

    def __init__(self, renderer: Optional['QtWidgetRenderer'] = None):
        """
        Initialize binding manager.

        Args:
            renderer: Optional renderer for widget updates
        """
        self.renderer = renderer

        # All bindings: list of Binding objects
        self._bindings: List[Binding] = []

        # Source -> bindings map for efficient updates
        # Key is "component.property", value is list of bindings that depend on it
        self._source_map: Dict[str, List[Binding]] = {}

        # Cached component values for change detection
        self._value_cache: Dict[str, Any] = {}

    def add_binding(
        self,
        target_component: str,
        target_property: str,
        source_expression: str,
        transform: Optional[Callable[[Any], Any]] = None
    ) -> Binding:
        """
        Add a new binding.

        Args:
            target_component: Component to update
            target_property: Property to set
            source_expression: Source expression (e.g., "input.value")
            transform: Optional transform function

        Returns:
            The created Binding object
        """
        binding = Binding(
            target_component=target_component,
            target_property=target_property,
            source_expression=source_expression,
            transform=transform
        )

        self._bindings.append(binding)

        # Add to source map
        source_key = self._normalize_source(source_expression)
        if source_key not in self._source_map:
            self._source_map[source_key] = []
        self._source_map[source_key].append(binding)

        logger.debug(
            f"[Binding] Added: {target_component}.{target_property} <- {source_expression}"
        )

        return binding

    def _normalize_source(self, expression: str) -> str:
        """
        Normalize source expression to a cache key.

        Handles:
        - "component.value" -> "component.value"
        - "component.text" -> "component.text"
        - "component" -> "component.value" (default to .value)
        """
        if '.' not in expression:
            return f"{expression}.value"
        return expression

    def _parse_expression(self, expression: str) -> tuple:
        """
        Parse a source expression into component and property.

        Returns:
            (component_name, property_name)
        """
        if '.' in expression:
            parts = expression.split('.', 1)
            return parts[0], parts[1]
        return expression, 'value'

    def _get_source_value(self, expression: str) -> Any:
        """
        Get the current value of a source expression.

        Args:
            expression: Source expression (e.g., "input.value")

        Returns:
            The current value, or None if not found
        """
        if not self.renderer:
            return None

        component_name, property_name = self._parse_expression(expression)
        component = self.renderer.get_component(component_name)

        if not component:
            return None

        # Get property value
        if property_name == 'value' and hasattr(component, 'value'):
            return component.value
        elif property_name == 'text' and hasattr(component, 'text'):
            return component.text
        elif property_name == 'visible':
            return component.visible
        elif property_name == 'enabled':
            return component.enabled
        elif property_name == 'checked' and hasattr(component, 'checked'):
            return component.checked
        elif hasattr(component, property_name):
            return getattr(component, property_name)

        return None

    def _set_target_value(self, component_name: str, property_name: str, value: Any):
        """
        Set a value on a target component.

        Updates both the component model and the widget if available.
        """
        if not self.renderer:
            return

        component = self.renderer.get_component(component_name)
        widget = self.renderer.get_widget(component_name)

        if not component:
            logger.warning(f"[Binding] Target component not found: {component_name}")
            return

        # Update component model
        if property_name == 'value' and hasattr(component, 'value'):
            component.value = value
        elif property_name == 'text':
            if hasattr(component, 'text'):
                component.text = value
            elif hasattr(component, 'value'):
                component.value = value
        elif property_name == 'visible':
            component.visible = bool(value)
        elif property_name == 'enabled':
            component.enabled = bool(value)
        elif hasattr(component, property_name):
            setattr(component, property_name, value)

        # Update widget
        if widget:
            if property_name == 'text' or property_name == 'value':
                if hasattr(widget, 'setText'):
                    widget.setText(str(value) if value is not None else '')
                elif hasattr(widget, 'setValue'):
                    widget.setValue(value)
            elif property_name == 'visible':
                if bool(value):
                    widget.show()
                else:
                    widget.hide()
            elif property_name == 'enabled':
                widget.setEnabled(bool(value))

    def notify_change(self, component_name: str, property_name: str = 'value'):
        """
        Notify that a source value has changed.

        Call this when a component's value changes to trigger binding updates.

        Args:
            component_name: Name of component that changed
            property_name: Property that changed (default: 'value')
        """
        source_key = f"{component_name}.{property_name}"

        if source_key not in self._source_map:
            return

        # Get current value
        new_value = self._get_source_value(source_key)

        # Check if actually changed
        old_value = self._value_cache.get(source_key)
        if new_value == old_value:
            return

        # Update cache
        self._value_cache[source_key] = new_value

        # Update all bindings that depend on this source
        for binding in self._source_map[source_key]:
            value = new_value

            # Apply transform if any
            if binding.transform:
                try:
                    value = binding.transform(value)
                except Exception as e:
                    logger.error(f"[Binding] Transform error: {e}")
                    continue

            # Set target value
            self._set_target_value(
                binding.target_component,
                binding.target_property,
                value
            )

            logger.debug(
                f"[Binding] Updated {binding.target_component}.{binding.target_property} = {value}"
            )

    def evaluate_all(self):
        """
        Evaluate all bindings and update targets.

        Call this after initial load or when re-syncing.
        """
        for binding in self._bindings:
            value = self._get_source_value(binding.source_expression)

            if binding.transform:
                try:
                    value = binding.transform(value)
                except Exception as e:
                    logger.error(f"[Binding] Transform error: {e}")
                    continue

            self._set_target_value(
                binding.target_component,
                binding.target_property,
                value
            )

    def clear(self):
        """Clear all bindings."""
        self._bindings.clear()
        self._source_map.clear()
        self._value_cache.clear()

    def get_bindings_for_component(self, component_name: str) -> List[Binding]:
        """Get all bindings that target a specific component."""
        return [b for b in self._bindings if b.target_component == component_name]

    def get_stats(self) -> Dict[str, Any]:
        """Get binding statistics."""
        return {
            'total_bindings': len(self._bindings),
            'source_expressions': len(self._source_map),
            'cached_values': len(self._value_cache)
        }


def parse_bindings_from_yaml(
    component_name: str,
    bindings_data: Dict[str, str],
    manager: BindingManager
) -> List[Binding]:
    """
    Parse bindings from YAML format and add to manager.

    YAML format:
        bindings:
          text: "input.value"
          visible: "checkbox.checked"

    Args:
        component_name: Target component name
        bindings_data: Dict of property -> source expression
        manager: Binding manager to add to

    Returns:
        List of created Binding objects
    """
    created = []

    for target_property, source_expression in bindings_data.items():
        binding = manager.add_binding(
            target_component=component_name,
            target_property=target_property,
            source_expression=source_expression
        )
        created.append(binding)

    return created


# Module test
if __name__ == "__main__":
    print("=== BindingManager Test ===\n")

    # Create mock renderer
    class MockComponent:
        def __init__(self, name, value=None, text=None):
            self.name = name
            self.value = value
            self.text = text
            self.visible = True
            self.enabled = True

    class MockWidget:
        def __init__(self):
            self.text = ""
            self.visible = True

        def setText(self, text):
            self.text = text
            print(f"    [Widget] setText('{text}')")

        def show(self):
            self.visible = True
            print("    [Widget] show()")

        def hide(self):
            self.visible = False
            print("    [Widget] hide()")

        def setEnabled(self, enabled):
            print(f"    [Widget] setEnabled({enabled})")

    class MockRenderer:
        def __init__(self):
            self._component_map = {}
            self._widget_map = {}

        def add_component(self, name, component, widget=None):
            self._component_map[name] = component
            if widget:
                self._widget_map[name] = widget

        def get_component(self, name):
            return self._component_map.get(name)

        def get_widget(self, name):
            return self._widget_map.get(name)

    # Set up test
    renderer = MockRenderer()
    renderer.add_component('input', MockComponent('input', value='Hello'), MockWidget())
    renderer.add_component('output', MockComponent('output', text=''), MockWidget())
    renderer.add_component('toggle', MockComponent('toggle', value=True), MockWidget())

    manager = BindingManager(renderer)

    # Test 1: Simple value binding
    print("Test 1: Value binding (input.value -> output.text)")
    manager.add_binding('output', 'text', 'input.value')
    manager.evaluate_all()
    print(f"  Output text: {renderer.get_component('output').text}")

    # Test 2: Update source
    print("\nTest 2: Source update")
    renderer.get_component('input').value = 'World'
    manager.notify_change('input', 'value')
    print(f"  Output text: {renderer.get_component('output').text}")

    # Test 3: Boolean binding
    print("\nTest 3: Boolean binding (toggle.value -> output.visible)")
    manager.add_binding('output', 'visible', 'toggle.value')
    manager.evaluate_all()

    renderer.get_component('toggle').value = False
    manager.notify_change('toggle', 'value')
    print(f"  Output visible: {renderer.get_component('output').visible}")

    # Test 4: Transform function
    print("\nTest 4: Transform binding")
    manager.add_binding(
        'output',
        'text',
        'input.value',
        transform=lambda v: f"[{v.upper()}]" if v else ""
    )
    renderer.get_component('input').value = 'test'
    manager.notify_change('input', 'value')

    print(f"\n=== Stats: {manager.get_stats()} ===")
