"""
UI Script Executor - Lightweight script execution for UI events

Executes JavaScript code in response to UI events (button clicks, etc.)
without the full ScriptedFacet overhead. Provides a UI-focused API.

Scripts have access to:
    - ui: Component value access (get, set, show, hide)
    - event: Event information (type, source)
    - app: NoodleApp instance (if available)
    - console: Logging functions

Author: Caitlyn + Claude
Date: January 3, 2026
"""

import time
import json
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path

# JavaScript execution via QuickJS
try:
    from quickjs import Context as QuickJSContext
    JS_AVAILABLE = True
except ImportError:
    JS_AVAILABLE = False
    QuickJSContext = None

if TYPE_CHECKING:
    from .renderer import QtWidgetRenderer

logger = logging.getLogger(__name__)


class UIScriptExecutor:
    """
    Lightweight JavaScript executor for UI event handlers.

    Unlike ScriptedFacet (which is designed for cognitive processing),
    this executor is optimized for UI interactions:
    - Simpler API focused on component manipulation
    - No async/await complexity
    - Direct widget access through the renderer

    Usage:
        executor = UIScriptExecutor(renderer)

        # Execute inline script
        executor.execute(
            script="ui.set('label', event.value);",
            event_type="onChange",
            source_component="input"
        )

        # Execute script file
        executor.execute_file(
            "scripts/on_submit.js",
            event_type="onSubmit",
            source_component="form"
        )
    """

    def __init__(self, renderer: 'QtWidgetRenderer', app: Optional[Any] = None):
        """
        Initialize UI script executor.

        Args:
            renderer: QtWidgetRenderer for component access
            app: Optional NoodleApp for deeper integration
        """
        self.renderer = renderer
        self.app = app

        # Script cache (file_path -> compiled script)
        self._script_cache: Dict[str, str] = {}

        # Execution stats
        self.execution_count = 0
        self.total_time = 0.0
        self.last_error: Optional[str] = None

        # Initialize JavaScript context
        if not JS_AVAILABLE:
            logger.warning("QuickJS not available - UI scripts will be disabled")
            self._js_context = None
        else:
            self._js_context = QuickJSContext()
            self._setup_context()

    def _setup_context(self):
        """Set up JavaScript context with UI helper functions."""
        if not self._js_context:
            return

        # Inject UI helper library
        helper_lib = """
        // =============================================
        // UI Script API - Available to all UI scripts
        // =============================================

        // Console/logging
        var __logs__ = [];
        var console = {
            log: function() {
                var args = Array.prototype.slice.call(arguments);
                __logs__.push({level: 'log', message: args.map(String).join(' ')});
            },
            warn: function() {
                var args = Array.prototype.slice.call(arguments);
                __logs__.push({level: 'warn', message: args.map(String).join(' ')});
            },
            error: function() {
                var args = Array.prototype.slice.call(arguments);
                __logs__.push({level: 'error', message: args.map(String).join(' ')});
            }
        };

        // Component values (populated by Python before script runs)
        var __component_values__ = {};
        var __component_visibility__ = {};
        var __component_enabled__ = {};

        // Value changes to apply after script runs
        var __value_changes__ = {};
        var __visibility_changes__ = {};
        var __enabled_changes__ = {};

        // UI API object
        var ui = {
            // Get component value
            get: function(componentName) {
                return __component_values__[componentName];
            },

            // Set component value
            set: function(componentName, value) {
                __value_changes__[componentName] = value;
            },

            // Check if component is visible
            isVisible: function(componentName) {
                return __component_visibility__[componentName] !== false;
            },

            // Show component
            show: function(componentName) {
                __visibility_changes__[componentName] = true;
            },

            // Hide component
            hide: function(componentName) {
                __visibility_changes__[componentName] = false;
            },

            // Toggle component visibility
            toggle: function(componentName) {
                var current = __component_visibility__[componentName] !== false;
                __visibility_changes__[componentName] = !current;
            },

            // Check if component is enabled
            isEnabled: function(componentName) {
                return __component_enabled__[componentName] !== false;
            },

            // Enable component
            enable: function(componentName) {
                __enabled_changes__[componentName] = true;
            },

            // Disable component
            disable: function(componentName) {
                __enabled_changes__[componentName] = false;
            },

            // Get all component names
            getComponents: function() {
                return Object.keys(__component_values__);
            }
        };

        // Event object (populated by Python)
        var event = {
            type: '',
            source: '',
            value: null,
            timestamp: 0
        };

        // Clear state between executions
        function __clear_state__() {
            __logs__ = [];
            __value_changes__ = {};
            __visibility_changes__ = {};
            __enabled_changes__ = {};
        }

        // Collect results after execution
        function __get_results__() {
            return JSON.stringify({
                logs: __logs__,
                value_changes: __value_changes__,
                visibility_changes: __visibility_changes__,
                enabled_changes: __enabled_changes__
            });
        }
        """

        self._js_context.eval(helper_lib)

    def execute(
        self,
        script: str,
        event_type: str = "onClick",
        source_component: str = "",
        event_value: Any = None
    ) -> Dict[str, Any]:
        """
        Execute a UI script.

        Args:
            script: JavaScript code to execute
            event_type: Type of event that triggered this (onClick, onChange, etc.)
            source_component: Name of component that triggered the event
            event_value: Value associated with the event (e.g., input value)

        Returns:
            Dict with execution results:
                - success: Whether execution succeeded
                - logs: Console output from script
                - error: Error message if failed
        """
        if not self._js_context:
            return {
                'success': False,
                'error': 'JavaScript not available',
                'logs': []
            }

        start_time = time.time()

        try:
            # Clear previous state
            self._js_context.eval("__clear_state__();")

            # Populate component values
            component_values = {}
            component_visibility = {}
            component_enabled = {}

            for name, component in self.renderer._component_map.items():
                # Get value if component has one
                if hasattr(component, 'value'):
                    component_values[name] = component.value
                elif hasattr(component, 'text'):
                    component_values[name] = component.text
                else:
                    component_values[name] = None

                component_visibility[name] = component.visible
                component_enabled[name] = component.enabled

            # Inject component state
            self._js_context.eval(
                f"__component_values__ = {json.dumps(component_values)};"
            )
            self._js_context.eval(
                f"__component_visibility__ = {json.dumps(component_visibility)};"
            )
            self._js_context.eval(
                f"__component_enabled__ = {json.dumps(component_enabled)};"
            )

            # Set up event object
            event_data = {
                'type': event_type,
                'source': source_component,
                'value': event_value,
                'timestamp': start_time
            }
            self._js_context.eval(f"event = {json.dumps(event_data)};")

            # Execute the script
            self._js_context.eval(script)

            # Collect results
            results_json = self._js_context.eval("__get_results__();")
            results = json.loads(results_json)

            # Apply changes
            self._apply_changes(results)

            # Log any console output
            for log_entry in results.get('logs', []):
                level = log_entry.get('level', 'log')
                message = log_entry.get('message', '')
                if level == 'error':
                    logger.error(f"[UIScript] {message}")
                elif level == 'warn':
                    logger.warning(f"[UIScript] {message}")
                else:
                    logger.debug(f"[UIScript] {message}")

            # Update stats
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_time += elapsed
            self.last_error = None

            return {
                'success': True,
                'logs': results.get('logs', []),
                'error': None
            }

        except Exception as e:
            elapsed = time.time() - start_time
            self.last_error = str(e)
            logger.error(f"[UIScript] Execution error: {e}")

            return {
                'success': False,
                'logs': [],
                'error': str(e)
            }

    def _apply_changes(self, results: Dict[str, Any]):
        """Apply component changes from script execution."""
        # Apply value changes
        for name, value in results.get('value_changes', {}).items():
            component = self.renderer.get_component(name)
            widget = self.renderer.get_widget(name)

            if component:
                if hasattr(component, 'value'):
                    component.value = value
                elif hasattr(component, 'text'):
                    component.text = value

            if widget:
                if hasattr(widget, 'setText'):
                    widget.setText(str(value))
                elif hasattr(widget, 'setValue'):
                    widget.setValue(value)

        # Apply visibility changes
        for name, visible in results.get('visibility_changes', {}).items():
            component = self.renderer.get_component(name)
            widget = self.renderer.get_widget(name)

            if component:
                component.visible = visible

            if widget:
                if visible:
                    widget.show()
                else:
                    widget.hide()

        # Apply enabled changes
        for name, enabled in results.get('enabled_changes', {}).items():
            component = self.renderer.get_component(name)
            widget = self.renderer.get_widget(name)

            if component:
                component.enabled = enabled

            if widget:
                widget.setEnabled(enabled)

    def execute_file(
        self,
        file_path: str,
        event_type: str = "onClick",
        source_component: str = "",
        event_value: Any = None,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a script from a file.

        Args:
            file_path: Path to JavaScript file (relative to project or absolute)
            event_type: Event type that triggered this
            source_component: Component that triggered the event
            event_value: Value associated with event
            project_path: Base project path for resolving relative paths

        Returns:
            Execution results (same as execute())
        """
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute() and project_path:
            path = Path(project_path) / file_path

        # Check cache
        cache_key = str(path.resolve())
        if cache_key not in self._script_cache:
            try:
                with open(path, 'r') as f:
                    self._script_cache[cache_key] = f.read()
            except FileNotFoundError:
                return {
                    'success': False,
                    'error': f'Script file not found: {file_path}',
                    'logs': []
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Failed to read script: {e}',
                    'logs': []
                }

        return self.execute(
            script=self._script_cache[cache_key],
            event_type=event_type,
            source_component=source_component,
            event_value=event_value
        )

    def clear_cache(self):
        """Clear the script file cache."""
        self._script_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.execution_count if self.execution_count > 0 else 0,
            'last_error': self.last_error,
            'cached_scripts': len(self._script_cache),
            'js_available': JS_AVAILABLE
        }


# Module test
if __name__ == "__main__":
    print("=== UIScriptExecutor Test ===\n")

    if not JS_AVAILABLE:
        print("QuickJS not available - cannot run tests")
        exit(1)

    # Create mock renderer
    class MockComponent:
        def __init__(self, name, value=None):
            self.name = name
            self.value = value
            self.visible = True
            self.enabled = True

    class MockWidget:
        def __init__(self):
            self.text = ""
            self.visible = True
            self.enabled = True

        def setText(self, text):
            self.text = text

        def show(self):
            self.visible = True

        def hide(self):
            self.visible = False

        def setEnabled(self, enabled):
            self.enabled = enabled

    class MockRenderer:
        def __init__(self):
            self._component_map = {
                'input': MockComponent('input', 'Hello'),
                'output': MockComponent('output', ''),
                'button': MockComponent('button'),
            }
            self._widget_map = {
                'input': MockWidget(),
                'output': MockWidget(),
                'button': MockWidget(),
            }

        def get_component(self, name):
            return self._component_map.get(name)

        def get_widget(self, name):
            return self._widget_map.get(name)

    renderer = MockRenderer()
    executor = UIScriptExecutor(renderer)

    # Test 1: Simple value transfer
    print("Test 1: Value transfer")
    result = executor.execute(
        script="""
        var input = ui.get('input');
        ui.set('output', 'You said: ' + input);
        console.log('Transferred value:', input);
        """,
        event_type="onClick",
        source_component="button"
    )
    print(f"  Success: {result['success']}")
    print(f"  Output value: {renderer.get_component('output').value}")
    print(f"  Logs: {result['logs']}")

    # Test 2: Visibility toggle
    print("\nTest 2: Visibility toggle")
    result = executor.execute(
        script="""
        ui.toggle('output');
        console.log('Toggled output visibility');
        """,
        event_type="onClick"
    )
    print(f"  Success: {result['success']}")
    print(f"  Output visible: {renderer.get_component('output').visible}")

    # Test 3: Using event object
    print("\nTest 3: Event object")
    result = executor.execute(
        script="""
        console.log('Event type:', event.type);
        console.log('Source:', event.source);
        console.log('Value:', event.value);
        ui.set('output', 'Changed to: ' + event.value);
        """,
        event_type="onChange",
        source_component="input",
        event_value="New value!"
    )
    print(f"  Success: {result['success']}")
    print(f"  Output: {renderer.get_component('output').value}")
    print(f"  Logs: {[l['message'] for l in result['logs']]}")

    # Test 4: Error handling
    print("\nTest 4: Error handling")
    result = executor.execute(
        script="""
        undefined_function();
        """,
        event_type="onClick"
    )
    print(f"  Success: {result['success']}")
    print(f"  Error: {result['error']}")

    print(f"\n=== Stats: {executor.get_stats()} ===")
