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
#   UI Event Dispatcher
#
#   Routes UI events to noodlings, scripts, assemblies, and h...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.event_dispatcher
# PURPOSE:  UI Event Dispatcher
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   UIEventDispatcher
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import asyncio
import logging
import threading

if TYPE_CHECKING:
    from .component import UIComponent, EventBinding
    from .event_data import UIEventData
    from .renderer import QtWidgetRenderer
    from .components.chat_history import ChatHistory, MessageRole
    from .script_executor import UIScriptExecutor

logger = logging.getLogger(__name__)


class UIEventDispatcher:
    """
    Dispatches UI events to noodlings, scripts, assemblies, and external handlers.

    Supported actions:
        - send_to_noodling: Send message to a noodling and get response
        - call_script: Execute inline JavaScript or script file
        - set_value: Set a component's value
        - show/hide/toggle_visible: Control component visibility
        - run_assembly: Execute a facet assembly one-shot
        - custom: Call a custom handler function

    Usage:
        dispatcher = UIEventDispatcher(renderer, app)
        renderer.set_event_dispatcher(dispatcher.dispatch)
    """

    def __init__(self, renderer: 'QtWidgetRenderer', app: Optional[Any] = None):
        """
        Initialize dispatcher.

        Args:
            renderer: The QtWidgetRenderer instance
            app: Optional NoodleApp instance for noodling communication
        """
        self.renderer = renderer
        self.app = app

        # Custom event handlers: action_name -> handler_func
        self._custom_handlers: Dict[str, Callable] = {}

        # Default chat history component name (for automatic response display)
        self.default_chat_history: str = "chat_history"

        # Script executor (lazy-initialized)
        self._script_executor: Optional['UIScriptExecutor'] = None

        # Project path for resolving script files
        self.project_path: Optional[str] = None

        # Facet executor for run_assembly action
        self._facet_executor = None

        # Assembly cache (path -> FacetAssembly)
        self._assembly_cache: Dict[str, Any] = {}

        # Root UI component (for FacetAssembly lookups)
        self.root_component: Optional['UIComponent'] = None

    def _get_script_executor(self) -> 'UIScriptExecutor':
        """Get or create the script executor."""
        if self._script_executor is None:
            from .script_executor import UIScriptExecutor
            self._script_executor = UIScriptExecutor(self.renderer, self.app)
        return self._script_executor

    def set_app(self, app: Any) -> None:
        """Set the NoodleApp instance for noodling communication."""
        self.app = app
        # Update script executor if it exists
        if self._script_executor:
            self._script_executor.app = app

    def set_project_path(self, path: str) -> None:
        """Set project path for resolving script files."""
        self.project_path = path

    def set_facet_executor(self, executor) -> None:
        """Set the FacetExecutor instance for run_assembly action."""
        self._facet_executor = executor

    def register_handler(self, action: str, handler: Callable) -> None:
        """
        Register a custom event handler.

        Args:
            action: Action name to handle
            handler: Callable(component, binding, value) -> None
        """
        self._custom_handlers[action] = handler

    def dispatch(
        self,
        event_name: str,
        component: 'UIComponent',
        binding: 'EventBinding',
        event_data: Optional['UIEventData'] = None
    ) -> None:
        """
        Dispatch a UI event based on its binding.

        Args:
            event_name: Name of the event (onClick, onSubmit, etc.)
            component: The component that triggered the event
            binding: Event binding with action and parameters
            event_data: Rich event metadata (mouse position, keys, etc.)
        """
        # Create basic event data if none provided (backward compatibility)
        if event_data is None:
            from .event_data import UIEventData
            event_value = None
            if hasattr(component, 'value'):
                event_value = component.value
            elif hasattr(component, 'text'):
                event_value = component.text
            event_data = UIEventData(
                type=event_name,
                source=component.name,
                value=event_value,
            )

        action = binding.action

        logger.debug(f"Dispatching event: {event_name} -> {action} (target: {binding.target})")

        if action == "send_to_noodling":
            self._handle_send_to_noodling(component, binding)
        elif action == "call_script":
            self._handle_call_script(component, binding, event_data)
        elif action == "set_value":
            self._handle_set_value(binding)
        elif action == "show":
            self._handle_show(binding)
        elif action == "hide":
            self._handle_hide(binding)
        elif action == "toggle_visible":
            self._handle_toggle_visible(binding)
        elif action == "run_assembly":
            self._handle_run_assembly(component, binding, event_data)
        elif action in self._custom_handlers:
            self._custom_handlers[action](component, binding)
        else:
            logger.warning(f"Unknown action: {action}")

    def _handle_send_to_noodling(self, component: 'UIComponent', binding: 'EventBinding') -> None:
        """
        Send a message to a noodling.

        Binding parameters:
            target: Noodling name/ID to send to
            message_source: Component name containing message (or "self" for current component)
            chat_history: Component name for chat history display (optional)
        """
        from .components.text_input import TextInput
        from .components.chat_input import ChatInput
        from .components.chat_history import ChatHistory, MessageRole

        # Get message text
        message_source = binding.message_source or "self"

        if message_source == "self":
            # Get value from the triggering component
            if isinstance(component, (TextInput, ChatInput)):
                message = component.value
            else:
                logger.warning(f"Cannot get message from component type: {type(component)}")
                return
        else:
            # Get value from named component
            source_component = self.renderer.get_component(message_source)
            if source_component and hasattr(source_component, 'value'):
                message = source_component.value
            else:
                logger.warning(f"Message source component not found: {message_source}")
                return

        if not message or not message.strip():
            return

        message = message.strip()

        # Get target noodling
        target = binding.target
        if not target:
            logger.warning("No target noodling specified")
            return

        # Get chat history component (if any)
        chat_history_name = binding.chat_history or self.default_chat_history
        chat_history = self.renderer.get_component(chat_history_name)

        # Add user message to chat history
        if chat_history and isinstance(chat_history, ChatHistory):
            chat_history.add_message(
                role=MessageRole.USER,
                content=message,
                sender_name="You"
            )

        # Send to noodling asynchronously
        if self.app:
            asyncio.create_task(self._send_and_receive(target, message, chat_history))
        else:
            logger.warning("No app instance for noodling communication")
            # In demo mode, echo the message
            if chat_history and isinstance(chat_history, ChatHistory):
                chat_history.add_message(
                    role=MessageRole.NOODLING,
                    content=f"[Demo mode] Received: {message}",
                    sender_name=target.title()
                )

    async def _send_and_receive(self, target: str, message: str, chat_history: Optional['ChatHistory']) -> None:
        """Send message to noodling and display response."""
        from .components.chat_history import ChatHistory, MessageRole

        try:
            # Call the app's run method to process through the noodling
            result = await self.app.run(message, noodling=target)

            response = result.get('response', '')

            # Add noodling response to chat history
            if chat_history and isinstance(chat_history, ChatHistory):
                chat_history.add_message(
                    role=MessageRole.NOODLING,
                    content=response,
                    sender_name=target.title()
                )
        except Exception as e:
            logger.error(f"Error sending to noodling: {e}")
            if chat_history and isinstance(chat_history, ChatHistory):
                chat_history.add_message(
                    role=MessageRole.SYSTEM,
                    content=f"Error: {str(e)}"
                )

    def _handle_call_script(
        self,
        component: 'UIComponent',
        binding: 'EventBinding',
        event_data: 'UIEventData'
    ) -> None:
        """
        Execute a script in response to a UI event.

        Binding parameters:
            script: Inline JavaScript code
            script_file: Path to JavaScript file (relative to project)

        Scripts have access to:
            - ui: Component value access (get, set, show, hide)
            - event: Rich event data (type, source, value, x, y, key, modifiers, etc.)
            - console: Logging functions
        """
        executor = self._get_script_executor()

        # Execute inline script or file
        if binding.script:
            result = executor.execute(
                script=binding.script,
                event_data=event_data
            )
        elif binding.script_file:
            result = executor.execute_file(
                file_path=binding.script_file,
                event_data=event_data,
                project_path=self.project_path
            )
        else:
            logger.warning("call_script action requires 'script' or 'script_file'")
            return

        if not result.get('success'):
            logger.error(f"Script execution failed: {result.get('error')}")

    def _handle_set_value(self, binding: 'EventBinding') -> None:
        """Set a component's value."""
        target = binding.target
        value = getattr(binding, 'value', '')

        component = self.renderer.get_component(target)
        if component and hasattr(component, 'value'):
            component.value = value

            # Update widget if rendered
            widget = self.renderer.get_widget(target)
            if widget and hasattr(widget, 'setText'):
                widget.setText(value)

    def _handle_show(self, binding: 'EventBinding') -> None:
        """Show a component."""
        target = binding.target
        widget = self.renderer.get_widget(target)
        if widget:
            widget.show()

        component = self.renderer.get_component(target)
        if component:
            component.visible = True

    def _handle_hide(self, binding: 'EventBinding') -> None:
        """Hide a component."""
        target = binding.target
        widget = self.renderer.get_widget(target)
        if widget:
            widget.hide()

        component = self.renderer.get_component(target)
        if component:
            component.visible = False

    def _handle_toggle_visible(self, binding: 'EventBinding') -> None:
        """Toggle component visibility."""
        target = binding.target
        widget = self.renderer.get_widget(target)
        component = self.renderer.get_component(target)

        if widget and component:
            if component.visible:
                widget.hide()
                component.visible = False
            else:
                widget.show()
                component.visible = True

    def _handle_run_assembly(
        self,
        component: 'UIComponent',
        binding: 'EventBinding',
        event_data: 'UIEventData'
    ) -> None:
        """
        Run a facet assembly one-shot.

        Binding parameters:
            target: Name of FacetAssembly component (uses its configured bindings)
            assembly: Path to assembly YAML file (relative to project)
            inputs: Dict mapping input pad names to values or component references
            outputs: Dict mapping output pad names to target component properties

        Input binding syntax:
            - Static value: {"text": "Hello world"}
            - Component reference: {"text": "{input_field.value}"}
            - Event value: {"text": "{event.value}"}

        Output binding syntax:
            - {"result": "result_label.text"}
            - {"sentiment": "mood_indicator.color"}

        Example binding (inline):
            onClick:
              action: run_assembly
              assembly: assemblies/sentiment-analysis.yaml
              inputs:
                text: "{text_field.value}"
              outputs:
                result: result_label.text

        Example binding (FacetAssembly component):
            onClick:
              action: run_assembly
              target: guide_assembly  # References FacetAssembly component
        """
        from pathlib import Path

        # Check for FacetAssembly component reference
        target_name = getattr(binding, 'target', None)
        if target_name and self.root_component:
            facet_component = self._find_facet_assembly_component(target_name)
            if facet_component:
                # Use FacetAssembly component's configuration
                assembly_path = facet_component.assembly_path
                # Convert input/output bindings to dicts
                input_bindings = {b.pad_name: "{" + b.source + "}" for b in facet_component.input_bindings}
                output_bindings = facet_component.get_output_targets()
                logger.info(f"Using FacetAssembly component '{target_name}': {assembly_path}")
            else:
                logger.warning(f"FacetAssembly component not found: {target_name}")
                return
        else:
            # Get inline assembly path
            assembly_path = getattr(binding, 'assembly', None)
            if not assembly_path:
                logger.warning("run_assembly action requires 'assembly' or 'target' parameter")
                return
            input_bindings = None  # Will use binding.inputs
            output_bindings = None  # Will use binding.outputs

        # Resolve relative path
        if self.project_path and not Path(assembly_path).is_absolute():
            assembly_path = str(Path(self.project_path) / assembly_path)

        # Check executor
        if not self._facet_executor:
            logger.warning("No facet executor available for run_assembly action")
            return

        # Load assembly (with caching)
        assembly = self._get_or_load_assembly(assembly_path)
        if not assembly:
            logger.error(f"Failed to load assembly: {assembly_path}")
            return

        # Resolve input bindings (use FacetAssembly bindings if available)
        if input_bindings is not None:
            # FacetAssembly component - resolve its bindings
            inputs = self._resolve_facet_assembly_inputs(input_bindings, component, event_data)
        else:
            # Inline binding config
            inputs = self._resolve_assembly_inputs(binding, component, event_data)

        # Get output bindings (use FacetAssembly bindings if available)
        if output_bindings is None:
            output_bindings = getattr(binding, 'outputs', {})

        # UX feedback: show "Thinking..." and clear input immediately
        thinking_target = getattr(binding, 'thinking_target', None)
        clear_input = getattr(binding, 'clear_input', False)
        logger.info(f"[run_assembly] thinking_target={thinking_target}, clear_input={clear_input}")

        if thinking_target:
            target_widget = self.renderer.get_widget(thinking_target)
            if target_widget and hasattr(target_widget, 'setText'):
                target_widget.setText("Thinking...")

        if clear_input and component:
            source_widget = self.renderer.get_widget(component.name)
            if source_widget and hasattr(source_widget, 'clear'):
                source_widget.clear()
            elif source_widget and hasattr(source_widget, 'setText'):
                source_widget.setText("")

        # Execute asynchronously in a thread
        # Note: Qt's event loop doesn't process asyncio tasks, so we always use threads
        import traceback
        logger.info(f"[run_assembly] Using thread for: {assembly.name}")

        def run_in_thread():
            print(f"[run_assembly] Thread started for: {assembly.name}", flush=True)
            logger.info(f"[run_assembly] Thread starting for assembly: {assembly.name}")
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)
            try:
                thread_loop.run_until_complete(
                    self._run_assembly_and_bind(
                        assembly, inputs, output_bindings, component,
                        from_worker_thread=True
                    )
                )
                print(f"[run_assembly] Thread completed for: {assembly.name}", flush=True)
                logger.info(f"[run_assembly] Thread completed for assembly: {assembly.name}")
            except Exception as e:
                print(f"[run_assembly] Thread CRASHED: {e}", flush=True)
                traceback.print_exc()
                logger.error(f"[run_assembly] Thread error: {e}", exc_info=True)
            finally:
                thread_loop.close()

        logger.info(f"[run_assembly] Starting thread for: {assembly.name}")
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        logger.info(f"[run_assembly] Thread.start() called")

    def _find_facet_assembly_component(self, name: str):
        """Find a FacetAssembly component by name in the UI tree."""
        from .components.facet_assembly import FacetAssembly

        def find_recursive(component):
            if component.name == name and isinstance(component, FacetAssembly):
                return component
            for child in component.children:
                result = find_recursive(child)
                if result:
                    return result
            return None

        return find_recursive(self.root_component) if self.root_component else None

    def _resolve_facet_assembly_inputs(
        self,
        input_bindings: dict,
        component: 'UIComponent',
        event_data: 'UIEventData'
    ) -> dict:
        """
        Resolve FacetAssembly input bindings to actual values.

        Args:
            input_bindings: Dict of pad_name -> source string (e.g., "{input.value}")
            component: Source component that triggered the event
            event_data: Event data
        """
        resolved = {}
        for pad_name, source in input_bindings.items():
            resolved[pad_name] = self._resolve_reference(source, component, event_data)
        return resolved

    def _resolve_reference(
        self,
        value: str,
        component: 'UIComponent',
        event_data: 'UIEventData'
    ) -> Any:
        """Resolve a single reference value like {input.value} or {event.value}."""
        if not isinstance(value, str) or not value.startswith('{') or not value.endswith('}'):
            return value

        ref = value[1:-1]  # Remove braces

        if ref.startswith('event.'):
            # Event reference
            attr = ref[6:]  # Remove "event."
            return getattr(event_data, attr, None)
        elif '.' in ref:
            # Component.property reference
            comp_name, prop_name = ref.split('.', 1)
            widget = self.renderer.get_widget(comp_name) if self.renderer else None
            if widget:
                if prop_name == 'value':
                    if hasattr(widget, 'text'):
                        return widget.text()
                    elif hasattr(widget, 'value'):
                        return widget.value()
                elif prop_name == 'text':
                    if hasattr(widget, 'text'):
                        return widget.text()
                elif prop_name == 'checked':
                    if hasattr(widget, 'isChecked'):
                        return widget.isChecked()
                else:
                    return widget.property(prop_name)
        return value

    def _get_or_load_assembly(self, assembly_path: str):
        """Load assembly from cache or disk."""
        if assembly_path in self._assembly_cache:
            return self._assembly_cache[assembly_path]

        try:
            from ...core.facet_system import FacetAssembly
            assembly = FacetAssembly.load_yaml(assembly_path)
            self._assembly_cache[assembly_path] = assembly
            logger.info(f"Loaded assembly for UI: {assembly.name}")
            return assembly
        except Exception as e:
            logger.error(f"Failed to load assembly {assembly_path}: {e}")
            return None

    def _resolve_assembly_inputs(
        self,
        binding: 'EventBinding',
        component: 'UIComponent',
        event_data: 'UIEventData'
    ) -> Any:
        """
        Resolve input bindings to actual values.

        Supports:
            - Static values: "Hello"
            - Component references: "{input_field.value}"
            - Event references: "{event.value}"
        """
        inputs = getattr(binding, 'inputs', {})
        resolved = {}

        for pad_name, value in inputs.items():
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                # Resolve reference
                ref = value[1:-1]  # Remove braces

                if ref.startswith('event.'):
                    # Event reference
                    attr = ref[6:]  # Remove "event."
                    resolved[pad_name] = getattr(event_data, attr, None)
                else:
                    # Component reference: "component_name.property"
                    parts = ref.split('.', 1)
                    if len(parts) == 2:
                        comp_name, prop_name = parts
                        comp = self.renderer.get_component(comp_name)
                        if comp:
                            resolved[pad_name] = getattr(comp, prop_name, None)
                        else:
                            logger.warning(f"Component not found: {comp_name}")
                            resolved[pad_name] = None
                    else:
                        resolved[pad_name] = value
            else:
                # Static value
                resolved[pad_name] = value

        # If only one input, return it directly (for INCOMING node)
        if len(resolved) == 1:
            return list(resolved.values())[0]
        return resolved

    async def _run_assembly_and_bind(
        self,
        assembly,
        inputs: Any,
        output_bindings: Dict[str, str],
        source_component: 'UIComponent',
        from_worker_thread: bool = False
    ) -> None:
        """
        Execute assembly and bind outputs to components.

        Args:
            assembly: Loaded FacetAssembly
            inputs: Input values for INCOMING node
            output_bindings: Map of output pad -> "component.property"
            source_component: The component that triggered the event
            from_worker_thread: If True, marshal widget updates to main thread
        """
        try:
            # Build execution context
            context = {
                'source_component': source_component.name,
                'ui_event': True
            }

            # Get Brenda's direction if available and inject into context
            # This allows the facet assembly to incorporate stage direction
            user_input_text = None
            if self.app:
                # Get the direction text from Brenda via GuideCueHandler
                brenda_direction = self.app.get_brenda_direction()
                if brenda_direction:
                    context['brenda_direction'] = brenda_direction
                    logger.debug("[UIEventDispatcher] Injected Brenda direction into context")

                # Capture user input for publishing and reporting
                if isinstance(inputs, str):
                    user_input_text = inputs
                elif isinstance(inputs, dict) and 'text' in inputs:
                    user_input_text = inputs['text']

                # Publish user input to #user.input for Brenda and other listeners
                if user_input_text:
                    self.app.publish_user_input(user_input_text)

            # Execute assembly
            result = await self._facet_executor.execute(
                assembly,
                inputs,
                context=context
            )

            # Apply output bindings
            for pad_name, target in output_bindings.items():
                # Find output value
                output_value = None

                # Check OUTGOING outputs first
                for facet_id, outputs in result.facet_outputs.items():
                    if pad_name in outputs:
                        output_value = outputs[pad_name]
                        break

                # Fallback to response
                if output_value is None and pad_name == 'result':
                    output_value = result.response

                if output_value is None:
                    continue

                # Parse target: "component_name.property"
                parts = target.split('.', 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid output binding format: {target}")
                    continue

                comp_name, prop_name = parts
                target_comp = self.renderer.get_component(comp_name)
                target_widget = self.renderer.get_widget(comp_name)

                if target_comp:
                    # Set component property
                    if hasattr(target_comp, prop_name):
                        setattr(target_comp, prop_name, output_value)

                    # Update widget - must be on main thread for Qt
                    if target_widget:
                        if from_worker_thread:
                            # Marshal to main thread via QMetaObject
                            self._update_widget_threadsafe(
                                target_widget, prop_name, output_value
                            )
                        else:
                            # Direct update (already on main thread)
                            if prop_name == 'text' and hasattr(target_widget, 'setText'):
                                target_widget.setText(str(output_value))
                            elif prop_name == 'value' and hasattr(target_widget, 'setValue'):
                                target_widget.setValue(output_value)

            logger.debug(f"Assembly {assembly.name} completed, outputs bound")

            # Report response back to Brenda for play advancement
            if self.app and result.response:
                self.app.report_actor_response(result.response, user_input_text)
                logger.debug("[UIEventDispatcher] Reported response to Brenda")

        except Exception as e:
            logger.error(f"Assembly execution error: {e}")

    def _update_widget_threadsafe(self, widget, prop_name: str, value: Any) -> None:
        """
        Update a Qt widget from a worker thread.

        Uses QMetaObject.invokeMethod with Qt.QueuedConnection to marshal
        the update to the main thread.
        """
        try:
            from PyQt6.QtCore import QMetaObject, Qt, Q_ARG

            if prop_name == 'text' and hasattr(widget, 'setText'):
                QMetaObject.invokeMethod(
                    widget, "setText",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, str(value))
                )
            elif prop_name == 'value' and hasattr(widget, 'setValue'):
                # Note: setValue signature varies by widget type
                # For QLabel text, use setText
                if hasattr(widget, 'setText'):
                    QMetaObject.invokeMethod(
                        widget, "setText",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, str(value))
                    )
        except Exception as e:
            logger.warning(f"Thread-safe widget update failed: {e}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
