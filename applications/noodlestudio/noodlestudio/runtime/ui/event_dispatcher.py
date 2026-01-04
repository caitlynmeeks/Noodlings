"""
UI Event Dispatcher

Routes UI events to noodlings, scripts, and handles responses.
"""

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import asyncio
import logging

if TYPE_CHECKING:
    from .component import UIComponent, EventBinding
    from .renderer import QtWidgetRenderer
    from .components.chat_history import ChatHistory, MessageRole
    from .script_executor import UIScriptExecutor

logger = logging.getLogger(__name__)


class UIEventDispatcher:
    """
    Dispatches UI events to noodlings, scripts, and external handlers.

    Supported actions:
        - send_to_noodling: Send message to a noodling and get response
        - call_script: Execute inline JavaScript or script file
        - set_value: Set a component's value
        - show/hide/toggle_visible: Control component visibility
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

    def register_handler(self, action: str, handler: Callable) -> None:
        """
        Register a custom event handler.

        Args:
            action: Action name to handle
            handler: Callable(component, binding, value) -> None
        """
        self._custom_handlers[action] = handler

    def dispatch(self, event_name: str, component: 'UIComponent', binding: 'EventBinding') -> None:
        """
        Dispatch a UI event based on its binding.

        Args:
            event_name: Name of the event (onClick, onSubmit, etc.)
            component: The component that triggered the event
            binding: Event binding with action and parameters
        """
        action = binding.action

        logger.debug(f"Dispatching event: {event_name} -> {action} (target: {binding.target})")

        if action == "send_to_noodling":
            self._handle_send_to_noodling(component, binding)
        elif action == "call_script":
            self._handle_call_script(component, binding, event_name)
        elif action == "set_value":
            self._handle_set_value(binding)
        elif action == "show":
            self._handle_show(binding)
        elif action == "hide":
            self._handle_hide(binding)
        elif action == "toggle_visible":
            self._handle_toggle_visible(binding)
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
        event_name: str
    ) -> None:
        """
        Execute a script in response to a UI event.

        Binding parameters:
            script: Inline JavaScript code
            script_file: Path to JavaScript file (relative to project)

        Scripts have access to:
            - ui: Component value access (get, set, show, hide)
            - event: Event info (type, source, value)
            - console: Logging functions
        """
        executor = self._get_script_executor()

        # Get component value for the event
        event_value = None
        if hasattr(component, 'value'):
            event_value = component.value
        elif hasattr(component, 'text'):
            event_value = component.text

        # Execute inline script or file
        if binding.script:
            result = executor.execute(
                script=binding.script,
                event_type=event_name,
                source_component=component.name,
                event_value=event_value
            )
        elif binding.script_file:
            result = executor.execute_file(
                file_path=binding.script_file,
                event_type=event_name,
                source_component=component.name,
                event_value=event_value,
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
