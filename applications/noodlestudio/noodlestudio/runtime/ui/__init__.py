"""
NoodleStudio UI Canvas System

Delphi-style visual UI components for building application interfaces.

Architecture:
    ui.yaml (user's design - stable contract)
        ↓
    UIComponent classes (this module)
        ↓
    Renderer backend (QtWidgetRenderer for v1)

Usage:
    from noodlestudio.runtime.ui import load_ui, QtWidgetRenderer

    # Load UI definition
    root = load_ui("path/to/ui.yaml")

    # Render to Qt widgets
    renderer = QtWidgetRenderer()
    widget = renderer.render(root)
    widget.show()
"""

# Core classes
from .component import (
    UIComponent,
    Anchors,
    Geometry,
    EventBinding,
    register_component,
    get_component_class,
    list_component_types,
)

# Loader
from .loader import (
    UILoader,
    load_ui,
    create_default_ui,
    create_default_ui_yaml,
)

# Renderer
from .renderer import (
    QtWidgetRenderer,
    AnchoredWidget,
    ChatHistoryWidget,
    ChatInputWidget,
)

# Event data
from .event_data import (
    UIEventData,
    Modifiers,
    MouseButton,
    # Event type constants
    EVENT_CLICK,
    EVENT_DOUBLE_CLICK,
    EVENT_MOUSE_DOWN,
    EVENT_MOUSE_UP,
    EVENT_MOUSE_MOVE,
    EVENT_MOUSE_ENTER,
    EVENT_MOUSE_LEAVE,
    EVENT_MOUSE_WHEEL,
    EVENT_CONTEXT_MENU,
    EVENT_KEY_DOWN,
    EVENT_KEY_UP,
    EVENT_KEY_PRESS,
    EVENT_FOCUS,
    EVENT_BLUR,
    EVENT_CHANGE,
    EVENT_SUBMIT,
    ALL_EVENT_TYPES,
)

# Event dispatcher
from .event_dispatcher import UIEventDispatcher

# Script executor
from .script_executor import UIScriptExecutor

# Bindings
from .bindings import Binding, BindingManager, parse_bindings_from_yaml

# Components (importing ensures registration)
from .components import (
    Panel,
    Label,
    Button,
    TextInput,
    RadianceViewport,
    ChatHistory,
    ChatMessage,
    MessageRole,
    ChatInput,
)

__all__ = [
    # Core
    'UIComponent',
    'Anchors',
    'Geometry',
    'EventBinding',
    'register_component',
    'get_component_class',
    'list_component_types',
    # Loader
    'UILoader',
    'load_ui',
    'create_default_ui',
    'create_default_ui_yaml',
    # Renderer
    'QtWidgetRenderer',
    'AnchoredWidget',
    'ChatHistoryWidget',
    'ChatInputWidget',
    # Event data
    'UIEventData',
    'Modifiers',
    'MouseButton',
    'EVENT_CLICK',
    'EVENT_DOUBLE_CLICK',
    'EVENT_MOUSE_DOWN',
    'EVENT_MOUSE_UP',
    'EVENT_MOUSE_MOVE',
    'EVENT_MOUSE_ENTER',
    'EVENT_MOUSE_LEAVE',
    'EVENT_MOUSE_WHEEL',
    'EVENT_CONTEXT_MENU',
    'EVENT_KEY_DOWN',
    'EVENT_KEY_UP',
    'EVENT_KEY_PRESS',
    'EVENT_FOCUS',
    'EVENT_BLUR',
    'EVENT_CHANGE',
    'EVENT_SUBMIT',
    'ALL_EVENT_TYPES',
    # Event dispatcher
    'UIEventDispatcher',
    # Script executor
    'UIScriptExecutor',
    # Bindings
    'Binding',
    'BindingManager',
    'parse_bindings_from_yaml',
    # Components
    'Panel',
    'Label',
    'Button',
    'TextInput',
    'RadianceViewport',
    'ChatHistory',
    'ChatMessage',
    'MessageRole',
    'ChatInput',
]
