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

# Event dispatcher
from .event_dispatcher import UIEventDispatcher

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
    # Event dispatcher
    'UIEventDispatcher',
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
