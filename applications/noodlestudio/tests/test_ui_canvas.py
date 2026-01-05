"""
Tests for the UI Canvas System (Phase 3a)

Tests the Delphi-style UI component system including:
- Component creation and serialization
- Anchor system
- YAML loading
- Qt widget rendering
"""

import pytest
from pathlib import Path
import tempfile
import yaml


# ============================================================================
# Component Tests
# ============================================================================

class TestUIComponent:
    """Test UIComponent base class."""

    def test_component_creation(self):
        """Test basic component creation."""
        from noodlestudio.runtime.ui import UIComponent

        comp = UIComponent(name="test")
        assert comp.name == "test"
        assert comp.visible is True
        assert comp.enabled is True
        assert comp.geometry.x == 0
        assert comp.geometry.y == 0

    def test_component_geometry(self):
        """Test setting component geometry."""
        from noodlestudio.runtime.ui import UIComponent

        comp = UIComponent(name="test")
        comp.set_geometry(10, 20, 100, 50)

        assert comp.geometry.x == 10
        assert comp.geometry.y == 20
        assert comp.geometry.width == 100
        assert comp.geometry.height == 50

    def test_component_anchors(self):
        """Test anchor system."""
        from noodlestudio.runtime.ui import UIComponent, Anchors

        comp = UIComponent(name="test")
        comp.set_anchors(left=True, top=True, right=True, bottom=False)

        assert comp.anchors.left is True
        assert comp.anchors.top is True
        assert comp.anchors.right is True
        assert comp.anchors.bottom is False

    def test_anchors_from_list(self):
        """Test creating anchors from list."""
        from noodlestudio.runtime.ui import Anchors

        anchors = Anchors.from_list(["left", "right", "bottom"])
        assert anchors.left is True
        assert anchors.right is True
        assert anchors.top is False
        assert anchors.bottom is True

    def test_anchors_to_list(self):
        """Test converting anchors to list."""
        from noodlestudio.runtime.ui import Anchors

        anchors = Anchors(left=True, top=False, right=True, bottom=True)
        edges = anchors.to_list()

        assert "left" in edges
        assert "right" in edges
        assert "bottom" in edges
        assert "top" not in edges

    def test_component_hierarchy(self):
        """Test parent/child relationships."""
        from noodlestudio.runtime.ui import UIComponent

        parent = UIComponent(name="parent")
        child1 = UIComponent(name="child1")
        child2 = UIComponent(name="child2")

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert child1.parent == parent
        assert child2.parent == parent

    def test_find_by_name(self):
        """Test finding components by name."""
        from noodlestudio.runtime.ui import UIComponent

        root = UIComponent(name="root")
        child = UIComponent(name="child")
        grandchild = UIComponent(name="grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        found = root.find_by_name("grandchild")
        assert found == grandchild

        not_found = root.find_by_name("nonexistent")
        assert not_found is None


class TestPanel:
    """Test Panel component."""

    def test_panel_creation(self):
        """Test Panel creation with defaults."""
        from noodlestudio.runtime.ui import Panel

        panel = Panel(name="test_panel")
        assert panel.component_type == "Panel"
        assert panel.background == "#2a2a2a"
        assert panel.border_radius == 0

    def test_panel_properties(self):
        """Test Panel custom properties."""
        from noodlestudio.runtime.ui import Panel

        panel = Panel(name="styled")
        panel.background = "#ff0000"
        panel.border_color = "#00ff00"
        panel.border_width = 2
        panel.border_radius = 8

        assert panel.background == "#ff0000"
        assert panel.border_color == "#00ff00"
        assert panel.border_width == 2
        assert panel.border_radius == 8


class TestLabel:
    """Test Label component."""

    def test_label_creation(self):
        """Test Label creation."""
        from noodlestudio.runtime.ui import Label

        label = Label(name="test", text="Hello World")
        assert label.component_type == "Label"
        assert label.text == "Hello World"
        assert label.text_color == "#ffffff"
        assert label.font_size == 14

    def test_label_alignment(self):
        """Test Label text alignment."""
        from noodlestudio.runtime.ui.components.label import Label, TextAlign, TextVAlign

        label = Label(name="centered")
        label.align = TextAlign.CENTER
        label.valign = TextVAlign.MIDDLE

        assert label.align == TextAlign.CENTER
        assert label.valign == TextVAlign.MIDDLE


class TestButton:
    """Test Button component."""

    def test_button_creation(self):
        """Test Button creation."""
        from noodlestudio.runtime.ui import Button

        btn = Button(name="test", text="Click Me")
        assert btn.component_type == "Button"
        assert btn.text == "Click Me"
        assert btn.background == "#3b82f6"

    def test_button_events(self):
        """Test Button event binding."""
        from noodlestudio.runtime.ui import Button, EventBinding

        btn = Button(name="action")
        btn.bind_event("onClick", EventBinding(
            action="send_to_noodling",
            target="red"
        ))

        assert "onClick" in btn.events
        assert btn.events["onClick"].action == "send_to_noodling"
        assert btn.events["onClick"].target == "red"


class TestTextInput:
    """Test TextInput component."""

    def test_text_input_creation(self):
        """Test TextInput creation."""
        from noodlestudio.runtime.ui import TextInput

        inp = TextInput(name="test", placeholder="Enter text...")
        assert inp.component_type == "TextInput"
        assert inp.placeholder == "Enter text..."
        assert inp.value == ""

    def test_text_input_properties(self):
        """Test TextInput properties."""
        from noodlestudio.runtime.ui import TextInput

        inp = TextInput(name="limited")
        inp.max_length = 100
        inp.read_only = True

        assert inp.max_length == 100
        assert inp.read_only is True


class TestRadianceViewport:
    """Test RadianceViewport component."""

    def test_viewport_creation(self):
        """Test RadianceViewport creation."""
        from noodlestudio.runtime.ui import RadianceViewport

        vp = RadianceViewport(name="main")
        assert vp.component_type == "RadianceViewport"
        assert vp.interactive is True
        assert vp.background == "#000000"
        # RadianceViewport is a focused renderer - it doesn't know about stages
        # Content is added via set_component(), add_component(), or load_file()

    def test_viewport_camera_config(self):
        """Test RadianceViewport camera configuration."""
        from noodlestudio.runtime.ui import RadianceViewport

        vp = RadianceViewport(name="viewer")
        vp.camera.distance = 5.0
        vp.camera.azimuth = 90.0
        vp.camera.elevation = 30.0

        assert vp.camera.distance == 5.0
        assert vp.camera.azimuth == 90.0
        assert vp.camera.elevation == 30.0


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Test component serialization."""

    def test_panel_serialization(self):
        """Test Panel serializes to dict."""
        from noodlestudio.runtime.ui import Panel, Label

        panel = Panel(name="root")
        panel.background = "#1a1a1a"

        label = Label(name="title", text="Hello")
        label.set_geometry(10, 10, 100, 24)
        panel.add_child(label)

        data = panel.to_dict()

        assert data["type"] == "Panel"
        assert data["name"] == "root"
        assert data["background"] == "#1a1a1a"
        assert len(data["children"]) == 1
        assert data["children"][0]["type"] == "Label"

    def test_button_event_serialization(self):
        """Test Button with events serializes correctly."""
        from noodlestudio.runtime.ui import Button, EventBinding

        btn = Button(name="send", text="Send")
        btn.bind_event("onClick", EventBinding(
            action="send_to_noodling",
            target="red",
            message_source="input"
        ))

        data = btn.to_dict()

        assert "events" in data
        assert "onClick" in data["events"]
        assert data["events"]["onClick"]["action"] == "send_to_noodling"
        assert data["events"]["onClick"]["target"] == "red"


# ============================================================================
# YAML Loader Tests
# ============================================================================

class TestUILoader:
    """Test YAML loading."""

    def test_load_simple_ui(self):
        """Test loading a simple UI from YAML."""
        from noodlestudio.runtime.ui import UILoader, Panel, Label

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  background: "#1a1a1a"
  children:
    - type: Label
      name: "title"
      text: "Test"
      x: 10
      y: 10
      width: 100
      height: 24
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            assert isinstance(root, Panel)
            assert root.name == "root"
            assert root.background == "#1a1a1a"
            assert len(root.children) == 1
            assert isinstance(root.children[0], Label)
            assert root.children[0].text == "Test"

    def test_load_with_anchors(self):
        """Test loading UI with anchor definitions."""
        from noodlestudio.runtime.ui import UILoader

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: Panel
      name: "header"
      height: 50
      anchors: [left, right, top]
    - type: Panel
      name: "footer"
      height: 30
      anchors: [left, right, bottom]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            header = root.find_by_name("header")
            footer = root.find_by_name("footer")

            assert header.anchors.left is True
            assert header.anchors.right is True
            assert header.anchors.top is True
            assert header.anchors.bottom is False

            assert footer.anchors.bottom is True
            assert footer.anchors.top is False

    def test_create_default_ui(self):
        """Test default UI creation."""
        from noodlestudio.runtime.ui import create_default_ui, Panel

        root = create_default_ui()

        assert isinstance(root, Panel)
        assert root.name == "root"
        assert len(root.children) >= 1  # Has welcome labels


# ============================================================================
# Component Registry Tests
# ============================================================================

class TestComponentRegistry:
    """Test component registration system."""

    def test_list_component_types(self):
        """Test listing registered component types."""
        from noodlestudio.runtime.ui import list_component_types

        types = list_component_types()

        assert "Panel" in types
        assert "Label" in types
        assert "Button" in types
        assert "TextInput" in types
        assert "RadianceViewport" in types

    def test_get_component_class(self):
        """Test getting component class by name."""
        from noodlestudio.runtime.ui import get_component_class, Panel, Label

        panel_cls = get_component_class("Panel")
        label_cls = get_component_class("Label")

        assert panel_cls == Panel
        assert label_cls == Label

    def test_unknown_component_type(self):
        """Test unknown component type returns None."""
        from noodlestudio.runtime.ui import get_component_class

        cls = get_component_class("NonExistent")
        assert cls is None


# ============================================================================
# Qt Renderer Tests (require Qt)
# ============================================================================

class TestQtRenderer:
    """Test Qt widget rendering."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_render_panel(self, qapp):
        """Test rendering a Panel to QFrame."""
        from noodlestudio.runtime.ui import Panel, QtWidgetRenderer
        from PyQt6.QtWidgets import QFrame

        panel = Panel(name="test")
        panel.set_geometry(0, 0, 200, 100)

        renderer = QtWidgetRenderer()
        widget = renderer.render(panel)

        assert isinstance(widget, QFrame)
        assert widget.objectName() == "test"

    def test_render_label(self, qapp):
        """Test rendering a Label to QLabel."""
        from noodlestudio.runtime.ui import Label, QtWidgetRenderer
        from PyQt6.QtWidgets import QLabel

        label = Label(name="title", text="Hello")
        label.set_geometry(10, 10, 100, 24)

        renderer = QtWidgetRenderer()
        widget = renderer.render(label)

        assert isinstance(widget, QLabel)
        assert widget.text() == "Hello"

    def test_render_button(self, qapp):
        """Test rendering a Button to QPushButton."""
        from noodlestudio.runtime.ui import Button, QtWidgetRenderer
        from PyQt6.QtWidgets import QPushButton

        btn = Button(name="action", text="Click")
        btn.set_geometry(10, 10, 80, 32)

        renderer = QtWidgetRenderer()
        widget = renderer.render(btn)

        assert isinstance(widget, QPushButton)
        assert widget.text() == "Click"

    def test_render_text_input(self, qapp):
        """Test rendering a TextInput to QLineEdit."""
        from noodlestudio.runtime.ui import TextInput, QtWidgetRenderer
        from PyQt6.QtWidgets import QLineEdit

        inp = TextInput(name="field", placeholder="Type...")
        inp.set_geometry(10, 10, 200, 32)

        renderer = QtWidgetRenderer()
        widget = renderer.render(inp)

        assert isinstance(widget, QLineEdit)
        assert widget.placeholderText() == "Type..."

    def test_render_hierarchy(self, qapp):
        """Test rendering nested components."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer

        root = Panel(name="root")
        root.set_geometry(0, 0, 400, 300)

        child = Label(name="child", text="Nested")
        child.set_geometry(10, 10, 100, 24)
        root.add_child(child)

        renderer = QtWidgetRenderer()
        widget = renderer.render(root)

        # Check hierarchy
        assert len(widget.children()) >= 1
        assert renderer.get_widget("child") is not None

    def test_widget_map(self, qapp):
        """Test widget lookup by name."""
        from noodlestudio.runtime.ui import Panel, Label, Button, QtWidgetRenderer

        root = Panel(name="root")
        label = Label(name="title", text="Title")
        btn = Button(name="action", text="Go")

        root.add_child(label)
        root.add_child(btn)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        assert renderer.get_widget("root") is not None
        assert renderer.get_widget("title") is not None
        assert renderer.get_widget("action") is not None
        assert renderer.get_widget("nonexistent") is None


# ============================================================================
# Chat Component Tests (Phase 3b)
# ============================================================================

class TestChatHistory:
    """Test ChatHistory component."""

    def test_chat_history_creation(self):
        """Test basic ChatHistory creation."""
        from noodlestudio.runtime.ui import ChatHistory

        chat = ChatHistory(name="chat")
        assert chat.component_type == "ChatHistory"
        assert chat.name == "chat"
        assert len(chat.messages) == 0

    def test_add_message(self):
        """Test adding messages."""
        from noodlestudio.runtime.ui import ChatHistory, MessageRole

        chat = ChatHistory(name="chat")
        msg = chat.add_message(MessageRole.USER, "Hello!", sender_name="User")

        assert len(chat.messages) == 1
        assert msg.content == "Hello!"
        assert msg.role == MessageRole.USER
        assert msg.sender_name == "User"

    def test_multiple_messages(self):
        """Test adding multiple messages."""
        from noodlestudio.runtime.ui import ChatHistory, MessageRole

        chat = ChatHistory(name="chat")
        chat.add_message(MessageRole.USER, "Hello")
        chat.add_message(MessageRole.NOODLING, "Hi there!", sender_name="Red")
        chat.add_message(MessageRole.SYSTEM, "Connected")

        assert len(chat.messages) == 3
        assert chat.messages[0].role == MessageRole.USER
        assert chat.messages[1].role == MessageRole.NOODLING
        assert chat.messages[2].role == MessageRole.SYSTEM

    def test_clear_messages(self):
        """Test clearing messages."""
        from noodlestudio.runtime.ui import ChatHistory, MessageRole

        chat = ChatHistory(name="chat")
        chat.add_message(MessageRole.USER, "Test")
        chat.add_message(MessageRole.USER, "Test 2")
        chat.clear_messages()

        assert len(chat.messages) == 0

    def test_chat_history_serialization(self):
        """Test ChatHistory serializes correctly."""
        from noodlestudio.runtime.ui import ChatHistory, MessageRole

        chat = ChatHistory(name="chat")
        chat.user_bubble_color = "#ff0000"
        chat.add_message(MessageRole.USER, "Test message")

        data = chat.to_dict()

        assert data["type"] == "ChatHistory"
        assert data["name"] == "chat"
        assert data["user_bubble_color"] == "#ff0000"
        assert len(data["messages"]) == 1

    def test_chat_message_serialization(self):
        """Test ChatMessage serializes correctly."""
        from noodlestudio.runtime.ui import ChatMessage, MessageRole
        from datetime import datetime

        msg = ChatMessage(
            role=MessageRole.NOODLING,
            content="Hello!",
            sender_name="Red",
            timestamp=datetime(2026, 1, 3, 12, 0, 0)
        )

        data = msg.to_dict()
        assert data["role"] == "noodling"
        assert data["content"] == "Hello!"
        assert data["sender_name"] == "Red"

        # Round-trip
        restored = ChatMessage.from_dict(data)
        assert restored.role == MessageRole.NOODLING
        assert restored.content == "Hello!"


class TestChatInput:
    """Test ChatInput component."""

    def test_chat_input_creation(self):
        """Test basic ChatInput creation."""
        from noodlestudio.runtime.ui import ChatInput

        inp = ChatInput(name="input", placeholder="Type here...")
        assert inp.component_type == "ChatInput"
        assert inp.name == "input"
        assert inp.placeholder == "Type here..."
        assert inp.value == ""

    def test_chat_input_properties(self):
        """Test ChatInput properties."""
        from noodlestudio.runtime.ui import ChatInput

        inp = ChatInput(name="input")
        inp.button_background = "#ff0000"
        inp.send_button_text = "Go"
        inp.clear_on_submit = False

        assert inp.button_background == "#ff0000"
        assert inp.send_button_text == "Go"
        assert inp.clear_on_submit is False

    def test_chat_input_serialization(self):
        """Test ChatInput serializes correctly."""
        from noodlestudio.runtime.ui import ChatInput

        inp = ChatInput(name="input", placeholder="Message...")
        inp.send_button_text = "Submit"

        data = inp.to_dict()

        assert data["type"] == "ChatInput"
        assert data["placeholder"] == "Message..."
        assert data["send_button_text"] == "Submit"


class TestChatComponents:
    """Integration tests for chat components."""

    def test_chat_components_registered(self):
        """Test chat components are in registry."""
        from noodlestudio.runtime.ui import list_component_types

        types = list_component_types()

        assert "ChatHistory" in types
        assert "ChatInput" in types

    def test_load_chat_ui(self):
        """Test loading UI with chat components."""
        from noodlestudio.runtime.ui import UILoader, ChatHistory, ChatInput
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: ChatHistory
      name: "history"
      x: 0
      y: 0
      height: 300
      anchors: [left, right, top]
    - type: ChatInput
      name: "input"
      y: 310
      height: 50
      anchors: [left, right, bottom]
      placeholder: "Type..."
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            history = root.find_by_name("history")
            inp = root.find_by_name("input")

            assert isinstance(history, ChatHistory)
            assert isinstance(inp, ChatInput)
            assert inp.placeholder == "Type..."


class TestEventBinding:
    """Test event binding system."""

    def test_event_binding_creation(self):
        """Test creating event bindings."""
        from noodlestudio.runtime.ui.component import EventBinding

        binding = EventBinding(
            action="send_to_noodling",
            target="red",
            message_source="input",
            chat_history="history"
        )

        assert binding.action == "send_to_noodling"
        assert binding.target == "red"
        assert binding.message_source == "input"
        assert binding.chat_history == "history"

    def test_event_binding_from_yaml(self):
        """Test loading event binding from YAML."""
        from noodlestudio.runtime.ui import UILoader, ChatInput
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: ChatInput
      name: "input"
      events:
        onSubmit:
          action: send_to_noodling
          target: red
          message_source: self
          chat_history: chat
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            inp = root.find_by_name("input")
            assert "onSubmit" in inp.events

            binding = inp.events["onSubmit"]
            assert binding.action == "send_to_noodling"
            assert binding.target == "red"
            assert binding.chat_history == "chat"


class TestChatQtRenderer:
    """Test Qt rendering of chat components."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_render_chat_history(self, qapp):
        """Test rendering ChatHistory to widget."""
        from noodlestudio.runtime.ui import ChatHistory, QtWidgetRenderer, ChatHistoryWidget

        chat = ChatHistory(name="chat")
        chat.set_geometry(0, 0, 400, 300)

        renderer = QtWidgetRenderer()
        widget = renderer.render(chat)

        assert isinstance(widget, ChatHistoryWidget)
        assert widget.objectName() == "chat"

    def test_render_chat_input(self, qapp):
        """Test rendering ChatInput to widget."""
        from noodlestudio.runtime.ui import ChatInput, QtWidgetRenderer, ChatInputWidget

        inp = ChatInput(name="input", placeholder="Type...")
        inp.set_geometry(0, 0, 400, 50)

        renderer = QtWidgetRenderer()
        widget = renderer.render(inp)

        assert isinstance(widget, ChatInputWidget)
        assert widget.objectName() == "input"
        assert widget.input_field.placeholderText() == "Type..."

    def test_chat_history_add_message_widget(self, qapp):
        """Test adding message updates widget."""
        from noodlestudio.runtime.ui import ChatHistory, MessageRole, QtWidgetRenderer

        chat = ChatHistory(name="chat")
        chat.set_geometry(0, 0, 400, 300)

        renderer = QtWidgetRenderer()
        widget = renderer.render(chat)

        # Add message after rendering
        chat.add_message(MessageRole.USER, "Hello!", sender_name="User")

        # Check message was added to widget
        assert chat.component_type == "ChatHistory"
        assert len(chat.messages) == 1


# ============================================================================
# Phase 3d Tests - Event Wiring Extensions
# ============================================================================

class TestEventBindingScript:
    """Test script-related EventBinding features."""

    def test_event_binding_with_script(self):
        """Test creating event binding with inline script."""
        from noodlestudio.runtime.ui.component import EventBinding

        binding = EventBinding(
            action="call_script",
            script="ui.set('output', 'clicked!');"
        )

        assert binding.action == "call_script"
        assert binding.script == "ui.set('output', 'clicked!');"
        assert binding.script_file is None

    def test_event_binding_with_script_file(self):
        """Test creating event binding with script file."""
        from noodlestudio.runtime.ui.component import EventBinding

        binding = EventBinding(
            action="call_script",
            script_file="scripts/on_click.js"
        )

        assert binding.action == "call_script"
        assert binding.script is None
        assert binding.script_file == "scripts/on_click.js"

    def test_event_binding_script_serialization(self):
        """Test script binding serializes correctly."""
        from noodlestudio.runtime.ui import Button, EventBinding

        btn = Button(name="test", text="Click")
        btn.bind_event("onClick", EventBinding(
            action="call_script",
            script="console.log('clicked');",
        ))

        data = btn.to_dict()

        assert "events" in data
        assert "onClick" in data["events"]
        assert data["events"]["onClick"]["action"] == "call_script"
        assert data["events"]["onClick"]["script"] == "console.log('clicked');"

    def test_load_script_binding_from_yaml(self):
        """Test loading script binding from YAML."""
        from noodlestudio.runtime.ui import UILoader
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: Button
      name: "btn"
      text: "Click Me"
      events:
        onClick:
          action: call_script
          script: |
            ui.set('output', 'clicked!');
            console.log('Button clicked');
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            btn = root.find_by_name("btn")
            assert "onClick" in btn.events
            assert btn.events["onClick"].action == "call_script"
            assert "ui.set('output', 'clicked!')" in btn.events["onClick"].script


class TestUIScriptExecutor:
    """Test UIScriptExecutor."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_script_executor_creation(self, qapp):
        """Test creating script executor."""
        from noodlestudio.runtime.ui import QtWidgetRenderer, UIScriptExecutor

        renderer = QtWidgetRenderer()
        executor = UIScriptExecutor(renderer)

        assert executor.renderer == renderer
        assert executor.execution_count == 0

    def test_execute_simple_script(self, qapp):
        """Test executing a simple script."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer, UIScriptExecutor

        # Create UI
        root = Panel(name="root")
        label = Label(name="output", text="Initial")
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Execute script
        executor = UIScriptExecutor(renderer)
        result = executor.execute(
            script="ui.set('output', 'Changed!');",
            event_type="onClick",
            source_component="root"
        )

        assert result['success'] is True
        # Check value was updated
        output = renderer.get_component('output')
        assert output.text == "Changed!"

    def test_execute_script_with_console(self, qapp):
        """Test script console.log output."""
        from noodlestudio.runtime.ui import Panel, QtWidgetRenderer, UIScriptExecutor

        root = Panel(name="root")
        renderer = QtWidgetRenderer()
        renderer.render(root)

        executor = UIScriptExecutor(renderer)
        result = executor.execute(
            script="console.log('Hello'); console.warn('Warning');",
            event_type="onClick"
        )

        assert result['success'] is True
        logs = result['logs']
        assert len(logs) >= 2
        assert any('Hello' in l['message'] for l in logs)

    def test_execute_script_error_handling(self, qapp):
        """Test script error handling."""
        from noodlestudio.runtime.ui import Panel, QtWidgetRenderer, UIScriptExecutor

        root = Panel(name="root")
        renderer = QtWidgetRenderer()
        renderer.render(root)

        executor = UIScriptExecutor(renderer)
        result = executor.execute(
            script="undefined_function();",
            event_type="onClick"
        )

        assert result['success'] is False
        assert result['error'] is not None

    def test_execute_script_with_event_value(self, qapp):
        """Test script access to event.value."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer, UIScriptExecutor

        root = Panel(name="root")
        label = Label(name="output", text="")
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        executor = UIScriptExecutor(renderer)
        result = executor.execute(
            script="ui.set('output', 'Value: ' + event.value);",
            event_type="onChange",
            source_component="input",
            event_value="test_value"
        )

        assert result['success'] is True
        output = renderer.get_component('output')
        assert output.text == "Value: test_value"


class TestBindingManager:
    """Test BindingManager."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_binding_manager_creation(self):
        """Test creating binding manager."""
        from noodlestudio.runtime.ui import BindingManager

        manager = BindingManager()
        assert manager._bindings == []

    def test_add_binding(self, qapp):
        """Test adding a binding."""
        from noodlestudio.runtime.ui import QtWidgetRenderer, BindingManager

        renderer = QtWidgetRenderer()
        manager = BindingManager(renderer)

        binding = manager.add_binding(
            target_component="output",
            target_property="text",
            source_expression="input.value"
        )

        assert binding.target_component == "output"
        assert binding.target_property == "text"
        assert binding.source_expression == "input.value"
        assert len(manager._bindings) == 1

    def test_binding_evaluate(self, qapp):
        """Test evaluating bindings."""
        from noodlestudio.runtime.ui import Panel, Label, TextInput, QtWidgetRenderer, BindingManager

        # Create UI
        root = Panel(name="root")
        inp = TextInput(name="input", placeholder="Type...")
        inp.value = "Hello"
        label = Label(name="output", text="")
        root.add_child(inp)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Set up binding
        manager = BindingManager(renderer)
        manager.add_binding("output", "text", "input.value")
        manager.evaluate_all()

        # Check binding was applied
        output = renderer.get_component("output")
        assert output.text == "Hello"

    def test_binding_notify_change(self, qapp):
        """Test binding updates on source change."""
        from noodlestudio.runtime.ui import Panel, Label, TextInput, QtWidgetRenderer, BindingManager

        # Create UI
        root = Panel(name="root")
        inp = TextInput(name="input", placeholder="Type...")
        inp.value = "Initial"
        label = Label(name="output", text="")
        root.add_child(inp)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Set up binding
        manager = BindingManager(renderer)
        manager.add_binding("output", "text", "input.value")
        manager.evaluate_all()

        # Change source value
        inp.value = "Updated"
        manager.notify_change("input", "value")

        # Check binding was updated
        output = renderer.get_component("output")
        assert output.text == "Updated"


class TestComponentBindings:
    """Test component-level bindings property."""

    def test_component_bindings_property(self):
        """Test bindings property on UIComponent."""
        from noodlestudio.runtime.ui import Label

        label = Label(name="output", text="")
        label.bindings = {"text": "input.value", "visible": "toggle.checked"}

        assert "text" in label.bindings
        assert label.bindings["text"] == "input.value"
        assert label.bindings["visible"] == "toggle.checked"

    def test_bindings_serialization(self):
        """Test bindings serialize to dict."""
        from noodlestudio.runtime.ui import Label

        label = Label(name="output", text="")
        label.bindings = {"text": "input.value"}

        data = label.to_dict()

        assert "bindings" in data
        assert data["bindings"]["text"] == "input.value"

    def test_bindings_from_yaml(self):
        """Test loading bindings from YAML."""
        from noodlestudio.runtime.ui import UILoader
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: TextInput
      name: "input"
      placeholder: "Type..."
    - type: Label
      name: "output"
      text: ""
      bindings:
        text: "input.value"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            output = root.find_by_name("output")
            assert "text" in output.bindings
            assert output.bindings["text"] == "input.value"

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_renderer_sets_up_bindings(self, qapp):
        """Test renderer automatically sets up bindings."""
        from noodlestudio.runtime.ui import Panel, Label, TextInput, QtWidgetRenderer

        # Create UI with bindings
        root = Panel(name="root")
        inp = TextInput(name="input", placeholder="Type...")
        inp.value = "Test"
        label = Label(name="output", text="")
        label.bindings = {"text": "input.value"}

        root.add_child(inp)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Bindings should be set up and evaluated
        output = renderer.get_component("output")
        assert output.text == "Test"


class TestUIEventDispatcher:
    """Test UIEventDispatcher event routing."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_dispatcher_creation(self, qapp):
        """Test creating event dispatcher."""
        from noodlestudio.runtime.ui import QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher

        renderer = QtWidgetRenderer()
        dispatcher = UIEventDispatcher(renderer)

        assert dispatcher.renderer == renderer
        assert dispatcher.app is None
        assert dispatcher.default_chat_history == "chat_history"

    def test_dispatcher_set_value_action(self, qapp):
        """Test set_value action updates component."""
        from noodlestudio.runtime.ui import Panel, TextInput, Button, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI - use TextInput which has .value property
        root = Panel(name="root")
        button = Button(name="btn", text="Click")
        text_input = TextInput(name="output", placeholder="Type...")
        text_input.value = "Initial"
        root.add_child(button)
        root.add_child(text_input)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)

        # Create binding for set_value
        binding = EventBinding(action="set_value", target="output")
        binding.value = "Updated!"

        # Dispatch event
        dispatcher.dispatch("onClick", button, binding)

        # Check value was set
        output = renderer.get_component("output")
        assert output.value == "Updated!"

    def test_dispatcher_show_action(self, qapp):
        """Test show action makes component visible."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI with hidden label
        root = Panel(name="root")
        label = Label(name="target", text="Hidden")
        label.visible = False
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Hide the widget initially
        widget = renderer.get_widget("target")
        widget.hide()

        dispatcher = UIEventDispatcher(renderer)
        binding = EventBinding(action="show", target="target")

        # Dispatch show event
        dispatcher.dispatch("onClick", root, binding)

        # Check visibility
        assert renderer.get_component("target").visible is True
        assert widget.isVisible() is True

    def test_dispatcher_hide_action(self, qapp):
        """Test hide action hides component."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI with visible label
        root = Panel(name="root")
        label = Label(name="target", text="Visible")
        label.visible = True
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)
        binding = EventBinding(action="hide", target="target")

        # Dispatch hide event
        dispatcher.dispatch("onClick", root, binding)

        # Check hidden
        assert renderer.get_component("target").visible is False
        widget = renderer.get_widget("target")
        assert widget.isVisible() is False

    def test_dispatcher_toggle_visible_action(self, qapp):
        """Test toggle_visible action toggles visibility."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI
        root = Panel(name="root")
        label = Label(name="target", text="Toggle me")
        label.visible = True
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)
        binding = EventBinding(action="toggle_visible", target="target")

        # Toggle off
        dispatcher.dispatch("onClick", root, binding)
        assert renderer.get_component("target").visible is False

        # Toggle on
        dispatcher.dispatch("onClick", root, binding)
        assert renderer.get_component("target").visible is True

    def test_dispatcher_custom_handler(self, qapp):
        """Test registering and calling custom handler."""
        from noodlestudio.runtime.ui import Panel, Button, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI
        root = Panel(name="root")
        button = Button(name="btn", text="Custom")
        root.add_child(button)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)

        # Track custom handler calls
        calls = []

        def custom_handler(component, binding):
            calls.append((component.name, binding.target))

        dispatcher.register_handler("my_custom_action", custom_handler)

        binding = EventBinding(action="my_custom_action", target="some_target")
        dispatcher.dispatch("onClick", button, binding)

        assert len(calls) == 1
        assert calls[0] == ("btn", "some_target")

    def test_dispatcher_call_script_action(self, qapp):
        """Test call_script action executes script."""
        from noodlestudio.runtime.ui import Panel, Button, Label, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI
        root = Panel(name="root")
        button = Button(name="btn", text="Run Script")
        label = Label(name="output", text="Initial")
        root.add_child(button)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)

        binding = EventBinding(
            action="call_script",
            script="ui.set('output', 'Script ran!');"
        )

        dispatcher.dispatch("onClick", button, binding)

        output = renderer.get_component("output")
        assert output.text == "Script ran!"


class TestUIIntegration:
    """Integration tests for full UI event flow."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_button_click_updates_label(self, qapp):
        """Integration: Load YAML, click button, verify label updates."""
        from noodlestudio.runtime.ui import UILoader, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: Button
      name: "clickMe"
      text: "Click Me"
      events:
        onClick:
          action: call_script
          script: "ui.set('result', 'Clicked!');"
    - type: Label
      name: "result"
      text: "Not clicked"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Load UI
            loader = UILoader()
            root = loader.load_file(f.name)

            # Render
            renderer = QtWidgetRenderer()
            widget = renderer.render(root)

            # Set up dispatcher
            dispatcher = UIEventDispatcher(renderer)
            renderer.set_event_dispatcher(dispatcher.dispatch)

            # Verify initial state
            result = renderer.get_component("result")
            assert result.text == "Not clicked"

            # Simulate button click by triggering the event directly
            button = renderer.get_component("clickMe")
            binding = button.events["onClick"]
            dispatcher.dispatch("onClick", button, binding)

            # Verify label updated
            assert result.text == "Clicked!"

    def test_toggle_panel_visibility(self, qapp):
        """Integration: Toggle panel visibility via button."""
        from noodlestudio.runtime.ui import UILoader, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        import tempfile

        yaml_content = """
version: 1
root:
  type: Panel
  name: "root"
  children:
    - type: Button
      name: "toggleBtn"
      text: "Toggle"
      events:
        onClick:
          action: toggle_visible
          target: details
    - type: Panel
      name: "details"
      background: "#333333"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            loader = UILoader()
            root = loader.load_file(f.name)

            renderer = QtWidgetRenderer()
            renderer.render(root)

            dispatcher = UIEventDispatcher(renderer)
            renderer.set_event_dispatcher(dispatcher.dispatch)

            # Initial: visible
            details = renderer.get_component("details")
            assert details.visible is True

            # Click toggle
            button = renderer.get_component("toggleBtn")
            binding = button.events["onClick"]
            dispatcher.dispatch("onClick", button, binding)

            # Now hidden
            assert details.visible is False

            # Click again
            dispatcher.dispatch("onClick", button, binding)

            # Visible again
            assert details.visible is True

    def test_binding_updates_on_input(self, qapp):
        """Integration: Text input binding updates label."""
        from noodlestudio.runtime.ui import Panel, Label, TextInput, QtWidgetRenderer

        # Create UI with bindings programmatically
        root = Panel(name="root")
        inp = TextInput(name="nameInput", placeholder="Enter name")
        inp.value = "Alice"
        label = Label(name="greeting", text="")
        label.bindings = {"text": "nameInput.value"}  # Simple binding

        root.add_child(inp)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        # Bindings should be evaluated on render
        greeting = renderer.get_component("greeting")
        assert greeting.text == "Alice"

        # Simulate input change
        name_input = renderer.get_component("nameInput")
        name_input.value = "Bob"

        # Notify binding manager
        renderer.notify_binding_change("nameInput", "value")

        # Check label updated
        assert greeting.text == "Bob"


# ============================================================================
# Phase 7A: UIEventData Tests
# ============================================================================

class TestUIEventData:
    """Test UIEventData dataclass and factory methods."""

    def test_event_data_creation(self):
        """Test basic UIEventData creation."""
        from noodlestudio.runtime.ui import UIEventData, MouseButton

        event = UIEventData(
            type="onClick",
            source="myButton",
            x=100,
            y=50,
            button=MouseButton.LEFT
        )

        assert event.type == "onClick"
        assert event.source == "myButton"
        assert event.x == 100
        assert event.y == 50
        assert event.button == MouseButton.LEFT
        assert event.timestamp > 0

    def test_event_data_modifiers(self):
        """Test Modifiers dataclass."""
        from noodlestudio.runtime.ui import Modifiers

        mods = Modifiers(shift=True, ctrl=False, alt=True, meta=False)

        assert mods.shift is True
        assert mods.ctrl is False
        assert mods.alt is True
        assert mods.meta is False

        d = mods.to_dict()
        assert d["shift"] is True
        assert d["ctrl"] is False
        assert d["alt"] is True
        assert d["meta"] is False

    def test_event_data_to_dict(self):
        """Test UIEventData.to_dict() for script access."""
        from noodlestudio.runtime.ui import UIEventData, MouseButton, Modifiers

        event = UIEventData(
            type="onMouseDown",
            source="panel",
            x=10,
            y=20,
            global_x=110,
            global_y=220,
            button=MouseButton.RIGHT,
            modifiers=Modifiers(shift=True)
        )

        d = event.to_dict()

        assert d["type"] == "onMouseDown"
        assert d["source"] == "panel"
        assert d["x"] == 10
        assert d["y"] == 20
        assert d["globalX"] == 110
        assert d["globalY"] == 220
        assert d["button"] == "right"
        assert d["modifiers"]["shift"] is True

    def test_event_data_click_factory(self):
        """Test UIEventData.click() factory method."""
        from noodlestudio.runtime.ui import UIEventData, MouseButton

        event = UIEventData.click("submitBtn")

        assert event.type == "onClick"
        assert event.source == "submitBtn"
        assert event.button == MouseButton.LEFT

    def test_event_data_value_change_factory(self):
        """Test UIEventData.value_change() factory method."""
        from noodlestudio.runtime.ui import UIEventData

        event = UIEventData.value_change("inputField", "new text", "old text")

        assert event.type == "onChange"
        assert event.source == "inputField"
        assert event.value == "new text"
        assert event.previous_value == "old text"

    def test_event_data_submit_factory(self):
        """Test UIEventData.submit() factory method."""
        from noodlestudio.runtime.ui import UIEventData

        event = UIEventData.submit("form", {"username": "test"})

        assert event.type == "onSubmit"
        assert event.source == "form"
        assert event.value == {"username": "test"}

    def test_event_data_focus_factory(self):
        """Test UIEventData.focus() factory method."""
        from noodlestudio.runtime.ui import UIEventData

        focus_event = UIEventData.focus("onFocus", "inputField")
        blur_event = UIEventData.focus("onBlur", "inputField")

        assert focus_event.type == "onFocus"
        assert focus_event.source == "inputField"
        assert blur_event.type == "onBlur"

    def test_event_data_keyboard_fields(self):
        """Test keyboard event fields."""
        from noodlestudio.runtime.ui import UIEventData, Modifiers

        event = UIEventData(
            type="onKeyDown",
            source="textInput",
            key="Enter",
            key_code=13,
            text="",
            modifiers=Modifiers(ctrl=True)
        )

        d = event.to_dict()

        assert d["key"] == "Enter"
        assert d["keyCode"] == 13
        assert d["modifiers"]["ctrl"] is True

    def test_event_data_wheel_fields(self):
        """Test mouse wheel event fields."""
        from noodlestudio.runtime.ui import UIEventData

        event = UIEventData(
            type="onMouseWheel",
            source="scrollPanel",
            x=50,
            y=50,
            delta_x=0.0,
            delta_y=-3.0
        )

        d = event.to_dict()

        assert d["type"] == "onMouseWheel"
        assert d["deltaY"] == -3.0

    def test_event_data_3d_fields(self):
        """Test RadianceViewport 3D event fields."""
        from noodlestudio.runtime.ui import UIEventData

        event = UIEventData(
            type="onGaussianClick",
            source="viewport",
            hit_position=(1.0, 2.0, 3.0),
            hit_entity="noodling_red",
            hit_semantics={"body_part": "head", "emotion": "happy"}
        )

        d = event.to_dict()

        assert d["hitPosition"]["x"] == 1.0
        assert d["hitPosition"]["y"] == 2.0
        assert d["hitPosition"]["z"] == 3.0
        assert d["hitEntity"] == "noodling_red"
        assert d["hitSemantics"]["body_part"] == "head"

    def test_event_data_propagation_control(self):
        """Test stop_propagation and prevent_default."""
        from noodlestudio.runtime.ui import UIEventData

        event = UIEventData(type="onClick", source="btn")

        assert event.stopped is False
        assert event.prevented is False

        event.stop_propagation()
        assert event.stopped is True

        event.prevent_default()
        assert event.prevented is True


class TestEventConstants:
    """Test event type constants."""

    def test_event_constants_defined(self):
        """Test all event constants are defined."""
        from noodlestudio.runtime.ui import (
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
            EVENT_FOCUS,
            EVENT_BLUR,
            EVENT_CHANGE,
            EVENT_SUBMIT,
            ALL_EVENT_TYPES,
        )

        assert EVENT_CLICK == "onClick"
        assert EVENT_DOUBLE_CLICK == "onDoubleClick"
        assert EVENT_MOUSE_DOWN == "onMouseDown"
        assert EVENT_KEY_DOWN == "onKeyDown"
        assert EVENT_FOCUS == "onFocus"
        assert EVENT_CHANGE == "onChange"

        # All constants should be in ALL_EVENT_TYPES
        assert EVENT_CLICK in ALL_EVENT_TYPES
        assert EVENT_KEY_DOWN in ALL_EVENT_TYPES
        assert EVENT_SUBMIT in ALL_EVENT_TYPES


class TestEventEmittingWidgets:
    """Test EventEmitting widget mixins."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_event_emitting_frame_creation(self, qapp):
        """Test creating EventEmittingFrame."""
        from noodlestudio.runtime.ui import Panel, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_widgets import EventEmittingFrame

        panel = Panel(name="testPanel")
        renderer = QtWidgetRenderer()

        frame = EventEmittingFrame(panel, renderer)

        assert frame._ui_component == panel
        assert frame._ui_renderer == renderer

    def test_event_emitting_button_creation(self, qapp):
        """Test creating EventEmittingButton."""
        from noodlestudio.runtime.ui import Button, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_widgets import EventEmittingButton

        button = Button(name="testBtn", text="Click")
        renderer = QtWidgetRenderer()

        widget = EventEmittingButton(button, renderer, "Click")

        assert widget._ui_component == button
        assert widget.text() == "Click"

    def test_event_emitting_line_edit_creation(self, qapp):
        """Test creating EventEmittingLineEdit."""
        from noodlestudio.runtime.ui import TextInput, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_widgets import EventEmittingLineEdit

        text_input = TextInput(name="testInput")
        renderer = QtWidgetRenderer()

        widget = EventEmittingLineEdit(text_input, renderer)

        assert widget._ui_component == text_input


class TestUIEventDataWithDispatcher:
    """Test UIEventData integration with event dispatcher."""

    @pytest.fixture
    def qapp(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    def test_dispatcher_receives_event_data(self, qapp):
        """Test dispatcher receives and uses UIEventData."""
        from noodlestudio.runtime.ui import Panel, Button, Label, QtWidgetRenderer, UIEventData
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI
        root = Panel(name="root")
        button = Button(name="btn", text="Click")
        label = Label(name="output", text="")
        root.add_child(button)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)

        # Create binding for call_script that uses event data
        binding = EventBinding(
            action="call_script",
            script="ui.set('output', 'Clicked at ' + event.x + ',' + event.y);"
        )

        # Create event data with position
        event_data = UIEventData(
            type="onClick",
            source="btn",
            x=100,
            y=50
        )

        # Dispatch with event data
        dispatcher.dispatch("onClick", button, binding, event_data)

        # Check script received event data
        output = renderer.get_component("output")
        assert output.text == "Clicked at 100,50"

    def test_dispatcher_creates_default_event_data(self, qapp):
        """Test dispatcher creates basic event data if none provided."""
        from noodlestudio.runtime.ui import Panel, Button, Label, QtWidgetRenderer
        from noodlestudio.runtime.ui.event_dispatcher import UIEventDispatcher
        from noodlestudio.runtime.ui.component import EventBinding

        # Create UI
        root = Panel(name="root")
        button = Button(name="btn", text="Click")
        label = Label(name="output", text="")
        root.add_child(button)
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        dispatcher = UIEventDispatcher(renderer)

        # Create binding
        binding = EventBinding(
            action="call_script",
            script="ui.set('output', event.type + ' from ' + event.source);"
        )

        # Dispatch WITHOUT event data - should create default
        dispatcher.dispatch("onClick", button, binding)

        # Check script received basic event data
        output = renderer.get_component("output")
        assert output.text == "onClick from btn"

    def test_script_executor_with_full_event_data(self, qapp):
        """Test script executor receives full UIEventData."""
        from noodlestudio.runtime.ui import Panel, Label, QtWidgetRenderer, UIScriptExecutor, UIEventData, Modifiers

        root = Panel(name="root")
        label = Label(name="output", text="")
        root.add_child(label)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        executor = UIScriptExecutor(renderer)

        # Create rich event data
        event_data = UIEventData(
            type="onKeyDown",
            source="input",
            key="Enter",
            key_code=13,
            modifiers=Modifiers(shift=True, ctrl=True)
        )

        result = executor.execute(
            script="""
            var msg = 'Key: ' + event.key;
            msg += ', Shift: ' + event.modifiers.shift;
            msg += ', Ctrl: ' + event.modifiers.ctrl;
            ui.set('output', msg);
            """,
            event_data=event_data
        )

        assert result['success'] is True
        output = renderer.get_component('output')
        assert "Key: Enter" in output.text
        assert "Shift: true" in output.text
        assert "Ctrl: true" in output.text


# ============================================================================
# Phase 7B: Event Wiring UI Tests
# ============================================================================

class TestEventBindingWidget:
    """Test EventBindingWidget for Inspector event wiring."""

    def test_widget_creation(self, qapp):
        """Test basic widget creation."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        widget = EventBindingWidget(event_name="onClick")
        assert widget.event_name == "onClick"
        assert widget.event_label.text() == "onClick"

    def test_widget_with_binding_data(self, qapp):
        """Test widget initialized with binding data."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        binding_data = {
            "action": "send_to_noodling",
            "target": "red",
            "message_source": "input",
        }

        widget = EventBindingWidget(
            event_name="onSubmit",
            binding_data=binding_data
        )

        data = widget.get_binding_data()
        assert data["action"] == "send_to_noodling"
        assert data["target"] == "red"
        assert data["message_source"] == "input"

    def test_widget_action_change(self, qapp):
        """Test changing action type."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        widget = EventBindingWidget(event_name="onClick")

        # Default is send_to_noodling
        data = widget.get_binding_data()
        assert data["action"] == "send_to_noodling"

        # Change to call_script
        for i in range(widget.action_combo.count()):
            if widget.action_combo.itemData(i) == "call_script":
                widget.action_combo.setCurrentIndex(i)
                break

        data = widget.get_binding_data()
        assert data["action"] == "call_script"

    def test_widget_delete_signal(self, qapp):
        """Test delete button emits signal."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        widget = EventBindingWidget(event_name="onClick")

        delete_called = []
        widget.delete_requested.connect(lambda: delete_called.append(True))

        widget.delete_btn.click()

        assert len(delete_called) == 1

    def test_widget_changed_signal(self, qapp):
        """Test that changes emit changed signal."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        widget = EventBindingWidget(
            event_name="onClick",
            available_noodlings=["red", "blue"]
        )

        changed_count = []
        widget.changed.connect(lambda: changed_count.append(True))

        # Change target
        widget.target_combo.setCurrentText("blue")

        assert len(changed_count) >= 1

    def test_widget_script_content(self, qapp):
        """Test setting script content."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        binding_data = {
            "action": "call_script",
            "script": "console.log('test');"
        }

        widget = EventBindingWidget(
            event_name="onClick",
            binding_data=binding_data
        )

        # Set new script
        widget.set_script_content("ui.set('label', 'clicked');")

        data = widget.get_binding_data()
        assert data["script"] == "ui.set('label', 'clicked');"

    def test_widget_available_components(self, qapp):
        """Test setting available components for dropdown."""
        from noodlestudio.widgets.event_binding_widget import EventBindingWidget

        widget = EventBindingWidget(
            event_name="onClick",
            available_components=["panel1", "button1", "label1"]
        )

        # Switch to show action (which shows components in target)
        for i in range(widget.action_combo.count()):
            if widget.action_combo.itemData(i) == "show":
                widget.action_combo.setCurrentIndex(i)
                break

        # Check components are in target dropdown
        targets = [widget.target_combo.itemText(i) for i in range(widget.target_combo.count())]
        assert "panel1" in targets
        assert "button1" in targets
        assert "label1" in targets


class TestScriptEditorDialog:
    """Test ScriptEditorDialog for inline JavaScript editing."""

    def test_dialog_creation(self, qapp):
        """Test basic dialog creation."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(
            script_content="console.log('hello');",
            event_name="onClick"
        )

        assert dialog.event_name == "onClick"
        assert "hello" in dialog.code_editor.toPlainText()

    def test_syntax_highlighter(self, qapp):
        """Test JavaScript syntax highlighter exists."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(
            script_content="function test() { return true; }",
            event_name="onClick"
        )

        # Highlighter should be attached
        assert dialog.highlighter is not None
        assert dialog.highlighter.document() == dialog.code_editor.document()

    def test_validation_balanced_braces(self, qapp):
        """Test validation catches unbalanced braces."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(event_name="onClick")

        # Unbalanced braces
        error = dialog._validate_script("function test() {")
        assert error is not None
        assert "brace" in error.lower()

        # Balanced braces
        error = dialog._validate_script("function test() { return 1; }")
        assert error is None

    def test_validation_balanced_parens(self, qapp):
        """Test validation catches unbalanced parentheses."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(event_name="onClick")

        # Unbalanced parens
        error = dialog._validate_script("console.log('test'")
        assert error is not None
        assert "parenthes" in error.lower()

        # Balanced parens
        error = dialog._validate_script("console.log('test');")
        assert error is None

    def test_validation_unclosed_string(self, qapp):
        """Test validation catches unclosed strings."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(event_name="onClick")

        # Unclosed string
        error = dialog._validate_script('var x = "hello')
        assert error is not None
        assert "string" in error.lower()

        # Closed string
        error = dialog._validate_script('var x = "hello";')
        assert error is None

    def test_validation_empty_script_valid(self, qapp):
        """Test empty script is considered valid (clears binding)."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(event_name="onClick")

        error = dialog._validate_script("")
        assert error is None

        error = dialog._validate_script("   ")
        assert error is None

    def test_get_script(self, qapp):
        """Test getting script content after edit."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(
            script_content="initial",
            event_name="onClick"
        )

        dialog.code_editor.setPlainText("modified")
        dialog._result_script = dialog.code_editor.toPlainText()

        assert dialog.get_script() == "modified"

    def test_api_tree_exists(self, qapp):
        """Test API reference tree is populated."""
        from noodlestudio.dialogs.script_editor_dialog import ScriptEditorDialog

        dialog = ScriptEditorDialog(event_name="onClick")

        # Should have top-level items for ui, event, console
        root = dialog.api_tree.invisibleRootItem()
        top_level_texts = [root.child(i).text(0) for i in range(root.childCount())]

        assert "ui" in top_level_texts
        assert "event" in top_level_texts
        assert "console" in top_level_texts


class TestJavaScriptHighlighter:
    """Test JavaScript syntax highlighter."""

    def test_highlighter_creation(self, qapp):
        """Test highlighter creation."""
        from noodlestudio.dialogs.script_editor_dialog import JavaScriptHighlighter
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        highlighter = JavaScriptHighlighter(editor.document())

        assert highlighter is not None
        assert highlighter.document() == editor.document()

    def test_keywords_defined(self, qapp):
        """Test keywords list is populated."""
        from noodlestudio.dialogs.script_editor_dialog import JavaScriptHighlighter
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        highlighter = JavaScriptHighlighter(editor.document())

        assert "function" in highlighter.keywords
        assert "var" in highlighter.keywords
        assert "const" in highlighter.keywords
        assert "return" in highlighter.keywords

    def test_api_objects_defined(self, qapp):
        """Test API objects list is populated."""
        from noodlestudio.dialogs.script_editor_dialog import JavaScriptHighlighter
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        highlighter = JavaScriptHighlighter(editor.document())

        assert "ui" in highlighter.api_objects
        assert "event" in highlighter.api_objects
        assert "console" in highlighter.api_objects


# ============================================================================
# Phase 7C: New Component Tests
# ============================================================================

class TestCheckboxComponent:
    """Test Checkbox component."""

    def test_checkbox_creation(self):
        """Test basic checkbox creation."""
        from noodlestudio.runtime.ui.components import Checkbox

        cb = Checkbox(name="test_cb", text="Enable Feature", checked=True)
        assert cb.name == "test_cb"
        assert cb.text == "Enable Feature"
        assert cb.checked is True
        assert cb.value is True  # Alias

    def test_checkbox_toggle(self):
        """Test checkbox toggle method."""
        from noodlestudio.runtime.ui.components import Checkbox

        cb = Checkbox(checked=False)
        assert cb.checked is False

        result = cb.toggle()
        assert result is True
        assert cb.checked is True

        result = cb.toggle()
        assert result is False
        assert cb.checked is False

    def test_checkbox_serialization(self):
        """Test checkbox serialization."""
        from noodlestudio.runtime.ui.components import Checkbox

        cb = Checkbox(name="my_checkbox", text="Accept Terms", checked=True)
        cb.check_color = "#ff0000"

        data = cb.to_dict()
        assert data["name"] == "my_checkbox"
        assert data["text"] == "Accept Terms"
        assert data["checked"] is True
        assert data["check_color"] == "#ff0000"

    def test_checkbox_deserialization(self):
        """Test checkbox deserialization."""
        from noodlestudio.runtime.ui.components import Checkbox

        data = {
            "name": "loaded_cb",
            "text": "Remember Me",
            "checked": True,
            "box_size": 20
        }

        cb = Checkbox.from_dict(data)
        assert cb.name == "loaded_cb"
        assert cb.text == "Remember Me"
        assert cb.checked is True
        assert cb.box_size == 20

    def test_checkbox_render(self, qapp):
        """Test checkbox Qt rendering."""
        from noodlestudio.runtime.ui.components import Panel, Checkbox
        from noodlestudio.runtime.ui import QtWidgetRenderer

        root = Panel(name="root")
        cb = Checkbox(name="cb", text="Test", checked=True)
        root.add_child(cb)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        widget = renderer.get_widget("cb")
        assert widget is not None
        assert widget.isChecked() is True


class TestDropdownComponent:
    """Test Dropdown component."""

    def test_dropdown_creation(self):
        """Test basic dropdown creation."""
        from noodlestudio.runtime.ui.components import Dropdown

        dd = Dropdown(
            name="color_select",
            options=["Red", "Green", "Blue"],
            selected_index=1
        )
        assert dd.name == "color_select"
        assert dd.options == ["Red", "Green", "Blue"]
        assert dd.selected_index == 1
        assert dd.value == "Green"

    def test_dropdown_value_property(self):
        """Test setting value by string."""
        from noodlestudio.runtime.ui.components import Dropdown

        dd = Dropdown(options=["A", "B", "C"])
        dd.value = "C"
        assert dd.selected_index == 2
        assert dd.value == "C"

        dd.value = "NotInList"
        assert dd.selected_index == -1
        assert dd.value is None

    def test_dropdown_add_remove_options(self):
        """Test adding and removing options."""
        from noodlestudio.runtime.ui.components import Dropdown

        dd = Dropdown()
        assert len(dd.options) == 0

        idx = dd.add_option("First")
        assert idx == 0
        assert "First" in dd.options

        dd.add_option("Second")
        dd.selected_index = 1

        dd.remove_option("First")
        assert dd.selected_index == 0  # Adjusted
        assert dd.value == "Second"

    def test_dropdown_serialization(self):
        """Test dropdown serialization."""
        from noodlestudio.runtime.ui.components import Dropdown

        dd = Dropdown(name="dd", options=["X", "Y"], selected_index=0)
        data = dd.to_dict()

        assert data["options"] == ["X", "Y"]
        assert data["selected_index"] == 0

    def test_dropdown_render(self, qapp):
        """Test dropdown Qt rendering."""
        from noodlestudio.runtime.ui.components import Panel, Dropdown
        from noodlestudio.runtime.ui import QtWidgetRenderer

        root = Panel(name="root")
        dd = Dropdown(name="dd", options=["One", "Two"], selected_index=1)
        root.add_child(dd)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        widget = renderer.get_widget("dd")
        assert widget is not None


class TestSliderComponent:
    """Test Slider component."""

    def test_slider_creation(self):
        """Test basic slider creation."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(name="volume", value=50, min_value=0, max_value=100)
        assert slider.name == "volume"
        assert slider.value == 50
        assert slider.min_value == 0
        assert slider.max_value == 100

    def test_slider_value_clamping(self):
        """Test value is clamped to range."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(min_value=0, max_value=100)

        slider.value = 150
        assert slider.value == 100

        slider.value = -50
        assert slider.value == 0

    def test_slider_percentage(self):
        """Test percentage property."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(min_value=0, max_value=100, value=25)
        assert slider.percentage == 0.25

        slider.percentage = 0.5
        assert slider.value == 50

    def test_slider_step(self):
        """Test step snapping."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(min_value=0, max_value=100)
        slider.step = 10

        slider.value = 23
        assert slider.value == 20  # Snapped to nearest step

        slider.value = 27
        assert slider.value == 30

    def test_slider_formatted_value(self):
        """Test formatted value display."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(value=33.333)
        slider.value_format = "{:.1f}%"

        assert slider.formatted_value == "33.3%"

    def test_slider_serialization(self):
        """Test slider serialization."""
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(name="s", value=50, min_value=0, max_value=100)
        slider.step = 5
        data = slider.to_dict()

        assert data["value"] == 50
        assert data["min_value"] == 0
        assert data["max_value"] == 100
        assert data["step"] == 5

    def test_slider_render(self, qapp):
        """Test slider Qt rendering."""
        from noodlestudio.runtime.ui.components import Panel, Slider
        from noodlestudio.runtime.ui import QtWidgetRenderer

        root = Panel(name="root")
        slider = Slider(name="slider", value=50)
        slider.show_value = True
        root.add_child(slider)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        widget = renderer.get_widget("slider")
        assert widget is not None


class TestRadioComponents:
    """Test RadioButton and RadioGroup components."""

    def test_radio_button_creation(self):
        """Test basic radio button creation."""
        from noodlestudio.runtime.ui.components import RadioButton

        rb = RadioButton(name="opt1", text="Option 1", value="opt1_value", checked=True)
        assert rb.name == "opt1"
        assert rb.text == "Option 1"
        assert rb.option_value == "opt1_value"
        assert rb.checked is True

    def test_radio_group_creation(self):
        """Test radio group creation."""
        from noodlestudio.runtime.ui.components import RadioGroup

        rg = RadioGroup(name="size", options=["Small", "Medium", "Large"])
        assert rg.name == "size"
        assert rg.options == ["Small", "Medium", "Large"]
        assert rg.selected_index == -1
        assert rg.value is None

    def test_radio_group_selection(self):
        """Test radio group selection."""
        from noodlestudio.runtime.ui.components import RadioGroup

        rg = RadioGroup(options=["A", "B", "C"])

        rg.selected_index = 1
        assert rg.value == "B"

        rg.value = "C"
        assert rg.selected_index == 2

    def test_radio_group_select_method(self):
        """Test select method returns change status."""
        from noodlestudio.runtime.ui.components import RadioGroup

        rg = RadioGroup(options=["X", "Y"])

        changed = rg.select(0)
        assert changed is True
        assert rg.selected_index == 0

        changed = rg.select(0)  # Same selection
        assert changed is False

        changed = rg.select(99)  # Invalid index
        assert changed is False

    def test_radio_group_serialization(self):
        """Test radio group serialization."""
        from noodlestudio.runtime.ui.components import RadioGroup

        rg = RadioGroup(name="rg", options=["Opt1", "Opt2"])
        rg.selected_index = 1
        rg.orientation = "horizontal"

        data = rg.to_dict()
        assert data["options"] == ["Opt1", "Opt2"]
        assert data["selected_index"] == 1
        assert data["orientation"] == "horizontal"

    def test_radio_group_render(self, qapp):
        """Test radio group Qt rendering."""
        from noodlestudio.runtime.ui.components import Panel, RadioGroup
        from noodlestudio.runtime.ui import QtWidgetRenderer

        root = Panel(name="root")
        rg = RadioGroup(name="rg", options=["Yes", "No"])
        rg.selected_index = 0
        root.add_child(rg)

        renderer = QtWidgetRenderer()
        renderer.render(root)

        widget = renderer.get_widget("rg")
        assert widget is not None


class TestNewComponentsInYAML:
    """Test new components can be loaded from YAML."""

    def test_checkbox_yaml_round_trip(self):
        """Test checkbox survives YAML serialization."""
        import yaml
        from noodlestudio.runtime.ui.components import Checkbox

        cb = Checkbox(name="yaml_cb", text="YAML Test", checked=True)
        data = cb.to_dict()

        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        cb2 = Checkbox.from_dict(loaded_data)
        assert cb2.name == "yaml_cb"
        assert cb2.text == "YAML Test"
        assert cb2.checked is True

    def test_dropdown_yaml_round_trip(self):
        """Test dropdown survives YAML serialization."""
        import yaml
        from noodlestudio.runtime.ui.components import Dropdown

        dd = Dropdown(name="dd", options=["A", "B", "C"], selected_index=2)
        data = dd.to_dict()

        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        dd2 = Dropdown.from_dict(loaded_data)
        assert dd2.options == ["A", "B", "C"]
        assert dd2.selected_index == 2
        assert dd2.value == "C"

    def test_slider_yaml_round_trip(self):
        """Test slider survives YAML serialization."""
        import yaml
        from noodlestudio.runtime.ui.components import Slider

        slider = Slider(name="s", value=75, min_value=0, max_value=100)
        slider.step = 25
        data = slider.to_dict()

        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        slider2 = Slider.from_dict(loaded_data)
        assert slider2.value == 75
        assert slider2.step == 25
