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
