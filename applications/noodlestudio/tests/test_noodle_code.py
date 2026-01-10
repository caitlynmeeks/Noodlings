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
#   Tests for Noodle Code AI assistant system.
#
#   Covers: - NoodleCodeTools execution - ComputerUseControll...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_noodle_code
# PURPOSE:  Tests for Noodle Code AI assistant system.
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestNoodleCodeTools, TestNoodleCodeToolDefinitions, TestComputerUseController, TestNoodleCodeEngine, TestModelLabelRouting
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import os
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import test fixtures
from conftest import qapp


def run_async(coro):
    """Helper to run async functions in sync tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestNoodleCodeTools:
    """Tests for NoodleCodeTools."""

    @pytest.fixture
    def tools(self, tmp_path):
        """Create NoodleCodeTools instance with temp project."""
        from noodlestudio.core.noodle_code_tools import NoodleCodeTools
        tools = NoodleCodeTools(tmp_path)
        return tools

    def test_read_file(self, tools, tmp_path):
        """Test read_file tool."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        result = run_async(tools.tool_read_file("test.txt"))
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output

    def test_read_file_not_found(self, tools):
        """Test read_file with missing file."""
        result = run_async(tools.tool_read_file("nonexistent.txt"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_write_file(self, tools, tmp_path):
        """Test write_file tool."""
        result = run_async(tools.tool_write_file("new_file.txt", "Hello World"))
        assert result.success

        # Verify file was created
        assert (tmp_path / "new_file.txt").exists()
        assert (tmp_path / "new_file.txt").read_text() == "Hello World"

    def test_edit_file(self, tools, tmp_path):
        """Test edit_file tool."""
        # Create file to edit
        test_file = tmp_path / "edit_me.txt"
        test_file.write_text("Hello World")

        result = run_async(tools.tool_edit_file("edit_me.txt", "World", "Universe"))
        assert result.success
        assert test_file.read_text() == "Hello Universe"

    def test_edit_file_not_found(self, tools):
        """Test edit_file with string not in file."""
        result = run_async(tools.tool_edit_file("nonexistent.txt", "old", "new"))
        assert not result.success

    def test_glob(self, tools, tmp_path):
        """Test glob tool."""
        # Create some files
        (tmp_path / "file1.py").write_text("# Python")
        (tmp_path / "file2.py").write_text("# Python")
        (tmp_path / "file3.txt").write_text("Text")

        result = run_async(tools.tool_glob("*.py"))
        assert result.success
        assert "file1.py" in result.output
        assert "file2.py" in result.output
        assert "file3.txt" not in result.output

    def test_grep(self, tools, tmp_path):
        """Test grep tool."""
        # Create files with content
        (tmp_path / "search_me.txt").write_text("The quick brown fox\njumps over the lazy dog")

        result = run_async(tools.tool_grep("quick"))
        assert result.success
        assert "quick" in result.output

    def test_bash(self, tools):
        """Test bash tool."""
        result = run_async(tools.tool_bash("echo 'Hello from bash'"))
        assert result.success
        assert "Hello from bash" in result.output

    def test_bash_timeout(self, tools):
        """Test bash tool timeout."""
        result = run_async(tools.tool_bash("sleep 10", timeout=1))
        assert not result.success
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    def test_list_directory(self, tools, tmp_path):
        """Test list_directory tool."""
        # Create some files/folders
        (tmp_path / "folder").mkdir()
        (tmp_path / "file.txt").write_text("content")

        result = run_async(tools.tool_list_directory())
        assert result.success
        assert "folder" in result.output
        assert "file.txt" in result.output


class TestNoodleCodeToolDefinitions:
    """Test tool definitions are properly formatted."""

    def test_all_tools_have_definitions(self):
        """Verify all tools are defined."""
        from noodlestudio.core.noodle_code_tools import NoodleCodeTools
        tools = NoodleCodeTools(None)
        definitions = tools.get_tool_definitions()

        tool_names = [d["name"] for d in definitions]

        # Check essential tools are defined
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names
        assert "bash" in tool_names
        assert "list_directory" in tool_names
        assert "hot_reload" in tool_names
        assert "soft_restart" in tool_names
        assert "computer_use" in tool_names
        assert "github" in tool_names

    def test_tool_definitions_have_required_fields(self):
        """Verify tool definitions have name, description, input_schema."""
        from noodlestudio.core.noodle_code_tools import NoodleCodeTools
        tools = NoodleCodeTools(None)
        definitions = tools.get_tool_definitions()

        for defn in definitions:
            assert "name" in defn, f"Tool missing 'name'"
            assert "description" in defn, f"Tool {defn.get('name')} missing 'description'"
            assert "input_schema" in defn, f"Tool {defn.get('name')} missing 'input_schema'"


class TestComputerUseController:
    """Tests for ComputerUseController."""

    def test_controller_singleton(self):
        """Test controller is singleton."""
        from noodlestudio.core.computer_use_controller import get_computer_use_controller
        c1 = get_computer_use_controller()
        c2 = get_computer_use_controller()
        assert c1 is c2

    def test_screenshot_without_window(self):
        """Test screenshot fails gracefully without main window."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        controller = ComputerUseController()
        # Don't set main_window - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Main window not set"):
            controller.screenshot()


class TestNoodleCodeEngine:
    """Tests for NoodleCodeEngine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with temp project."""
        from noodlestudio.core.noodle_code_engine import NoodleCodeEngine
        engine = NoodleCodeEngine(
            model_label_manager=None,
            provider_manager=None,
            project_path=tmp_path
        )
        return engine

    def test_build_system_prompt_with_project(self, engine, tmp_path):
        """Test system prompt includes project context."""
        prompt = engine._build_system_prompt()
        assert str(tmp_path.name) in prompt or str(tmp_path) in prompt

    def test_build_system_prompt_without_project(self):
        """Test system prompt without project."""
        from noodlestudio.core.noodle_code_engine import NoodleCodeEngine
        engine = NoodleCodeEngine(project_path=None)
        prompt = engine._build_system_prompt()
        assert "No project" in prompt

    def test_noodle_code_md_loading(self, engine, tmp_path):
        """Test NOODLE_CODE.md is loaded into prompt."""
        # Create NOODLE_CODE.md
        noodle_code_path = tmp_path / "NOODLE_CODE.md"
        noodle_code_path.write_text("## Custom Project Context\nThis is a test project.")

        prompt = engine._build_system_prompt()
        assert "Custom Project Context" in prompt

    def test_history_management(self, engine):
        """Test conversation history."""
        from noodlestudio.core.noodle_code_engine import Message

        assert len(engine.get_history()) == 0

        # Add message
        engine.history.append(Message(role="user", content="test"))
        assert len(engine.get_history()) == 1

        # Clear
        engine.clear_history()
        assert len(engine.get_history()) == 0


class TestModelLabelRouting:
    """Tests for model label routing."""

    def test_protected_labels(self):
        """Test protected labels include Noodle Code and Computer Use."""
        from noodlestudio.core.model_label_manager import ModelLabelManager
        manager = ModelLabelManager()

        assert manager.is_protected_label("Noodle Code")
        assert manager.is_protected_label("Computer Use")
        assert manager.is_protected_label("Small")
        assert manager.is_protected_label("Medium")
        assert manager.is_protected_label("Large")

    def test_get_noodle_code_model(self):
        """Test get_noodle_code_model helper."""
        from noodlestudio.core.model_label_manager import ModelLabelManager
        manager = ModelLabelManager()

        # Returns tuple
        result = manager.get_noodle_code_model()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_computer_use_model(self):
        """Test get_computer_use_model helper."""
        from noodlestudio.core.model_label_manager import ModelLabelManager
        manager = ModelLabelManager()

        result = manager.get_computer_use_model()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestProjectNoodleCodeMd:
    """Tests for NOODLE_CODE.md auto-generation."""

    def test_create_project_generates_noodle_code_md(self, tmp_path):
        """Test new projects get NOODLE_CODE.md."""
        from noodlestudio.core.project_manager import ProjectManager
        manager = ProjectManager()

        project_name = "TestProject"
        success = manager.create_project(str(tmp_path), project_name, "Test description")

        if success:
            project_path = tmp_path / project_name
            noodle_code_path = project_path / "NOODLE_CODE.md"
            assert noodle_code_path.exists()

            content = noodle_code_path.read_text()
            assert project_name in content
            assert "Test description" in content or "NoodleStudio project" in content

    def test_noodle_code_md_has_required_sections(self, tmp_path):
        """Test generated NOODLE_CODE.md has required sections."""
        from noodlestudio.core.project_manager import ProjectManager
        manager = ProjectManager()

        success = manager.create_project(str(tmp_path), "TestProject")

        if success:
            noodle_code_path = tmp_path / "TestProject" / "NOODLE_CODE.md"
            content = noodle_code_path.read_text()

            assert "Project Info" in content
            assert "Project Structure" in content
            assert "Project-Specific Context" in content
            assert "Tips for Noodle Code" in content


class TestGitHubTool:
    """Tests for GitHub CLI tool."""

    def test_github_tool_security(self, tmp_path):
        """Test GitHub tool blocks shell injection."""
        from noodlestudio.core.noodle_code_tools import NoodleCodeTools
        tools = NoodleCodeTools(tmp_path)

        # Try shell injection
        result = run_async(tools.tool_github("issue list; rm -rf /"))
        assert not result.success
        assert "Invalid characters" in result.error

    def test_github_tool_blocks_operators(self, tmp_path):
        """Test GitHub tool blocks various shell operators."""
        from noodlestudio.core.noodle_code_tools import NoodleCodeTools
        tools = NoodleCodeTools(tmp_path)

        dangerous = ["issue list | cat", "issue list && echo", "$(whoami)", "`id`"]
        for cmd in dangerous:
            result = run_async(tools.tool_github(cmd))
            assert not result.success, f"Should block: {cmd}"


class TestNoodleCodePanelUI:
    """Tests for NoodleCodePanel UI components."""

    @pytest.fixture
    def panel(self, qapp):
        """Create panel for testing."""
        from noodlestudio.panels.noodle_code_panel import NoodleCodePanel
        return NoodleCodePanel()

    def test_font_size_initialization(self, panel):
        """Test font size is initialized."""
        assert hasattr(panel, 'font_size')
        assert 8 <= panel.font_size <= 36

    def test_font_size_increase(self, panel):
        """Test increase_font_size."""
        initial = panel.font_size
        panel.increase_font_size()
        if initial < 36:
            assert panel.font_size == initial + 2

    def test_font_size_decrease(self, panel):
        """Test decrease_font_size."""
        panel.font_size = 14  # Start from known value
        panel.decrease_font_size()
        assert panel.font_size == 12

    def test_font_size_bounds(self, panel):
        """Test font size stays in bounds."""
        # Test lower bound
        panel.font_size = 8
        panel.decrease_font_size()
        assert panel.font_size == 8  # Should not go below 8

        # Test upper bound
        panel.font_size = 36
        panel.increase_font_size()
        assert panel.font_size == 36  # Should not go above 36

    def test_chat_view_exists(self, panel):
        """Test chat view QTextEdit exists."""
        assert hasattr(panel, 'chat_view')
        from PyQt6.QtWidgets import QTextEdit
        assert isinstance(panel.chat_view, QTextEdit)

    def test_input_history_initialized(self, panel):
        """Test input history list is initialized."""
        assert hasattr(panel, 'input_history')
        assert isinstance(panel.input_history, list)
        assert hasattr(panel, 'history_index')
        assert panel.history_index == -1

    def test_thinking_indicator_exists(self, panel):
        """Test thinking indicator is initialized."""
        assert hasattr(panel, 'thinking_indicator')
        from noodlestudio.panels.noodle_code_panel import ThinkingIndicator
        assert isinstance(panel.thinking_indicator, ThinkingIndicator)

    def test_chat_history_tracking(self, panel):
        """Test chat history list exists."""
        assert hasattr(panel, '_chat_history')
        assert isinstance(panel._chat_history, list)

    def test_stop_mode_toggle(self, panel):
        """Test stop mode can be toggled."""
        assert hasattr(panel, '_is_stop_mode')
        assert panel._is_stop_mode == False
        panel._set_stop_mode(True)
        assert panel._is_stop_mode == True
        assert panel.send_button.text() == "Stop"
        panel._set_stop_mode(False)
        assert panel._is_stop_mode == False
        assert panel.send_button.text() == "Send"

    def test_demo_mode_button_exists(self, panel):
        """Test demo mode toggle button is initialized."""
        assert hasattr(panel, 'demo_mode_btn')
        assert panel.demo_mode_btn.isCheckable()
        assert "Demo mode" in panel.demo_mode_btn.toolTip()


class TestNoodleCodeProfiles:
    """Tests for Noodle Code personality profiles."""

    def test_profile_manager_singleton(self):
        """Test profile manager is singleton."""
        from noodlestudio.core.noodle_code_profiles import get_profile_manager
        m1 = get_profile_manager()
        m2 = get_profile_manager()
        assert m1 is m2

    def test_default_profile_exists(self):
        """Test default profile is always available."""
        from noodlestudio.core.noodle_code_profiles import get_profile_manager
        manager = get_profile_manager()

        assert "default" in manager.get_profile_names()
        profile = manager.get_profile("default")
        assert profile is not None

    def test_builtin_profiles_loaded(self):
        """Test built-in profiles are loaded."""
        from noodlestudio.core.noodle_code_profiles import get_profile_manager
        manager = get_profile_manager()

        names = manager.get_profile_names()
        assert "default" in names
        assert "creative" in names
        assert "architect" in names
        assert "reviewer" in names

    def test_set_current_profile(self):
        """Test setting current profile."""
        from noodlestudio.core.noodle_code_profiles import get_profile_manager
        manager = get_profile_manager()

        # Set to architect
        assert manager.set_current_profile("architect")
        assert manager.current_profile_name == "architect"

        # Set back to default
        manager.set_current_profile("default")

    def test_get_profile_prompt(self):
        """Test getting profile prompt content."""
        from noodlestudio.core.noodle_code_profiles import get_profile_manager
        manager = get_profile_manager()

        prompt = manager.get_profile_prompt("creative")
        assert "Creative" in prompt or "creative" in prompt.lower()
        assert len(prompt) > 0


class TestGhostCursor:
    """Tests for ghost cursor visualization system."""

    def test_cursor_animation_bezier(self):
        """Test bezier curve interpolation."""
        from noodlestudio.core.ghost_cursor import CursorAnimation
        from PyQt6.QtCore import QPointF

        anim = CursorAnimation(
            start=QPointF(0, 0),
            end=QPointF(100, 100),
            control1=QPointF(30, 50),
            control2=QPointF(70, 50),
            duration_ms=500,
            start_time=0.0
        )

        # Test start position
        pos = anim.position_at(0.0)
        assert abs(pos.x() - 0) < 0.01
        assert abs(pos.y() - 0) < 0.01

        # Test end position
        pos = anim.position_at(1.0)
        assert abs(pos.x() - 100) < 0.01
        assert abs(pos.y() - 100) < 0.01

        # Test midpoint is somewhere along the curve
        pos = anim.position_at(0.5)
        assert 0 < pos.x() < 100
        assert 0 < pos.y() < 100

    def test_cursor_animation_easing(self):
        """Test ease-in-out cubic easing."""
        from noodlestudio.core.ghost_cursor import CursorAnimation
        from PyQt6.QtCore import QPointF

        anim = CursorAnimation(
            start=QPointF(0, 0),
            end=QPointF(100, 0),
            control1=QPointF(33, 0),
            control2=QPointF(66, 0),
            duration_ms=1000,
            start_time=0.0
        )

        # Ease-in-out starts slow
        t1 = anim._ease_in_out_cubic(0.1)
        t2 = anim._ease_in_out_cubic(0.2)
        # Progress should be slower at start (less than linear)
        assert t1 < 0.1
        assert t2 - t1 < 0.1  # Still accelerating

        # Midpoint
        t_mid = anim._ease_in_out_cubic(0.5)
        assert abs(t_mid - 0.5) < 0.01  # At 0.5 the easing should be at 0.5

    def test_click_ripple_progress(self):
        """Test click ripple animation progress."""
        from noodlestudio.core.ghost_cursor import ClickRipple
        from PyQt6.QtCore import QPointF

        ripple = ClickRipple(
            center=QPointF(50, 50),
            start_time=0.0,
            duration_ms=400
        )

        # At start
        assert ripple.progress(0.0) == 0.0
        assert not ripple.is_finished(0.0)

        # At end
        assert ripple.progress(0.4) >= 1.0
        assert ripple.is_finished(0.4)

    def test_ghost_overlay_creation(self, qapp):
        """Test ghost overlay widget can be created."""
        from noodlestudio.core.ghost_cursor import GhostCursorOverlay
        from PyQt6.QtWidgets import QWidget

        parent = QWidget()
        overlay = GhostCursorOverlay(parent)

        assert overlay is not None
        assert not overlay.is_enabled
        assert not overlay._cursor_visible

    def test_ghost_controller_demo_mode(self, qapp):
        """Test ghost controller demo mode toggle."""
        from noodlestudio.core.ghost_cursor import GhostCursorOverlay, GhostCursorController
        from PyQt6.QtWidgets import QWidget

        parent = QWidget()
        overlay = GhostCursorOverlay(parent)
        controller = GhostCursorController(overlay)

        assert not controller.demo_mode
        controller.set_demo_mode(True)
        assert controller.demo_mode
        assert overlay.is_enabled


class TestUIElementMap:
    """Tests for UI Element Map functionality."""

    def test_get_ui_element_map_without_window(self):
        """Test UI element map returns empty without window."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        controller = ComputerUseController()
        # No main window set
        elements = controller.get_ui_element_map()
        assert elements == []

    def test_get_ui_summary_without_window(self):
        """Test UI summary returns message without window."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        controller = ComputerUseController()
        summary = controller.get_ui_summary()
        assert "No UI elements found" in summary

    def test_get_ui_element_map_with_window(self, qapp):
        """Test UI element map with main window."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        from PyQt6.QtWidgets import QMainWindow, QPushButton, QTabWidget, QWidget

        # Create a simple main window with some widgets
        main_window = QMainWindow()
        central = QWidget()
        main_window.setCentralWidget(central)

        # Add a button
        btn = QPushButton("Test Button", central)
        btn.setGeometry(50, 50, 100, 30)

        main_window.show()
        main_window.resize(400, 300)

        controller = ComputerUseController()
        controller.set_main_window(main_window)

        elements = controller.get_ui_element_map()

        # Should find the button
        button_elements = [e for e in elements if e["type"] == "button"]
        assert len(button_elements) >= 1

        # Check element has required fields
        for elem in elements:
            assert "name" in elem
            assert "type" in elem
            assert "x" in elem
            assert "y" in elem
            assert "bounds" in elem

        main_window.close()

    def test_ui_summary_format(self, qapp):
        """Test UI summary is properly formatted."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        from PyQt6.QtWidgets import QMainWindow, QPushButton, QWidget

        main_window = QMainWindow()
        central = QWidget()
        main_window.setCentralWidget(central)

        btn = QPushButton("Click Me", central)
        btn.setGeometry(50, 50, 100, 30)

        main_window.show()
        main_window.resize(400, 300)

        controller = ComputerUseController()
        controller.set_main_window(main_window)

        summary = controller.get_ui_summary()

        # Summary should include headers
        assert "CLICKABLE UI ELEMENTS" in summary
        # Should have coordinates format
        assert "->" in summary
        assert "(" in summary and ")" in summary

        main_window.close()

    def test_tab_detection(self, qapp):
        """Test tabs are detected in UI element map."""
        from noodlestudio.core.computer_use_controller import ComputerUseController
        from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget

        main_window = QMainWindow()
        tabs = QTabWidget()
        tabs.addTab(QWidget(), "Tab One")
        tabs.addTab(QWidget(), "Tab Two")
        tabs.addTab(QWidget(), "Tab Three")
        main_window.setCentralWidget(tabs)

        main_window.show()
        main_window.resize(400, 300)

        controller = ComputerUseController()
        controller.set_main_window(main_window)

        elements = controller.get_ui_element_map()
        tab_elements = [e for e in elements if e["type"] == "tab"]

        # Should find all three tabs
        assert len(tab_elements) >= 3
        tab_names = [e["name"] for e in tab_elements]
        assert any("Tab One" in name for name in tab_names)
        assert any("Tab Two" in name for name in tab_names)

        main_window.close()


class TestChatHistory:
    """Tests for chat history persistence."""

    def test_history_file_path(self):
        """Test history file path is correct."""
        from noodlestudio.panels.noodle_code_panel import NoodleCodePanel
        from pathlib import Path

        expected = Path.home() / ".noodlestudio" / "noodlecode_history.json"
        assert NoodleCodePanel.HISTORY_FILE == expected

    def test_chat_history_max_limit(self):
        """Test max history messages constant exists."""
        from noodlestudio.panels.noodle_code_panel import NoodleCodePanel
        assert hasattr(NoodleCodePanel, 'MAX_HISTORY_MESSAGES')
        assert NoodleCodePanel.MAX_HISTORY_MESSAGES > 0


class TestImageInToolResult:
    """Tests for image handling in tool results."""

    def test_message_has_image_field(self):
        """Test Message dataclass has image_base64 field."""
        from noodlestudio.core.noodle_code_engine import Message

        msg = Message(
            role="tool_result",
            content="Screenshot taken",
            image_base64="base64data"
        )
        assert msg.image_base64 == "base64data"

    def test_message_image_defaults_none(self):
        """Test Message image_base64 defaults to None."""
        from noodlestudio.core.noodle_code_engine import Message

        msg = Message(role="user", content="test")
        assert msg.image_base64 is None

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
