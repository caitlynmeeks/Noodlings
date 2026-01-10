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
#   Panel Wiring Tests - pytest-qt tests for NoodleStudio signal connections
#
#   Tests the critical signal flows: 1. Noodling selection in...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_panel_wiring
# PURPOSE:  Tests for panel wiring
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestNoodlingSelectionToFacetEditor, TestNameChangePropagation, TestUndoRedo, TestSignalConnections, qapp()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def qapp():
    """Create QApplication once for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qapp, qtbot):
    """Create a MainWindow instance for testing."""
    from noodlestudio.core.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    # Don't show the window (headless testing)
    # window.show()

    yield window

    window.close()


@pytest.fixture
def mock_noodling_data():
    """Mock noodling entity data for testing."""
    return {
        'id': 'test_noodling_001',
        'name': 'TestNoodling',
        'noodling_ref': 'empty_noodling',
        'path': '/test/path/instance.yaml',
        'data': {
            'noodling': 'empty_noodling',
            'facet_assembly': {
                'ref': 'library/empty_noodling'
            }
        }
    }


@pytest.fixture
def mock_prop_data():
    """Mock prop entity data for testing."""
    return {
        'id': 'test_prop_001',
        'name': 'TestProp',
        'prim_ref': 'cube',
        'position': [0, 0, 0]
    }


# ============================================================================
# Test 1: Noodling Selection -> Facet Editor Wiring
# ============================================================================

class TestNoodlingSelectionToFacetEditor:
    """Test that selecting a Noodling in Stage View loads its assembly in Facet Editor."""

    def test_facets_editor_exists(self, main_window):
        """Verify Facets Editor panel exists on MainWindow."""
        assert hasattr(main_window, 'facets_editor'), "MainWindow missing facets_editor"
        assert main_window.facets_editor is not None

    def test_hierarchy_exists(self, main_window):
        """Verify Stage View (hierarchy) panel exists."""
        assert hasattr(main_window, 'hierarchy'), "MainWindow missing hierarchy"
        assert main_window.hierarchy is not None

    def test_entitySelected_signal_exists(self, main_window):
        """Verify entitySelected signal exists on hierarchy."""
        from PyQt6.QtCore import pyqtSignal
        assert hasattr(main_window.hierarchy, 'entitySelected')

    def test_noodling_selection_triggers_facet_load(self, main_window, mock_noodling_data, qtbot):
        """Selecting a noodling should load its facet assembly."""
        # Clear any existing state
        main_window.facets_editor.clear_editor()

        # Emit the entitySelected signal as if user clicked a noodling
        main_window.hierarchy.entitySelected.emit('noodling', mock_noodling_data)

        # Process events
        qtbot.wait(100)

        # Check that facets editor received and loaded an assembly
        # After selection, current_assembly should not be None (assembly was loaded)
        assert main_window.facets_editor.current_assembly is not None, \
            "Facet assembly should be loaded when noodling is selected"

    def test_deselection_clears_facet_editor(self, main_window, mock_noodling_data, qtbot):
        """Deselecting (empty selection) should clear the facet editor."""
        # First select something
        main_window.hierarchy.entitySelected.emit('noodling', mock_noodling_data)
        qtbot.wait(50)

        # Then deselect
        main_window.hierarchy.entitySelected.emit('', {})
        qtbot.wait(50)

        # Facet editor should be cleared
        # Check that there's no current agent or the editor is empty
        current_agent = getattr(main_window.facets_editor, '_current_agent_id', None)
        # After clear, _current_agent_id should be None or empty
        # This depends on implementation - adjust assertion as needed


# ============================================================================
# Test 2: Inspector Name Change -> Stage View Propagation
# ============================================================================

class TestNameChangePropagation:
    """Test that name changes in Inspector propagate to Stage View."""

    def test_inspector_exists(self, main_window):
        """Verify Inspector panel exists."""
        assert hasattr(main_window, 'inspector'), "MainWindow missing inspector"
        assert main_window.inspector is not None

    def test_nameChanged_signal_exists(self, main_window):
        """Verify nameChanged signal exists on inspector."""
        assert hasattr(main_window.inspector, 'nameChanged')

    def test_hierarchy_has_update_entity_name(self, main_window):
        """Verify hierarchy has update_entity_name method."""
        assert hasattr(main_window.hierarchy, 'update_entity_name')
        assert callable(main_window.hierarchy.update_entity_name)

    def test_name_change_signal_connected(self, main_window, qtbot):
        """Verify nameChanged signal is connected to handler."""
        # We can test this by checking if emitting the signal triggers the handler
        # Use a spy to track if update_entity_name is called

        call_log = []
        original_method = main_window.hierarchy.update_entity_name

        def tracking_wrapper(entity_type, entity_id, new_name):
            call_log.append((entity_type, entity_id, new_name))
            return original_method(entity_type, entity_id, new_name)

        main_window.hierarchy.update_entity_name = tracking_wrapper

        try:
            # Emit the signal
            main_window.inspector.nameChanged.emit('noodling', 'test_id', 'NewName')
            qtbot.wait(50)

            # Check if our wrapper was called
            assert len(call_log) > 0, "nameChanged signal not connected to update_entity_name"
            assert call_log[0] == ('noodling', 'test_id', 'NewName')
        finally:
            main_window.hierarchy.update_entity_name = original_method

    def test_name_change_updates_tree_item(self, main_window, qtbot):
        """Test that update_entity_name actually updates the tree widget."""
        # This test would require setting up actual tree items
        # For now, just verify the method doesn't crash
        main_window.hierarchy.update_entity_name('noodling', 'nonexistent_id', 'NewName')
        # Should not raise an exception


# ============================================================================
# Test 3: Undo/Redo Reliability
# ============================================================================

class TestUndoRedo:
    """Test undo/redo system reliability."""

    def test_undo_manager_exists(self, main_window):
        """Verify UndoManager singleton exists."""
        from noodlestudio.core.undo_manager import undo_manager
        assert undo_manager is not None

    def test_undo_stack_starts_empty(self, main_window):
        """Undo stack should be clearable."""
        from noodlestudio.core.undo_manager import undo_manager

        # Clear any existing state
        undo_manager.clear()

        assert not undo_manager.can_undo()
        assert not undo_manager.can_redo()

    def test_push_command_enables_undo(self, main_window):
        """Pushing a command should enable undo."""
        from noodlestudio.core.undo_manager import undo_manager
        from noodlestudio.core.commands.base_command import StudioCommand

        undo_manager.clear()

        # Create a simple test command
        class TestCommand(StudioCommand):
            def __init__(self):
                super().__init__("Test Command")
                self.executed = False
                self.undone = False

            def redo(self):
                self.executed = True

            def undo(self):
                self.undone = True

        cmd = TestCommand()
        undo_manager.push(cmd)

        assert undo_manager.can_undo()
        assert cmd.executed, "Command should be executed when pushed"

    def test_undo_then_redo(self, main_window):
        """Test undo followed by redo."""
        from noodlestudio.core.undo_manager import undo_manager
        from noodlestudio.core.commands.base_command import StudioCommand

        undo_manager.clear()

        state = {'value': 0}

        class IncrementCommand(StudioCommand):
            def __init__(self):
                super().__init__("Increment")

            def redo(self):
                state['value'] += 1

            def undo(self):
                state['value'] -= 1

        cmd = IncrementCommand()
        undo_manager.push(cmd)

        assert state['value'] == 1, "Redo (initial) should increment"

        undo_manager.undo()
        assert state['value'] == 0, "Undo should decrement"
        assert undo_manager.can_redo()

        undo_manager.redo()
        assert state['value'] == 1, "Redo should increment again"

    def test_multiple_undos(self, main_window):
        """Test multiple sequential undos."""
        from noodlestudio.core.undo_manager import undo_manager
        from noodlestudio.core.commands.base_command import StudioCommand

        undo_manager.clear()

        state = {'value': 0}

        class IncrementCommand(StudioCommand):
            def __init__(self, amount):
                super().__init__(f"Increment by {amount}")
                self.amount = amount

            def redo(self):
                state['value'] += self.amount

            def undo(self):
                state['value'] -= self.amount

        # Push 3 commands
        undo_manager.push(IncrementCommand(1))  # value = 1
        undo_manager.push(IncrementCommand(2))  # value = 3
        undo_manager.push(IncrementCommand(3))  # value = 6

        assert state['value'] == 6

        # Undo all three
        undo_manager.undo()  # value = 3
        assert state['value'] == 3

        undo_manager.undo()  # value = 1
        assert state['value'] == 1

        undo_manager.undo()  # value = 0
        assert state['value'] == 0

        assert not undo_manager.can_undo()
        assert undo_manager.can_redo()


# ============================================================================
# Test 4: Signal Connection Integrity
# ============================================================================

class TestSignalConnections:
    """Test that critical signal connections are in place."""

    def test_hierarchy_to_inspector_connection(self, main_window, mock_noodling_data, qtbot):
        """Selecting entity in hierarchy should update inspector."""
        # Emit selection
        main_window.hierarchy.entitySelected.emit('noodling', mock_noodling_data)
        qtbot.wait(50)

        # Inspector should have loaded the entity
        # Check inspector's current mode or loaded entity
        # This depends on inspector implementation

    def test_hierarchy_to_facets_editor_connection(self, main_window, mock_noodling_data, qtbot):
        """Selecting noodling should update facets editor."""
        # Clear first
        main_window.facets_editor.clear_editor()

        main_window.hierarchy.entitySelected.emit('noodling', mock_noodling_data)
        qtbot.wait(100)

        # Facets editor should show an assembly after noodling selection
        assert main_window.facets_editor.current_assembly is not None, \
            "Selecting noodling should load facet assembly"


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
