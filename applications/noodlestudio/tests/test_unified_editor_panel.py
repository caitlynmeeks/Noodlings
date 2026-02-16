"""Tests for UnifiedEditorPanel depth-stack navigation."""

import pytest
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent

from noodlestudio.panels.editors.unified_editor_panel import UnifiedEditorPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class StubDepthView(QWidget):
    """Minimal depth view for testing the stack."""

    def __init__(self, label="stub", parent=None):
        super().__init__(parent)
        self._label = label
        self._saved = False
        self._loaded_path = None
        self._loaded_context = None
        self._has_changes = False
        # Give it visible content so it has a size
        self._content = QLabel(label, self)

    def load_data(self, data_path, context):
        self._loaded_path = data_path
        self._loaded_context = context

    def save_data(self):
        self._saved = True

    def get_breadcrumb_label(self):
        return self._label

    def has_unsaved_changes(self):
        return self._has_changes


@pytest.fixture
def panel(qapp):
    """Fresh UnifiedEditorPanel for each test."""
    p = UnifiedEditorPanel()
    p.show()
    # Clear class-level registry to avoid cross-test pollution
    UnifiedEditorPanel._depth_view_registry.clear()
    yield p
    p.close()


class TestStackOperations:
    """Test push, pop, depth, clear."""

    def test_initial_state_empty(self, panel):
        assert panel.depth() == 0
        assert panel.current_view() is None

    def test_push_view(self, panel):
        view = StubDepthView("assembly")
        panel.push_view(view, "assembly")
        assert panel.depth() == 1
        assert panel.current_view() is view

    def test_push_two_views(self, panel):
        v1 = StubDepthView("assembly")
        v2 = StubDepthView("charm network")
        panel.push_view(v1, "assembly")
        panel.push_view(v2, "Charm Network")
        assert panel.depth() == 2
        assert panel.current_view() is v2
        # First view should be hidden
        assert not v1.isVisible()
        assert v2.isVisible()

    def test_pop_one(self, panel):
        v1 = StubDepthView("assembly")
        v2 = StubDepthView("charm network")
        panel.push_view(v1, "assembly")
        panel.push_view(v2, "Charm Network")
        panel.pop_one()
        assert panel.depth() == 1
        assert panel.current_view() is v1
        assert v1.isVisible()

    def test_pop_one_at_root_is_noop(self, panel):
        v1 = StubDepthView("assembly")
        panel.push_view(v1, "assembly")
        panel.pop_one()
        # Should stay at depth 1 (root), not pop to empty
        assert panel.depth() == 1
        assert panel.current_view() is v1

    def test_pop_one_when_empty_is_noop(self, panel):
        panel.pop_one()
        assert panel.depth() == 0

    def test_pop_to_index(self, panel):
        views = []
        for i in range(4):
            v = StubDepthView(f"level-{i}")
            panel.push_view(v, f"Level {i}")
            views.append(v)

        assert panel.depth() == 4
        panel.pop_to(1)
        assert panel.depth() == 2
        assert panel.current_view() is views[1]

    def test_pop_to_saves_popped_views(self, panel):
        views = []
        for i in range(3):
            v = StubDepthView(f"level-{i}")
            panel.push_view(v, f"Level {i}")
            views.append(v)

        panel.pop_to(0)
        # Views at index 1 and 2 were popped, should have been saved
        assert views[1]._saved is True
        assert views[2]._saved is True
        # Root view (index 0) should NOT have been saved by the pop
        assert views[0]._saved is False

    def test_clear_stack(self, panel):
        for i in range(3):
            panel.push_view(StubDepthView(f"level-{i}"), f"Level {i}")

        assert panel.depth() == 3
        panel.clear_stack()
        assert panel.depth() == 0
        assert panel.current_view() is None

    def test_clear_stack_saves_all(self, panel):
        views = []
        for i in range(3):
            v = StubDepthView(f"level-{i}")
            panel.push_view(v, f"Level {i}")
            views.append(v)

        panel.clear_stack()
        for v in views:
            assert v._saved is True

    def test_save_all(self, panel):
        views = []
        for i in range(3):
            v = StubDepthView(f"level-{i}")
            panel.push_view(v, f"Level {i}")
            views.append(v)

        panel.save_all()
        for v in views:
            assert v._saved is True


class TestBreadcrumbIntegration:
    """Test breadcrumb bar updates with stack operations."""

    def test_empty_breadcrumb(self, panel):
        assert panel._breadcrumb.segment_count() == 0

    def test_single_view_breadcrumb(self, panel):
        panel.push_view(StubDepthView("assembly"), "assembly")
        assert panel._breadcrumb.segment_count() == 1

    def test_multi_level_breadcrumb(self, panel):
        panel.push_view(StubDepthView("ajo"), "Ajo Majo")
        panel.push_view(StubDepthView("assembly"), "assembly")
        panel.push_view(StubDepthView("charm"), "Charm Network")
        assert panel._breadcrumb.segment_count() == 3

    def test_breadcrumb_updates_on_pop(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        panel.push_view(StubDepthView("b"), "Level 1")
        panel.push_view(StubDepthView("c"), "Level 2")
        assert panel._breadcrumb.segment_count() == 3
        panel.pop_one()
        assert panel._breadcrumb.segment_count() == 2

    def test_breadcrumb_click_pops_to_index(self, panel, qtbot):
        views = []
        for i in range(3):
            v = StubDepthView(f"level-{i}")
            panel.push_view(v, f"Level {i}")
            views.append(v)

        assert panel.depth() == 3

        # Click the first breadcrumb segment (index 0)
        with qtbot.waitSignal(panel.depthChanged, timeout=1000):
            panel._breadcrumb.segmentClicked.emit(0)

        assert panel.depth() == 1
        assert panel.current_view() is views[0]

    def test_breadcrumb_clears_on_clear_stack(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        panel.push_view(StubDepthView("b"), "Level 1")
        panel.clear_stack()
        assert panel._breadcrumb.segment_count() == 0


class TestDepthViewRegistry:
    """Test class-level depth view registry and dispatch."""

    def test_register_and_lookup(self, panel):
        UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", StubDepthView)
        cls = UnifiedEditorPanel.get_registered_view_class("NeuralCanvasFacet")
        assert cls is StubDepthView

    def test_lookup_unknown_returns_none(self, panel):
        cls = UnifiedEditorPanel.get_registered_view_class("NonexistentFacet")
        assert cls is None

    def test_on_double_click_container_pushes_view(self, panel):
        UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", StubDepthView)
        panel.push_view(StubDepthView("assembly"), "assembly")

        panel.on_double_click_container(
            facet_type="NeuralCanvasFacet",
            data_path="/path/to/charm.nncanvas",
            breadcrumb_label="Charm Network",
            context={"project_root": "/project"}
        )

        assert panel.depth() == 2
        top = panel.current_view()
        assert isinstance(top, StubDepthView)
        assert top._loaded_path == "/path/to/charm.nncanvas"
        assert top._loaded_context == {"project_root": "/project"}

    def test_on_double_click_unregistered_type_noop(self, panel):
        panel.push_view(StubDepthView("assembly"), "assembly")
        panel.on_double_click_container(
            facet_type="UnknownFacet",
            data_path="/nowhere",
            breadcrumb_label="Unknown"
        )
        assert panel.depth() == 1  # No new view pushed


class TestSignals:
    """Test signal emissions."""

    def test_depth_changed_on_push(self, panel, qtbot):
        view = StubDepthView("assembly")
        with qtbot.waitSignal(panel.depthChanged, timeout=1000) as blocker:
            panel.push_view(view, "assembly")
        assert blocker.args == [1]

    def test_depth_changed_on_pop(self, panel, qtbot):
        panel.push_view(StubDepthView("a"), "Level 0")
        panel.push_view(StubDepthView("b"), "Level 1")

        with qtbot.waitSignal(panel.depthChanged, timeout=1000) as blocker:
            panel.pop_one()
        assert blocker.args == [1]

    def test_depth_changed_on_clear(self, panel, qtbot):
        panel.push_view(StubDepthView("a"), "Level 0")

        with qtbot.waitSignal(panel.depthChanged, timeout=1000) as blocker:
            panel.clear_stack()
        assert blocker.args == [0]


class TestKeyboardNavigation:
    """Test Backspace key binding."""

    def test_backspace_pops_one(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        panel.push_view(StubDepthView("b"), "Level 1")
        assert panel.depth() == 2

        from PyQt6.QtCore import QEvent
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace,
                          Qt.KeyboardModifier.NoModifier)
        panel.keyPressEvent(event)
        assert panel.depth() == 1

    def test_backspace_at_root_is_noop(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        from PyQt6.QtCore import QEvent
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace,
                          Qt.KeyboardModifier.NoModifier)
        panel.keyPressEvent(event)
        assert panel.depth() == 1

    def test_non_backspace_key_passes_through(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        from PyQt6.QtCore import QEvent
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A,
                          Qt.KeyboardModifier.NoModifier)
        panel.keyPressEvent(event)
        assert panel.depth() == 1  # No change


class TestEmptyState:
    """Test empty state label behavior."""

    def test_empty_state_visible_initially(self, panel):
        assert panel._empty_label.isVisible()

    def test_empty_state_hidden_after_push(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        assert not panel._empty_label.isVisible()

    def test_empty_state_restored_after_clear(self, panel):
        panel.push_view(StubDepthView("a"), "Level 0")
        panel.clear_stack()
        assert panel._empty_label.isVisible()
