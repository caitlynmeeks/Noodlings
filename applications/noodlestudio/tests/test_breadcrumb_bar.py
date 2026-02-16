"""Tests for BreadcrumbBar widget."""

import pytest
from PyQt6.QtWidgets import QApplication

from noodlestudio.panels.editors.breadcrumb_bar import BreadcrumbBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestBreadcrumbBar:
    """Test breadcrumb path navigation bar."""

    def test_initial_state_is_empty(self, qapp):
        bar = BreadcrumbBar()
        assert bar.segment_count() == 0
        assert len(bar._buttons) == 0

    def test_set_path_creates_segments(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["Ajo Majo", "assembly", "Charm Network"])
        assert bar.segment_count() == 3
        assert len(bar._buttons) == 3

    def test_segment_labels_match(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["Root", "Level 1", "Level 2"])
        labels = [btn.text() for btn in bar._buttons]
        assert labels == ["Root", "Level 1", "Level 2"]

    def test_last_segment_is_disabled(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["Root", "Child", "Current"])
        # Last segment (current depth) should be disabled
        assert not bar._buttons[-1].isEnabled()
        # Earlier segments should be enabled
        assert bar._buttons[0].isEnabled()
        assert bar._buttons[1].isEnabled()

    def test_click_emits_depth_index(self, qapp, qtbot):
        bar = BreadcrumbBar()
        bar.set_path(["Root", "Child", "Current"])

        with qtbot.waitSignal(bar.segmentClicked, timeout=1000) as blocker:
            bar._buttons[0].click()

        assert blocker.args == [0]

    def test_click_middle_segment(self, qapp, qtbot):
        bar = BreadcrumbBar()
        bar.set_path(["Root", "Child", "Current"])

        with qtbot.waitSignal(bar.segmentClicked, timeout=1000) as blocker:
            bar._buttons[1].click()

        assert blocker.args == [1]

    def test_separators_between_segments(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["A", "B", "C"])
        # 3 segments -> 2 separators (the > characters)
        sep_labels = [s for s in bar._separators if s.text() == ">"]
        assert len(sep_labels) == 2

    def test_clear_removes_all(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["A", "B", "C"])
        assert bar.segment_count() == 3
        bar.clear()
        assert bar.segment_count() == 0
        assert len(bar._buttons) == 0
        assert len(bar._separators) == 0

    def test_set_path_replaces_previous(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["A", "B"])
        assert bar.segment_count() == 2
        bar.set_path(["X", "Y", "Z"])
        assert bar.segment_count() == 3
        labels = [btn.text() for btn in bar._buttons]
        assert labels == ["X", "Y", "Z"]

    def test_single_segment_is_disabled(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["Root"])
        assert bar.segment_count() == 1
        assert not bar._buttons[0].isEnabled()

    def test_truncation_with_many_segments(self, qapp):
        bar = BreadcrumbBar()
        bar.set_path(["A", "B", "C", "D", "E", "F", "G", "H"])
        # Full path is 8 segments
        assert bar.segment_count() == 8
        # But visible buttons should be limited to MAX_VISIBLE_SEGMENTS
        assert len(bar._buttons) <= bar.MAX_VISIBLE_SEGMENTS

    def test_fixed_height(self, qapp):
        bar = BreadcrumbBar()
        assert bar.maximumHeight() == 28
