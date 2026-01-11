# ──────────────────────────────────────────────────────────────
#   Tests for Panel Fold Manager
#
#   Tests for the fold/unfold animation system.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QSplitter, QWidget
from PyQt6.QtCore import Qt

from noodlestudio.core.panel_fold_manager import PanelFoldManager


class TestPanelFoldManager:
    """Tests for PanelFoldManager."""

    @pytest.fixture
    def splitters(self, qtbot):
        """Create splitters for testing."""
        # Top splitter: left | center | right
        top = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        center_widget = QWidget()
        right_widget = QWidget()
        top.addWidget(left_widget)
        top.addWidget(center_widget)
        top.addWidget(right_widget)
        top.setSizes([250, 800, 280])
        qtbot.addWidget(top)

        # Main splitter: top | bottom
        main = QSplitter(Qt.Orientation.Vertical)
        bottom_widget = QWidget()
        main.addWidget(top)
        main.addWidget(bottom_widget)
        main.setSizes([600, 180])
        qtbot.addWidget(main)

        return main, top

    @pytest.fixture
    def manager(self, splitters, qtbot):
        """Create manager with splitters."""
        main, top = splitters
        mgr = PanelFoldManager(main, top)
        return mgr

    def test_initial_state(self, manager):
        """Manager starts unfolded."""
        assert manager.is_folded is False
        assert manager.is_animating is False

    def test_fold_instant(self, manager, splitters):
        """Instant fold sets correct sizes."""
        main, top = splitters

        manager.fold(animated=False)

        assert manager.is_folded is True
        # Left and right should be 0
        sizes = top.sizes()
        assert sizes[0] == 0  # Left
        assert sizes[2] == 0  # Right
        # Bottom should be 0
        assert main.sizes()[1] == 0

    def test_unfold_instant(self, manager, splitters):
        """Instant unfold restores sizes."""
        main, top = splitters

        # Fold first
        manager.fold(animated=False)
        assert manager.is_folded is True

        # Unfold
        manager.unfold(animated=False)

        assert manager.is_folded is False
        # Panels should be visible again
        sizes = top.sizes()
        assert sizes[0] > 0  # Left
        assert sizes[2] > 0  # Right
        assert main.sizes()[1] > 0  # Bottom

    def test_toggle(self, manager):
        """Toggle switches between states."""
        assert manager.is_folded is False

        manager.toggle(animated=False)
        assert manager.is_folded is True

        manager.toggle(animated=False)
        assert manager.is_folded is False

    def test_set_folded(self, manager):
        """set_folded sets explicit state."""
        manager.set_folded(True, animated=False)
        assert manager.is_folded is True

        manager.set_folded(False, animated=False)
        assert manager.is_folded is False

    def test_fold_emits_signals(self, manager, qtbot):
        """Fold emits appropriate signals."""
        fold_started = []
        fold_complete = []
        state_changed = []

        manager.fold_started.connect(lambda: fold_started.append(True))
        manager.fold_complete.connect(lambda: fold_complete.append(True))
        manager.state_changed.connect(lambda s: state_changed.append(s))

        manager.fold(animated=False)

        assert len(fold_started) == 1
        assert len(fold_complete) == 1
        assert state_changed == [True]

    def test_unfold_emits_signals(self, manager, qtbot):
        """Unfold emits appropriate signals."""
        manager.fold(animated=False)

        unfold_started = []
        unfold_complete = []
        state_changed = []

        manager.unfold_started.connect(lambda: unfold_started.append(True))
        manager.unfold_complete.connect(lambda: unfold_complete.append(True))
        manager.state_changed.connect(lambda s: state_changed.append(s))

        manager.unfold(animated=False)

        assert len(unfold_started) == 1
        assert len(unfold_complete) == 1
        assert state_changed == [False]

    def test_fold_when_already_folded(self, manager):
        """Folding when already folded does nothing."""
        manager.fold(animated=False)
        assert manager.is_folded is True

        # Second fold should do nothing
        manager.fold(animated=False)
        assert manager.is_folded is True

    def test_unfold_when_already_unfolded(self, manager):
        """Unfolding when already unfolded does nothing."""
        assert manager.is_folded is False

        # Unfold should do nothing
        manager.unfold(animated=False)
        assert manager.is_folded is False

    def test_saved_sizes_restored(self, manager, splitters):
        """Saved sizes are restored on unfold (proportionally)."""
        main, top = splitters

        # Get initial sizes
        initial_top = top.sizes()
        initial_main = main.sizes()

        # Fold
        manager.fold(animated=False)

        # Unfold
        manager.unfold(animated=False)

        # All panels should be visible again
        restored_top = top.sizes()
        restored_main = main.sizes()

        # Left and right should be non-zero
        assert restored_top[0] > 0  # Left
        assert restored_top[2] > 0  # Right
        assert restored_main[1] > 0  # Bottom

    def test_set_default_sizes(self, manager):
        """set_default_sizes updates default values."""
        manager.set_default_sizes(left=200, right=200, bottom=150)

        assert manager.DEFAULT_LEFT == 200
        assert manager.DEFAULT_RIGHT == 200
        assert manager.DEFAULT_BOTTOM == 150


class TestEasingFunctions:
    """Tests for easing functions."""

    def test_ease_out_at_zero(self):
        """Ease-out returns 0 at t=0."""
        assert PanelFoldManager._ease_out(0.0) == 0.0

    def test_ease_out_at_one(self):
        """Ease-out returns 1 at t=1."""
        assert PanelFoldManager._ease_out(1.0) == 1.0

    def test_ease_out_faster_start(self):
        """Ease-out is faster at start (>0.5 at t=0.5)."""
        result = PanelFoldManager._ease_out(0.5)
        assert result > 0.5  # Progress faster at start

    def test_ease_in_at_zero(self):
        """Ease-in returns 0 at t=0."""
        assert PanelFoldManager._ease_in(0.0) == 0.0

    def test_ease_in_at_one(self):
        """Ease-in returns 1 at t=1."""
        assert PanelFoldManager._ease_in(1.0) == 1.0

    def test_ease_in_slower_start(self):
        """Ease-in is slower at start (<0.5 at t=0.5)."""
        result = PanelFoldManager._ease_in(0.5)
        assert result < 0.5  # Progress slower at start
