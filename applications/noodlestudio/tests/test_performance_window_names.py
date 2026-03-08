# ------------------------------------------------------------------
#   Performance Window Name Bar Tests
#
#   Verifies: ensemble performer names, active speaker highlight,
#   and name click emits noodlingSelected signal.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_performance_window_names
# PURPOSE:  Performance Window Name Bar Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture
def ensemble_window(qapp, qtbot):
    """Create an ensemble performance panel for testing."""
    from noodlestudio.runtime.ui.guide_performance_window import (
        PerformancePanel,
    )

    window = PerformancePanel(ensemble_mode=True)
    qtbot.addWidget(window)

    # Set up two performers
    window.set_performer_name('ajo', 'Ajo Majo')
    window.set_performer_name('yuki', 'Yuki Cyberfox')

    yield window

    window.close()


class TestEnsembleWindowNames:
    """Ensemble window must show performer names in the name bar."""

    def test_performer_names_displayed(self, ensemble_window):
        """Both performer names must be visible in the name bar."""
        labels = ensemble_window._performer_labels
        names = {slot: label.text() for slot, label in labels.items()}

        assert 'Ajo Majo' in names.values()
        assert 'Yuki Cyberfox' in names.values()

    def test_active_speaker_highlighted(self, ensemble_window):
        """Active speaker label must get gold highlight color."""
        ensemble_window.set_active_speaker('ajo')

        slot = ensemble_window._noodling_to_slot.get('ajo')
        label = ensemble_window._performer_labels.get(slot)
        assert label is not None

        style = label.styleSheet()
        # Gold color #E8C547 from set_active_speaker
        assert '#E8C547' in style

    def test_inactive_speaker_not_highlighted(self, ensemble_window):
        """Inactive speaker label must not have gold highlight."""
        ensemble_window.set_active_speaker('ajo')

        yuki_slot = ensemble_window._noodling_to_slot.get('yuki')
        yuki_label = ensemble_window._performer_labels.get(yuki_slot)
        assert yuki_label is not None

        style = yuki_label.styleSheet()
        assert '#E8C547' not in style


class TestNameBarClickSignal:
    """Clicking a performer name must emit noodlingSelected signal."""

    def test_noodling_selected_signal_exists(self, ensemble_window):
        """Panel must have a noodlingSelected signal."""
        assert hasattr(ensemble_window, 'noodlingSelected')

    def test_name_click_emits_signal(self, ensemble_window):
        """Clicking a performer name label must emit noodlingSelected(str)."""
        from PyQt6.QtCore import QEvent, QPointF
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import Qt

        received = []
        ensemble_window.noodlingSelected.connect(lambda nid: received.append(nid))

        # Find Ajo's label and simulate a click
        ajo_slot = ensemble_window._noodling_to_slot.get('ajo')
        label = ensemble_window._performer_labels.get(ajo_slot)
        assert label is not None

        # Create mouse press event
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        # Deliver via event filter
        ensemble_window.eventFilter(label, event)

        assert len(received) == 1
        assert received[0] == 'ajo'

    def test_click_different_performer(self, ensemble_window):
        """Clicking Yuki's name must emit her noodling_id."""
        from PyQt6.QtCore import QEvent, QPointF
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import Qt

        received = []
        ensemble_window.noodlingSelected.connect(lambda nid: received.append(nid))

        yuki_slot = ensemble_window._noodling_to_slot.get('yuki')
        label = ensemble_window._performer_labels.get(yuki_slot)

        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        ensemble_window.eventFilter(label, event)

        assert len(received) == 1
        assert received[0] == 'yuki'
