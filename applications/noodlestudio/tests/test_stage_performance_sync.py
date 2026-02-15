# ------------------------------------------------------------------
#   Stage-Performance Sync Tests
#
#   Verifies: hierarchy selection syncs to performance window,
#   facets editor switches noodling on hierarchy selection.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_stage_performance_sync
# PURPOSE:  Stage-Performance Sync Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from conftest import StubMainWindow, StubFacetsEditor, StubWindow, FakeLLMClient


class TestHierarchySelectsSpeaker:
    """Hierarchy noodling selection must sync to performance window."""

    def _make_ensemble_manager(self):
        """Build a GuidePerformanceManager in ensemble mode with stubs."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        editor = StubFacetsEditor()
        main_window = StubMainWindow(facets_editor=editor)
        manager = GuidePerformanceManager(main_window)
        manager._facets_editor = editor

        # Create stub window that tracks active speaker
        window = StubWindow()
        window._active_speaker = None
        original_set = window.set_active_speaker

        def tracking_set_active_speaker(noodling_id=None):
            window._active_speaker = noodling_id
            original_set(noodling_id)

        window.set_active_speaker = tracking_set_active_speaker
        manager._window = window

        # Create two performers (no real assemblies needed for this test)
        ajo = NoodlingPerformer(
            noodling_id='ajo', name='Ajo', llm_client=FakeLLMClient()
        )
        yuki = NoodlingPerformer(
            noodling_id='yuki', name='Yuki', llm_client=FakeLLMClient()
        )

        manager._ensemble_mode = True
        manager._performers = {'ajo': ajo, 'yuki': yuki}
        manager._performer = ajo

        # Set up ensemble noodlings on the editor
        editor.set_ensemble_noodlings([
            {'id': 'ajo', 'name': 'Ajo Majo',
             'assembly': None, 'assembly_path': None},
            {'id': 'yuki', 'name': 'Yuki Cyberfox',
             'assembly': None, 'assembly_path': None},
        ])

        return manager, window, editor

    def test_selecting_noodling_sets_active_speaker(self):
        """Selecting a noodling in hierarchy must highlight it in the window."""
        manager, window, editor = self._make_ensemble_manager()

        manager.on_hierarchy_noodling_selected('yuki')
        assert window._active_speaker == 'yuki'

    def test_selecting_noodling_switches_facets_editor(self):
        """Selecting a noodling must switch the facets editor to that noodling."""
        manager, window, editor = self._make_ensemble_manager()

        # Initially Ajo is selected (index 0)
        assert editor._selected_noodling_id == 'ajo'

        manager.on_hierarchy_noodling_selected('yuki')
        assert editor._selected_noodling_id == 'yuki'

    def test_selecting_unknown_noodling_is_noop(self):
        """Selecting an unknown noodling_id must not crash or change state."""
        manager, window, editor = self._make_ensemble_manager()

        manager.on_hierarchy_noodling_selected('nonexistent')
        # Active speaker should not change
        assert window._active_speaker is None

    def test_not_in_ensemble_mode_is_noop(self):
        """In single-performer mode, noodling selection is a no-op."""
        manager, window, editor = self._make_ensemble_manager()
        manager._ensemble_mode = False

        manager.on_hierarchy_noodling_selected('yuki')
        assert window._active_speaker is None

    def test_no_window_is_safe(self):
        """If no window exists, noodling selection must not crash."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        manager = GuidePerformanceManager(StubMainWindow())
        manager._ensemble_mode = True
        manager._window = None

        # Should not raise
        manager.on_hierarchy_noodling_selected('ajo')


class TestFacetsEditorSelectNoodling:
    """Facets editor must support selecting a noodling by ID."""

    def test_select_noodling_updates_selected_id(self):
        """select_noodling(id) must update _selected_noodling_id."""
        editor = StubFacetsEditor()
        editor.set_ensemble_noodlings([
            {'id': 'ajo', 'name': 'Ajo', 'assembly': None, 'assembly_path': None},
            {'id': 'yuki', 'name': 'Yuki', 'assembly': None, 'assembly_path': None},
        ])

        assert editor._selected_noodling_id == 'ajo'
        editor.select_noodling('yuki')
        assert editor._selected_noodling_id == 'yuki'

    def test_select_unknown_noodling_is_noop(self):
        """select_noodling with unknown ID must not change selection."""
        editor = StubFacetsEditor()
        editor.set_ensemble_noodlings([
            {'id': 'ajo', 'name': 'Ajo', 'assembly': None, 'assembly_path': None},
        ])

        editor.select_noodling('nonexistent')
        assert editor._selected_noodling_id == 'ajo'
