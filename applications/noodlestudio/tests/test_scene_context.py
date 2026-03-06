# ------------------------------------------------------------------
#   Scene Context Pipeline Tests
#
#   Tests that build_scene_context integrates correctly with
#   GuidePerformanceManager, including mark resolution, backward
#   compatibility, and WHO IS HERE formatting.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_scene_context
# PURPOSE:  Scene Context Pipeline Tests
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
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from noodlestudio.core.set_dressing import (
    SetObject, StageSet, BlockingMark,
    build_scene_context, save_set, save_mark,
)


# =====================================================================
# build_scene_context Integration
# =====================================================================

class TestBuildSceneContextIntegration:
    """Test the context builder with realistic multi-noodling scenarios."""

    @pytest.fixture
    def cafe_set(self):
        return StageSet(
            name='Hearthwood Cafe',
            description='A warm cafe nestled in the mountains.',
            objects=[
                SetObject('fireplace', 'Stone Fireplace', 'Crackling warmth.'),
                SetObject('counter', 'Counter', 'Worn volcanic stone.'),
                SetObject('bookshelf', 'Bookshelf', 'Old dog-eared books.'),
                SetObject('windows', 'Front Windows', 'View of the forest.'),
            ],
        )

    @pytest.fixture
    def ajo_mark(self):
        return BlockingMark(
            id='behind_counter',
            name='Behind the Counter',
            perspective="You're behind the counter, polishing glasses.",
            can_see=['counter', 'fireplace', 'bookshelf', 'windows'],
        )

    @pytest.fixture
    def juanita_mark(self):
        return BlockingMark(
            id='window_seat',
            name='Window Seat',
            perspective="You're by the window, watching the forest.",
            can_see=['windows', 'bookshelf'],
        )

    @pytest.fixture
    def krampus_mark(self):
        return BlockingMark(
            id='by_the_fire',
            name='By the Fire',
            perspective="You're sprawled on a rug by the fireplace.",
            can_see=['fireplace', 'bookshelf'],
        )

    def test_full_set_mark_others(self, cafe_set, ajo_mark):
        """Full context with set, mark, and other noodlings."""
        others = [
            {'name': 'Juanita', 'mark_name': 'Window Seat'},
            {'name': 'Krampus', 'mark_name': 'By the Fire'},
        ]
        ctx = build_scene_context(cafe_set, ajo_mark, others)

        assert 'THE SPACE:' in ctx
        assert 'warm cafe' in ctx
        assert 'WHERE YOU ARE:' in ctx
        assert 'polishing glasses' in ctx
        assert 'WHAT YOU CAN SEE:' in ctx
        assert 'Counter:' in ctx
        assert 'WHO IS HERE:' in ctx
        assert 'Juanita is at Window Seat.' in ctx
        assert 'Krampus is at By the Fire.' in ctx

    def test_no_set_returns_empty(self):
        """No set dressing returns empty (backward compat)."""
        ctx = build_scene_context(None, None)
        assert ctx == ''

    def test_no_mark_shows_all_objects(self, cafe_set):
        """No mark assigned shows all objects under AROUND YOU."""
        ctx = build_scene_context(cafe_set, None)
        assert 'AROUND YOU:' in ctx
        assert 'Stone Fireplace:' in ctx
        assert 'Counter:' in ctx
        assert 'Bookshelf:' in ctx
        assert 'Front Windows:' in ctx

    def test_mark_filters_visible_objects(self, cafe_set, juanita_mark):
        """Mark's can_see filters objects to only visible ones."""
        ctx = build_scene_context(cafe_set, juanita_mark)
        assert 'Front Windows:' in ctx
        assert 'Bookshelf:' in ctx
        # Counter and Fireplace not in juanita's can_see
        assert 'Counter:' not in ctx
        assert 'Stone Fireplace:' not in ctx

    def test_empty_can_see_no_objects_section(self, cafe_set):
        """Empty can_see list produces no WHAT YOU CAN SEE section."""
        mark = BlockingMark(
            id='isolated', name='Isolated', perspective='In a closet.',
            can_see=[],
        )
        ctx = build_scene_context(cafe_set, mark)
        assert 'WHAT YOU CAN SEE:' not in ctx
        assert 'WHERE YOU ARE:' in ctx

    def test_who_is_here_includes_mark_names(self, cafe_set, ajo_mark):
        """Others with mark names show their position."""
        others = [
            {'name': 'Juanita', 'mark_name': 'Window Seat'},
        ]
        ctx = build_scene_context(cafe_set, ajo_mark, others)
        assert 'Juanita is at Window Seat.' in ctx

    def test_who_is_here_without_mark(self, cafe_set, ajo_mark):
        """Others without mark names show 'is here'."""
        others = [{'name': 'Visitor'}]
        ctx = build_scene_context(cafe_set, ajo_mark, others)
        assert 'Visitor is here.' in ctx

    def test_different_perspectives_per_mark(self, cafe_set, ajo_mark, krampus_mark):
        """Each mark produces a different perspective."""
        ctx_ajo = build_scene_context(cafe_set, ajo_mark)
        ctx_krampus = build_scene_context(cafe_set, krampus_mark)

        assert 'polishing glasses' in ctx_ajo
        assert 'sprawled on a rug' in ctx_krampus
        assert 'polishing glasses' not in ctx_krampus

    def test_no_others_omits_who_section(self, cafe_set, ajo_mark):
        """No others list omits WHO IS HERE section."""
        ctx = build_scene_context(cafe_set, ajo_mark)
        assert 'WHO IS HERE:' not in ctx


# =====================================================================
# Update Mark
# =====================================================================

class TestUpdateMark:
    """Test live mark reassignment via update_mark."""

    def test_update_mark_changes_metadata(self):
        """update_mark should update _instance_metadata dict."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'ajo': {'noodling_id': 'ajo', 'name': 'Ajo', 'mark': 'behind_counter'},
        }

        manager.update_mark('ajo', 'window_seat')
        assert manager._instance_metadata['ajo']['mark'] == 'window_seat'

    def test_update_mark_nonexistent_noodling(self):
        """update_mark for unknown noodling does not crash."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {}
        # Should not raise
        manager.update_mark('unknown', 'some_mark')
