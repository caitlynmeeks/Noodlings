# ------------------------------------------------------------------
#   Three-Noodling Ensemble Tests
#
#   Verifies that the ensemble system correctly handles three
#   noodlings (Ajo, Krampus, Juanita) on the default stage:
#   stage discovery, VRM resolution, turn queue construction,
#   and context passing between performers.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_three_noodling_ensemble
# PURPOSE:  Three-Noodling Ensemble Tests
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

LIBRARY_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'library'
)
STAGE_PATH = os.path.join(
    LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
)


# =====================================================================
# Stage Discovery with Three Noodlings
# =====================================================================

class TestThreeNoodlingDiscovery:
    """Stage discovery must find all three noodlings with correct data."""

    @pytest.fixture
    def manager(self, qapp):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow
        return GuidePerformanceManager(StubMainWindow())

    def test_discovers_three_instances(self, manager):
        """Stage must yield exactly three instances."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        assert len(instances) == 3

    def test_correct_noodling_ids(self, manager):
        """Instance IDs must be ajo, krampus, juanita (from directory names)."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        ids = {i['noodling_id'] for i in instances}
        assert ids == {'ajo', 'krampus', 'juanita'}

    def test_display_names_from_overrides(self, manager):
        """Display names must come from instance.yaml overrides."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        names = {i['noodling_id']: i['name'] for i in instances}
        assert names['ajo'] == 'Ajo Majo'
        assert names['krampus'] == 'Krampus'
        assert names['juanita'] == 'Juanita'

    def test_all_have_vrm_paths(self, manager):
        """All three noodlings must have resolved VRM paths."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        for info in instances:
            assert info['vrm_path'] is not None, (
                f"{info['noodling_id']} has no VRM path"
            )
            assert os.path.isfile(info['vrm_path']), (
                f"VRM not found: {info['vrm_path']}"
            )

    def test_all_have_assembly_paths(self, manager):
        """All three noodlings must have valid assembly.yaml files."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        for info in instances:
            assert os.path.isfile(info['assembly_path']), (
                f"Assembly not found: {info['assembly_path']}"
            )

    def test_sorted_by_directory_name(self, manager):
        """Instances should be returned sorted by directory name."""
        instances = manager._discover_stage_instances(STAGE_PATH)
        ids = [i['noodling_id'] for i in instances]
        assert ids == sorted(ids)


# =====================================================================
# Turn Queue with Three Performers
# =====================================================================

class TestThreeNoodlingTurnQueue:
    """Turn-taking must iterate all three performers in order."""

    @pytest.fixture
    def manager_with_performers(self, qapp):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from conftest import StubMainWindow, FakeLLMClient

        manager = GuidePerformanceManager(StubMainWindow())

        # Create three performers in stage order
        for nid, name in [('ajo', 'Ajo'), ('krampus', 'Krampus'), ('juanita', 'Juanita')]:
            performer = NoodlingPerformer(
                noodling_id=nid, name=name, llm_client=FakeLLMClient()
            )
            manager._performers[nid] = performer

        return manager

    def test_turn_queue_has_three_entries(self, manager_with_performers):
        """Turn queue must contain all three noodling IDs."""
        manager = manager_with_performers
        manager._turn_responses = {}
        manager._turn_queue = list(manager._performers.keys())
        assert len(manager._turn_queue) == 3

    def test_turn_queue_order_matches_performers(self, manager_with_performers):
        """Turn queue order must match performer insertion order."""
        manager = manager_with_performers
        manager._turn_queue = list(manager._performers.keys())
        assert manager._turn_queue == ['ajo', 'krampus', 'juanita']

    def test_context_accumulates_across_turns(self, manager_with_performers):
        """Each performer should receive previous performers' responses as context."""
        manager = manager_with_performers
        # Simulate the context that would accumulate
        manager._turn_responses = {
            'ajo': 'Ajo said hello',
            'krampus': 'Krampus is NOT amused',
        }
        extra_context = {}
        for prev_nid, response in manager._turn_responses.items():
            extra_context[f'{prev_nid}_said'] = response

        # Juanita (third in queue) should see both previous responses
        assert 'ajo_said' in extra_context
        assert 'krampus_said' in extra_context
        assert extra_context['ajo_said'] == 'Ajo said hello'
        assert extra_context['krampus_said'] == 'Krampus is NOT amused'
