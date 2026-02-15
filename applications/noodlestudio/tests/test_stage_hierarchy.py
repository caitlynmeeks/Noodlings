# ──────────────────────────────────────────────────────────────
#   Stage Hierarchy Tests
#
#   Verifies: stage dropdown behavior with and without projects.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_stage_hierarchy
# PURPOSE:  Stage Hierarchy Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestStageDropdown:
    """Stage dropdown must show clean states for project/no-project."""

    def test_stage_dropdown_empty_without_project(self, qapp):
        """Stage dropdown must show placeholder when no project is open."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        hierarchy = SceneHierarchy()
        pm = ProjectManager()
        hierarchy.set_project_manager(pm)
        # No project opened — dropdown should show placeholder
        assert hierarchy.stage_selector.count() == 1
        text = hierarchy.stage_selector.itemText(0)
        assert "No project" in text, f"Expected placeholder, got: {text}"
        assert not hierarchy.stage_selector.isEnabled(), \
            "Stage selector should be disabled when no project is open"

    def test_stage_dropdown_populates_with_project(self, qapp, tmp_path):
        """Stage dropdown must show project stages when project is open."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        # Create a real project with a stage
        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'test_project')
        pm.create_stage('test_stage', 'A test stage')

        hierarchy = SceneHierarchy()
        hierarchy.set_project_manager(pm)

        assert hierarchy.stage_selector.isEnabled(), \
            "Stage selector should be enabled when project is open"
        assert hierarchy.stage_selector.count() >= 1
        # Check that the test_stage appears
        texts = [hierarchy.stage_selector.itemText(i)
                 for i in range(hierarchy.stage_selector.count())]
        found = any('test_stage' in t for t in texts)
        assert found, f"test_stage not found in dropdown items: {texts}"
