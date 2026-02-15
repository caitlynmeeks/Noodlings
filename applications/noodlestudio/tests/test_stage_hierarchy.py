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
import shutil
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


class TestHierarchyWithoutServer:
    """Hierarchy must populate from disk without requiring the server."""

    @staticmethod
    def _template_path():
        """Return path to the Getting Started template."""
        return os.path.join(
            os.path.dirname(__file__), '..', 'library',
            'templates', 'Getting Started'
        )

    @pytest.fixture
    def project_copy(self, tmp_path):
        """Copy the Getting Started template to a temp directory.

        Tests must never open the template directly -- _save_hierarchy()
        would write a hierarchy.yaml with absolute paths back into it.
        """
        src = os.path.abspath(self._template_path())
        if not os.path.exists(src):
            pytest.skip("Default project template not found")
        dest = str(tmp_path / 'Getting Started')
        shutil.copytree(src, dest)
        return dest

    def test_hierarchy_populates_without_server(self, qapp, project_copy):
        """Hierarchy shows stage content even when server is not running."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        pm = ProjectManager()
        pm.open_project(project_copy)

        hierarchy = SceneHierarchy()
        hierarchy._server_running = False  # Server explicitly OFF
        hierarchy.set_project_manager(pm)

        # Tree must have items despite server being off
        item_count = hierarchy.tree.topLevelItemCount()
        assert item_count > 0, (
            f"Hierarchy should populate from disk without server, "
            f"got {item_count} items"
        )

    def test_hierarchy_shows_instances_from_stage(self, qapp, project_copy):
        """Hierarchy must show Ajo and Yuki from the default project."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        pm = ProjectManager()
        pm.open_project(project_copy)

        hierarchy = SceneHierarchy()
        hierarchy._server_running = False
        hierarchy.set_project_manager(pm)

        # Collect all item names from the tree
        names = []
        for i in range(hierarchy.tree.topLevelItemCount()):
            item = hierarchy.tree.topLevelItem(i)
            names.append(item.text(0))

        assert any('Ajo' in n for n in names), (
            f"Expected 'Ajo' in hierarchy items, got: {names}"
        )
        assert any('Yuki' in n for n in names), (
            f"Expected 'Yuki' in hierarchy items, got: {names}"
        )

    def test_hierarchy_no_duplicate_items(self, qapp, project_copy):
        """Each hierarchy item must appear exactly once (no duplicates)."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        pm = ProjectManager()
        pm.open_project(project_copy)

        hierarchy = SceneHierarchy()
        hierarchy._server_running = False
        hierarchy.set_project_manager(pm)

        # Collect all item names from the tree
        names = []
        for i in range(hierarchy.tree.topLevelItemCount()):
            item = hierarchy.tree.topLevelItem(i)
            names.append(item.text(0))

        # Each name must appear exactly once
        from collections import Counter
        counts = Counter(names)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        assert not duplicates, (
            f"Hierarchy contains duplicate items: {duplicates}. "
            f"All items: {names}"
        )

    def test_hierarchy_shows_no_project_message_when_closed(self, qapp):
        """Hierarchy shows 'No project open' when no project is loaded."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        pm = ProjectManager()  # No project opened

        hierarchy = SceneHierarchy()
        hierarchy.set_project_manager(pm)

        # Status label should show no-project message
        # Note: use isHidden() not isVisible() -- isVisible() requires
        # the full parent chain to be shown, which it isn't in tests
        assert not hierarchy.status_label.isHidden(), \
            "Status label should be visible when no project is open"
        assert "No project" in hierarchy.status_label.text(), \
            f"Expected 'No project' message, got: {hierarchy.status_label.text()}"
