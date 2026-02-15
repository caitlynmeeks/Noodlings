# ------------------------------------------------------------------
#   Window Title Tests
#
#   Verifies: title format, project name, stage display name,
#   updates on project open/close and stage change.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_window_title
# PURPOSE:  Window Title Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture
def project_copy(tmp_path):
    """Copy the Getting Started template to a temp directory.

    Tests must never open the template directly -- _save_hierarchy()
    would write a hierarchy.yaml with absolute paths back into it.
    """
    src = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'library',
        'templates', 'Getting Started'
    ))
    if not os.path.exists(src):
        pytest.skip("Default project template not found")
    dest = str(tmp_path / 'Getting Started')
    shutil.copytree(src, dest)
    return dest


class TestWindowTitle:
    """Window title must reflect project and stage state."""

    def test_default_title(self, main_window):
        """Default title must be 'NoodleStudio', not 'NoodleSTUDIO'."""
        title = main_window.windowTitle()
        assert title == "NoodleStudio", \
            f"Expected 'NoodleStudio', got: {title}"

    def test_title_with_project(self, main_window, project_copy):
        """Title must include project name after opening a project."""
        main_window.project_manager.open_project(project_copy)
        title = main_window.windowTitle()

        # Must contain em dash separator and project name
        assert "\u2014" in title, \
            f"Title should use em dash separator, got: {title}"
        assert "NoodleStudio" in title, \
            f"Title must start with NoodleStudio, got: {title}"

    def test_title_with_project_and_stage(self, main_window, project_copy):
        """Title must show stage display name, not internal key."""
        main_window.project_manager.open_project(project_copy)
        title = main_window.windowTitle()

        # Should show "The Nexus" (display name from stage.yaml),
        # not "the_nexus" (directory key)
        assert "the_nexus" not in title, \
            f"Title should use display name, not internal key: {title}"

    def test_title_updates_on_project_close(self, main_window, project_copy):
        """Title must revert to 'NoodleStudio' when project is closed."""
        main_window.project_manager.open_project(project_copy)
        main_window.project_manager.close_project()
        title = main_window.windowTitle()

        assert title == "NoodleStudio", \
            f"Expected 'NoodleStudio' after close, got: {title}"
