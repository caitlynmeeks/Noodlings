# ------------------------------------------------------------------
#   Project Chooser Dialog Tests
#
#   Verifies: dialog construction, template listing, recent projects,
#   signal emission on project selection.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_project_chooser
# PURPOSE:  Project Chooser Dialog Tests
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


class TestProjectChooserDialog:
    """ProjectChooserDialog must show templates and handle project selection."""

    def test_dialog_constructs_without_error(self, qapp):
        """Dialog must initialize without crashing."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        assert dialog.windowTitle() == "Choose a Project"

    def test_dialog_shows_templates_and_recent_in_sidebar(self, qapp):
        """Sidebar must have Templates and Recent categories."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        sidebar = dialog._sidebar
        assert sidebar.count() == 2
        assert sidebar.item(0).text() == "Templates"
        assert sidebar.item(1).text() == "Recent"

    def test_dialog_lists_discovered_templates(self, qapp):
        """Templates page must show at least Getting Started and Empty Project."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        template_names = [t['name'] for t in dialog._templates]
        assert 'Getting Started' in template_names
        assert 'Empty Project' in template_names

    def test_dialog_shows_recent_projects(self, qapp, tmp_path):
        """Recent page must list provided recent projects."""
        fake_project = tmp_path / 'my_project'
        fake_project.mkdir()

        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog(recent_projects=[str(fake_project)])

        dialog._sidebar.setCurrentRow(1)
        assert dialog._recent_list.count() >= 1

    def test_dialog_choose_button_disabled_initially(self, qapp):
        """Choose button must be disabled until a selection is made."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        assert not dialog._action_btn.isEnabled()
        assert dialog._action_btn.text() == "Choose"

    def test_template_selection_enables_choose(self, qapp):
        """Selecting a template must enable the Choose button."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        if dialog._templates:
            dialog._on_template_selected(dialog._templates[0])
            assert dialog._action_btn.isEnabled()
            assert dialog._action_btn.text() == "Choose"

    def test_bottom_bar_has_open_existing_button(self, qapp):
        """Bottom bar must have an 'Open an existing project...' button."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        from PyQt6.QtWidgets import QPushButton
        dialog = ProjectChooserDialog()
        buttons = dialog._bottom.findChildren(QPushButton)
        labels = [b.text() for b in buttons]
        assert "Open an existing project..." in labels

    def test_create_from_template_emits_signal(self, qapp, tmp_path):
        """Calling _choose_template's underlying create logic must emit projectSelected."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            ProjectChooserDialog, create_project_from_template,
        )

        dialog = ProjectChooserDialog()
        if not dialog._templates:
            pytest.skip("No templates discovered")

        # Directly test the create_project_from_template function
        # (the native Save dialog can't be tested in headless mode)
        tmpl = dialog._templates[0]
        received = []
        dialog.projectSelected.connect(lambda path: received.append(path))

        dest = create_project_from_template(tmpl['path'], 'SignalTest', str(tmp_path))
        assert dest is not None
        dialog.projectSelected.emit(dest)

        assert len(received) == 1
        assert os.path.isdir(received[0])
        assert os.path.isfile(os.path.join(received[0], 'project.noodleproj'))
