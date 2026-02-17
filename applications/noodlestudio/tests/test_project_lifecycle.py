# ──────────────────────────────────────────────────────────────
#   Project Lifecycle Tests
#
#   Verifies: default project structure, instance references,
#   assembly parsing, first-run auto-open.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_project_lifecycle
# PURPOSE:  Project Lifecycle Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'library')


class TestDefaultProjectStructure:
    """Default project must have valid structure with noodlings and stage instances."""

    def test_default_project_exists(self):
        """Default project directory and project file must exist."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        assert os.path.isdir(project_dir), f"Default project not found: {project_dir}"
        assert os.path.isfile(os.path.join(project_dir, 'project.noodleproj')), \
            "project.noodleproj not found"

    def test_default_project_has_noodling_templates(self):
        """Default project must contain Ajo, Krampus, and Juanita noodling templates."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        for noodling in ('ajo_majo', 'krampus', 'juanita'):
            noodling_dir = os.path.join(project_dir, 'Noodlings', noodling)
            assert os.path.isdir(noodling_dir), f"Noodling dir not found: {noodling_dir}"
            assert os.path.isfile(os.path.join(noodling_dir, 'assembly.yaml')), \
                f"assembly.yaml missing for {noodling}"
            assert os.path.isfile(os.path.join(noodling_dir, 'recipe.yaml')), \
                f"recipe.yaml missing for {noodling}"
            assert os.path.isfile(os.path.join(noodling_dir, 'noodling.yaml')), \
                f"noodling.yaml missing for {noodling}"

    def test_default_project_has_stage_with_instances(self):
        """Default project must have noodling instances on the nexus stage."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        assert os.path.isdir(instances_dir), f"Instances dir not found: {instances_dir}"
        assert os.path.isdir(os.path.join(instances_dir, 'ajo'))
        assert os.path.isdir(os.path.join(instances_dir, 'krampus'))
        assert os.path.isdir(os.path.join(instances_dir, 'juanita'))

    def test_default_project_stage_yaml_valid(self):
        """Stage YAML must parse and contain expected fields."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        stage_yaml = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'stage.yaml'
        )
        with open(stage_yaml) as f:
            data = yaml.safe_load(f)
        assert data['name'] == 'The Nexus'
        assert 'geometry' in data


class TestDefaultProjectInstanceRefs:
    """Each instance.yaml must reference a noodling template that exists."""

    def test_instances_reference_valid_noodlings(self):
        """Instance noodling refs must resolve to real template directories."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        for instance_name in ('ajo', 'krampus', 'juanita'):
            instance_path = os.path.join(
                instances_dir, instance_name, 'instance.yaml'
            )
            assert os.path.isfile(instance_path), \
                f"instance.yaml not found for {instance_name}"

            with open(instance_path) as f:
                instance = yaml.safe_load(f)

            noodling_ref = instance['noodling']
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(instance_path), noodling_ref)
            )
            assert os.path.isdir(resolved), \
                f"Noodling template not found: {resolved} (from {instance_name})"
            assert os.path.isfile(os.path.join(resolved, 'assembly.yaml')), \
                f"No assembly.yaml in {resolved}"

    def test_assemblies_parse_without_error(self):
        """Assembly YAMLs in the default project must parse correctly."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        for noodling in ('ajo_majo', 'krampus', 'juanita'):
            assembly_path = os.path.join(
                project_dir, 'Noodlings', noodling, 'assembly.yaml'
            )
            with open(assembly_path) as f:
                data = yaml.safe_load(f)
            assert 'name' in data, f"Assembly missing 'name' field: {noodling}"
            assert 'facets' in data, f"Assembly missing 'facets' field: {noodling}"
            assert len(data['facets']) >= 3, \
                f"Assembly should have at least 3 facets: {noodling}"


class TestAlwaysShowChooserOnLaunch:
    """auto_open_last_project must always show the Project Chooser (Logic Pro model)."""

    def test_chooser_shown_even_with_recent_project(self, qapp, tmp_path):
        """Chooser must always appear on launch, even when recent projects exist."""
        from noodlestudio.core.project_manager import ProjectManager
        from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

        # Create a real project on disk
        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'test_project')
        project_path = str(tmp_path / 'test_project')
        pm.close_project()

        # Build a minimal host that has the mixin
        class StubProjectHost(MainWindowProjectMixin):
            def __init__(self):
                self.project_manager = ProjectManager()
                self._recent = [project_path]
                self.chooser_shown = False

            def load_recent_projects(self):
                return self._recent

            def save_recent_projects(self, projects):
                self._recent = projects

            def update_recent_projects_menu(self):
                pass

            def _show_project_chooser(self):
                self.chooser_shown = True

        host = StubProjectHost()
        host.auto_open_last_project()

        assert host.chooser_shown, \
            "Chooser must ALWAYS be shown on launch (Logic Pro model)"
        assert not host.project_manager.is_project_open(), \
            "No project should be auto-opened — user must choose from the dialog"

    def test_chooser_shown_with_no_recent_projects(self, qapp):
        """Chooser must appear on launch even with no recent projects."""
        from noodlestudio.core.project_manager import ProjectManager
        from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

        class StubProjectHost(MainWindowProjectMixin):
            def __init__(self):
                self.project_manager = ProjectManager()
                self._recent = []
                self.chooser_shown = False

            def load_recent_projects(self):
                return self._recent

            def save_recent_projects(self, projects):
                self._recent = projects

            def update_recent_projects_menu(self):
                pass

            def _show_project_chooser(self):
                self.chooser_shown = True

        host = StubProjectHost()
        host.auto_open_last_project()

        assert not host.project_manager.is_project_open(), \
            "No project should be open when chooser is shown"
        assert host.chooser_shown, \
            "Chooser must be shown on launch with no recent projects"
