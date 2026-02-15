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
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        assert os.path.isdir(project_dir), f"Default project not found: {project_dir}"
        assert os.path.isfile(os.path.join(project_dir, 'project.noodleproj')), \
            "project.noodleproj not found"

    def test_default_project_has_noodling_templates(self):
        """Default project must contain Ajo and Yuki noodling templates."""
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        for noodling in ('ajo_majo', 'yuki_cyberfox'):
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
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        assert os.path.isdir(instances_dir), f"Instances dir not found: {instances_dir}"
        assert os.path.isdir(os.path.join(instances_dir, 'ajo'))
        assert os.path.isdir(os.path.join(instances_dir, 'yuki'))

    def test_default_project_stage_yaml_valid(self):
        """Stage YAML must parse and contain expected fields."""
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
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
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        for instance_name in ('ajo', 'yuki'):
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
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        for noodling in ('ajo_majo', 'yuki_cyberfox'):
            assembly_path = os.path.join(
                project_dir, 'Noodlings', noodling, 'assembly.yaml'
            )
            with open(assembly_path) as f:
                data = yaml.safe_load(f)
            assert 'name' in data, f"Assembly missing 'name' field: {noodling}"
            assert 'facets' in data, f"Assembly missing 'facets' field: {noodling}"
            assert len(data['facets']) >= 3, \
                f"Assembly should have at least 3 facets: {noodling}"


class TestFirstRunAutoOpen:
    """First run with no recent projects should open the default project."""

    def test_first_run_opens_default_project(self, qapp, tmp_path, monkeypatch):
        """auto_open_last_project must copy and open default project on first run."""
        from noodlestudio.core.project_manager import ProjectManager
        from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

        # Build a minimal host that has the mixin
        class StubProjectHost(MainWindowProjectMixin):
            def __init__(self):
                self.project_manager = ProjectManager()
                self._recent_projects = []

            def load_recent_projects(self):
                return self._recent_projects

            def save_recent_projects(self, projects):
                self._recent_projects = projects

            def update_recent_projects_menu(self):
                pass

        host = StubProjectHost()

        # Monkeypatch user home to tmp_path so we don't pollute real Documents
        monkeypatch.setattr(
            os.path, 'expanduser',
            lambda path: str(tmp_path) if path == '~' else os.path.expanduser(path)
        )

        host.auto_open_last_project()

        # Verify project was copied and opened
        dest = os.path.join(
            str(tmp_path), 'Documents', 'NoodleStudio Projects',
            'Welcome to NoodleStudio'
        )
        assert os.path.isdir(dest), f"Default project not copied to {dest}"
        assert os.path.isfile(os.path.join(dest, 'project.noodleproj'))
        assert host.project_manager.is_project_open(), \
            "Project should be open after first-run auto-open"
